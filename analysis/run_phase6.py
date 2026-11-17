"""Phase 6: implied volatility and the volatility smile on real NSE data.

    python analysis/run_phase6.py          (about 2 seconds)

Phases 1-5 asked "given a volatility, what is this option worth, and how
accurately can I compute it?". The answer was always measured against a model
that was true by assumption. Phase 6 turns the question round and points it at
the market: take the prices the exchange actually settled at, invert
Black-Scholes for the volatility implied by each one, and see whether the
single constant volatility the model assumes is there.

It is not, and the shape of its absence is the result.

Data
----
``data/nifty_fo_bhavcopy_<date>.csv`` -- the NIFTY rows of one trading day's
NSE F&O bhavcopy, downloaded by ``analysis/fetch_phase6_data.py``. See the
README for every market input and where it came from.

Why Nifty index options
-----------------------
They are **European**: exercise only at expiry. That matters more than it
sounds. The whole project's closed-form machinery -- and therefore the
inversion -- assumes European exercise, and American options on dividend-free
stock would need the Phase 2 binomial tree inverted instead. Indian single
stock options are American; the index options are not.

Why out-of-the-money options only
---------------------------------
At a given strike and expiry a call and a put contain the *same* volatility
information -- put-call parity is a model-free identity, so one can be
converted into the other with no volatility input at all. Having both would
double-count. Choosing the out-of-the-money one of the pair is the standard
convention and there is a concrete reason for it:

* an out-of-the-money option is **all** optionality. Its entire price is the
  premium that volatility is being read from.
* an in-the-money option is mostly intrinsic value. A deep in-the-money call
  worth 5,000 points might carry 3 points of optionality, so a one-tick
  rounding in the close moves the implied volatility far more than it moves
  the price. The same information arrives with a much worse signal-to-noise
  ratio.

So: puts for strikes below spot, calls for strikes above.

Outputs
-------
``iv_smile.csv``          every surviving strike, with its implied volatility.
``iv_filter_log.csv``     how many strikes each filter removed, and why.
``iv_term_structure.csv`` one row per expiry: the smile in numbers, the
                          forward diagnostics, and what the solver cost.
"""

import csv
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # the package is not installed; see pytest.ini

from pricing.black_scholes import vega                              # noqa: E402
from pricing.implied_vol import implied_vol_detail                  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"

BHAVCOPY = DATA / "nifty_fo_bhavcopy_20261116.csv"

# ---------------------------------------------------------------------------
# Market inputs, with provenance
# ---------------------------------------------------------------------------
#
# Every number here is stated in the README too. Nothing is estimated or
# simulated: the spot and the option prices come out of the bhavcopy, and the
# rate comes from a published auction result.

#: The trading day. Monday 16 November 2026 -- read back from the file itself
#: rather than trusted, see ``load_chain``.
TRADE_DATE = date(2026, 11, 16)

#: NSE's minimum price step for index options. One tick is the finest
#: distinction the close can make, so it sets the resolution of every implied
#: volatility reported here.
TICK = 0.05

#: Cut-off price of the 91-day Government of India Treasury bill at the RBI
#: auction of 11 November 2026: face 100, accepted at 98.5581, implicit yield
#: at cut-off 5.8681% on a simple 365-day basis.
TBILL_PRICE = 98.5581
TBILL_FACE = 100.0
TBILL_DAYS = 91.0

#: Day-count convention, used for both the bill and the options: calendar
#: days over 365. Simple, stated once, and applied to everything. (An Indian
#: money-market desk would use ACT/365 for the bill and the exchange's own
#: business-day count for the option, which would move the rate by a few
#: basis points and the option prices by far less than one tick.)
DAYS_PER_YEAR = 365.0


def continuously_compounded_rate() -> float:
    """Turn the T-bill's cut-off *price* into Black-Scholes' ``r``.

    A Treasury bill pays nothing until maturity: you buy it at 98.5581 and it
    repays 100 after 91 days. So the gross return over the period is
    ``100 / 98.5581``, and the rate that compounds continuously to the same
    thing over ``T = 91/365`` years solves ``exp(r*T) = 100/98.5581``::

        r = ln(face / price) / T

    This is not the 5.8681% RBI quotes. That figure is the *simple* annualised
    yield, ``(face - price)/price * 365/91``. Continuous compounding reaches
    the same total return with a slightly smaller rate, because the rate is
    applied to a balance that is growing: 5.8255% against 5.8681%, a
    difference of 4.3 basis points. Black-Scholes discounts with
    ``exp(-r*T)``, so the continuously compounded number is the one it wants
    -- feeding it the quoted yield would be a small but entirely avoidable
    error.
    """
    return math.log(TBILL_FACE / TBILL_PRICE) / (TBILL_DAYS / DAYS_PER_YEAR)


RISK_FREE_RATE = continuously_compounded_rate()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

#: Minimum close price to keep, in index points. Twenty ticks.
#:
#: The reason for a floor at all: implied volatility is read from the premium,
#: and near the price floor the premium is mostly quantisation. At a close of
#: 1.00 one tick is a 5% change in the price; at 0.90 -- the cheapest close in
#: this file, and the price of dozens of far out-of-the-money weekly puts --
#: a single tick moves the implied volatility by about 0.3 of a vol point, and
#: the clustering of those quotes at exactly 0.90 shows the price has hit a
#: practical floor rather than a valuation.
#:
#: Twenty ticks rather than, say, two hundred, because the aim is to remove
#: quotes that are *mostly* rounding, not to prune the wing until the smile
#: looks tidy. ``iv_smile.csv`` carries the per-tick sensitivity of every
#: surviving strike so the reader can see where the remaining noise is.
PRICE_FLOOR = 20.0 * TICK

#: Filters are applied in this order and counted at each stage, so the drop
#: accounting in ``iv_filter_log.csv`` adds up exactly.
FILTER_ORDER = [
    "in_the_money",
    "zero_volume",
    "zero_open_interest",
    "below_price_floor",
    "no_implied_vol",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_chain(path: Path) -> tuple[list[dict], float, dict[str, float], date]:
    """Read the bhavcopy slice: option rows, spot, futures closes, trade date.

    The spot comes from the file's own ``UndrlygPric`` column, which carries
    the underlying's close on every contract row. It is checked for
    consistency across all 2,242 rows rather than taken from the first one --
    if the exchange ever shipped a file with two different underlying prices
    in it, silently using row zero would be the wrong thing to do.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    options = [r for r in rows if r["OptnTp"] in ("CE", "PE")]
    futures = {r["XpryDt"]: float(r["ClsPric"]) for r in rows
               if r["FinInstrmTp"] in ("IDF", "FUTIDX")}

    spots = {float(r["UndrlygPric"]) for r in rows}
    if len(spots) != 1:
        raise ValueError(f"file carries more than one underlying price: {spots}")
    trade_dates = {r["TradDt"] for r in rows}
    if len(trade_dates) != 1:
        raise ValueError(f"file spans more than one trading day: {trade_dates}")

    return options, spots.pop(), futures, date.fromisoformat(trade_dates.pop())


def monthly_expiries(options: list[dict], futures: dict[str, float]) -> list[str]:
    """The monthly expiries, identified as the ones that have a future.

    NSE lists NIFTY options weekly but futures only monthly, so "has a futures
    contract in the same file" picks out the monthly series exactly, with no
    calendar rules to get wrong. That matters here because the obvious rule --
    "the last Tuesday of the month" -- is wrong for two of the three expiries
    in this file: 24 November 2026 is Guru Nanak Jayanti and 26 January 2027 is
    Republic Day, so both expiries move back a day to the Monday. The data
    knows; a hard-coded calendar would not.

    Monthly rather than weekly because the monthlies carry the open interest
    and quote a wider strike range, which is what a smile needs.
    """
    listed = {r["XpryDt"] for r in options}
    return sorted(x for x in futures if x in listed)


# ---------------------------------------------------------------------------
# The smile for one expiry
# ---------------------------------------------------------------------------

def build_smile(options: list[dict], expiry: str, spot: float, rate: float,
                forward: float) -> tuple[list[dict], Counter]:
    """Implied volatility at every usable strike of one expiry.

    Returns the surviving rows and a count of what each filter removed. The
    filters run in ``FILTER_ORDER``: a strike dropped for zero volume is not
    also counted as below the price floor, so the counts sum to the number of
    contracts that started.

    Each row carries **two** implied volatilities. ``implied_vol`` is the one
    the phase is about: Black-Scholes on the stated inputs, spot and the
    T-bill rate, no dividends. ``implied_vol_fwd`` re-inverts the same quote
    against ``forward``, the forward the option market is itself quoting via
    put-call parity -- a diagnostic, kept beside the main number rather than
    replacing it, so the plots can show what the no-dividend assumption costs.
    """
    expiry_date = date.fromisoformat(expiry)
    days = (expiry_date - TRADE_DATE).days
    years = days / DAYS_PER_YEAR

    dropped: Counter = Counter()
    smile = []

    for row in options:
        if row["XpryDt"] != expiry:
            continue

        strike = float(row["StrkPric"])
        kind = "call" if row["OptnTp"] == "CE" else "put"

        # 1. Out of the money only: puts below spot, calls above.
        out_of_the_money = (strike > spot) if kind == "call" else (strike < spot)
        if not out_of_the_money:
            dropped["in_the_money"] += 1
            continue

        # 2 and 3. Liquidity. Zero traded volume means the close is a stale
        # print from an earlier session or a theoretical value, not a price
        # anyone transacted at today. Zero open interest means nobody holds
        # the contract at all.
        if float(row["TtlTradgVol"]) == 0:
            dropped["zero_volume"] += 1
            continue
        if float(row["OpnIntrst"]) == 0:
            dropped["zero_open_interest"] += 1
            continue

        # 4. Price floor: see PRICE_FLOOR.
        close = float(row["ClsPric"])
        if close < PRICE_FLOOR:
            dropped["below_price_floor"] += 1
            continue

        # 5. The inversion itself. A nan here is a quote no volatility can
        # produce -- outside the no-arbitrage bounds -- and it is dropped and
        # counted by its reason rather than replaced with a guess.
        result = implied_vol_detail(close, spot, strike, years, rate, kind)
        if math.isnan(result.sigma) or result.sigma <= 0.0:
            dropped["no_implied_vol"] += 1
            dropped[f"status:{result.status}"] += 1
            continue

        # What one tick of price is worth in volatility points: the resolution
        # of this particular number, and the honest error bar on it.
        option_vega = vega(spot, strike, rate, result.sigma, years, kind)
        iv_per_tick = TICK / option_vega if option_vega > 0 else float("nan")

        # The same quote, inverted against the market's own forward. See the
        # docstring and ``forward_adjusted_jump``.
        forward_spot = forward * math.exp(-rate * years)
        forward_result = implied_vol_detail(close, forward_spot, strike, years,
                                            rate, kind)

        smile.append({
            "expiry": expiry,
            "days_to_expiry": days,
            "years_to_expiry": f"{years:.8f}",
            "kind": kind,
            "strike": f"{strike:.2f}",
            "moneyness": f"{strike / spot:.6f}",
            "forward_moneyness": f"{strike / forward:.6f}",
            "close": f"{close:.2f}",
            "settlement": row["SttlmPric"],
            "volume": row["TtlTradgVol"],
            "open_interest": row["OpnIntrst"],
            "implied_vol": f"{result.sigma:.8f}",
            "implied_vol_fwd": f"{forward_result.sigma:.8f}",
            "vega": f"{option_vega:.6f}",
            "iv_per_tick": f"{iv_per_tick:.8f}",
            "solver_status": result.status,
            "iterations": result.iterations,
            "newton_steps": result.newton_steps,
            "bisection_steps": result.bisection_steps,
        })

    smile.sort(key=lambda r: float(r["strike"]))
    return smile, dropped


# ---------------------------------------------------------------------------
# Reading the smile off in numbers
# ---------------------------------------------------------------------------

def interpolate(smile: list[dict], target: float, kind: str) -> float:
    """Implied volatility at a given moneyness, linearly between strikes.

    One side of the smile at a time, because the two sides are different
    options: below 1.0 the quotes are puts, above it they are calls. Linear
    interpolation is used rather than a fitted curve so the reported numbers
    stay a direct reading of the data -- no shape is assumed, and nothing is
    extrapolated beyond the strikes that exist.
    """
    points = sorted((float(r["moneyness"]), float(r["implied_vol"]))
                    for r in smile if r["kind"] == kind)
    if len(points) < 2:
        return float("nan")
    if not points[0][0] <= target <= points[-1][0]:
        return float("nan")

    for (m0, v0), (m1, v1) in zip(points, points[1:]):
        if m0 <= target <= m1:
            if m1 == m0:
                return v0
            weight = (target - m0) / (m1 - m0)
            return v0 + weight * (v1 - v0)
    return float("nan")


def at_the_money_edges(smile: list[dict]) -> tuple[float, float]:
    """The two one-sided limits of the smile at moneyness 1.0.

    The highest-strike put and the lowest-strike call: the last point of the
    put branch and the first point of the call branch. They are the closest
    thing the data has to "the at-the-money volatility", and reporting both
    rather than averaging them is the point -- the gap between them is a
    measurement, and the forward diagnostic below explains it.
    """
    puts = [r for r in smile if r["kind"] == "put"]
    calls = [r for r in smile if r["kind"] == "call"]
    put_side = float(puts[-1]["implied_vol"]) if puts else float("nan")
    call_side = float(calls[0]["implied_vol"]) if calls else float("nan")
    return put_side, call_side


def parity_forward(options: list[dict], expiry: str, spot: float, rate: float,
                   years: float, n_strikes: int = 10) -> float:
    """The forward price the option market itself is quoting.

    Put-call parity is model-free -- it follows from the payoffs alone, with no
    volatility, no distribution and no Black-Scholes::

        C - P = exp(-r*T) * (F - K)     =>     F = K + (C - P) * exp(r*T)

    So every strike where both a call and a put traded gives an independent
    reading of the forward, and they should all agree. Averaging the ten
    nearest the money keeps the reading on the most liquid pairs, where the
    closes are least stale.

    This is the diagnostic that explains the step in the middle of the smile.
    Black-Scholes with no dividends assumes the forward is ``S*exp(r*T)``; if
    the market is trading a different forward, then the strike that is
    genuinely at the money is not the one at ``K = S*exp(r*T)``, and the put
    and call branches get read off at slightly different moneyness -- which
    shows up as a discontinuity.
    """
    paired: dict[float, dict] = {}
    for row in options:
        if row["XpryDt"] != expiry or float(row["TtlTradgVol"]) == 0:
            continue
        paired.setdefault(float(row["StrkPric"]), {})[row["OptnTp"]] = row

    both = [(abs(strike - spot), strike, legs) for strike, legs in paired.items()
            if "CE" in legs and "PE" in legs]
    both.sort()
    if not both:
        return float("nan")

    forwards = []
    for _, strike, legs in both[:n_strikes]:
        call = float(legs["CE"]["ClsPric"])
        put = float(legs["PE"]["ClsPric"])
        forwards.append(strike + (call - put) * math.exp(rate * years))
    return sum(forwards) / len(forwards)


def forward_adjusted_jump(smile: list[dict], spot: float, rate: float,
                          years: float, forward: float) -> float:
    """Re-invert the two at-the-money quotes against the market's forward.

    Black-Scholes only ever uses the spot through the forward it implies, so
    replacing ``S`` by ``F*exp(-r*T)`` prices against the forward the market is
    actually trading instead of the dividend-free one the model assumes. It is
    exactly the Black-76 adjustment, and it needs no new code -- just a
    different number in the existing ``S`` slot.

    If the step in the middle of the smile is caused by the forward, this
    closes it. If it is something else, it will not, and that is worth
    knowing. Reported as a diagnostic only: the smile itself stays on the
    stated inputs.
    """
    puts = [r for r in smile if r["kind"] == "put"]
    calls = [r for r in smile if r["kind"] == "call"]
    if not puts or not calls or math.isnan(forward):
        return float("nan")

    effective_spot = forward * math.exp(-rate * years)
    edges = []
    for row in (puts[-1], calls[0]):
        result = implied_vol_detail(float(row["close"]), effective_spot,
                                    float(row["strike"]), years, rate,
                                    row["kind"])
        edges.append(result.sigma)
    return edges[0] - edges[1]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_csv(name: str, rows: list[dict], fieldnames=None) -> None:
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def main() -> int:
    if not BHAVCOPY.exists():
        print(f"missing {BHAVCOPY.relative_to(ROOT)}")
        print("run: python analysis/fetch_phase6_data.py 2026-11-16")
        return 1

    options, spot, futures, trade_date = load_chain(BHAVCOPY)
    if trade_date != TRADE_DATE:
        raise ValueError(f"file is for {trade_date}, expected {TRADE_DATE}")

    print(f"NIFTY option chain, {trade_date.isoformat()} "
          f"({trade_date.strftime('%A')})")
    print(f"  spot close                {spot:,.2f}")
    print(f"  risk-free rate (cc)       {RISK_FREE_RATE:.6f}  "
          f"(91-day T-bill, cut-off price {TBILL_PRICE})")
    print(f"  option contracts in file  {len(options):,}")
    print(f"  price floor               {PRICE_FLOOR:.2f} "
          f"({PRICE_FLOOR / TICK:.0f} ticks)")

    expiries = monthly_expiries(options, futures)
    print(f"  monthly expiries          {', '.join(expiries)}")

    all_rows: list[dict] = []
    filter_rows: list[dict] = []
    summary_rows: list[dict] = []

    for expiry in expiries:
        days = (date.fromisoformat(expiry) - TRADE_DATE).days
        years = days / DAYS_PER_YEAR
        market_forward = parity_forward(options, expiry, spot,
                                        RISK_FREE_RATE, years)

        smile, dropped = build_smile(options, expiry, spot, RISK_FREE_RATE,
                                     market_forward)
        all_rows.extend(smile)
        started = sum(1 for r in options if r["XpryDt"] == expiry)

        print(f"\n{expiry}  ({days} days, T = {years:.6f})")
        print(f"  contracts listed          {started}")
        for reason in FILTER_ORDER:
            print(f"  dropped: {reason:<22} {dropped[reason]}")
        for status, count in sorted(dropped.items()):
            if status.startswith("status:"):
                print(f"      of which {status[7:]:<17} {count}")
        print(f"  kept                      {len(smile)}")

        for reason in FILTER_ORDER:
            filter_rows.append({
                "expiry": expiry,
                "contracts_listed": started,
                "filter": reason,
                "dropped": dropped[reason],
                "kept_after": started - sum(dropped[f] for f in
                                            FILTER_ORDER[:FILTER_ORDER.index(reason) + 1]),
            })

        if not smile:
            continue

        put_side, call_side = at_the_money_edges(smile)
        spot_forward = spot * math.exp(RISK_FREE_RATE * years)
        iv_095 = interpolate(smile, 0.95, "put")
        iv_105 = interpolate(smile, 1.05, "call")

        iterations = [int(r["iterations"]) for r in smile]
        bisections = sum(int(r["bisection_steps"]) for r in smile)
        newtons = sum(int(r["newton_steps"]) for r in smile)
        ticks = [float(r["iv_per_tick"]) for r in smile]

        summary_rows.append({
            "expiry": expiry,
            "days_to_expiry": days,
            "years_to_expiry": f"{years:.8f}",
            "strikes_kept": len(smile),
            "moneyness_min": min(float(r["moneyness"]) for r in smile),
            "moneyness_max": max(float(r["moneyness"]) for r in smile),
            "iv_at_0_95": f"{iv_095:.6f}",
            "iv_atm_put_side": f"{put_side:.6f}",
            "iv_atm_call_side": f"{call_side:.6f}",
            "iv_atm_midpoint": f"{0.5 * (put_side + call_side):.6f}",
            "iv_at_1_05": f"{iv_105:.6f}",
            "skew_095_minus_105": f"{iv_095 - iv_105:.6f}",
            "atm_jump_put_minus_call": f"{put_side - call_side:.6f}",
            "spot_forward": f"{spot_forward:.4f}",
            "parity_forward": f"{market_forward:.4f}",
            "futures_close": f"{futures[expiry]:.4f}",
            "forward_gap_points": f"{market_forward - spot_forward:.4f}",
            "atm_jump_forward_adjusted": f"{forward_adjusted_jump(smile, spot, RISK_FREE_RATE, years, market_forward):.6f}",
            "mean_iterations": f"{sum(iterations) / len(iterations):.3f}",
            "max_iterations": max(iterations),
            "newton_steps": newtons,
            "bisection_steps": bisections,
            "median_iv_per_tick": f"{sorted(ticks)[len(ticks) // 2]:.6f}",
            "max_iv_per_tick": f"{max(ticks):.6f}",
        })

    print()
    write_csv("iv_smile.csv", all_rows)
    write_csv("iv_filter_log.csv", filter_rows)
    write_csv("iv_term_structure.csv", summary_rows)
    print("\nnow run: python analysis/plot_phase6.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
