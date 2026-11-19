"""Phase 8: validate the engine against QuantLib.

    python analysis/run_phase8_quantlib.py          (about 2 minutes)

Every earlier phase checked this engine against itself: the tree against
Black-Scholes, Monte Carlo against Black-Scholes, LSM against the tree. If one
shared misunderstanding sat underneath all of them, those checks would agree
with each other and still be wrong. QuantLib is written by other people, in
another language, from other references, so agreement with it is evidence of
a different kind.

The grid: S in {80, 100, 120}, K = 100, T in {0.25, 1, 2}, sigma in
{0.15, 0.3, 0.5}, r = 0.05, q = 0, calls and puts -- 54 options (27 puts).

a. Black-Scholes          vs QuantLib AnalyticEuropeanEngine
b. CRR tree, American     vs QuantLib BinomialVanillaEngine("crr"), same steps
c. CRR tree, American     vs QuantLib FdBlackScholesVanillaEngine, fine grid
d. Longstaff-Schwartz put vs the same FD reference, in standard errors
e. implied volatility     vs QuantLib impliedVolatility(), on the 417 NSE quotes

Two diagnostic rows are added because they explain the main ones:

* ``crr_linearised_p`` -- the tree in (b) does **not** match to 1e-6, and the
  reason is a convention, not a bug. QuantLib's "crr" tree uses the
  *linearised* up-probability ``p = 1/2 + 1/2 (r - sigma^2/2) dt / dx``; ours
  uses the exact no-arbitrage one ``p = (e^{r dt} - d) / (u - d)``. Both are
  O(dt) approximations and converge to the same limit. Re-running our own
  backward induction with QuantLib's ``p`` reproduces QuantLib to rounding,
  which proves the whole gap is that one line.
* ``fd_european_vs_bs`` -- the FD reference's own error, measured where the
  truth is known. American FD is less accurate than this (exercise is only
  checked on the time grid, so FD is itself a Bermudan with that many dates),
  which is why the time grid is far finer than the space grid.

Time is put on QuantLib's date axis exactly: the grid uses Actual/360, so
T = 0.25, 1, 2 are 90, 360, 720 whole days; the NSE quotes use Actual/365
with their integer day counts, which is the convention Phase 6 used.

Output: ``results/quantlib_parity.csv``, one row per comparison.
"""

import csv
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import QuantLib as ql

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # the package is not installed; see pytest.ini
sys.path.insert(0, str(ROOT / "analysis"))

from pricing.binomial import american_binomial                        # noqa: E402
from pricing.black_scholes import price as black_scholes              # noqa: E402
from pricing.common import payoff                                     # noqa: E402
from pricing.implied_vol import implied_vol                           # noqa: E402
from pricing.lsm import lsm_price                                     # noqa: E402

RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# The grid and the settings, fixed here
# ---------------------------------------------------------------------------
K, R = 100.0, 0.05
SPOTS = (80.0, 100.0, 120.0)
MATURITIES = (0.25, 1.0, 2.0)
VOLS = (0.15, 0.30, 0.50)
KINDS = ("call", "put")
GRID = [(S, T, sigma, kind) for S in SPOTS for T in MATURITIES
        for sigma in VOLS for kind in KINDS]

TREE_STEPS = 500             # (b) same step count on both sides
REFERENCE_TREE_STEPS = 10_000  # (c) our American price against FD
FD_TIME_STEPS = 20_000       # (c, d) American FD error is set by the time grid
FD_SPACE_STEPS = 2_000
LSM_PATHS = 100_000          # (d) Phase 7 settings: 50 dates, Laguerre degree 3
LSM_DATES = 50
LSM_SEED = 12345

TODAY = ql.Date(16, 11, 2026)
GRID_DAY_COUNT = ql.Actual360()    # 0.25, 1, 2 years = 90, 360, 720 days exactly
NSE_DAY_COUNT = ql.Actual365Fixed()  # Phase 6: calendar days / 365


def _set_today() -> None:
    ql.Settings.instance().evaluationDate = TODAY


def _maturity(T: float, day_count) -> ql.Date:
    days = T * (360.0 if day_count == GRID_DAY_COUNT else 365.0)
    if abs(days - round(days)) > 1e-9:
        raise ValueError(f"T = {T} is not a whole number of days under {day_count}")
    return TODAY + int(round(days))


def ql_process(S: float, r: float, sigma: float, day_count=GRID_DAY_COUNT):
    """A Black-Scholes-Merton process with flat r, zero dividend yield, flat vol."""
    _set_today()
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, 0.0, day_count, ql.Continuous)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, day_count, ql.Continuous)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(TODAY, ql.NullCalendar(), sigma, day_count)),
    )


def ql_option(K: float, T: float, kind: str, american: bool, day_count=GRID_DAY_COUNT):
    maturity = _maturity(T, day_count)
    option_type = ql.Option.Call if kind == "call" else ql.Option.Put
    exercise = (ql.AmericanExercise(TODAY, maturity) if american
                else ql.EuropeanExercise(maturity))
    return ql.VanillaOption(ql.PlainVanillaPayoff(option_type, K), exercise)


def ql_price(S, K, r, sigma, T, kind, american, engine_factory) -> float:
    process = ql_process(S, r, sigma)
    option = ql_option(K, T, kind, american)
    option.setPricingEngine(engine_factory(process))
    return option.NPV()


def analytic_engine(process):
    return ql.AnalyticEuropeanEngine(process)


def binomial_engine(steps: int):
    return lambda process: ql.BinomialVanillaEngine(process, "crr", steps)


def fd_engine(time_steps: int = FD_TIME_STEPS, space_steps: int = FD_SPACE_STEPS):
    return lambda process: ql.FdBlackScholesVanillaEngine(
        process, time_steps, space_steps, 0, ql.FdmSchemeDesc.Douglas())


# ---------------------------------------------------------------------------
# The diagnostic tree: ours, with QuantLib's probability
# ---------------------------------------------------------------------------

def crr_linearised_p(S, K, r, sigma, T, steps, kind, american=True) -> float:
    """``pricing.binomial.binomial_price`` with one line changed.

    Same lattice (``dx = sigma*sqrt(dt)``, ``d = 1/u``), same discounting,
    same backward induction. Only ``p`` differs: QuantLib's CoxRossRubinstein
    tree takes the drift of the *log* price, ``r - sigma^2/2``, and matches
    its mean to first order, ``p = 1/2 + 1/2 * drift * dt / dx``, where ours
    matches the mean of the price itself exactly. Diagnostic only -- it lives
    here, not in ``src/``, because it is not how the engine prices.
    """
    dt = T / steps
    dx = sigma * math.sqrt(dt)
    p = 0.5 + 0.5 * (r - 0.5 * sigma * sigma) * dt / dx
    discount = math.exp(-r * dt)
    values = payoff(S * np.exp(dx * (steps - 2 * np.arange(steps + 1))), K, kind)
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            spots = S * np.exp(dx * (step - 2 * np.arange(step + 1)))
            values = np.maximum(values, payoff(spots, K, kind))
    return float(values[0])


# ---------------------------------------------------------------------------
# The comparisons. Each returns one record per option.
# ---------------------------------------------------------------------------

def _record(label, ours, reference, **extra) -> dict:
    return {"option": label, "ours": ours, "reference": reference,
            "abs_error": abs(ours - reference), **extra}


def _label(S, T, sigma, kind) -> str:
    return f"{kind} S={S:g} T={T:g} sigma={sigma:g}"


def compare_european(grid=GRID) -> list[dict]:
    return [_record(_label(S, T, sigma, kind),
                    black_scholes(S, K, R, sigma, T, kind),
                    ql_price(S, K, R, sigma, T, kind, False, analytic_engine))
            for S, T, sigma, kind in grid]


def compare_binomial(grid=GRID, steps=TREE_STEPS) -> list[dict]:
    engine = binomial_engine(steps)
    return [_record(_label(S, T, sigma, kind),
                    american_binomial(S, K, R, sigma, T, steps, kind),
                    ql_price(S, K, R, sigma, T, kind, True, engine))
            for S, T, sigma, kind in grid]


def compare_linearised_tree(grid=GRID, steps=TREE_STEPS) -> list[dict]:
    engine = binomial_engine(steps)
    return [_record(_label(S, T, sigma, kind),
                    crr_linearised_p(S, K, R, sigma, T, steps, kind),
                    ql_price(S, K, R, sigma, T, kind, True, engine))
            for S, T, sigma, kind in grid]


def fd_american_references(grid=GRID) -> dict:
    """QuantLib FD American prices, keyed by (S, T, sigma, kind). Computed once."""
    engine = fd_engine()
    return {(S, T, sigma, kind): ql_price(S, K, R, sigma, T, kind, True, engine)
            for S, T, sigma, kind in grid}


def compare_fd_european(grid=GRID) -> list[dict]:
    """The FD reference's own error, where the true answer is known."""
    engine = fd_engine()
    return [_record(_label(S, T, sigma, kind),
                    ql_price(S, K, R, sigma, T, kind, False, engine),
                    black_scholes(S, K, R, sigma, T, kind))
            for S, T, sigma, kind in grid]


def compare_american_fd(fd: dict, steps=REFERENCE_TREE_STEPS) -> list[dict]:
    return [_record(_label(S, T, sigma, kind),
                    american_binomial(S, K, R, sigma, T, steps, kind), reference)
            for (S, T, sigma, kind), reference in fd.items()]


def compare_lsm(fd: dict, n_paths=LSM_PATHS) -> list[dict]:
    rows = []
    for (S, T, sigma, kind), reference in fd.items():
        if kind != "put":
            continue
        result = lsm_price(S, K, R, sigma, T, "put", n_paths=n_paths,
                           n_dates=LSM_DATES, seed=LSM_SEED)
        # When the rule exercises at t = 0 every path is worth K - S exactly,
        # so the standard error is 0 and a z-score does not exist. Those cases
        # are compared on absolute error only and counted separately.
        se = result.standard_error
        z = (result.price - reference) / se if se > 0 else math.nan
        rows.append(_record(_label(S, T, sigma, kind), result.price, reference,
                            standard_error=se, z=z,
                            exercise_now=result.exercise_now))
    return rows


def nse_quotes() -> list[dict]:
    """The 417 out-of-the-money NSE quotes of Phase 6, selected by Phase 6's own code.

    Each carries the inputs the inversion used: the quote, the strike, the
    integer days to expiry, and the spot that makes Black-Scholes price
    against the parity forward (``F * exp(-rT)``).
    """
    import run_phase6 as p6

    options, spot, _futures, trade_date = p6.load_chain(p6.BHAVCOPY)
    if trade_date != p6.TRADE_DATE:
        raise ValueError(f"bhavcopy is for {trade_date}, expected {p6.TRADE_DATE}")
    quotes = []
    for expiry in p6.monthly_expiries(options, _futures):
        days = (date.fromisoformat(expiry) - p6.TRADE_DATE).days
        years = days / p6.DAYS_PER_YEAR
        forward = p6.parity_forward(options, expiry, spot, p6.RISK_FREE_RATE, years)
        smile, _dropped = p6.build_smile(options, expiry, spot, p6.RISK_FREE_RATE, forward)
        for row in smile:
            quotes.append({
                "label": f"{row['kind']} {expiry} K={float(row['strike']):g}",
                "kind": row["kind"],
                "price": float(row["close"]),
                "strike": float(row["strike"]),
                "days": days,
                "years": years,
                "spot": forward * math.exp(-p6.RISK_FREE_RATE * years),
                "rate": p6.RISK_FREE_RATE,
            })
    return quotes


def ql_implied_vol(q: dict) -> float:
    process = ql_process(q["spot"], q["rate"], 0.2, NSE_DAY_COUNT)
    option = ql_option(q["strike"], q["days"] / 365.0, q["kind"], False, NSE_DAY_COUNT)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    # accuracy is on sigma; 1e-12 matches the engine's own SIGMA_TOL.
    return option.impliedVolatility(q["price"], process, 1e-12, 1000, 1e-7, 10.0)


def compare_implied_vol(quotes=None) -> list[dict]:
    quotes = nse_quotes() if quotes is None else quotes
    return [_record(q["label"],
                    implied_vol(q["price"], q["spot"], q["strike"], q["years"],
                                q["rate"], q["kind"]),
                    ql_implied_vol(q))
            for q in quotes]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(method: str, rows: list[dict]) -> dict:
    abs_err = np.array([r["abs_error"] for r in rows])
    rel_err = np.array([r["abs_error"] / abs(r["reference"]) for r in rows])
    summary = {
        "method": method,
        "n_options": len(rows),
        "max_abs_error": f"{abs_err.max():.3e}",
        "max_rel_error": f"{rel_err.max():.3e}",
        "mean_abs_error": f"{abs_err.mean():.3e}",
        "mean_z": "",
        "max_abs_z": "",
    }
    if rows and "z" in rows[0]:
        z = np.array([r["z"] for r in rows if not math.isnan(r["z"])])
        summary["mean_z"] = f"{z.mean():+.3f}"
        summary["max_abs_z"] = f"{np.abs(z).max():.3f}"
    return summary


def _worst(rows: list[dict], key="abs_error") -> dict:
    return max(rows, key=lambda r: abs(r[key]))


def main() -> int:
    if not (ROOT / "data" / "nifty_fo_bhavcopy_20261116.csv").exists():
        print("missing data/nifty_fo_bhavcopy_20261116.csv; run analysis/fetch_phase6_data.py")
        return 1
    print(f"QuantLib {ql.__version__}, grid of {len(GRID)} options "
          f"({sum(k == 'put' for *_, k in GRID)} puts)\n")

    summaries = []

    def run(method, compute):
        started = time.perf_counter()
        rows = compute()
        summary = summarise(method, rows)
        summaries.append(summary)
        worst = _worst(rows)
        print(f"{method:<26} n={summary['n_options']:<4} max abs {summary['max_abs_error']}  "
              f"max rel {summary['max_rel_error']}  mean abs {summary['mean_abs_error']}"
              + (f"  mean z {summary['mean_z']}  max |z| {summary['max_abs_z']}"
                 if summary["mean_z"] else "")
              + f"   ({time.perf_counter() - started:.1f} s)")
        print(f"{'':<26} worst: {worst['option']}  ours {worst['ours']:.10f}  "
              f"reference {worst['reference']:.10f}")
        return rows

    run("european_analytic", compare_european)
    run(f"binomial_crr_{TREE_STEPS}", compare_binomial)
    run(f"crr_linearised_p_{TREE_STEPS}", compare_linearised_tree)
    run("fd_european_vs_bs", compare_fd_european)

    started = time.perf_counter()
    fd = fd_american_references()
    print(f"\n(FD American references: {FD_TIME_STEPS} x {FD_SPACE_STEPS} grid, "
          f"{time.perf_counter() - started:.1f} s)")
    run(f"american_tree_{REFERENCE_TREE_STEPS}_vs_fd", lambda: compare_american_fd(fd))
    lsm_rows = run("lsm_put_vs_fd", lambda: compare_lsm(fd))
    print("\n  LSM put detail (price - FD in standard errors):")
    for row in lsm_rows:
        print(f"    {row['option']:<32} LSM {row['ours']:9.5f}  FD {row['reference']:9.5f}  "
              f"SE {row['standard_error']:.5f}  z {row['z']:+.2f}"
              + ("  (exercises at t=0)" if row["exercise_now"] else ""))
    print(f"  {sum(r['exercise_now'] for r in lsm_rows)} of {len(lsm_rows)} exercise at t=0 "
          f"(SE = 0, no z); z statistics are over the rest")
    print()
    run("implied_vol_nse", compare_implied_vol)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "quantlib_parity.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nwrote {(RESULTS / 'quantlib_parity.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
