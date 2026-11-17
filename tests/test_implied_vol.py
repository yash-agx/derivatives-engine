"""Tests for the implied-volatility solver.

Inverting a function is easy to get *nearly* right, so the checks are built
around what "right" actually means here:

1. **Round trip.** Price at a known sigma, invert the price, get the sigma
   back. This is the only test that can catch a solver that converges
   confidently to the wrong root, and it is run over a grid of moneyness,
   maturity, volatility and option type.
2. **No-arbitrage edges.** A price no volatility can produce must come back as
   ``nan``, not as a clipped or capped number. Both ends, both option types,
   and the exact-boundary cases.
3. **The hard corners.** Deep in and out of the money, and expiries down to an
   hour -- the places where vega collapses and plain Newton-Raphson diverges.
4. **Identities.** The answer must not depend on whether the quote is read as
   a call or as the put that put-call parity makes equivalent to it.

Note the argument order: ``implied_vol(price, S, K, T, r, kind)``, price
first, because the price is what is being inverted.
"""

import math

import pytest

from pricing.black_scholes import CALL, PUT, call_price, price, put_price, vega
from pricing.implied_vol import (
    SIGMA_MAX,
    implied_vol,
    implied_vol_detail,
    no_arbitrage_bounds,
)

ONE_HOUR = 1.0 / (365.0 * 24.0)
ONE_DAY = 1.0 / 365.0
ONE_WEEK = 7.0 / 365.0

#: How far a strike is from the forward, measured in standard deviations of
#: the log return. This is the right ruler for "is this quote invertible?",
#: and moneyness on its own is not: ``K/S = 0.6`` is a routine strike at 25%
#: vol with a year to run (3.7 sd) and an impossible one at 5% vol with a week
#: to run (74 sd), where the option is worth discounted intrinsic value plus
#: 1e-80 and the premium that carries the volatility information has
#: underflowed. Beyond this cut the limit is the *price*, not the solver --
#: see the identifiability tests at the bottom.
MAX_STANDARD_DEVIATIONS = 5.0


def standard_deviations_from_forward(S, K, r, sigma, T):
    """``|ln(K/F)| / (sigma*sqrt(T))`` -- strike distance on the model's ruler."""
    forward = S * math.exp(r * T)
    return abs(math.log(K / forward)) / (sigma * math.sqrt(T))


def invertible(grid):
    """Keep the grid points where the premium is still representable."""
    return [row for row in grid
            if standard_deviations_from_forward(*row[:5]) <= MAX_STANDARD_DEVIATIONS]


#: A textbook-scale grid: in, at and out of the money, a week to five years,
#: calm to wild vols, and a negative-to-high range of rates.
ROUND_TRIP_GRID = invertible([
    (S, K, r, sigma, T, kind)
    for S in (100.0,)
    for K in (60.0, 85.0, 100.0, 115.0, 160.0)
    for r in (0.0, 0.058, 0.10)
    for sigma in (0.05, 0.12, 0.25, 0.60, 1.50)
    for T in (ONE_WEEK, 0.1, 1.0, 5.0)
    for kind in (CALL, PUT)
])

#: The Phase 6 data, in round numbers: Nifty 50 near 25,300, strikes on a
#: 500-point grid, a day to six months, vols in the low teens. Scale matters --
#: an index four orders of magnitude larger than a textbook stock is where an
#: absolute price tolerance would quietly stop meaning anything.
NIFTY_GRID = invertible([
    (25_300.0, K, 0.0583, sigma, T, kind)
    for K in (24_000.0, 25_000.0, 25_300.0, 25_600.0, 26_500.0)
    for sigma in (0.08, 0.11, 0.14, 0.22, 0.40)
    for T in (ONE_DAY, ONE_WEEK, 43.0 / 365.0, 0.5)
    for kind in (CALL, PUT)
])


def test_the_grids_are_not_empty_and_are_not_everything():
    """Guard the guard: the filter must bite, but not swallow the whole grid.

    If a future edit made ``invertible`` a no-op, or made it reject
    everything, every other test here would still pass while testing nothing.
    """
    assert 200 < len(ROUND_TRIP_GRID) < 600
    assert 100 < len(NIFTY_GRID) < 200


# ---------------------------------------------------------------------------
# 1. Round trip: recover the volatility that generated the price
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T,kind", ROUND_TRIP_GRID)
def test_round_trip_recovers_sigma(S, K, r, sigma, T, kind):
    """The headline requirement: sigma back to within 1e-6.

    The tolerance is absolute and in volatility points, so 1e-6 is a ten
    thousandth of a vol point -- far finer than any market quote resolves,
    and the solver's own tolerance is tighter again (1e-12 relative).
    """
    quote = price(S, K, r, sigma, T, kind)
    recovered = implied_vol(quote, S, K, T, r, kind)
    assert recovered == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("S,K,r,sigma,T,kind", NIFTY_GRID)
def test_round_trip_at_index_scale(S, K, r, sigma, T, kind):
    """Same test at Nifty 50 scale, where prices are 10^4 rather than 10^2."""
    quote = price(S, K, r, sigma, T, kind)
    recovered = implied_vol(quote, S, K, T, r, kind)
    assert recovered == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("S,K,r,sigma,T,kind", ROUND_TRIP_GRID)
def test_round_trip_reprices_the_quote(S, K, r, sigma, T, kind):
    """And the recovered sigma must reprice the quote.

    A weaker condition than the one above -- many wrong sigmas can reprice a
    quote where vega is small -- but it is the condition the solver actually
    enforces, so it should hold to much tighter than the vol tolerance.
    """
    quote = price(S, K, r, sigma, T, kind)
    recovered = implied_vol(quote, S, K, T, r, kind)
    assert price(S, K, r, recovered, T, kind) == pytest.approx(quote,
                                                              rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("S,K,r,sigma,T,kind", ROUND_TRIP_GRID)
def test_solver_reports_success_and_stays_in_range(S, K, r, sigma, T, kind):
    """Status ``ok``, a sigma inside the search range, and a bounded cost."""
    quote = price(S, K, r, sigma, T, kind)
    result = implied_vol_detail(quote, S, K, T, r, kind)
    assert result.status == "ok"
    assert 0.0 <= result.sigma <= SIGMA_MAX
    assert result.iterations <= 100


def test_most_quotes_cost_fewer_than_ten_evaluations():
    """Newton should be doing the work, not the bisection safety net.

    Not a correctness check -- a check that the safeguard has not silently
    taken over, which would still give right answers at ten times the cost.
    """
    counts = []
    for S, K, r, sigma, T, kind in ROUND_TRIP_GRID:
        quote = price(S, K, r, sigma, T, kind)
        counts.append(implied_vol_detail(quote, S, K, T, r, kind).iterations)
    assert sum(counts) / len(counts) < 10.0


def test_implied_vol_is_increasing_in_the_quote():
    """A dearer option implies a higher volatility, always.

    This is the monotonicity the bracket relies on. If it failed, the root
    would not be unique and no bracketing method would be meaningful.
    """
    S, K, r, T = 100.0, 105.0, 0.05, 0.5
    quotes = [price(S, K, r, s, T, CALL) for s in (0.05, 0.1, 0.2, 0.4, 0.8)]
    vols = [implied_vol(q, S, K, T, r, CALL) for q in quotes]
    assert vols == sorted(vols)
    assert all(b > a for a, b in zip(vols, vols[1:]))


# ---------------------------------------------------------------------------
# 2. No-arbitrage edges: nan, never a made-up number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", [CALL, PUT])
def test_bounds_are_the_sigma_limits(kind):
    """The bounds must equal the price at zero and at infinite volatility."""
    S, K, r, T = 100.0, 90.0, 0.05, 1.0
    lower, upper = no_arbitrage_bounds(S, K, r, T, kind)

    assert price(S, K, r, 0.0, T, kind) == pytest.approx(lower)
    # sigma -> infinity, approached from a very large but finite value.
    assert price(S, K, r, 50.0, T, kind) == pytest.approx(upper, abs=1e-9)


def test_lower_bound_uses_the_discounted_strike():
    """``S - K*exp(-r*T)``, not ``S - K``, and the difference cuts both ways.

    For a **call** the discounted strike is smaller, so the bound is *higher*
    than undiscounted intrinsic value: a European call must be worth more than
    ``S - K``, because the strike is not paid until expiry. Using ``S - K``
    here would accept arbitrageable quotes.

    For a **put** it is the other way round -- the bound is *lower* than
    ``K - S``, so a deep in-the-money European put legitimately trades below
    undiscounted intrinsic value, and using ``K - S`` would reject real
    quotes. Both halves are tested because the two mistakes have opposite
    symptoms.
    """
    S, K, r, T = 100.0, 90.0, 0.10, 2.0
    discounted_k = K * math.exp(-r * T)

    call_lower, _ = no_arbitrage_bounds(S, K, r, T, CALL)
    assert call_lower == pytest.approx(S - discounted_k)
    assert call_lower > S - K

    S_itm_put = 50.0
    put_lower, _ = no_arbitrage_bounds(S_itm_put, K, r, T, PUT)
    assert put_lower == pytest.approx(discounted_k - S_itm_put)
    assert put_lower < K - S_itm_put


def test_bounds_satisfy_put_call_parity():
    """The two lower bounds differ by exactly the parity amount."""
    S, K, r, T = 100.0, 120.0, 0.07, 1.5
    call_lower, _ = no_arbitrage_bounds(S, K, r, T, CALL)
    put_lower, _ = no_arbitrage_bounds(S, K, r, T, PUT)
    assert call_lower - put_lower == pytest.approx(S - K * math.exp(-r * T))


#: Spots that put each option type deep *in* the money, so its lower bound is
#: strictly positive and "below the bound" is a meaningful place to be. A
#: single spot would give one of the two a zero bound and test nothing.
IN_THE_MONEY = {CALL: 100.0, PUT: 60.0}


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_price_below_lower_bound_is_nan(kind):
    """Below intrinsic value there is an arbitrage, and no implied vol."""
    S, K, r, T = IN_THE_MONEY[kind], 80.0, 0.06, 1.0
    lower, _ = no_arbitrage_bounds(S, K, r, T, kind)
    assert lower > 0.0

    assert math.isnan(implied_vol(lower * 0.99, S, K, T, r, kind))
    assert implied_vol_detail(lower * 0.99, S, K, T, r,
                              kind).status == "below_lower_bound"


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_negative_price_is_nan(kind):
    """Negative premiums cannot be inverted (and cannot exist)."""
    assert math.isnan(implied_vol(-1.0, 100.0, 100.0, 1.0, 0.05, kind))


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_price_above_upper_bound_is_nan(kind):
    """A call dearer than the share, or a put dearer than the strike's PV."""
    S, K, r, T = 100.0, 110.0, 0.05, 1.0
    _, upper = no_arbitrage_bounds(S, K, r, T, kind)
    assert math.isnan(implied_vol(upper * 1.01, S, K, T, r, kind))
    assert implied_vol_detail(upper * 1.01, S, K, T, r,
                              kind).status == "above_upper_bound"


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_price_exactly_at_upper_bound_is_nan(kind):
    """The upper bound is the sigma -> infinity limit: approached, never hit.

    So there is no *finite* volatility that produces it, and the honest answer
    is nan rather than ``SIGMA_MAX``.
    """
    S, K, r, T = 100.0, 110.0, 0.05, 1.0
    _, upper = no_arbitrage_bounds(S, K, r, T, kind)
    assert math.isnan(implied_vol(upper, S, K, T, r, kind))


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_price_exactly_at_lower_bound_implies_zero_vol(kind):
    """On the lower bound the answer is 0, and it is reported as such.

    Not nan: the quote *is* the zero-volatility price, and "the market is
    pricing no uncertainty" is a real statement about it. The status flags it
    so an analysis can treat it separately from a genuine inversion.
    """
    S, K, r, T = IN_THE_MONEY[kind], 80.0, 0.06, 1.0
    lower, _ = no_arbitrage_bounds(S, K, r, T, kind)
    assert lower > 0.0

    result = implied_vol_detail(lower, S, K, T, r, kind)
    assert result.sigma == 0.0
    assert result.status == "at_lower_bound"


def test_volatility_above_the_cap_is_nan_not_clipped():
    """A quote needing more than ``SIGMA_MAX`` is reported, not squashed.

    A short expiry is used so that a 1,200% volatility still leaves the price
    well inside the upper bound -- at one year it would saturate at ``S`` and
    be rejected as above the upper bound instead, which is a different
    (equally correct) answer and not what this test is about.
    """
    S, K, r, T = 100.0, 100.0, 0.0, 0.01
    quote = call_price(S, K, r, SIGMA_MAX * 1.2, T)
    _, upper = no_arbitrage_bounds(S, K, r, T, CALL)
    assert quote < 0.9 * upper

    result = implied_vol_detail(quote, S, K, T, r, CALL)
    assert math.isnan(result.sigma)
    assert result.status == "above_sigma_max"


def test_volatility_just_below_the_cap_is_still_found():
    """The cap is tested, not just overshot: 950% vol must still invert.

    Guards against the off-by-one where doubling the bracket from 100% jumps
    1, 2, 4, 8, 16 and declares failure at 16 without ever evaluating 10.
    """
    S, K, r, T = 100.0, 100.0, 0.0, 1.0
    quote = call_price(S, K, r, 9.5, T)
    assert implied_vol(quote, S, K, T, r, CALL) == pytest.approx(9.5, abs=1e-6)


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_zero_time_to_expiry_is_nan(kind):
    """At expiry the payoff is settled, so the price says nothing about vol."""
    result = implied_vol_detail(5.0, 100.0, 95.0, 0.0, 0.05, kind)
    assert math.isnan(result.sigma)
    assert result.status == "zero_time"


# ---------------------------------------------------------------------------
# Invalid inputs raise rather than returning nan
# ---------------------------------------------------------------------------
#
# The distinction matters: nan means "this quote has no implied volatility",
# which is information. A negative spot means the caller has a bug.

@pytest.mark.parametrize("bad_price", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_price_raises(bad_price):
    with pytest.raises(ValueError, match="finite"):
        implied_vol(bad_price, 100.0, 100.0, 1.0, 0.05, CALL)


@pytest.mark.parametrize("override", [
    {"S": 0.0}, {"S": -100.0}, {"K": 0.0}, {"K": -100.0}, {"T": -1.0},
])
def test_invalid_market_inputs_raise(override):
    args = {"price": 5.0, "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05,
            "kind": CALL}
    args.update(override)
    with pytest.raises(ValueError):
        implied_vol(**args)


def test_unknown_option_type_raises():
    with pytest.raises(ValueError, match="call"):
        implied_vol(5.0, 100.0, 100.0, 1.0, 0.05, "straddle")


# ---------------------------------------------------------------------------
# 3. The hard corners: tiny vega
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [ONE_HOUR, ONE_DAY, 2.0 * ONE_DAY, ONE_WEEK])
@pytest.mark.parametrize("sigma", [0.10, 0.25, 0.80])
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_short_expiry_round_trip(T, sigma, kind):
    """An hour to expiry still inverts, at the money where vega is largest."""
    S, K, r = 100.0, 100.0, 0.05
    quote = price(S, K, r, sigma, T, kind)
    assert implied_vol(quote, S, K, T, r, kind) == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("K", [40.0, 60.0, 80.0, 125.0, 170.0, 250.0])
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_deep_in_and_out_of_the_money_round_trip(K, kind):
    """Moneyness from 0.4 to 2.5 at one year, where vega is small but real."""
    S, r, sigma, T = 100.0, 0.05, 0.25, 1.0
    quote = price(S, K, r, sigma, T, kind)
    assert implied_vol(quote, S, K, T, r, kind) == pytest.approx(sigma, abs=1e-6)


def test_bisection_rescues_newton_somewhere_on_the_grid():
    """The fallback is not decoration: some real case has to need it.

    If nothing on a grid this wide ever triggers bisection, the safeguard is
    untested code and this test should fail so that it gets noticed.
    """
    fallbacks = 0
    for S, K, r, sigma, T, kind in ROUND_TRIP_GRID + NIFTY_GRID:
        quote = price(S, K, r, sigma, T, kind)
        fallbacks += implied_vol_detail(quote, S, K, T, r, kind).bisection_steps
    assert fallbacks > 0


def test_tiny_vega_quote_does_not_diverge():
    """The case plain Newton-Raphson gets wrong.

    A far out-of-the-money option one day from expiry has a vega of order
    1e-9 per vol point. An unguarded ``sigma - f/vega`` from a starting guess
    of 100% vol lands thousands of vol points away or negative; the bracket
    makes that impossible by construction.
    """
    S, K, r, sigma, T = 100.0, 160.0, 0.05, 0.30, ONE_DAY
    assert vega(S, K, r, sigma, T, CALL) < 1e-6
    quote = call_price(S, K, r, sigma, T)
    result = implied_vol_detail(quote, S, K, T, r, CALL)
    assert 0.0 <= result.sigma <= SIGMA_MAX
    assert not math.isnan(result.sigma)


# ---------------------------------------------------------------------------
# 4. Identities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.1, 0.5, 2.0])
def test_call_and_put_quotes_imply_the_same_volatility(K, T):
    """Put-call parity means a call quote and its parity put quote are the
    same piece of information, so they must invert to the same sigma.

    ::

        C - P = S - K*exp(-r*T)

    This is the strongest consistency check available without a second
    implementation: it exercises the call branch and the put branch of both
    the pricer and the solver and demands they agree, with no reference value
    needed.
    """
    S, r, sigma = 100.0, 0.06, 0.28
    call_quote = call_price(S, K, r, sigma, T)
    put_quote = call_quote - (S - K * math.exp(-r * T))

    assert put_quote == pytest.approx(put_price(S, K, r, sigma, T))
    call_iv = implied_vol(call_quote, S, K, T, r, CALL)
    put_iv = implied_vol(put_quote, S, K, T, r, PUT)
    assert call_iv == pytest.approx(put_iv, abs=1e-8)
    assert call_iv == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("r", [-0.02, 0.0, 0.0583, 0.15])
def test_inversion_works_at_any_rate(r):
    """Including a negative rate, which the rest of the project also allows."""
    S, K, sigma, T = 100.0, 105.0, 0.22, 1.0
    quote = call_price(S, K, r, sigma, T)
    assert implied_vol(quote, S, K, T, r, CALL) == pytest.approx(sigma, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Identifiability: where the *price*, not the solver, loses the answer
# ---------------------------------------------------------------------------
#
# These document a real limit rather than a bug, and they are here so the
# limit is pinned down instead of being discovered on live data.

def test_a_price_rounded_to_the_bound_implies_zero_vol():
    """Beyond some point the price is the zero-vol price to machine precision.

    A deep in-the-money call three days from expiry is worth discounted
    intrinsic value plus a few times 1e-13. Stored as a double -- let alone
    quoted in exchange ticks of 0.05 -- that premium is gone, and with it any
    information about volatility. The solver says 0 and flags it rather than
    inventing the 25% that produced the number.
    """
    S, K, r, sigma, T = 100.0, 20.0, 0.05, 0.25, 3.0 * ONE_DAY
    quote = call_price(S, K, r, sigma, T)
    lower, _ = no_arbitrage_bounds(S, K, r, T, CALL)
    assert quote - lower < 1e-10          # the premium has vanished

    result = implied_vol_detail(quote, S, K, T, r, CALL)
    assert result.status == "at_lower_bound"
    assert result.sigma == 0.0


def test_tick_rounding_is_the_real_precision_limit():
    """What a 0.05 price tick costs in volatility points, at index scale.

    Nothing about the solver changes this: it is ``tick / vega``. For the
    at-the-money Nifty weekly it is a few hundredths of a vol point, which is
    why the Phase 6 smile is reported to one decimal place and not three.
    """
    S, K, r, sigma, T = 25_300.0, 25_300.0, 0.0583, 0.12, ONE_WEEK
    quote = call_price(S, K, r, sigma, T)
    up = implied_vol(quote + 0.05, S, K, T, r, CALL)
    down = implied_vol(quote - 0.05, S, K, T, r, CALL)
    spread = up - down
    assert 0.0 < spread < 0.01            # under one vol point, but not zero
    assert spread == pytest.approx(0.10 / vega(S, K, r, sigma, T, CALL), rel=0.05)
