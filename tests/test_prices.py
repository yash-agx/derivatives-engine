"""Tests for the Black-Scholes European call and put prices.

Three kinds of check, in increasing order of how much they prove:

1. Known textbook values (Hull) -- catches a wrong formula.
2. Put-call parity -- a no-arbitrage identity that must hold for *any* inputs,
   so it catches wrong-but-plausible formulas that happen to match one example.
3. Bounds and monotonicity -- the qualitative behaviour a trader would expect.
"""

import math

import pytest

from pricing.black_scholes import (
    CALL,
    PUT,
    call_price,
    d1_d2,
    price,
    put_price,
)

# A grid of inputs that every identity below is checked on: in, at and out of
# the money, short and long expiries, calm and wild vols, zero/negative rates.
GRID = [
    (S, K, r, sigma, T)
    for S in (50.0, 100.0, 150.0)
    for K in (80.0, 100.0, 120.0)
    for r in (-0.01, 0.0, 0.05, 0.10)
    for sigma in (0.05, 0.20, 0.60)
    for T in (0.02, 0.5, 2.0)
]


# ---------------------------------------------------------------------------
# 1. Known textbook values
# ---------------------------------------------------------------------------

def test_hull_d1_d2():
    """Hull's worked example: S=42, K=40, r=10%, sigma=20%, T=0.5 years.

    Hull reports d1 = 0.7693 and d2 = 0.6278.
    """
    d1, d2 = d1_d2(42.0, 40.0, 0.10, 0.20, 0.5)
    assert d1 == pytest.approx(0.7693, abs=5e-5)
    assert d2 == pytest.approx(0.6278, abs=5e-5)


def test_hull_call_and_put_price():
    """Same example: Hull reports call = 4.76, put = 0.81."""
    assert call_price(42.0, 40.0, 0.10, 0.20, 0.5) == pytest.approx(4.76, abs=0.005)
    assert put_price(42.0, 40.0, 0.10, 0.20, 0.5) == pytest.approx(0.81, abs=0.005)


def test_hull_greeks_chapter_call_price():
    """Hull's Greeks example: S=49, K=50, r=5%, sigma=20%, T=20/52 years.

    Hull reports a call value of 2.40 (used as the 100,000-option
    delta-hedging example, worth $240,000).
    """
    assert call_price(49.0, 50.0, 0.05, 0.20, 20 / 52) == pytest.approx(2.40, abs=0.005)


def test_at_the_money_call_matches_closed_form():
    """Independent cross-check with no stock drift and K = S.

    With r = 0 and K = S, d1 = sigma*sqrt(T)/2 and d2 = -sigma*sqrt(T)/2, so
    the call price is S * (2*N(sigma*sqrt(T)/2) - 1) -- a formula simple
    enough to write out by hand, and it must agree with the general code.
    """
    S, sigma, T = 100.0, 0.25, 1.5
    from scipy.stats import norm

    expected = S * (2 * norm.cdf(sigma * math.sqrt(T) / 2) - 1)
    assert call_price(S, S, 0.0, sigma, T) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Put-call parity:  C - P = S - K * exp(-r*T)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T", GRID)
def test_put_call_parity(S, K, r, sigma, T):
    """Holds by arbitrage, independent of the model.

    Buying a call and selling a put gives exactly the payoff ``S_T - K`` at
    expiry, whatever happens -- the same payoff as holding the stock and
    owing ``K``. So the two must cost the same today, which fixes
    ``C - P = S - K*exp(-r*T)``.
    """
    lhs = call_price(S, K, r, sigma, T) - put_price(S, K, r, sigma, T)
    rhs = S - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Bounds and monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T", GRID)
def test_price_bounds(S, K, r, sigma, T):
    """No-arbitrage bounds.

    A call is worth at least its discounted intrinsic value (otherwise buy it,
    sell the forward) and never more than the stock itself. A put is worth at
    least its discounted intrinsic value and never more than the discounted
    strike, which is the most it can ever pay out.
    """
    discounted_k = K * math.exp(-r * T)
    call = call_price(S, K, r, sigma, T)
    put = put_price(S, K, r, sigma, T)

    assert max(S - discounted_k, 0.0) - 1e-12 <= call <= S + 1e-12
    assert max(discounted_k - S, 0.0) - 1e-12 <= put <= discounted_k + 1e-12


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_monotonic_in_spot(kind):
    """Calls rise with the spot, puts fall. (This is the sign of delta.)"""
    prices = [price(S, 100.0, 0.05, 0.2, 1.0, kind) for S in range(60, 141, 5)]
    diffs = [b - a for a, b in zip(prices, prices[1:])]
    if kind == CALL:
        assert all(d > 0 for d in diffs)
    else:
        assert all(d < 0 for d in diffs)


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_monotonic_in_volatility(kind):
    """More volatility is worth more to the holder of either option.

    The downside is capped (you just don't exercise) while the upside is not,
    so extra uncertainty can only help. This is the sign of vega.
    """
    prices = [price(100.0, 100.0, 0.05, sigma, 1.0, kind) for sigma in
              (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_monotonic_in_expiry_at_the_money(kind):
    """At the money, more time to run is worth more (the sign of theta)."""
    prices = [price(100.0, 100.0, 0.05, 0.2, T, kind) for T in
              (0.05, 0.25, 0.5, 1.0, 2.0)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


# ---------------------------------------------------------------------------
# The kind dispatcher
# ---------------------------------------------------------------------------

def test_price_dispatcher_matches_named_functions():
    args = (105.0, 100.0, 0.03, 0.25, 0.75)
    assert price(*args, CALL) == call_price(*args)
    assert price(*args, PUT) == put_price(*args)


def test_bad_kind_raises():
    """'straddle' is a valid kind from Phase 4 on, so this uses a real non-kind."""
    with pytest.raises(ValueError, match="call.*put"):
        price(100.0, 100.0, 0.05, 0.2, 1.0, "butterfly")
