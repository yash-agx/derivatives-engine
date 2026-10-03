"""Edge cases: the inputs where a pricing library quietly breaks.

Four families, all of them things that happen in real use:

* **Very short expiry** -- an option on its expiry day, or valued at the
  moment of expiry (``T = 0``).
* **Very low / very high volatility** -- a frozen stock, or a meme stock at
  500% vol. Both ends of ``sigma`` push ``d1``/``d2`` towards infinity.
* **Deep in / out of the money** -- where ``N(d1)`` saturates at 1 or 0 and a
  naive implementation returns NaN or a negative price.
* **Invalid inputs** -- which must raise rather than return nonsense.

The general rule being tested: the price must stay inside its no-arbitrage
bounds, put-call parity must still hold, and nothing may be NaN.
"""

import math

import pytest

from pricing.black_scholes import (
    CALL,
    PUT,
    call_price,
    greeks,
    price,
    put_price,
)

ONE_HOUR = 1.0 / (365.0 * 24.0)
ONE_SECOND = 1.0 / (365.0 * 24.0 * 3600.0)

EXTREME_GRID = [
    (S, K, r, sigma, T)
    for S in (1.0, 100.0, 10_000.0)
    for K in (10.0, 100.0, 1_000.0)
    for r in (0.0, 0.05)
    for sigma in (1e-8, 1e-4, 0.2, 3.0, 10.0)
    for T in (0.0, ONE_SECOND, ONE_HOUR, 1.0, 30.0)
]


# ---------------------------------------------------------------------------
# Nothing may be NaN, and the bounds must hold, anywhere on the extreme grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,r,sigma,T", EXTREME_GRID)
def test_prices_stay_finite_and_within_bounds(S, K, r, sigma, T):
    call = call_price(S, K, r, sigma, T)
    put = put_price(S, K, r, sigma, T)
    discounted_k = K * math.exp(-r * T)
    slack = 1e-9 * max(1.0, S, discounted_k)

    assert math.isfinite(call) and math.isfinite(put)
    assert max(S - discounted_k, 0.0) - slack <= call <= S + slack
    assert max(discounted_k - S, 0.0) - slack <= put <= discounted_k + slack


@pytest.mark.parametrize("S,K,r,sigma,T", EXTREME_GRID)
def test_parity_survives_the_extremes(S, K, r, sigma, T):
    """Parity is the strongest single check, so it is worth running out here."""
    lhs = call_price(S, K, r, sigma, T) - put_price(S, K, r, sigma, T)
    rhs = S - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("S,K,r,sigma,T", EXTREME_GRID)
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_greeks_stay_finite_except_where_documented(S, K, r, sigma, T, kind):
    """No NaN anywhere.

    The single documented exception is gamma on the knife edge
    ``S == K*exp(-r*T)`` with no volatility left, where the payoff has a kink
    and the curvature really is unbounded -- there we return +inf on purpose.
    """
    for name, value in greeks(S, K, r, sigma, T, kind).items():
        assert not math.isnan(value), f"{name} is NaN"
        if math.isinf(value):
            assert name == "gamma"
            assert sigma * math.sqrt(T) < 1e-12
            assert S == pytest.approx(K * math.exp(-r * T))


# ---------------------------------------------------------------------------
# Very short expiry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [0.0, ONE_SECOND, ONE_HOUR, 1.0 / 365.0])
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_short_expiry_collapses_to_intrinsic_value(T, kind):
    """As expiry approaches, the option becomes its payoff.

    With a day or less to run there is almost no time for the stock to move,
    so the optionality is worth almost nothing and only the intrinsic value
    remains.
    """
    S, K, r, sigma = 110.0, 100.0, 0.05, 0.2
    intrinsic = max(S - K, 0.0) if kind == CALL else max(K - S, 0.0)
    assert price(S, K, r, sigma, T, kind) == pytest.approx(intrinsic, abs=0.06)


def test_price_at_expiry_is_exactly_the_payoff():
    """``T = 0`` must be exact, not approximate -- it is the payoff itself."""
    assert call_price(110.0, 100.0, 0.05, 0.2, 0.0) == 10.0
    assert call_price(90.0, 100.0, 0.05, 0.2, 0.0) == 0.0
    assert put_price(90.0, 100.0, 0.05, 0.2, 0.0) == 10.0
    assert put_price(110.0, 100.0, 0.05, 0.2, 0.0) == 0.0


def test_greeks_at_expiry_are_the_payoff_derivatives():
    """At expiry an in-the-money call is just the stock minus the strike."""
    in_the_money = greeks(110.0, 100.0, 0.05, 0.2, 0.0, CALL)
    assert in_the_money["delta"] == 1.0
    assert in_the_money["gamma"] == 0.0
    assert in_the_money["vega"] == 0.0
    assert in_the_money["rho"] == 0.0          # rho carries a factor T

    worthless = greeks(90.0, 100.0, 0.05, 0.2, 0.0, CALL)
    assert all(value == 0.0 for value in worthless.values())


def test_at_the_money_gamma_explodes_as_expiry_approaches():
    """The textbook warning about short-dated at-the-money gamma."""
    gammas = [greeks(100.0, 100.0, 0.05, 0.2, T, CALL)["gamma"]
              for T in (1.0, 0.1, 0.01, 0.001)]
    assert all(b > a for a, b in zip(gammas, gammas[1:]))
    # Gamma grows like 1/sqrt(T), so a 1000x shorter expiry multiplies it by
    # about sqrt(1000) = 32 (measured: 33.6x, the drift terms adding a little).
    assert gammas[-1] > 25 * gammas[0]


def test_at_the_money_theta_explodes_as_expiry_approaches():
    """The other side of the same coin: the last day is where value drains."""
    thetas = [greeks(100.0, 100.0, 0.05, 0.2, T, CALL)["theta"]
              for T in (1.0, 0.1, 0.01, 0.001)]
    assert all(b < a for a, b in zip(thetas, thetas[1:]))   # more and more negative


# ---------------------------------------------------------------------------
# Very low volatility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.0, 1e-12, 1e-8, 1e-4])
def test_low_volatility_gives_the_discounted_forward_payoff(sigma):
    """With no volatility the stock is a bond: it ends at the forward, period.

    So the call is worth ``max(S - K*exp(-r*T), 0)`` -- note the *discounted*
    strike, not the plain intrinsic value, because the payment happens at
    expiry.
    """
    S, K, r, T = 110.0, 100.0, 0.05, 1.0
    discounted_k = K * math.exp(-r * T)
    assert call_price(S, K, r, sigma, T) == pytest.approx(S - discounted_k, abs=1e-6)
    assert put_price(S, K, r, sigma, T) == pytest.approx(0.0, abs=1e-6)


def test_zero_volatility_out_of_the_money_is_worthless():
    """No volatility and below the forward: there is no way back."""
    assert call_price(90.0, 100.0, 0.05, 0.0, 1.0) == 0.0
    assert put_price(110.0, 100.0, 0.0, 0.0, 1.0) == 0.0


def test_zero_volatility_greeks_are_the_forward_greeks():
    S, K, r, T = 110.0, 100.0, 0.05, 1.0
    g = greeks(S, K, r, 0.0, T, CALL)
    discounted_k = K * math.exp(-r * T)
    assert g["delta"] == 1.0                                    # one share, for certain
    assert g["gamma"] == 0.0 and g["vega"] == 0.0               # nothing left to bend
    assert g["theta"] == pytest.approx(-r * discounted_k)       # interest on the strike
    assert g["rho"] == pytest.approx(T * discounted_k)


def test_knife_edge_at_the_forward_with_no_volatility():
    """``S == K*exp(-r*T)`` exactly: the documented limits, not NaN."""
    K, r, T = 100.0, 0.05, 1.0
    S = K * math.exp(-r * T)
    g = greeks(S, K, r, 0.0, T, CALL)
    assert g["delta"] == 0.5                       # midpoint of the jump
    assert math.isinf(g["gamma"])                  # a kink has unbounded curvature
    assert g["vega"] == pytest.approx(S * math.sqrt(T) / math.sqrt(2 * math.pi))
    assert greeks(S, K, r, 0.0, T, PUT)["delta"] == -0.5


# ---------------------------------------------------------------------------
# Very high volatility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [5.0, 20.0])
def test_high_volatility_call_approaches_the_stock_price(sigma):
    """As vol explodes the call converges to the stock and the put to the cash.

    With enormous uncertainty, exercising a call is near-certain to be worth
    *something* and the strike becomes irrelevant, so the call is worth about
    the share. The put converges to the discounted strike, the most it can
    ever pay.

    The convergence is slow, because what matters is ``N(sigma*sqrt(T)/2)``:
    measured relative error for the call is 0.60 at sigma=1, 0.31 at 2,
    0.012 at 5, and below 1e-4 from 10 up. Hence testing it at 5 and 20.
    """
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    discounted_k = K * math.exp(-r * T)
    assert call_price(S, K, r, sigma, T) == pytest.approx(S, rel=0.02)
    assert put_price(S, K, r, sigma, T) == pytest.approx(discounted_k, rel=0.02)


def test_call_rises_towards_the_stock_price_as_volatility_grows():
    """The approach to the bound is monotone and never crosses it."""
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    prices = [call_price(S, K, r, sigma, T) for sigma in (1.0, 2.0, 5.0, 10.0, 20.0)]
    assert all(b >= a for a, b in zip(prices, prices[1:]))
    assert all(p <= S for p in prices)
    assert prices[-1] == pytest.approx(S, rel=1e-6)


def test_high_volatility_prices_never_break_the_bounds():
    """The bound ``call <= S`` is the one a sloppy implementation violates."""
    for sigma in (1.0, 5.0, 10.0, 50.0, 100.0):
        call = call_price(100.0, 100.0, 0.05, sigma, 5.0)
        assert 0.0 <= call <= 100.0


def test_high_volatility_delta_stays_a_probability():
    for sigma in (1.0, 5.0, 10.0, 50.0):
        assert 0.0 <= greeks(100.0, 100.0, 0.05, sigma, 1.0, CALL)["delta"] <= 1.0


# ---------------------------------------------------------------------------
# Deep in and out of the money
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S", [500.0, 1_000.0, 100_000.0])
def test_deep_in_the_money_call_is_the_discounted_forward(S):
    """Exercise is a near-certainty, so the call is the stock minus a loan."""
    K, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    assert call_price(S, K, r, sigma, T) == pytest.approx(
        S - K * math.exp(-r * T), rel=1e-9
    )
    assert greeks(S, K, r, sigma, T, CALL)["delta"] == pytest.approx(1.0, abs=1e-9)
    assert greeks(S, K, r, sigma, T, CALL)["gamma"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("S", [20.0, 1.0, 0.01])
def test_deep_out_of_the_money_call_is_worthless_but_not_negative(S):
    """The failure mode here is a tiny negative price from cancellation."""
    K, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    call = call_price(S, K, r, sigma, T)
    assert 0.0 <= call < 1e-3
    g = greeks(S, K, r, sigma, T, CALL)
    assert g["delta"] == pytest.approx(0.0, abs=1e-6)
    assert g["vega"] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("S", [20.0, 1.0, 0.01])
def test_deep_in_the_money_put_approaches_the_discounted_strike(S):
    K, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    assert put_price(S, K, r, sigma, T) == pytest.approx(
        K * math.exp(-r * T) - S, rel=1e-6
    )
    assert greeks(S, K, r, sigma, T, PUT)["delta"] == pytest.approx(-1.0, abs=1e-6)


def test_extreme_moneyness_does_not_produce_nan():
    """``d1`` reaches +/-1e4 here; scipy saturates rather than returning NaN."""
    for S in (1e-6, 1e6):
        for kind in (CALL, PUT):
            assert math.isfinite(price(S, 100.0, 0.05, 0.2, 1.0, kind))
            assert all(math.isfinite(v) or math.isinf(v)
                       for v in greeks(S, 100.0, 0.05, 0.2, 1.0, kind).values())


# ---------------------------------------------------------------------------
# Long expiry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [10.0, 50.0, 100.0])
def test_very_long_expiry_stays_sane(T):
    """A 100-year call is worth nearly the whole share (the strike discounts away)."""
    S, K, r, sigma = 100.0, 100.0, 0.05, 0.2
    call = call_price(S, K, r, sigma, T)
    assert max(S - K * math.exp(-r * T), 0.0) <= call <= S
    assert call_price(S, K, r, sigma, 100.0) == pytest.approx(S, rel=0.02)


# ---------------------------------------------------------------------------
# Invalid inputs must raise, not return nonsense
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_args,message", [
    ((0.0, 100.0, 0.05, 0.2, 1.0), "S"),
    ((-10.0, 100.0, 0.05, 0.2, 1.0), "S"),
    ((100.0, 0.0, 0.05, 0.2, 1.0), "K"),
    ((100.0, -10.0, 0.05, 0.2, 1.0), "K"),
    ((100.0, 100.0, 0.05, -0.2, 1.0), "sigma"),
    ((100.0, 100.0, 0.05, 0.2, -1.0), "T"),
    ((math.nan, 100.0, 0.05, 0.2, 1.0), "finite"),
    ((math.inf, 100.0, 0.05, 0.2, 1.0), "finite"),
    ((100.0, 100.0, math.nan, 0.2, 1.0), "finite"),
    ((100.0, 100.0, 0.05, math.inf, 1.0), "finite"),
])
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_invalid_inputs_raise_value_error(bad_args, message, kind):
    with pytest.raises(ValueError, match=message):
        price(*bad_args, kind)
    with pytest.raises(ValueError, match=message):
        greeks(*bad_args, kind)


def test_zero_rate_and_negative_rate_are_allowed():
    """Not an error: both have been the state of the world recently."""
    assert call_price(100.0, 100.0, 0.0, 0.2, 1.0) > 0.0
    assert call_price(100.0, 100.0, -0.02, 0.2, 1.0) > 0.0
