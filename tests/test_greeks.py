"""Tests for the five Greeks.

The important test here is the finite-difference one. A Greek is a derivative,
so instead of trusting the algebra we bump an input by a small amount ``h``,
re-price, and check the slope the code claims matches the slope the price
function actually has. If a formula is wrong -- a missing discount factor, a
sign error, the wrong ``d`` in a CDF -- this catches it, because the price
function and the Greek formula are derived independently.

Central differences are used because their error shrinks like ``h**2`` rather
than ``h``::

    f'(x)  ~ (f(x+h) - f(x-h)) / (2h)
    f''(x) ~ (f(x+h) - 2*f(x) + f(x-h)) / h**2

``h`` is a compromise: too big and the curvature of the price spoils the
estimate, too small and subtracting two nearly equal prices loses precision.
"""

import math

import pytest

from pricing.black_scholes import (
    CALL,
    GREEK_NAMES,
    PUT,
    delta,
    gamma,
    greeks,
    price,
    rho,
    theta,
    vega,
)

# Well-behaved inputs: the finite-difference comparison is only meaningful
# where the price function is smooth. The extremes (T -> 0, sigma -> 0, deep
# out of the money) live in test_edge_cases.py.
FD_GRID = [
    (100.0, 100.0, 0.05, 0.20, 1.00),      # at the money, one year
    (100.0, 90.0, 0.05, 0.20, 0.50),       # call in the money
    (100.0, 110.0, 0.03, 0.35, 2.00),      # call out of the money, long dated
    (42.0, 40.0, 0.10, 0.20, 0.50),        # Hull's pricing example
    (49.0, 50.0, 0.05, 0.20, 20 / 52),     # Hull's Greeks example
    (100.0, 100.0, 0.00, 0.10, 0.25),      # zero rate, calm, three months
    (100.0, 100.0, -0.01, 0.25, 1.50),     # negative rate
    (80.0, 120.0, 0.05, 0.45, 0.75),       # well out of the money, wild
    (120.0, 80.0, 0.05, 0.15, 0.30),       # well in the money, calm
]
KINDS = [CALL, PUT]
CASES = [(args, kind) for args in FD_GRID for kind in KINDS]


def central_diff(f, x, h):
    """First derivative of ``f`` at ``x`` by central difference."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def second_diff(f, x, h):
    """Second derivative of ``f`` at ``x`` by central difference."""
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


# ---------------------------------------------------------------------------
# Known textbook values: Hull's Greeks example
# ---------------------------------------------------------------------------

HULL = (49.0, 50.0, 0.05, 0.20, 20 / 52)


def test_hull_call_greeks():
    """Hull: delta 0.522, gamma 0.066, vega 12.1, theta -4.31/year, rho 8.91.

    Hull quotes vega and rho per 1% move (0.121 and 0.0891) and theta per
    calendar day (-0.0118); this module works per unit, hence the x100 / x365.
    """
    assert delta(*HULL, CALL) == pytest.approx(0.522, abs=0.001)
    assert gamma(*HULL, CALL) == pytest.approx(0.066, abs=0.001)
    assert vega(*HULL, CALL) == pytest.approx(12.1, abs=0.05)
    assert theta(*HULL, CALL) == pytest.approx(-4.31, abs=0.01)
    assert rho(*HULL, CALL) == pytest.approx(8.91, abs=0.01)
    # The same numbers in the units a desk would quote.
    assert theta(*HULL, CALL) / 365 == pytest.approx(-0.0118, abs=0.0001)
    assert vega(*HULL, CALL) / 100 == pytest.approx(0.121, abs=0.0005)
    assert rho(*HULL, CALL) / 100 == pytest.approx(0.0891, abs=0.0001)


def test_hull_put_greeks_follow_from_parity():
    """The put Greeks for the same option, derived by hand from parity.

    ``P = C - S + K*exp(-r*T)``, so differentiating: put delta = call delta - 1,
    put rho = call rho - K*T*exp(-r*T), put theta = call theta + r*K*exp(-r*T).
    """
    S, K, r, sigma, T = HULL
    discounted_k = K * math.exp(-r * T)
    assert delta(*HULL, PUT) == pytest.approx(delta(*HULL, CALL) - 1.0, rel=1e-12)
    assert rho(*HULL, PUT) == pytest.approx(rho(*HULL, CALL) - T * discounted_k, rel=1e-12)
    assert theta(*HULL, PUT) == pytest.approx(theta(*HULL, CALL) + r * discounted_k, rel=1e-12)


# ---------------------------------------------------------------------------
# Finite differences: every Greek, both option types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args,kind", CASES)
def test_delta_matches_finite_difference(args, kind):
    """delta = dP/dS."""
    S, K, r, sigma, T = args
    h = S * 1e-4
    numeric = central_diff(lambda s: price(s, K, r, sigma, T, kind), S, h)
    assert delta(*args, kind) == pytest.approx(numeric, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("args,kind", CASES)
def test_gamma_matches_finite_difference(args, kind):
    """gamma = d2P/dS2, so it needs the second difference.

    Step size matters more here than for the first derivatives, because the
    ``h**2`` in the denominator amplifies both error sources. Measured on the
    sharpest case in the grid (S=K=100, sigma=10%, T=0.25, where one standard
    deviation of the terminal price is only 5 points), the relative error
    behaves exactly as the theory says::

        h = S*1e-2   1.3e-4       truncation error, falling like h**2
        h = S*1e-3   3.3e-5
        h = S*1e-4   3.3e-7
        h = S*1e-5   6.0e-8       the floor
        h = S*1e-6   7.3e-6       round-off taking over, rising like 1/h**2

    ``S*1e-4`` sits comfortably inside that valley, with two orders of
    magnitude of margin before either error source matters.
    """
    S, K, r, sigma, T = args
    h = S * 1e-4
    numeric = second_diff(lambda s: price(s, K, r, sigma, T, kind), S, h)
    assert gamma(*args, kind) == pytest.approx(numeric, rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("args,kind", CASES)
def test_vega_matches_finite_difference(args, kind):
    """vega = dP/dsigma, per 1.0 of volatility."""
    S, K, r, sigma, T = args
    h = 1e-5
    numeric = central_diff(lambda sg: price(S, K, r, sg, T, kind), sigma, h)
    assert vega(*args, kind) == pytest.approx(numeric, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("args,kind", CASES)
def test_theta_matches_finite_difference(args, kind):
    """theta = dP/dt = -dP/dT: time passing is expiry approaching.

    The minus sign is the whole subtlety of theta, and this test is what pins
    it down.
    """
    S, K, r, sigma, T = args
    h = min(1e-5, T / 100.0)
    numeric = -central_diff(lambda t: price(S, K, r, sigma, t, kind), T, h)
    assert theta(*args, kind) == pytest.approx(numeric, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("args,kind", CASES)
def test_rho_matches_finite_difference(args, kind):
    """rho = dP/dr, per 1.0 of rate."""
    S, K, r, sigma, T = args
    h = 1e-6
    numeric = central_diff(lambda rate: price(S, K, r=rate, sigma=sigma, T=T, kind=kind), r, h)
    assert rho(*args, kind) == pytest.approx(numeric, rel=1e-6, abs=1e-8)


# ---------------------------------------------------------------------------
# Relations between the Greeks that must hold exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args", FD_GRID)
def test_greek_parity_relations(args):
    """Differentiate put-call parity ``C - P = S - K*exp(-r*T)`` term by term.

    * by ``S``:     call delta - put delta = 1
    * by ``S`` twice: gammas are equal (the right-hand side is linear in S)
    * by ``sigma``: vegas are equal (sigma does not appear on the right)
    * by ``r``:     call rho - put rho = K*T*exp(-r*T)
    * by ``t``:     call theta - put theta = -r*K*exp(-r*T)
    """
    S, K, r, sigma, T = args
    discounted_k = K * math.exp(-r * T)
    assert delta(*args, CALL) - delta(*args, PUT) == pytest.approx(1.0, rel=1e-12)
    assert gamma(*args, CALL) == pytest.approx(gamma(*args, PUT), rel=1e-12)
    assert vega(*args, CALL) == pytest.approx(vega(*args, PUT), rel=1e-12)
    assert rho(*args, CALL) - rho(*args, PUT) == pytest.approx(T * discounted_k, rel=1e-12)
    assert theta(*args, CALL) - theta(*args, PUT) == pytest.approx(
        -r * discounted_k, rel=1e-10, abs=1e-12
    )


@pytest.mark.parametrize("args", FD_GRID)
def test_greek_signs_and_ranges(args):
    """The qualitative facts a trader would state without calculating.

    Call delta is a probability-like number in [0, 1] and put delta in
    [-1, 0]; gamma and vega are positive for both option types, because
    curvature and the value of uncertainty do not care which way you are
    betting.
    """
    assert 0.0 <= delta(*args, CALL) <= 1.0
    assert -1.0 <= delta(*args, PUT) <= 0.0
    assert gamma(*args, CALL) > 0.0
    assert vega(*args, CALL) > 0.0
    assert rho(*args, CALL) > 0.0      # waiting to pay the strike is good
    assert rho(*args, PUT) < 0.0       # waiting to receive it is not


def test_call_theta_is_negative_at_the_money():
    """An at-the-money call with a positive rate only loses from time passing."""
    assert theta(100.0, 100.0, 0.05, 0.2, 1.0, CALL) < 0.0


def test_deep_in_the_money_put_can_have_positive_theta():
    """The one case where waiting *helps*: interest on the strike dominates.

    A deep-in-the-money put is nearly certain to be exercised, so it is almost
    a promise to receive K later. With a high rate, every day that passes
    brings that payment closer, which is worth money -- enough to outweigh the
    optionality draining away.
    """
    assert theta(40.0, 100.0, 0.15, 0.15, 1.0, PUT) > 0.0


# ---------------------------------------------------------------------------
# Gamma and vega peak where the textbooks say they do
# ---------------------------------------------------------------------------

def test_gamma_peaks_near_the_money():
    """Gamma is largest where the option is most undecided."""
    strikes = [60.0, 80.0, 100.0, 120.0, 140.0]
    gammas = [gamma(100.0, K, 0.05, 0.2, 0.5, CALL) for K in strikes]
    assert gammas.index(max(gammas)) == 2       # K = 100, at the money


def test_gamma_grows_and_vega_shrinks_as_expiry_approaches():
    """Gamma is a short-dated phenomenon, vega a long-dated one.

    Same at-the-money option, shrinking time to expiry: gamma goes up (the
    ``1/sqrt(T)`` in its denominator) while vega goes down (the ``sqrt(T)``
    factor). This is why gamma scalpers trade the front month and vol traders
    trade the back month.
    """
    expiries = [2.0, 1.0, 0.5, 0.25, 0.05]
    gammas = [gamma(100.0, 100.0, 0.05, 0.2, T, CALL) for T in expiries]
    vegas = [vega(100.0, 100.0, 0.05, 0.2, T, CALL) for T in expiries]
    assert all(b > a for a, b in zip(gammas, gammas[1:]))
    assert all(b < a for a, b in zip(vegas, vegas[1:]))


# ---------------------------------------------------------------------------
# The greeks() convenience wrapper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args,kind", CASES)
def test_greeks_dict_matches_individual_functions(args, kind):
    bundle = greeks(*args, kind)
    assert tuple(bundle) == GREEK_NAMES
    assert bundle["delta"] == delta(*args, kind)
    assert bundle["gamma"] == gamma(*args, kind)
    assert bundle["vega"] == vega(*args, kind)
    assert bundle["theta"] == theta(*args, kind)
    assert bundle["rho"] == rho(*args, kind)


@pytest.mark.parametrize("args,kind", CASES)
def test_greeks_are_plain_floats(args, kind):
    """Not numpy scalars: the scalar API should return plain Python floats."""
    for value in greeks(*args, kind).values():
        assert type(value) is float


def test_greeks_rejects_bad_kind():
    with pytest.raises(ValueError, match="call.*put"):
        greeks(100.0, 100.0, 0.05, 0.2, 1.0, "butterfly")
