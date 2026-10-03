"""Tests for the Cox-Ross-Rubinstein binomial tree.

The tree has one big advantage as a thing to test: it is a completely
different algorithm from Black-Scholes -- a discrete lattice and a backward
recursion, with no normal CDF anywhere -- yet it must converge to the same
number. Agreement between two unrelated methods is strong evidence that both
are right.

What is checked here:

* a two-step tree whose arithmetic can be done by hand
* Hull's five-step American put example
* European prices converging to Black-Scholes, at the expected 1/N rate
* put-call parity holding *exactly* on the tree
* American call = European call with no dividends (early exercise never pays)
* American put >= European put, strictly greater when early exercise is
  optimal, and equal again when the rate is zero
"""

import math

import pytest

from pricing.binomial import (
    american_binomial,
    binomial_price,
    crr_parameters,
    early_exercise_premium,
    european_binomial,
)
from pricing.black_scholes import CALL, PUT, call_price, price, put_price

ATM = (100.0, 100.0, 0.05, 0.20, 1.0)       # S, K, r, sigma, T

CONVERGENCE_CASES = [
    (100.0, 100.0, 0.05, 0.20, 1.0),
    (42.0, 40.0, 0.10, 0.20, 0.5),           # Hull's Black-Scholes example
    (100.0, 110.0, 0.03, 0.35, 2.0),
    (120.0, 80.0, 0.05, 0.15, 0.3),
    (80.0, 120.0, 0.02, 0.50, 0.5),
    (100.0, 100.0, 0.00, 0.25, 1.5),
]


# ---------------------------------------------------------------------------
# The tree parameters, and a tree small enough to check by hand
# ---------------------------------------------------------------------------

def test_crr_parameters_two_steps():
    """S=100, K=100, r=5%, sigma=20%, T=1, N=2, so dt = 0.5.

    By hand::

        u = exp(0.20 * sqrt(0.5)) = exp(0.141421) = 1.151910
        d = 1 / u                               = 0.868123
        p = (exp(0.05*0.5) - d) / (u - d)
          = (1.025315 - 0.868123) / 0.283787    = 0.553908
    """
    params = crr_parameters(0.05, 0.20, 1.0, 2)
    assert params["dt"] == 0.5
    assert params["u"] == pytest.approx(1.151910, abs=1e-6)
    assert params["d"] == pytest.approx(0.868123, abs=1e-6)
    assert params["p"] == pytest.approx(0.553908, abs=1e-6)
    assert params["discount"] == pytest.approx(math.exp(-0.025), rel=1e-12)
    assert params["u"] * params["d"] == pytest.approx(1.0, rel=1e-15)   # recombining


def test_two_step_european_call_by_hand():
    """Only one of the three terminal nodes pays off, which makes this easy.

    Terminal prices: ``100*u*u = 132.69``, ``100*u*d = 100``, ``100*d*d = 75.36``.
    Against a strike of 100 the payoffs are 32.69, 0 and 0, so only the
    up-up path matters and the price is::

        exp(-0.05) * p**2 * 32.6900
          = 0.951229 * 0.306814 * 32.6900 = 9.5405
    """
    expected = math.exp(-0.05) * 0.553908**2 * (100.0 * 1.151910**2 - 100.0)
    assert european_binomial(*ATM, 2, CALL) == pytest.approx(expected, rel=1e-5)
    assert european_binomial(*ATM, 2, CALL) == pytest.approx(9.5405, abs=1e-4)


def test_two_step_european_put_by_hand():
    """Mirror image: only the down-down node pays.

        exp(-0.05) * (1-p)**2 * (100 - 75.3639) = 4.6634
    """
    down_down = 100.0 * 0.868123**2
    expected = math.exp(-0.05) * (1 - 0.553908) ** 2 * (100.0 - down_down)
    assert european_binomial(*ATM, 2, PUT) == pytest.approx(expected, rel=1e-5)


def test_single_step_tree_is_the_one_period_model():
    """N=1: two outcomes, one discounted average. The simplest case that works."""
    S, K, r, sigma, T = ATM
    p = crr_parameters(r, sigma, T, 1)
    up_payoff = max(S * p["u"] - K, 0.0)
    down_payoff = max(S * p["d"] - K, 0.0)
    expected = p["discount"] * (p["p"] * up_payoff + (1 - p["p"]) * down_payoff)
    assert european_binomial(*ATM, 1, CALL) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Known textbook value: Hull's five-step American put
# ---------------------------------------------------------------------------

def test_hull_five_step_american_put():
    """Hull's worked example: S=50, K=50, r=10%, sigma=40%, T=5 months, N=5.

    Hull reports 4.49 for the American put. This is the example that shows
    early exercise mattering: the European price on the same tree is 4.32.
    """
    args = (50.0, 50.0, 0.10, 0.40, 5 / 12)
    assert american_binomial(*args, 5, PUT) == pytest.approx(4.49, abs=0.005)
    assert european_binomial(*args, 5, PUT) == pytest.approx(4.32, abs=0.005)
    assert early_exercise_premium(*args, 5, PUT) > 0.15


# ---------------------------------------------------------------------------
# Convergence to Black-Scholes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args", CONVERGENCE_CASES)
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_european_binomial_converges_to_black_scholes(args, kind):
    """With 2,000 steps the tree should agree with the closed form to a cent."""
    tree = european_binomial(*args, 2000, kind)
    closed_form = price(*args, kind)
    assert tree == pytest.approx(closed_form, abs=0.01)


AT_THE_MONEY_CASES = [
    (100.0, 100.0, 0.05, 0.20, 1.0),
    (100.0, 100.0, 0.00, 0.25, 1.5),
]


@pytest.mark.parametrize("args", AT_THE_MONEY_CASES)
def test_at_the_money_error_shrinks_like_one_over_steps(args):
    """The CRR error is O(1/N), measured rather than assumed.

    Steps are quadrupled each time and all of them are even, so the odd/even
    oscillation is held constant and only the envelope is left. At the money
    this is textbook clean, because the strike sits exactly on a lattice node
    for every even N -- the geometry does not change as N grows, so
    ``error * N`` is constant::

        N = 50    200    800    3200
        err*N   1.995  1.998  1.999  2.000      (S=K=100, r=5%, sigma=20%, T=1)
        err*N   3.034  3.040  3.041  3.042      (S=K=100, r=0,  sigma=25%, T=1.5)
    """
    closed_form = call_price(*args)
    steps_list = (50, 200, 800, 3200)
    errors = [abs(european_binomial(*args, steps, CALL) - closed_form)
              for steps in steps_list]

    for coarse, fine in zip(errors, errors[1:]):
        assert 3.8 < coarse / fine < 4.2          # quadruple the steps, quarter the error

    scaled = [err * steps for err, steps in zip(errors, steps_list)]
    assert max(scaled) / min(scaled) < 1.01       # error * N is constant to 1%


@pytest.mark.parametrize("args", CONVERGENCE_CASES)
def test_convergence_error_is_bounded_by_order_one_over_steps(args):
    """Away from the money the *envelope* is still 1/N, but it is not smooth.

    The reason is geometric: as N changes, the strike sits in a different
    position between two lattice nodes, and how well the lattice happens to
    straddle the kink in the payoff changes with it. So the error still falls
    like 1/N overall, but individual ratios jump around -- measured ratios for
    Hull's 42/40 call are 1.08, 15.7, 2.3, and for the 80/120 case 6.07, 23.2,
    1.01. This is the jitter the Phase 2 convergence plot shows.

    What can be asserted is the envelope: ``error * N`` stays bounded.
    """
    closed_form = call_price(*args)
    steps_list = (50, 200, 800, 3200)
    errors = [abs(european_binomial(*args, steps, CALL) - closed_form)
              for steps in steps_list]

    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse                      # still improving at every stage
    for err, steps in zip(errors, steps_list):
        assert err * steps < 10.0                 # i.e. error < 10/N


def test_error_at_many_steps_is_small():
    """A concrete size for the error, so a regression would be obvious."""
    closed_form = call_price(*ATM)
    assert abs(european_binomial(*ATM, 3200, CALL) - closed_form) < 1e-3


def test_odd_and_even_step_counts_straddle_the_true_price():
    """The oscillation that the Phase 2 experiment plots.

    At the money, an even number of steps puts a terminal node *exactly* at
    the strike, while an odd number leaves the strike between two nodes. The
    two cases approach Black-Scholes from opposite sides: even step counts
    undershoot, odd ones overshoot. Measured errors here are -0.0399 (N=50)
    against +0.0344 (N=51), and -0.00062 (N=3200) against +0.00055 (N=3201).
    """
    closed_form = call_price(*ATM)
    for even in (10, 50, 200, 800, 3200):
        assert european_binomial(*ATM, even, CALL) < closed_form
        assert european_binomial(*ATM, even + 1, CALL) > closed_form


def test_averaging_an_odd_and_even_pair_beats_both():
    """Why the oscillation is worth knowing about: averaging cancels it.

    The two errors have opposite signs and similar size, so the mean of the
    N and N+1 prices is far more accurate than either -- a free accuracy gain
    for one extra tree.
    """
    closed_form = call_price(*ATM)
    low = european_binomial(*ATM, 200, CALL)
    high = european_binomial(*ATM, 201, CALL)
    averaged = 0.5 * (low + high)
    assert abs(averaged - closed_form) < 0.1 * min(
        abs(low - closed_form), abs(high - closed_form)
    )


# ---------------------------------------------------------------------------
# Put-call parity holds exactly on the tree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args", CONVERGENCE_CASES)
@pytest.mark.parametrize("steps", [1, 2, 7, 50, 201])
def test_european_tree_satisfies_put_call_parity_exactly(args, steps):
    """Not approximately -- exactly, at every step count.

    The risk-neutral probability is chosen so that each step's expected growth
    is exactly ``exp(r*dt)``, which makes the tree price the forward exactly.
    Parity follows for any ``N``, even ``N = 1`` where the price itself is
    nowhere near Black-Scholes. So this test checks the tree is internally
    arbitrage-free, separately from whether it has converged.
    """
    S, K, r, sigma, T = args
    call = european_binomial(*args, steps, CALL)
    put = european_binomial(*args, steps, PUT)
    assert call - put == pytest.approx(S - K * math.exp(-r * T), rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# American options: when early exercise is worth something
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args", CONVERGENCE_CASES)
@pytest.mark.parametrize("steps", [25, 200])
def test_american_call_equals_european_call_without_dividends(args, steps):
    """The classic result: never exercise an American call early (no dividends).

    Exercising early pays the strike sooner and throws away the protection of
    not being committed, so continuation always wins and the ``max`` in the
    backward induction never bites. The two prices come out equal to within
    floating-point noise (exactly equal in most cases; the ties happen at
    ``r = 0``, where deferring has no interest benefit to offer).
    """
    american = binomial_price(*args, steps, CALL, american=True)
    european = binomial_price(*args, steps, CALL, american=False)
    assert american == pytest.approx(european, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("args", CONVERGENCE_CASES)
@pytest.mark.parametrize("steps", [25, 200])
def test_american_put_is_never_worth_less_than_european(args, steps):
    """A strictly larger set of rights cannot be worth less."""
    american = binomial_price(*args, steps, PUT, american=True)
    european = binomial_price(*args, steps, PUT, american=False)
    assert american >= european - 1e-12


@pytest.mark.parametrize("S,expected_premium", [
    (60.0, 9.4),        # deep in the money: exercise immediately
    (70.0, 8.8),
    (80.0, 6.7),
    (90.0, 3.0),
    (100.0, 1.07),      # at the money
    (110.0, 0.37),
    (130.0, 0.04),      # out of the money: the right is nearly worthless
])
def test_american_put_premium_is_largest_deep_in_the_money(S, expected_premium):
    """Measured early-exercise premiums, K=100, r=10%, sigma=20%, T=1, N=500.

    The pattern is the point: the deeper in the money, the more the right to
    exercise early is worth, because there is more strike to collect interest
    on and less insurance value being given up.
    """
    premium = early_exercise_premium(S, 100.0, 0.10, 0.20, 1.0, 500, PUT)
    assert premium == pytest.approx(expected_premium, rel=0.02)
    assert premium > 0.0


def test_deep_in_the_money_american_put_equals_its_intrinsic_value():
    """Far enough in the money, the right answer is "exercise now".

    At S=60 with K=100 the American put is worth exactly 40 -- the tree has
    found that waiting is never better than taking the cash today.
    """
    assert american_binomial(60.0, 100.0, 0.10, 0.20, 1.0, 500, PUT) == pytest.approx(
        40.0, abs=1e-9
    )
    # The European put cannot do that: it must wait, and 40 discounted is less.
    assert european_binomial(60.0, 100.0, 0.10, 0.20, 1.0, 500, PUT) < 40.0


@pytest.mark.parametrize("S", [60.0, 80.0, 100.0, 120.0])
def test_zero_rate_removes_the_early_exercise_premium(S):
    """With r = 0 there is no interest to gain, so early exercise never pays.

    This is the cleanest demonstration that the premium really is about the
    time value of money: set the rate to zero and the American put collapses
    onto the European one (agreement here is to 3e-14).
    """
    american = american_binomial(S, 100.0, 0.0, 0.20, 1.0, 500, PUT)
    european = european_binomial(S, 100.0, 0.0, 0.20, 1.0, 500, PUT)
    assert american == pytest.approx(european, abs=1e-9)


@pytest.mark.parametrize("args", CONVERGENCE_CASES)
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_american_price_is_never_below_intrinsic_value(args, kind):
    """An American option can always be exercised, so it is worth at least that."""
    S, K, _, _, _ = args
    intrinsic = max(S - K, 0.0) if kind == CALL else max(K - S, 0.0)
    assert binomial_price(*args, 200, kind, american=True) >= intrinsic - 1e-12


def test_american_put_converges_too():
    """Different step counts agree, so the American price is itself converging."""
    args = (90.0, 100.0, 0.08, 0.30, 1.0)
    prices = [american_binomial(*args, steps, PUT) for steps in (500, 1000, 2000, 4000)]
    assert max(prices) - min(prices) < 0.01


# ---------------------------------------------------------------------------
# Inputs the tree cannot accept
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("steps", [0, -1, -100, 2.5])
def test_bad_step_count_raises(steps):
    with pytest.raises(ValueError, match="steps"):
        european_binomial(*ATM, steps, CALL)


def test_float_step_count_that_is_whole_is_accepted():
    """``2.0`` is a whole number, so it is allowed and equals ``2``."""
    assert european_binomial(*ATM, 2.0, CALL) == european_binomial(*ATM, 2, CALL)


def test_zero_volatility_raises_with_a_useful_message():
    """u and d would both be 1: there is no tree. Black-Scholes handles it."""
    with pytest.raises(ValueError, match="sigma must be > 0"):
        european_binomial(100.0, 100.0, 0.05, 0.0, 1.0, 100, CALL)


def test_step_too_long_for_the_volatility_raises():
    """``p`` outside (0, 1) means the tree implies an arbitrage, so refuse.

    Here one five-year step with 10% vol cannot reach the risk-free growth of
    ``exp(0.5)``, so no probability can make the tree fair.
    """
    with pytest.raises(ValueError, match="not in"):
        european_binomial(100.0, 100.0, 0.10, 0.10, 5.0, 1, CALL)


def test_expiry_now_returns_the_payoff():
    assert european_binomial(110.0, 100.0, 0.05, 0.2, 0.0, 50, CALL) == 10.0
    assert american_binomial(90.0, 100.0, 0.05, 0.2, 0.0, 50, PUT) == 10.0
    assert european_binomial(90.0, 100.0, 0.05, 0.2, 0.0, 50, CALL) == 0.0


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="call.*put"):
        european_binomial(*ATM, 10, "condor")


# ---------------------------------------------------------------------------
# Agreement with Phase 1 on the Greeks' worth of inputs
# ---------------------------------------------------------------------------

def test_tree_reproduces_hull_black_scholes_example():
    """Hull's 42/40 call: closed form 4.76, tree with 5,000 steps must agree."""
    args = (42.0, 40.0, 0.10, 0.20, 0.5)
    assert european_binomial(*args, 5000, CALL) == pytest.approx(
        call_price(*args), abs=0.002
    )
    assert european_binomial(*args, 5000, PUT) == pytest.approx(
        put_price(*args), abs=0.002
    )
