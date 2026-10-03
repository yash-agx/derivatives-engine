"""Tests for the variance-reduced estimators.

Two things have to be true of every method here, and they pull in opposite
directions:

1. **It must still be right.** A variance reduction that quietly biases the
   price is worse than useless, because it looks *more* convincing while being
   wrong. Every estimator is checked for unbiasedness against Black-Scholes.
2. **Its error bar must still be honest.** This is where these methods are
   easiest to get wrong. Antithetic variates deliberately make two paths
   dependent, so the textbook ``s/sqrt(N)`` over individual paths no longer
   applies; the standard error has to be computed over *pairs*. The
   Phase 3 harness checks this directly by comparing the reported error with
   the actual spread over hundreds of seeds.

The tests also pin down where each method **fails**, which matters as much as
where it works: antithetic on a symmetric payoff, and importance sampling on an
option that did not need it.
"""

import math

import numpy as np
import pytest

from experiments.harness import Option, plain_monte_carlo, run_seed_study
from pricing.black_scholes import call_price, price
from pricing.monte_carlo import monte_carlo_price
from pricing.variance_reduction import (
    antithetic_control_monte_carlo,
    antithetic_monte_carlo,
    control_variate_monte_carlo,
    importance_sampling_monte_carlo,
    importance_sampling_shift,
    optimal_control_coefficient,
)

ATM = (100.0, 100.0, 0.05, 0.20, 1.0)
DEEP_OTM = (100.0, 160.0, 0.05, 0.20, 0.5)      # the Phase 3 case that breaks
SEEDS = range(300)

ATM_CALL = Option("ATM call", S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, kind="call")
STRADDLE = Option("straddle", S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
                  kind="straddle")
BROKEN = Option("K=160 6m", S=100.0, K=160.0, r=0.05, sigma=0.20, T=0.5, kind="call")


def estimator(function, **kwargs):
    """Wrap a pricer into the harness's (option, n_paths, seed) signature."""
    def wrapped(option, n_paths, seed):
        return function(*option.args, option.kind, n_paths=n_paths, seed=seed, **kwargs)
    return wrapped


ANTITHETIC = estimator(antithetic_monte_carlo)
CONTROL = estimator(control_variate_monte_carlo)
BOTH = estimator(antithetic_control_monte_carlo)
IMPORTANCE = estimator(importance_sampling_monte_carlo)


# ---------------------------------------------------------------------------
# 1. Every method still prices correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,method", [
    ("antithetic", ANTITHETIC),
    ("control", CONTROL),
    ("anti+control", BOTH),
    ("importance", IMPORTANCE),
])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_estimator_is_unbiased(name, method, kind):
    """Averaged over 300 seeds, each method lands on Black-Scholes."""
    option = Option(f"ATM {kind}", S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
                    kind=kind)
    study = run_seed_study(method, option, 20_000, SEEDS, name)
    standard_error_of_mean = study.spread / math.sqrt(study.n_seeds)
    assert abs(study.bias) < 4 * standard_error_of_mean


def test_methods_agree_with_each_other_at_many_paths():
    """Five estimators of the same number, each inside its own error bar.

    Judged against each method's own standard error rather than one shared
    dollar tolerance, because the point of these methods is that their spreads
    differ enormously: at the money the combined estimator's error bar is about
    a twentieth of importance sampling's on the same option.
    """
    truth = call_price(*ATM)
    results = [
        monte_carlo_price(*ATM, "call", n_paths=500_000, seed=11),
        antithetic_monte_carlo(*ATM, "call", n_paths=500_000, seed=11),
        control_variate_monte_carlo(*ATM, "call", n_paths=500_000, seed=11),
        antithetic_control_monte_carlo(*ATM, "call", n_paths=500_000, seed=11),
        importance_sampling_monte_carlo(*ATM, "call", n_paths=500_000, seed=11),
    ]
    for result in results:
        assert abs(result.price - truth) < 4 * result.standard_error
        assert result.price == pytest.approx(truth, abs=0.1)


# ---------------------------------------------------------------------------
# 2. Every method's error bar is still honest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,method", [
    ("antithetic", ANTITHETIC),
    ("control", CONTROL),
    ("anti+control", BOTH),
    ("importance", IMPORTANCE),
])
def test_reported_error_matches_the_actual_spread(name, method):
    """The harness's honesty check, applied to each variance-reduced method.

    Measured ratios on the at-the-money call are 1.03 (antithetic), 1.00
    (control), 0.99 (both) and 0.99 (importance).
    """
    study = run_seed_study(method, ATM_CALL, 20_000, SEEDS, name)
    assert study.se_ratio == pytest.approx(1.0, abs=0.15)
    assert 0.90 <= study.coverage <= 0.99


def test_antithetic_standard_error_uses_pairs_not_paths():
    """The specific mistake this method invites, pinned down two ways.

    First directly: rebuild the pair averages by hand and check the reported
    standard error is their spread over sqrt(number of pairs).

    Then by consequence: the same quantity computed over individual paths is a
    different number, and using it would misreport the error.
    """
    S, K, r, sigma, T = ATM
    n_paths, seed = 20_000, 4
    result = antithetic_monte_carlo(S, K, r, sigma, T, "call",
                                    n_paths=n_paths, seed=seed)

    n_pairs = n_paths // 2
    normals = np.random.default_rng(seed).standard_normal(n_pairs)
    drift = (r - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)
    discount = math.exp(-r * T)
    up = np.maximum(S * np.exp(drift + vol * normals) - K, 0.0)
    down = np.maximum(S * np.exp(drift - vol * normals) - K, 0.0)

    pair_means = 0.5 * discount * (up + down)
    by_pairs = pair_means.std(ddof=1) / math.sqrt(n_pairs)
    assert result.standard_error == pytest.approx(by_pairs, rel=1e-12)
    assert result.price == pytest.approx(pair_means.mean(), rel=1e-12)

    all_paths = np.concatenate([discount * up, discount * down])
    by_paths = all_paths.std(ddof=1) / math.sqrt(n_paths)
    assert by_paths != pytest.approx(by_pairs, rel=0.02)


def test_antithetic_path_based_error_would_be_wrong_in_both_directions():
    """Why the pair rule is not merely pedantic.

    For a call the two legs are negatively correlated, so a path-based standard
    error *overstates* the true error; for a straddle they are positively
    correlated, so it *understates* it. One wrong formula, two opposite
    failures -- which is why it has to be the pair.
    """
    S, K, r, sigma, T = ATM
    n_paths, seed = 20_000, 9
    drift = (r - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)
    discount = math.exp(-r * T)
    normals = np.random.default_rng(seed).standard_normal(n_paths // 2)
    up_spot = S * np.exp(drift + vol * normals)
    down_spot = S * np.exp(drift - vol * normals)

    for kind, expect_pairs_smaller in (("call", True), ("straddle", False)):
        if kind == "call":
            up = np.maximum(up_spot - K, 0.0)
            down = np.maximum(down_spot - K, 0.0)
        else:
            up = np.abs(up_spot - K)
            down = np.abs(down_spot - K)
        pair_means = 0.5 * discount * (up + down)
        by_pairs = pair_means.std(ddof=1) / math.sqrt(len(pair_means))
        by_paths = (np.concatenate([discount * up, discount * down])
                    .std(ddof=1) / math.sqrt(n_paths))
        assert bool(by_pairs < by_paths) is expect_pairs_smaller


# ---------------------------------------------------------------------------
# 3. Where each method works
# ---------------------------------------------------------------------------

def variance_ratio(method, option, n_paths=20_000, seeds=SEEDS) -> float:
    """Variance of plain Monte Carlo over variance of ``method``, at equal paths.

    Above 1 means the method helped. Equal path counts, because Phase 3 showed
    runtime per path is not constant in N.
    """
    plain = run_seed_study(plain_monte_carlo, option, n_paths, seeds, "plain")
    reduced = run_seed_study(method, option, n_paths, seeds, "reduced")
    return (plain.spread ** 2) / (reduced.spread ** 2)


def test_antithetic_helps_a_call():
    """Measured 2.1x on the at-the-money call."""
    assert variance_ratio(ANTITHETIC, ATM_CALL) > 1.7


def test_control_variate_helps_more():
    """Measured 6.9x: the stock is strongly correlated with a call on it."""
    assert variance_ratio(CONTROL, ATM_CALL) > 5.0


def test_both_together_help_most():
    """Measured 28x -- the two techniques attack different things."""
    assert variance_ratio(BOTH, ATM_CALL) > 15.0


def test_combining_is_neither_additive_nor_multiplicative():
    """The combined factor is not the product of the two separate factors.

    With Y the discounted payoff of a path and X its discounted terminal stock
    price (the control)::

        antithetic = 1 / (1 + rho_A)        rho_A  = Corr(Y(Z), Y(-Z))
        control    = 1 / (1 - rho_XY^2)     rho_XY = Corr(Y, X)
        both       = 1 / [(1 + rho_A)(1 - rho_AB^2)]

    where rho_AB is the correlation of the *pair averages*. So "both" is the
    antithetic factor times a control factor recomputed on pairs, and the ratio
    to the naive product is (1 - rho_XY^2) / (1 - rho_AB^2) -- which lands on
    either side of 1 depending on whether pairing helps the control.

    Checked here against measured variances rather than against the formula
    alone, so it is the estimators that are being tested, not the algebra.
    """
    S, K, r, sigma, T = ATM
    n_pairs = 2_000_000
    normals = np.random.default_rng(4242).standard_normal(n_pairs)
    drift = (r - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    up, down = S * np.exp(drift + vol * normals), S * np.exp(drift - vol * normals)
    y_up, y_down = discount * np.maximum(up - K, 0.0), discount * np.maximum(down - K, 0.0)
    x_up, x_down = discount * up, discount * down

    rho_a = float(np.corrcoef(y_up, y_down)[0, 1])
    rho_xy = float(np.corrcoef(np.concatenate([y_up, y_down]),
                               np.concatenate([x_up, x_down]))[0, 1])
    rho_ab = float(np.corrcoef(0.5 * (y_up + y_down), 0.5 * (x_up + x_down))[0, 1])

    antithetic_factor = 1.0 / (1.0 + rho_a)
    control_factor = 1.0 / (1.0 - rho_xy ** 2)
    both_factor = antithetic_factor / (1.0 - rho_ab ** 2)

    # The identity: the ratio to the product is governed by the two control
    # correlations alone.
    assert both_factor / (antithetic_factor * control_factor) == pytest.approx(
        (1 - rho_xy ** 2) / (1 - rho_ab ** 2), rel=1e-9
    )
    # And for this option the combination beats the product, because pairing
    # lifts the control's correlation (0.924 -> 0.964).
    assert rho_ab > rho_xy
    assert both_factor > 1.5 * antithetic_factor * control_factor

    # The formulas describe the actual estimators, not just themselves.
    assert variance_ratio(ANTITHETIC, ATM_CALL) == pytest.approx(
        antithetic_factor, rel=0.25)
    assert variance_ratio(BOTH, ATM_CALL) == pytest.approx(both_factor, rel=0.25)


def test_deep_in_the_money_combination_loses_to_the_product():
    """The other side of the same identity.

    Deep in the money the payoff is already nearly an exact linear function of
    the control, so pairing discards the very component the control was
    exploiting and the combination comes in *below* the product.
    """
    S, K, r, sigma, T = 130.0, 100.0, 0.05, 0.20, 1.0
    normals = np.random.default_rng(99).standard_normal(2_000_000)
    drift = (r - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)
    discount = math.exp(-r * T)
    up, down = S * np.exp(drift + vol * normals), S * np.exp(drift - vol * normals)
    y_up, y_down = discount * np.maximum(up - K, 0.0), discount * np.maximum(down - K, 0.0)
    x_up, x_down = discount * up, discount * down

    rho_xy = float(np.corrcoef(np.concatenate([y_up, y_down]),
                               np.concatenate([x_up, x_down]))[0, 1])
    rho_ab = float(np.corrcoef(0.5 * (y_up + y_down), 0.5 * (x_up + x_down))[0, 1])
    assert rho_xy > 0.99          # already almost perfect per path
    assert rho_ab < rho_xy        # and pairing makes it worse, not better
    assert (1 - rho_xy ** 2) / (1 - rho_ab ** 2) < 1.0


def test_control_variate_estimates_satisfy_put_call_parity_exactly():
    """A free consequence worth knowing, and a strong correctness check.

    Phase 2 found that the whole of plain Monte Carlo's parity error comes from
    the sampled mean of ``S_T`` missing the forward. The control variate's one
    job is to remove exactly that error, so the adjusted call and put must
    satisfy parity *exactly* -- measured here to 5e-15, where plain Monte Carlo
    on the same seed is off by up to 0.28.
    """
    S, K, r, sigma, T = ATM
    exact = S - K * math.exp(-r * T)
    for seed in (1, 2, 3):
        call = control_variate_monte_carlo(*ATM, "call", n_paths=5_000, seed=seed).price
        put = control_variate_monte_carlo(*ATM, "put", n_paths=5_000, seed=seed).price
        assert call - put == pytest.approx(exact, abs=1e-12)


def test_optimal_coefficient_is_covariance_over_variance():
    rng = np.random.default_rng(0)
    control = rng.standard_normal(5_000)
    target = 3.0 * control + rng.standard_normal(5_000)
    assert optimal_control_coefficient(target, control) == pytest.approx(3.0, rel=0.05)
    expected = np.cov(target, control, ddof=1)[0, 1] / np.var(control, ddof=1)
    assert optimal_control_coefficient(target, control) == pytest.approx(expected,
                                                                        rel=1e-12)


def test_coefficient_of_a_constant_control_is_zero():
    """No variance to exploit, and no division by zero either."""
    assert optimal_control_coefficient(np.arange(10.0), np.ones(10)) == 0.0


# ---------------------------------------------------------------------------
# 4. Where each method fails
# ---------------------------------------------------------------------------

def test_antithetic_makes_a_straddle_worse():
    """The documented failure case, and the reason it happens.

    Antithetic variates rely on the payoff being monotone in the random draw,
    so that one leg of a pair is low when the other is high. A straddle pays
    ``|S_T - K|``: it is large when the draw is extreme in *either* direction,
    so the two legs move together instead of opposing. Pairing then concentrates
    the variance rather than cancelling it.

    Measured: a variance ratio of 0.7, i.e. about 40% *more* variance than
    plain Monte Carlo for the same number of paths.
    """
    assert variance_ratio(ANTITHETIC, STRADDLE) < 0.9


def test_antithetic_still_reports_an_honest_error_bar_when_it_hurts():
    """Failing is acceptable; lying about it is not.

    Even on the straddle, where the method increases variance, the reported
    standard error still matches the actual spread (measured ratio 1.03). The
    estimator is worse, and says so.
    """
    study = run_seed_study(ANTITHETIC, STRADDLE, 20_000, SEEDS, "antithetic")
    assert study.se_ratio == pytest.approx(1.0, abs=0.15)
    assert 0.90 <= study.coverage <= 0.99


def test_importance_sampling_hurts_an_at_the_money_option():
    """Shifting the drift is a cost when nothing needed shifting.

    At the money about half the paths already pay, so there is no rare event to
    fix -- and the likelihood-ratio weights add variance of their own. Measured
    ratio 0.7: a 30% loss for solving a problem this option does not have.
    """
    assert variance_ratio(IMPORTANCE, ATM_CALL) < 1.0


def test_control_variate_barely_helps_far_out_of_the_money():
    """The control only works through correlation, and far out of the money
    the option payoff stops tracking the stock price at all."""
    assert variance_ratio(CONTROL, BROKEN) < 1.5


DEEP_ITM = Option("deep ITM 1m", S=130.0, K=100.0, r=0.05, sigma=0.20,
                  T=1.0 / 12.0, kind="call")


def test_control_variate_breaks_its_own_error_bar_deep_in_the_money():
    """Variance reduction can *manufacture* the Phase 3 rare-event failure.

    Deep in the money a call is almost surely exercised, so its discounted
    payoff is almost surely exactly ``exp(-rT)*S_T - K*exp(-rT)`` -- an exact
    affine function of the control. The optimal coefficient is then 1, the
    residual is a constant, and the variance collapses by a factor of 14
    million.

    But the option is not *exactly* the forward: it is the forward minus a put
    worth 2.6e-6, and that put only shows up on paths finishing below the
    strike, which happen with probability 2.2e-6. At 100,000 paths, 83% of runs
    contain no such path at all. Those runs report a standard error of
    essentially zero around a price that is wrong by the put value, so the
    interval is a point that misses. Measured coverage: 0.158, and exactly the
    same 83% of runs report se < 1e-12.

    The method removed all the variance it could see and left behind a pure
    rare-event problem -- the same pathology Phase 3 documented, created here
    by the variance reduction itself.
    """
    runs = [control_variate_monte_carlo(*DEEP_ITM.args, DEEP_ITM.kind,
                                        n_paths=100_000, seed=seed)
            for seed in range(200)]
    truth = DEEP_ITM.black_scholes()

    near_zero_error_bars = sum(r.standard_error < 1e-12 for r in runs) / len(runs)
    coverage = sum(r.contains(truth) for r in runs) / len(runs)
    assert near_zero_error_bars > 0.7       # most runs claim perfect precision
    assert coverage < 0.4                   # and most of those are wrong

    # Plain Monte Carlo, with far more variance, keeps an honest interval.
    plain_coverage = sum(
        plain_monte_carlo(DEEP_ITM, 100_000, seed).contains(truth)
        for seed in range(200)
    ) / 200
    assert plain_coverage > 0.90


def test_importance_sampling_fails_badly_deep_in_the_money():
    """Shifting towards a strike the spot has already passed is actively harmful.

    The shift is negative here (the strike sits below the median terminal
    price), so paths are pushed away from where the payoff actually is and the
    likelihood-ratio weights blow up. Measured: variance about 1,750x *worse*
    than plain, and coverage 0.296.
    """
    assert importance_sampling_shift(*DEEP_ITM.args) < 0
    assert variance_ratio(IMPORTANCE, DEEP_ITM, n_paths=20_000,
                          seeds=range(150)) < 0.1


# ---------------------------------------------------------------------------
# 5. The key experiment: does anything fix the Phase 3 coverage failure?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,method", [
    ("antithetic", ANTITHETIC),
    ("control", CONTROL),
    ("anti+control", BOTH),
])
def test_classical_variance_reduction_does_not_fix_broken_coverage(name, method):
    """Making the spread smaller does not make a rare event less rare.

    On the K=160 six-month call, plain Monte Carlo covers 87% of the time at
    N=20,000. Antithetic, control and both together all leave it at 86-87%:
    they shrink the variance of an estimator whose *shape* is the problem.
    """
    study = run_seed_study(method, BROKEN, 20_000, SEEDS, name)
    assert study.coverage < 0.92


def test_importance_sampling_restores_coverage():
    """The one method that fixes it, because it fixes the actual cause.

    Shifting the drift by -d2 = +3.22 standard deviations puts the strike at
    the median, so about half the paths pay instead of one in 1,500. Measured:
    coverage back to 0.957 and a 1,745x variance reduction on the same option
    where the other three methods did nothing.
    """
    study = run_seed_study(IMPORTANCE, BROKEN, 20_000, SEEDS, "importance")
    assert study.coverage >= 0.93
    assert study.se_ratio == pytest.approx(1.0, abs=0.15)
    assert variance_ratio(IMPORTANCE, BROKEN) > 100.0


def test_importance_sampling_is_unbiased_where_plain_is_unusable():
    truth = call_price(*DEEP_OTM)
    study = run_seed_study(IMPORTANCE, BROKEN, 20_000, SEEDS, "importance")
    assert study.mean_price == pytest.approx(truth, rel=0.01)


# ---------------------------------------------------------------------------
# The importance-sampling shift
# ---------------------------------------------------------------------------

def test_shift_puts_the_strike_at_the_median():
    """The shifted median path should finish exactly at the strike."""
    S, K, r, sigma, T = DEEP_OTM
    mu = importance_sampling_shift(S, K, r, sigma, T)
    median_terminal = S * math.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * mu)
    assert median_terminal == pytest.approx(K, rel=1e-12)


def test_shift_equals_minus_d2():
    from pricing.black_scholes import d1_d2

    _, d2 = d1_d2(*DEEP_OTM)
    assert importance_sampling_shift(*DEEP_OTM) == pytest.approx(-d2, rel=1e-12)


def test_zero_shift_reproduces_plain_monte_carlo_exactly():
    """A shift of zero is not an approximation of plain Monte Carlo; it is it."""
    shifted = importance_sampling_monte_carlo(*ATM, "call", n_paths=10_000, seed=5,
                                              shift=0.0)
    plain = monte_carlo_price(*ATM, "call", n_paths=10_000, seed=5)
    assert shifted.price == plain.price
    assert shifted.standard_error == plain.standard_error


# ---------------------------------------------------------------------------
# Bookkeeping: equal path counts, reproducibility, rejected inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("function", [
    antithetic_monte_carlo,
    control_variate_monte_carlo,
    antithetic_control_monte_carlo,
    importance_sampling_monte_carlo,
])
def test_result_reports_the_paths_actually_spent(function):
    """Comparisons are only fair if every method reports the same budget."""
    result = function(*ATM, "call", n_paths=10_000, seed=1)
    assert result.n_paths == 10_000
    assert result.seed == 1


@pytest.mark.parametrize("function", [
    antithetic_monte_carlo,
    control_variate_monte_carlo,
    antithetic_control_monte_carlo,
    importance_sampling_monte_carlo,
])
def test_same_seed_reproduces_exactly(function):
    first = function(*ATM, "call", n_paths=5_000, seed=42)
    again = function(*ATM, "call", n_paths=5_000, seed=42)
    other = function(*ATM, "call", n_paths=5_000, seed=43)
    assert first == again
    assert first.price != other.price


@pytest.mark.parametrize("function", [antithetic_monte_carlo,
                                      antithetic_control_monte_carlo])
def test_antithetic_rejects_an_odd_path_count(function):
    with pytest.raises(ValueError, match="even"):
        function(*ATM, "call", n_paths=1_001, seed=1)


@pytest.mark.parametrize("function", [control_variate_monte_carlo,
                                      antithetic_control_monte_carlo])
@pytest.mark.parametrize("pilot", [-0.1, 1.0, 1.5])
def test_bad_pilot_fraction_raises(function, pilot):
    with pytest.raises(ValueError, match="pilot_fraction"):
        function(*ATM, "call", n_paths=1_000, seed=1, pilot_fraction=pilot)


def test_importance_sampling_needs_randomness_to_shift():
    with pytest.raises(ValueError, match="sigma > 0"):
        importance_sampling_monte_carlo(100.0, 100.0, 0.05, 0.0, 1.0, "call",
                                        n_paths=1_000, seed=1)


@pytest.mark.parametrize("function", [
    antithetic_monte_carlo,
    control_variate_monte_carlo,
    antithetic_control_monte_carlo,
    importance_sampling_monte_carlo,
])
def test_bad_kind_raises(function):
    with pytest.raises(ValueError, match="call.*put"):
        function(*ATM, "butterfly", n_paths=1_000, seed=1)


def test_straddle_is_priced_by_every_applicable_method():
    """The straddle is a real payoff, not just a failure case to point at."""
    truth = price(*ATM, "straddle")
    for function in (monte_carlo_price, antithetic_monte_carlo,
                     control_variate_monte_carlo, antithetic_control_monte_carlo):
        result = function(*ATM, "straddle", n_paths=200_000, seed=3)
        assert result.price == pytest.approx(truth, rel=0.02)
