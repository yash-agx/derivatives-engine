"""Tests for Phase 7: Longstaff-Schwartz, the Bermudan tree and exercise boundaries.

The validation tests at the bottom re-run exactly the pre-registered
configuration (docs/phase7_preregistration.md): same cases, same seeds, same
settings, against the reference values written down before any LSM price was
computed.
"""

import dataclasses
import math

import numpy as np
import pytest

from pricing.binomial import american_binomial, european_binomial
from pricing.black_scholes import price as black_scholes
from pricing.exercise_tree import (
    bermudan_binomial,
    bermudan_exercise_boundary,
    tree_exercise_boundary,
)
from pricing.lsm import (
    ExerciseRule,
    apply_exercise_rule,
    fit_exercise_rule,
    laguerre_basis,
    lsm_exercise_boundary,
    lsm_price,
    simulate_paths,
)

K, R = 40.0, 0.06


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------

class TestLaguerreBasis:
    x = np.linspace(0.0, 3.0, 31)

    def test_first_four_match_the_closed_forms(self):
        cols = laguerre_basis(self.x, 3)
        x = self.x
        assert np.allclose(cols[:, 0], 1.0)
        assert np.allclose(cols[:, 1], 1.0 - x)
        assert np.allclose(cols[:, 2], 1.0 - 2.0 * x + x ** 2 / 2.0)
        assert np.allclose(cols[:, 3], 1.0 - 3.0 * x + 1.5 * x ** 2 - x ** 3 / 6.0)

    def test_recurrence_matches_numpy_laguerre(self):
        from numpy.polynomial import laguerre
        cols = laguerre_basis(self.x, 6)
        for n in range(7):
            coef = np.zeros(n + 1)
            coef[n] = 1.0
            assert np.allclose(cols[:, n], laguerre.lagval(self.x, coef))

    @pytest.mark.parametrize("degree", range(0, 7))
    def test_degree_d_has_d_plus_one_columns(self, degree):
        assert laguerre_basis(self.x, degree).shape == (self.x.size, degree + 1)

    def test_weighted_is_a_constant_plus_weighted_polynomials(self):
        cols = laguerre_basis(self.x, 3, weighted=True)
        weight = np.exp(-self.x / 2.0)
        assert cols.shape == (self.x.size, 4)
        assert np.allclose(cols[:, 0], 1.0)
        assert np.allclose(cols[:, 1], weight)
        assert np.allclose(cols[:, 2], weight * (1.0 - self.x))
        assert np.allclose(cols[:, 3], weight * (1.0 - 2.0 * self.x + self.x ** 2 / 2.0))

    @pytest.mark.parametrize("bad", [-1, 1.5])
    def test_rejects_bad_degree(self, bad):
        with pytest.raises(ValueError):
            laguerre_basis(self.x, bad)

    def test_weighted_needs_degree_one(self):
        with pytest.raises(ValueError):
            laguerre_basis(self.x, 0, weighted=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

class TestPaths:
    def test_shape(self):
        paths = simulate_paths(36.0, R, 0.2, 1.0, 50, 1000, np.random.default_rng(0))
        assert paths.shape == (1000, 50)
        assert (paths > 0).all()

    def test_same_seed_same_paths(self):
        a = simulate_paths(36.0, R, 0.2, 1.0, 10, 100, np.random.default_rng(3))
        b = simulate_paths(36.0, R, 0.2, 1.0, 10, 100, np.random.default_rng(3))
        assert np.array_equal(a, b)

    def test_risk_neutral_mean_on_every_date(self):
        # E[S_t] = S * exp(r t): the discounted stock is a martingale.
        S, sigma, T, n_dates, n = 36.0, 0.2, 1.0, 10, 400_000
        paths = simulate_paths(S, R, sigma, T, n_dates, n, np.random.default_rng(1))
        for m in range(n_dates):
            t = (m + 1) * T / n_dates
            sd = S * math.exp(R * t) * math.sqrt(math.exp(sigma ** 2 * t) - 1.0)
            assert abs(paths[:, m].mean() - S * math.exp(R * t)) < 4 * sd / math.sqrt(n)

    def test_log_increments_have_the_right_variance(self):
        sigma, T, n_dates = 0.4, 2.0, 8
        paths = simulate_paths(40.0, R, sigma, T, n_dates, 200_000, np.random.default_rng(2))
        steps = np.diff(np.log(paths), axis=1)
        assert np.allclose(steps.var(axis=0), sigma ** 2 * T / n_dates, rtol=0.02)


# ---------------------------------------------------------------------------
# The Bermudan tree and the tree boundary
# ---------------------------------------------------------------------------

class TestBermudanTree:
    @pytest.mark.parametrize("S", [36.0, 40.0, 44.0])
    def test_one_step_per_date_is_the_american_tree(self, S):
        assert bermudan_binomial(S, K, R, 0.2, 1.0, 200, 1, "put") == pytest.approx(
            american_binomial(S, K, R, 0.2, 1.0, 200, "put"), abs=1e-12)

    @pytest.mark.parametrize("S", [36.0, 40.0, 44.0])
    def test_one_date_is_european_or_exercise_now(self, S):
        european = european_binomial(S, K, R, 0.2, 1.0, 400, "put")
        expected = max(european, max(K - S, 0.0))
        assert bermudan_binomial(S, K, R, 0.2, 1.0, 1, 400, "put") == pytest.approx(
            expected, abs=1e-12)

    @pytest.mark.parametrize("S,sigma,T", [(36, 0.2, 1), (40, 0.4, 2), (44, 0.2, 2)])
    def test_european_below_bermudan_below_american(self, S, sigma, T):
        steps = 2000
        european = european_binomial(S, K, R, sigma, T, steps, "put")
        bermudan = bermudan_binomial(S, K, R, sigma, T, 50, steps // 50, "put")
        american = american_binomial(S, K, R, sigma, T, steps, "put")
        assert european < bermudan < american

    def test_more_dates_is_worth_more_when_dates_are_nested(self):
        # 5 dates are a subset of 10, which are a subset of 20, ...: more
        # choices can never be worth less on the same tree.
        values = [bermudan_binomial(36.0, K, R, 0.2, 1.0, m, 4000 // m, "put")
                  for m in (5, 10, 20, 40, 80)]
        assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))

    def test_bermudan_call_equals_european_call(self):
        # No dividends: exercising a call early is never optimal, on any dates.
        assert bermudan_binomial(40.0, K, R, 0.2, 1.0, 50, 40, "call") == pytest.approx(
            european_binomial(40.0, K, R, 0.2, 1.0, 2000, "call"), abs=1e-12)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            bermudan_binomial(36.0, K, R, 0.2, 1.0, 0, 10, "put")
        with pytest.raises(ValueError):
            bermudan_binomial(36.0, K, R, 0.0, 1.0, 10, 10, "put")


class TestTreeBoundary:
    def test_put_boundary_is_below_strike_and_rises_to_it(self):
        times, boundary = tree_exercise_boundary(36.0, K, R, 0.2, 1.0, 1000, "put")
        finite = boundary[np.isfinite(boundary)]
        assert (finite < K).all()
        # Close to expiry the critical price approaches the strike.
        assert boundary[-1] > 39.0
        # Early in the year the boundary is lower (more to lose by exercising).
        late, early = np.nanmean(boundary[-100:]), np.nanmean(boundary[200:300])
        assert early < late

    def test_call_has_no_boundary_without_dividends(self):
        _, boundary = tree_exercise_boundary(40.0, K, R, 0.2, 1.0, 500, "call")
        assert np.isnan(boundary).all()


class TestBermudanBoundary:
    def test_one_step_per_date_is_the_american_tree(self):
        times, berm = bermudan_exercise_boundary(36.0, K, R, 0.2, 1.0, 200, 1, "put")
        tree_t, tree_b = tree_exercise_boundary(36.0, K, R, 0.2, 1.0, 200, "put")
        np.testing.assert_allclose(times, tree_t)
        np.testing.assert_array_equal(berm, tree_b)

    def test_dates_match_the_lsm_rule(self):
        times, berm = bermudan_exercise_boundary(36.0, K, R, 0.2, 1.0, 50, 20, "put")
        assert times.size == 49
        np.testing.assert_allclose(times, np.arange(1, 50) / 50)

    @pytest.mark.parametrize("sigma", [0.2, 0.4])
    def test_sits_above_the_american_boundary(self, sigma):
        # Same lattice (50 dates x 40 steps = 2000 steps), so the nodes are
        # identical. Holding a Bermudan is worth less at every node, so the
        # exercise region can only be larger: the put boundary is never lower.
        times, berm = bermudan_exercise_boundary(36.0, K, R, sigma, 1.0, 50, 40, "put")
        tree_t, tree_b = tree_exercise_boundary(36.0, K, R, sigma, 1.0, 2000, "put")
        at_dates = tree_b[40 * np.arange(1, 50) - 1]
        both = np.isfinite(berm) & np.isfinite(at_dates)
        assert both.sum() > 40
        assert (berm[both] >= at_dates[both]).all()
        assert (berm[both] < K).all()
        assert np.mean(berm[both] - at_dates[both]) > 0.2

    def test_call_has_no_boundary_without_dividends(self):
        _, boundary = bermudan_exercise_boundary(40.0, K, R, 0.2, 1.0, 50, 10, "call")
        assert np.isnan(boundary).all()

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            bermudan_exercise_boundary(36.0, K, R, 0.2, 1.0, 50, 0, "put")


# ---------------------------------------------------------------------------
# LSM mechanics
# ---------------------------------------------------------------------------

class TestLSMMechanics:
    def test_reproducible_with_the_same_seed(self):
        a = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=5_000, seed=11)
        b = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=5_000, seed=11)
        assert a.price == b.price and a.in_sample_price == b.in_sample_price

    def test_different_seeds_give_different_prices(self):
        a = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=5_000, seed=11)
        b = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=5_000, seed=12)
        assert a.price != b.price

    def test_training_and_pricing_paths_are_different(self):
        # The out-of-sample estimate is only out of sample if the two
        # streams spawned from one seed differ.
        train_seq, price_seq = np.random.SeedSequence(5).spawn(2)
        a = simulate_paths(36.0, R, 0.2, 1.0, 5, 1000, np.random.default_rng(train_seq))
        b = simulate_paths(36.0, R, 0.2, 1.0, 5, 1000, np.random.default_rng(price_seq))
        assert not np.allclose(a, b)
        assert abs(np.corrcoef(a[:, -1], b[:, -1])[0, 1]) < 0.1

    def test_in_sample_is_the_training_cash_flows(self):
        res = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=4_000, n_dates=10, seed=8)
        train_seq, _ = np.random.SeedSequence(8).spawn(2)
        _, cash = fit_exercise_rule(36.0, K, R, 0.2, 1.0, "put", 4_000, 10,
                                    np.random.default_rng(train_seq))
        assert res.in_sample_price == pytest.approx(cash.mean(), abs=1e-14)

    def test_applying_the_rule_to_its_own_paths_reproduces_in_sample(self):
        # Fitting and then re-applying on the same paths must take the same
        # decisions, so the two code paths agree exactly.
        rng = np.random.default_rng(21)
        state = rng.bit_generator.state
        rule, cash = fit_exercise_rule(36.0, K, R, 0.2, 1.0, "put", 3_000, 20, rng)
        rng.bit_generator.state = state
        assert np.allclose(apply_exercise_rule(rule, 3_000, rng), cash, atol=1e-12)

    def test_rule_matches_a_forward_walk_to_first_exercise(self):
        rng = np.random.default_rng(4)
        rule, _ = fit_exercise_rule(36.0, K, R, 0.2, 1.0, "put", 5_000, 12, rng)
        state = rng.bit_generator.state
        cash = apply_exercise_rule(rule, 500, rng)
        rng.bit_generator.state = state
        paths = simulate_paths(36.0, R, 0.2, 1.0, 12, 500, rng)
        forward = np.empty(500)
        for i, path in enumerate(paths):
            forward[i] = math.exp(-R) * max(K - path[-1], 0.0)
            for m in range(11):
                exercise_value = K - path[m]
                if exercise_value > 0 and exercise_value > rule.continuation(m, path[m:m + 1])[0]:
                    forward[i] = math.exp(-R * (m + 1) / 12) * exercise_value
                    break
        assert np.allclose(cash, forward, atol=1e-12)

    def test_ci_and_gap(self):
        res = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=5_000, seed=1)
        assert res.ci_low < res.price < res.ci_high
        assert res.ci_high - res.price == pytest.approx(1.959963985 * res.standard_error)
        assert res.gap == pytest.approx(res.in_sample_price - res.price)

    def test_one_date_is_european(self):
        # With only the expiry date there is no decision: plain Monte Carlo.
        res = lsm_price(44.0, K, R, 0.2, 1.0, "put", n_paths=200_000, n_dates=1, seed=2)
        assert all(c is None for c in res.rule.coefficients)
        assert res.contains(black_scholes(44.0, K, R, 0.2, 1.0, "put"))

    def test_exercises_immediately_when_deep_in_the_money(self):
        # Spot 20, strike 40: holding on can only lose interest on the strike.
        res = lsm_price(20.0, K, R, 0.2, 1.0, "put", n_paths=5_000, seed=3)
        assert res.exercise_now
        assert res.price == 20.0 and res.standard_error == 0.0
        assert american_binomial(20.0, K, R, 0.2, 1.0, 1000, "put") == pytest.approx(20.0)

    def test_time_zero_decision_is_made_on_the_training_paths(self):
        rule, cash = fit_exercise_rule(20.0, K, R, 0.2, 1.0, "put", 5_000, 50,
                                       np.random.default_rng(4))
        assert rule.exercise_now
        assert (cash == 20.0).all()
        # The pricing paths only follow the rule: they cannot overturn it.
        priced = apply_exercise_rule(rule, 1_000, np.random.default_rng(5))
        assert (priced == 20.0).all()
        # Without the time-0 exercise the same rule is simulated, and is worth less.
        held = apply_exercise_rule(dataclasses.replace(rule, exercise_now=False),
                                   1_000, np.random.default_rng(5))
        assert held.std() > 0 and held.mean() < 20.0

    def test_holds_at_time_zero_when_waiting_is_worth_more(self):
        res = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=5_000, seed=1)
        assert not res.exercise_now and not res.rule.exercise_now
        assert res.standard_error > 0

    def test_str_reports_the_confidence_level_used(self):
        res = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=2_000, seed=1, confidence=0.9)
        assert "90% CI" in str(res)
        assert "95% CI" in str(lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=2_000, seed=1))

    def test_dates_with_too_few_paths_in_the_money_never_exercise(self):
        # Deep out-of-the-money put, few paths and a short life: early dates
        # have almost nothing in the money to regress on.
        rule, cash = fit_exercise_rule(80.0, K, R, 0.2, 0.1, "put", 50, 10,
                                       np.random.default_rng(0))
        assert rule.coefficients[0] is None
        assert np.isinf(rule.continuation(0, np.array([30.0]))).all()
        assert np.isfinite(cash).all()

    def test_lower_bias_more_paths_closer_to_bermudan(self):
        # Low-biased: averaged over seeds, the out-of-sample price sits at or
        # below the Bermudan value it estimates.
        bermudan = bermudan_binomial(36.0, K, R, 0.2, 1.0, 50, 100, "put")
        prices = [lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=2_000, seed=s).price
                  for s in range(60)]
        mean, se = np.mean(prices), np.std(prices, ddof=1) / math.sqrt(len(prices))
        assert mean < bermudan + 2 * se

    @pytest.mark.parametrize("kwargs", [
        {"n_dates": 0}, {"n_paths": 1}, {"degree": -1}, {"sigma": 0.0},
        {"T": 0.0}, {"kind": "straddle"}, {"n_price_paths": 1}, {"confidence": 1.0},
    ])
    def test_rejects_bad_inputs(self, kwargs):
        args = {"S": 36.0, "K": K, "r": R, "sigma": 0.2, "T": 1.0, "kind": "put",
                "n_paths": 1000}
        args.update(kwargs)
        with pytest.raises(ValueError):
            lsm_price(**args)


class TestLSMBoundary:
    def test_boundary_is_below_strike_and_near_the_tree(self):
        res = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=100_000, seed=40_000)
        times, boundary = lsm_exercise_boundary(res.rule)
        assert isinstance(res.rule, ExerciseRule)
        assert times.size == 49
        assert (boundary[np.isfinite(boundary)] < K).all()
        tree_t, tree_b = tree_exercise_boundary(36.0, K, R, 0.2, 1.0, 2000, "put")
        tree_at_dates = np.interp(times, tree_t, tree_b)
        middle = slice(10, 40)
        assert np.nanmax(np.abs(boundary[middle] - tree_at_dates[middle])) < 1.0

    def test_boundary_is_close_to_the_bermudan_tree_mid_year(self):
        # The like-for-like curve: the tree that exercises on the same 50 dates.
        res = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=100_000, seed=40_000)
        _, boundary = lsm_exercise_boundary(res.rule)
        _, berm = bermudan_exercise_boundary(36.0, K, R, 0.2, 1.0, 50, 100, "put")
        middle = slice(10, 35)
        assert np.nanmedian(np.abs(boundary[middle] - berm[middle])) < 0.3


# ---------------------------------------------------------------------------
# The American call: equal to the European call without dividends
# ---------------------------------------------------------------------------

class TestAmericanCall:
    @pytest.mark.parametrize("S", [36.0, 40.0, 44.0])
    def test_lsm_call_matches_black_scholes(self, S):
        res = lsm_price(S, K, R, 0.2, 1.0, "call", n_paths=100_000, seed=50_000)
        assert res.contains(black_scholes(S, K, R, 0.2, 1.0, "call"))

    def test_mistaken_early_exercise_costs_almost_nothing(self):
        # The fitted rule is not exactly "never exercise", but on the same
        # paths it gives up a tiny fraction of a standard error.
        S = 40.0
        res = lsm_price(S, K, R, 0.2, 1.0, "call", n_paths=100_000, seed=50_001)
        rng = np.random.default_rng(99)
        state = rng.bit_generator.state
        cash = apply_exercise_rule(res.rule, 100_000, rng)
        rng.bit_generator.state = state
        paths = simulate_paths(S, R, 0.2, 1.0, 50, 100_000, rng)
        european = math.exp(-R) * np.maximum(paths[:, -1] - K, 0.0)
        assert abs(cash.mean() - european.mean()) < 0.25 * res.standard_error

    def test_tree_american_call_equals_european(self):
        assert american_binomial(40.0, K, R, 0.2, 1.0, 2000, "call") == pytest.approx(
            european_binomial(40.0, K, R, 0.2, 1.0, 2000, "call"), abs=1e-12)


# ---------------------------------------------------------------------------
# The pre-registered validation
# ---------------------------------------------------------------------------

# (S, sigma, T, American tree at 10,000 steps, Bermudan tree on 50 dates/yr),
# copied from docs/phase7_preregistration.md.
PREREGISTERED = [
    (36, 0.2, 1, 4.4867, 4.4779), (36, 0.2, 2, 4.8483, 4.8403),
    (36, 0.4, 1, 7.1090, 7.1013), (36, 0.4, 2, 8.5143, 8.5069),
    (38, 0.2, 1, 3.2572, 3.2502), (38, 0.2, 2, 3.7514, 3.7448),
    (38, 0.4, 1, 6.1547, 6.1477), (38, 0.4, 2, 7.6750, 7.6682),
    (40, 0.2, 1, 2.3195, 2.3141), (40, 0.2, 2, 2.8899, 2.8846),
    (40, 0.4, 1, 5.3182, 5.3119), (40, 0.4, 2, 6.9234, 6.9170),
    (42, 0.2, 1, 1.6212, 1.6170), (42, 0.2, 2, 2.2168, 2.2124),
    (42, 0.4, 1, 4.5882, 4.5825), (42, 0.4, 2, 6.2504, 6.2445),
    (44, 0.2, 1, 1.1130, 1.1099), (44, 0.2, 2, 1.6933, 1.6899),
    (44, 0.4, 1, 3.9528, 3.9477), (44, 0.4, 2, 5.6469, 5.6414),
]
CASE_IDS = [f"case{i}-S{c[0]}-v{c[1]}-T{c[2]}" for i, c in enumerate(PREREGISTERED)]


@pytest.fixture(scope="module")
def validation_runs():
    """The pre-registered runs: 100k + 100k paths, 50 dates/yr, degree 3, seed 7000+i."""
    return [lsm_price(S, K, R, sigma, T, "put", n_paths=100_000, n_dates=50 * T,
                      degree=3, seed=7000 + i)
            for i, (S, sigma, T, _, _) in enumerate(PREREGISTERED)]


@pytest.mark.parametrize("i", range(20), ids=CASE_IDS)
def test_preregistered_references_match_the_tree(i):
    # A 2,500-step tree is within ~1e-4 of the 10,000-step one, and cheap.
    S, sigma, T, american, bermudan = PREREGISTERED[i]
    assert american_binomial(S, K, R, sigma, T, 2_500, "put") == pytest.approx(american, abs=1e-3)
    assert bermudan_binomial(S, K, R, sigma, T, 50 * T, 2_500 // (50 * T), "put") == \
        pytest.approx(bermudan, abs=1e-3)


@pytest.mark.parametrize("i", range(20), ids=CASE_IDS)
def test_lsm_american_put_is_at_least_the_european_put(validation_runs, i):
    S, sigma, T, _, _ = PREREGISTERED[i]
    assert validation_runs[i].price >= black_scholes(S, K, R, sigma, T, "put")


# Case 11 (S=40, sigma=0.4, T=2) is the one pre-registered miss: z = -1.99
# against the American tree, on the low side the theory predicts (it passes
# against the Bermudan tree). Strict, so a change that silently "fixes" it
# fails here and has to be explained rather than absorbed.
KNOWN_MISSES = {11}
AMERICAN_CASES = [
    pytest.param(i, id=CASE_IDS[i], marks=pytest.mark.xfail(
        strict=True, reason="pre-registered miss, reported in the README"))
    if i in KNOWN_MISSES else pytest.param(i, id=CASE_IDS[i])
    for i in range(20)
]


@pytest.mark.parametrize("i", AMERICAN_CASES)
def test_american_reference_inside_lsm_95_ci(validation_runs, i):
    american = PREREGISTERED[i][3]
    res = validation_runs[i]
    assert res.contains(american), (
        f"LSM {res.price:.4f} +/- {res.standard_error:.4f} misses the tree's {american:.4f}")


@pytest.mark.parametrize("i", range(20), ids=CASE_IDS)
def test_bermudan_reference_inside_lsm_95_ci(validation_runs, i):
    bermudan = PREREGISTERED[i][4]
    assert validation_runs[i].contains(bermudan)


def test_overall_pass_rule(validation_runs):
    passes = sum(res.contains(case[3]) for res, case in zip(validation_runs, PREREGISTERED))
    assert passes >= 18


# ---------------------------------------------------------------------------
# Fixed-seed regression
# ---------------------------------------------------------------------------

def test_fixed_seed_regression():
    """Pins the exact output for one seed, so any change to the algorithm,
    the path construction or the seeding shows up as a failure here."""
    res = lsm_price(36.0, K, R, 0.2, 1.0, "put", n_paths=20_000, n_dates=50,
                    degree=3, seed=2026)
    assert res.price == pytest.approx(REGRESSION_PRICE, abs=1e-10)
    assert res.standard_error == pytest.approx(REGRESSION_SE, abs=1e-10)
    assert res.in_sample_price == pytest.approx(REGRESSION_IN_SAMPLE, abs=1e-10)


REGRESSION_PRICE = 4.418850383621461
REGRESSION_SE = 0.0208271213958893
REGRESSION_IN_SAMPLE = 4.499184057697395
