"""Tests for the experiment harness.

The harness is measurement equipment, so it has to be calibrated before it is
trusted to judge anything. The technique here is **fake estimators**: feed the
harness methods whose bias, spread and honesty are known in advance by
construction, and check it reports exactly those.

That matters because the harness's job in Phase 4 is to decide whether a
variance-reduced estimator is genuinely better. If the harness itself
miscounted coverage or confused "reported error" with "real error", every
conclusion downstream would be wrong and nothing else in the project would
catch it.
"""

import math

import numpy as np
import pytest

from experiments.harness import (
    Option,
    PayoffShape,
    clt_diagnostic,
    measure_payoff_shape,
    paths_for_target,
    plain_monte_carlo,
    run_grid,
    run_seed_study,
)
from pricing.black_scholes import call_price, price
from pricing.monte_carlo import Z_95, MonteCarloResult, monte_carlo_price

ATM_CALL = Option("atm call", S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, kind="call")
SEEDS = tuple(range(200))


# ---------------------------------------------------------------------------
# Fake estimators: known behaviour by construction
# ---------------------------------------------------------------------------

def make_fake(truth, scatter, reported_se, bias=0.0):
    """An estimator whose estimates are normal around ``truth + bias``.

    ``scatter`` is how much the estimates really move; ``reported_se`` is what
    the estimator *claims* about itself. Setting them differently is how the
    honesty checks below are tested.
    """
    def estimator(option, n_paths, seed):
        rng = np.random.default_rng(seed)
        estimate = truth + bias + scatter * float(rng.standard_normal())
        half_width = Z_95 * reported_se
        return MonteCarloResult(
            price=estimate,
            standard_error=reported_se,
            ci_low=estimate - half_width,
            ci_high=estimate + half_width,
            n_paths=n_paths,
            seed=seed,
        )
    return estimator


def test_harness_measures_an_honest_estimator_as_honest():
    """Scatter and reported error agree: ratio 1, coverage 95%, no bias."""
    truth = ATM_CALL.black_scholes()
    study = run_seed_study(make_fake(truth, scatter=0.5, reported_se=0.5),
                           ATM_CALL, n_paths=1_000, seeds=range(2_000))

    assert study.spread == pytest.approx(0.5, rel=0.05)
    assert study.se_ratio == pytest.approx(1.0, rel=0.05)
    assert study.coverage == pytest.approx(0.95, abs=0.02)
    assert study.bias == pytest.approx(0.0, abs=4 * 0.5 / math.sqrt(2_000))
    assert study.rmse == pytest.approx(0.5, rel=0.05)


def test_harness_catches_an_over_confident_estimator():
    """The bug a price check cannot see.

    This estimator is perfectly unbiased and its prices look fine -- but it
    reports an error bar four times too small. The harness has to notice,
    because Phase 4 could easily produce exactly this failure.
    """
    truth = ATM_CALL.black_scholes()
    study = run_seed_study(make_fake(truth, scatter=0.8, reported_se=0.2),
                           ATM_CALL, n_paths=1_000, seeds=range(2_000))

    assert study.bias == pytest.approx(0.0, abs=0.1)        # the price is fine
    assert study.se_ratio == pytest.approx(0.25, rel=0.05)  # the error bar is not
    assert study.coverage < 0.65                            # and coverage collapses


def test_harness_catches_an_under_confident_estimator():
    """The opposite failure: error bars so wide they are useless."""
    truth = ATM_CALL.black_scholes()
    study = run_seed_study(make_fake(truth, scatter=0.2, reported_se=0.8),
                           ATM_CALL, n_paths=1_000, seeds=range(1_000))
    assert study.se_ratio == pytest.approx(4.0, rel=0.05)
    assert study.coverage > 0.999


def test_harness_catches_a_biased_estimator():
    """Aimed at the wrong number: RMSE stays large however many seeds are run."""
    truth = ATM_CALL.black_scholes()
    study = run_seed_study(make_fake(truth, scatter=0.1, reported_se=0.1, bias=0.5),
                           ATM_CALL, n_paths=1_000, seeds=range(1_000))
    assert study.bias == pytest.approx(0.5, abs=0.02)
    assert study.rmse == pytest.approx(math.sqrt(0.5**2 + 0.1**2), rel=0.05)
    assert study.coverage < 0.01        # the intervals are in the wrong place


def test_rmse_decomposes_into_bias_and_spread():
    """An identity the harness must satisfy: MSE = bias^2 + variance.

    (With the 1/n convention for the variance, hence the (n-1)/n factor on the
    sample spread, which uses ddof=1.)
    """
    study = run_seed_study(make_fake(10.0, scatter=0.3, reported_se=0.3, bias=0.2),
                           ATM_CALL, n_paths=1_000, seeds=range(500))
    n = study.n_seeds
    expected = math.sqrt(study.bias ** 2 + study.spread ** 2 * (n - 1) / n)
    assert study.rmse == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# The real estimator through the harness
# ---------------------------------------------------------------------------

def test_plain_monte_carlo_matches_calling_the_pricer_directly():
    """The harness's estimator wrapper must not change any numbers."""
    through_harness = plain_monte_carlo(ATM_CALL, n_paths=10_000, seed=7)
    direct = monte_carlo_price(*ATM_CALL.args, ATM_CALL.kind, n_paths=10_000, seed=7)
    assert through_harness == direct


def test_real_estimator_is_measured_as_unbiased_and_honest():
    """Phase 3's headline result, in miniature."""
    study = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=20_000, seeds=SEEDS,
                           estimator_name="plain")

    assert study.estimator == "plain"
    assert study.n_seeds == len(SEEDS)
    assert study.truth == pytest.approx(call_price(*ATM_CALL.args))
    # Unbiased to within a few standard errors of the mean of 200 runs.
    assert abs(study.bias) < 4 * study.spread / math.sqrt(study.n_seeds)
    assert study.se_ratio == pytest.approx(1.0, abs=0.15)
    assert 0.90 <= study.coverage <= 0.99


def test_rmse_and_reported_error_agree_for_an_unbiased_estimator():
    """When there is no bias, RMSE should be the same size as the error bar.

    This is what makes the standard error useful at all: it is a live estimate
    of an accuracy you cannot otherwise see without knowing the answer.
    """
    study = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=20_000, seeds=SEEDS)
    assert study.rmse == pytest.approx(study.mean_reported_se, rel=0.2)


def test_payoff_std_is_recovered_from_the_standard_error():
    """``s = se * sqrt(N)`` must not depend on N -- that is the 1/sqrt(N) law."""
    stds = [run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=n,
                           seeds=range(50)).payoff_std
            for n in (10_000, 100_000, 1_000_000)]
    assert max(stds) / min(stds) < 1.05


def test_runtime_grows_with_paths():
    fast = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=10_000, seeds=range(20))
    slow = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=1_000_000, seeds=range(20))
    assert slow.median_runtime > fast.median_runtime
    assert slow.total_runtime > slow.median_runtime     # summed over 20 runs


def test_study_is_reproducible():
    first = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=5_000, seeds=range(30))
    again = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=5_000, seeds=range(30))
    assert first.prices == again.prices
    assert first.coverage == again.coverage


def test_row_is_csv_ready():
    study = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=5_000, seeds=range(30),
                           estimator_name="plain")
    row = study.row()
    assert row["estimator"] == "plain"
    assert row["option"] == "atm call"
    assert int(row["n_paths"]) == 5_000
    assert all(isinstance(key, str) for key in row)


# ---------------------------------------------------------------------------
# Efficiency arithmetic
# ---------------------------------------------------------------------------

def test_paths_for_target_is_quadratic_in_the_target():
    """Ten times the accuracy costs a hundred times the paths."""
    coarse = paths_for_target(payoff_std=15.0, target=0.01)
    fine = paths_for_target(payoff_std=15.0, target=0.001)
    assert fine / coarse == pytest.approx(100.0, rel=1e-12)


def test_paths_for_target_matches_the_formula():
    assert paths_for_target(10.0, 0.01, confidence=0.95) == pytest.approx(
        (Z_95 * 10.0 / 0.01) ** 2, rel=1e-12
    )
    # confidence=None targets the standard error itself, so no z factor.
    assert paths_for_target(10.0, 0.01, confidence=None) == pytest.approx(
        (10.0 / 0.01) ** 2, rel=1e-12
    )


def test_paths_for_target_rejects_a_nonsense_target():
    with pytest.raises(ValueError, match="target"):
        paths_for_target(10.0, 0.0)


def test_predicted_paths_actually_deliver_the_target_accuracy():
    """The prediction is the point, so check it against a real run.

    Work out how many paths the harness thinks are needed for a 1-cent error
    bar, run that many, and confirm the error bar really comes out at a cent.
    """
    pilot = run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=20_000, seeds=range(20))
    needed = int(pilot.paths_for_target(0.01))
    achieved = monte_carlo_price(*ATM_CALL.args, ATM_CALL.kind,
                                 n_paths=needed, seed=999)
    assert achieved.ci_width / 2 == pytest.approx(0.01, rel=0.05)


def test_variance_falls_exactly_like_one_over_paths():
    """The 1/sqrt(N) law, stated as a variance: double the paths, halve it.

    Measured ratios per doubling from N=50,000 to 800,000 are 2.06, 2.01, 1.90
    and 2.03. This is the half of the efficiency story that behaves.
    """
    variances = [run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=n,
                                seeds=range(300)).spread ** 2
                 for n in (50_000, 100_000)]
    assert variances[0] / variances[1] == pytest.approx(2.0, rel=0.15)


def test_efficiency_is_only_roughly_constant_because_runtime_is_superlinear():
    """Variance times runtime is *not* flat in N here, and the reason is cache.

    The theory says this product should be constant: variance falls like 1/N
    and runtime rises like N. The variance half holds exactly. The runtime half
    does not -- past about 100,000 paths the working arrays no longer fit in
    cache, so per-path cost rises and the product climbs from 3.2e-6 at
    N=100,000 to 7.7e-6 at N=400,000, flattening near 7.6e-6 by N=800,000.

    So the test asserts the real behaviour (within a factor of three) rather
    than the textbook claim. The practical rule it protects: compare estimators
    at the same path count, never across path counts.
    """
    products = [run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=n,
                               seeds=range(100)).variance_times_runtime
                for n in (100_000, 800_000)]
    ratio = products[1] / products[0]
    assert 0.33 < ratio < 3.0


def test_runtime_per_path_rises_with_path_count():
    """The memory effect, measured directly rather than inferred."""
    per_path = [run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=n,
                               seeds=range(50)).runtime_per_path
                for n in (50_000, 800_000)]
    assert per_path[1] > per_path[0]


# ---------------------------------------------------------------------------
# Payoff shape, the explanation for coverage failures
# ---------------------------------------------------------------------------

def test_payoff_shape_of_an_at_the_money_call():
    """Right-skewed: capped at zero below, unbounded above."""
    shape = measure_payoff_shape(ATM_CALL, n_paths=500_000)
    assert shape.mean == pytest.approx(ATM_CALL.black_scholes(), abs=0.1)
    assert shape.skewness > 1.0
    assert shape.excess_kurtosis > 1.0
    assert 0.3 < shape.zero_fraction < 0.6


def test_zero_fraction_matches_the_closed_form_probability():
    """A call pays nothing when it finishes below the strike.

    Under the risk-neutral measure that happens with probability ``1 - N(d2)``,
    so the measured fraction of zero payoffs is a direct check on the
    simulation as well as on the shape measurement.
    """
    from scipy.stats import norm as normal

    from pricing.black_scholes import d1_d2

    _, d2 = d1_d2(*ATM_CALL.args)
    shape = measure_payoff_shape(ATM_CALL, n_paths=500_000)
    assert shape.zero_fraction == pytest.approx(1 - normal.cdf(d2), abs=0.005)


def test_further_out_of_the_money_means_more_skew_and_more_zeros():
    """The ordering that explains everything about coverage in Phase 3."""
    shapes = [
        measure_payoff_shape(
            Option(f"K={K}", S=100.0, K=K, r=0.05, sigma=0.2, T=1.0, kind="call"),
            n_paths=400_000,
        )
        for K in (80.0, 100.0, 140.0)
    ]
    skews = [s.skewness for s in shapes]
    zeros = [s.zero_fraction for s in shapes]
    assert skews[0] < skews[1] < skews[2]
    assert zeros[0] < zeros[1] < zeros[2]


def test_deep_in_the_money_call_is_nearly_symmetric():
    """Almost every path pays, so the payoff is close to the stock itself.

    The strike is 3.6 standard deviations below the forward here, so the
    closed form puts the chance of expiring worthless at 1.5e-4; the measured
    fraction over 400,000 paths is 1.8e-4. Skewness comes out at 0.62, against
    1.4 for the at-the-money call -- much closer to symmetric, which is why
    its confidence intervals behave.
    """
    deep = Option("deep itm", S=200.0, K=100.0, r=0.05, sigma=0.2, T=1.0, kind="call")
    shape = measure_payoff_shape(deep, n_paths=400_000)
    assert shape.zero_fraction < 1e-3
    assert shape.zero_fraction == pytest.approx(1.5e-4, abs=1e-4)
    assert abs(shape.skewness) < 1.5


def test_clt_diagnostic_combines_skew_and_paths():
    assert clt_diagnostic(10.0, 100) == pytest.approx(1.0)
    assert clt_diagnostic(10.0, 10_000) == pytest.approx(0.1)
    assert clt_diagnostic(-4.0, 4) == pytest.approx(2.0)       # sign does not matter


def test_paths_for_clt_rule_of_thumb():
    shape = PayoffShape(ATM_CALL, n_paths=1, seed=0, mean=0.0, std=1.0,
                        skewness=5.0, excess_kurtosis=0.0, zero_fraction=0.0)
    assert shape.paths_for_clt(threshold=0.1) == pytest.approx(2_500.0)
    assert clt_diagnostic(5.0, shape.paths_for_clt(0.1)) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# The grid runner
# ---------------------------------------------------------------------------

def test_run_grid_covers_every_combination():
    options = [ATM_CALL,
               Option("otm call", S=100.0, K=120.0, r=0.05, sigma=0.2, T=1.0)]
    messages = []
    studies = run_grid(
        {"plain": plain_monte_carlo},
        options,
        path_counts=[1_000, 5_000],
        seeds=range(20),
        progress=messages.append,
    )
    assert len(studies) == 4
    assert {(s.option.label, s.n_paths) for s in studies} == {
        ("atm call", 1_000), ("atm call", 5_000),
        ("otm call", 1_000), ("otm call", 5_000),
    }
    assert all(s.estimator == "plain" for s in studies)
    assert len(messages) == 4


def test_run_grid_takes_several_estimators():
    """The Phase 4 shape: two methods, same options, same seeds, one table."""
    truth = ATM_CALL.black_scholes()
    studies = run_grid(
        {"plain": plain_monte_carlo, "fake": make_fake(truth, 0.1, 0.1)},
        [ATM_CALL],
        path_counts=[1_000],
        seeds=range(30),
    )
    assert {s.estimator for s in studies} == {"plain", "fake"}
    # The fake one is deliberately much tighter, which the harness should show.
    by_name = {s.estimator: s for s in studies}
    assert by_name["fake"].spread < by_name["plain"].spread


# ---------------------------------------------------------------------------
# Option
# ---------------------------------------------------------------------------

def test_option_knows_its_own_black_scholes_price():
    assert ATM_CALL.black_scholes() == price(*ATM_CALL.args, "call")
    put = Option("atm put", S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, kind="put")
    assert put.black_scholes() == price(*put.args, "put")


def test_option_moneyness():
    assert ATM_CALL.moneyness == 1.0
    assert Option("x", S=130.0, K=100.0, r=0.0, sigma=0.2, T=1.0).moneyness == 1.3


def test_option_rejects_a_bad_kind():
    with pytest.raises(ValueError, match="call.*put"):
        Option("bad", S=100.0, K=100.0, r=0.0, sigma=0.2, T=1.0, kind="binary")


def test_seed_study_needs_at_least_one_seed():
    with pytest.raises(ValueError, match="seed"):
        run_seed_study(plain_monte_carlo, ATM_CALL, n_paths=1_000, seeds=[])
