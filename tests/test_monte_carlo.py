"""Tests for the Monte Carlo European pricer.

Testing a random algorithm needs a different approach from testing a
deterministic one. "Is the price right?" is the wrong question -- the price is
a random variable, and a run that happened to land far from Black-Scholes is
not necessarily a bug. The right questions are statistical:

1. **Is the estimator unbiased?** Averaged over many seeds, does it converge
   to the closed-form price?
2. **Is the reported uncertainty honest?** A 95% confidence interval is a
   claim: build it many times and 95% of the intervals should contain the true
   value. That is the central test in this file, and it is the one that would
   catch a wrong standard-error formula -- a bug that leaves the price itself
   looking perfectly fine.
3. **Does the error shrink at the right rate?** 1/sqrt(N), so a hundred times
   the paths should buy exactly one decimal place.

Every run is seeded, so each test is deterministic and reproducible.
"""

import math

import numpy as np
import pytest

from pricing.black_scholes import CALL, PUT, call_price, price
from pricing.monte_carlo import Z_95, MonteCarloResult, monte_carlo_price

ATM = (100.0, 100.0, 0.05, 0.20, 1.0)       # S, K, r, sigma, T
DEEP_OTM = (100.0, 160.0, 0.05, 0.20, 0.5)  # a call worth 0.004: payoffs are rare


# ---------------------------------------------------------------------------
# 1. Is it unbiased? Does it land near Black-Scholes?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", [CALL, PUT])
@pytest.mark.parametrize("n_paths", [1_000, 10_000, 100_000, 1_000_000])
def test_price_is_within_a_few_standard_errors_of_black_scholes(kind, n_paths):
    """The honest version of "is the price right?".

    The error is measured in standard errors, not in dollars: landing within
    4 standard errors is what a correct estimator does (99.99% of the time if
    the normal approximation holds). Measured errors for these seeds run from
    0.05 to 1.9 standard errors.
    """
    result = monte_carlo_price(*ATM, kind, n_paths=n_paths, seed=12345)
    closed_form = price(*ATM, kind)
    error_in_standard_errors = abs(result.price - closed_form) / result.standard_error
    assert error_in_standard_errors < 4.0


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_averaging_many_seeds_converges_to_black_scholes(kind):
    """Unbiasedness, directly: average 200 independent runs.

    Each run has standard error ``se``; the average of 200 of them has
    standard error ``se/sqrt(200)``, so the mean should be very close indeed.
    """
    results = [monte_carlo_price(*ATM, kind, n_paths=20_000, seed=s)
               for s in range(200)]
    mean_price = sum(r.price for r in results) / len(results)
    standard_error_of_mean = results[0].standard_error / math.sqrt(len(results))
    closed_form = price(*ATM, kind)
    assert abs(mean_price - closed_form) < 4.0 * standard_error_of_mean


# ---------------------------------------------------------------------------
# 2. Is the reported uncertainty honest? (the central test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", [CALL, PUT])
def test_confidence_interval_covers_black_scholes_about_95_percent_of_the_time(kind):
    """The claim a 95% interval makes, tested by building 1,000 of them.

    This is the test that justifies reporting a confidence interval at all. It
    would catch a standard error that is too small (over-confident intervals,
    coverage well below 95%) or too large (useless intervals, coverage near
    100%) -- neither of which shows up in the price itself.

    Measured: 949/1000 for the call, 956/1000 for the put. With 1,000 trials
    the standard error of a coverage estimate is sqrt(0.95*0.05/1000) = 0.007,
    so anything in 0.93 to 0.97 is consistent with a correct interval, and
    both land inside.
    """
    closed_form = price(*ATM, kind)
    covered = sum(
        monte_carlo_price(*ATM, kind, n_paths=20_000, seed=s).contains(closed_form)
        for s in range(1_000)
    )
    coverage = covered / 1_000
    assert 0.93 <= coverage <= 0.97


def test_a_99_percent_interval_is_wider_and_covers_more_often():
    """The confidence level does what it says.

    Measured over 300 seeds: the 95% interval is 0.404 wide and covers 94.3%
    of the time, the 99% interval is 0.531 wide and covers 99.3%.
    """
    closed_form = call_price(*ATM)
    narrow = monte_carlo_price(*ATM, CALL, n_paths=20_000, seed=1, confidence=0.95)
    wide = monte_carlo_price(*ATM, CALL, n_paths=20_000, seed=1, confidence=0.99)
    assert wide.ci_width > narrow.ci_width
    assert wide.ci_width / narrow.ci_width == pytest.approx(2.5758 / 1.9600, rel=1e-3)

    coverage_99 = sum(
        monte_carlo_price(*ATM, CALL, n_paths=20_000, seed=s,
                          confidence=0.99).contains(closed_form)
        for s in range(300)
    ) / 300
    assert coverage_99 > 0.97


def test_rare_payoffs_make_the_interval_over_confident():
    """A documented limitation, not a bug -- and a real one worth knowing.

    The confidence interval relies on the central limit theorem: the *mean* of
    the payoffs is treated as normally distributed. For a deep out-of-the-money
    call (here worth 0.004, so almost every path pays nothing and a handful pay
    a lot) the payoff distribution is so skewed that 20,000 paths are not
    enough for that approximation to hold, and the intervals come out too
    narrow.

    Measured coverage is 0.873 at N=20,000 and 0.913 at N=200,000 -- clearly
    below the nominal 95%, and improving with N as the theorem predicts. The
    lesson for Phase 3/4: a reported standard error is only as good as the
    normal approximation behind it.
    """
    closed_form = call_price(*DEEP_OTM)
    coverage = {}
    for n_paths in (20_000, 200_000):
        covered = sum(
            monte_carlo_price(*DEEP_OTM, CALL, n_paths=n_paths,
                              seed=s).contains(closed_form)
            for s in range(300)
        )
        coverage[n_paths] = covered / 300

    assert coverage[20_000] < 0.93                      # over-confident
    assert coverage[200_000] > coverage[20_000]         # improving with N
    assert coverage[20_000] > 0.80                      # but not wildly broken


# ---------------------------------------------------------------------------
# 3. Does the error shrink like 1/sqrt(N)?
# ---------------------------------------------------------------------------

def test_standard_error_falls_like_one_over_root_n():
    """A hundred times the paths should divide the standard error by ten.

    Measured: 0.4187 at N=1,000, 0.04635 at N=100,000, 0.004653 at
    N=10,000,000 -- ratios of 9.03 and 9.96. (The first is slightly under 10
    because at N=1,000 the sample standard deviation is itself a noisy
    estimate.)
    """
    standard_errors = {
        n: monte_carlo_price(*ATM, CALL, n_paths=n, seed=7).standard_error
        for n in (1_000, 100_000, 10_000_000)
    }
    assert standard_errors[1_000] / standard_errors[100_000] == pytest.approx(10, rel=0.2)
    assert standard_errors[100_000] / standard_errors[10_000_000] == pytest.approx(10, rel=0.1)


def test_standard_error_times_root_n_is_roughly_constant():
    """The same statement without ratios: se * sqrt(N) estimates one number.

    That number is the standard deviation of a single discounted payoff, which
    does not depend on N at all -- which is exactly why the error can only
    improve as 1/sqrt(N).
    """
    scaled = [
        monte_carlo_price(*ATM, CALL, n_paths=n, seed=7).standard_error * math.sqrt(n)
        for n in (10_000, 100_000, 1_000_000, 10_000_000)
    ]
    assert max(scaled) / min(scaled) < 1.05


def test_more_paths_narrow_the_interval():
    widths = [monte_carlo_price(*ATM, CALL, n_paths=n, seed=3).ci_width
              for n in (1_000, 10_000, 100_000, 1_000_000)]
    assert all(b < a for a, b in zip(widths, widths[1:]))


# ---------------------------------------------------------------------------
# The structure of the estimator
# ---------------------------------------------------------------------------

def test_confidence_interval_is_the_price_plus_or_minus_z_times_se():
    result = monte_carlo_price(*ATM, CALL, n_paths=50_000, seed=5)
    assert result.ci_low == pytest.approx(result.price - Z_95 * result.standard_error)
    assert result.ci_high == pytest.approx(result.price + Z_95 * result.standard_error)
    assert result.ci_width == pytest.approx(2 * Z_95 * result.standard_error)
    assert result.contains(result.price)


def test_z_95_is_the_exact_critical_value():
    """1.959963..., not the 1.96 of the textbooks' rounded tables."""
    assert Z_95 == pytest.approx(1.959963984540054, abs=1e-12)


def test_same_seed_reproduces_exactly_and_different_seeds_do_not():
    """The project rule: every number must be reproducible."""
    first = monte_carlo_price(*ATM, CALL, n_paths=50_000, seed=99)
    again = monte_carlo_price(*ATM, CALL, n_paths=50_000, seed=99)
    other = monte_carlo_price(*ATM, CALL, n_paths=50_000, seed=100)
    assert first == again                    # frozen dataclass: compares field by field
    assert first.price != other.price


def test_result_records_what_produced_it():
    result = monte_carlo_price(*ATM, PUT, n_paths=1234, seed=777)
    assert result.n_paths == 1234
    assert result.seed == 777
    assert result.confidence == 0.95
    assert isinstance(result, MonteCarloResult)
    assert "seed=777" in str(result)


# ---------------------------------------------------------------------------
# Put-call parity, which the simulation satisfies path by path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_paths", [1_000, 100_000, 1_000_000])
def test_same_seed_call_minus_put_is_exactly_the_sampled_forward(n_paths):
    """Parity holds *within* the simulation, exactly, however few the paths.

    On every single path ``max(S_T-K, 0) - max(K-S_T, 0) = S_T - K``, with no
    approximation. So with the same seed (hence the same terminal prices)::

        MC_call - MC_put = exp(-r*T) * mean(S_T) - K*exp(-r*T)

    which this test checks to 2e-14. What it does *not* equal is the true
    ``S - K*exp(-r*T)``, because ``mean(S_T)`` is only an estimate of the
    forward: the gap is -0.626 at N=1,000, -0.069 at N=100,000 and +0.004 at
    N=1,000,000.

    That gap is the whole idea behind the control variate in Phase 4: the
    simulation's error in pricing the *stock* -- something whose true value we
    know exactly -- tells us about its error in pricing the option.
    """
    S, K, r, sigma, T = ATM
    seed = 42
    call = monte_carlo_price(*ATM, CALL, n_paths=n_paths, seed=seed).price
    put = monte_carlo_price(*ATM, PUT, n_paths=n_paths, seed=seed).price

    # Rebuild the same terminal prices the pricer used.
    normals = np.random.default_rng(seed).standard_normal(n_paths)
    terminal = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * normals)
    sampled_forward = math.exp(-r * T) * terminal.mean()

    assert call - put == pytest.approx(sampled_forward - K * math.exp(-r * T), abs=1e-12)


def test_parity_error_shrinks_with_more_paths():
    """And the gap to true parity closes like 1/sqrt(N), as it must."""
    S, K, r, _, T = ATM
    exact = S - K * math.exp(-r * T)
    errors = []
    for n_paths in (1_000, 100_000, 1_000_000):
        call = monte_carlo_price(*ATM, CALL, n_paths=n_paths, seed=42).price
        put = monte_carlo_price(*ATM, PUT, n_paths=n_paths, seed=42).price
        errors.append(abs((call - put) - exact))
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.01


# ---------------------------------------------------------------------------
# Sanity: the simulation agrees with the tree as well as with the closed form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", [CALL, PUT])
def test_simulation_agrees_with_the_binomial_tree(kind):
    """Three unrelated methods, one answer.

    Closed form, lattice and simulation share no code beyond the payoff, so
    agreement between them is about as independent a check as this project can
    make.
    """
    from pricing.binomial import european_binomial

    simulated = monte_carlo_price(*ATM, kind, n_paths=2_000_000, seed=2024)
    tree = european_binomial(*ATM, 2000, kind)
    assert simulated.contains(tree) or abs(simulated.price - tree) < 0.02


# ---------------------------------------------------------------------------
# Inputs the simulation cannot accept
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_paths", [0, 1, -5, 10.5])
def test_bad_path_count_raises(n_paths):
    """Fewer than two paths cannot produce a standard error."""
    with pytest.raises(ValueError, match="n_paths"):
        monte_carlo_price(*ATM, CALL, n_paths=n_paths, seed=1)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
def test_bad_confidence_raises(confidence):
    with pytest.raises(ValueError, match="confidence"):
        monte_carlo_price(*ATM, CALL, n_paths=100, seed=1, confidence=confidence)


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="call.*put"):
        monte_carlo_price(*ATM, "strangle", n_paths=100, seed=1)


def test_invalid_market_inputs_raise():
    with pytest.raises(ValueError, match="S"):
        monte_carlo_price(-100.0, 100.0, 0.05, 0.2, 1.0, CALL, n_paths=100, seed=1)
    with pytest.raises(ValueError, match="sigma"):
        monte_carlo_price(100.0, 100.0, 0.05, -0.2, 1.0, CALL, n_paths=100, seed=1)


def test_zero_volatility_gives_the_certain_payoff_with_no_error():
    """With sigma = 0 every path is identical, so the standard error is zero.

    The simulation still works -- it just has nothing to average over -- and it
    must agree with the closed-form degenerate case.
    """
    result = monte_carlo_price(110.0, 100.0, 0.05, 0.0, 1.0, CALL,
                               n_paths=1_000, seed=1)
    assert result.standard_error == pytest.approx(0.0, abs=1e-12)
    assert result.price == pytest.approx(110.0 - 100.0 * math.exp(-0.05), abs=1e-9)


def test_expiry_now_gives_the_payoff_exactly():
    result = monte_carlo_price(110.0, 100.0, 0.05, 0.2, 0.0, CALL,
                               n_paths=1_000, seed=1)
    assert result.price == pytest.approx(10.0, abs=1e-12)
    assert result.standard_error == pytest.approx(0.0, abs=1e-12)
