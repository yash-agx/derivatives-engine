"""Tests for the C++ Monte Carlo engine.

Being fast is not the claim that needs testing -- being *right* is. A rewritten
Monte Carlo core can be wrong in ways a spot-check of one price will not catch:
a weak random number generator, correlated streams across threads, or a
standard error computed over the wrong unit. So the C++ estimators are put
through exactly the same Phase 3 harness as the Python ones, and judged on the
same three things: unbiasedness, an honest standard error, and coverage.

The C++ generator is counter-based and completely different from NumPy's
PCG64, so these tests can never compare numbers path by path. They compare
distributions, which is the right level anyway.

Everything here skips cleanly if the DLL has not been built.
"""

import math

import pytest

from experiments.harness import Option, run_seed_study
from pricing.black_scholes import call_price, price
from pricing.cpp_backend import (
    CppUnavailable,
    cpp_available,
    cpp_monte_carlo,
    engine_version,
    hardware_threads,
    importance_shift,
    make_estimator,
)
from pricing.variance_reduction import importance_sampling_shift

pytestmark = pytest.mark.skipif(
    not cpp_available(),
    reason='cpp/build/mcengine.dll not built (cmake -S cpp -B cpp/build -G "MinGW Makefiles")',
)

ATM = (100.0, 100.0, 0.05, 0.20, 1.0)
ATM_CALL = Option("ATM call", S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, kind="call")
BROKEN = Option("K=160 6m", S=100.0, K=160.0, r=0.05, sigma=0.20, T=0.5, kind="call")
SEEDS = range(300)

METHOD_NAMES = ["plain", "antithetic", "control", "importance"]


# ---------------------------------------------------------------------------
# The engine loads and reports itself
# ---------------------------------------------------------------------------

def test_engine_loads():
    assert "mc_engine" in engine_version()
    assert hardware_threads() >= 1


# ---------------------------------------------------------------------------
# Statistical correctness, through the Phase 3 harness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHOD_NAMES)
def test_cpp_estimator_is_unbiased(method):
    """Averaged over 300 seeds, each C++ method lands on Black-Scholes.

    This is the test that would fail on a weak random number generator: a
    biased or short-period stream shows up as a mean that does not converge to
    the right answer, however small the reported error bar is.
    """
    study = run_seed_study(make_estimator(method), ATM_CALL, 20_000, SEEDS, method)
    standard_error_of_mean = study.spread / math.sqrt(study.n_seeds)
    assert abs(study.bias) < 4 * standard_error_of_mean


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_cpp_error_bar_is_honest(method):
    """Reported standard error against the real spread over 300 seeds.

    For the antithetic method this is also the check that the C++ side computed
    its standard error over *pairs*: getting that wrong would show up here as a
    ratio well away from 1, exactly as it would in Python.
    """
    study = run_seed_study(make_estimator(method), ATM_CALL, 20_000, SEEDS, method)
    assert study.se_ratio == pytest.approx(1.0, abs=0.15)
    assert 0.90 <= study.coverage <= 0.99


@pytest.mark.parametrize("kind", ["call", "put", "straddle"])
def test_cpp_prices_every_payoff_kind(kind):
    truth = price(*ATM, kind)
    result = cpp_monte_carlo(*ATM, kind, n_paths=2_000_000, seed=5)
    assert abs(result.price - truth) < 4 * result.standard_error


def test_cpp_agrees_with_the_python_implementation():
    """Two independent implementations with unrelated generators, one answer.

    Averaged over 200 seeds each, the C++ and Python plain estimators must
    agree to within the error of the comparison -- which is what makes this a
    real cross-check rather than a restatement.
    """
    from experiments.harness import plain_monte_carlo

    cpp = run_seed_study(make_estimator("plain"), ATM_CALL, 50_000, range(200), "cpp")
    python = run_seed_study(plain_monte_carlo, ATM_CALL, 50_000, range(200), "python")
    gap = abs(cpp.mean_price - python.mean_price)
    combined = math.sqrt(cpp.spread ** 2 + python.spread ** 2) / math.sqrt(200)
    assert gap < 4 * combined


def test_cpp_control_variate_satisfies_put_call_parity_exactly():
    """The same free identity the Python control variate has.

    Both legs remove the sampled forward's error exactly, so the difference of
    the two adjusted estimates is the exact parity value whatever the paths did.
    """
    S, K, r, _, T = ATM
    exact = S - K * math.exp(-r * T)
    for seed in (1, 2, 3):
        call = cpp_monte_carlo(*ATM, "call", n_paths=50_000, seed=seed,
                               method="control").price
        put = cpp_monte_carlo(*ATM, "put", n_paths=50_000, seed=seed,
                              method="control").price
        assert call - put == pytest.approx(exact, abs=1e-9)


def test_cpp_importance_sampling_restores_broken_coverage():
    """The Phase 4 rescue, reproduced by the C++ engine.

    Plain Monte Carlo covers about 87% of the time on this option at 20,000
    paths; importance sampling should put it back on 95%.
    """
    plain = run_seed_study(make_estimator("plain"), BROKEN, 20_000, SEEDS, "plain")
    shifted = run_seed_study(make_estimator("importance"), BROKEN, 20_000, SEEDS,
                             "importance")
    assert plain.coverage < 0.92
    assert shifted.coverage >= 0.93
    assert (plain.spread ** 2) / (shifted.spread ** 2) > 100.0


def test_cpp_shift_matches_the_python_one():
    assert importance_shift(*BROKEN.args) == pytest.approx(
        importance_sampling_shift(*BROKEN.args), rel=1e-12)


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_cpp_variance_falls_like_one_over_paths(method):
    """A broken generator would show up as an error that stops improving."""
    errors = [cpp_monte_carlo(*ATM, "call", n_paths=n, seed=3,
                              method=method).standard_error
              for n in (100_000, 1_000_000)]
    assert errors[0] / errors[1] == pytest.approx(math.sqrt(10), rel=0.1)


# ---------------------------------------------------------------------------
# Threads: independent streams, and reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHOD_NAMES)
def test_thread_count_does_not_change_the_answer_at_all(method):
    """Bitwise identical from 1 thread to 16, not merely statistically equal.

    Two things make this hold. The random numbers are a pure function of
    (seed, index), so a path's draws do not depend on which thread computes it;
    and the work is split into a fixed number of chunks whatever the thread
    count, so the floating-point additions happen in the same order every time.
    Without the second, the answer would wobble in the last bits as threads
    changed -- correct, but not reproducible.
    """
    results = [cpp_monte_carlo(*ATM, "call", n_paths=1_000_000, seed=11,
                               method=method, n_threads=threads)
               for threads in (1, 2, 3, 4, 8, 16)]
    for result in results[1:]:
        assert result.price == results[0].price
        assert result.standard_error == results[0].standard_error


def test_more_threads_is_not_more_paths():
    """A thread must not quietly add work: the path budget is the path budget."""
    single = cpp_monte_carlo(*ATM, "call", n_paths=500_000, seed=2, n_threads=1)
    many = cpp_monte_carlo(*ATM, "call", n_paths=500_000, seed=2, n_threads=8)
    assert single.n_paths == many.n_paths == 500_000
    assert single.price == many.price


def test_threads_do_not_share_a_stream():
    """If two threads drew the same numbers, the variance would collapse.

    Overlapping streams duplicate paths, which makes the estimates look more
    stable than they are: the reported standard error would stay put while the
    real spread across seeds grew. The honesty check above would catch that,
    and so does this more direct one -- the standard error on 16 threads is the
    same number as on 1, because it is literally the same computation.
    """
    one = cpp_monte_carlo(*ATM, "call", n_paths=2_000_000, seed=77, n_threads=1)
    many = cpp_monte_carlo(*ATM, "call", n_paths=2_000_000, seed=77, n_threads=16)
    assert one.standard_error == many.standard_error
    assert one.standard_error > 0


def test_different_seeds_give_different_streams():
    first = cpp_monte_carlo(*ATM, "call", n_paths=100_000, seed=1)
    second = cpp_monte_carlo(*ATM, "call", n_paths=100_000, seed=2)
    assert first.price != second.price


def test_chunk_count_is_part_of_the_reproducibility_contract():
    """Changing the chunking changes the summation order, by design.

    Holding the chunk count fixed is what makes the thread count irrelevant, so
    the chunk count itself has to be a stated input rather than something tuned
    per machine. Different chunking gives an equally valid answer that differs
    in the last bits.
    """
    a = cpp_monte_carlo(*ATM, "call", n_paths=1_000_000, seed=4, n_chunks=16)
    b = cpp_monte_carlo(*ATM, "call", n_paths=1_000_000, seed=4, n_chunks=64)
    assert a.price == pytest.approx(b.price, rel=1e-12)
    # Same chunking, any threads: exactly equal.
    c = cpp_monte_carlo(*ATM, "call", n_paths=1_000_000, seed=4, n_chunks=16,
                        n_threads=8)
    assert a.price == c.price


# ---------------------------------------------------------------------------
# Inputs the engine rejects
# ---------------------------------------------------------------------------

def test_antithetic_rejects_an_odd_path_count():
    with pytest.raises(ValueError, match="even"):
        cpp_monte_carlo(*ATM, "call", n_paths=1_001, seed=1, method="antithetic")


def test_importance_sampling_needs_randomness():
    with pytest.raises(ValueError, match="sigma > 0"):
        cpp_monte_carlo(100.0, 100.0, 0.05, 0.0, 1.0, "call", n_paths=1_000,
                        seed=1, method="importance")


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        cpp_monte_carlo(*ATM, "call", n_paths=1_000, seed=1, method="quasi")


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="call.*put"):
        cpp_monte_carlo(*ATM, "butterfly", n_paths=1_000, seed=1)


def test_bad_market_inputs_raise():
    with pytest.raises(ValueError, match="S"):
        cpp_monte_carlo(-100.0, 100.0, 0.05, 0.2, 1.0, "call", n_paths=1_000, seed=1)


def test_too_few_paths_raises():
    with pytest.raises(ValueError, match="n_paths"):
        cpp_monte_carlo(*ATM, "call", n_paths=1, seed=1)


def test_missing_library_error_names_the_build_command():
    """If the DLL is absent, the error has to say how to produce it.

    Simulated by pointing the loader at a path that does not exist, since on
    this machine the real one does.
    """
    from pricing import cpp_backend

    original_path, original_engine = cpp_backend._LIBRARY_PATH, cpp_backend._engine
    try:
        cpp_backend._LIBRARY_PATH = original_path.with_name("definitely_not_here.dll")
        cpp_backend._engine = None
        assert not cpp_backend.cpp_available()
        with pytest.raises(CppUnavailable, match="cmake"):
            cpp_backend.load_engine()
    finally:
        cpp_backend._LIBRARY_PATH, cpp_backend._engine = original_path, original_engine
