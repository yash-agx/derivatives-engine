"""A reusable harness for measuring how well a Monte Carlo estimator works.

The point of the harness
------------------------
A single Monte Carlo run tells you almost nothing about the *method*. It gives
one price and one self-reported error bar, and both are random. To say anything
about a method you have to run it many times, from independent seeds, and look
at the distribution of what comes back.

That is what this module does. Give it an estimator and an option, and it runs
the estimator ``n_seeds`` times at each path count and reports:

* **bias** -- is the method aimed at the right answer?
* **RMSE** -- how far off is a typical run, measured against the truth rather
  than against the method's own opinion of itself?
* **spread** -- the actual standard deviation of the estimates across seeds.
* **mean reported standard error**, and the ratio of the two. This is the
  honesty check: a method that reports an error bar of 0.10 while its estimates
  actually scatter by 0.40 is lying, and no amount of looking at prices alone
  would reveal it.
* **coverage** -- how often the 95% interval really contains the true price.
* **runtime** -- the median over the repeated runs.
* **efficiency** -- variance times runtime. This is a *cost*: the price of a
  unit of accuracy, so lower is better and a rising number means worse.

Why it is built around an estimator *function*
-----------------------------------------------
Phase 4 adds antithetic variates and control variates. Those are different
estimators of the same quantity, and the only fair way to judge them is to run
them through exactly the same measurement code. So the harness never calls the
pricer directly: it takes any callable with the signature::

    estimator(option, n_paths, seed) -> MonteCarloResult

``plain_monte_carlo`` below is the Phase 3 instance. Phase 4 adds more, changes
nothing here, and the comparison is automatically like for like.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import kurtosis, norm, skew

from pricing.black_scholes import price as black_scholes_price
from pricing.common import CALL, check_payoff_kind
from pricing.monte_carlo import (
    MonteCarloResult,
    monte_carlo_price,
    simulate_discounted_payoffs,
)

__all__ = [
    "Option",
    "Estimator",
    "plain_monte_carlo",
    "SeedStudy",
    "PayoffShape",
    "run_seed_study",
    "run_grid",
    "measure_payoff_shape",
    "clt_diagnostic",
    "paths_for_target",
]


@dataclass(frozen=True)
class Option:
    """One option contract, plus a short label for tables and charts."""

    label: str
    S: float
    K: float
    r: float
    sigma: float
    T: float
    kind: str = CALL

    def __post_init__(self) -> None:
        # The wider set: an Option only ever needs payoff() and price(), both of
        # which handle a straddle. The Greeks keep the strict check.
        check_payoff_kind(self.kind)

    @property
    def args(self) -> tuple:
        """``(S, K, r, sigma, T)`` -- the argument order every pricer uses."""
        return (self.S, self.K, self.r, self.sigma, self.T)

    @property
    def moneyness(self) -> float:
        """Spot over strike. Above 1 a call is in the money."""
        return self.S / self.K

    def black_scholes(self) -> float:
        """The truth every measurement in this module is taken against."""
        return black_scholes_price(*self.args, self.kind)


# An estimator is anything that prices an option from a path count and a seed.
Estimator = Callable[[Option, int, int], MonteCarloResult]


def plain_monte_carlo(option: Option, n_paths: int, seed: int) -> MonteCarloResult:
    """The Phase 3 estimator: ordinary Monte Carlo, no variance reduction."""
    return monte_carlo_price(*option.args, option.kind, n_paths=n_paths, seed=seed)


@dataclass(frozen=True)
class SeedStudy:
    """What one estimator did on one option at one path count, over many seeds.

    The raw per-seed numbers are kept so that anything not thought of yet can
    still be computed later; the properties below are the summary.
    """

    estimator: str
    option: Option
    n_paths: int
    truth: float
    prices: tuple
    standard_errors: tuple
    covered: tuple
    runtimes: tuple
    seeds: tuple

    # -- accuracy, measured against the truth -------------------------------

    @property
    def n_seeds(self) -> int:
        return len(self.prices)

    @property
    def mean_price(self) -> float:
        return float(np.mean(self.prices))

    @property
    def bias(self) -> float:
        """Average error. Should be ~0: Monte Carlo is an unbiased estimator."""
        return self.mean_price - self.truth

    @property
    def rmse(self) -> float:
        """Root mean squared error against Black-Scholes.

        This is the honest measure of accuracy, and it is *not* the same thing
        as the standard error the estimator reports about itself.
        """
        errors = np.asarray(self.prices) - self.truth
        return float(np.sqrt(np.mean(errors ** 2)))

    @property
    def spread(self) -> float:
        """Standard deviation of the estimates across seeds: the real scatter."""
        if self.n_seeds < 2:
            return float("nan")
        return float(np.std(self.prices, ddof=1))

    # -- honesty: does the reported error bar match reality? ----------------

    @property
    def mean_reported_se(self) -> float:
        return float(np.mean(self.standard_errors))

    @property
    def se_ratio(self) -> float:
        """Reported standard error divided by the actual spread.

        1.0 means the estimator's self-assessment is right. Below 1.0 it is
        over-confident: the error bars are too small for how much the estimates
        actually move.
        """
        spread = self.spread
        if not spread > 0:
            return float("nan")
        return self.mean_reported_se / spread

    @property
    def coverage(self) -> float:
        """Fraction of 95% intervals that contained the true price."""
        return float(np.mean(self.covered))

    @property
    def payoff_std(self) -> float:
        """The per-path payoff standard deviation ``s``, backed out of the run.

        Since ``standard error = s / sqrt(N)``, this is ``se * sqrt(N)``. It is
        the quantity that does not depend on N at all, and the one variance
        reduction attacks.
        """
        return self.mean_reported_se * math.sqrt(self.n_paths)

    # -- cost ---------------------------------------------------------------

    @property
    def median_runtime(self) -> float:
        """Median of the per-run timings, in seconds."""
        return float(statistics.median(self.runtimes))

    @property
    def total_runtime(self) -> float:
        return float(sum(self.runtimes))

    @property
    def runtime_per_path(self) -> float:
        """Seconds per path. Not constant -- see ``variance_times_runtime``."""
        return self.median_runtime / self.n_paths

    @property
    def variance_times_runtime(self) -> float:
        """Efficiency: estimator variance times the time one run costs.

        This is a **cost** -- what a unit of accuracy is worth paying for -- so
        **lower is better and a rising number means the method got worse**.

        In theory it is *constant in N*: doubling the paths halves the variance
        and doubles the time. That would make it a property of the method
        rather than of the budget, and so the right yardstick for comparing
        Phase 4's estimators against this one.

        Measured here, only half of that holds. The variance does fall exactly
        like 1/N (2.06x, 2.01x, 1.90x, 2.03x per doubling from N=50,000 to
        800,000). The runtime does not rise like N: 1.95x, 3.23x, 2.81x, 2.01x
        per doubling over the same range, because somewhere past 100,000 paths
        the arrays stop fitting in cache and each path starts costing more. So
        the cost **rises** from 3.2e-6 at N=100,000 to 7.7e-6 at N=400,000
        before flattening off around 7.6e-6: a large batch buys less accuracy
        per second than a medium one, and the cheapest accuracy on this machine
        is near N=100,000.

        The practical consequence, and it matters for Phase 4: compare
        estimators **at the same path count**. Comparing across path counts
        measures the memory system as much as the method.
        """
        spread = self.spread
        return float(spread * spread * self.median_runtime)

    def paths_for_target(self, target: float, confidence: float = 0.95) -> float:
        """Paths needed for a 95% error bar of at most ``target``.

        From ``half width = z * s / sqrt(N)``::

            N = (z * s / target)**2

        Quadratic in ``1/target``: one more decimal place costs a hundred times
        the paths. Pass ``confidence=None`` for a plain standard-error target
        instead of a confidence-interval one.
        """
        return paths_for_target(self.payoff_std, target, confidence)

    def row(self) -> dict:
        """Flat dict of everything, for writing to CSV."""
        return {
            "estimator": self.estimator,
            "option": self.option.label,
            "kind": self.option.kind,
            "S": self.option.S,
            "K": self.option.K,
            "r": self.option.r,
            "sigma": self.option.sigma,
            "T": f"{self.option.T:.6f}",
            "moneyness": f"{self.option.moneyness:.4f}",
            "n_paths": self.n_paths,
            "n_seeds": self.n_seeds,
            "bs_price": f"{self.truth:.10f}",
            "mean_price": f"{self.mean_price:.10f}",
            "bias": f"{self.bias:.10e}",
            "rmse": f"{self.rmse:.10e}",
            "spread": f"{self.spread:.10e}",
            "mean_reported_se": f"{self.mean_reported_se:.10e}",
            "se_ratio": f"{self.se_ratio:.6f}",
            "coverage": f"{self.coverage:.4f}",
            "payoff_std": f"{self.payoff_std:.8f}",
            "median_runtime_s": f"{self.median_runtime:.8f}",
            "runtime_per_path_ns": f"{self.runtime_per_path * 1e9:.4f}",
            "total_runtime_s": f"{self.total_runtime:.4f}",
            "variance_times_runtime": f"{self.variance_times_runtime:.6e}",
            "paths_for_1_cent": f"{self.paths_for_target(0.01):.0f}",
            "paths_for_0.1_cent": f"{self.paths_for_target(0.001):.0f}",
        }


def paths_for_target(payoff_std: float, target: float,
                     confidence: float | None = 0.95) -> float:
    """Paths needed to reach ``target`` accuracy, given a per-path spread.

    With ``confidence``, ``target`` is the half-width of that confidence
    interval; with ``confidence=None`` it is the standard error itself.
    """
    if not target > 0:
        raise ValueError(f"target must be > 0, got {target}")
    z = 1.0 if confidence is None else float(norm.ppf(0.5 + 0.5 * confidence))
    return (z * payoff_std / target) ** 2


def run_seed_study(
    estimator: Estimator,
    option: Option,
    n_paths: int,
    seeds: Sequence[int],
    estimator_name: str = "",
) -> SeedStudy:
    """Run ``estimator`` once per seed and collect everything it did.

    Each run is timed individually, so the reported runtime is a median over
    ``len(seeds)`` repeats rather than a single stopwatch reading.
    """
    if len(seeds) < 1:
        raise ValueError("need at least one seed")
    truth = option.black_scholes()

    prices, standard_errors, covered, runtimes = [], [], [], []
    for seed in seeds:
        started = time.perf_counter()
        result = estimator(option, n_paths, seed)
        runtimes.append(time.perf_counter() - started)
        prices.append(result.price)
        standard_errors.append(result.standard_error)
        covered.append(result.contains(truth))

    return SeedStudy(
        estimator=estimator_name or getattr(estimator, "__name__", "estimator"),
        option=option,
        n_paths=int(n_paths),
        truth=truth,
        prices=tuple(prices),
        standard_errors=tuple(standard_errors),
        covered=tuple(covered),
        runtimes=tuple(runtimes),
        seeds=tuple(seeds),
    )


def run_grid(
    estimators: dict,
    options: Iterable[Option],
    path_counts: Iterable[int],
    seeds: Sequence[int],
    progress: Callable[[str], None] | None = None,
) -> list:
    """Every estimator, on every option, at every path count.

    ``estimators`` maps a display name to an estimator function. Returns a flat
    list of ``SeedStudy``. The same seeds are used everywhere, so differences
    between rows are differences between methods and settings, not luck.
    """
    options = list(options)
    path_counts = list(path_counts)
    studies = []
    for name, estimator in estimators.items():
        for option in options:
            for n_paths in path_counts:
                study = run_seed_study(estimator, option, n_paths, seeds, name)
                studies.append(study)
                if progress is not None:
                    progress(
                        f"  {name:>14}  {option.label:<22} N={n_paths:>9,}  "
                        f"rmse {study.rmse:.5f}  coverage {study.coverage:.3f}  "
                        f"{study.total_runtime:.1f}s"
                    )
    return studies


@dataclass(frozen=True)
class PayoffShape:
    """The shape of the payoff distribution a single path draws from.

    The confidence interval assumes the *mean* of these draws is normally
    distributed. The central limit theorem promises that eventually, but how
    many paths "eventually" means depends on how lopsided this distribution is
    -- which is what these numbers measure.
    """

    option: Option
    n_paths: int
    seed: int
    mean: float
    std: float
    skewness: float
    excess_kurtosis: float
    zero_fraction: float

    def paths_for_clt(self, threshold: float = 0.1) -> float:
        """Paths needed before the normal approximation should be trusted.

        The Berry-Esseen theorem bounds how far the mean's distribution can be
        from normal by roughly ``|skewness| / sqrt(N)``. Requiring that to stay
        under ``threshold`` gives ``N > (skewness / threshold)**2``. It is a
        rule of thumb, not a guarantee, but it gets the order of magnitude
        right -- and this project can check it directly against measured
        coverage.
        """
        return (self.skewness / threshold) ** 2

    def row(self) -> dict:
        return {
            "option": self.option.label,
            "kind": self.option.kind,
            "S": self.option.S,
            "K": self.option.K,
            "sigma": self.option.sigma,
            "T": f"{self.option.T:.6f}",
            "moneyness": f"{self.option.moneyness:.4f}",
            "bs_price": f"{self.option.black_scholes():.10f}",
            "sample_paths": self.n_paths,
            "payoff_mean": f"{self.mean:.8f}",
            "payoff_std": f"{self.std:.8f}",
            "skewness": f"{self.skewness:.6f}",
            "excess_kurtosis": f"{self.excess_kurtosis:.6f}",
            "zero_fraction": f"{self.zero_fraction:.6f}",
            "paths_for_clt": f"{self.paths_for_clt():.0f}",
        }


def measure_payoff_shape(option: Option, n_paths: int = 2_000_000,
                         seed: int = 20_260_101) -> PayoffShape:
    """Measure how lopsided one option's payoff distribution is.

    Uses a large single sample rather than the study runs, because these are
    properties of the option itself and should be measured once, accurately.
    """
    payoffs = simulate_discounted_payoffs(
        *option.args, option.kind, n_paths=n_paths, seed=seed
    )
    std = float(payoffs.std(ddof=1))
    return PayoffShape(
        option=option,
        n_paths=int(n_paths),
        seed=int(seed),
        mean=float(payoffs.mean()),
        std=std,
        # Degenerate samples (every payoff identical) have no shape to report.
        skewness=float(skew(payoffs)) if std > 0 else float("nan"),
        excess_kurtosis=float(kurtosis(payoffs)) if std > 0 else float("nan"),
        zero_fraction=float(np.mean(payoffs == 0.0)),
    )


def clt_diagnostic(skewness: float, n_paths: int) -> float:
    """``|skewness| / sqrt(N)`` -- how far from normal the sample mean still is.

    This is the natural x-axis for asking when confidence intervals can be
    trusted: it combines the two things that matter (how skewed the payoff is,
    and how many paths are averaging it away) into one number, so cases that
    look unrelated should fall on a single curve.
    """
    return abs(skewness) / math.sqrt(n_paths)
