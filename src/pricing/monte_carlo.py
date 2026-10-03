"""Monte Carlo pricing of European options under geometric Brownian motion.

The idea
--------
Black-Scholes says the option is worth the discounted risk-neutral *average*
of its payoff. The closed form does that average with calculus. Monte Carlo
does it the brute-force way: simulate a large number of possible final stock
prices, work out the payoff in each one, average them, discount. The law of
large numbers says that average converges to the true one.

Why no time-stepping is needed
------------------------------
Geometric Brownian motion has an exact solution, so the terminal price can be
drawn in a single step rather than walked to::

    S_T = S * exp( (r - sigma**2 / 2) * T  +  sigma * sqrt(T) * Z ),   Z ~ N(0, 1)

Reading it: ``sigma*sqrt(T)*Z`` is the random part, one standard deviation of
the log return scaled by a standard normal draw. The ``(r - sigma**2/2)*T``
drift is the deterministic part, and the ``-sigma**2/2`` is there so that the
*average* of ``S_T`` comes out at the forward ``S*exp(r*T)`` -- without it,
the exponential of a zero-mean normal would average to more than one and the
simulation would drift upwards. (That term is the same Ito correction that
appears in Black-Scholes' ``d1``.)

A European payoff depends only on ``S_T``, so one normal draw per path is
exact -- there is no discretisation error at all, unlike the binomial tree.
The only error is statistical. Path-dependent options (barriers, Asians) would
need the full path, and that is where time-stepping and its own discretisation
error come in; this project does not price those.

Why the error shrinks like 1/sqrt(N)
-----------------------------------
Each path gives one payoff draw, from a distribution with some standard
deviation ``s``. We report the mean of ``N`` such draws, and the standard
deviation of a mean of ``N`` independent draws is::

    standard error = s / sqrt(N)

That is the whole story, and it is a property of averaging, not of options:
to halve the error you need four times the paths, and to add one decimal
place (ten times the accuracy) you need a hundred times the work. A
hundredfold cost for one digit is why variance reduction (Phase 4) is worth
the effort -- it attacks ``s`` instead of fighting ``sqrt(N)``.

The standard error is also what makes a Monte Carlo price *honest*: it is an
estimate, and quoting it without an error bar hides the fact that a different
seed would have given a different number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .common import check_kind, payoff, validate_inputs

__all__ = ["MonteCarloResult", "monte_carlo_price", "Z_95"]

# Two-sided 95% normal critical value: 1.959963... , not the rounded 1.96.
# P(-Z_95 < Z < Z_95) = 0.95 for a standard normal Z.
Z_95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class MonteCarloResult:
    """What a Monte Carlo run produced, with its uncertainty attached.

    ``price`` alone is not an answer -- a different seed gives a different
    number -- so the estimator always travels with its standard error and
    confidence interval.
    """

    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    seed: int
    confidence: float = 0.95

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval: the practical measure of accuracy."""
        return self.ci_high - self.ci_low

    def contains(self, value: float) -> bool:
        """Does the confidence interval cover ``value``?

        Used by the coverage test: if the interval is built correctly, it
        should contain the true Black-Scholes price about 95% of the time.
        """
        return self.ci_low <= value <= self.ci_high

    def __str__(self) -> str:
        return (
            f"{self.price:.4f} +/- {self.standard_error:.4f} (se), "
            f"95% CI [{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"N={self.n_paths:,}, seed={self.seed}"
        )


def monte_carlo_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    kind: str,
    n_paths: int = 100_000,
    seed: int = 12345,
    confidence: float = 0.95,
) -> MonteCarloResult:
    """Price a European call or put by simulating terminal stock prices.

    ``seed`` is required (with a default) rather than optional-and-random:
    every number in this project has to be reproducible, so a run that cannot
    be repeated is not useful. The generator is NumPy's PCG64 via
    ``default_rng``, so the same seed gives the same paths on any machine.

    The returned standard error uses the *sample* standard deviation of the
    discounted payoffs with ``ddof=1``, which is the unbiased estimator -- the
    textbook ``s / sqrt(N)``.
    """
    validate_inputs(S, K, r, sigma, T)
    check_kind(kind)
    if int(n_paths) != n_paths or n_paths < 2:
        raise ValueError(
            f"n_paths must be a whole number >= 2 (a standard error needs at "
            f"least two draws), got {n_paths!r}"
        )
    n_paths = int(n_paths)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_paths)

    # The exact GBM solution: one draw per path, no time-stepping.
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T) * normals
    terminal_spots = S * np.exp(drift + diffusion)

    discounted_payoffs = math.exp(-r * T) * payoff(terminal_spots, K, kind)

    price = float(discounted_payoffs.mean())
    # ddof=1: the unbiased sample standard deviation, because the mean we are
    # measuring spread around was itself estimated from the same data.
    standard_error = float(discounted_payoffs.std(ddof=1) / math.sqrt(n_paths))

    z = float(norm.ppf(0.5 + 0.5 * confidence))
    half_width = z * standard_error

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        ci_low=price - half_width,
        ci_high=price + half_width,
        n_paths=n_paths,
        seed=seed,
        confidence=confidence,
    )
