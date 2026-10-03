"""Variance-reduced Monte Carlo estimators.

Phase 3 established the problem: the error falls like ``s / sqrt(N)``, so one
extra decimal place costs a hundred times the paths. Fighting ``sqrt(N)`` is
hopeless. The alternative is to attack ``s`` -- to find a different random
variable with the *same mean* and a *smaller spread*, and average that instead.

Every estimator here does exactly that, and every one of them is still an
honest Monte Carlo: unbiased (or nearly so, with one documented exception),
with a standard error computed from whatever the independent unit of its
construction happens to be. Getting that unit right is the whole subtlety.

The four methods
----------------
**Antithetic variates.** For each draw ``Z``, also use ``-Z``. Both are valid
standard normal draws, so both legs price the option correctly; but one is high
exactly when the other is low. Average the pair first, then average the pairs.
If the payoff is a monotone function of ``Z`` the two legs are negatively
correlated and the pair average is much more stable than either leg.

*The unit of independence is the pair, not the path.* Two paths within a pair
are deliberately dependent, so computing the standard error over individual
paths would be using a formula whose assumption has just been broken on
purpose. This matters: it would understate the error for a call, and overstate
it for a straddle.

**Control variates.** Suppose there is a second quantity ``X`` whose true mean
you already know exactly, and which moves with the thing you care about. Then::

    Y_adjusted = Y - c * (X - E[X])

has the same mean as ``Y`` for any ``c``, because ``E[X - E[X]] = 0``. But its
variance is ``Var(Y) - 2c*Cov(X,Y) + c^2*Var(X)``, minimised at::

    c* = Cov(X, Y) / Var(X)

giving ``Var(Y) * (1 - rho^2)``. The better ``X`` correlates with ``Y``, the
more variance disappears. Here ``X`` is the discounted terminal stock price,
whose mean is known exactly to be today's spot ``S`` -- that is the
risk-neutral property the whole model is built on. When the simulation prices
the *stock* too high, it has probably priced the *option* too high as well,
and the control subtracts off that shared error.

**Antithetic and control together.** Pair first, then control on the pair
averages. The two techniques attack different things, so they compose.

**Importance sampling.** The others reduce spread; this one fixes a different
failure. Phase 3 found that a deep out-of-the-money option breaks Monte Carlo
not because the spread is large but because almost no path pays anything --
below about 50 paying paths per run even the error bar becomes untrustworthy.
No amount of averaging fixes a sample with nothing in it.

So change where the paths go. Simulate under a drift that puts the strike in
the middle of the distribution, then undo the lie by weighting each path with
the likelihood ratio between the true distribution and the sampled one::

    Z ~ Normal(mu, 1)  instead of  Normal(0, 1)
    weight = exp(-mu*Z + mu^2 / 2)

The weighted average is still an unbiased estimate of the true price -- the
weights are exactly the ratio of the two densities -- but now half the paths
land in the money instead of one in a thousand.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .common import check_payoff_kind, payoff, validate_inputs
from .monte_carlo import MonteCarloResult

__all__ = [
    "antithetic_monte_carlo",
    "control_variate_monte_carlo",
    "antithetic_control_monte_carlo",
    "importance_sampling_monte_carlo",
    "optimal_control_coefficient",
    "importance_sampling_shift",
]


def _check_inputs(S, K, r, sigma, T, kind, n_paths, confidence, minimum=2):
    validate_inputs(S, K, r, sigma, T)
    check_payoff_kind(kind)
    if int(n_paths) != n_paths or n_paths < minimum:
        raise ValueError(
            f"n_paths must be a whole number >= {minimum}, got {n_paths!r}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def _result(values, n_paths, seed, confidence, price=None) -> MonteCarloResult:
    """Package a sample of independent units into a result with its error bar.

    ``values`` must be the *independent* units of the estimator -- pair
    averages for an antithetic run, adjusted payoffs for a control-variate one.
    ``n_paths`` is the number of path evaluations spent, which is what makes
    methods comparable at equal cost.
    """
    count = len(values)
    estimate = float(np.mean(values)) if price is None else float(price)
    standard_error = float(np.std(values, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
    z = float(norm.ppf(0.5 + 0.5 * confidence))
    half_width = z * standard_error
    return MonteCarloResult(
        price=estimate,
        standard_error=standard_error,
        ci_low=estimate - half_width,
        ci_high=estimate + half_width,
        n_paths=int(n_paths),
        seed=int(seed),
        confidence=confidence,
    )


def _terminal(S, r, sigma, T, normals):
    """Terminal prices from a vector of standard normal draws."""
    return S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * normals)


# ---------------------------------------------------------------------------
# Antithetic variates
# ---------------------------------------------------------------------------

def antithetic_monte_carlo(
    S: float, K: float, r: float, sigma: float, T: float, kind: str,
    n_paths: int = 100_000, seed: int = 12345, confidence: float = 0.95,
) -> MonteCarloResult:
    """Price using ``n_paths/2`` antithetic pairs.

    Spends the same number of path evaluations as plain Monte Carlo with
    ``n_paths``, so the two can be compared directly -- though it needs only
    half as many random numbers, which is a small speed gain on its own.

    The standard error comes from the spread of the **pair averages**. See the
    module docstring: the pair is the independent unit, and using the paths
    instead would be applying a formula whose independence assumption this
    method deliberately breaks.
    """
    _check_inputs(S, K, r, sigma, T, kind, n_paths, confidence, minimum=4)
    if n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even for antithetic pairs, got {n_paths}")

    n_pairs = int(n_paths) // 2
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_pairs)

    discount = math.exp(-r * T)
    up = payoff(_terminal(S, r, sigma, T, normals), K, kind)
    down = payoff(_terminal(S, r, sigma, T, -normals), K, kind)
    pair_means = 0.5 * discount * (up + down)

    return _result(pair_means, n_paths, seed, confidence)


# ---------------------------------------------------------------------------
# Control variates
# ---------------------------------------------------------------------------

def optimal_control_coefficient(target, control) -> float:
    """``c* = Cov(target, control) / Var(control)``, the variance-minimising mix."""
    control_variance = float(np.var(control, ddof=1))
    if not control_variance > 0:
        return 0.0
    covariance = float(np.cov(target, control, ddof=1)[0, 1])
    return covariance / control_variance


def control_variate_monte_carlo(
    S: float, K: float, r: float, sigma: float, T: float, kind: str,
    n_paths: int = 100_000, seed: int = 12345, confidence: float = 0.95,
    pilot_fraction: float = 0.0,
) -> MonteCarloResult:
    """Price using the discounted terminal stock price as a control.

    ``E[exp(-rT) * S_T] = S`` exactly under the risk-neutral measure, so the
    control's error is directly observable on every run: whatever the
    simulation gets wrong about the stock, it has probably got wrong about the
    option in the same direction.

    ``pilot_fraction`` splits the paths, estimating ``c`` on the first slice and
    pricing on the rest. With the default of 0 the same paths do both, which
    makes ``c`` dependent on the data and introduces a small bias of order
    ``1/N``. The experiment measures how large that actually is; the split is
    there so the two can be compared honestly at the same total path count.
    """
    _check_inputs(S, K, r, sigma, T, kind, n_paths, confidence, minimum=4)
    if not 0.0 <= pilot_fraction < 1.0:
        raise ValueError(f"pilot_fraction must be in [0, 1), got {pilot_fraction}")

    n_paths = int(n_paths)
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_paths)

    discount = math.exp(-r * T)
    terminal = _terminal(S, r, sigma, T, normals)
    target = discount * payoff(terminal, K, kind)
    control = discount * terminal              # known mean: exactly S

    if pilot_fraction > 0.0:
        n_pilot = max(2, int(round(n_paths * pilot_fraction)))
        n_pilot = min(n_pilot, n_paths - 2)
        coefficient = optimal_control_coefficient(target[:n_pilot], control[:n_pilot])
        target, control = target[n_pilot:], control[n_pilot:]
    else:
        coefficient = optimal_control_coefficient(target, control)

    adjusted = target - coefficient * (control - S)
    return _result(adjusted, n_paths, seed, confidence)


# ---------------------------------------------------------------------------
# Both together
# ---------------------------------------------------------------------------

def antithetic_control_monte_carlo(
    S: float, K: float, r: float, sigma: float, T: float, kind: str,
    n_paths: int = 100_000, seed: int = 12345, confidence: float = 0.95,
    pilot_fraction: float = 0.0,
) -> MonteCarloResult:
    """Antithetic pairing first, then a control variate on the pair averages.

    The pair is still the independent unit, so the control coefficient and the
    standard error are both computed over pairs.
    """
    _check_inputs(S, K, r, sigma, T, kind, n_paths, confidence, minimum=4)
    if n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even for antithetic pairs, got {n_paths}")
    if not 0.0 <= pilot_fraction < 1.0:
        raise ValueError(f"pilot_fraction must be in [0, 1), got {pilot_fraction}")

    n_pairs = int(n_paths) // 2
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_pairs)

    discount = math.exp(-r * T)
    up_terminal = _terminal(S, r, sigma, T, normals)
    down_terminal = _terminal(S, r, sigma, T, -normals)

    target = 0.5 * discount * (payoff(up_terminal, K, kind)
                               + payoff(down_terminal, K, kind))
    control = 0.5 * discount * (up_terminal + down_terminal)

    if pilot_fraction > 0.0:
        n_pilot = max(2, int(round(n_pairs * pilot_fraction)))
        n_pilot = min(n_pilot, n_pairs - 2)
        coefficient = optimal_control_coefficient(target[:n_pilot], control[:n_pilot])
        target, control = target[n_pilot:], control[n_pilot:]
    else:
        coefficient = optimal_control_coefficient(target, control)

    adjusted = target - coefficient * (control - S)
    return _result(adjusted, n_paths, seed, confidence)


# ---------------------------------------------------------------------------
# Importance sampling
# ---------------------------------------------------------------------------

def importance_sampling_shift(S: float, K: float, r: float, sigma: float,
                              T: float) -> float:
    """The drift shift that lands the *median* path exactly on the strike.

    Solving ``S * exp((r - sigma^2/2)T + sigma*sqrt(T)*mu) = K`` gives::

        mu = (ln(K/S) - (r - sigma^2/2)T) / (sigma*sqrt(T))  =  -d2

    which is a pleasing way to meet ``d2`` again: the number of standard
    deviations the option is in the money by is exactly the number of standard
    deviations the sampling has to be shifted to put it at the money.

    Centring on the strike is the standard textbook choice and gets roughly
    half the paths paying. It is not quite variance-optimal -- shifting a
    little further out is better still -- but it is the one that can be
    explained in a sentence.
    """
    return -(math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def importance_sampling_monte_carlo(
    S: float, K: float, r: float, sigma: float, T: float, kind: str,
    n_paths: int = 100_000, seed: int = 12345, confidence: float = 0.95,
    shift: float | None = None,
) -> MonteCarloResult:
    """Price by sampling where the payoff is, and reweighting to stay honest.

    ``shift`` defaults to ``importance_sampling_shift`` (the strike at the
    median). Passing ``0.0`` recovers ordinary Monte Carlo exactly, which is a
    useful thing to be able to assert in a test.
    """
    _check_inputs(S, K, r, sigma, T, kind, n_paths, confidence)
    if sigma <= 0 or T <= 0:
        raise ValueError(
            "importance sampling needs sigma > 0 and T > 0: with no randomness "
            "there is no distribution to shift"
        )

    mu = importance_sampling_shift(S, K, r, sigma, T) if shift is None else float(shift)
    n_paths = int(n_paths)
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_paths) + mu

    discount = math.exp(-r * T)
    terminal = _terminal(S, r, sigma, T, normals)
    # Likelihood ratio: density of the true N(0,1) over the sampled N(mu,1).
    weights = np.exp(-mu * normals + 0.5 * mu * mu)
    weighted = discount * payoff(terminal, K, kind) * weights

    return _result(weighted, n_paths, seed, confidence)
