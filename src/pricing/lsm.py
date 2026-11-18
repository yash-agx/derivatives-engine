"""American options by Monte Carlo: Longstaff-Schwartz least-squares (LSM).

Why American options are hard to simulate
-----------------------------------------
A European option is an *average*: simulate final prices, average the payoff,
discount (``monte_carlo.py``). An American option is a *decision*: at every
moment the holder asks "is exercising now worth more than holding on?", and
the answer depends on what holding on is worth -- the **continuation value**.
That value is itself an expectation over the future, conditional on today's
price. The tree gets it for free because every node knows its two children.
A simulated path does not: it has exactly one future, so it cannot tell you
the *average* future from its own position.

The Longstaff-Schwartz idea
---------------------------
Take many paths at once. At a given date, every path that is in the money
gives one pair ``(S_t, realised discounted cash flow from holding on)``. The
realised cash flow is a noisy, unbiased sample of the continuation value at
that ``S_t``. So **regress** the cash flows on functions of ``S_t``: the
fitted curve is an estimate of the conditional expectation::

    continuation(S) ~ beta_0 * L_0(S/K) + beta_1 * L_1(S/K) + ... + beta_d * L_d(S/K)

Least squares is exactly the tool for this: the best predictor of ``Y`` from
``X`` in the mean-square sense *is* ``E[Y | X]``, and least squares finds the
best one inside the span of the chosen functions.

Then walk backwards from expiry, one exercise date at a time:

1. at expiry the cash flow is the payoff;
2. at each earlier date, discount every path's cash flow back one date, fit
   the regression on the in-the-money paths, and on any path where the
   immediate payoff beats the fitted continuation value, replace the path's
   cash flow with the payoff (the holder exercises there, so the later cash
   flow never happens);
3. at time 0, average and discount: that average is the value of holding,
   and the holder exercises immediately instead if the payoff now beats it.

Three design choices, and why
-----------------------------
* **Only in-the-money paths go into the regression.** Out of the money the
  decision is trivial (exercising pays nothing), so those paths say nothing
  about the decision that matters, and including them spends the fit's
  flexibility on a region where it is never used.
* **The regression target is the realised cash flow, not the fitted value
  from the next date.** Fitted values carry the fitting error forward and
  compound it; realised cash flows keep each date's estimate unbiased given
  the rule used after it. This is the main difference from earlier
  regression methods.
* **Laguerre polynomials in ``x = S/K``.** Any polynomials of degree ``d``
  span the same functions, so the fitted curve is the same in exact
  arithmetic. Laguerre polynomials are the original paper's choice, and
  dividing by ``K`` keeps ``x`` near 1 so the columns are well scaled. The
  weighted form the paper uses (each ``L_n`` times ``exp(-x/2)``) is
  available as ``weighted=True``.

The bias, and the two estimates
-------------------------------
Using the *same* paths to fit the rule and to price with it is a mild form of
looking into the future: the regression has seen these paths' futures, so the
rule is slightly tuned to their particular noise and the price comes out a
little high. That is the **in-sample** estimate.

Fitting the rule on one set of paths and pricing on a second, independent set
removes the foresight. What is left is a *fixed*, legitimate exercise
strategy. No strategy beats the optimal one, so its expected value can only be
at or below the true price: the **out-of-sample** estimate is **low-biased** --
a lower bound, up to its own sampling noise. Because the new paths are
independent of the rule, they are also independent of each other given the
rule, so the ordinary standard error ``s / sqrt(N)`` is valid for it, which
is not strictly true in-sample.

``lsm_price`` reports both. The two independent sets come from one seed via
``numpy.random.SeedSequence.spawn``, which is designed to produce streams that
do not overlap.

What LSM actually prices
------------------------
With ``M`` exercise dates the simulation can only stop on those dates, so it
prices a **Bermudan** option. It converges to the American price only as
``M`` grows. ``exercise_tree.bermudan_binomial`` prices the same Bermudan
contract on a tree, so the two kinds of error can be told apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .common import CALL, check_kind, critical_value, payoff, validate_inputs

__all__ = [
    "LSMResult",
    "ExerciseRule",
    "laguerre_basis",
    "simulate_paths",
    "fit_exercise_rule",
    "apply_exercise_rule",
    "lsm_price",
    "lsm_exercise_boundary",
]


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------

def laguerre_basis(x: np.ndarray, degree: int, weighted: bool = False) -> np.ndarray:
    """Regression columns for the continuation value, one row per path.

    Unweighted (the default): ``L_0(x), ..., L_degree(x)``. ``L_0 = 1`` is
    the constant, so ``degree + 1`` columns spanning all polynomials up to
    ``degree``. Built from the three-term recurrence::

        L_0 = 1,   L_1 = 1 - x,   (n + 1) L_{n+1} = (2n + 1 - x) L_n - n L_{n-1}

    Weighted (Longstaff and Schwartz's form): a constant, then
    ``exp(-x/2) L_0(x), ..., exp(-x/2) L_{degree-1}(x)`` -- also
    ``degree + 1`` columns, so the two are compared like for like. Degree 3
    weighted is the paper's "constant plus three weighted Laguerre
    polynomials".
    """
    if int(degree) != degree or degree < 0:
        raise ValueError(f"degree must be a whole number >= 0, got {degree!r}")
    degree = int(degree)
    if weighted and degree < 1:
        raise ValueError("the weighted basis needs degree >= 1 (a constant plus L_0)")
    x = np.asarray(x, dtype=float)

    n_polys = degree if weighted else degree + 1
    polys = [np.ones_like(x)]
    if n_polys > 1:
        polys.append(1.0 - x)
    for n in range(1, n_polys - 1):
        polys.append(((2 * n + 1 - x) * polys[n] - n * polys[n - 1]) / (n + 1))

    if not weighted:
        return np.column_stack(polys)
    weight = np.exp(-0.5 * x)
    return np.column_stack([np.ones_like(x)] + [weight * p for p in polys])


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def simulate_paths(
    S: float,
    r: float,
    sigma: float,
    T: float,
    n_dates: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stock prices at the ``n_dates`` exercise dates, shape ``(n_paths, n_dates)``.

    Column ``m`` is the price at ``t = (m + 1) * T / n_dates``. Each step uses
    the exact GBM solution over that interval::

        S_{t+dt} = S_t * exp( (r - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z )

    so the simulated prices on the dates are exactly distributed: the only
    discretisation in LSM is *when* the holder may exercise, never the path.
    A running sum of the log increments builds the whole path in one go.
    """
    dt = T / n_dates
    increments = ((r - 0.5 * sigma * sigma) * dt
                  + sigma * math.sqrt(dt) * rng.standard_normal((n_paths, n_dates)))
    return S * np.exp(np.cumsum(increments, axis=1))


# ---------------------------------------------------------------------------
# Fitting and applying an exercise rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExerciseRule:
    """A fitted exercise strategy: one set of regression coefficients per date.

    ``coefficients[m]`` is ``None`` on a date where too few paths were in the
    money to fit the regression; the rule then never exercises early there.
    ``exercise_now`` is the decision at ``t = 0``, made on the training paths
    like every other decision: every path starts at the same ``S``, so the
    regression there is just the training paths' mean, and the rule exercises
    immediately if the payoff beats it.
    ``fit_ranges[m]`` is the span of ``S`` the fit at date ``m`` actually saw,
    because a regression says nothing reliable outside its data. The last
    date is expiry, where no regression is needed (the payoff is the value).
    """

    S: float
    K: float
    r: float
    sigma: float
    T: float
    kind: str
    n_dates: int
    degree: int
    weighted: bool
    coefficients: tuple
    fit_ranges: tuple
    exercise_now: bool = False

    @property
    def dt(self) -> float:
        return self.T / self.n_dates

    @property
    def times(self) -> np.ndarray:
        return np.arange(1, self.n_dates + 1) * self.dt

    def continuation(self, m: int, spots: np.ndarray) -> np.ndarray:
        """Fitted continuation value at date index ``m`` (``0 .. n_dates - 2``)."""
        beta = self.coefficients[m]
        if beta is None:
            return np.full(np.shape(spots), np.inf)
        return laguerre_basis(np.asarray(spots) / self.K, self.degree, self.weighted) @ beta


def _backward_sweep(paths, K, r, dt, kind, degree, weighted, rule=None):
    """The LSM backward induction over a matrix of paths.

    With ``rule=None`` it fits a fresh regression at each date and returns the
    coefficients it found; with a rule it only *applies* the stored
    coefficients. Both return each path's cash flow discounted to time 0.
    Applying a rule backwards gives the same answer as walking each path
    forwards to its first exercise date, since every earlier exercise
    overwrites every later one.
    """
    n_paths, n_dates = paths.shape
    n_columns = degree + 1
    step_discount = math.exp(-r * dt)

    cash = payoff(paths[:, -1], K, kind)           # value at expiry, in expiry money
    coefficients = [None] * (n_dates - 1)
    fit_ranges = [None] * (n_dates - 1)

    for m in range(n_dates - 2, -1, -1):
        cash *= step_discount                       # now in date-m money
        spots = paths[:, m]
        exercise_value = payoff(spots, K, kind)
        itm = np.flatnonzero(exercise_value > 0)

        if rule is None:
            if itm.size <= n_columns:               # not enough data to fit
                continue
            design = laguerre_basis(spots[itm] / K, degree, weighted)
            beta, *_ = np.linalg.lstsq(design, cash[itm], rcond=None)
            coefficients[m] = beta
            fit_ranges[m] = (float(spots[itm].min()), float(spots[itm].max()))
        else:
            beta = rule.coefficients[m]
            if beta is None or itm.size == 0:
                continue
            design = laguerre_basis(spots[itm] / K, degree, weighted)

        continuation = design @ beta
        exercise = itm[exercise_value[itm] > continuation]
        cash[exercise] = exercise_value[exercise]

    cash *= step_discount                           # from the first date to t = 0
    return cash, tuple(coefficients), tuple(fit_ranges)


def _check(S, K, r, sigma, T, kind, n_dates, degree, n_paths):
    validate_inputs(S, K, r, sigma, T)
    check_kind(kind)
    if T <= 0 or sigma <= 0:
        raise ValueError("LSM needs T > 0 and sigma > 0")
    for name, value, low in (("n_dates", n_dates, 1), ("n_paths", n_paths, 2)):
        if int(value) != value or value < low:
            raise ValueError(f"{name} must be a whole number >= {low}, got {value!r}")
    if int(degree) != degree or degree < 0:
        raise ValueError(f"degree must be a whole number >= 0, got {degree!r}")


def fit_exercise_rule(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    kind: str,
    n_paths: int,
    n_dates: int,
    rng: np.random.Generator,
    degree: int = 3,
    weighted: bool = False,
) -> tuple[ExerciseRule, np.ndarray]:
    """Fit an exercise rule on fresh paths; also return their in-sample cash flows.

    The time-0 decision is part of the rule: exercise now if the payoff beats
    the training paths' mean discounted cash flow (the in-sample value of
    holding). If it does, every training path's cash flow is that payoff.
    """
    _check(S, K, r, sigma, T, kind, n_dates, degree, n_paths)
    paths = simulate_paths(S, r, sigma, T, int(n_dates), int(n_paths), rng)
    cash, coefficients, fit_ranges = _backward_sweep(
        paths, K, r, T / n_dates, kind, int(degree), weighted)
    immediate = float(payoff(S, K, kind))
    exercise_now = immediate > float(cash.mean())
    if exercise_now:
        cash = np.full_like(cash, immediate)
    rule = ExerciseRule(S, K, r, sigma, T, kind, int(n_dates), int(degree),
                        weighted, coefficients, fit_ranges, exercise_now)
    return rule, cash


def apply_exercise_rule(rule: ExerciseRule, n_paths: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Discounted cash flows from following a *fixed* rule on new paths.

    If the rule exercises at ``t = 0``, every path pays the payoff now and the
    simulation is never needed.
    """
    if int(n_paths) != n_paths or n_paths < 2:
        raise ValueError(f"n_paths must be a whole number >= 2, got {n_paths!r}")
    if rule.exercise_now:
        return np.full(int(n_paths), float(payoff(rule.S, rule.K, rule.kind)))
    paths = simulate_paths(rule.S, rule.r, rule.sigma, rule.T, rule.n_dates,
                           int(n_paths), rng)
    cash, _, _ = _backward_sweep(paths, rule.K, rule.r, rule.dt, rule.kind,
                                 rule.degree, rule.weighted, rule=rule)
    return cash


# ---------------------------------------------------------------------------
# The price
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LSMResult:
    """Both LSM estimates, with their error bars.

    ``price`` is the out-of-sample estimate: the rule fitted on the training
    paths, applied to independent pricing paths. It is the one to quote --
    low-biased, with an honest standard error. ``in_sample_price`` re-uses
    the training paths and is reported to show how large the foresight bias
    is; its standard error is the naive ``s / sqrt(N)`` and is only
    indicative, because those cash flows all depend on a rule fitted to them.

    ``exercise_now`` is ``True`` when the rule exercises at time 0 -- decided
    on the training paths, never on the pricing paths. Both prices are then
    the payoff with zero standard error, which is honest: the strategy
    "exercise now" pays exactly that amount. (Deciding on the pricing paths'
    mean instead would let their noise choose the strategy, and near the
    critical spot report a zero-width interval around the wrong number.)
    """

    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    in_sample_price: float
    in_sample_standard_error: float
    n_train: int
    n_price: int
    n_dates: int
    degree: int
    weighted: bool
    seed: int
    rule: ExerciseRule
    exercise_now: bool = False
    confidence: float = 0.95

    @property
    def gap(self) -> float:
        """In-sample minus out-of-sample: the size of the foresight bias plus noise."""
        return self.in_sample_price - self.price

    def contains(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high

    def __str__(self) -> str:
        return (
            f"{self.price:.4f} +/- {self.standard_error:.4f} (se), "
            f"{self.confidence:.0%} CI [{self.ci_low:.4f}, {self.ci_high:.4f}]; in-sample "
            f"{self.in_sample_price:.4f}; {self.n_dates} dates, degree {self.degree}"
        )


def lsm_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    kind: str,
    n_paths: int = 100_000,
    n_dates: int = 50,
    degree: int = 3,
    seed: int = 12345,
    n_price_paths: int | None = None,
    weighted: bool = False,
    confidence: float = 0.95,
) -> LSMResult:
    """Price an American (strictly, Bermudan) option by Longstaff-Schwartz.

    ``n_paths`` paths fit the rule; ``n_price_paths`` (default: the same
    number) independent paths price it. ``n_dates`` exercise dates are spread
    evenly over ``(0, T]``, and the holder may also exercise at ``t = 0``.
    """
    _check(S, K, r, sigma, T, kind, n_dates, degree, n_paths)
    n_price_paths = int(n_paths) if n_price_paths is None else n_price_paths
    if int(n_price_paths) != n_price_paths or n_price_paths < 2:
        raise ValueError(f"n_price_paths must be a whole number >= 2, got {n_price_paths!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # One seed, two streams that are independent by construction.
    train_seq, price_seq = np.random.SeedSequence(seed).spawn(2)
    rule, train_cash = fit_exercise_rule(
        S, K, r, sigma, T, kind, n_paths, n_dates,
        np.random.default_rng(train_seq), degree, weighted)
    price_cash = apply_exercise_rule(rule, n_price_paths, np.random.default_rng(price_seq))

    price = float(price_cash.mean())
    se = float(price_cash.std(ddof=1) / math.sqrt(price_cash.size))
    in_sample = float(train_cash.mean())
    in_sample_se = float(train_cash.std(ddof=1) / math.sqrt(train_cash.size))

    half_width = critical_value(confidence) * se
    return LSMResult(
        price=price,
        standard_error=se,
        ci_low=price - half_width,
        ci_high=price + half_width,
        in_sample_price=in_sample,
        in_sample_standard_error=in_sample_se,
        n_train=int(n_paths),
        n_price=int(n_price_paths),
        n_dates=int(n_dates),
        degree=int(degree),
        weighted=weighted,
        seed=seed,
        rule=rule,
        exercise_now=rule.exercise_now,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# The exercise boundary the rule implies
# ---------------------------------------------------------------------------

def lsm_exercise_boundary(rule: ExerciseRule, grid_points: int = 2001
                          ) -> tuple[np.ndarray, np.ndarray]:
    """The critical stock price the fitted rule implies at each early date.

    At date ``m``, exercise wins wherever ``payoff(S) > continuation(S)``.
    For a put that is the region below some ``S*``; the boundary is the
    highest grid point where exercising wins (the lowest, for a call). The
    search is restricted to the range of in-the-money prices the regression
    at that date actually saw -- outside it the polynomial is extrapolating
    and its crossings mean nothing. ``NaN`` where the rule never exercises.

    Returns ``(times, boundary)`` for the ``n_dates - 1`` dates before expiry.
    """
    boundary = np.full(rule.n_dates - 1, np.nan)
    for m in range(rule.n_dates - 1):
        if rule.coefficients[m] is None:
            continue
        low, high = rule.fit_ranges[m]
        grid = np.linspace(low, high, grid_points)
        wins = payoff(grid, rule.K, rule.kind) > rule.continuation(m, grid)
        wins &= payoff(grid, rule.K, rule.kind) > 0
        if wins.any():
            boundary[m] = grid[wins].min() if rule.kind == CALL else grid[wins].max()
    return rule.times[:-1], boundary
