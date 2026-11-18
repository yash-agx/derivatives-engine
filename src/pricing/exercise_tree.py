"""Two things the binomial tree can tell Phase 7 that ``binomial.py`` does not return.

Phase 7 prices American puts by Monte Carlo (Longstaff-Schwartz, ``lsm.py``)
and checks them against the Phase 2 tree. Two questions need more from the
tree than a single price, and both are answered here, by the same backward
induction, without touching ``binomial.py``.

1. A Bermudan price
-------------------
A simulation can only exercise on the dates it simulates. With 50 exercise
dates a year, LSM is pricing a **Bermudan** option -- exercisable on those 50
dates only -- not a true American one exercisable at every instant. Fewer
choices can never be worth more, so::

    European  <=  Bermudan (M dates)  <=  American

``bermudan_binomial`` builds a tree whose steps line up with the exercise
dates (``steps_per_date`` steps between consecutive dates) and only applies
the early-exercise ``max`` on those steps. Comparing LSM against it isolates
the error of LSM itself; comparing it against the American tree measures what
the discrete dates cost. Without the split, any gap would be blamed on the
wrong thing.

The holder may also exercise at time 0 ("now"), exactly as on the American
tree, whose backward induction applies the ``max`` at the root as well.

2. The exercise boundary
------------------------
For a put, at each moment there is a critical stock price ``S*(t)``: below it
exercising now beats holding on, above it holding wins. That curve is the
whole content of the optimal strategy. On the tree it can be read straight off
the backward induction: at each step, the highest node where the exercise
value is strictly larger than the continuation value. (For a call it would be
the lowest such node; without dividends there is none.)

The tree's boundary is a staircase, because node prices are spaced by a
factor ``u = exp(sigma*sqrt(dt))``. More steps make the stairs finer.

The same split as for the price applies to the boundary. LSM's fitted rule
can only exercise on its dates, so the right curve to hold it against is the
boundary of the Bermudan tree on those dates (``bermudan_exercise_boundary``),
not the American one. Holding a Bermudan option is worth less (fewer chances
later), so exercising wins over a wider range of prices and its put boundary
sits *above* the American one.
"""

from __future__ import annotations

import math

import numpy as np

from .binomial import crr_parameters
from .common import CALL, check_kind, payoff, validate_inputs

__all__ = ["bermudan_binomial", "tree_exercise_boundary", "bermudan_exercise_boundary"]


def _check_tree_inputs(S, K, r, sigma, T, kind) -> None:
    validate_inputs(S, K, r, sigma, T)
    check_kind(kind)
    if T <= 0 or sigma <= 0:
        raise ValueError("a tree needs T > 0 and sigma > 0")


def _whole(name: str, value) -> int:
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive whole number, got {value!r}")
    return int(value)


def bermudan_binomial(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_dates: int,
    steps_per_date: int,
    kind: str,
) -> float:
    """Price an option exercisable at ``t = 0`` and on ``n_dates`` equally spaced dates.

    The dates are ``T/n_dates, 2T/n_dates, ..., T``. The tree has
    ``n_dates * steps_per_date`` steps, so every exercise date falls exactly
    on a layer of nodes.

    Two limits pin it down, and the tests check both: one step per date is
    the American tree with ``n_dates`` steps, and one date is a European
    option that may also be exercised immediately.
    """
    _check_tree_inputs(S, K, r, sigma, T, kind)
    n_dates = _whole("n_dates", n_dates)
    steps_per_date = _whole("steps_per_date", steps_per_date)
    steps = n_dates * steps_per_date

    params = crr_parameters(r, sigma, T, steps)
    p, discount = params["p"], params["discount"]
    log_u = sigma * math.sqrt(params["dt"])

    values = payoff(S * np.exp(log_u * (steps - 2 * np.arange(steps + 1))), K, kind)
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[:-1] + (1.0 - p) * values[1:])
        if step % steps_per_date == 0:          # an exercise date, or t = 0
            spots = S * np.exp(log_u * (step - 2 * np.arange(step + 1)))
            values = np.maximum(values, payoff(spots, K, kind))
    return float(values[0])


def tree_exercise_boundary(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """The American tree's critical stock price at every step.

    Returns ``(times, boundary)`` for steps ``1 .. steps - 1`` (at expiry the
    rule is trivially "exercise if in the money", so it is left out). For a
    put the boundary is the highest node where exercising now is strictly
    better than holding; for a call, the lowest. ``NaN`` where no node of
    that layer should be exercised -- early on, when the tree has not yet
    spread far enough from the spot to reach the exercise region.
    """
    _check_tree_inputs(S, K, r, sigma, T, kind)
    steps = _whole("steps", steps)

    params = crr_parameters(r, sigma, T, steps)
    p, discount = params["p"], params["discount"]
    log_u = sigma * math.sqrt(params["dt"])

    times = np.arange(1, steps) * params["dt"]
    boundary = np.full(steps - 1, np.nan)

    values = payoff(S * np.exp(log_u * (steps - 2 * np.arange(steps + 1))), K, kind)
    for step in range(steps - 1, 0, -1):
        continuation = discount * (p * values[:-1] + (1.0 - p) * values[1:])
        spots = S * np.exp(log_u * (step - 2 * np.arange(step + 1)))
        exercise_value = payoff(spots, K, kind)
        exercise = (exercise_value > continuation) & (exercise_value > 0)
        if exercise.any():
            chosen = spots[exercise]
            boundary[step - 1] = chosen.min() if kind == CALL else chosen.max()
        values = np.maximum(continuation, exercise_value)
    return times, boundary


def bermudan_exercise_boundary(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_dates: int,
    steps_per_date: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """The Bermudan tree's critical stock price on each exercise date before expiry.

    Same tree as ``bermudan_binomial``; on each exercise date it records the
    highest node (lowest, for a call) where exercising is strictly better than
    holding. Returns ``(times, boundary)`` for dates ``1 .. n_dates - 1``,
    i.e. ``t = T/n_dates, ..., (n_dates - 1) T/n_dates`` -- the same dates
    ``lsm.lsm_exercise_boundary`` reports. ``NaN`` where no node is exercised.
    """
    _check_tree_inputs(S, K, r, sigma, T, kind)
    n_dates = _whole("n_dates", n_dates)
    steps_per_date = _whole("steps_per_date", steps_per_date)
    steps = n_dates * steps_per_date

    params = crr_parameters(r, sigma, T, steps)
    p, discount = params["p"], params["discount"]
    log_u = sigma * math.sqrt(params["dt"])

    times = np.arange(1, n_dates) * (T / n_dates)
    boundary = np.full(n_dates - 1, np.nan)

    values = payoff(S * np.exp(log_u * (steps - 2 * np.arange(steps + 1))), K, kind)
    for step in range(steps - 1, 0, -1):
        values = discount * (p * values[:-1] + (1.0 - p) * values[1:])
        if step % steps_per_date == 0:              # an exercise date
            spots = S * np.exp(log_u * (step - 2 * np.arange(step + 1)))
            exercise_value = payoff(spots, K, kind)
            exercise = (exercise_value > values) & (exercise_value > 0)
            if exercise.any():
                chosen = spots[exercise]
                boundary[step // steps_per_date - 1] = (
                    chosen.min() if kind == CALL else chosen.max())
            values = np.maximum(values, exercise_value)
    return times, boundary
