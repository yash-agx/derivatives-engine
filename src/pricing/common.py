"""Pieces every pricing model in this project shares.

Phase 1 had one model, so validation and the option-type constants lived in
``black_scholes.py``. Phase 2 adds a binomial tree and a Monte Carlo
simulation, and all three need the same three things: a way to say "call or
put", a check that the market inputs make sense, and the exercise payoff. They
live here so there is exactly one copy of each.

Conventions (the same everywhere in the project): ``r`` and ``sigma`` are
decimals per year, ``T`` is in years, and no dividends are modelled.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.stats import norm

CALL = "call"
PUT = "put"
STRADDLE = "straddle"

#: What the Greeks and the binomial tree accept. They branch on call-versus-put
#: internally, so a third kind would quietly produce a wrong number.
KINDS = (CALL, PUT)
#: What ``payoff()`` and the Black-Scholes ``price()`` accept. A straddle is
#: just a call plus a put on the same strike, so both are well defined for it.
PAYOFF_KINDS = (CALL, PUT, STRADDLE)


def check_kind(kind: str) -> str:
    """Reject anything that is not ``"call"`` or ``"put"``."""
    if kind not in KINDS:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind


def check_payoff_kind(kind: str) -> str:
    """Reject anything that is not a payoff this project can price."""
    if kind not in PAYOFF_KINDS:
        raise ValueError(
            f"kind must be 'call', 'put' or 'straddle', got {kind!r}"
        )
    return kind


def validate_inputs(S: float, K: float, r: float, sigma: float, T: float) -> None:
    """Reject inputs the models cannot mean anything for.

    ``r`` is deliberately unconstrained -- negative rates are real.
    ``S`` and ``K`` must be strictly positive: Black-Scholes takes
    ``ln(S/K)``, and a tree built from a zero spot never leaves zero.
    """
    # Finiteness first: NaN fails every comparison below, so checking it here
    # is what makes the error message say what is actually wrong.
    for name, value in (("S", S), ("K", K), ("r", r), ("sigma", sigma), ("T", T)):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be a finite number, got {value}")
    if not S > 0:
        raise ValueError(f"S (spot) must be > 0, got {S}")
    if not K > 0:
        raise ValueError(f"K (strike) must be > 0, got {K}")
    if sigma < 0:
        raise ValueError(f"sigma (volatility) must be >= 0, got {sigma}")
    if T < 0:
        raise ValueError(f"T (years to expiry) must be >= 0, got {T}")


@lru_cache(maxsize=None)
def critical_value(confidence: float) -> float:
    """The two-sided normal critical value for a confidence level.

    Cached because it depends on nothing else: every Monte Carlo run used to
    recompute the same 1.959963... through scipy, which costs 27 microseconds
    and is most of the fixed overhead of a small run.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(norm.ppf(0.5 + 0.5 * confidence))


def payoff(spot, K: float, kind: str):
    """Exercise value of the option: what it is worth if exercised right now.

    ``max(S - K, 0)`` for a call, ``max(K - S, 0)`` for a put, and ``|S - K|``
    for a straddle (owning both). This is the whole definition of an option --
    every pricing model in the project is ultimately just a different way of
    averaging and discounting this number.

    Works on a single spot or on a NumPy array of them, which is what the tree
    (a whole layer of nodes) and the simulation (a whole batch of paths) need.
    """
    check_payoff_kind(kind)
    if kind == CALL:
        return np.maximum(spot - K, 0.0)
    if kind == PUT:
        return np.maximum(K - spot, 0.0)
    return np.abs(spot - K)
