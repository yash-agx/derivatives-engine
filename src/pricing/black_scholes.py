"""Black-Scholes closed-form prices for European options.

Conventions used everywhere in this project
-------------------------------------------
``S``      spot price of the underlying today
``K``      strike price
``r``      continuously compounded risk-free rate, per year, as a decimal (5% -> 0.05)
``sigma``  volatility of the underlying, per year, as a decimal (20% -> 0.20)
``T``      time to expiry in years (6 months -> 0.5)
``kind``   ``"call"`` or ``"put"``

No dividends (q = 0) and European exercise only: those are Black-Scholes'
assumptions, and they are the limitations the write-up has to discuss.

The formulas
------------
::

    call = S * N(d1) - K * exp(-r*T) * N(d2)
    put  = K * exp(-r*T) * N(-d2) - S * N(-d1)

    d1 = (ln(S/K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

``N()`` is the standard normal CDF: ``N(x) = P(Z <= x)`` for ``Z ~ Normal(0, 1)``.

What the pieces actually mean
-----------------------------
In this model the stock ends up log-normally distributed, and under the
risk-neutral measure::

    ln(S_T) ~ Normal(ln(S) + (r - sigma**2 / 2) * T,  sigma**2 * T)

so ``sigma * sqrt(T)`` is the standard deviation of the log return over the
whole life of the option. That is the model's natural ruler: it is the only
sensible unit for "how far away" the strike is, because a 5-point move means
something completely different for a calm stock with a week to run than for a
wild one with a year.

* ``d2`` is how many of those standard deviations the option is in the money
  by, measured in log terms. Equivalently, with the forward price
  ``F = S * exp(r*T)``::

        d2 = ln(F / K) / (sigma*sqrt(T)) - (sigma*sqrt(T)) / 2

  ``N(d2)`` is then the risk-neutral probability of finishing in the money,
  i.e. ``P(S_T > K)``.
* ``d1 = d2 + sigma*sqrt(T)`` is that same distance measured under a
  probability weighting in which each outcome counts in proportion to the
  stock price itself (the "share measure"). The useful consequence is that
  ``S * N(d1)`` is today's value of *the share we receive, but only in the
  cases where we exercise*.
* So the call formula reads as **what you get minus what you pay**:
  ``S * N(d1)`` is the present value of the share received, and
  ``K * exp(-r*T) * N(d2)`` is the present value of the cash paid -- the
  strike ``K``, discounted, and handed over only in the ``N(d2)`` fraction of
  scenarios where exercise happens.
* The ``sigma**2 / 2`` term in ``d1`` is not a fudge factor: it is the
  Ito/Jensen correction that keeps the *expected* stock price equal to the
  forward even though we are averaging an exponential. It is why ``d1`` and
  ``d2`` differ by exactly one standard deviation.
"""

from __future__ import annotations

import math

from scipy.stats import norm

CALL = "call"
PUT = "put"
KINDS = (CALL, PUT)

# If sigma*sqrt(T) is below this, there is (numerically) no randomness left in
# the payoff: d1 and d2 would blow up to +/- infinity. We switch to the
# certain-payoff formulas instead. See _degenerate_price.
_NO_VOL = 1e-12


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind


def _validate(S: float, K: float, r: float, sigma: float, T: float) -> None:
    """Reject inputs the formulas cannot mean anything for.

    ``r`` is deliberately unconstrained -- negative rates are real.
    ``S`` and ``K`` must be strictly positive because the formulas take
    ``ln(S/K)``.
    """
    if not S > 0:
        raise ValueError(f"S (spot) must be > 0, got {S}")
    if not K > 0:
        raise ValueError(f"K (strike) must be > 0, got {K}")
    if sigma < 0:
        raise ValueError(f"sigma (volatility) must be >= 0, got {sigma}")
    if T < 0:
        raise ValueError(f"T (years to expiry) must be >= 0, got {T}")
    for name, value in (("S", S), ("K", K), ("r", r), ("sigma", sigma), ("T", T)):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be a finite number, got {value}")


def d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    """Return ``(d1, d2)``.

    Both are distances to the strike measured in standard deviations of the
    log return (see the module docstring). Only valid when
    ``sigma * sqrt(T) > 0``; callers must handle the degenerate case first.
    """
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def _is_degenerate(sigma: float, T: float) -> bool:
    """True when the payoff carries no uncertainty (T == 0 or sigma == 0)."""
    return sigma * math.sqrt(T) < _NO_VOL


def _degenerate_price(S: float, K: float, r: float, T: float, kind: str) -> float:
    """Price when there is no randomness left: a certain payoff, discounted.

    With ``sigma == 0`` the stock grows at the risk-free rate with certainty,
    so it ends at the forward ``F = S*exp(r*T)``, and::

        call = exp(-r*T) * max(F - K, 0) = max(S - K*exp(-r*T), 0)

    With ``T == 0`` the same expression collapses to the intrinsic value
    ``max(S - K, 0)``, so one branch covers both cases.
    """
    discounted_k = K * math.exp(-r * T)
    if kind == CALL:
        return max(S - discounted_k, 0.0)
    return max(discounted_k - S, 0.0)


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a European call."""
    _validate(S, K, r, sigma, T)
    if _is_degenerate(sigma, T):
        return _degenerate_price(S, K, r, T, CALL)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a European put.

    Same reading as the call, mirrored: ``K*exp(-r*T)*N(-d2)`` is the present
    value of the cash received, ``S*N(-d1)`` the value of the share handed
    over. ``N(-d2) = 1 - N(d2)`` is the risk-neutral probability of finishing
    in the money for a put.
    """
    _validate(S, K, r, sigma, T)
    if _is_degenerate(sigma, T):
        return _degenerate_price(S, K, r, T, PUT)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price(S: float, K: float, r: float, sigma: float, T: float, kind: str) -> float:
    """``call_price`` or ``put_price``, chosen by ``kind``.

    Convenience for loops and tests that run over both option types.
    """
    _check_kind(kind)
    if kind == CALL:
        return call_price(S, K, r, sigma, T)
    return put_price(S, K, r, sigma, T)
