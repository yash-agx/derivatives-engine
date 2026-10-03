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
    # float(): scipy returns numpy scalars, and this is a plain-scalar API.
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


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
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def price(S: float, K: float, r: float, sigma: float, T: float, kind: str) -> float:
    """``call_price`` or ``put_price``, chosen by ``kind``.

    Convenience for loops and tests that run over both option types.
    """
    _check_kind(kind)
    if kind == CALL:
        return call_price(S, K, r, sigma, T)
    return put_price(S, K, r, sigma, T)


# ---------------------------------------------------------------------------
# The Greeks
# ---------------------------------------------------------------------------
#
# A Greek is just a partial derivative of the price: "if one input moves a
# little, how much does the option move?" They are what a desk actually
# trades on, because they say how to hedge and what will hurt you.
#
#     delta = dP/dS        gamma = d2P/dS2      vega = dP/dsigma
#     theta = dP/dt        rho   = dP/dr
#
# Units in this module (stated once, used everywhere):
#   * delta -- per 1.0 of spot (dimensionless; also "shares per option").
#   * gamma -- per 1.0 of spot, per 1.0 of spot.
#   * vega  -- per 1.0 of volatility, i.e. per 100 vol points.
#              Divide by 100 for the usual desk quote "per 1% of vol".
#   * theta -- per year. Divide by 365 for "per calendar day".
#   * rho   -- per 1.0 of rate. Divide by 100 for "per 1% of rate".
# Keeping everything per-unit is what makes the finite-difference tests a fair
# comparison: bumping sigma by h and dividing by h gives a per-unit number too.

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


def _degenerate_greeks(S: float, K: float, r: float, T: float, kind: str) -> dict:
    """Greeks when there is no randomness left (``T == 0`` or ``sigma == 0``).

    The payoff is already decided, so the option is just a forward contract
    (if it will be exercised) or nothing (if it will not), and the Greeks are
    the derivatives of ``max(S - K*exp(-r*T), 0)`` for a call::

        in the money :  delta = +/-1, gamma = 0, vega = 0,
                        theta = -/+ r*K*exp(-r*T), rho = +/- T*K*exp(-r*T)
        out of it    :  every Greek is 0

    The one awkward point is exactly on the boundary ``S == K*exp(-r*T)``,
    where the payoff has a kink and the derivatives strictly do not exist. We
    report the limits of the real formulas as ``sigma -> 0``: delta is +/-0.5
    (the midpoint of the jump), gamma is infinite (the curvature of a kink),
    and vega is ``S*sqrt(T)*phi(0)``, which is *not* zero -- right at the
    kink, the tiniest bit of volatility is still worth something.
    """
    discounted_k = K * math.exp(-r * T)
    out = dict.fromkeys(GREEK_NAMES, 0.0)

    if S == discounted_k:  # knife edge: see the docstring
        half_interest = 0.5 * r * discounted_k
        half_rho = 0.5 * T * discounted_k
        out["delta"] = 0.5 if kind == CALL else -0.5
        out["gamma"] = math.inf
        out["vega"] = float(S * math.sqrt(T) * norm.pdf(0.0))
        out["theta"] = -half_interest if kind == CALL else half_interest
        out["rho"] = half_rho if kind == CALL else -half_rho
        return out

    call_is_in_the_money = S > discounted_k
    if kind == CALL and call_is_in_the_money:
        out["delta"] = 1.0
        out["theta"] = -r * discounted_k
        out["rho"] = T * discounted_k
    elif kind == PUT and not call_is_in_the_money:
        out["delta"] = -1.0
        out["theta"] = r * discounted_k
        out["rho"] = -T * discounted_k
    # Otherwise the option expires worthless and all five Greeks stay 0.0.
    return out


def delta(S: float, K: float, r: float, sigma: float, T: float, kind: str) -> float:
    """Delta: change in option price per 1.0 change in the spot.

    ``call delta = N(d1)``, ``put delta = N(d1) - 1``.

    Three ways to read it, all useful:

    * **Sensitivity.** A call with delta 0.52 gains about $0.52 if the stock
      gains $1.
    * **Hedge ratio.** To be flat against small stock moves, hold ``-delta``
      shares per call you own. This is the number Hull's delta-hedging
      examples are about.
    * **Share equivalent.** The call behaves like ``N(d1)`` shares right now.
      Deep in the money ``N(d1) -> 1`` (it *is* the stock, minus a loan);
      deep out of the money ``N(d1) -> 0`` (it is nothing).

    Call delta runs 0 to 1, put delta -1 to 0, and
    ``call delta - put delta = 1`` always -- that is the derivative of
    put-call parity with respect to ``S``.
    """
    _validate(S, K, r, sigma, T)
    _check_kind(kind)
    if _is_degenerate(sigma, T):
        return _degenerate_greeks(S, K, r, T, kind)["delta"]
    d1, _ = d1_d2(S, K, r, sigma, T)
    if kind == CALL:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def gamma(S: float, K: float, r: float, sigma: float, T: float, kind: str = CALL) -> float:
    """Gamma: change in *delta* per 1.0 change in the spot (curvature).

    ``gamma = phi(d1) / (S * sigma * sqrt(T))``, where ``phi`` is the normal
    PDF. It is identical for a call and a put: by parity they differ by
    ``S - K*exp(-r*T)``, which is linear in ``S`` and so has no curvature.
    ``kind`` is accepted only so that all five Greeks share one signature.

    What it tells a trader: how fast the hedge goes stale. Delta says how many
    shares to hold *now*; gamma says how quickly that number changes, so it is
    the re-hedging cost of a position. Long gamma means the hedge keeps buying
    low and selling high; short gamma means the opposite, which is how option
    sellers get hurt in a fast market.

    Gamma is largest for at-the-money options close to expiry -- exactly where
    a small move flips the option between "worth nothing" and "worth
    intrinsic".
    """
    _validate(S, K, r, sigma, T)
    _check_kind(kind)
    if _is_degenerate(sigma, T):
        return _degenerate_greeks(S, K, r, T, kind)["gamma"]
    d1, _ = d1_d2(S, K, r, sigma, T)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def vega(S: float, K: float, r: float, sigma: float, T: float, kind: str = CALL) -> float:
    """Vega: change in option price per 1.0 change in volatility.

    ``vega = S * phi(d1) * sqrt(T)``. Same for calls and puts, for the same
    parity reason as gamma (the difference does not involve ``sigma`` at all).

    What it tells a trader: how much of the position is a bet on volatility
    rather than on direction. It is positive for both calls and puts, because
    more uncertainty can only help someone whose loss is capped at the premium
    and whose upside is not capped.

    Note the ``sqrt(T)``: vega grows with time to expiry, so long-dated
    options are the vol-sensitive ones, while gamma is a short-dated
    phenomenon. Divide by 100 for the desk convention "dollars per vol point".
    """
    _validate(S, K, r, sigma, T)
    _check_kind(kind)
    if _is_degenerate(sigma, T):
        return _degenerate_greeks(S, K, r, T, kind)["vega"]
    d1, _ = d1_d2(S, K, r, sigma, T)
    return float(S * norm.pdf(d1) * math.sqrt(T))


def theta(S: float, K: float, r: float, sigma: float, T: float, kind: str) -> float:
    """Theta: change in option price per year of *calendar time passing*.

    ::

        call theta = -S*phi(d1)*sigma / (2*sqrt(T)) - r*K*exp(-r*T)*N(d2)
        put  theta = -S*phi(d1)*sigma / (2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)

    Theta is ``dP/dt`` with ``t`` running forwards, which is ``-dP/dT``: that
    is where the minus signs come from, and why theta is usually negative.
    Options are wasting assets -- hold one with nothing else changing and it is
    worth less tomorrow, because there is less time left for the stock to move
    your way.

    The two terms are two separate effects:

    * ``-S*phi(d1)*sigma / (2*sqrt(T))`` is the loss of optionality, the
      valuable uncertainty draining away. Always negative, and it blows up
      like ``1/sqrt(T)`` at the very end -- the last day of an at-the-money
      option is brutal.
    * the ``r*K*exp(-r*T)`` term is interest on the strike you have not paid
      yet. For a call that works against you as expiry approaches (negative);
      for a put you are the one due to *receive* the strike, so it works in
      your favour (positive) and can make a deep-in-the-money put's theta
      positive overall.

    Divide by 365 for the "per day" number desks quote.
    """
    _validate(S, K, r, sigma, T)
    _check_kind(kind)
    if _is_degenerate(sigma, T):
        return _degenerate_greeks(S, K, r, T, kind)["theta"]
    d1, d2 = d1_d2(S, K, r, sigma, T)
    optionality_decay = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
    interest_on_strike = r * K * math.exp(-r * T)
    if kind == CALL:
        return float(optionality_decay - interest_on_strike * norm.cdf(d2))
    return float(optionality_decay + interest_on_strike * norm.cdf(-d2))


def rho(S: float, K: float, r: float, sigma: float, T: float, kind: str) -> float:
    """Rho: change in option price per 1.0 change in the risk-free rate.

    ``call rho = K*T*exp(-r*T)*N(d2)``, ``put rho = -K*T*exp(-r*T)*N(-d2)``.

    What it tells a trader: the interest-rate exposure. A call has positive
    rho, because owning a call means you have *not* yet paid the strike, and a
    higher rate makes that deferred payment cheaper in present-value terms. A
    put has negative rho for the mirror reason: you are waiting to receive the
    strike, and a higher rate discounts it harder.

    Rho is the smallest of the five for short-dated equity options -- note the
    factor ``T`` out front -- which is why it is usually the last one anyone
    mentions. Divide by 100 for "per 1% of rate".
    """
    _validate(S, K, r, sigma, T)
    _check_kind(kind)
    if _is_degenerate(sigma, T):
        return _degenerate_greeks(S, K, r, T, kind)["rho"]
    _, d2 = d1_d2(S, K, r, sigma, T)
    discounted_k_times_t = K * T * math.exp(-r * T)
    if kind == CALL:
        return float(discounted_k_times_t * norm.cdf(d2))
    return float(-discounted_k_times_t * norm.cdf(-d2))


def greeks(S: float, K: float, r: float, sigma: float, T: float, kind: str) -> dict:
    """All five Greeks in one dict, keyed by ``GREEK_NAMES``.

    This recomputes ``d1``/``d2`` once per Greek instead of sharing them. That
    is deliberate for now: each formula lives in exactly one place, which is
    what keeps the tests meaningful. Phase 5 is where speed matters.
    """
    _check_kind(kind)
    return {
        "delta": delta(S, K, r, sigma, T, kind),
        "gamma": gamma(S, K, r, sigma, T, kind),
        "vega": vega(S, K, r, sigma, T, kind),
        "theta": theta(S, K, r, sigma, T, kind),
        "rho": rho(S, K, r, sigma, T, kind),
    }
