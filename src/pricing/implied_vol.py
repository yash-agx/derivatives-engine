"""Implied volatility: inverting Black-Scholes for sigma.

Phase 6. Every earlier phase went forwards -- given a volatility, what is the
option worth? The market works the other way round. The price is the thing you
can see; the volatility is the thing you cannot. So the question becomes:

    **which sigma, put into Black-Scholes, reproduces the price the market
    actually traded at?**

That number is the *implied volatility*. It is not a forecast and not a
measurement of past moves -- it is a restatement of the price in units of
volatility, using Black-Scholes as the dictionary. Quoting an option at "14.2
vol" instead of "212.55 rupees" is useful because the vol number is roughly
comparable across strikes, expiries and underlyings, while the price is not.

Why there is no formula for it
------------------------------
Black-Scholes is::

    C(sigma) = S*N(d1) - K*exp(-r*T)*N(d2)

and ``sigma`` appears inside two normal CDFs, which have no elementary
inverse. There is no closed form for sigma, so it has to be found
numerically. The good news is that the function being inverted is about as
well behaved as a function can be:

* **Strictly increasing in sigma.** ``dC/dsigma`` is vega,
  ``S*phi(d1)*sqrt(T)``, which is positive for every ``sigma > 0``. So the
  solution is unique if it exists -- there is no risk of finding "an" answer
  that is the wrong one.
* **Its derivative is free.** Phase 1 already has vega in closed form, which
  is exactly what Newton-Raphson needs. No finite differencing.
* **Its range is known in advance.** As ``sigma -> 0`` the price falls to the
  discounted intrinsic value; as ``sigma -> infinity`` a call rises to ``S``
  and a put to ``K*exp(-r*T)``. Those are the no-arbitrage bounds, and they
  are what makes it possible to say "no sigma can produce this price" rather
  than returning a number that means nothing.

The method: safeguarded Newton-Raphson
--------------------------------------
Newton-Raphson treats the curve as a straight line through the current point
and jumps to where that line crosses the target::

    sigma_next = sigma - (C(sigma) - price) / vega(sigma)

Read plainly: "I am 3 rupees too cheap, and each vol point is worth 14
rupees, so add 3/14 of a vol point." It converges quadratically -- the number
of correct digits roughly doubles each step -- which is why it is the standard
choice.

It also fails in two ways that matter on real market data:

1. **Vega goes to zero.** Far out of the money, or very close to expiry, the
   price barely responds to volatility at all. Dividing by a near-zero vega
   throws the iterate to sigma = 400% or to a negative number.
2. **The jump overshoots.** The price is convex in sigma on one side of the
   at-the-money point and concave on the other, so a tangent taken in the
   flat tail can land outside the region where a solution can live.

The fix used here is a *bracket*. The solution is known to lie between a
``lo`` where the model is too cheap and a ``hi`` where it is too expensive,
and because the price is monotone in sigma, every evaluation tightens that
bracket whichever way it comes out. Newton's step is accepted only when it
lands **inside** the current bracket and vega is not negligible; otherwise
the step is replaced by bisection -- the midpoint, which is guaranteed to
halve the bracket. So the iteration keeps Newton's speed where Newton works
and bisection's guaranteed progress where it does not. It cannot diverge, and
it cannot return a sigma outside the bracket it started with.

What this module will not do
----------------------------
If the price lies outside the no-arbitrage bounds, **no** volatility
reproduces it, and the function returns ``nan``. That is not a failure of the
solver; it is information about the quote. Real exchange closes violate the
bounds regularly -- a stale last trade on an illiquid deep-in-the-money
strike can settle below intrinsic value -- and the Phase 6 analysis counts how
often. Returning a clipped 0.001 or a capped 500% instead would hide exactly
the data problem that needs reporting.

Argument order note
-------------------
``implied_vol(price, S, K, T, r, kind)`` leads with the price, because the
price is the input being inverted. The rest of the project uses
``(S, K, r, sigma, T)``, and the pricing calls inside here are written in that
order.
"""

from __future__ import annotations

import math
import sys
from typing import NamedTuple

from . import black_scholes as bs
from .common import CALL, check_kind, validate_inputs

__all__ = [
    "SIGMA_MAX",
    "no_arbitrage_bounds",
    "ImpliedVolResult",
    "implied_vol_detail",
    "implied_vol",
]

#: Volatilities above this are treated as "no solution in a sensible range".
#: 1000% annualised is far beyond anything a listed equity index option
#: implies; a price needing more than this is a broken quote, not a volatile
#: market. Reported as nan rather than clipped, for the reason above.
SIGMA_MAX = 10.0

#: Newton's step is only trusted when vega is at least this fraction of its
#: largest possible value, ``S*sqrt(T)*phi(0)``. Below it the price carries
#: essentially no information about sigma and ``f/vega`` is numerical noise
#: amplified by a huge factor, so bisection takes over.
VEGA_FLOOR_FRACTION = 1e-12

#: ``phi(0) = 1/sqrt(2*pi)``, the peak of the standard normal density. Used
#: only to turn VEGA_FLOOR_FRACTION into an absolute vega threshold.
_PHI_0 = 0.3989422804014327

#: Stopping tolerance, on **sigma** and **relative**: stop once the bracket is
#: narrower than ``SIGMA_TOL * max(1, sigma)``, i.e. the answer is pinned to
#: about twelve significant figures. Far tighter than the 1e-6 the round-trip
#: tests ask for.
#:
#: Converging on sigma rather than on the price error is a deliberate choice.
#: A price tolerance sounds natural -- "stop when the model matches the quote"
#: -- but it is an absolute currency amount, and the sigma it corresponds to
#: is ``price_tol / vega``. Where vega is small (far out of the money, or
#: nearly expired) that is a wide band of volatilities, so a price-based stop
#: returns an answer that matches the quote to ten decimal places and is wrong
#: in the *first*. The bracket, by contrast, measures the thing actually being
#: asked for.
SIGMA_TOL = 1e-12

#: Slack allowed when checking the bounds, as a fraction of ``max(S, K)``. It
#: absorbs floating-point error in computing the bound itself (one exp and one
#: subtraction), not noise in the quote: a price that violates intrinsic value
#: by more than a part in 10^12 of the spot is violating it for real.
BOUND_SLACK = 1e-12


class ImpliedVolResult(NamedTuple):
    """What the solver did, not just what it returned.

    ``sigma``            the implied volatility, or ``nan`` if there is none.
    ``iterations``       how many times the price was evaluated.
    ``newton_steps``     of the moves made, how many were Newton's.
    ``bisection_steps``  of the moves made, how many fell back to bisection.
    ``status``           ``"ok"``, ``"at_lower_bound"``, ``"below_lower_bound"``,
                         ``"above_upper_bound"``, ``"zero_time"``,
                         ``"above_sigma_max"`` or ``"not_converged"``.

    The counts exist because "how often did Newton need rescuing, and where?"
    is a measurable property of a real option chain, and Phase 6 reports it
    rather than asserting that the safeguard was necessary.
    """

    sigma: float
    iterations: int
    newton_steps: int
    bisection_steps: int
    status: str


def no_arbitrage_bounds(S: float, K: float, r: float, T: float,
                        kind: str) -> tuple[float, float]:
    """The range of prices a European option can have, for *any* volatility.

    ::

        call:  max(S - K*exp(-r*T), 0)  <=  C  <=  S
        put:   max(K*exp(-r*T) - S, 0)  <=  P  <=  K*exp(-r*T)

    Both ends are the limits of the Black-Scholes formula itself, which is why
    they are exactly the right test for "can this price be inverted?":

    * **Lower bound** (``sigma -> 0``): with no uncertainty the stock ends at
      the forward ``S*exp(r*T)`` for certain, so the call is worth the
      discounted certain payoff ``S - K*exp(-r*T)``, floored at zero. It is
      also an arbitrage bound: a call cheaper than that could be bought,
      hedged by shorting the stock and lending the strike, for a riskless
      profit. Note it is ``S - K*exp(-r*T)``, not ``S - K`` -- the
      *discounted* strike, because the strike is not paid until expiry. For a
      call that makes the bound **higher** than undiscounted intrinsic value;
      for a put it makes it **lower**, which is why a deep in-the-money
      European put legitimately trades below ``K - S``.
    * **Upper bound** (``sigma -> infinity``): a call can never be worth more
      than the share it delivers, and a put never more than the present value
      of the strike it pays you, because that is the most the payoff can ever
      be.

    Prices strictly inside the range invert to a unique finite sigma; prices
    *on* either end need sigma = 0 or sigma = infinity.
    """
    check_kind(kind)
    discounted_k = K * math.exp(-r * T)
    if kind == CALL:
        return max(S - discounted_k, 0.0), S
    return max(discounted_k - S, 0.0), discounted_k


def implied_vol_detail(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: str,
    *,
    sigma_tol: float = SIGMA_TOL,
    sigma_max: float = SIGMA_MAX,
    max_iter: int = 100,
) -> ImpliedVolResult:
    """Implied volatility, with a record of how it was found.

    See ``implied_vol`` for the plain-float version and the module docstring
    for the method. The steps are:

    1. **Reject what cannot be inverted.** ``T <= 0`` leaves no volatility to
       imply (the payoff is already decided). A price outside the
       no-arbitrage bounds has no sigma at all.
    2. **Bracket the root.** ``lo = 0`` is guaranteed too cheap by step 1.
       ``hi`` starts at 100% vol and doubles until the model is too
       expensive, giving a bracket ``[lo, hi]`` that must contain the answer
       because the price is continuous and strictly increasing in sigma.
    3. **Iterate.** Each step evaluates the price and vega once, tightens the
       bracket using the sign of the error, and then moves by Newton if the
       step is trustworthy and inside the bracket, or by bisection if not.
    """
    check_kind(kind)
    # sigma is the unknown, so 0.0 stands in for it here: the rest of the
    # validation (finite, S > 0, K > 0, T >= 0) is what is wanted.
    validate_inputs(S, K, r, 0.0, T)
    if not math.isfinite(price):
        raise ValueError(f"price must be a finite number, got {price}")

    nan = float("nan")
    if T <= 0.0:
        # At expiry the option is worth its intrinsic value whatever the
        # volatility was, so the price says nothing about sigma.
        return ImpliedVolResult(nan, 0, 0, 0, "zero_time")

    lower, upper = no_arbitrage_bounds(S, K, r, T, kind)
    slack = BOUND_SLACK * max(S, K)
    if price < lower - slack:
        return ImpliedVolResult(nan, 0, 0, 0, "below_lower_bound")
    if price <= lower + slack:
        # The quote *is* the zero-volatility price, so the implied volatility
        # is zero. Not a fudge: it is the only statement the quote supports.
        # A deep in-the-money option trading at discounted intrinsic value is
        # the usual way this happens, and it is also the point past which no
        # solver could do better -- vega there is far below the resolution of
        # a price quoted in exchange ticks.
        return ImpliedVolResult(0.0, 0, 0, 0, "at_lower_bound")
    if price > upper - slack:
        # Includes price == upper exactly, which is the sigma -> infinity
        # limit: approached, never attained, so there is no finite answer.
        return ImpliedVolResult(nan, 0, 0, 0, "above_upper_bound")

    def error(sigma: float) -> float:
        """Model price minus market price: zero at the implied volatility."""
        return bs.price(S, K, r, sigma, T, kind) - price

    # --- step 2: bracket the root ----------------------------------------
    # error(0) <= 0 is guaranteed: the bounds check above ruled out a price
    # below the sigma -> 0 price. Only the upper end has to be searched for.
    # Doubling from 100% reaches the cap in four steps, and the cap itself is
    # tested before giving up, so a genuine 950% volatility is still found.
    lo, hi = 0.0, 1.0
    iterations = 1
    while error(hi) < 0.0:
        if hi >= sigma_max:
            return ImpliedVolResult(nan, iterations, 0, 0, "above_sigma_max")
        lo = hi                    # already known too cheap, so it is a floor
        hi = min(2.0 * hi, sigma_max)
        iterations += 1

    # --- step 3: safeguarded Newton --------------------------------------
    # Brenner-Subrahmanyam as the starting guess: for an at-the-money option
    # the Black-Scholes price is very nearly S*sigma*sqrt(T/(2*pi)), so
    # inverting that linear approximation costs nothing and usually lands
    # within a few vol points. Replaced by the bracket midpoint if it falls
    # outside the bracket, which happens for options far from the money.
    sigma = math.sqrt(2.0 * math.pi / T) * price / S
    if not lo < sigma < hi:
        sigma = 0.5 * (lo + hi)

    vega_floor = VEGA_FLOOR_FRACTION * S * math.sqrt(T) * _PHI_0
    newton_steps = bisection_steps = 0

    # How finely sigma can be resolved at all. The price is computed to within
    # a few units in the last place of its own magnitude, and one vol point is
    # worth ``vega``, so an uncertainty of ``noise`` in the price is an
    # uncertainty of ``noise / vega`` in sigma. Asking for more than that is
    # asking the iteration to chase rounding error, which is exactly what a
    # fixed 1e-12 does to a 950%-vol option: there, vega is 2e-4 against a
    # price of 100, so sigma is only knowable to about 4e-10 whatever the
    # solver does.
    price_noise = 4.0 * sys.float_info.epsilon * max(abs(price), S)

    for _ in range(max_iter):
        iterations += 1
        f = error(sigma)

        # Monotonicity is what makes this safe: too cheap means the answer is
        # above, too expensive means it is below, with no exceptions. ``f == 0``
        # goes to the upper branch, so an exactly flat stretch of the price
        # function collapses to its lower end -- the smallest volatility that
        # reproduces the quote bit for bit.
        if f < 0.0:
            lo = sigma
        else:
            hi = sigma
        if hi - lo <= sigma_tol * max(1.0, hi):
            return ImpliedVolResult(0.5 * (lo + hi), iterations, newton_steps,
                                    bisection_steps, "ok")

        # Newton's step, taken only if vega carries real information and the
        # step stays inside the bracket. Everything else bisects.
        v = bs.vega(S, K, r, sigma, T, kind)
        if v > vega_floor:
            step = sigma - f / v
            resolution = max(sigma_tol * max(1.0, sigma), price_noise / v)
            if abs(step - sigma) <= resolution:
                # Newton's own correction is now smaller than the tolerance,
                # so the answer will not move again. This is the exit that
                # almost every real quote takes. Without it the loop would
                # keep bisecting a bracket whose *other* end is still far
                # away -- the bracket narrows from one side only once Newton
                # has locked on, so waiting for it costs ~25 evaluations per
                # option and sometimes never arrives at all.
                return ImpliedVolResult(step, iterations, newton_steps + 1,
                                        bisection_steps, "ok")
            if lo <= step <= hi:
                sigma = step
                newton_steps += 1
                continue

        sigma = 0.5 * (lo + hi)
        bisection_steps += 1

    # Unreachable in practice: 100 bisections alone shrink any starting
    # bracket by 2^-100. Reported rather than raised, so one pathological
    # quote cannot stop a sweep over a whole option chain.
    return ImpliedVolResult(0.5 * (lo + hi), iterations, newton_steps,
                            bisection_steps, "not_converged")


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: str,
    **kwargs,
) -> float:
    """The volatility that makes Black-Scholes reproduce ``price``.

    ``nan`` when no such volatility exists -- price outside the no-arbitrage
    bounds, ``T <= 0``, or a solution beyond ``SIGMA_MAX``. Never a clipped or
    invented number: see the module docstring.

    >>> from pricing import black_scholes as bs
    >>> quote = bs.call_price(100.0, 100.0, 0.05, 0.20, 1.0)
    >>> round(implied_vol(quote, 100.0, 100.0, 1.0, 0.05, "call"), 12)
    0.2
    """
    return implied_vol_detail(price, S, K, T, r, kind, **kwargs).sigma
