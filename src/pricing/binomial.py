"""Cox-Ross-Rubinstein binomial tree: European and American options.

How the tree works, in words
----------------------------
Chop the option's life into ``N`` short steps. In each step the stock is
allowed to do exactly two things: multiply by ``u`` (up) or by ``d`` (down).
Repeat that ``N`` times and you get a lattice of possible prices -- a tree
that fans out from today's spot.

Two questions have to be answered to make this a pricing model.

**1. How big are the up and down moves?** CRR picks::

    u = exp(sigma * sqrt(dt))      d = 1 / u        dt = T / N

So one step up is one standard deviation of the log return over that step.
Setting ``d = 1/u`` makes the tree *recombine*: up-then-down returns to the
same price as down-then-up, so after ``N`` steps there are only ``N+1``
distinct prices instead of ``2**N`` paths. That is what makes the whole thing
computable. It is also why the tree converges to Black-Scholes: the binomial
distribution of ``N`` up/down moves tends to a normal distribution for the
log price, which is exactly Black-Scholes' assumption.

**2. What is the probability of an up move?** Not the real-world one -- the
*risk-neutral* one, the probability under which the stock's expected growth is
the risk-free rate::

    p * u + (1 - p) * d = exp(r * dt)   =>   p = (exp(r*dt) - d) / (u - d)

This is the pricing trick of the whole subject. We are not forecasting the
stock; we are asking what the option must cost so that nobody can build a
risk-free money machine out of it and the stock. The answer involves ``r``
only, and never the stock's real expected return.

Then price it by **backward induction**. At expiry the option is worth its
payoff, which we know. Step back one layer: the value at a node is the
discounted average of the two values it leads to::

    V = exp(-r*dt) * (p * V_up + (1-p) * V_down)

Repeat until you reach today's single node. That number is the price.

Early exercise and American options
-----------------------------------
A European option can only be exercised at expiry. An American option can be
exercised at *any* time, so at every node the holder has a choice:

* **continue** holding, worth ``exp(-r*dt) * (p*V_up + (1-p)*V_down)``, or
* **exercise now**, worth the payoff ``max(S - K, 0)`` / ``max(K - S, 0)``.

A rational holder takes whichever is larger, so the American rule is one
``max`` added to the backward induction::

    V = max(payoff(S_node),  exp(-r*dt) * (p*V_up + (1-p)*V_down))

This is why the tree is worth building even though Black-Scholes exists:
American options have no simple closed-form price, and the tree handles them
with one extra line, because backward induction visits every node at every
time and can ask the question there.

When is exercising early actually optimal?

* **Calls, no dividends: never.** Exercising a call early means paying ``K``
  sooner than you have to and throwing away the insurance that the stock might
  fall below ``K``. Both are losses, so the ``max`` above never bites and the
  American call equals the European call. (With dividends it can bite -- a
  dividend you only receive if you own the share can be worth more than the
  interest and the insurance. This project has no dividends.)
* **Puts: sometimes.** Exercising a put early means receiving ``K`` sooner,
  and interest on ``K`` is a real gain. Deep in the money there is little
  insurance value left to give up (the stock is unlikely to climb back above
  ``K``), so the interest wins and exercising is correct. The American put is
  therefore worth strictly more than the European put whenever rates are
  positive and the option is far enough in the money.
"""

from __future__ import annotations

import math

import numpy as np

from .common import CALL, PUT, check_kind, payoff, validate_inputs

__all__ = [
    "crr_parameters",
    "binomial_price",
    "european_binomial",
    "american_binomial",
    "early_exercise_premium",
]


def crr_parameters(r: float, sigma: float, T: float, steps: int) -> dict:
    """The four numbers that define a CRR tree, for inspection and testing.

    Returns ``dt``, ``u``, ``d``, ``p`` and the one-step discount factor.

    ``p`` must land strictly inside ``(0, 1)`` to be a probability. It fails
    when the step is so long that the risk-free growth ``exp(r*dt)`` falls
    outside the range ``[d, u]`` the tree can reach -- the tree would then
    imply an arbitrage. The cure is more steps (smaller ``dt``), and the
    condition is roughly ``sigma * sqrt(dt) > |r| * dt``.
    """
    dt = T / steps
    up_move = math.exp(sigma * math.sqrt(dt))
    down_move = 1.0 / up_move
    growth = math.exp(r * dt)
    p = (growth - down_move) / (up_move - down_move)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p = {p:.6g} is not in (0, 1): the step "
            f"dt = {dt:.6g} is too long for sigma = {sigma:.6g} and r = {r:.6g}. "
            f"Use more steps."
        )
    return {
        "dt": dt,
        "u": up_move,
        "d": down_move,
        "p": p,
        "discount": math.exp(-r * dt),
    }


def binomial_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    kind: str,
    american: bool = False,
) -> float:
    """Price a European or American option on a CRR tree with ``steps`` steps.

    The implementation is the backward induction described in the module
    docstring, done one whole layer at a time with NumPy rather than node by
    node: ``values`` holds every node value at the current time step, and each
    iteration collapses it to the layer before.
    """
    validate_inputs(S, K, r, sigma, T)
    check_kind(kind)
    if int(steps) != steps or steps < 1:
        raise ValueError(f"steps must be a positive whole number, got {steps!r}")
    steps = int(steps)
    if T == 0:
        return float(payoff(S, K, kind))
    if sigma <= 0:
        raise ValueError(
            "sigma must be > 0 to build a tree (u and d would both be 1). "
            "The zero-volatility case is a certain payoff: use black_scholes."
        )

    params = crr_parameters(r, sigma, T, steps)
    p, discount = params["p"], params["discount"]
    log_u = sigma * math.sqrt(params["dt"])

    # Terminal layer. Node j of step i has j down-moves and (i-j) up-moves, so
    # its price is S * u**(i-j) * d**j = S * exp(log_u * (i - 2j)).
    down_moves = np.arange(steps + 1)
    spots = S * np.exp(log_u * (steps - 2 * down_moves))
    values = payoff(spots, K, kind)

    for step in range(steps - 1, -1, -1):
        # Discounted risk-neutral average of the two children.
        values = discount * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            # The holder's choice at every node of this layer.
            spots = S * np.exp(log_u * (step - 2 * np.arange(step + 1)))
            values = np.maximum(values, payoff(spots, K, kind))

    return float(values[0])


def european_binomial(S, K, r, sigma, T, steps, kind) -> float:
    """``binomial_price`` with early exercise switched off."""
    return binomial_price(S, K, r, sigma, T, steps, kind, american=False)


def american_binomial(S, K, r, sigma, T, steps, kind) -> float:
    """``binomial_price`` with early exercise allowed at every node."""
    return binomial_price(S, K, r, sigma, T, steps, kind, american=True)


def early_exercise_premium(S, K, r, sigma, T, steps, kind) -> float:
    """How much the right to exercise early is worth, on the same tree.

    Comparing American and European prices from the *same* tree is the honest
    way to measure this: the discretisation error is common to both and
    cancels, so what is left is the value of the early-exercise right itself.
    """
    american = binomial_price(S, K, r, sigma, T, steps, kind, american=True)
    european = binomial_price(S, K, r, sigma, T, steps, kind, american=False)
    return american - european
