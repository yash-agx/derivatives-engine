# Computational Derivatives Pricing & Risk Engine

A computational research/engineering project — not just "enter stock price → get option price".

Research question for the final write-up:
**How do numerical and variance-reduction techniques affect the accuracy and computational efficiency of Monte Carlo option pricing?**

## About me (Yash)
- IB DP2 student (Math AA HL, Physics HL, CS HL), aspiring quant developer. Comfortable with Python; **new to C++**.
- I need to be able to explain every formula and line of code in an interview, so:
  - Work one phase at a time, one feature at a time. Don't jump ahead.
  - Explain the maths intuitively first (what the formula *means* and why it has that shape), then the code.
  - After writing each piece, say what it does and *why* it's built that way, in simple language.
  - Point out the concepts I'm meeting for the first time (risk-neutral pricing, variance reduction, and in
    Phase 5 the C++ ones: templates, references, RAII, etc.).
  - Keep explanations brief (what it does + why). Don't stop to quiz me after each feature — I'll ask if I
    want to be tested.

## Stack
- Python first (NumPy, SciPy, matplotlib, pytest) for Phases 1–4.
- C++17 for the performance-critical parts in Phase 5.

## Phases

### Phase 1 — Analytical pricing ✅
- [x] Black-Scholes price for European call and put — `src/pricing/black_scholes.py`
- [x] Greeks: Delta, Gamma, Vega, Theta, Rho (both option types, plus a `greeks()` dict)
- [x] Check put-call parity holds; check Greeks against finite differences

2,669 tests pass (`.\.venv\Scripts\python.exe -m pytest`):
- `tests/test_prices.py` — Hull's worked examples, parity over a 108-point grid, no-arbitrage
  bounds, monotonicity in S, sigma and T
- `tests/test_greeks.py` — Hull's Greeks example, every Greek against a central finite difference,
  the five Greek parity relations, signs and ranges, where gamma and vega peak
- `tests/test_edge_cases.py` — T = 0 and sigma = 0, one-second and 100-year expiries, vol from
  1e-8 to 10, moneyness from 1e-6 to 1e6, and invalid inputs

### Phase 2 — Numerical pricing ✅
- [x] Binomial tree (Cox-Ross-Rubinstein), European and American options — `src/pricing/binomial.py`
- [x] Show binomial price converging to Black-Scholes as steps increase (plot) —
      `results/binomial_convergence.png`, `results/binomial_oscillation.png`
- [x] Monte Carlo pricing using geometric Brownian motion — `src/pricing/monte_carlo.py`
      (exact terminal-price draw, with standard error and 95% confidence interval)

2,825 tests pass (117 for the tree, 39 for the simulation). Experiments: `python analysis/run_phase2.py` (~25 s, writes `results/*.csv`),
then `python analysis/plot_phase2.py` (tables and `results/*.png`).

Key numbers to carry into Phase 3:
- CRR error is O(1/N): at the money `error × N` is constant to 1% (1.995 → 2.000 from N=50 to
  3200), and the measured log-log slope is −1.00.
- Odd/even oscillation: at the money, even N undershoots Black-Scholes and odd N overshoots,
  every time. Averaging an N/N+1 pair leaves 7.1% of the better single error — a free accuracy
  gain worth remembering.
- Away from the money the sign flips in slow groups instead of alternating, because the strike
  drifts between lattice nodes as N changes.
- American call premium is exactly 0 everywhere (no dividends). American put premium is 9.41 at
  S=60 (K=100, r=10%) and exactly 0 at every spot when r=0 — the premium is interest on the
  strike, not optionality.
- Monte Carlo 95% CI coverage is 94.4–96.0% for a normal option, but only 75.2% at N=5,000 for a
  deep out-of-the-money call (rare payoffs, CLT not yet valid), recovering to 93.2% at N=100,000.
  A reported standard error is only as good as the normal approximation behind it.

### Phase 3 — Monte Carlo investigation ✅
- [x] Run 1,000 → 10,000 → 100,000 → 1,000,000 simulations
- [x] Measure error vs Black-Scholes, runtime, standard error, confidence intervals
- [x] Show the error shrinks roughly like 1/√N (log-log plot) — `results/mc_convergence.png`
- [x] Repeat runs with different seeds to measure variance properly — 500 seeds per point

Built as a reusable harness (`src/experiments/harness.py`), not a script: it takes any
`estimator(option, n_paths, seed) -> MonteCarloResult`, so Phase 4's variance-reduced estimators
run through identical measurement code and the comparison is like for like by construction.

Experiments: `python analysis/run_phase3.py` (~9 min, writes `results/mc_*.csv`), then
`python analysis/plot_phase3.py` (tables and `results/mc_*.png`). 2,856 tests pass.

#### Findings

**The 1/√N law holds exactly.** Fitted log-log slope of RMSE against paths is −0.496 for the base
case and −0.490 to −0.501 across all 16 options; `RMSE × √N` is flat. A 1,000× increase in paths
cuts the error by 30.5× (theory: √1000 = 31.6). The law does not care what the option is — only
the constant in front changes.

**The reported standard error is honest — where the estimator works at all.** Across all 64
grid points, the mean reported standard error matches the actual spread of 500 estimates to
within 0.966–1.038, over four orders of magnitude of error size. Bias is negligible
(−0.0007 at N=1,000,000 on a price of 10.45). So the standard error is a usable live estimate
of an accuracy you otherwise could not see without knowing the answer.

**Coverage does not break with moneyness or maturity.** This contradicted my expectation. Over
the whole 5 moneyness × 3 maturity grid, coverage stayed at 0.93–0.96 against a nominal 95%
(±0.019 is sampling noise at 500 seeds) — including a deep out-of-the-money call with payoff
skewness 8.71. Being far out of the money is *not* by itself enough to break the interval.

**What does break it is a payoff that is a rare event.** A dedicated sweep (strike pushed from
120 to 200 at six months) locates the failure precisely, and the governing quantity turns out to
be the expected number of paths that pay anything at all:

| expected paying paths | 0.03 | 0.65 | 3.2 | 13 | 57 | 118 | 1,145 | 236,016 |
|---|---|---|---|---|---|---|---|---|
| coverage | 0.026 | 0.404 | 0.752 | 0.874 | 0.928 | 0.948 | 0.952 | 0.952 |

Both experiments collapse onto this one curve regardless of strike or path count. The practical
rule: **below ~50 paying paths per run the interval is not trustworthy, and below ~5 it is
worthless.** In `|skewness|/√N` terms the threshold is around 0.3, not the 0.1 I assumed from the
Berry-Esseen rule of thumb — the rule is roughly 3× too conservative here.

**The mechanism, seen directly** (`results/mc_sampling_distribution.png`): standardising the
estimates by their own reported standard error should give a standard normal. For the K=160
six-month call at N=1,000 it gives a spike just below +1 with a long left tail (skew −10.98,
coverage 40.4%). The spike is the "exactly one paying path" signature: with a single payoff `X`
among `N` paths, the estimate is `X/N` and the standard error is also ≈ `X/N`, so their ratio
pins near +1 whatever `X` was. The estimator is most confident exactly when it has least
information. By N=100,000 the same option is back to skew −0.62 and 93.2% coverage.

**Cost.** A 95% error bar of ±1 cent needs 8.3M paths (0.23 s) for the at-the-money one-year
call, and 130M paths (3.6 s) for the deep-in-the-money five-year. One extra decimal place costs
100× the paths, which is the whole case for Phase 4.

**Efficiency gets worse at large N, and the reason is the machine.** `variance × runtime` is a
**cost**: lower is better, because it is the price of a unit of accuracy. In theory it is flat in
N (variance falls like 1/N, runtime rises like N). Measured, the variance half is exact but
runtime per path is U-shaped: 189 ns at N=1,000 (dominated by fixed per-call overhead), 27 ns at
N=100,000, 33 ns at N=1,000,000 (arrays no longer fitting in cache). So the cost **rises** from
3.2e-6 at N=100,000 to 7.7e-6 at N=400,000 before flattening — i.e. a big batch buys *less*
accuracy per second than a medium one. The cheapest accuracy on this machine is around
N=100,000. **Consequence for Phase 4: compare estimators at the same path count.** Comparing
across path counts measures the memory system as much as the method.

#### Limitations
- Measured against Black-Scholes, which exists only because these are vanilla European options.
  The whole point of Monte Carlo is problems with no closed form, where none of these errors
  could be measured this way — only the self-reported standard error would be available, which
  is exactly why establishing when it can be trusted matters.
- One machine, one process, no pinning. Runtimes move with machine state; the U-shape was stable
  across runs but the absolute nanosecond figures are not portable.
- The same 500 seeds are used at every path count, so results across path counts are correlated
  (seed 7 starts the same generator each time). Fine for each point on its own, but the curves
  are not 64 independent experiments.
- 500 seeds gives coverage to ±0.019, so nothing finer than about 2 percentage points can be
  resolved. The flagged failures are far outside that; the 0.93-vs-0.95 wobbles inside the grid
  are not.
- Only GBM with constant volatility and no dividends — the Black-Scholes world. Real payoff
  distributions are fatter-tailed, so the paths-needed figures here are optimistic.
- The rare-event failure is a property of crude Monte Carlo, not of simulation in general.
  Importance sampling targets exactly this case and is not in this project's scope.

### Phase 4 — Advanced Monte Carlo
- [ ] Antithetic variates
- [ ] Control variates (e.g. underlying asset as control)
- [ ] Compare standard vs variance-reduced: accuracy, variance, runtime, convergence rate
- [ ] Variance reduction factor and "time-adjusted efficiency" (variance × runtime)

### Phase 5 — Performance engineering
- [ ] Re-implement the Monte Carlo core in C++
- [ ] Benchmark: pure Python loop vs NumPy vectorised vs C++
- [ ] Optional: multithreading in C++
- [ ] Results tables and charts

### Write-up
- [ ] Method, results, analysis, limitations (e.g. Black-Scholes assumptions: constant volatility, no dividends)

## Rules
- Every pricing function gets tests (known textbook values, put-call parity, convergence checks).
- Fix random seeds so every result is reproducible.
- Save all experiment results to CSV, and generate plots from the CSVs with a script.
- Do not add Co-Authored-By or Claude-Session lines to commit messages.
