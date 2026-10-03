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

### Phase 3 — Monte Carlo investigation
- [ ] Run 1,000 → 10,000 → 100,000 → 1,000,000 simulations
- [ ] Measure error vs Black-Scholes, runtime, standard error, confidence intervals
- [ ] Show the error shrinks roughly like 1/√N (log-log plot)
- [ ] Repeat runs with different seeds to measure variance properly

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
