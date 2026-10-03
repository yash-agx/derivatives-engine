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

### Phase 1 — Analytical pricing
- [ ] Black-Scholes price for European call and put
- [ ] Greeks: Delta, Gamma, Vega, Theta, Rho
- [ ] Check put-call parity holds; check Greeks against finite differences

### Phase 2 — Numerical pricing
- [ ] Binomial tree (Cox-Ross-Rubinstein), European and American options
- [ ] Show binomial price converging to Black-Scholes as steps increase (plot)
- [ ] Monte Carlo pricing using geometric Brownian motion

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
