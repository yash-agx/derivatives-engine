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

### Phase 4 — Advanced Monte Carlo ✅
- [x] Antithetic variates — standard error computed over **pairs**, not paths
- [x] Control variates (discounted terminal stock price, known mean exactly `S`)
- [x] Both combined; plus importance sampling, added to attack the Phase 3 coverage failure
- [x] Compare standard vs variance-reduced: accuracy, variance, runtime, convergence rate
- [x] Variance reduction factor and "time-adjusted efficiency" (variance × runtime)

All four live in `src/pricing/variance_reduction.py` with the same signature as plain Monte
Carlo, so they run through the Phase 3 harness unchanged. **Every comparison is at equal path
counts**, because runtime per path is not constant in N.

Experiments: `python analysis/run_phase4.py` (~2.5 min), then `python analysis/plot_phase4.py`.
2,913 tests pass.

#### Findings

**Variance reduction at 100,000 paths** (factor = variance of plain / variance of the method;
above 1 is an improvement):

| option | antithetic | control | both | importance |
|---|---|---|---|---|
| ATM 1y | 1.7x | 6.2x | **28x** | 0.7x |
| OTM 5y | 1.4x | 10.7x | **44x** | 0.8x |
| deep ITM 1y | 9.3x | 111x | **521x** | 0.0x |
| ATM 1y put | 1.5x | 2.3x | **10x** | 1.4x |
| K=160 6m (the Phase 3 failure) | 0.9x | 1.0x | 0.9x | **1,745x** |

Antithetic is the weakest and the cheapest; the control variate is much stronger. Runtime cost is
small (antithetic is actually ~0.75x plain, since it needs half as many random numbers), so the
cost ranking matches the variance ranking almost everywhere.

**Combining them is neither additive nor multiplicative.** Writing `Y` for a path's discounted
payoff and `X` for its discounted terminal stock price:

| method | factor | correlation it uses |
|---|---|---|
| antithetic | `1 / (1 + rho_A)` | `rho_A = Corr(Y(Z), Y(-Z))` |
| control | `1 / (1 - rho_XY^2)` | `rho_XY = Corr(Y, X)` |
| both | `1 / [(1 + rho_A)(1 - rho_AB^2)]` | `rho_AB = Corr(pair mean of Y, pair mean of X)` |

So the combined factor is the antithetic factor times a control factor **recomputed on the pair
averages**, and the ratio to the naive product is `(1 - rho_XY^2) / (1 - rho_AB^2)`. Measured on a
4,000,000-path sample, this predicts the observed ratio exactly (2.110 predicted against 2.11
measured for ATM 1m, 6.697 against 6.70 for OTM 1y).

Which way it goes depends on whether pairing *helps* the control:
- **Usually it helps a lot.** Antithetic averaging removes the odd part of both the payoff and the
  control, and what is left correlates more tightly: `rho_XY = 0.924 -> rho_AB = 0.964` for the
  ATM one-year call, so the combination comes in at **2.08x the product** (28.7x against a product
  of 13.8x). Out of the money the lift is larger still (OTM 1y: 0.840 -> 0.978, **6.7x** the
  product).
- **It hurts when the control was already almost perfect.** Deep in the money the payoff is nearly
  an exact linear function of the control (`rho_XY = 0.995`), and pairing throws away the very
  component that made it work (`rho_AB = 0.990`): the combination is **0.46x the product** (536x
  against 1171x). The one-month deep-ITM case is the extreme: a product of 4.0e9 against an actual
  1.3e7.

The earlier phrasing here ("multiplies rather than adds") was wrong in both directions, and the
measured table is what corrected it: 1.7 x 6.2 = 10.5 but both = 28, while 9.3 x 111 = 1,032 but
both = 521.

**Every error bar stayed honest** where the estimator itself was sound: reported standard error
over actual spread stayed near 1 for all four methods. The antithetic SE was checked hardest
because it is the easy one to get wrong — computed over pairs it matches the true standard error
to **1.0002**; computed over paths it would be wrong in *both* directions (too small for a call,
where the legs are negatively correlated, too large for a straddle, where they are positively
correlated).

**Where each method fails** — the more useful half of the results:

- **Antithetic on a symmetric payoff.** A straddle pays `|S_T - K|`, which is not monotone in
  the draw, so the two legs of a pair move together and pairing concentrates variance instead of
  cancelling it. Sweeping the strike: 10.5x at K=70, 2.1x at K=85, **0.60x at K=100**, 0.78x at
  K=115, 3.2x at K=130. The worst point brackets the *median terminal price* (103.05), not the
  spot — which is exactly where the reflected payoff is most nearly even.
- **Importance sampling on anything that did not need it.** At the money it costs 30% (0.7x); in
  the money it is far worse. The shift solves a problem those options do not have, and the
  likelihood-ratio weights add variance of their own.
- **The control variate succeeding too well.** Deep in the money (S=130, K=100, one month) the
  payoff is an *exact* affine function of the control, so the residual is a constant and variance
  falls by 14 million times. But the option is the forward minus a put worth 2.6e-6, and that put
  only appears on paths finishing below the strike (probability 2.2e-6). At 100,000 paths **83%
  of runs contain no such path**, report a standard error below 1e-12, and miss. Coverage: 0.158,
  against 0.952 for plain Monte Carlo on the same option. **Variance reduction manufactured the
  Phase 3 rare-event failure**: it removed all the variance it could see and left a pure
  rare-event problem behind.

**The control coefficient must not be fitted on the paths it is used on** — at small N. Measured
over 20,000 seeds, the same-sample bias is **-0.147 (-25 standard errors)** at N=50, **-0.038
(-13 se)** at N=200 and **-0.0058 (-4.7 se)** at N=1,000, falling like 1/N as theory says. A
pilot sample removes it completely (every pilot run within 1.4 se of zero) at the cost of 7-35%
more spread. At the N=10,000-100,000 used for the comparisons the bias is under ~1% of a standard
deviation, which is why the default fits on the same paths.

**The key experiment: only importance sampling fixes broken coverage.** Antithetic, control and
both together leave it exactly where it was, because they shrink the spread of an estimator whose
*shape* is the problem:

| strike (6m) | plain | antithetic | control | both | importance |
|---|---|---|---|---|---|
| K=140 | 0.876 | 0.886 | 0.876 | 0.878 | **0.962** |
| K=160 | 0.404 | 0.426 | 0.404 | 0.418 | **0.940** |
| K=180 | 0.026 | 0.024 | 0.026 | 0.024 | **0.942** |
| K=200 | 0.000 | 0.000 | 0.000 | 0.000 | **0.934** |

(coverage at N=1,000, 500 seeds). Importance sampling also restores the fully degenerate Phase 3
case — the deep-OTM one-month call where every single path pays zero — from coverage 0.000 to
0.940. It is the only method here that changes *where the paths go* rather than how they are
averaged, and the rare-event failure is a problem of where the paths go.

#### Limitations
- The control variate is the discounted stock, which correlates well with a vanilla call and
  badly with anything far out of the money (1.0x at K=160). A better control for those would be a
  closed-form option at a nearer strike; not tried.
- The importance-sampling shift puts the strike at the median, which is the explainable choice,
  not the variance-optimal one. A larger shift is known to be better for deep out-of-the-money
  options, and the optimal shift is not computed here.
- Importance sampling needs to know *where* the payoff lives. That is obvious for a vanilla call
  and not obvious in general, which is the practical catch: the method that fixes rare events
  requires already understanding the rare event.
- Coverage is measured over 500 seeds, resolving it only to ±0.019 — and **all 16 options share
  the same seed set**, so an unlucky set moves them together rather than independently. Antithetic
  coverage read ~0.92 across the whole grid at 500 seeds, which looked systematic; at 5,000 seeds
  it is 0.944-0.948, i.e. nominal. Treat any single 500-seed coverage figure within ~0.03 of 0.95
  as noise.
- Variance reduction factors above about 1,000 should be read with the coverage column beside
  them — see the deep-in-the-money control variate above.
- Still only GBM with constant volatility and no dividends, and only European payoffs.

### Phase 5 — Performance engineering ✅
- [x] Re-implement the Monte Carlo core in C++ — `cpp/mc_engine.cpp`, all four Phase 4 methods
- [x] Benchmark: pure Python loop vs NumPy vectorised vs C++ — `results/perf_*.csv`
- [x] Multithreading in C++, with reproducibility guaranteed
- [x] Results tables and charts

Build and run:
```
cmake -S cpp -B cpp/build -G "MinGW Makefiles"
cmake --build cpp/build
python analysis\run_phase5.py          # ~20 s
python analysis\plot_phase5.py
```
2,950 tests pass, 35 of them exercising the C++ engine.

#### How Python calls the C++ (and why this way)

A **C-ABI DLL loaded with `ctypes`**, not a Python extension module. This machine has
MSVC-built CPython and WinLibs g++; an extension module (pybind11, Cython, a raw
`PyMODINIT_FUNC`) links against CPython's own runtime and object layout, so mixing a
MinGW-built one with an MSVC CPython risks two different C runtimes allocating and freeing
each other's memory. The C ABI is the one interface both compilers agree on exactly — it is how
Windows itself is called. Nothing of CPython crosses the boundary, `ctypes` ships with Python,
and the boundary only has to carry seven doubles in and two out. Measured call overhead is
**~30 µs**, of which 27 µs was scipy recomputing the same normal critical value on every run
until that was cached.

The DLL links `-static`: without it, it imports `libwinpthread-1.dll` from the compiler's own
bin directory and Python fails to load it with an error that names the DLL rather than the
missing dependency.

#### Threads: independent streams and reproducibility

The generator is **counter-based**: the draws for path `i` are a pure function of
`(seed, i)` via a SplitMix64 mix, with Box-Muller on top. Two consequences, both of which the
tests check:

- **Streams cannot overlap.** Each path index is handed out exactly once, so no two threads can
  ever draw the same numbers. This is a structural guarantee, not a probabilistic one — unlike
  splitting a sequential generator and hoping the substreams do not collide.
- **The answer does not depend on the thread count.** Because the work is also divided into a
  **fixed number of chunks regardless of thread count**, the floating-point additions happen in
  the same order every time. Measured: 1, 2, 3, 4, 8 and 16 threads return a **bitwise identical**
  price and standard error. Without the fixed chunking the result would wobble in the last bits
  as threads changed — still correct, but not reproducible.

#### Findings

**Throughput, European call, fastest of interleaved repeats** (million paths/second):

| N | python loop | numpy | c++ 1 thread | c++ 24 threads |
|---|---|---|---|---|
| 10^3 | 2.85 | 5.77 | **15.1** | 0.27 |
| 10^4 | 3.07 | **60.1** | 36.9 | 3.73 |
| 10^5 | 2.94 | **51.8** | 38.6 | 32.0 |
| 10^6 | 2.92 | 43.4 | 38.7 | **185.7** |
| 10^7 | — | 39.4 | 38.5 | **324.1** |

**Where each speedup comes from:**

- **Interpreter → NumPy: 15–20x.** The pure Python loop costs ~340 ns per path — roughly a
  thousand cycles for about twenty floating-point operations. Almost all of it is interpreter
  overhead: boxing floats, dispatching method calls, stepping bytecode. NumPy moves the loop
  into compiled, SIMD code and the per-path cost drops to 16.6 ns.
- **NumPy → single-threaded C++: nothing at all (0.98x at 10^7).** This was the surprise, and it
  is the honest headline: NumPy's inner loops are *already* compiled and vectorised, so a scalar
  C++ loop has nothing to win on arithmetic. It processes one double at a time where NumPy does
  four or eight. What C++ wins back is memory and call overhead; the two roughly cancel.
- **Memory: the Phase 3 U-shape is a NumPy property, not a hardware one.** NumPy's per-path cost
  is U-shaped — 173 ns at 10^3 (call overhead), 16.6 ns at 10^4, rising to 25.4 ns at 10^7 as its
  temporary arrays leave cache. The C++ core keeps **no arrays at all** (Welford accumulation,
  O(1) memory), and its per-path cost is **flat at 25.9 ns from 10^4 to 10^7**. It still has the
  left half of the curve (66 ns at 10^3) from fixed per-call cost. So the rising half of Phase 3's
  U-shape was the array traffic, and streaming removes it.
- **Threads: 7.3x, saturating at 8.** Speedup is 1.99x / 3.95x / 7.34x at 2 / 4 / 8 threads
  (99.6% / 98.8% / 91.8% efficiency), then **flat**: 16 threads gives 7.27x and 24 gives 7.04x.
  This CPU (i7-14650HX) has 8 performance cores plus 8 efficiency cores and 24 logical threads,
  so the eight P-cores are what this floating-point-bound loop can use.
- **Threads are a liability on small batches.** At N=10^3, 24 threads costs **3,726 ns per path**
  against 66 ns single-threaded — creating and joining the threads takes ~3.7 ms, hundreds of
  times the actual work. The crossover is around N=10^5.

**Cost (variance × runtime, lower is better), at 10^7 paths:** numpy 5.50e-6, c++ 1 thread
5.62e-6, c++ 24 threads **6.68e-7** — 8.2x cheaper per unit of accuracy than NumPy. Since the
variance per path is identical across implementations, this is purely the speed showing up in
the metric that Phases 3 and 4 used.

**All four estimators, Python against C++, at 10^6 paths** (million paths/second):

| method | python | c++ 1 thread | c++ 24 threads |
|---|---|---|---|
| plain | 30.4 | 21.0 (0.69x) | 113.4 (**3.73x**) |
| antithetic | 37.1 | 36.1 (0.97x) | 155.7 (**4.19x**) |
| control | 16.7 | 20.3 (**1.21x**) | 109.1 (**6.52x**) |
| importance | 20.2 | 18.9 (0.94x) | 103.4 (**5.13x**) |

The control variate is where single-threaded C++ actually wins, because the Python version makes
several passes over several temporary arrays to get a covariance while the C++ accumulates it in
one pass with no memory at all.

**And they are still statistically correct.** The C++ estimators run through the Phase 3 harness
unchanged: unbiased on 300 seeds, reported standard error matching the real spread, coverage in
range, antithetic standard errors computed over pairs, and importance sampling restoring the
Phase 4 coverage rescue (0.87 → 0.95 on the K=160 call). The control-variate call and put satisfy
put-call parity exactly, as in Python.

#### Limitations
- **Scalar C++ against vectorised NumPy is not a like-for-like language comparison.** The C++
  core does not use SIMD intrinsics; NumPy does. `-march=native` changed nothing measurable
  (25.2 vs 26.2 ns/path) because the loop is dominated by `exp`/`log`/`cos`/`sin`, which do not
  auto-vectorise with the accumulation dependency present. A hand-vectorised C++ core, or one
  using a vector math library, would be the real comparison and was not written.
- Box-Muller costs two transcendentals per pair of normals; NumPy's ziggurat is cheaper per
  draw. Some of the single-threaded parity is C++ doing more work per normal, not the language.
- The machine is noisy: median and fastest timings differed by up to 1.9x, and running all the
  repeats of one implementation together (rather than interleaving) cost a spurious factor of
  1.9 on a first pass. Headline numbers use the fastest repeat; medians and spreads are in the
  CSVs. Nothing here is pinned and turbo is on.
- Thread saturation at 8 is a property of this CPU's core layout, not a general result.
- `n_chunks` is part of the reproducibility contract: changing it changes the summation order
  (though not the answer beyond rounding). It defaults to 64.
- SplitMix64 + Box-Muller is a sound counter-based generator, but it has not been run through a
  statistical test battery here; the evidence for it is the harness's bias, standard-error and
  coverage checks, which are sensitive but not a substitute for TestU01.
- Only one option and one machine; no attempt at GPU, SIMD intrinsics, or a thread pool that
  would amortise thread creation at small N.

### Write-up
- [ ] Method, results, analysis, limitations (e.g. Black-Scholes assumptions: constant volatility, no dividends)

## Rules
- Every pricing function gets tests (known textbook values, put-call parity, convergence checks).
- Fix random seeds so every result is reproducible.
- Save all experiment results to CSV, and generate plots from the CSVs with a script.
- Do not add Co-Authored-By or Claude-Session lines to commit messages.
