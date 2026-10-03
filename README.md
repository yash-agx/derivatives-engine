# Computational Derivatives Pricing & Risk Engine

**How do numerical and variance-reduction techniques affect the accuracy and computational
efficiency of Monte Carlo option pricing?**

A computational research project rather than a calculator: price the same European option by
closed form, by binomial tree and by Monte Carlo, then measure how accuracy and runtime respond
to the numerical choices. Python (NumPy, SciPy, matplotlib, pytest) for Phases 1–4; C++17 for the
performance work in Phase 5.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Black-Scholes prices and the five Greeks | in progress |
| 2 | Binomial tree (CRR), European and American; Monte Carlo under GBM | not started |
| 3 | Monte Carlo convergence: error, standard error, runtime, the 1/√N law | not started |
| 4 | Antithetic and control variates; variance reduction and time-adjusted efficiency | not started |
| 5 | C++ Monte Carlo core; Python vs NumPy vs C++ benchmarks | not started |

Phase 1 so far: `call_price`, `put_price` and `delta`/`gamma`/`vega`/`theta`/`rho` for both option
types, validated against Hull's worked examples, put-call parity, and finite differences.

## Layout

```
src/pricing/black_scholes.py   closed-form prices and Greeks (docstrings explain the maths)
tests/                         pytest suite: textbook values, parity, finite differences, edge cases
requirements.txt               numpy, scipy, matplotlib, pytest
pytest.ini                     puts src/ on the import path
```

## Running

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

Conventions: rates and volatilities are decimals per year (5% → `0.05`), `T` is in years, vega is
per 1.0 of volatility and theta is per year. No dividends, European exercise only — the
Black-Scholes assumptions the write-up has to discuss.
