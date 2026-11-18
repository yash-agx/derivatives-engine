"""Run the Phase 7 experiments and write the CSVs in results/.

    python analysis/run_phase7.py          (about 17 minutes)

Every setting of the validation run is fixed in advance in
docs/phase7_preregistration.md; this script implements exactly that. Fixed
seeds throughout, so re-running rewrites the same numbers.

Six experiments:

1. ``lsm_validation.csv`` -- the twenty American puts of Longstaff & Schwartz
   (2001) Table 1: LSM out-of-sample and in-sample prices against the American
   tree and against a Bermudan tree on the same exercise dates.
2. ``lsm_paths.csv`` -- error against path count, 100 seeds per point.
3. ``lsm_dates.csv`` -- error against the number of exercise dates, with the
   Bermudan tree for each date count.
4. ``lsm_basis.csv`` -- how much the regression basis matters.
5. ``lsm_boundary_tree.csv``, ``lsm_boundary_lsm.csv`` -- the critical stock
   price over time, from the tree and from 20 independently fitted LSM rules.
6. ``lsm_call.csv`` -- an American call on a non-dividend stock, where early
   exercise is never optimal: how often the fitted rule exercises anyway, and
   what that costs, measured on the same paths as the European payoff.
"""

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # the package is not installed; see pytest.ini

from pricing.binomial import american_binomial                        # noqa: E402
from pricing.black_scholes import price as black_scholes              # noqa: E402
from pricing.common import payoff                                     # noqa: E402
from pricing.exercise_tree import (bermudan_binomial,                 # noqa: E402
                                   tree_exercise_boundary)
from pricing.lsm import (apply_exercise_rule, lsm_exercise_boundary,  # noqa: E402
                         lsm_price, simulate_paths)

RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Settings fixed by the pre-registration
# ---------------------------------------------------------------------------
K, R = 40.0, 0.06
CASES = [(S, sigma, T)
         for S in (36.0, 38.0, 40.0, 42.0, 44.0)
         for sigma in (0.20, 0.40)
         for T in (1.0, 2.0)]
DATES_PER_YEAR = 50
DEGREE = 3
N_TRAIN = N_PRICE = 100_000
CASE_SEED = 7000                  # case i uses seed 7000 + i
TREE_STEPS = 10_000               # reference; uncertainty from doubling it
Z_95 = 1.959963984540054

# ---------------------------------------------------------------------------
# Convergence experiments, on case 0 (S=36, sigma=0.2, T=1: the largest
# early-exercise premium of the sigma=0.2 cases)
# ---------------------------------------------------------------------------
BASE = (36.0, 0.20, 1.0)
PATH_COUNTS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
PATH_SEEDS = range(10_000, 10_100)          # 100 seeds
DATE_COUNTS = [1, 2, 5, 10, 25, 50, 100, 200]
DATE_SEEDS = range(20_000, 20_020)          # 20 seeds
BASES = [(d, False) for d in range(0, 7)] + [(3, True)]   # (degree, weighted)
BASIS_SEEDS = range(30_000, 30_020)         # 20 seeds
BOUNDARY_CASES = {"case0": (36.0, 0.20, 1.0), "case2": (36.0, 0.40, 1.0)}
BOUNDARY_TREE_STEPS = 5_000
BOUNDARY_SEEDS = range(40_000, 40_020)      # 20 seeds
CALL_SPOTS = [36.0, 40.0, 44.0, 50.0]
CALL_SEEDS = range(50_000, 50_020)


def write_csv(name: str, rows: list) -> None:
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def n_dates_for(T: float) -> int:
    return int(round(DATES_PER_YEAR * T))


def bermudan_reference(S, sigma, T, n_dates, total_steps=TREE_STEPS):
    return bermudan_binomial(S, K, R, sigma, T, n_dates, total_steps // n_dates, "put")


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------

def run_validation() -> None:
    print("\n1. Validation on Longstaff & Schwartz Table 1 (20 cases)")
    rows = []
    for i, (S, sigma, T) in enumerate(CASES):
        n_dates = n_dates_for(T)
        american = american_binomial(S, K, R, sigma, T, TREE_STEPS, "put")
        american_2x = american_binomial(S, K, R, sigma, T, 2 * TREE_STEPS, "put")
        bermudan = bermudan_reference(S, sigma, T, n_dates)
        bermudan_2x = bermudan_reference(S, sigma, T, n_dates, 2 * TREE_STEPS)
        european = black_scholes(S, K, R, sigma, T, "put")

        start = time.perf_counter()
        res = lsm_price(S, K, R, sigma, T, "put", n_paths=N_TRAIN, n_price_paths=N_PRICE,
                        n_dates=n_dates, degree=DEGREE, seed=CASE_SEED + i)
        runtime = time.perf_counter() - start

        rows.append({
            "case": i, "S": S, "sigma": sigma, "T": T, "n_dates": n_dates,
            "seed": CASE_SEED + i,
            "european": f"{european:.6f}",
            "american_tree": f"{american:.6f}",
            "american_tree_uncertainty": f"{abs(american - american_2x):.2e}",
            "bermudan_tree": f"{bermudan:.6f}",
            "bermudan_tree_uncertainty": f"{abs(bermudan - bermudan_2x):.2e}",
            "lsm_price": f"{res.price:.6f}",
            "lsm_se": f"{res.standard_error:.6f}",
            "ci_low": f"{res.ci_low:.6f}",
            "ci_high": f"{res.ci_high:.6f}",
            "in_sample_price": f"{res.in_sample_price:.6f}",
            "in_sample_se": f"{res.in_sample_standard_error:.6f}",
            "gap_in_minus_out": f"{res.gap:+.6f}",
            "error_vs_american": f"{res.price - american:+.6f}",
            "z_vs_american": f"{(res.price - american) / res.standard_error:+.3f}",
            "pass_american": int(res.contains(american)),
            "error_vs_bermudan": f"{res.price - bermudan:+.6f}",
            "z_vs_bermudan": f"{(res.price - bermudan) / res.standard_error:+.3f}",
            "pass_bermudan": int(res.contains(bermudan)),
            "exercise_now": int(res.exercise_now),
            "runtime_s": f"{runtime:.3f}",
        })
        flag = "pass" if res.contains(american) else "MISS"
        print(f"  case {i:>2}: S={S:.0f} sigma={sigma} T={T:.0f}  LSM {res.price:.4f} "
              f"+/- {res.standard_error:.4f}  tree {american:.4f}  "
              f"z={(res.price - american) / res.standard_error:+.2f}  {flag}")
    write_csv("lsm_validation.csv", rows)


# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------

def run_paths() -> None:
    print("\n2. Error against number of paths (case 0, 100 seeds per count)")
    S, sigma, T = BASE
    n_dates = n_dates_for(T)
    american = american_binomial(S, K, R, sigma, T, TREE_STEPS, "put")
    bermudan = bermudan_reference(S, sigma, T, n_dates)
    rows = []
    for n in PATH_COUNTS:
        start = time.perf_counter()
        for seed in PATH_SEEDS:
            t0 = time.perf_counter()
            res = lsm_price(S, K, R, sigma, T, "put", n_paths=n, n_dates=n_dates,
                            degree=DEGREE, seed=seed)
            rows.append({
                "n_paths": n, "seed": seed,
                "american_tree": f"{american:.6f}", "bermudan_tree": f"{bermudan:.6f}",
                "lsm_price": f"{res.price:.6f}", "lsm_se": f"{res.standard_error:.6f}",
                "in_sample_price": f"{res.in_sample_price:.6f}",
                "covers_american": int(res.contains(american)),
                "covers_bermudan": int(res.contains(bermudan)),
                "runtime_s": f"{time.perf_counter() - t0:.4f}",
            })
        print(f"  N={n:>7,}: {time.perf_counter() - start:.1f} s")
    write_csv("lsm_paths.csv", rows)


# ---------------------------------------------------------------------------
# 3. Exercise dates
# ---------------------------------------------------------------------------

def run_dates() -> None:
    print("\n3. Error against number of exercise dates (case 0, 20 seeds per count)")
    S, sigma, T = BASE
    american = american_binomial(S, K, R, sigma, T, TREE_STEPS, "put")
    european = black_scholes(S, K, R, sigma, T, "put")
    rows = []
    for n_dates in DATE_COUNTS:
        bermudan = bermudan_reference(S, sigma, T, n_dates)
        start = time.perf_counter()
        for seed in DATE_SEEDS:
            res = lsm_price(S, K, R, sigma, T, "put", n_paths=N_TRAIN, n_dates=n_dates,
                            degree=DEGREE, seed=seed)
            rows.append({
                "n_dates": n_dates, "seed": seed,
                "european": f"{european:.6f}", "american_tree": f"{american:.6f}",
                "bermudan_tree": f"{bermudan:.6f}",
                "lsm_price": f"{res.price:.6f}", "lsm_se": f"{res.standard_error:.6f}",
                "in_sample_price": f"{res.in_sample_price:.6f}",
                "covers_bermudan": int(res.contains(bermudan)),
                "covers_american": int(res.contains(american)),
            })
        print(f"  M={n_dates:>3}: Bermudan tree {bermudan:.4f}  "
              f"({time.perf_counter() - start:.1f} s)")
    write_csv("lsm_dates.csv", rows)


# ---------------------------------------------------------------------------
# 4. Basis
# ---------------------------------------------------------------------------

def run_basis() -> None:
    print("\n4. Basis degree (case 0 and case 3, 20 seeds per basis)")
    rows = []
    for label, (S, sigma, T) in {"case0": BASE, "case3": (36.0, 0.40, 2.0)}.items():
        n_dates = n_dates_for(T)
        american = american_binomial(S, K, R, sigma, T, TREE_STEPS, "put")
        bermudan = bermudan_reference(S, sigma, T, n_dates)
        for degree, weighted in BASES:
            start = time.perf_counter()
            for seed in BASIS_SEEDS:
                res = lsm_price(S, K, R, sigma, T, "put", n_paths=N_TRAIN, n_dates=n_dates,
                                degree=degree, weighted=weighted, seed=seed)
                rows.append({
                    "case": label, "degree": degree, "weighted": int(weighted),
                    "seed": seed,
                    "american_tree": f"{american:.6f}", "bermudan_tree": f"{bermudan:.6f}",
                    "lsm_price": f"{res.price:.6f}", "lsm_se": f"{res.standard_error:.6f}",
                    "in_sample_price": f"{res.in_sample_price:.6f}",
                })
            print(f"  {label} degree {degree}{' weighted' if weighted else ''}: "
                  f"{time.perf_counter() - start:.1f} s")
    write_csv("lsm_basis.csv", rows)


# ---------------------------------------------------------------------------
# 5. Exercise boundary
# ---------------------------------------------------------------------------

def run_boundary() -> None:
    print("\n5. Exercise boundary (tree vs 20 independently fitted LSM rules)")
    tree_rows, lsm_rows = [], []
    for label, (S, sigma, T) in BOUNDARY_CASES.items():
        times, boundary = tree_exercise_boundary(S, K, R, sigma, T, BOUNDARY_TREE_STEPS, "put")
        tree_rows += [{"case": label, "t": f"{t:.6f}", "boundary": f"{b:.6f}"}
                      for t, b in zip(times, boundary)]
        for seed in BOUNDARY_SEEDS:
            res = lsm_price(S, K, R, sigma, T, "put", n_paths=N_TRAIN,
                            n_dates=n_dates_for(T), degree=DEGREE, seed=seed)
            times, boundary = lsm_exercise_boundary(res.rule)
            lsm_rows += [{"case": label, "seed": seed, "t": f"{t:.6f}",
                          "boundary": f"{b:.6f}"} for t, b in zip(times, boundary)]
        print(f"  {label} done")
    write_csv("lsm_boundary_tree.csv", tree_rows)
    write_csv("lsm_boundary_lsm.csv", lsm_rows)


# ---------------------------------------------------------------------------
# 6. The American call
# ---------------------------------------------------------------------------

def run_call() -> None:
    print("\n6. American call, no dividends: does the fitted rule exercise early?")
    sigma, T = 0.20, 1.0
    n_dates = n_dates_for(T)
    rows = []
    for S in CALL_SPOTS:
        european = black_scholes(S, K, R, sigma, T, "call")
        for seed in CALL_SEEDS:
            res = lsm_price(S, K, R, sigma, T, "call", n_paths=N_TRAIN, n_dates=n_dates,
                            degree=DEGREE, seed=seed)
            # Re-run the fitted rule on fresh paths, beside the European payoff
            # on the *same* paths: any difference is the rule exercising early.
            rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
            paths_rng_state = rng.bit_generator.state
            cash = apply_exercise_rule(res.rule, N_PRICE, rng)
            rng.bit_generator.state = paths_rng_state
            paths = simulate_paths(S, R, sigma, T, n_dates, N_PRICE, rng)
            euro_cash = math.exp(-R * T) * payoff(paths[:, -1], K, "call")
            changed = np.abs(cash - euro_cash) > 1e-12
            diff = cash - euro_cash
            rows.append({
                "S": S, "seed": seed, "european_bs": f"{european:.6f}",
                "lsm_price": f"{res.price:.6f}", "lsm_se": f"{res.standard_error:.6f}",
                "covers_european": int(res.contains(european)),
                "fraction_exercised_early": f"{changed.mean():.6f}",
                "paired_cost": f"{-diff.mean():.6f}",
                "paired_cost_se": f"{diff.std(ddof=1) / math.sqrt(diff.size):.6f}",
            })
        print(f"  S={S:.0f} done")
    write_csv("lsm_call.csv", rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    start = time.perf_counter()
    run_validation()
    run_paths()
    run_dates()
    run_basis()
    run_boundary()
    run_call()
    print(f"\nAll Phase 7 experiments done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
