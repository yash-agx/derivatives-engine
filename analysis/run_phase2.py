"""Run the Phase 2 experiments and write the CSVs in results/.

    python analysis/run_phase2.py          (about 30 seconds)

Everything here is deterministic: fixed grids, fixed seeds. Re-running
rewrites the same numbers, which is what makes the charts and the write-up
citable.

Three experiments:

1. ``binomial_convergence.csv`` -- European call and put on CRR trees from 10
   to 5,001 steps, against the closed form. Run for an at-the-money option
   (where the odd/even oscillation is cleanest) and an out-of-the-money one
   (where the error is jittery instead, because the strike's position between
   lattice nodes keeps changing).
2. ``american_premium.csv`` -- American and European puts and calls across
   moneyness and three interest rates, to measure what the right to exercise
   early is worth.
3. ``mc_ci_coverage.csv`` and ``mc_coverage_summary.csv`` -- 1,000 seeded
   Monte Carlo runs, and whether each 95% confidence interval covers the
   Black-Scholes price. The summary repeats that at several path counts for a
   well-behaved option and for a deep out-of-the-money one, where the normal
   approximation behind the interval starts to fail.
"""

import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # the package is not installed; see pytest.ini

from pricing.binomial import binomial_price, european_binomial       # noqa: E402
from pricing.black_scholes import price                              # noqa: E402
from pricing.monte_carlo import monte_carlo_price                    # noqa: E402

RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Experiment 1: binomial convergence
# ---------------------------------------------------------------------------
# Dense up to 500 so consecutive step counts can be compared (that is what
# makes the oscillation visible), then pairs (N, N+1) further out to show the
# oscillation never goes away -- only its size does.
DENSE_STEPS = list(range(10, 501))
LARGE_STEPS = [n + offset
               for n in (600, 800, 1000, 1500, 2000, 3000, 4000, 5000)
               for offset in (0, 1)]
CONVERGENCE_STEPS = DENSE_STEPS + LARGE_STEPS

CONVERGENCE_CASES = {
    # label: (S, K, r, sigma, T)
    "atm": (100.0, 100.0, 0.05, 0.20, 1.0),
    "otm": (100.0, 110.0, 0.05, 0.20, 1.0),
}

# ---------------------------------------------------------------------------
# Experiment 2: the early-exercise premium
# ---------------------------------------------------------------------------
PREMIUM_SPOTS = [float(s) for s in range(60, 141, 5)]
PREMIUM_RATES = [0.0, 0.05, 0.10]
PREMIUM_STRIKE = 100.0
PREMIUM_SIGMA = 0.20
PREMIUM_T = 1.0
PREMIUM_STEPS = 1000

# ---------------------------------------------------------------------------
# Experiment 3: Monte Carlo confidence-interval coverage
# ---------------------------------------------------------------------------
COVERAGE_CASES = {
    "atm": (100.0, 100.0, 0.05, 0.20, 1.0),      # a normal option
    "deep_otm": (100.0, 160.0, 0.05, 0.20, 0.5),  # a call worth 0.004: rare payoffs
}
PER_SEED_PATHS = 20_000
PER_SEED_COUNT = 1_000
SUMMARY_PATHS = [5_000, 20_000, 100_000]
SUMMARY_SEEDS = 500


def write_csv(name: str, fieldnames: list, rows: list) -> None:
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def run_binomial_convergence() -> None:
    print("1/3 binomial convergence: "
          f"{len(CONVERGENCE_STEPS)} step counts x {len(CONVERGENCE_CASES)} cases x 2 kinds")
    rows = []
    for label, args in CONVERGENCE_CASES.items():
        closed_form = {kind: price(*args, kind) for kind in ("call", "put")}
        for steps in CONVERGENCE_STEPS:
            row = {
                "case": label,
                "S": args[0], "K": args[1], "r": args[2], "sigma": args[3], "T": args[4],
                "steps": steps,
                "parity": "even" if steps % 2 == 0 else "odd",
            }
            for kind in ("call", "put"):
                tree = european_binomial(*args, steps, kind)
                error = tree - closed_form[kind]
                row[f"binomial_{kind}"] = f"{tree:.10f}"
                row[f"bs_{kind}"] = f"{closed_form[kind]:.10f}"
                row[f"error_{kind}"] = f"{error:.10e}"
                row[f"abs_error_{kind}"] = f"{abs(error):.10e}"
                # error * N: constant if the error really is O(1/N).
                row[f"scaled_error_{kind}"] = f"{abs(error) * steps:.6f}"
            rows.append(row)

    write_csv(
        "binomial_convergence.csv",
        ["case", "S", "K", "r", "sigma", "T", "steps", "parity",
         "binomial_call", "bs_call", "error_call", "abs_error_call", "scaled_error_call",
         "binomial_put", "bs_put", "error_put", "abs_error_put", "scaled_error_put"],
        rows,
    )


def run_american_premium() -> None:
    print(f"2/3 early-exercise premium: {len(PREMIUM_SPOTS)} spots x "
          f"{len(PREMIUM_RATES)} rates, {PREMIUM_STEPS}-step trees")
    rows = []
    for rate in PREMIUM_RATES:
        for spot in PREMIUM_SPOTS:
            args = (spot, PREMIUM_STRIKE, rate, PREMIUM_SIGMA, PREMIUM_T)
            row = {
                "S": spot, "K": PREMIUM_STRIKE, "r": rate,
                "sigma": PREMIUM_SIGMA, "T": PREMIUM_T, "steps": PREMIUM_STEPS,
                "moneyness": f"{spot / PREMIUM_STRIKE:.3f}",
            }
            for kind in ("put", "call"):
                american = binomial_price(*args, PREMIUM_STEPS, kind, american=True)
                european = binomial_price(*args, PREMIUM_STEPS, kind, american=False)
                intrinsic = (max(spot - PREMIUM_STRIKE, 0.0) if kind == "call"
                             else max(PREMIUM_STRIKE - spot, 0.0))
                row[f"american_{kind}"] = f"{american:.8f}"
                row[f"european_{kind}"] = f"{european:.8f}"
                row[f"premium_{kind}"] = f"{american - european:.8f}"
                row[f"premium_pct_{kind}"] = (
                    f"{100.0 * (american - european) / european:.4f}" if european > 1e-12 else ""
                )
                row[f"intrinsic_{kind}"] = f"{intrinsic:.8f}"
                row[f"bs_{kind}"] = f"{price(*args, kind):.8f}"
            rows.append(row)

    write_csv(
        "american_premium.csv",
        ["S", "K", "r", "sigma", "T", "steps", "moneyness",
         "american_put", "european_put", "premium_put", "premium_pct_put",
         "intrinsic_put", "bs_put",
         "american_call", "european_call", "premium_call", "premium_pct_call",
         "intrinsic_call", "bs_call"],
        rows,
    )


def run_monte_carlo_coverage() -> None:
    print(f"3/3 Monte Carlo coverage: {PER_SEED_COUNT} seeds per kind at "
          f"N={PER_SEED_PATHS:,}, plus the summary sweep")

    # Per-seed detail for the at-the-money option: one row per run.
    args = COVERAGE_CASES["atm"]
    rows = []
    for kind in ("call", "put"):
        closed_form = price(*args, kind)
        for seed in range(PER_SEED_COUNT):
            result = monte_carlo_price(*args, kind, n_paths=PER_SEED_PATHS, seed=seed)
            rows.append({
                "case": "atm",
                "kind": kind,
                "seed": seed,
                "n_paths": result.n_paths,
                "price": f"{result.price:.8f}",
                "standard_error": f"{result.standard_error:.8f}",
                "ci_low": f"{result.ci_low:.8f}",
                "ci_high": f"{result.ci_high:.8f}",
                "bs_price": f"{closed_form:.8f}",
                "error": f"{result.price - closed_form:.8f}",
                "error_in_se": f"{(result.price - closed_form) / result.standard_error:.4f}",
                "covers_bs": int(result.contains(closed_form)),
            })
    write_csv(
        "mc_ci_coverage.csv",
        ["case", "kind", "seed", "n_paths", "price", "standard_error",
         "ci_low", "ci_high", "bs_price", "error", "error_in_se", "covers_bs"],
        rows,
    )

    # Summary: coverage by case, option type and path count.
    summary = []
    for label, case_args in COVERAGE_CASES.items():
        for kind in ("call", "put"):
            closed_form = price(*case_args, kind)
            for n_paths in SUMMARY_PATHS:
                results = [monte_carlo_price(*case_args, kind, n_paths=n_paths, seed=seed)
                           for seed in range(SUMMARY_SEEDS)]
                covered = sum(r.contains(closed_form) for r in results)
                mean_price = sum(r.price for r in results) / len(results)
                mean_se = sum(r.standard_error for r in results) / len(results)
                coverage = covered / len(results)
                # How far coverage can stray by chance with this many seeds.
                sampling_error = math.sqrt(0.95 * 0.05 / len(results))
                summary.append({
                    "case": label,
                    "kind": kind,
                    "n_paths": n_paths,
                    "seeds": len(results),
                    "bs_price": f"{closed_form:.8f}",
                    "mean_price": f"{mean_price:.8f}",
                    "bias": f"{mean_price - closed_form:.8f}",
                    "mean_standard_error": f"{mean_se:.8f}",
                    "covered": covered,
                    "coverage": f"{coverage:.4f}",
                    "coverage_sampling_error": f"{sampling_error:.4f}",
                })
    write_csv(
        "mc_coverage_summary.csv",
        ["case", "kind", "n_paths", "seeds", "bs_price", "mean_price", "bias",
         "mean_standard_error", "covered", "coverage", "coverage_sampling_error"],
        summary,
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    run_binomial_convergence()
    run_american_premium()
    run_monte_carlo_coverage()
    print(f"done in {time.perf_counter() - started:.1f} s")
    print("now run: python analysis/plot_phase2.py")


if __name__ == "__main__":
    main()
