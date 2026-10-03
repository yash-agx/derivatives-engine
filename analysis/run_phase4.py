"""Run the Phase 4 variance-reduction experiments and write the CSVs in results/.

    python analysis/run_phase4.py          (about 8 minutes)

Every comparison is made **at equal path counts**, because Phase 3 showed that
runtime per path is not constant in N: comparing methods across different path
budgets would measure the memory system as much as the method.

Four experiments:

1. ``vr_comparison.csv``  -- all five estimators on the Phase 3 option grid, at
   two path counts: variance reduction factor, RMSE, coverage, runtime and the
   cost metric (variance x runtime, lower is better) against plain Monte Carlo.
2. ``vr_straddle.csv``    -- where antithetic variates *fail*. A straddle pays
   |S_T - K|, which is not monotone in the random draw, so pairing concentrates
   variance instead of cancelling it. Sweeping the strike away from the forward
   turns the payoff gradually back into something monotone, so the method
   should recover.
3. ``vr_control_bias.csv`` -- estimating the control coefficient from the same
   paths it is then applied to makes it dependent on the data, which biases the
   estimate by O(1/N). Measured at small path counts, where that bias is
   largest, against a pilot-sample version that spends the same total paths.
4. ``vr_coverage_rescue.csv`` -- the key experiment. Phase 3 found that a deep
   out-of-the-money option breaks the confidence interval because almost no
   path pays. Do any of these methods fix it?
"""

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # the package is not installed; see pytest.ini

from experiments.harness import (                                    # noqa: E402
    ESTIMATORS,
    Option,
    measure_payoff_shape,
    plain_monte_carlo,
    run_seed_study,
)
from pricing.variance_reduction import (                             # noqa: E402
    control_variate_monte_carlo,
    importance_sampling_shift,
)

RESULTS = ROOT / "results"
SEEDS = tuple(range(500))

# ---------------------------------------------------------------------------
# 1. The main comparison, on the Phase 3 grid
# ---------------------------------------------------------------------------
RATE, VOLATILITY, STRIKE = 0.05, 0.20, 100.0
MONEYNESS = [("deep OTM", 70.0), ("OTM", 90.0), ("ATM", 100.0),
             ("ITM", 110.0), ("deep ITM", 130.0)]
MATURITIES = [("1m", 1.0 / 12.0), ("1y", 1.0), ("5y", 5.0)]
GRID = [
    Option(f"{money} {maturity}", S=spot, K=STRIKE, r=RATE, sigma=VOLATILITY,
           T=years, kind="call")
    for money, spot in MONEYNESS
    for maturity, years in MATURITIES
]
GRID.append(Option("ATM 1y put", S=100.0, K=STRIKE, r=RATE, sigma=VOLATILITY,
                   T=1.0, kind="put"))
COMPARISON_PATHS = [10_000, 100_000]

# ---------------------------------------------------------------------------
# 2. Where antithetic variates fail
# ---------------------------------------------------------------------------
# At the forward a straddle is symmetric in the draw, which is the worst case
# for pairing. Moving the strike away makes one side dominate, so the payoff
# becomes closer to monotone and the method should start helping again.
STRADDLES = [
    Option(f"straddle K={strike:.0f}", S=100.0, K=strike, r=RATE,
           sigma=VOLATILITY, T=1.0, kind="straddle")
    for strike in (70.0, 85.0, 100.0, 115.0, 130.0)
]
STRADDLE_METHODS = ["plain", "antithetic", "control", "anti+control"]
STRADDLE_PATHS = 20_000

# ---------------------------------------------------------------------------
# 3. Control-variate bias from reusing the paths
# ---------------------------------------------------------------------------
# The bias is O(1/N), so it is only visible at small N -- and resolving it needs
# many seeds, since it is far smaller than the run-to-run spread.
BIAS_OPTION = Option("ATM 1y", S=100.0, K=100.0, r=RATE, sigma=VOLATILITY,
                     T=1.0, kind="call")
BIAS_PATHS = [50, 200, 1_000]
BIAS_PILOTS = [0.0, 0.10, 0.25]
BIAS_SEEDS = tuple(range(20_000))

# ---------------------------------------------------------------------------
# 4. The coverage rescue
# ---------------------------------------------------------------------------
RESCUE_OPTIONS = [
    Option(f"K={strike:.0f}", S=100.0, K=strike, r=RATE, sigma=VOLATILITY,
           T=0.5, kind="call")
    for strike in (140.0, 160.0, 180.0, 200.0)
]
RESCUE_PATHS = [1_000, 5_000, 20_000, 100_000]
SHAPE_PATHS = 2_000_000


def write_csv(name: str, fieldnames: list, rows: list) -> None:
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def comparison_rows(options, path_counts, method_names, label_progress=True) -> list:
    """Run every method on every option at every path count, and compare.

    The plain run is done first for each (option, path count) so that every
    other method can be expressed as a ratio against it.
    """
    rows = []
    for option in options:
        for n_paths in path_counts:
            baseline = run_seed_study(plain_monte_carlo, option, n_paths, SEEDS,
                                      "plain")
            for name in method_names:
                study = (baseline if name == "plain"
                         else run_seed_study(ESTIMATORS[name], option, n_paths,
                                             SEEDS, name))
                row = study.row()
                variance = study.spread ** 2
                base_variance = baseline.spread ** 2
                row["variance_reduction"] = ("%.4f" % (base_variance / variance)
                                             if variance > 0 else "")
                row["rmse_ratio"] = ("%.4f" % (baseline.rmse / study.rmse)
                                     if study.rmse > 0 else "")
                row["runtime_ratio"] = "%.4f" % (study.median_runtime
                                                 / baseline.median_runtime)
                # Cost is variance x runtime and lower is better, so a ratio
                # above 1 means this method is cheaper per unit of accuracy.
                row["cost_ratio"] = (
                    "%.4f" % (baseline.variance_times_runtime
                              / study.variance_times_runtime)
                    if study.variance_times_runtime > 0 else "")
                rows.append(row)
                if label_progress:
                    print(f"  {name:>12} {option.label:<16} N={n_paths:>7,}  "
                          f"var cut {row['variance_reduction'] or 'n/a':>8}  "
                          f"coverage {study.coverage:.3f}")
    return rows


def run_comparison() -> None:
    print(f"1/4 main comparison: {len(ESTIMATORS)} methods x {len(GRID)} options "
          f"x {len(COMPARISON_PATHS)} path counts x {len(SEEDS)} seeds")
    rows = comparison_rows(GRID, COMPARISON_PATHS, list(ESTIMATORS))
    write_csv("vr_comparison.csv", list(rows[0]), rows)


def run_straddle() -> None:
    print(f"\n2/4 where antithetic fails: {len(STRADDLES)} straddles x "
          f"{len(STRADDLE_METHODS)} methods")
    rows = comparison_rows(STRADDLES, [STRADDLE_PATHS], STRADDLE_METHODS)
    for row in rows:
        option = next(o for o in STRADDLES if o.label == row["option"])
        row["moneyness"] = "%.4f" % option.moneyness
    write_csv("vr_straddle.csv", list(rows[0]), rows)


def run_control_bias() -> None:
    print(f"\n3/4 control-variate bias: {len(BIAS_PATHS)} path counts x "
          f"{len(BIAS_PILOTS)} pilot fractions x {len(BIAS_SEEDS):,} seeds")
    truth = BIAS_OPTION.black_scholes()
    rows = []
    for n_paths in BIAS_PATHS:
        for pilot in BIAS_PILOTS:
            def estimator(option, paths, seed, pilot=pilot):
                return control_variate_monte_carlo(
                    *option.args, option.kind, n_paths=paths, seed=seed,
                    pilot_fraction=pilot)

            study = run_seed_study(estimator, BIAS_OPTION, n_paths, BIAS_SEEDS,
                                   f"control pilot={pilot:.2f}")
            # How many standard errors of the mean the bias is: the only way to
            # tell a real bias from noise with this many seeds.
            standard_error_of_mean = study.spread / len(BIAS_SEEDS) ** 0.5
            rows.append({
                "option": BIAS_OPTION.label,
                "n_paths": n_paths,
                "pilot_fraction": "%.2f" % pilot,
                "seeds": len(BIAS_SEEDS),
                "bs_price": "%.10f" % truth,
                "mean_price": "%.10f" % study.mean_price,
                "bias": "%.10e" % study.bias,
                "bias_in_standard_errors": "%.4f" % (study.bias / standard_error_of_mean),
                "spread": "%.10e" % study.spread,
                "rmse": "%.10e" % study.rmse,
                "se_ratio": "%.4f" % study.se_ratio,
                "coverage": "%.4f" % study.coverage,
            })
            print(f"  N={n_paths:>5}  pilot {pilot:.2f}  bias {study.bias:+.3e} "
                  f"({study.bias / standard_error_of_mean:+.2f} se)  "
                  f"coverage {study.coverage:.4f}")
    write_csv("vr_control_bias.csv", list(rows[0]), rows)


def run_coverage_rescue() -> None:
    print(f"\n4/4 coverage rescue: {len(RESCUE_OPTIONS)} strikes x "
          f"{len(ESTIMATORS)} methods x {len(RESCUE_PATHS)} path counts")
    shapes = {option.label: measure_payoff_shape(option, n_paths=SHAPE_PATHS)
              for option in RESCUE_OPTIONS}
    rows = comparison_rows(RESCUE_OPTIONS, RESCUE_PATHS, list(ESTIMATORS))
    for row in rows:
        shape = shapes[row["option"]]
        paying = 1.0 - shape.zero_fraction
        row["skewness"] = "%.6f" % shape.skewness
        row["zero_fraction"] = "%.8f" % shape.zero_fraction
        row["expected_paying_paths"] = "%.4f" % (paying * int(row["n_paths"]))
        option = next(o for o in RESCUE_OPTIONS if o.label == row["option"])
        row["is_shift"] = "%.4f" % importance_sampling_shift(*option.args)
    write_csv("vr_coverage_rescue.csv", list(rows[0]), rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    run_comparison()
    run_straddle()
    run_control_bias()
    run_coverage_rescue()
    print(f"\ndone in {time.perf_counter() - started:.0f} s")
    print("now run: python analysis/plot_phase4.py")


if __name__ == "__main__":
    main()
