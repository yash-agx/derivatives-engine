"""Run the Phase 3 convergence study and write the CSVs in results/.

    python analysis/run_phase3.py          (about 9 minutes)

The question: how does ordinary Monte Carlo behave as paths increase, across
options of very different shape, and when can its self-reported error bar be
trusted?

Every estimate is run 500 times from independent seeds, because a single run
reveals nothing about a *method*. Measured against Black-Scholes, which is the
luxury of this project: in a real pricing problem there is no closed form to
check against, and the only error estimate available is the one the simulation
reports about itself -- so finding out when that self-report is reliable is the
practical point of the exercise.

Writes:

* ``mc_convergence.csv``  -- one row per (option, path count): bias, RMSE, the
  real spread across seeds, the mean reported standard error and their ratio,
  coverage, runtime, efficiency and the paths needed for a target error.
* ``mc_payoff_shape.csv`` -- one row per option: the skewness, excess kurtosis
  and zero fraction of its payoff distribution, which is what explains the
  coverage results.
* ``mc_seed_detail.csv``  -- every individual estimate for two contrasting
  options (one well-behaved, one that breaks), so the sampling distribution
  itself can be drawn.
* ``mc_skew_sweep.csv``   -- strikes pushed far out of the money, where a payoff
  becomes a rare event, to locate the point at which the confidence interval
  stops working.

The grid is 16 options x 4 path counts x 500 seeds. The same seeds are used
everywhere, so differences between rows are differences in the problem, not
luck. (One consequence worth knowing: results at different path counts are not
independent of each other, since seed 7 starts the same generator each time.)
"""

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # the package is not installed; see pytest.ini

from experiments.harness import (                                    # noqa: E402
    Option,
    clt_diagnostic,
    measure_payoff_shape,
    plain_monte_carlo,
    run_grid,
)

RESULTS = ROOT / "results"

PATH_COUNTS = [1_000, 10_000, 100_000, 1_000_000]
SEEDS = tuple(range(500))

# Fixed market, varying contract: five moneyness levels x three maturities.
RATE = 0.05
VOLATILITY = 0.20
STRIKE = 100.0
MONEYNESS = [
    ("deep OTM", 70.0),
    ("OTM", 90.0),
    ("ATM", 100.0),
    ("ITM", 110.0),
    ("deep ITM", 130.0),
]
MATURITIES = [("1m", 1.0 / 12.0), ("1y", 1.0), ("5y", 5.0)]

OPTIONS = [
    Option(f"{money} {maturity}", S=spot, K=STRIKE, r=RATE, sigma=VOLATILITY,
           T=years, kind="call")
    for money, spot in MONEYNESS
    for maturity, years in MATURITIES
]
# One put, as a check that nothing here is specific to calls.
OPTIONS.append(Option("ATM 1y put", S=100.0, K=STRIKE, r=RATE, sigma=VOLATILITY,
                      T=1.0, kind="put"))

# The two options whose individual estimates get saved, at matched path counts:
# one well-behaved, one that genuinely breaks. Drawing their sampling
# distributions side by side is the clearest picture of why coverage fails.
# (The first choice here was "deep OTM 1y", which turned out to have perfectly
# good coverage -- being far out of the money is not by itself enough.)
DETAIL_OPTIONS = [
    Option("ATM 1y", S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, kind="call"),
    Option("K=160 6m", S=100.0, K=160.0, r=0.05, sigma=0.20, T=0.5, kind="call"),
]
DETAIL_PATHS = [1_000, 5_000, 20_000, 100_000]

SHAPE_PATHS = 2_000_000

# ---------------------------------------------------------------------------
# The skewness sweep
# ---------------------------------------------------------------------------
# The grid above turns out not to break: coverage holds across every ordinary
# moneyness and maturity. To find where it *does* break, this sweep pushes the
# strike far enough out that a payoff becomes a rare event, and varies the path
# count over the range where the expected number of paying paths falls from
# thousands to less than one. Six months to expiry, matching the Phase 2 case
# that first showed the failure.
SWEEP_STRIKES = [120.0, 140.0, 160.0, 180.0, 200.0]
SWEEP_PATHS = [1_000, 5_000, 20_000, 100_000, 500_000, 2_000_000]
SWEEP_T = 0.5
SWEEP_OPTIONS = [
    Option(f"K={strike:.0f}", S=100.0, K=strike, r=RATE, sigma=VOLATILITY,
           T=SWEEP_T, kind="call")
    for strike in SWEEP_STRIKES
]


def write_csv(name: str, fieldnames: list, rows: list) -> None:
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def run_seed_detail() -> None:
    """Save every individual estimate for two contrasting options.

    Keeping the raw estimates is what makes it possible to draw the sampling
    distribution itself, rather than only its summary statistics -- and the
    shape of that distribution is the whole explanation for coverage.
    """
    print("\nseed detail: %d options x %d path counts x %d seeds"
          % (len(DETAIL_OPTIONS), len(DETAIL_PATHS), len(SEEDS)))
    studies = run_grid({"plain": plain_monte_carlo}, DETAIL_OPTIONS, DETAIL_PATHS,
                       SEEDS, progress=print)
    detail = []
    for study in studies:
        for seed, price, se, covered in zip(study.seeds, study.prices,
                                            study.standard_errors, study.covered):
            detail.append({
                "option": study.option.label,
                "n_paths": study.n_paths,
                "seed": seed,
                "price": "%.10f" % price,
                "standard_error": "%.10f" % se,
                "bs_price": "%.10f" % study.truth,
                "error": "%.10e" % (price - study.truth),
                "error_in_se": ("%.6f" % ((price - study.truth) / se)) if se > 0 else "",
                "covers_bs": int(covered),
            })
    write_csv("mc_seed_detail.csv", list(detail[0]), detail)


def run_skew_sweep() -> None:
    """Push the strike out until the confidence interval fails, and watch it go."""
    print("\nskewness sweep: %d strikes x %d path counts x %d seeds"
          % (len(SWEEP_OPTIONS), len(SWEEP_PATHS), len(SEEDS)))
    shapes = {option.label: measure_payoff_shape(option, n_paths=SHAPE_PATHS)
              for option in SWEEP_OPTIONS}
    studies = run_grid({"plain": plain_monte_carlo}, SWEEP_OPTIONS, SWEEP_PATHS,
                       SEEDS, progress=print)

    rows = []
    for study in studies:
        row = study.row()
        shape = shapes[study.option.label]
        paying = 1.0 - shape.zero_fraction
        row["skewness"] = "%.6f" % shape.skewness
        row["excess_kurtosis"] = "%.6f" % shape.excess_kurtosis
        row["zero_fraction"] = "%.8f" % shape.zero_fraction
        row["paying_fraction"] = "%.10f" % paying
        # The quantity that actually governs the failure: how many paths in a
        # single run contribute anything at all.
        row["expected_paying_paths"] = "%.4f" % (paying * study.n_paths)
        row["clt_diagnostic"] = "%.6f" % clt_diagnostic(shape.skewness, study.n_paths)
        rows.append(row)
    write_csv("mc_skew_sweep.csv", list(rows[0]), rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()

    print(f"payoff shape: {len(OPTIONS)} options x {SHAPE_PATHS:,} paths")
    shapes = {option.label: measure_payoff_shape(option, n_paths=SHAPE_PATHS)
              for option in OPTIONS}
    write_csv("mc_payoff_shape.csv", list(next(iter(shapes.values())).row()),
              [shape.row() for shape in shapes.values()])

    print(f"\nconvergence study: {len(OPTIONS)} options x {len(PATH_COUNTS)} path "
          f"counts x {len(SEEDS)} seeds")
    studies = run_grid(
        {"plain": plain_monte_carlo},
        OPTIONS,
        PATH_COUNTS,
        SEEDS,
        progress=print,
    )

    rows = []
    for study in studies:
        row = study.row()
        # Carry the shape numbers alongside, so the coverage analysis needs
        # only one file.
        shape = shapes[study.option.label]
        row["skewness"] = f"{shape.skewness:.6f}"
        row["excess_kurtosis"] = f"{shape.excess_kurtosis:.6f}"
        row["zero_fraction"] = f"{shape.zero_fraction:.6f}"
        row["clt_diagnostic"] = f"{clt_diagnostic(shape.skewness, study.n_paths):.6f}"
        rows.append(row)
    write_csv("mc_convergence.csv", list(rows[0]), rows)

    run_seed_detail()

    run_skew_sweep()

    print(f"\ndone in {time.perf_counter() - started:.0f} s")
    print("now run: python analysis/plot_phase3.py")


if __name__ == "__main__":
    main()
