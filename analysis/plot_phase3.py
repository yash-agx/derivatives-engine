"""Read the Phase 3 CSVs in results/, print the tables and draw the charts.

    python analysis/plot_phase3.py

Reads only the CSVs, so the figures always show exactly the numbers that were
saved and can be redrawn without re-running the eight-minute study.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")                   # write straight to files, no window needed
import matplotlib.pyplot as plt         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Shared with the Phase 2 charts and the order-book project.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Moneyness is an *ordered* variable, so it gets a sequential ramp (one hue,
# light to dark) rather than categorical hues. Lightness is monotonic, checked.
MONEYNESS_ORDER = ["deep OTM", "OTM", "ATM", "ITM", "deep ITM"]
MONEYNESS_RAMP = dict(zip(MONEYNESS_ORDER,
                          ["#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8", "#1e3a8a"]))
# Maturity is categorical and only three wide.
MATURITY_COLOUR = {"1m": BLUE, "1y": ORANGE, "5y": AQUA}
MATURITY_ORDER = ["1m", "1y", "5y"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})

LINE = {"linewidth": 2.0}
MARK = {"markersize": 7, "markeredgecolor": SURFACE, "markeredgewidth": 1.0}


def read_csv(name: str) -> list:
    with open(RESULTS / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value) -> float:
    """CSV cell to float; blanks and 'nan' become nan rather than raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


MATURITY_TOKENS = {"1m", "1y", "5y"}


def split_label(label: str) -> tuple:
    """Split a label into its contract part and its maturity.

    'deep OTM 1y' -> ('deep OTM', '1y');  'ATM 1y put' -> ('ATM put', '1y').
    Finding the maturity token rather than slicing the end keeps the put, whose
    label has a trailing word, in the maturity-coloured charts.
    """
    tokens = label.split()
    maturity = next((token for token in tokens if token in MATURITY_TOKENS), "")
    rest = " ".join(token for token in tokens if token != maturity)
    return rest, maturity


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.918, subtitle, fontsize=10, color=INK_SECONDARY, va="top")


def fit_slope(xs, ys) -> float:
    """Least-squares slope of log(y) against log(x)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        return float("nan")
    return float(np.polyfit([math.log(x) for x, _ in pairs],
                            [math.log(y) for _, y in pairs], 1)[0])


# With 500 seeds, a correct 95% interval still lands anywhere in roughly
# 0.95 +/- 2*sqrt(0.95*0.05/500) = 0.95 +/- 0.019 by chance alone. Drawing that
# band stops the eye reading noise as signal.
COVERAGE_SEEDS = 500
COVERAGE_NOISE = 2.0 * (0.95 * 0.05 / COVERAGE_SEEDS) ** 0.5


def sweep_series(rows: list) -> dict:
    """The skewness sweep, keyed by strike label, sorted by path count."""
    series = defaultdict(list)
    for row in rows:
        series[row["option"]].append(row)
    for label in series:
        series[label].sort(key=lambda r: int(r["n_paths"]))
    return dict(sorted(series.items(), key=lambda kv: float(kv[0].split("=")[1])))


def one_year_calls(rows: list) -> dict:
    """The T=1y call rows, keyed by moneyness, each sorted by path count."""
    series = defaultdict(list)
    for row in rows:
        money, maturity = split_label(row["option"])
        if maturity == "1y" and row["kind"] == "call":
            series[money].append(row)
    for money in series:
        series[money].sort(key=lambda r: int(r["n_paths"]))
    return series


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_base_case(rows: list) -> None:
    print("\nTHE BASE CASE: at-the-money call, 1 year, 500 seeds per path count")
    base = sorted((r for r in rows if r["option"] == "ATM 1y"),
                  key=lambda r: int(r["n_paths"]))
    if not base:
        return
    print(f"Black-Scholes price {num(base[0]['bs_price']):.6f}\n")
    print(f"{'paths':>10} {'bias':>11} {'RMSE':>10} {'spread':>10} {'mean se':>10} "
          f"{'se/spread':>10} {'coverage':>9} {'runtime':>10} {'var x t':>10}")
    for row in base:
        print(f"{int(row['n_paths']):>10,} {num(row['bias']):>+11.6f} "
              f"{num(row['rmse']):>10.6f} {num(row['spread']):>10.6f} "
              f"{num(row['mean_reported_se']):>10.6f} {num(row['se_ratio']):>10.3f} "
              f"{num(row['coverage']):>9.3f} {num(row['median_runtime_s']) * 1e3:>9.2f}ms "
              f"{num(row['variance_times_runtime']):>10.2e}")

    rmse_slope = fit_slope([int(r["n_paths"]) for r in base],
                           [num(r["rmse"]) for r in base])
    print(f"\nfitted log-log slope of RMSE against paths: {rmse_slope:.4f} "
          f"(theory: -0.5)")


def print_slopes(rows: list) -> None:
    print("\n\nCONVERGENCE RATE BY OPTION  (log-log slope of RMSE against paths)")
    print("Theory says -0.5 for every one of them: the 1/sqrt(N) law does not care")
    print("what the option is.\n")
    by_option = defaultdict(list)
    for row in rows:
        by_option[row["option"]].append(row)
    print(f"{'option':>14} {'slope':>8} {'RMSE at 1e3':>12} {'RMSE at 1e6':>12} "
          f"{'ratio':>8}")
    for label, group in by_option.items():
        group.sort(key=lambda r: int(r["n_paths"]))
        slope = fit_slope([int(r["n_paths"]) for r in group],
                          [num(r["rmse"]) for r in group])
        coarse, fine = num(group[0]["rmse"]), num(group[-1]["rmse"])
        ratio = coarse / fine if fine > 0 else float("nan")
        print(f"{label:>14} {slope:>8.3f} {coarse:>12.6f} {fine:>12.6f} {ratio:>8.1f}")
    print("\n(a ratio near 31.6 = sqrt(1000) is exactly the 1/sqrt(N) law over "
          "1,000x the paths)")


def print_coverage_matrix(rows: list) -> None:
    print("\n\nWHERE THE 95% CONFIDENCE INTERVAL BREAKS DOWN")
    print("Coverage over 500 seeds. Anything below 0.93 is flagged: with 500 seeds")
    print("the sampling error is +/-0.010, so 0.93 is about 2 standard errors low.\n")
    path_counts = sorted({int(r["n_paths"]) for r in rows})
    by_option = defaultdict(dict)
    for row in rows:
        by_option[row["option"]][int(row["n_paths"])] = row

    header = f"{'option':>14} " + " ".join(f"{f'N={n:,}':>12}" for n in path_counts)
    print(header + f"  {'skewness':>9}")
    for label, by_paths in by_option.items():
        cells = []
        for n in path_counts:
            row = by_paths.get(n)
            if row is None:
                cells.append(f"{'':>12}")
                continue
            coverage = num(row["coverage"])
            flag = " *" if coverage < 0.93 else "  "
            cells.append(f"{coverage:>10.3f}{flag}")
        skew = num(next(iter(by_paths.values()))["skewness"])
        print(f"{label:>14} " + " ".join(cells) + f"  {skew:>9.2f}")
    print("\n* = materially below the nominal 95%")


def print_sweep_table(rows: list) -> None:
    print("\n\nTHE SKEWNESS SWEEP: pushing the strike out until the interval fails")
    print("S=100, six months, sigma=20%. As the strike rises the payoff becomes a")
    print("rare event, and the question is how many paths still pay anything.\n")
    series = sweep_series(rows)
    path_counts = sorted({int(r["n_paths"]) for r in rows})

    print(f"{'strike':>8} {'BS price':>11} {'skewness':>9} {'zero %':>10}")
    for label, group in series.items():
        row = group[0]
        print(f"{label:>8} {num(row['bs_price']):>11.6f} {num(row['skewness']):>9.2f} "
              f"{num(row['zero_fraction']) * 100:>9.4f}%")

    print("\ncoverage:")
    print(f"{'strike':>8} " + " ".join(f"{f'N={n:,}':>12}" for n in path_counts))
    for label, group in series.items():
        by_paths = {int(r["n_paths"]): r for r in group}
        cells = [f"{num(by_paths[n]['coverage']):>12.3f}" if n in by_paths
                 else f"{'':>12}" for n in path_counts]
        print(f"{label:>8} " + " ".join(cells))

    print("\nexpected number of paths that pay anything:")
    print(f"{'strike':>8} " + " ".join(f"{f'N={n:,}':>12}" for n in path_counts))
    for label, group in series.items():
        by_paths = {int(r["n_paths"]): r for r in group}
        cells = [f"{num(by_paths[n]['expected_paying_paths']):>12.2f}" if n in by_paths
                 else f"{'':>12}" for n in path_counts]
        print(f"{label:>8} " + " ".join(cells))

    print("\ncoverage against paying paths, both experiments pooled and sorted:")
    pairs = sorted((num(r["expected_paying_paths"]), num(r["coverage"]))
                   for r in rows if num(r["expected_paying_paths"]) > 0)
    for paying, coverage in pairs:
        bar = "#" * int(round(coverage * 40))
        print(f"   {paying:>12,.2f} paying   {coverage:>5.3f}  {bar}")


def print_shape_table(rows: list) -> None:
    print("\n\nPAYOFF SHAPE: why coverage fails where it does")
    print("Measured on 2,000,000 paths per option. 'paths for CLT' is the")
    print("Berry-Esseen rule of thumb (skewness/0.1)^2 -- the point past which the")
    print("normal approximation behind the interval should be safe.\n")
    print(f"{'option':>14} {'BS price':>12} {'payoff sd':>10} {'skewness':>9} "
          f"{'kurtosis':>9} {'zero %':>8} {'paths for CLT':>15}")
    for row in rows:
        print(f"{row['option']:>14} {num(row['bs_price']):>12.6f} "
              f"{num(row['payoff_std']):>10.4f} {num(row['skewness']):>9.2f} "
              f"{num(row['excess_kurtosis']):>9.1f} "
              f"{num(row['zero_fraction']) * 100:>7.2f}% "
              f"{num(row['paths_for_clt']):>15,.0f}")


def print_efficiency_table(rows: list) -> None:
    print("\n\nEFFICIENCY AND COST")
    print("variance x runtime is a COST (lower is better): the price of a unit of")
    print("accuracy. In theory it is flat in N; here it rises, so large batches are")
    print("less efficient than medium ones.")
    print("'paths for 1 cent' is what a 95% error bar of +/-$0.01 costs.\n")
    print(f"{'option':>14} {'paths':>10} {'var x t':>11} {'ns/path':>9} "
          f"{'paths for 1c':>16} {'implied time':>13}")
    for row in sorted(rows, key=lambda r: (r["option"], int(r["n_paths"]))):
        if int(row["n_paths"]) != 100_000:       # one representative row per option
            continue
        needed = num(row["paths_for_1_cent"])
        seconds = needed * num(row["runtime_per_path_ns"]) * 1e-9
        print(f"{row['option']:>14} {int(row['n_paths']):>10,} "
              f"{num(row['variance_times_runtime']):>11.2e} "
              f"{num(row['runtime_per_path_ns']):>9.1f} {needed:>16,.0f} "
              f"{seconds:>12.2f}s")

    print("\nruntime per path against path count (the same work, so this should be flat):")
    base = sorted((r for r in rows if r["option"] == "ATM 1y"),
                  key=lambda r: int(r["n_paths"]))
    for row in base:
        print(f"   N={int(row['n_paths']):>9,}  {num(row['runtime_per_path_ns']):>6.1f} ns/path"
              f"   median runtime {num(row['median_runtime_s']) * 1e3:>8.3f} ms")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_convergence(rows: list) -> None:
    series = one_year_calls(rows)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.90, wspace=0.26)
    titles(fig, "Monte Carlo error falls like 1/sqrt(N), whatever the option",
           "RMSE against Black-Scholes over 500 seeds per point. One-year calls, "
           "K=100, r=5%, sigma=20%.")

    slope_lines = []
    for money in MONEYNESS_ORDER:
        group = series.get(money)
        if not group:
            continue
        xs = [int(r["n_paths"]) for r in group]
        ys = [num(r["rmse"]) for r in group]
        colour = MONEYNESS_RAMP[money]
        left.plot(xs, ys, color=colour, marker="o", label=money, **LINE, **MARK)
        left.text(xs[-1] * 1.25, ys[-1], money, color=INK_SECONDARY, fontsize=9,
                  va="center")
        slope_lines.append(f"{money} {fit_slope(xs, ys):+.3f}")
        # RMSE * sqrt(N): flat if the law holds exactly.
        right.plot(xs, [y * math.sqrt(x) for x, y in zip(xs, ys)], color=colour,
                   marker="o", label=money, **LINE, **MARK)
        right.text(xs[-1] * 1.25, ys[-1] * math.sqrt(xs[-1]), money,
                   color=INK_SECONDARY, fontsize=9, va="center")

    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("paths per run (log scale)")
    left.set_ylabel("RMSE vs Black-Scholes (log scale)")
    left.set_title("Slope -0.5 on every option", fontsize=10, color=INK, loc="left")
    # Top right: the only empty corner, since every line runs top-left to
    # bottom-right. At the bottom it collided with the "deep OTM" end label.
    left.text(0.98, 0.97, "fitted slopes\n" + "\n".join(slope_lines),
              transform=left.transAxes, fontsize=8, color=INK_SECONDARY,
              ha="right", va="top", linespacing=1.5)

    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("paths per run (log scale)")
    right.set_ylabel("RMSE x sqrt(N)")
    right.set_title("The same claim as a constant", fontsize=10, color=INK, loc="left")

    fig.savefig(RESULTS / "mc_convergence.png", dpi=160)
    plt.close(fig)
    print(f"\n  wrote {(RESULTS / 'mc_convergence.png').relative_to(ROOT)}")


def plot_error_honesty(rows: list) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.90, wspace=0.26)
    titles(fig, "Does the simulation's self-reported error bar match reality?",
           "Reported standard error against the actual spread of 500 estimates. "
           "On the diagonal means honest.")

    usable = [r for r in rows if num(r["spread"]) > 0]
    for maturity in MATURITY_ORDER:
        group = [r for r in usable if split_label(r["option"])[1] == maturity]
        left.plot([num(r["spread"]) for r in group],
                  [num(r["mean_reported_se"]) for r in group],
                  "o", color=MATURITY_COLOUR[maturity], linestyle="none",
                  label=f"T = {maturity}", **MARK)
    lo = min(num(r["spread"]) for r in usable) * 0.6
    hi = max(num(r["spread"]) for r in usable) * 1.6
    left.plot([lo, hi], [lo, hi], color=INK_SECONDARY, linewidth=1.3, linestyle="--",
              zorder=1, label="perfect agreement")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("actual spread across seeds (log scale)")
    left.set_ylabel("mean reported standard error (log scale)")
    left.set_title("Every point sits on the line", fontsize=10, color=INK, loc="left")
    left.legend(loc="upper left", labelcolor=INK_SECONDARY, fontsize=9)

    series = one_year_calls(rows)
    right.axhline(1.0, color=INK_SECONDARY, linewidth=1.3, linestyle="--", zorder=1)
    for money in MONEYNESS_ORDER:
        group = series.get(money)
        if not group:
            continue
        xs = [int(r["n_paths"]) for r in group]
        ys = [num(r["se_ratio"]) for r in group]
        right.plot(xs, ys, color=MONEYNESS_RAMP[money], marker="o", label=money,
                   **LINE, **MARK)
        right.text(xs[-1] * 1.25, ys[-1], money, color=INK_SECONDARY, fontsize=9,
                   va="center")
    right.set_xscale("log")
    right.set_xlabel("paths per run (log scale)")
    right.set_ylabel("reported se / actual spread")
    right.set_ylim(0.9, 1.1)        # context: the deviation really is tiny
    right.set_title("Ratio stays at 1.0 — the error bar is honest",
                    fontsize=10, color=INK, loc="left")

    fig.savefig(RESULTS / "mc_error_honesty.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'mc_error_honesty.png').relative_to(ROOT)}")


def plot_coverage(grid_rows: list, sweep_rows: list) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.90, wspace=0.26)
    titles(fig, "Where the confidence interval stops being trustworthy",
           "Coverage of the 95% interval over 500 seeds. The grey band is what "
           "sampling noise alone can do; anything outside it is real.")

    # Left: the sweep, which is where the breakdown actually lives.
    left.axhspan(0.95 - COVERAGE_NOISE, 0.95 + COVERAGE_NOISE, color=GRID,
                 zorder=0)
    left.axhline(0.95, color=INK_SECONDARY, linewidth=1.3, linestyle="--", zorder=3)
    for index, (label, group) in enumerate(sweep_series(sweep_rows).items()):
        xs = [int(r["n_paths"]) for r in group]
        ys = [num(r["coverage"]) for r in group]
        colour = MONEYNESS_RAMP[MONEYNESS_ORDER[min(index, 4)]]
        left.plot(xs, ys, color=colour, marker="o", **LINE, **MARK)
        # Label at the left end: the curves converge on the right, where five
        # labels would sit on top of each other.
        left.text(xs[0] * 0.8, ys[0], label, color=INK_SECONDARY, fontsize=9,
                  va="center", ha="right")
    left.set_xscale("log")
    left.set_xlim(400, 3.5e6)
    left.set_ylim(-0.03, 1.02)
    left.set_xlabel("paths per run (log scale)")
    left.set_ylabel("fraction of intervals covering the true price")
    left.set_title("Six-month calls, strike pushed further out", fontsize=10,
                   color=INK, loc="left")

    # Right: every point from both experiments against the one quantity that
    # explains them -- how many paths in a run actually pay anything.
    right.axhspan(0.95 - COVERAGE_NOISE, 0.95 + COVERAGE_NOISE, color=GRID, zorder=0)
    right.axhline(0.95, color=INK_SECONDARY, linewidth=1.3, linestyle="--", zorder=3)

    sweep_points = [(num(r["expected_paying_paths"]), num(r["coverage"]))
                    for r in sweep_rows if num(r["expected_paying_paths"]) > 0]
    grid_points = []
    for row in grid_rows:
        paying = (1.0 - num(row["zero_fraction"])) * int(row["n_paths"])
        if paying > 0 and not math.isnan(num(row["coverage"])):
            grid_points.append((paying, num(row["coverage"])))
    right.plot([x for x, _ in grid_points], [y for _, y in grid_points], "o",
               color=MUTED, linestyle="none", markersize=5,
               markeredgecolor=SURFACE, markeredgewidth=0.8,
               label="moneyness x maturity grid")
    right.plot([x for x, _ in sweep_points], [y for _, y in sweep_points], "o",
               color=ORANGE, linestyle="none",
               label="skewness sweep", **MARK)
    right.axvline(50, color=MUTED, linewidth=1.2, linestyle=":", zorder=1)
    right.text(60, 0.30, "about 50 paying paths:\nbelow this the interval\nbreaks down",
               color=MUTED, fontsize=9, va="center")
    right.set_xscale("log")
    right.set_ylim(-0.03, 1.02)
    right.set_xlabel("expected number of paths that pay anything (log scale)")
    right.set_ylabel("coverage")
    right.set_title("Both experiments collapse onto one curve", fontsize=10,
                    color=INK, loc="left")
    right.legend(loc="lower right", labelcolor=INK_SECONDARY, fontsize=9)

    fig.savefig(RESULTS / "mc_coverage.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'mc_coverage.png').relative_to(ROOT)}")


def plot_sampling_distribution(detail: list) -> None:
    labels = ["ATM 1y", "K=160 6m"]
    path_counts = sorted({int(r["n_paths"]) for r in detail})
    fig, axes = plt.subplots(len(labels), len(path_counts), figsize=(12.5, 6.4),
                             sharex=True, sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.07, right=0.98,
                        wspace=0.12, hspace=0.30)
    titles(fig, "Why a rare payoff breaks the interval",
           "500 estimates per panel, standardised as (estimate - true price) / "
           "reported standard error. The curve is the standard normal the interval "
           "assumes it has.")

    grid = np.linspace(-5, 5, 400)
    normal_density = np.exp(-0.5 * grid ** 2) / math.sqrt(2 * math.pi)
    for row_index, label in enumerate(labels):
        for col_index, n_paths in enumerate(path_counts):
            ax = axes[row_index][col_index]
            values = [num(r["error_in_se"]) for r in detail
                      if r["option"] == label and int(r["n_paths"]) == n_paths
                      and r["error_in_se"] != ""]
            colour = BLUE if label == "ATM 1y" else ORANGE
            if values:
                ax.hist(values, bins=np.linspace(-5, 5, 41), density=True,
                        color=colour, alpha=0.85)
                ax.text(0.04, 0.93, f"skew {float(np.mean([(v - np.mean(values)) ** 3 for v in values]) / np.std(values) ** 3):+.2f}",
                        transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
                        va="top")
            ax.plot(grid, normal_density, color=INK_SECONDARY, linewidth=1.4)
            ax.axvline(0.0, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
            covered = [r for r in detail
                       if r["option"] == label and int(r["n_paths"]) == n_paths]
            if covered:
                hit = sum(int(r["covers_bs"]) for r in covered) / len(covered)
                ax.text(0.96, 0.93, f"coverage {hit:.1%}", transform=ax.transAxes,
                        fontsize=8, color=INK_SECONDARY, ha="right", va="top")
            if row_index == 0:
                ax.set_title(f"N = {n_paths:,}", fontsize=10, color=INK, loc="left")
            if col_index == 0:
                ax.set_ylabel(f"{label}\ndensity", fontsize=9)
            if row_index == len(labels) - 1:
                ax.set_xlabel("standardised error")
            ax.set_xlim(-5, 5)
            ax.grid(axis="x", visible=False)

    fig.savefig(RESULTS / "mc_sampling_distribution.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'mc_sampling_distribution.png').relative_to(ROOT)}")


def plot_efficiency(rows: list) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.055, right=0.95, wspace=0.34)
    titles(fig, "What the accuracy costs",
           "variance x runtime is a cost -- the price of a unit of accuracy, so "
           "lower is better. Theory says it is flat in N; the memory system says "
           "otherwise.")

    series = one_year_calls(rows)
    runtime_ax, efficiency_ax, cost_ax = axes

    for money in MONEYNESS_ORDER:
        group = series.get(money)
        if not group:
            continue
        xs = [int(r["n_paths"]) for r in group]
        runtime_ax.plot(xs, [num(r["runtime_per_path_ns"]) for r in group],
                        color=MONEYNESS_RAMP[money], marker="o", label=money,
                        **LINE, **MARK)
        efficiency_ax.plot(xs, [num(r["variance_times_runtime"]) for r in group],
                           color=MONEYNESS_RAMP[money], marker="o", label=money,
                           **LINE, **MARK)
    runtime_ax.set_xscale("log")
    runtime_ax.set_xlabel("paths per run (log scale)")
    runtime_ax.set_ylabel("nanoseconds per path")
    runtime_ax.set_title("Per-path cost is U-shaped", fontsize=10, color=INK,
                         loc="left")
    # All five options lie on top of each other, which is the finding: this is a
    # property of the machine, not of the option. So no legend -- it would only
    # label five identical lines.
    runtime_ax.text(0.97, 0.95,
                    "all five options coincide:\nthis is the machine,\nnot the option",
                    transform=runtime_ax.transAxes, fontsize=8, color=INK_SECONDARY,
                    ha="right", va="top", linespacing=1.5)

    efficiency_ax.set_xscale("log")
    efficiency_ax.set_yscale("log")
    efficiency_ax.set_xlabel("paths per run (log scale)")
    efficiency_ax.set_ylabel("variance x runtime, lower is better (log scale)")
    efficiency_ax.set_title("So a big batch costs more per unit of accuracy",
                            fontsize=10, color=INK, loc="left")

    # Cost of a one-cent error bar, by option.
    # The deep-OTM one-month call is excluded here: every path pays zero, so its
    # measured payoff spread is 0 and "paths for a cent" comes out as 0, which
    # is not a cost but a failure to sample anything at all. It is in the tables.
    at_100k = [r for r in rows
               if int(r["n_paths"]) == 100_000 and r["kind"] == "call"
               and num(r["paths_for_1_cent"]) > 0]
    at_100k.sort(key=lambda r: num(r["paths_for_1_cent"]))
    labels = [r["option"] for r in at_100k]
    needed = [num(r["paths_for_1_cent"]) for r in at_100k]
    colours = [MONEYNESS_RAMP.get(split_label(label)[0], BLUE) for label in labels]
    positions = range(len(labels))
    cost_ax.barh(list(positions), needed, color=colours, height=0.72)
    cost_ax.set_yticks(list(positions))
    cost_ax.set_yticklabels(labels, fontsize=8)
    cost_ax.set_xscale("log")
    cost_ax.set_xlabel("paths needed for a 1 cent error bar (log scale)")
    cost_ax.set_title("A cent costs 100x more out of the money", fontsize=10,
                      color=INK, loc="left")
    cost_ax.grid(axis="y", visible=False)
    for position, value in zip(positions, needed):
        if value > 0:
            cost_ax.text(value * 1.3, position, f"{value:,.0f}", fontsize=8,
                         color=INK_SECONDARY, va="center")

    fig.savefig(RESULTS / "mc_efficiency.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'mc_efficiency.png').relative_to(ROOT)}")


def main() -> None:
    convergence = read_csv("mc_convergence.csv")
    shape = read_csv("mc_payoff_shape.csv")
    detail = read_csv("mc_seed_detail.csv")
    sweep = read_csv("mc_skew_sweep.csv")

    print_base_case(convergence)
    print_slopes(convergence)
    print_coverage_matrix(convergence)
    print_sweep_table(sweep)
    print_shape_table(shape)
    print_efficiency_table(convergence)

    print("\n\nCHARTS")
    plot_convergence(convergence)
    plot_error_honesty(convergence)
    plot_coverage(convergence, sweep)
    plot_sampling_distribution(detail)
    plot_efficiency(convergence)


if __name__ == "__main__":
    main()
