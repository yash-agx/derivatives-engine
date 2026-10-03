"""Read the Phase 4 CSVs in results/, print the tables and draw the charts.

    python analysis/plot_phase4.py

Reads only the CSVs, so the figures always show exactly the numbers that were
saved and can be redrawn without re-running the experiments.
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

# Shared with the Phase 2 and Phase 3 charts, and the order-book project.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# One colour per method, fixed everywhere. Plain is the deliberately recessive
# baseline, so it wears ink rather than a hue.
METHOD_ORDER = ["plain", "antithetic", "control", "anti+control", "importance"]
METHOD_COLOUR = {
    "plain": MUTED,
    "antithetic": BLUE,
    "control": ORANGE,
    "anti+control": AQUA,
    "importance": YELLOW,
}

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

COVERAGE_SEEDS = 500
COVERAGE_NOISE = 2.0 * (0.95 * 0.05 / COVERAGE_SEEDS) ** 0.5


def read_csv(name: str) -> list:
    with open(RESULTS / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.918, subtitle, fontsize=10, color=INK_SECONDARY, va="top")


def by_option_and_method(rows: list, n_paths: int) -> dict:
    table = defaultdict(dict)
    for row in rows:
        if int(row["n_paths"]) == n_paths:
            table[row["option"]][row["estimator"]] = row
    return table


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_comparison(rows: list) -> None:
    path_counts = sorted({int(r["n_paths"]) for r in rows})
    for n_paths in path_counts:
        print(f"\n\nVARIANCE REDUCTION AT {n_paths:,} PATHS (500 seeds, equal path budget)")
        print("Factor = variance of plain / variance of the method. Above 1 means it")
        print("helped; below 1 means it made things worse.\n")
        table = by_option_and_method(rows, n_paths)
        methods = [m for m in METHOD_ORDER if m != "plain"]
        print(f"{'option':>14} " + " ".join(f"{m:>13}" for m in methods))
        for option, by_method in table.items():
            cells = []
            for method in methods:
                row = by_method.get(method)
                cells.append(f"{num(row['variance_reduction']):>12.1f}x" if row
                             else f"{'':>13}")
            print(f"{option:>14} " + " ".join(cells))

        print(f"\nCoverage at {n_paths:,} paths (nominal 0.95 +/- {COVERAGE_NOISE:.3f}):\n")
        print(f"{'option':>14} " + " ".join(f"{m:>13}" for m in METHOD_ORDER))
        for option, by_method in table.items():
            cells = []
            for method in METHOD_ORDER:
                row = by_method.get(method)
                if not row:
                    cells.append(f"{'':>13}")
                    continue
                coverage = num(row["coverage"])
                flag = "*" if coverage < 0.95 - 3 * COVERAGE_NOISE else " "
                cells.append(f"{coverage:>12.3f}{flag}")
            print(f"{option:>14} " + " ".join(cells))


def print_cost(rows: list) -> None:
    n_paths = max(int(r["n_paths"]) for r in rows)
    print(f"\n\nCOST AT {n_paths:,} PATHS: variance x runtime, as a ratio to plain")
    print("Above 1 means cheaper per unit of accuracy than plain Monte Carlo.")
    print("This is the number that matters: a method that halves the variance but")
    print("triples the runtime has made things worse.\n")
    table = by_option_and_method(rows, n_paths)
    methods = [m for m in METHOD_ORDER if m != "plain"]
    print(f"{'option':>14} " + " ".join(f"{m:>26}" for m in methods))
    print(f"{'':>14} " + " ".join(f"{'var cut / time / cost':>26}" for _ in methods))
    for option, by_method in table.items():
        cells = []
        for method in methods:
            row = by_method.get(method)
            if not row:
                cells.append(f"{'':>26}")
                continue
            cells.append(f"{num(row['variance_reduction']):>8.1f}x /"
                         f"{num(row['runtime_ratio']):>6.2f}x /"
                         f"{num(row['cost_ratio']):>8.1f}x")
        print(f"{option:>14} " + " ".join(cells))


def print_straddle(rows: list) -> None:
    print("\n\nWHERE ANTITHETIC VARIATES FAIL: the straddle")
    print("A straddle pays |S_T - K|, which is not monotone in the random draw.")
    print("At the forward the two legs of a pair move together instead of opposing,")
    print("so pairing concentrates variance. Moving the strike away makes one side")
    print("dominate and the payoff becomes closer to monotone again.\n")
    methods = [m for m in METHOD_ORDER if m in {r["estimator"] for r in rows}
               and m != "plain"]
    by_option = defaultdict(dict)
    for row in rows:
        by_option[row["option"]][row["estimator"]] = row
    print(f"{'straddle':>18} {'S/K':>6} " + " ".join(f"{m:>14}" for m in methods))
    for option in sorted(by_option, key=lambda o: float(o.split("=")[1])):
        by_method = by_option[option]
        money = num(next(iter(by_method.values()))["moneyness"])
        cells = [f"{num(by_method[m]['variance_reduction']):>13.2f}x" if m in by_method
                 else f"{'':>14}" for m in methods]
        print(f"{option:>18} {money:>6.2f} " + " ".join(cells))
    print("\n(a factor below 1.00 means the method increased the variance)")


def print_combination(rows: list) -> None:
    print("\n\nHOW ANTITHETIC AND CONTROL VARIATES COMBINE")
    print("Writing Y for a path's discounted payoff and X for its discounted terminal")
    print("stock price (the control), the three factors are:")
    print("    antithetic = 1 / (1 + rho_A)          rho_A  = Corr(Y(Z), Y(-Z))")
    print("    control    = 1 / (1 - rho_XY^2)       rho_XY = Corr(Y, X)")
    print("    both       = 1 / [(1 + rho_A)(1 - rho_AB^2)]")
    print("with rho_AB the correlation of the PAIR AVERAGES of Y and X. So 'both' is")
    print("the antithetic factor times a control factor recomputed on pairs -- not the")
    print("product of the two separate factors. The ratio is (1-rho_XY^2)/(1-rho_AB^2).\n")
    print(f"{'option':>18} {'rho_A':>8} {'rho_XY':>8} {'rho_AB':>8} "
          f"{'anti':>9} {'control':>11} {'both':>13} {'product':>14} {'both/prod':>10}")
    for row in rows:
        if math.isnan(num(row["rho_antithetic"])):
            continue
        print(f"{row['option']:>18} {num(row['rho_antithetic']):>+8.3f} "
              f"{num(row['rho_payoff_control']):>+8.3f} "
              f"{num(row['rho_pair_payoff_control']):>+8.3f} "
              f"{num(row['antithetic_factor']):>8.1f}x {num(row['control_factor']):>10.1f}x "
              f"{num(row['both_factor']):>12.1f}x {num(row['product_of_factors']):>13.1f}x "
              f"{num(row['both_over_product']):>9.2f}x")
    print("\nThe combination beats the product when pairing makes the control work")
    print("better (rho_AB > rho_XY), and loses to it when the payoff was already")
    print("nearly an exact linear function of the control, because then pairing")
    print("throws away the very component the control was exploiting.")


def print_bias(rows: list) -> None:
    print("\n\nCONTROL-VARIATE BIAS FROM REUSING THE PATHS")
    print("Estimating c on the same paths it is applied to makes it depend on the")
    print("data, which biases the estimate by O(1/N). Measured over 20,000 seeds at")
    print("small path counts, where that bias is largest. 'bias in se' is how many")
    print("standard errors of the mean it is -- under about 2, it is not detectable.\n")
    print(f"{'paths':>7} {'pilot':>7} {'bias':>12} {'bias in se':>11} "
          f"{'spread':>11} {'coverage':>9}")
    for row in rows:
        print(f"{int(row['n_paths']):>7,} {num(row['pilot_fraction']):>7.2f} "
              f"{num(row['bias']):>+12.3e} {num(row['bias_in_standard_errors']):>+11.2f} "
              f"{num(row['spread']):>11.3e} {num(row['coverage']):>9.4f}")


def print_rescue(rows: list) -> None:
    print("\n\nTHE KEY EXPERIMENT: does anything fix the deep out-of-the-money failure?")
    print("Phase 3 found the interval breaks when almost no path pays. Coverage of")
    print("the 95% interval, 500 seeds:\n")
    path_counts = sorted({int(r["n_paths"]) for r in rows})
    by_option = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_option[row["option"]][row["estimator"]][int(row["n_paths"])] = row

    for option in sorted(by_option, key=lambda o: float(o.split("=")[1])):
        first = next(iter(next(iter(by_option[option].values())).values()))
        print(f"  {option}  (skewness {num(first['skewness']):.1f}, "
              f"importance shift {num(first['is_shift']):+.2f} sd)")
        print(f"    {'method':>13} " + " ".join(f"{f'N={n:,}':>11}" for n in path_counts)
              + f" {'paying paths at 1e3':>21}")
        for method in METHOD_ORDER:
            by_paths = by_option[option].get(method, {})
            if not by_paths:
                continue
            cells = [f"{num(by_paths[n]['coverage']):>11.3f}" if n in by_paths
                     else f"{'':>11}" for n in path_counts]
            paying = num(by_paths[path_counts[0]]["expected_paying_paths"])
            print(f"    {method:>13} " + " ".join(cells) + f" {paying:>21.2f}")
        print()

    print("Variance reduction on the same options (vs plain, at each path count):\n")
    for option in sorted(by_option, key=lambda o: float(o.split("=")[1])):
        print(f"  {option}")
        print(f"    {'method':>13} " + " ".join(f"{f'N={n:,}':>11}" for n in path_counts))
        for method in METHOD_ORDER:
            by_paths = by_option[option].get(method, {})
            if not by_paths or method == "plain":
                continue
            cells = [f"{num(by_paths[n]['variance_reduction']):>10.1f}x" if n in by_paths
                     else f"{'':>11}" for n in path_counts]
            print(f"    {method:>13} " + " ".join(cells))
        print()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_variance_reduction(rows: list) -> None:
    n_paths = max(int(r["n_paths"]) for r in rows)
    table = by_option_and_method(rows, n_paths)
    # Drop options whose plain baseline has no variance at all (every path pays
    # zero), where a "variance reduction factor" has nothing to divide by.
    options = [o for o in table
               if not math.isnan(num(table[o]["antithetic"]["variance_reduction"]))]
    methods = [m for m in METHOD_ORDER if m != "plain"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.subplots_adjust(top=0.78, bottom=0.26, left=0.07, right=0.98, wspace=0.22)
    titles(fig, "What each method buys, and what it costs",
           f"Measured at {n_paths:,} paths over 500 seeds, equal path budget for every "
           "method. Above the line is an improvement — but a cross marks a result "
           "whose own error bar is broken.")

    positions = np.arange(len(options))
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        values = [num(table[o][method]["variance_reduction"]) if method in table[o]
                  else float("nan") for o in options]
        costs = [num(table[o][method]["cost_ratio"]) if method in table[o]
                 else float("nan") for o in options]
        left.bar(positions + offset, values, width=width * 0.92,
                 color=METHOD_COLOUR[method], label=method)
        right.bar(positions + offset, costs, width=width * 0.92,
                  color=METHOD_COLOUR[method], label=method)
        # A big variance reduction is not a win if the interval stopped working.
        for position, option in zip(positions, options):
            row = table[option].get(method)
            if row is None:
                continue
            coverage = num(row["coverage"])
            if abs(coverage - 0.95) <= 3 * COVERAGE_NOISE:
                continue
            for ax, series in ((left, values), (right, costs)):
                index_of = options.index(option)
                height = series[index_of]
                if height > 0:
                    ax.plot(position + offset, height * 1.6, marker="x",
                            color=INK, markersize=5, markeredgewidth=1.4,
                            linestyle="none", zorder=5)

    for ax, label, title in (
        (left, "variance reduction factor", "Variance: how much the spread shrank"),
        (right, "cost improvement factor", "Cost: variance x runtime, vs plain"),
    ):
        ax.axhline(1.0, color=INK_SECONDARY, linewidth=1.3, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(options, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    left.legend(loc="upper left", labelcolor=INK_SECONDARY, fontsize=9, ncol=2)
    left.plot([], [], marker="x", color=INK, markersize=5, markeredgewidth=1.4,
              linestyle="none", label="coverage broken")
    left.legend(loc="upper left", labelcolor=INK_SECONDARY, fontsize=9, ncol=2)

    fig.savefig(RESULTS / "vr_variance_reduction.png", dpi=160)
    plt.close(fig)
    print(f"\n  wrote {(RESULTS / 'vr_variance_reduction.png').relative_to(ROOT)}")


def plot_straddle(rows: list) -> None:
    by_option = defaultdict(dict)
    for row in rows:
        by_option[row["option"]][row["estimator"]] = row
    options = sorted(by_option, key=lambda o: float(o.split("=")[1]))
    moneyness = [num(next(iter(by_option[o].values()))["moneyness"]) for o in options]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.10, right=0.97)
    titles(fig, "Where antithetic variates fail, and why",
           "Straddles paying |S_T - K|, one year, 20,000 paths, 500 seeds. Below the "
           "line the method made the variance worse.")

    ax.axhline(1.0, color=INK_SECONDARY, linewidth=1.3, zorder=3)
    for method in METHOD_ORDER:
        if method == "plain" or method not in by_option[options[0]]:
            continue
        values = [num(by_option[o][method]["variance_reduction"]) for o in options]
        ax.plot(moneyness, values, color=METHOD_COLOUR[method], marker="o",
                label=method, **LINE, **MARK)
    ax.set_yscale("log")
    # The antithetic reflection is symmetric about the MEDIAN terminal price,
    # S*exp((r - sigma^2/2)T) = 103.05 here -- not about the spot. That is where
    # the payoff |S_T - K| is most nearly even in the draw, and it is where the
    # measured minimum sits (between K=100 and K=115).
    median_terminal = 100.0 * math.exp((0.05 - 0.5 * 0.20 ** 2) * 1.0)
    ax.axvline(100.0 / median_terminal, color=MUTED, linewidth=1.2, linestyle=":",
               zorder=1)
    ax.text(0.40, 0.055,
            "strike at the median terminal price (%.1f):\nthe payoff is most "
            "symmetric here, and\nantithetic pairing has nothing to cancel"
            % median_terminal,
            transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel("spot / strike")
    ax.set_ylabel("variance reduction factor (log scale)")
    ax.legend(loc="upper left", labelcolor=INK_SECONDARY, fontsize=9)

    fig.savefig(RESULTS / "vr_straddle.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'vr_straddle.png').relative_to(ROOT)}")


def plot_coverage_rescue(rows: list) -> None:
    by_option = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_option[row["option"]][row["estimator"]][int(row["n_paths"])] = row
    options = sorted(by_option, key=lambda o: float(o.split("=")[1]))

    fig, axes = plt.subplots(1, len(options), figsize=(13.5, 5.1), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.06, right=0.985, wspace=0.10)
    titles(fig, "Does variance reduction fix a broken confidence interval?",
           "Coverage of the 95% interval, 500 seeds. Only importance sampling does, "
           "because only it changes where the paths go.")

    for ax, option in zip(axes, options):
        ax.axhspan(0.95 - COVERAGE_NOISE, 0.95 + COVERAGE_NOISE, color=GRID, zorder=0)
        ax.axhline(0.95, color=INK_SECONDARY, linewidth=1.2, linestyle="--", zorder=3)
        for method in METHOD_ORDER:
            by_paths = by_option[option].get(method, {})
            if not by_paths:
                continue
            xs = sorted(by_paths)
            ys = [num(by_paths[n]["coverage"]) for n in xs]
            ax.plot(xs, ys, color=METHOD_COLOUR[method], marker="o", label=method,
                    **LINE, **MARK)
        first = next(iter(next(iter(by_option[option].values())).values()))
        skew = num(first["skewness"])
        # The furthest strike has no finite skewness: in a 2,000,000-path sample
        # not one path finished in the money.
        caption = f"skew {skew:.0f}" if not math.isnan(skew) else "no path ever paid"
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("paths per run")
        ax.set_title(f"{option}  ({caption})", fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraction of intervals covering the true price")
    # Above the panels: inside the last one it sat on top of the K=200 lines.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.95),
               ncol=5, labelcolor=INK_SECONDARY, fontsize=9)

    fig.savefig(RESULTS / "vr_coverage_rescue.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'vr_coverage_rescue.png').relative_to(ROOT)}")


def plot_combination(rows: list) -> None:
    usable = [r for r in rows if not math.isnan(num(r["rho_antithetic"]))]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.98, wspace=0.26)
    titles(fig, "Antithetic and control variates do not simply multiply",
           "The combined factor is 1/[(1+rhoA)(1-rhoAB^2)] -- the antithetic factor "
           "times a control factor recomputed on pair averages.")

    # Left: pairing usually lifts the control's correlation, which is the whole
    # reason the combination usually beats the product.
    left.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1.3, linestyle="--",
              zorder=1, label="no change from pairing")
    left.plot([abs(num(r["rho_payoff_control"])) for r in usable],
              [abs(num(r["rho_pair_payoff_control"])) for r in usable],
              "o", color=ORANGE, linestyle="none", **MARK)
    left.set_xlim(0.3, 1.01)
    left.set_ylim(0.75, 1.005)
    left.set_xlabel("|correlation| of payoff with control, per path")
    left.set_ylabel("the same, on pair averages")
    left.set_title("Pairing usually improves the control", fontsize=10, color=INK,
                   loc="left")
    left.legend(loc="lower right", labelcolor=INK_SECONDARY, fontsize=9)

    # Right: and where it does not, the combination underperforms the product.
    right.axhline(1.0, color=INK_SECONDARY, linewidth=1.3, zorder=3)
    for row in usable:
        ratio = num(row["both_over_product"])
        colour = AQUA if ratio >= 1 else ORANGE
        right.plot(abs(num(row["rho_payoff_control"])), ratio, "o", color=colour,
                   linestyle="none", **MARK)
    right.set_yscale("log")
    right.set_xlabel("|correlation| of payoff with control, per path")
    right.set_ylabel("combined factor / product of factors")
    right.set_title("Above 1: better than the product. Below: worse.", fontsize=10,
                    color=INK, loc="left")
    right.text(0.985, 0.06,
               "when the payoff is already almost\nan exact linear function of the\n"
               "control, pairing throws away what\nthe control was exploiting",
               transform=right.transAxes, color=INK_SECONDARY, fontsize=8.5,
               ha="right", va="bottom", linespacing=1.5)

    fig.savefig(RESULTS / "vr_combination.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'vr_combination.png').relative_to(ROOT)}")


def plot_control_bias(rows: list) -> None:
    by_pilot = defaultdict(list)
    for row in rows:
        by_pilot[num(row["pilot_fraction"])].append(row)
    for pilot in by_pilot:
        by_pilot[pilot].sort(key=lambda r: int(r["n_paths"]))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.98, wspace=0.24)
    titles(fig, "Does reusing the paths to fit the control coefficient bias it?",
           "20,000 seeds per point, at small path counts where an O(1/N) bias would "
           "be largest. The band is +/-2 standard errors of the mean.")

    colours = [BLUE, ORANGE, AQUA]
    for index, pilot in enumerate(sorted(by_pilot)):
        group = by_pilot[pilot]
        xs = [int(r["n_paths"]) for r in group]
        biases = [num(r["bias"]) for r in group]
        label = "same paths" if pilot == 0 else f"pilot {pilot:.0%}"
        left.plot(xs, biases, color=colours[index % 3], marker="o", label=label,
                  **LINE, **MARK)
        right.plot(xs, [num(r["bias_in_standard_errors"]) for r in group],
                   color=colours[index % 3], marker="o", label=label, **LINE, **MARK)

    left.axhline(0.0, color=INK_SECONDARY, linewidth=1.2, zorder=3)
    left.set_xscale("log")
    left.set_xlabel("paths per run (log scale)")
    left.set_ylabel("bias (price units)")
    left.set_title("Bias in dollars", fontsize=10, color=INK, loc="left")
    left.legend(loc="best", labelcolor=INK_SECONDARY, fontsize=9)

    right.axhspan(-2, 2, color=GRID, zorder=0)
    right.axhline(0.0, color=INK_SECONDARY, linewidth=1.2, zorder=3)
    right.set_xscale("log")
    right.set_xlabel("paths per run (log scale)")
    right.set_ylabel("bias, in standard errors of the mean")
    right.set_title("Is it detectable at all?", fontsize=10, color=INK, loc="left")

    fig.savefig(RESULTS / "vr_control_bias.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'vr_control_bias.png').relative_to(ROOT)}")


def main() -> None:
    comparison = read_csv("vr_comparison.csv")
    straddle = read_csv("vr_straddle.csv")
    bias = read_csv("vr_control_bias.csv")
    rescue = read_csv("vr_coverage_rescue.csv")
    combination = read_csv("vr_combination.csv")

    print_comparison(comparison)
    print_cost(comparison)
    print_combination(combination)
    print_straddle(straddle)
    print_bias(bias)
    print_rescue(rescue)

    print("\n\nCHARTS")
    plot_variance_reduction(comparison)
    plot_combination(combination)
    plot_straddle(straddle)
    plot_coverage_rescue(rescue)
    plot_control_bias(bias)


if __name__ == "__main__":
    main()
