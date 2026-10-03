"""Read the Phase 2 CSVs in results/, print the tables and draw the charts.

    python analysis/plot_phase2.py

Reads only the CSVs, never re-runs the experiments -- so the charts and the
tables always show exactly the numbers that were saved, and the figures can be
redrawn without waiting for the trees and the simulations.
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

# Colours, matching the order-book project so the two read as one portfolio.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Categorical colours are assigned to an entity and never reused for anything
# else across the figures: even steps are always blue, odd always orange.
PARITY_COLOUR = {"even": BLUE, "odd": ORANGE}
RATE_COLOUR = {0.0: BLUE, 0.05: ORANGE, 0.10: AQUA}
SERIES_COLOUR = {
    ("atm", "call"): BLUE,
    ("atm", "put"): ORANGE,
    ("deep_otm", "call"): AQUA,
    ("deep_otm", "put"): YELLOW,
}
SERIES_LABEL = {
    ("atm", "call"): "at-the-money call",
    ("atm", "put"): "at-the-money put",
    ("deep_otm", "call"): "deep out-of-the-money call",
    ("deep_otm", "put"): "deep in-the-money put",
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
MARK = {"markersize": 8, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}


def read_csv(name: str) -> list:
    with open(RESULTS / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.918, subtitle, fontsize=10, color=INK_SECONDARY, va="top")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_convergence_table(rows: list) -> None:
    print("\nBINOMIAL CONVERGENCE TO BLACK-SCHOLES")
    print("European call, S=100, K=100, r=5%, sigma=20%, T=1. BS price 10.4505835722")
    print("error*N is constant if the error really is O(1/N).\n")
    atm = {int(r["steps"]): r for r in rows if r["case"] == "atm"}
    print(f"{'steps':>6} {'parity':>6} {'binomial':>14} {'error':>13} {'error*N':>9}")
    for steps in (10, 11, 50, 51, 100, 101, 200, 201, 500, 501,
                  1000, 1001, 2000, 2001, 5000, 5001):
        row = atm.get(steps)
        if row is None:
            continue
        print(f"{steps:>6} {row['parity']:>6} {float(row['binomial_call']):>14.8f} "
              f"{float(row['error_call']):>+13.2e} {float(row['scaled_error_call']):>9.4f}")

    print("\nAveraging an odd/even pair cancels most of the error:")
    print(f"{'pair':>12} {'mean price':>14} {'error':>13} {'vs best single':>15}")
    for even in (50, 200, 1000, 2000, 5000):
        low, high = atm.get(even), atm.get(even + 1)
        if low is None or high is None:
            continue
        bs = float(low["bs_call"])
        averaged = 0.5 * (float(low["binomial_call"]) + float(high["binomial_call"]))
        best_single = min(abs(float(low["error_call"])), abs(float(high["error_call"])))
        print(f"{f'{even}+{even + 1}':>12} {averaged:>14.8f} {averaged - bs:>+13.2e} "
              f"{abs(averaged - bs) / best_single:>14.1%}")

    print("\nThe same option 10% out of the money (K=110) is not a clean alternation:")
    otm = {int(r["steps"]): r for r in rows if r["case"] == "otm"}
    print(f"{'steps':>6} {'parity':>6} {'error':>13} {'error*N':>9}")
    for steps in (50, 51, 52, 53, 200, 201, 202, 203):
        row = otm.get(steps)
        if row is None:
            continue
        print(f"{steps:>6} {row['parity']:>6} {float(row['error_call']):>+13.2e} "
              f"{float(row['scaled_error_call']):>9.4f}")


def print_premium_table(rows: list) -> None:
    print("\n\nEARLY-EXERCISE PREMIUM (American minus European, same 1000-step tree)")
    print("Put, K=100, sigma=20%, T=1. The call premium is 0.00000000 at every")
    print("spot and every rate -- never exercise an American call early without dividends.\n")
    rates = sorted({float(r["r"]) for r in rows})
    by_spot = defaultdict(dict)
    for row in rows:
        by_spot[float(row["S"])][float(row["r"])] = row

    header = f"{'spot':>6} " + " ".join(f"{f'r={rate:.0%}':>22}" for rate in rates)
    print(header)
    print(f"{'':>6} " + " ".join(f"{'amer / euro / prem':>22}" for _ in rates))
    for spot in sorted(by_spot):
        cells = []
        for rate in rates:
            row = by_spot[spot][rate]
            cells.append(f"{float(row['american_put']):>6.2f} /"
                         f"{float(row['european_put']):>6.2f} /"
                         f"{float(row['premium_put']):>6.2f}")
        print(f"{spot:>6.0f} " + " ".join(f"{c:>22}" for c in cells))

    print("\nWhere immediate exercise is already optimal (American put = intrinsic):")
    for rate in rates:
        exercise_now = [float(r["S"]) for r in rows
                        if float(r["r"]) == rate
                        and abs(float(r["american_put"]) - float(r["intrinsic_put"])) < 1e-6
                        and float(r["intrinsic_put"]) > 0]
        spots = (f"spot <= {max(exercise_now):.0f}" if exercise_now else "never")
        print(f"  r = {rate:.0%}: {spots}")


def print_coverage_table(rows: list) -> None:
    print("\n\nMONTE CARLO 95% CONFIDENCE-INTERVAL COVERAGE")
    print("500 seeds per row. A correct interval covers 95% of the time;")
    print("with 500 seeds the sampling error on that figure is +/-0.0097.\n")
    print(f"{'case':>10} {'kind':>5} {'paths':>8} {'BS price':>11} {'mean se':>10} "
          f"{'bias':>11} {'coverage':>9}")
    for row in rows:
        flag = "" if abs(float(row["coverage"]) - 0.95) <= 3 * 0.0097 else "   <-- low"
        print(f"{row['case']:>10} {row['kind']:>5} {int(row['n_paths']):>8,} "
              f"{float(row['bs_price']):>11.6f} {float(row['mean_standard_error']):>10.6f} "
              f"{float(row['bias']):>+11.6f} {float(row['coverage']):>9.3f}{flag}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_convergence(rows: list) -> None:
    atm = [r for r in rows if r["case"] == "atm"]
    bs = float(atm[0]["bs_call"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.975, wspace=0.22)
    titles(fig, "The binomial tree converges to Black-Scholes, but not smoothly",
           "European call, S=100, K=100, r=5%, sigma=20%, T=1 year. "
           "Even step counts land below the true price, odd ones above.")

    # Left: the prices themselves, small step counts, so the zigzag is visible.
    near = [r for r in atm if int(r["steps"]) <= 60]
    steps = [int(r["steps"]) for r in near]
    prices = [float(r["binomial_call"]) for r in near]
    left.plot(steps, prices, color=AXIS, linewidth=1.0, zorder=1)
    for parity, colour in PARITY_COLOUR.items():
        xs = [s for s, r in zip(steps, near) if r["parity"] == parity]
        ys = [p for p, r in zip(prices, near) if r["parity"] == parity]
        left.plot(xs, ys, "o", color=colour, label=f"{parity} steps", zorder=3,
                  linestyle="none", **MARK)
    left.axhline(bs, color=INK_SECONDARY, linewidth=1.5, linestyle="--", zorder=2,
                 label=f"Black-Scholes  {bs:.4f}")
    left.set_xlabel("steps in the tree")
    left.set_ylabel("price")
    left.set_title("Price oscillates as steps increase", fontsize=10, color=INK, loc="left")
    left.legend(loc="lower right", labelcolor=INK_SECONDARY)

    # Right: the error envelope over the whole range, log-log.
    for parity, colour in PARITY_COLOUR.items():
        series = [r for r in atm if r["parity"] == parity]
        right.plot([int(r["steps"]) for r in series],
                   [float(r["abs_error_call"]) for r in series],
                   color=colour, label=f"{parity} steps", **LINE)
    # A 1/N guide. Drawn at 2.5x the data so it stays visible -- on top of the
    # series it would be invisible, which is itself the point: the slopes match.
    largest = max(atm, key=lambda r: int(r["steps"]))
    constant = float(largest["abs_error_call"]) * int(largest["steps"])
    guide_steps = [10, 5001]
    right.plot(guide_steps, [2.5 * constant / s for s in guide_steps],
               color=MUTED, linewidth=1.2, linestyle=":", zorder=1)
    right.text(300, 2.5 * constant / 300 * 1.25, "slope -1 (1/N), offset for visibility",
               color=MUTED, fontsize=9)

    # Measured slopes, so the 1/N claim is stated as a number, not a vibe.
    slopes = []
    for parity in ("even", "odd"):
        series = [r for r in atm if r["parity"] == parity and int(r["steps"]) >= 50]
        fit = np.polyfit([math.log(int(r["steps"])) for r in series],
                         [math.log(float(r["abs_error_call"])) for r in series], 1)
        slopes.append(f"{parity} {fit[0]:.2f}")
    right.text(0.98, 0.97, "measured slope:  " + ",  ".join(slopes),
               transform=right.transAxes, ha="right", va="top",
               fontsize=9, color=INK_SECONDARY)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("steps in the tree (log scale)")
    right.set_ylabel("|error| vs Black-Scholes (log scale)")
    right.set_title("Error falls like 1/N on both branches", fontsize=10, color=INK,
                    loc="left")
    right.legend(loc="lower left", labelcolor=INK_SECONDARY)

    fig.savefig(RESULTS / "binomial_convergence.png", dpi=160)
    plt.close(fig)
    print(f"\n  wrote {(RESULTS / 'binomial_convergence.png').relative_to(ROOT)}")


def plot_oscillation(rows: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.975, wspace=0.08)
    titles(fig, "Why the oscillation happens: where the strike sits in the lattice",
           "Signed error of the European call, steps 10 to 120. At the money the sign "
           "alternates perfectly; 10% out of the money it does not.")

    panels = [
        ("atm", "K = 100, at the money", axes[0]),
        ("otm", "K = 110, out of the money", axes[1]),
    ]
    for case, panel_title, ax in panels:
        series = [r for r in rows if r["case"] == case and 10 <= int(r["steps"]) <= 120]
        steps = [int(r["steps"]) for r in series]
        errors = [float(r["error_call"]) for r in series]
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.2, zorder=2)
        for parity, colour in PARITY_COLOUR.items():
            xs = [s for s, r in zip(steps, series) if r["parity"] == parity]
            ys = [e for e, r in zip(errors, series) if r["parity"] == parity]
            ax.vlines(xs, 0.0, ys, color=colour, linewidth=1.6, alpha=0.55, zorder=1)
            ax.plot(xs, ys, "o", color=colour, label=f"{parity} steps", linestyle="none",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        ax.set_xlabel("steps in the tree")
        ax.set_title(panel_title, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("binomial price - Black-Scholes price")
    axes[0].legend(loc="upper right", labelcolor=INK_SECONDARY)

    fig.savefig(RESULTS / "binomial_oscillation.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'binomial_oscillation.png').relative_to(ROOT)}")


def plot_american_premium(rows: list) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.975, wspace=0.22)
    titles(fig, "What the right to exercise early is worth",
           "American minus European put on the same 1000-step tree. "
           "K=100, sigma=20%, T=1 year. The call premium is exactly zero everywhere.")

    # Left: the two put curves and the intrinsic value, at the highest rate.
    high_rate = [r for r in rows if float(r["r"]) == 0.10]
    high_rate.sort(key=lambda r: float(r["S"]))
    spots = [float(r["S"]) for r in high_rate]
    left.plot(spots, [float(r["american_put"]) for r in high_rate], color=ORANGE,
              label="American put", zorder=2, **LINE)
    left.plot(spots, [float(r["european_put"]) for r in high_rate], color=BLUE,
              label="European put", zorder=2, **LINE)
    # On top: where this dashed line and the American curve coincide, immediate
    # exercise is optimal.
    left.plot(spots, [float(r["intrinsic_put"]) for r in high_rate], color=INK,
              linewidth=1.3, linestyle=(0, (4, 3)), zorder=4,
              label="intrinsic value  max(K-S, 0)")
    left.set_xlabel("spot price")
    left.set_ylabel("put value")
    left.set_title("r = 10%: the American put tracks intrinsic once deep in the money",
                   fontsize=10, color=INK, loc="left")
    left.legend(loc="upper right", labelcolor=INK_SECONDARY)

    # Right: the premium itself, one line per rate.
    for rate in sorted({float(r["r"]) for r in rows}):
        series = sorted((r for r in rows if float(r["r"]) == rate),
                        key=lambda r: float(r["S"]))
        xs = [float(r["S"]) for r in series]
        ys = [float(r["premium_put"]) for r in series]
        right.plot(xs, ys, color=RATE_COLOUR[rate], label=f"r = {rate:.0%}", **LINE)
        right.text(xs[0] - 1.0, ys[0], f"{ys[0]:.1f}", color=INK_SECONDARY,
                   fontsize=9, ha="right", va="center")
    right.set_xlabel("spot price")
    right.set_ylabel("premium (American - European)")
    right.set_title("No interest, no premium: the r = 0 line is flat zero",
                    fontsize=10, color=INK, loc="left")
    right.legend(loc="upper right", labelcolor=INK_SECONDARY)

    fig.savefig(RESULTS / "american_premium.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'american_premium.png').relative_to(ROOT)}")


def plot_coverage(per_seed: list, summary: list) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.975, wspace=0.22)
    titles(fig, "Is the reported uncertainty honest?",
           "A 95% confidence interval should contain the true price 95% of the time. "
           "Built 500 times per point, from independent seeds.")

    # Left: the first 60 intervals for the at-the-money call.
    calls = [r for r in per_seed if r["kind"] == "call" and int(r["seed"]) < 60]
    calls.sort(key=lambda r: int(r["seed"]))
    bs = float(calls[0]["bs_price"])
    left.axhline(bs, color=INK_SECONDARY, linewidth=1.5, zorder=3)
    misses = 0
    for row in calls:
        seed = int(row["seed"])
        covers = row["covers_bs"] == "1"
        colour = BLUE if covers else ORANGE
        misses += 0 if covers else 1
        left.vlines(seed, float(row["ci_low"]), float(row["ci_high"]),
                    color=colour, linewidth=2.0, zorder=2)
        left.plot(seed, float(row["price"]), "o", color=colour, markersize=3.5,
                  markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)
    left.plot([], [], color=BLUE, linewidth=2.0, label="interval covers Black-Scholes")
    left.plot([], [], color=ORANGE, linewidth=2.0, label=f"misses ({misses} of 60)")
    left.set_xlabel("seed")
    left.set_ylabel("price and 95% interval")
    left.set_title(f"60 runs at N = {int(calls[0]['n_paths']):,} paths "
                   f"(the line is Black-Scholes, {bs:.4f})",
                   fontsize=10, color=INK, loc="left")
    left.legend(loc="lower right", labelcolor=INK_SECONDARY)
    left.grid(axis="x", visible=False)

    # Right: coverage against path count, one line per case.
    right.axhline(0.95, color=INK_SECONDARY, linewidth=1.5, linestyle="--", zorder=3)
    right.text(100_000, 0.9535, "nominal 95%  ", color=INK_SECONDARY, fontsize=9,
               va="bottom", ha="right")
    for key, colour in SERIES_COLOUR.items():
        case, kind = key
        series = sorted((r for r in summary if r["case"] == case and r["kind"] == kind),
                        key=lambda r: int(r["n_paths"]))
        if not series:
            continue
        xs = [int(r["n_paths"]) for r in series]
        ys = [float(r["coverage"]) for r in series]
        right.plot(xs, ys, color=colour, marker="o", label=SERIES_LABEL[key],
                   **LINE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.0)
    # The one series that misbehaves gets a direct label.
    bad = sorted((r for r in summary if r["case"] == "deep_otm" and r["kind"] == "call"),
                 key=lambda r: int(r["n_paths"]))
    if bad:
        right.text(int(bad[0]["n_paths"]) * 1.15, float(bad[0]["coverage"]),
                   f"{float(bad[0]['coverage']):.2f}", color=INK_SECONDARY, fontsize=9,
                   va="center")
    right.set_xscale("log")
    right.set_xlabel("paths per run (log scale)")
    right.set_ylabel("fraction of intervals covering the true price")
    right.set_ylim(0.70, 1.0)
    right.set_title("Rare payoffs break the normal approximation", fontsize=10,
                    color=INK, loc="left")
    right.legend(loc="lower right", labelcolor=INK_SECONDARY, fontsize=9)

    fig.savefig(RESULTS / "mc_ci_coverage.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'mc_ci_coverage.png').relative_to(ROOT)}")


def main() -> None:
    convergence = read_csv("binomial_convergence.csv")
    premium = read_csv("american_premium.csv")
    per_seed = read_csv("mc_ci_coverage.csv")
    summary = read_csv("mc_coverage_summary.csv")

    print_convergence_table(convergence)
    print_premium_table(premium)
    print_coverage_table(summary)

    print("\n\nCHARTS")
    plot_convergence(convergence)
    plot_oscillation(convergence)
    plot_american_premium(premium)
    plot_coverage(per_seed, summary)


if __name__ == "__main__":
    main()
