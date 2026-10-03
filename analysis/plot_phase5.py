"""Read the Phase 5 CSVs in results/, print the tables and draw the charts.

    python analysis/plot_phase5.py
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                   # write straight to files, no window needed
import matplotlib.pyplot as plt         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# One colour per implementation, in the order they are reported.
IMPL_COLOUR = {}
IMPL_ORDER = []

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
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.918, subtitle, fontsize=10, color=INK_SECONDARY, va="top")


def by_implementation(rows: list) -> dict:
    series = defaultdict(list)
    for row in rows:
        series[row["implementation"]].append(row)
    for label in series:
        series[label].sort(key=lambda r: int(r["n_paths"]))
    return series


def assign_colours(rows: list) -> None:
    global IMPL_ORDER
    IMPL_ORDER = list(dict.fromkeys(r["implementation"] for r in rows))
    palette = [MUTED, BLUE, ORANGE, AQUA, YELLOW]
    for index, label in enumerate(IMPL_ORDER):
        IMPL_COLOUR[label] = palette[index % len(palette)]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_benchmark(rows: list) -> None:
    print("\nTHROUGHPUT BY IMPLEMENTATION (fastest of the interleaved repeats)")
    print("Headline numbers use the fastest repeat: background load can only make a")
    print("run slower, so the minimum is the cleanest estimate of what the code costs.")
    print("The median is in the CSV alongside, and differs by up to 1.9x on this")
    print("machine -- which is why the repeats are interleaved.\n")
    series = by_implementation(rows)
    path_counts = sorted({int(r["n_paths"]) for r in rows})

    print(f"{'implementation':>16} " + " ".join(f"{f'N={n:,}':>14}" for n in path_counts))
    print(f"{'':>16} " + " ".join(f"{'M paths/sec':>14}" for _ in path_counts))
    for label in IMPL_ORDER:
        by_paths = {int(r["n_paths"]): r for r in series[label]}
        cells = [f"{num(by_paths[n]['paths_per_second']) / 1e6:>14.2f}" if n in by_paths
                 else f"{'':>14}" for n in path_counts]
        print(f"{label:>16} " + " ".join(cells))

    print(f"\n{'implementation':>16} " +
          " ".join(f"{f'N={n:,}':>14}" for n in path_counts))
    print(f"{'':>16} " + " ".join(f"{'ns per path':>14}" for _ in path_counts))
    for label in IMPL_ORDER:
        by_paths = {int(r["n_paths"]): r for r in series[label]}
        cells = [f"{num(by_paths[n]['ns_per_path']):>14.2f}" if n in by_paths
                 else f"{'':>14}" for n in path_counts]
        print(f"{label:>16} " + " ".join(cells))

    print("\nSpeedup against the pure Python loop, and against NumPy:\n")
    print(f"{'implementation':>16} {'N':>12} {'vs python':>11} {'vs numpy':>10} "
          f"{'cost (var x t)':>15}")
    for label in IMPL_ORDER:
        for row in series[label]:
            print(f"{label:>16} {int(row['n_paths']):>12,} "
                  f"{row['speedup_vs_python_loop'] or '-':>11} "
                  f"{num(row['speedup_vs_numpy']):>10.2f} "
                  f"{num(row['variance_times_runtime']):>15.3e}")

    print("\nEvery implementation agrees on the price (they use different generators,")
    print("so they agree statistically, not to the last digit):\n")
    for label in IMPL_ORDER:
        row = series[label][-1]
        print(f"  {label:>16}  N={int(row['n_paths']):>10,}  "
              f"price {num(row['price']):.6f}  "
              f"error vs Black-Scholes {num(row['error_vs_bs']):+.6f}  "
              f"(se {num(row['standard_error']):.6f})")


def print_threads(rows: list) -> None:
    print("\n\nTHREAD SCALING")
    print("Same path count, more threads. Parallel efficiency is the fraction of the")
    print("ideal linear speedup that is actually realised.\n")
    print(f"{'threads':>8} {'ms':>10} {'M paths/s':>12} {'speedup':>9} "
          f"{'efficiency':>11} {'price':>20}")
    for row in rows:
        print(f"{int(row['n_threads']):>8} {num(row['median_seconds']) * 1e3:>10.1f} "
              f"{num(row['paths_per_second']) / 1e6:>12.1f} "
              f"{num(row['speedup']):>8.2f}x {num(row['parallel_efficiency']):>10.1%} "
              f"{row['price']:>20}")
    identical = len({row["price"] for row in rows}) == 1
    print(f"\nAll thread counts produced a bitwise identical price: {identical}")
    print("The random draws are a pure function of (seed, path index), and the work is")
    print("split into a fixed number of chunks whatever the thread count, so both the")
    print("values and the order they are added in are independent of how many threads")
    print("run.")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_throughput(rows: list) -> None:
    series = by_implementation(rows)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.98, wspace=0.24)
    titles(fig, "Four implementations of the same estimator",
           "Equal path counts, interleaved repeats, fastest of each taken. "
           "European call, S=K=100, r=5%, sigma=20%, T=1.")

    for label in IMPL_ORDER:
        group = series[label]
        xs = [int(r["n_paths"]) for r in group]
        left.plot(xs, [num(r["paths_per_second"]) / 1e6 for r in group],
                  color=IMPL_COLOUR[label], marker="o", label=label, **LINE, **MARK)
        right.plot(xs, [num(r["ns_per_path"]) for r in group],
                   color=IMPL_COLOUR[label], marker="o", label=label, **LINE, **MARK)

    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("paths per run (log scale)")
    left.set_ylabel("million paths per second (log scale)")
    left.set_title("Throughput", fontsize=10, color=INK, loc="left")
    left.legend(loc="upper left", labelcolor=INK_SECONDARY, fontsize=9)

    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("paths per run (log scale)")
    right.set_ylabel("nanoseconds per path (log scale)")
    right.set_title("Cost per path: NumPy is U-shaped, the C++ core is flat",
                    fontsize=10, color=INK, loc="left")

    fig.savefig(RESULTS / "perf_throughput.png", dpi=160)
    plt.close(fig)
    print(f"\n  wrote {(RESULTS / 'perf_throughput.png').relative_to(ROOT)}")


def plot_per_path(rows: list) -> None:
    """The Phase 3 U-shape, re-measured in every implementation."""
    series = by_implementation(rows)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.09, right=0.97)
    titles(fig, "Where the per-path cost goes, and why",
           "Phase 3 found this U-shaped in NumPy. The C++ core keeps no arrays, so it "
           "has the left half of the curve but not the right.")

    for label in IMPL_ORDER:
        group = series[label]
        xs = [int(r["n_paths"]) for r in group]
        ys = [num(r["ns_per_path"]) for r in group]
        ax.plot(xs, ys, color=IMPL_COLOUR[label], marker="o", label=label,
                **LINE, **MARK)
        best = min(range(len(ys)), key=lambda i: ys[i])
        ax.plot(xs[best], ys[best], marker="o", color=IMPL_COLOUR[label],
                markersize=13, markerfacecolor="none", markeredgewidth=1.8,
                linestyle="none", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("paths per run (log scale)")
    ax.set_ylabel("nanoseconds per path (log scale)")
    ax.text(0.98, 0.95, "rings mark each implementation's cheapest batch size",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=INK_SECONDARY)
    ax.legend(loc="lower left", labelcolor=INK_SECONDARY, fontsize=9)

    fig.savefig(RESULTS / "perf_per_path.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'perf_per_path.png').relative_to(ROOT)}")


def plot_threads(rows: list) -> None:
    threads = [int(r["n_threads"]) for r in rows]
    speedups = [num(r["speedup"]) for r in rows]
    efficiency = [num(r["parallel_efficiency"]) for r in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.98, wspace=0.24)
    titles(fig, "Thread scaling, and where it stops",
           f"C++ core at N={int(rows[0]['n_paths']):,}. Every thread count returns a "
           "bitwise identical price.")

    left.plot(threads, threads, color=MUTED, linewidth=1.3, linestyle="--",
              label="perfect scaling", zorder=1)
    left.plot(threads, speedups, color=BLUE, marker="o", label="measured",
              **LINE, **MARK)
    left.set_xscale("log", base=2)
    left.set_yscale("log", base=2)
    left.set_xticks(threads)
    left.set_xticklabels([str(t) for t in threads])
    left.set_yticks([1, 2, 4, 8, 16, 24])
    left.set_yticklabels(["1x", "2x", "4x", "8x", "16x", "24x"])
    left.set_xlabel("threads")
    left.set_ylabel("speedup over one thread")
    left.set_title("Near-linear to 8 threads, then flat", fontsize=10, color=INK,
                   loc="left")
    left.legend(loc="upper left", labelcolor=INK_SECONDARY, fontsize=9)

    right.axhline(1.0, color=MUTED, linewidth=1.3, linestyle="--", zorder=1)
    right.plot(threads, efficiency, color=ORANGE, marker="o", **LINE, **MARK)
    right.set_xscale("log", base=2)
    right.set_xticks(threads)
    right.set_xticklabels([str(t) for t in threads])
    right.set_ylim(0, 1.1)
    right.set_xlabel("threads")
    right.set_ylabel("fraction of ideal speedup realised")
    right.set_title("Efficiency collapses past the physical cores", fontsize=10,
                    color=INK, loc="left")
    for t, value in zip(threads, efficiency):
        right.text(t, value + 0.04, f"{value:.0%}", ha="center", fontsize=8,
                   color=INK_SECONDARY)

    fig.savefig(RESULTS / "perf_threads.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {(RESULTS / 'perf_threads.png').relative_to(ROOT)}")


def print_methods(rows: list) -> None:
    print("\n\nALL FOUR ESTIMATORS, PYTHON AGAINST C++")
    print("The variance each method removes is a property of the method, not the")
    print("language; what changes here is what the paths cost. Cost = variance x")
    print("runtime is the product of the two, and is what actually matters.\n")
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    print(f"{'method':>12} {'implementation':>16} {'M paths/s':>11} {'vs python':>10} "
          f"{'std error':>11} {'cost':>12}")
    for method, group in by_method.items():
        for row in group:
            print(f"{method:>12} {row['implementation']:>16} "
                  f"{num(row['paths_per_second']) / 1e6:>11.1f} "
                  f"{num(row['speedup_vs_python']):>9.2f}x "
                  f"{num(row['standard_error']):>11.6f} "
                  f"{num(row['variance_times_runtime']):>12.3e}")
        print()


def main() -> None:
    benchmark = read_csv("perf_benchmark.csv")
    threads = read_csv("perf_threads.csv")
    methods = read_csv("perf_methods.csv")
    assign_colours(benchmark)

    print_benchmark(benchmark)
    print_threads(threads)
    print_methods(methods)

    print("\n\nCHARTS")
    plot_throughput(benchmark)
    plot_per_path(benchmark)
    plot_threads(threads)


if __name__ == "__main__":
    main()
