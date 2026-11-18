"""Read the Phase 7 CSVs in results/, print the tables and draw the charts.

    python analysis/plot_phase7.py

Reads only the CSVs written by run_phase7.py, never re-runs the experiments.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
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

# One colour per entity, the same in every figure.
COLOUR = {"out": BLUE, "in": ORANGE, "american": INK, "bermudan": AQUA, "european": MUTED}

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
MARK = {"markersize": 7, "markeredgecolor": SURFACE, "markeredgewidth": 1.2}


def read_csv(name: str) -> list:
    with open(RESULTS / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.925, subtitle, fontsize=10, color=INK_SECONDARY, va="top")


def f(row, key):
    return float(row[key])


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) / math.sqrt(values.size)


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------

def validation(rows: list) -> None:
    print("\nLSM VALIDATION: Longstaff & Schwartz (2001) Table 1, American put, K=40, r=6%")
    print("100k training + 100k pricing paths, 50 dates/yr, Laguerre degree 3.")
    print("z = (LSM - reference) / SE. Pass = reference inside the 95% CI.\n")
    print(f"{'case':>4} {'S':>3} {'sig':>4} {'T':>2} {'European':>9} {'American':>9} "
          f"{'Bermudan':>9} {'LSM out':>9} {'SE':>7} {'z_Am':>6} {'z_Be':>6} "
          f"{'in-sample':>9} {'gap':>8}")
    for r in rows:
        flag = "" if r["pass_american"] == "1" else "  MISS (American)"
        if r["pass_bermudan"] != "1":
            flag += "  MISS (Bermudan)"
        print(f"{r['case']:>4} {f(r, 'S'):>3.0f} {f(r, 'sigma'):>4.1f} {f(r, 'T'):>2.0f} "
              f"{f(r, 'european'):>9.4f} {f(r, 'american_tree'):>9.4f} "
              f"{f(r, 'bermudan_tree'):>9.4f} {f(r, 'lsm_price'):>9.4f} {f(r, 'lsm_se'):>7.4f} "
              f"{f(r, 'z_vs_american'):>+6.2f} {f(r, 'z_vs_bermudan'):>+6.2f} "
              f"{f(r, 'in_sample_price'):>9.4f} {f(r, 'gap_in_minus_out'):>+8.4f}{flag}")

    z_am = np.array([f(r, "z_vs_american") for r in rows])
    z_be = np.array([f(r, "z_vs_bermudan") for r in rows])
    gaps = np.array([f(r, "gap_in_minus_out") for r in rows])
    print(f"\nPass vs American: {sum(r['pass_american'] == '1' for r in rows)}/20 "
          f"(pre-registered rule: >= 18)")
    print(f"Pass vs Bermudan: {sum(r['pass_bermudan'] == '1' for r in rows)}/20")
    print(f"Mean z vs American {z_am.mean():+.3f} (se of mean {z_am.std(ddof=1)/math.sqrt(20):.3f}); "
          f"below the tree in {int((z_am < 0).sum())}/20")
    print(f"Mean z vs Bermudan {z_be.mean():+.3f} (se of mean {z_be.std(ddof=1)/math.sqrt(20):.3f}); "
          f"below the tree in {int((z_be < 0).sum())}/20")
    print(f"In-sample minus out-of-sample: mean {gaps.mean():+.4f}, "
          f"positive in {int((gaps > 0).sum())}/20")
    print(f"Max tree uncertainty: American "
          f"{max(f(r, 'american_tree_uncertainty') for r in rows):.1e}, Bermudan "
          f"{max(f(r, 'bermudan_tree_uncertainty') for r in rows):.1e}")
    print(f"Runtime per case: {np.mean([f(r, 'runtime_s') for r in rows]):.2f} s on average")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(top=0.82, bottom=0.27, left=0.08, right=0.98)
    x = np.arange(len(rows))
    err = np.array([f(r, "lsm_price") - f(r, "american_tree") for r in rows])
    half = np.array([1.959963985 * f(r, "lsm_se") for r in rows])
    gap = np.array([f(r, "bermudan_tree") - f(r, "american_tree") for r in rows])
    ins = np.array([f(r, "in_sample_price") - f(r, "american_tree") for r in rows])
    ax.axhline(0, color=COLOUR["american"], linewidth=1.2, label="American tree (reference)")
    ax.plot(x, gap, "_", color=COLOUR["bermudan"], markersize=16, markeredgewidth=2.5,
            label="Bermudan tree, same 50/yr dates")
    ax.errorbar(x, err, yerr=half, fmt="o", color=COLOUR["out"], capsize=3,
                label="LSM out-of-sample, 95% CI", **MARK)
    ax.plot(x + 0.25, ins, "D", color=COLOUR["in"], markersize=4, label="LSM in-sample")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f(r, 'S'):.0f}\n{f(r, 'sigma'):.1f}\n{f(r, 'T'):.0f}" for r in rows],
                       fontsize=8)
    ax.text(-0.9, ax.get_ylim()[0], "S\nσ\nT", fontsize=8, color=MUTED, va="top", ha="right")
    ax.set_ylabel("price minus American tree")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9)
    titles(fig, "LSM against the binomial tree on the 20 Longstaff-Schwartz puts",
           f"{sum(r['pass_american'] == '1' for r in rows)}/20 confidence intervals contain "
           f"the American tree value; {sum(r['pass_bermudan'] == '1' for r in rows)}/20 "
           f"contain the Bermudan one")
    fig.savefig(RESULTS / "lsm_validation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------

def paths(rows: list) -> None:
    by_n = defaultdict(list)
    for r in rows:
        by_n[int(r["n_paths"])].append(r)
    american = f(rows[0], "american_tree")
    bermudan = f(rows[0], "bermudan_tree")
    print("\n\nERROR AGAINST PATHS (case 0: S=36, sigma=0.2, T=1; 100 seeds per N)")
    print(f"American tree {american:.4f}, Bermudan tree {bermudan:.4f}. "
          "Bias = mean over seeds minus the reference; +/- is the se of that mean.\n")
    print(f"{'N':>8} {'out bias vs Be':>17} {'in bias vs Be':>17} {'in-out gap':>16} "
          f"{'RMSE vs Be':>10} {'RMSE vs Am':>10} {'mean SE':>8} {'cov Be':>6} {'cov Am':>6} "
          f"{'s/run':>6}")
    table = []
    for n in sorted(by_n):
        group = by_n[n]
        out = np.array([f(r, "lsm_price") for r in group])
        ins = np.array([f(r, "in_sample_price") for r in group])
        ob, obse = mean_se(out - bermudan)
        ib, ibse = mean_se(ins - bermudan)
        g, gse = mean_se(ins - out)
        rmse_b = math.sqrt(np.mean((out - bermudan) ** 2))
        rmse_a = math.sqrt(np.mean((out - american) ** 2))
        se = np.mean([f(r, "lsm_se") for r in group])
        cov_b = np.mean([int(r["covers_bermudan"]) for r in group])
        cov_a = np.mean([int(r["covers_american"]) for r in group])
        runtime = np.median([f(r, "runtime_s") for r in group])
        table.append((n, ob, obse, ib, ibse, g, gse, rmse_b, rmse_a, se))
        print(f"{n:>8,} {ob:>+9.4f}±{obse:.4f} {ib:>+9.4f}±{ibse:.4f} {g:>+8.4f}±{gse:.4f} "
              f"{rmse_b:>10.4f} {rmse_a:>10.4f} {se:>8.4f} {cov_b:>6.2f} {cov_a:>6.2f} "
              f"{runtime:>6.3f}")
    ns = np.array([t[0] for t in table], dtype=float)
    rb = np.array([t[7] for t in table])
    slope = np.polyfit(np.log(ns), np.log(rb), 1)[0]
    print(f"\nLog-log slope of RMSE vs Bermudan against N: {slope:.3f} (1/sqrt(N) is -0.5)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(top=0.8, bottom=0.13, left=0.08, right=0.98, wspace=0.28)
    ax1.loglog(ns, rb, "o-", color=COLOUR["bermudan"], label="RMSE vs Bermudan tree",
               **LINE, **MARK)
    ax1.loglog(ns, [t[8] for t in table], "s-", color=COLOUR["american"],
               label="RMSE vs American tree", **LINE, **MARK)
    ax1.loglog(ns, [t[9] for t in table], "--", color=COLOUR["out"], label="mean reported SE",
               linewidth=1.5)
    ax1.axhline(abs(american - bermudan), color=MUTED, linestyle=":", linewidth=1.2)
    ax1.text(ns[-1], abs(american - bermudan) * 0.9, "American − Bermudan gap",
             fontsize=8.5, color=MUTED, va="top", ha="right")
    ax1.set_xlabel("paths (training = pricing)")
    ax1.set_ylabel("error")
    ax1.legend(fontsize=9)

    ax2.axhline(0, color=COLOUR["bermudan"], linewidth=1.2, label="Bermudan tree")
    ax2.axhline(american - bermudan, color=COLOUR["american"], linewidth=1.2,
                label="American tree")
    ax2.errorbar(ns, [t[1] for t in table], yerr=[1.96 * t[2] for t in table], fmt="o-",
                 color=COLOUR["out"], capsize=3, label="out-of-sample (mean of 100)",
                 **LINE, **MARK)
    ax2.errorbar(ns * 1.08, [t[3] for t in table], yerr=[1.96 * t[4] for t in table],
                 fmt="D-", color=COLOUR["in"], capsize=3, label="in-sample (mean of 100)",
                 **LINE, markersize=5)
    ax2.set_xscale("log")
    ax2.set_xlabel("paths (training = pricing)")
    ax2.set_ylabel("mean price minus Bermudan tree")
    ax2.legend(fontsize=9, loc="lower right")
    titles(fig, "Paths: noise falls like 1/√N; the in-sample/out-of-sample gap closes from 0.20 to 0.003",
           f"Case 0 (S=36, σ=0.2, T=1), 50 dates. RMSE slope {slope:.2f}. Bars are 95% "
           "intervals for the mean over seeds.")
    fig.savefig(RESULTS / "lsm_paths.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Dates
# ---------------------------------------------------------------------------

def dates(rows: list) -> None:
    by_m = defaultdict(list)
    for r in rows:
        by_m[int(r["n_dates"])].append(r)
    american = f(rows[0], "american_tree")
    european = f(rows[0], "european")
    print("\n\nERROR AGAINST EXERCISE DATES (case 0, 100k + 100k paths, 20 seeds per M)")
    print(f"American tree {american:.4f}, European {european:.4f}, intrinsic 4.0000\n")
    print(f"{'M':>4} {'Bermudan':>9} {'Am - Be':>8} {'LSM mean':>9} {'LSM - Be':>16} "
          f"{'in - Be':>9} {'cov Be':>6} {'cov Am':>6}")
    table = []
    for m in sorted(by_m):
        group = by_m[m]
        bermudan = f(group[0], "bermudan_tree")
        out = np.array([f(r, "lsm_price") for r in group])
        ins = np.array([f(r, "in_sample_price") for r in group])
        d, dse = mean_se(out - bermudan)
        table.append((m, bermudan, out.mean(), dse))
        print(f"{m:>4} {bermudan:>9.4f} {american - bermudan:>8.4f} {out.mean():>9.4f} "
              f"{d:>+9.4f}±{dse:.4f} {ins.mean() - bermudan:>+9.4f} "
              f"{np.mean([int(r['covers_bermudan']) for r in group]):>6.2f} "
              f"{np.mean([int(r['covers_american']) for r in group]):>6.2f}")

    ms = np.array([t[0] for t in table], dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(top=0.8, bottom=0.13, left=0.08, right=0.98, wspace=0.28)
    ax1.axhline(american, color=COLOUR["american"], linewidth=1.2, label="American tree")
    ax1.axhline(european, color=COLOUR["european"], linewidth=1.2, linestyle="--",
                label="European (Black-Scholes)")
    ax1.plot(ms, [t[1] for t in table], "-", color=COLOUR["bermudan"],
             label="Bermudan tree, M dates", **LINE)
    ax1.errorbar(ms, [t[2] for t in table], yerr=[1.96 * t[3] for t in table], fmt="o",
                 color=COLOUR["out"], capsize=3, label="LSM out-of-sample (mean of 20)",
                 **MARK)
    ax1.set_xscale("log")
    ax1.set_xlabel("exercise dates M")
    ax1.set_ylabel("price")
    ax1.legend(fontsize=9, loc="center right")

    ax2.loglog(ms, [american - t[1] for t in table], "o-", color=COLOUR["bermudan"],
               label="American − Bermudan tree", **LINE, **MARK)
    slope = np.polyfit(np.log(ms[ms >= 10]), np.log([american - t[1] for t in table
                                                    if t[0] >= 10]), 1)[0]
    print(f"\nLog-log slope of (American - Bermudan) against M, M >= 10: {slope:.3f}")
    ax2.set_xlabel("exercise dates M")
    ax2.set_ylabel("value lost by exercising on M dates only")
    ax2.legend(fontsize=9)
    titles(fig, "Exercise dates: LSM tracks the Bermudan contract, not the American one",
           f"Case 0 (S=36, σ=0.2, T=1). Discrete-exercise gap falls with slope {slope:.2f} in M. "
           "One date: the holder exercises now (4.00).")
    fig.savefig(RESULTS / "lsm_dates.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Basis
# ---------------------------------------------------------------------------

def basis(rows: list) -> None:
    print("\n\nBASIS (20 seeds per basis, 100k + 100k paths)")
    print("Out-of-sample is a lower bound: a better basis gives a HIGHER out-of-sample price.\n")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case"], int(r["degree"]), int(r["weighted"]))].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.08, right=0.98, wspace=0.28)
    for ax, case in zip(axes, ("case0", "case3")):
        keys = sorted(k for k in groups if k[0] == case)
        keys = [k for k in keys if not k[2]] + [k for k in keys if k[2]]
        bermudan = f(groups[keys[0]][0], "bermudan_tree")
        american = f(groups[keys[0]][0], "american_tree")
        print(f"{case}: Bermudan {bermudan:.4f}, American {american:.4f}")
        print(f"  {'basis':>12} {'out - Be':>16} {'in - Be':>16} {'in - out':>9}")
        labels, outs, out_se, ins = [], [], [], []
        for key in keys:
            group = groups[key]
            out = np.array([f(r, "lsm_price") for r in group])
            inn = np.array([f(r, "in_sample_price") for r in group])
            o, ose = mean_se(out - bermudan)
            i, ise = mean_se(inn - bermudan)
            label = f"{key[1]}" + (" (LS)" if key[2] else "")
            print(f"  {('degree ' + label):>12} {o:>+9.4f}±{ose:.4f} {i:>+9.4f}±{ise:.4f} "
                  f"{(inn - out).mean():>+9.4f}")
            labels.append(label)
            outs.append(o)
            out_se.append(ose)
            ins.append(i)
        x = np.arange(len(keys))
        ax.axhline(0, color=COLOUR["bermudan"], linewidth=1.2, label="Bermudan tree")
        ax.errorbar(x, outs, yerr=1.96 * np.array(out_se), fmt="o", color=COLOUR["out"],
                    capsize=3, label="out-of-sample (mean of 20)", **MARK)
        ax.plot(x + 0.15, ins, "D", color=COLOUR["in"], markersize=5,
                label="in-sample (mean of 20)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Laguerre degree  (LS = paper's weighted form, 4 columns)")
        ax.set_yscale("symlog", linthresh=0.02)
        ax.set_ylabel("mean price minus Bermudan tree (linear inside ±0.02)")
        S, sigma, T = {"case0": (36, 0.2, 1), "case3": (36, 0.4, 2)}[case]
        ax.set_title(f"S={S}, σ={sigma}, T={T}", fontsize=10, color=INK_SECONDARY)
        ax.legend(fontsize=9, loc="lower right")
    titles(fig, "Basis: the price climbs until degree 3, then stops",
           "Out-of-sample is a lower bound, so higher is better. Bars are 95% intervals for the mean over 20 seeds.")
    fig.savefig(RESULTS / "lsm_basis.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Boundary
# ---------------------------------------------------------------------------

def boundary(tree_rows: list, lsm_rows: list) -> None:
    print("\n\nEXERCISE BOUNDARY (tree at 5,000 steps vs 20 independently fitted LSM rules)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.subplots_adjust(top=0.8, bottom=0.13, left=0.08, right=0.98, wspace=0.08)
    for ax, case in zip(axes, ("case0", "case2")):
        tt = np.array([f(r, "t") for r in tree_rows if r["case"] == case])
        tb = np.array([f(r, "boundary") for r in tree_rows if r["case"] == case])
        per_seed = defaultdict(list)
        for r in lsm_rows:
            if r["case"] == case:
                per_seed[r["seed"]].append((f(r, "t"), f(r, "boundary")))
        times = np.array([t for t, _ in next(iter(per_seed.values()))])
        stack = np.array([[b for _, b in v] for v in per_seed.values()])
        median = np.nanmedian(stack, axis=0)
        low, high = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
        tree_at = np.interp(times, tt[np.isfinite(tb)], tb[np.isfinite(tb)])
        diff = median - tree_at
        spread = high - low
        print(f"{case}: median LSM minus tree, by thirds of the year: "
              + ", ".join(f"{np.nanmean(diff[s]):+.3f}" for s in
                          (slice(0, 16), slice(16, 33), slice(33, None)))
              + f"; mean |diff| {np.nanmean(np.abs(diff)):.3f}; "
              f"seed spread (max-min) median {np.nanmedian(spread):.3f}, "
              f"largest {np.nanmax(spread):.3f} at t={times[np.nanargmax(spread)]:.2f}")
        for row in stack:
            ax.plot(times, row, color=COLOUR["out"], alpha=0.18, linewidth=1)
        ax.plot(times, median, color=COLOUR["out"], label="LSM, median of 20 fitted rules", **LINE)
        ax.plot(tt, tb, color=COLOUR["american"], linewidth=1.4, label="American tree")
        ax.axhline(40, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xlabel("time (years)")
        sigma = {"case0": 0.2, "case2": 0.4}[case]
        ax.set_title(f"S=36, σ={sigma}, T=1", fontsize=10, color=INK_SECONDARY)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("critical stock price S*(t): exercise below it")
    titles(fig, "The early-exercise boundary: LSM against the tree",
           "Thin lines: 20 rules fitted on independent seeds. Strike 40 dotted.")
    fig.savefig(RESULTS / "lsm_boundary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Call
# ---------------------------------------------------------------------------

def call(rows: list) -> None:
    print("\n\nAMERICAN CALL, NO DIVIDENDS (sigma=0.2, T=1, 20 seeds per spot)")
    print("Early exercise is never optimal, so LSM should equal the European call.\n")
    print(f"{'S':>4} {'BS call':>8} {'LSM mean':>9} {'cov':>5} {'exercised early':>16} "
          f"{'paired cost':>18} {'mean SE':>8}")
    by_s = defaultdict(list)
    for r in rows:
        by_s[f(r, "S")].append(r)
    for S in sorted(by_s):
        group = by_s[S]
        cost, cost_se = mean_se([f(r, "paired_cost") for r in group])
        print(f"{S:>4.0f} {f(group[0], 'european_bs'):>8.4f} "
              f"{np.mean([f(r, 'lsm_price') for r in group]):>9.4f} "
              f"{np.mean([int(r['covers_european']) for r in group]):>5.2f} "
              f"{np.mean([f(r, 'fraction_exercised_early') for r in group]):>15.3%} "
              f"{cost:>+10.5f}±{cost_se:.5f} "
              f"{np.mean([f(r, 'lsm_se') for r in group]):>8.4f}")


def main() -> None:
    validation(read_csv("lsm_validation.csv"))
    paths(read_csv("lsm_paths.csv"))
    dates(read_csv("lsm_dates.csv"))
    basis(read_csv("lsm_basis.csv"))
    boundary(read_csv("lsm_boundary_tree.csv"), read_csv("lsm_boundary_lsm.csv"))
    call(read_csv("lsm_call.csv"))
    print("\nCharts written to results/lsm_*.png")


if __name__ == "__main__":
    main()
