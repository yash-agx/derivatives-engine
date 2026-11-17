"""Read the Phase 6 CSVs in results/, print the tables and draw the charts.

    python analysis/plot_phase6.py

Reads only the CSVs, so the figures always show exactly the numbers that were
saved and can be redrawn without re-running the inversion.

Two figures:

``iv_smile.png``             implied volatility against forward moneyness
                             ``K/F``, one line per expiry: the whole surviving
                             strike range, and a zoom on the liquid core where
                             the headline numbers are read off.
``iv_forward_diagnostic.png`` what the textbook construction costs: the same
                             quotes inverted against spot and the T-bill rate
                             with no dividends, which tears a step into the
                             middle of every smile.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

matplotlib.use("Agg")                   # write straight to files, no window needed
import matplotlib.pyplot as plt         # noqa: E402

# Shared with the Phase 2-5 charts.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

#: One colour per expiry, nearest first. Nearest gets the warm colour because
#: it is the one that behaves differently and the eye should find it.
EXPIRY_COLOUR = ["#eb6834", "#2a78d6", "#1baf7a"]

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

LINE = {"linewidth": 1.8}
MARK = {"markersize": 4.5, "markeredgecolor": SURFACE, "markeredgewidth": 0.6}

#: The zoom window of the second panel. Chosen because it is the range where
#: the strikes are liquid on every expiry and one price tick is worth well
#: under a tenth of a vol point, so the shape is signal rather than rounding.
CORE = (0.90, 1.10)


def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.928, subtitle, fontsize=10, color=INK_SECONDARY, va="top")


def label_for(row: dict) -> str:
    return f"{row['expiry']}  ({row['days_to_expiry']}d)"


def by_expiry(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["expiry"]].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: num(r["forward_moneyness"]))
    return dict(grouped)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_inputs(summary: list[dict]) -> None:
    print("\n\nPHASE 6 -- IMPLIED VOLATILITY, NIFTY 50 OPTIONS, 16 NOV 2026")
    print("Spot 25,291.10. r = 5.8255% continuously compounded (91-day T-bill,")
    print("cut-off price 98.5581, RBI auction 11 Nov 2026). T = calendar days / 365.")
    print("Priced against the forward put-call parity reads out of the quotes")
    print("themselves, so moneyness is K/F. Out-of-the-money only: puts below")
    print("the forward, calls above.\n")
    print(f"{'expiry':>12} {'days':>5} {'strikes':>8} {'forward':>11} "
          f"{'K/F range':>18}")
    for row in summary:
        span = f"{num(row['moneyness_min']):.3f} - {num(row['moneyness_max']):.3f}"
        print(f"{row['expiry']:>12} {row['days_to_expiry']:>5} "
              f"{row['strikes_kept']:>8} {num(row['parity_forward']):>11,.1f} "
              f"{span:>18}")


def print_filters(filters: list[dict]) -> None:
    print("\n\nWHAT WAS DROPPED, AND WHY")
    print("Filters are applied in this order, so each contract is counted once.\n")
    stages = []
    for row in filters:
        if row["filter"] not in stages:
            stages.append(row["filter"])
    expiries = []
    for row in filters:
        if row["expiry"] not in expiries:
            expiries.append(row["expiry"])

    table = {(r["expiry"], r["filter"]): r for r in filters}
    print(f"{'filter':>24} " + " ".join(f"{e:>12}" for e in expiries))
    listed = [table[(e, stages[0])]["contracts_listed"] for e in expiries]
    print(f"{'contracts listed':>24} " + " ".join(f"{n:>12}" for n in listed))
    for stage in stages:
        cells = [table[(e, stage)]["dropped"] for e in expiries]
        print(f"{stage:>24} " + " ".join(f"{-int(n) if int(n) else 0:>12}"
                                         for n in cells))
    kept = [table[(e, stages[-1])]["kept_after"] for e in expiries]
    print(f"{'kept':>24} " + " ".join(f"{n:>12}" for n in kept))


def print_smile_numbers(summary: list[dict]) -> None:
    print("\n\nTHE SMILE IN NUMBERS (implied volatility, % annualised)")
    print("Forward moneyness K/F. At 1.00 the data has two values, not one:")
    print("the last put below the forward and the first call above it, so the")
    print("gap between them is a measurement rather than a choice.\n")
    print(f"{'expiry':>12} {'0.95':>8} {'1.00 put':>10} {'1.00 call':>10} "
          f"{'1.05':>8} {'0.95-1.05':>11} {'ATM gap':>10}")
    for row in summary:
        print(f"{row['expiry']:>12} "
              f"{num(row['iv_at_0_95']) * 100:>8.2f} "
              f"{num(row['iv_atm_put_side']) * 100:>10.2f} "
              f"{num(row['iv_atm_call_side']) * 100:>10.2f} "
              f"{num(row['iv_at_1_05']) * 100:>8.2f} "
              f"{num(row['skew_095_minus_105']) * 100:>10.2f}pt "
              f"{num(row['atm_jump_put_minus_call']) * 100:>+9.2f}pt")


def print_forward_diagnostic(summary: list[dict]) -> None:
    print("\n\nWHAT THE TEXTBOOK CONSTRUCTION WOULD HAVE COST")
    print("Black-Scholes with no dividends assumes the forward is S*exp(r*T).")
    print("Put-call parity says what forward the option market is quoting, and")
    print("the NIFTY future says it independently. Where they differ, the")
    print("at-the-money strike is not where the model thinks it is, and the")
    print("two branches of the smile stop meeting.\n")
    print(f"{'expiry':>12} {'parity F':>11} {'future':>11} {'S*exp(rT)':>11} "
          f"{'gap':>8} {'ATM gap':>10} {'on spot':>10}")
    for row in summary:
        print(f"{row['expiry']:>12} "
              f"{num(row['parity_forward']):>11.1f} "
              f"{num(row['futures_close']):>11.1f} "
              f"{num(row['spot_forward']):>11.1f} "
              f"{num(row['forward_gap_points']):>+8.1f} "
              f"{num(row['atm_jump_put_minus_call']) * 100:>+9.2f}pt "
              f"{num(row['spot_atm_jump_put_minus_call']) * 100:>+9.2f}pt")

    print("\nThe same smile read off both ways (0.95 / ATM / 1.05, vol points):\n")
    print(f"{'expiry':>12} {'forward basis':>28} {'spot basis':>28}")
    for row in summary:
        forward = (f"{num(row['iv_at_0_95']) * 100:.2f} / "
                   f"{num(row['iv_atm_midpoint']) * 100:.2f} / "
                   f"{num(row['iv_at_1_05']) * 100:.2f}")
        spot = (f"{num(row['spot_iv_at_0_95']) * 100:.2f} / "
                f"{0.5 * (num(row['spot_iv_atm_put_side'])
                          + num(row['spot_iv_atm_call_side'])) * 100:.2f} / "
                f"{num(row['spot_iv_at_1_05']) * 100:.2f}")
        print(f"{row['expiry']:>12} {forward:>28} {spot:>28}")


def print_solver_cost(summary: list[dict]) -> None:
    print("\n\nWHAT THE INVERSION COST")
    print("Evaluations of the Black-Scholes price per strike, and how many of")
    print("the solver's moves were Newton's rather than the bisection fallback.\n")
    print(f"{'expiry':>12} {'strikes':>8} {'mean evals':>11} {'max':>5} "
          f"{'newton':>8} {'bisection':>10} {'bisect %':>9} "
          f"{'median vol pts/tick':>20}")
    for row in summary:
        newton = int(row["newton_steps"])
        bisect = int(row["bisection_steps"])
        share = 100.0 * bisect / (newton + bisect) if newton + bisect else 0.0
        print(f"{row['expiry']:>12} {row['strikes_kept']:>8} "
              f"{num(row['mean_iterations']):>11.2f} {row['max_iterations']:>5} "
              f"{newton:>8} {bisect:>10} {share:>8.1f}% "
              f"{num(row['median_iv_per_tick']) * 100:>20.4f}")


# ---------------------------------------------------------------------------
# Figure 1: the smile
# ---------------------------------------------------------------------------

def plot_smile(smile: list[dict], summary: list[dict]) -> None:
    grouped = by_expiry(smile)
    order = [row["expiry"] for row in summary]

    fig, (full, core) = plt.subplots(1, 2, figsize=(12.4, 5.4))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.11,
                        wspace=0.18)

    for index, expiry in enumerate(order):
        rows = grouped[expiry]
        colour = EXPIRY_COLOUR[index % len(EXPIRY_COLOUR)]
        label = f"{expiry}  ({rows[0]['days_to_expiry']} days)"

        for axis in (full, core):
            # Puts and calls drawn as separate segments on purpose: they are
            # different contracts, and joining them across moneyness 1.0 would
            # draw a line through a gap that is one of the findings.
            for kind, marker in (("put", "o"), ("call", "s")):
                branch = [r for r in rows if r["kind"] == kind]
                if not branch:
                    continue
                axis.plot([num(r["forward_moneyness"]) for r in branch],
                          [num(r["implied_vol"]) * 100 for r in branch],
                          marker=marker, color=colour,
                          label=label if kind == "put" else None,
                          **LINE, **MARK)

    for axis in (full, core):
        axis.axvline(1.0, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)),
                     zorder=0)
        axis.set_xlabel("forward moneyness  K / F")
    full.set_ylabel("implied volatility  (% annualised)")
    core.set_xlim(*CORE)

    # The zoom window needs its own y-limit: the full panel is dominated by a
    # 76% wing that would flatten the core into a horizontal line.
    core_values = [num(r["implied_vol"]) * 100 for r in smile
                   if CORE[0] <= num(r["forward_moneyness"]) <= CORE[1]]
    pad = 0.06 * (max(core_values) - min(core_values))
    core.set_ylim(min(core_values) - pad, max(core_values) + pad)

    full.set_title("every strike that survived the filters", fontsize=10,
                   color=INK_SECONDARY, loc="left", pad=8)
    core.set_title(f"the liquid core, {CORE[0]:.2f} to {CORE[1]:.2f}",
                   fontsize=10, color=INK_SECONDARY, loc="left", pad=8)
    full.legend(loc="upper right", fontsize=9)

    full.text(0.98, 0.74, "circles: puts      squares: calls", fontsize=9,
              color=MUTED, transform=full.transAxes, ha="right", va="top")

    titles(fig, "Nifty 50 implied volatility is not one number",
           "16 November 2026 close. Out-of-the-money options only, inverted from "
           "the exchange's own closing prices against the forward\nput-call parity "
           "reads out of those same prices. If Black-Scholes' constant volatility "
           "held, each line would be flat.")
    out = RESULTS / "iv_smile.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"\n  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2: the forward diagnostic
# ---------------------------------------------------------------------------

def plot_forward_diagnostic(smile: list[dict], summary: list[dict]) -> None:
    grouped = by_expiry(smile)
    order = [row["expiry"] for row in summary]
    lookup = {row["expiry"]: row for row in summary}

    fig, axes = plt.subplots(1, len(order), figsize=(12.4, 4.9), sharey=False)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.72, bottom=0.13,
                        wspace=0.22)

    for axis, expiry in zip(axes, order):
        rows = grouped[expiry]
        info = lookup[expiry]
        window = [r for r in rows if 0.97 <= num(r["forward_moneyness"]) <= 1.03]

        for kind, marker in (("put", "o"), ("call", "s")):
            branch = [r for r in window if r["kind"] == kind]
            if not branch:
                continue
            axis.plot([num(r["forward_moneyness"]) for r in branch],
                      [num(r["implied_vol"]) * 100 for r in branch],
                      marker=marker, color=BLUE,
                      label="market's forward (used)" if kind == "put" else None,
                      **LINE, **MARK)
            axis.plot([num(r["spot_moneyness"]) for r in branch],
                      [num(r["implied_vol_spot"]) * 100 for r in branch],
                      marker=marker, color=ORANGE,
                      label="spot, no dividends" if kind == "put" else None,
                      **LINE, **MARK)

        axis.axvline(1.0, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)),
                     zorder=0)
        gap = num(info["atm_jump_put_minus_call"]) * 100
        spot_gap = num(info["spot_atm_jump_put_minus_call"]) * 100
        axis.set_title(
            f"{expiry}  ({info['days_to_expiry']} days)\n"
            f"forward gap {num(info['forward_gap_points']):+.1f} pts\n"
            f"ATM step {gap:+.2f} pt on F,  {spot_gap:+.2f} pt on S",
            fontsize=9, color=INK_SECONDARY, loc="left", pad=8)
        # Each series is drawn against its own moneyness, because that is the
        # point: the two constructions disagree about where "at the money" is.
        axis.set_xlabel("moneyness   K/F in blue,  K/S in orange")

    axes[0].set_ylabel("implied volatility  (% annualised)")
    axes[0].legend(loc="upper right", fontsize=9)

    titles(fig, "What the no-dividend assumption would have cost",
           "The same quotes, inverted two ways. Against the spot and the T-bill "
           "rate the put branch and the call branch stop meeting;\nagainst the "
           "forward those same prices imply, they join -- and the sign of the step "
           "follows the sign of the forward gap.")
    out = RESULTS / "iv_forward_diagnostic.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def main() -> None:
    smile = read_csv("iv_smile.csv")
    summary = read_csv("iv_term_structure.csv")
    filters = read_csv("iv_filter_log.csv")

    print_inputs(summary)
    print_filters(filters)
    print_smile_numbers(summary)
    print_forward_diagnostic(summary)
    print_solver_cost(summary)

    plot_smile(smile, summary)
    plot_forward_diagnostic(smile, summary)


if __name__ == "__main__":
    main()
