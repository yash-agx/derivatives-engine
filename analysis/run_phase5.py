"""Benchmark the Monte Carlo core across four implementations.

    python analysis/run_phase5.py          (about 2 minutes)

The same algorithm -- price a European option by averaging discounted payoffs
from terminal prices drawn in one step -- written four ways:

1. **pure Python loop**: one path at a time, in the interpreter.
2. **NumPy vectorised**: the whole batch as array operations.
3. **C++ single-threaded**: the same arithmetic, compiled, streaming.
4. **C++ multithreaded**: the same again, across cores.

All four are run at equal path counts, so what is being compared is the cost of
the machinery rather than the amount of work, and the repeats are **interleaved**
-- one round of every implementation, then the next -- because running all the
repeats of one implementation together measures it partly against the machine's
clock state at that moment. That mistake cost a factor of 1.9 on a first pass. Two extra things are measured
because Phase 3 made them interesting:

* **per-path cost against batch size.** Phase 3 found this U-shaped in NumPy:
  fixed per-call overhead at small N, cache pressure at large N. The C++ core
  keeps no arrays at all -- it accumulates with Welford as it goes -- so it
  should not have the large-N half of that curve.
* **cost = variance x runtime**, the Phase 3/4 efficiency metric, which is what
  actually matters: a faster implementation buys accuracy more cheaply even
  though its variance per path is identical.

Writes ``perf_benchmark.csv``, ``perf_threads.csv`` and ``perf_methods.csv``
(all four estimators, Python against C++, since the C++ core implements the
Phase 4 methods too).
"""

import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))      # the package is not installed; see pytest.ini

from pricing.black_scholes import call_price                        # noqa: E402
from pricing.cpp_backend import (                                   # noqa: E402
    cpp_available,
    cpp_monte_carlo,
    engine_version,
    hardware_threads,
)
from pricing.monte_carlo import monte_carlo_price                   # noqa: E402
from pricing.variance_reduction import (                            # noqa: E402
    antithetic_monte_carlo,
    control_variate_monte_carlo,
    importance_sampling_monte_carlo,
)

RESULTS = ROOT / "results"

OPTION = (100.0, 100.0, 0.05, 0.20, 1.0)       # S, K, r, sigma, T
KIND = "call"
SEED = 12345

PATH_COUNTS = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
# The interpreter loop is roughly 1,000x slower than the compiled core, so
# running it at 10^7 would take minutes on its own. It is capped and the gap is
# extrapolated instead, which is noted in the results.
PYTHON_LOOP_MAX = 1_000_000
THREAD_COUNTS = [1, 2, 4, 8, 16, 24]
THREAD_PATHS = 10_000_000


def repeats_for(n_paths: int) -> int:
    """More repeats where a run is short and the timer is relatively coarse."""
    if n_paths <= 10_000:
        return 9
    if n_paths <= 1_000_000:
        return 5
    return 3


def python_loop_price(S, K, r, sigma, T, n_paths, seed):
    """The same estimator, one path at a time, in pure Python.

    Deliberately written the obvious way -- this is the baseline the other
    three are trying to beat, and making it clever would misrepresent the gap.
    Uses Welford so that it is doing the same arithmetic as the C++ version
    rather than a less accurate sum of squares.
    """
    rng = random.Random(seed)
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    count = 0
    mean = 0.0
    m2 = 0.0
    for _ in range(n_paths):
        spot = S * math.exp(drift + vol * rng.gauss(0.0, 1.0))
        value = discount * (spot - K if spot > K else 0.0)
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    standard_error = math.sqrt(m2 / (count - 1) / count)
    return mean, standard_error


def numpy_price(S, K, r, sigma, T, n_paths, seed):
    result = monte_carlo_price(S, K, r, sigma, T, KIND, n_paths=n_paths, seed=seed)
    return result.price, result.standard_error


def cpp_price(S, K, r, sigma, T, n_paths, seed, threads):
    result = cpp_monte_carlo(S, K, r, sigma, T, KIND, n_paths=n_paths, seed=seed,
                             method="plain", n_threads=threads)
    return result.price, result.standard_error


def time_once(function, n_paths: int) -> tuple:
    """One timed call, plus whatever it produced."""
    started = time.perf_counter()
    price, standard_error = function(*OPTION, n_paths, SEED)
    return time.perf_counter() - started, price, standard_error


def write_csv(name: str, fieldnames: list, rows: list) -> None:
    path = RESULTS / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def run_benchmark() -> None:
    threads = hardware_threads()
    implementations = [
        ("python loop", lambda *a: python_loop_price(*a), PYTHON_LOOP_MAX),
        ("numpy", lambda *a: numpy_price(*a), None),
        ("c++ 1 thread", lambda *a: cpp_price(*a, 1), None),
        (f"c++ {threads} threads", lambda *a: cpp_price(*a, threads), None),
    ]
    truth = call_price(*OPTION)
    print(f"1/3 benchmark: {len(implementations)} implementations x "
          f"{len(PATH_COUNTS)} path counts")
    print(f"    {engine_version()}, {threads} hardware threads")
    print(f"    Black-Scholes price {truth:.6f}\n")

    # Interleaved: one repeat of every (implementation, path count) per round,
    # so that drifting clock speeds hit all of them alike rather than
    # penalising whichever happened to run while the machine was hot.
    jobs = [(label, function, n_paths)
            for label, function, cap in implementations
            for n_paths in PATH_COUNTS
            if cap is None or n_paths <= cap]

    timings = {key: [] for key in ((label, n) for label, _, n in jobs)}
    outputs = {}
    rounds = max(repeats_for(n) for _, _, n in jobs)
    for round_index in range(rounds):
        for label, function, n_paths in jobs:
            if round_index >= repeats_for(n_paths):
                continue
            elapsed, price, standard_error = time_once(function, n_paths)
            timings[(label, n_paths)].append(elapsed)
            outputs[(label, n_paths)] = (price, standard_error)
        print(f"  round {round_index + 1}/{rounds} done")

    rows = []
    baseline = {}
    for label, _, n_paths in jobs:
        samples = timings[(label, n_paths)]
        median = statistics.median(samples)
        # Headline numbers use the FASTEST repeat. Background load and clock
        # drift can only ever make a run slower, so the minimum is the cleanest
        # estimate of what the implementation costs; the median and the full
        # spread are kept in the CSV so the noise is visible.
        fastest = min(samples)
        price, standard_error = outputs[(label, n_paths)]
        throughput = n_paths / fastest
        if label == "python loop":
            baseline[n_paths] = fastest
        rows.append({
            "implementation": label,
            "n_paths": n_paths,
            "repeats": len(samples),
            "median_seconds": "%.9f" % median,
            "fastest_seconds": "%.9f" % min(samples),
            "slowest_seconds": "%.9f" % max(samples),
            "paths_per_second": "%.1f" % throughput,
            "ns_per_path": "%.4f" % (fastest / n_paths * 1e9),
            "ns_per_path_median": "%.4f" % (median / n_paths * 1e9),
            "price": "%.10f" % price,
            "standard_error": "%.10f" % standard_error,
            "error_vs_bs": "%.10f" % (price - truth),
            # Cost of a unit of accuracy: lower is better. The variance per
            # path is the same for every implementation, so this isolates
            # what the machinery costs.
            "variance_times_runtime": "%.6e" % (standard_error ** 2 * fastest),
        })
        print(f"  {label:>16}  N={n_paths:>10,}  {fastest * 1e3:>10.3f} ms  "
              f"{throughput / 1e6:>8.2f} M paths/s  "
              f"{fastest / n_paths * 1e9:>8.2f} ns/path  "
              f"(median {median * 1e3:.2f}, slowest {max(samples) * 1e3:.2f} ms)")

    # Speedups, computed once every implementation has been timed.
    by_key = {(r["implementation"], int(r["n_paths"])): r for r in rows}
    numpy_times = {int(r["n_paths"]): float(r["fastest_seconds"])
                   for r in rows if r["implementation"] == "numpy"}
    for row in rows:
        n_paths = int(row["n_paths"])
        fastest = float(row["fastest_seconds"])
        row["speedup_vs_python_loop"] = ("%.2f" % (baseline[n_paths] / fastest)
                                         if n_paths in baseline else "")
        row["speedup_vs_numpy"] = "%.3f" % (numpy_times[n_paths] / fastest)
    del by_key

    write_csv("perf_benchmark.csv", list(rows[0]), rows)


def run_thread_scaling() -> None:
    print(f"\n2/3 thread scaling at N={THREAD_PATHS:,}")
    counts = [t for t in THREAD_COUNTS if t <= hardware_threads()]
    samples = {t: [] for t in counts}
    outputs = {}
    for _ in range(7):
        for threads in counts:
            started = time.perf_counter()
            price, standard_error = cpp_price(*OPTION, THREAD_PATHS, SEED, threads)
            samples[threads].append(time.perf_counter() - started)
            outputs[threads] = (price, standard_error)

    rows = []
    single = None
    for threads in counts:
        price, standard_error = outputs[threads]
        median = statistics.median(samples[threads])
        fastest = min(samples[threads])
        if single is None:
            single = fastest
        speedup = single / fastest
        rows.append({
            "n_threads": threads,
            "n_paths": THREAD_PATHS,
            "repeats": len(samples[threads]),
            "median_seconds": "%.9f" % median,
            "fastest_seconds": "%.9f" % min(samples[threads]),
            "paths_per_second": "%.1f" % (THREAD_PATHS / fastest),
            "ns_per_path": "%.4f" % (fastest / THREAD_PATHS * 1e9),
            "speedup": "%.4f" % speedup,
            # How much of the ideal linear speedup is actually realised.
            "parallel_efficiency": "%.4f" % (speedup / threads),
            "price": "%.17g" % price,
            "standard_error": "%.17g" % standard_error,
        })
        print(f"  {threads:>3} threads  {fastest * 1e3:>9.3f} ms  "
              f"{THREAD_PATHS / fastest / 1e6:>7.1f} M paths/s  "
              f"speedup {speedup:>5.2f}x  efficiency {speedup / threads:>5.1%}  "
              f"price {price:.12f}")

    identical = len({row["price"] for row in rows}) == 1
    print(f"\n  every thread count produced a bitwise identical price: {identical}")
    write_csv("perf_threads.csv", list(rows[0]), rows)


METHOD_PATHS = 1_000_000
PYTHON_METHODS = {
    "plain": lambda *a, **kw: monte_carlo_price(*a, **kw),
    "antithetic": lambda *a, **kw: antithetic_monte_carlo(*a, **kw),
    "control": lambda *a, **kw: control_variate_monte_carlo(*a, **kw),
    "importance": lambda *a, **kw: importance_sampling_monte_carlo(*a, **kw),
}


def run_method_comparison() -> None:
    """All four estimators, Python against C++, at one path count.

    The variance reduction each method buys was measured in Phase 4 and does
    not depend on the language; what changes here is what the paths cost. The
    product of the two is what actually matters, so the cost metric is carried
    through.
    """
    threads = hardware_threads()
    print(f"\n3/3 all four estimators at N={METHOD_PATHS:,}: Python vs C++")
    rows = []
    for method, python_function in PYTHON_METHODS.items():
        variants = [
            ("python", lambda m=method, f=python_function: f(
                *OPTION, KIND, n_paths=METHOD_PATHS, seed=SEED)),
            ("c++ 1 thread", lambda m=method: cpp_monte_carlo(
                *OPTION, KIND, n_paths=METHOD_PATHS, seed=SEED, method=m,
                n_threads=1)),
            (f"c++ {threads} threads", lambda m=method: cpp_monte_carlo(
                *OPTION, KIND, n_paths=METHOD_PATHS, seed=SEED, method=m,
                n_threads=threads)),
        ]
        samples = {label: [] for label, _ in variants}
        outputs = {}
        for _ in range(9):
            for label, call in variants:
                started = time.perf_counter()
                result = call()
                samples[label].append(time.perf_counter() - started)
                outputs[label] = result
        python_time = min(samples["python"])
        for label, _ in variants:
            fastest = min(samples[label])
            result = outputs[label]
            rows.append({
                "method": method,
                "implementation": label,
                "n_paths": METHOD_PATHS,
                "fastest_seconds": "%.9f" % fastest,
                "median_seconds": "%.9f" % statistics.median(samples[label]),
                "paths_per_second": "%.1f" % (METHOD_PATHS / fastest),
                "ns_per_path": "%.4f" % (fastest / METHOD_PATHS * 1e9),
                "speedup_vs_python": "%.3f" % (python_time / fastest),
                "price": "%.10f" % result.price,
                "standard_error": "%.10f" % result.standard_error,
                "variance_times_runtime": "%.6e" % (result.standard_error ** 2 * fastest),
            })
            print(f"  {method:>11} {label:>16}  {fastest * 1e3:>8.2f} ms  "
                  f"{METHOD_PATHS / fastest / 1e6:>7.1f} M paths/s  "
                  f"{python_time / fastest:>5.2f}x python  "
                  f"se {result.standard_error:.6f}")
    write_csv("perf_methods.csv", list(rows[0]), rows)


def main() -> None:
    if not cpp_available():
        sys.exit('cpp/build/mcengine.dll not found. Build it first:\n'
                 '    cmake -S cpp -B cpp/build -G "MinGW Makefiles"\n'
                 '    cmake --build cpp/build')
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    run_benchmark()
    run_thread_scaling()
    run_method_comparison()
    print(f"\ndone in {time.perf_counter() - started:.0f} s")
    print("now run: python analysis/plot_phase5.py")


if __name__ == "__main__":
    main()
