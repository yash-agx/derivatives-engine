"""The C++ Monte Carlo core, called from Python through a C ABI.

Why a C-ABI DLL and ctypes, and not a Python extension module
--------------------------------------------------------------
This machine has **MSVC-built CPython** and **WinLibs g++** (MinGW-w64). A
Python extension module -- what pybind11, Cython or a hand-written
``PyMODINIT_FUNC`` produce -- links against CPython's own C runtime and uses
its internal object layout, so it really wants to be built by the same
toolchain CPython was. Mixing MinGW-built extension modules with an MSVC
CPython is a known source of crashes: the two can end up with different C
runtimes, and anything that allocates memory in one and frees it in the other
is undefined behaviour. Making it work means installing MSVC build tools or
fighting import libraries.

The C ABI avoids the whole problem. A DLL that exports plain C functions
taking plain C scalars is the one interface every Windows compiler agrees on
exactly -- it is how the operating system itself is called. Nothing about
CPython's internals appears at the boundary: no Python objects are created or
destroyed across it, no memory is allocated on one side and freed on the
other, and no C++ exception is allowed to escape. ``ctypes`` is in the
standard library, so there is nothing extra to install either.

The cost is that the boundary can only carry numbers. That is not a real
constraint here: a pricing run takes seven doubles and a few integers in, and
returns a price and a standard error out. Arrays never cross it, so the cost
per call is a few microseconds, which Phase 5 measures.

The build
---------
::

    cmake -S cpp -B cpp/build -G "MinGW Makefiles"
    cmake --build cpp/build

That produces ``cpp/build/mcengine.dll``, compiled with ``-O3`` and with the
compiler runtimes statically linked so loading it does not depend on the
compiler's own DLLs being on PATH.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from .common import (
    CALL,
    PUT,
    STRADDLE,
    check_payoff_kind,
    critical_value,
    validate_inputs,
)
from .monte_carlo import MonteCarloResult

__all__ = [
    "CppUnavailable",
    "cpp_available",
    "load_engine",
    "cpp_monte_carlo",
    "hardware_threads",
    "engine_version",
    "METHODS",
]

#: Method codes, matching the enum in cpp/mc_engine.cpp.
METHODS = {"plain": 0, "antithetic": 1, "control": 2, "importance": 3}
_KIND_CODES = {CALL: 0, PUT: 1, STRADDLE: 2}

_ERRORS = {
    -1: "null output pointer",
    -2: "invalid market inputs (need S > 0, K > 0, sigma >= 0, T >= 0)",
    -3: "n_paths too small (need at least 2 independent units)",
    -4: "unknown method code",
    -5: "unknown option kind code",
    -6: "n_paths must be even for antithetic pairs",
    -7: "importance sampling needs sigma > 0 and T > 0",
    -8: "not enough samples to form a standard error",
}

_LIBRARY_PATH = Path(__file__).resolve().parents[2] / "cpp" / "build" / "mcengine.dll"
_engine = None


class CppUnavailable(RuntimeError):
    """The compiled engine is not present, so the C++ estimators cannot run."""


def load_engine():
    """Load ``mcengine.dll`` once and cache it.

    Raises ``CppUnavailable`` with the build command if it is missing, so the
    tests can skip rather than fail on a machine where it has not been built.
    """
    global _engine
    if _engine is not None:
        return _engine
    if not _LIBRARY_PATH.exists():
        raise CppUnavailable(
            f"{_LIBRARY_PATH} not found. Build it with:\n"
            f'    cmake -S cpp -B cpp/build -G "MinGW Makefiles"\n'
            f"    cmake --build cpp/build"
        )
    library = ctypes.CDLL(str(_LIBRARY_PATH))
    library.mc_run.restype = ctypes.c_int
    library.mc_run.argtypes = [
        ctypes.c_int,                      # method
        ctypes.c_double, ctypes.c_double,  # S, K
        ctypes.c_double, ctypes.c_double,  # r, sigma
        ctypes.c_double,                   # T
        ctypes.c_int,                      # kind
        ctypes.c_longlong,                 # n_paths
        ctypes.c_ulonglong,                # seed
        ctypes.c_int, ctypes.c_int,        # n_threads, n_chunks
        ctypes.POINTER(ctypes.c_double),   # out price
        ctypes.POINTER(ctypes.c_double),   # out standard error
    ]
    library.mc_hardware_threads.restype = ctypes.c_int
    library.mc_hardware_threads.argtypes = []
    library.mc_version.restype = ctypes.c_char_p
    library.mc_version.argtypes = []
    _engine = library
    return _engine


def cpp_available() -> bool:
    """True if the compiled engine can be loaded."""
    try:
        load_engine()
    except (CppUnavailable, OSError):
        return False
    return True


def hardware_threads() -> int:
    return int(load_engine().mc_hardware_threads())


def engine_version() -> str:
    return load_engine().mc_version().decode("ascii")


def cpp_monte_carlo(
    S: float, K: float, r: float, sigma: float, T: float, kind: str,
    n_paths: int = 100_000, seed: int = 12345, confidence: float = 0.95,
    method: str = "plain", n_threads: int = 1, n_chunks: int = 64,
) -> MonteCarloResult:
    """Price with the C++ engine, returning the same result type as Python.

    Returning a ``MonteCarloResult`` is the whole point: these estimators then
    drop into the Phase 3 harness unchanged, so the same code that checked the
    Python ones for bias, standard-error honesty and coverage checks these.

    ``n_chunks`` is the number of fixed work units the paths are divided into.
    It is deliberately independent of ``n_threads``: holding it constant keeps
    the floating-point reduction order the same, so results are bitwise
    identical however many threads run.
    """
    validate_inputs(S, K, r, sigma, T)
    check_payoff_kind(kind)
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")
    if int(n_paths) != n_paths or n_paths < 2:
        raise ValueError(f"n_paths must be a whole number >= 2, got {n_paths!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    engine = load_engine()
    price = ctypes.c_double()
    standard_error = ctypes.c_double()
    status = engine.mc_run(
        METHODS[method], float(S), float(K), float(r), float(sigma), float(T),
        _KIND_CODES[kind], int(n_paths), int(seed), int(n_threads), int(n_chunks),
        ctypes.byref(price), ctypes.byref(standard_error),
    )
    if status != 0:
        raise ValueError(f"C++ engine rejected the inputs: "
                         f"{_ERRORS.get(status, f'code {status}')}")

    half_width = critical_value(confidence) * standard_error.value
    return MonteCarloResult(
        price=price.value,
        standard_error=standard_error.value,
        ci_low=price.value - half_width,
        ci_high=price.value + half_width,
        n_paths=int(n_paths),
        seed=int(seed),
        confidence=confidence,
    )


def make_estimator(method: str, n_threads: int = 1, n_chunks: int = 64):
    """Wrap one C++ method into the harness's ``(option, n_paths, seed)`` shape."""
    def estimator(option, n_paths, seed):
        return cpp_monte_carlo(*option.args, option.kind, n_paths=n_paths,
                               seed=seed, method=method, n_threads=n_threads,
                               n_chunks=n_chunks)
    estimator.__name__ = f"cpp_{method}_{n_threads}t"
    return estimator


def importance_shift(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """The shift the C++ engine applies, mirrored here for tests: ``-d2``."""
    return -(math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
