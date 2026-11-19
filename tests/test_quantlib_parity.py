"""Phase 8: the engine against QuantLib, an independent implementation.

Skipped as a whole when QuantLib is not installed. The comparison code is
imported from ``analysis/run_phase8_quantlib.py`` so the tests and the
experiment cannot drift apart.

Tolerances, and where each comes from:

* Black-Scholes and implied volatility: 1e-6 as required. Both are the same
  closed form evaluated two ways, so anything above rounding is a bug.
* CRR tree, same steps: 1e-6 is **not** met (about 1e-3 at 500 steps), and
  that is kept as a strict expected failure rather than loosened. QuantLib's
  "crr" tree uses a linearised up-probability where ours uses the exact one;
  the test that our backward induction *with QuantLib's probability* matches
  to 1e-6 is what shows the gap is that convention and nothing else.
* American tree (10,000 steps) against QuantLib FD: 1e-3 absolute. The
  tree's own O(1/N) error reaches 6.6e-4 at sigma*sqrt(T) = 0.71, and the FD
  reference's own error on Europeans is up to 1.4e-4.
* LSM put against FD: within 3 standard errors, **chosen after seeing the
  worst case** (z = -2.79), so it guards against regressions rather than being
  an independent pass. LSM is biased low by construction (Bermudan on 50
  dates, and any fitted rule is suboptimal), so it is also checked one-sided:
  never more than 2 standard errors *above* the American value.
"""

import math
import sys
from pathlib import Path

import pytest

ql = pytest.importorskip("QuantLib")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import run_phase8_quantlib as p8  # noqa: E402

EXACT = 1e-6

# A spread of the grid for the slow comparisons (FD costs ~2 s per option):
# both kinds, every spot, short and long, and the worst case of each section.
SLOW_CASES = [
    (80.0, 1.0, 0.30, "put"),
    (100.0, 0.25, 0.15, "put"),
    (100.0, 2.0, 0.50, "call"),     # worst tree-vs-FD gap: the tree's own O(1/N) error
    (120.0, 0.25, 0.15, "put"),     # worst LSM z-score
    (120.0, 2.0, 0.15, "put"),
    (80.0, 2.0, 0.15, "put"),       # LSM exercises at t = 0
]


@pytest.fixture(scope="module")
def fd_references():
    return p8.fd_american_references(SLOW_CASES)


# --- a. Black-Scholes ------------------------------------------------------

@pytest.mark.parametrize("row", p8.compare_european(), ids=lambda r: r["option"])
def test_black_scholes_matches_analytic_engine(row):
    assert row["abs_error"] < EXACT


# --- b. CRR tree -----------------------------------------------------------

@pytest.mark.parametrize("row", p8.compare_linearised_tree(), ids=lambda r: r["option"])
def test_tree_with_quantlib_probability_matches_quantlib_crr(row):
    """Same lattice, same backward induction, QuantLib's p: identical prices."""
    assert row["abs_error"] < EXACT


@pytest.mark.parametrize("row", p8.compare_binomial(), ids=lambda r: r["option"])
def test_tree_agrees_with_quantlib_crr_to_first_order(row):
    """The two p's differ by O(dt^1.5) per step, so prices differ by O(1/N).

    At N = 500 that is ~1e-3 at worst; 5/N = 1e-2 bounds it with room.
    """
    assert row["abs_error"] < 5.0 / p8.TREE_STEPS


@pytest.mark.xfail(strict=True, reason=(
    "QuantLib's CRR uses p = 1/2 + 1/2 (r - sigma^2/2) dt/dx; ours uses the exact "
    "p = (e^{r dt} - d)/(u - d). Max gap 1.3e-3 at 500 steps. A convention, not a bug."))
def test_tree_matches_quantlib_crr_to_1e6():
    assert max(row["abs_error"] for row in p8.compare_binomial()) < EXACT


# --- c. American tree against QuantLib FD ----------------------------------

def test_american_tree_matches_fd(fd_references):
    rows = p8.compare_american_fd(fd_references)
    worst = max(rows, key=lambda r: r["abs_error"])
    assert worst["abs_error"] < 1e-3, worst


def test_fd_reference_is_accurate_on_europeans():
    """The reference's own error, measured where the answer is known."""
    rows = p8.compare_fd_european(SLOW_CASES)
    assert max(r["abs_error"] for r in rows) < 2e-4


# --- d. LSM put against QuantLib FD ----------------------------------------

@pytest.fixture(scope="module")
def lsm_rows(fd_references):
    return p8.compare_lsm(fd_references)


def test_lsm_within_three_standard_errors(lsm_rows):
    for row in lsm_rows:
        if row["exercise_now"]:
            assert row["abs_error"] < EXACT, row      # K - S exactly, on both sides
        else:
            assert abs(row["z"]) < 3.0, row


def test_lsm_is_not_above_the_american_value(lsm_rows):
    """Out-of-sample LSM is a lower bound up to noise."""
    for row in lsm_rows:
        if not math.isnan(row["z"]):
            assert row["z"] < 2.0, row


# --- e. Implied volatility on the NSE quotes -------------------------------

@pytest.fixture(scope="module")
def implied_vol_rows():
    if not (ROOT / "data" / "nifty_fo_bhavcopy_20261116.csv").exists():
        pytest.skip("NSE bhavcopy not present")
    return p8.compare_implied_vol()


def test_implied_vol_uses_all_417_phase6_quotes(implied_vol_rows):
    assert len(implied_vol_rows) == 417


def test_implied_vol_matches_quantlib(implied_vol_rows):
    worst = max(implied_vol_rows, key=lambda r: r["abs_error"])
    assert worst["abs_error"] < EXACT, worst
