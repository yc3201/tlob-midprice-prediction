"""Tests for the TLOB labelling scaffold.

Run:  python3 test_labelling.py        (no pytest needed)
"""
import numpy as np

from labelling import (DOWN, FLAT, UP, fi2010_labels, horizon_bias_demo,
                       label_shares, spread_aware_labels, tlob_labels,
                       tlob_labels_zscored, tlob_labelling_demo)


def test_tlob_labels_match_paper_formula():
    # Hand-computed from §4 eqs. 5–7: w+ = mean of k+1 mids ending at
    # t+h, w- = mean of k+1 mids ending at t, l = (w+ - w-) / w-.
    # mids = 100..109, h = 2, k = 2, theta = 0.002.
    mids = np.arange(100, 110, dtype=float)
    lab = tlob_labels(mids, horizon=2, k=2, theta=0.002)
    # t=2: w- = mean(100,101,102) = 101, w+ = mean(102,103,104) = 103,
    # l = 2/101 > theta -> UP. t=0,1 invalid (t < k); t=8,9 invalid
    # (t+h > n-1). Every valid t is UP on this steady climb.
    assert lab.tolist() == [-1, -1, 2, 2, 2, 2, 2, 2, -1, -1], lab.tolist()
    # Same series reversed: every valid label must be DOWN.
    lab_down = tlob_labels(mids[::-1].copy(), horizon=2, k=2, theta=0.002)
    assert lab_down.tolist() == [-1, -1, 0, 0, 0, 0, 0, 0, -1, -1]


def test_tlob_labels_antisymmetric():
    n = np.arange(500)
    mids_up = 100.0 * (1.001 ** n)
    mids_down = 100.0 * (0.999 ** n)
    up = label_shares(tlob_labels(mids_up, 10, 10))["up"]
    down = label_shares(tlob_labels(mids_down, 10, 10))["down"]
    assert abs(up - down) < 1e-9


def test_tlob_labels_head_and_tail_unlabelled():
    mids = np.full(100, 100.0)
    lab = tlob_labels(mids, horizon=10, k=5)
    # First k indices need a past window, last h need a future window.
    assert (lab[:5] == -1).all()
    assert (lab[-10:] == -1).all()
    assert (lab[5:-10] == FLAT).all()


def test_tlob_up_share_flat_on_stationary_series():
    # The bias the paper's labelling removes: on a stationary series
    # the label distribution must not depend on the horizon, because
    # both smoothing windows have fixed length k+1. (Measured on seed 0:
    # tlob up-shares 0.007/0.010/0.010/0.010 across h = 10..200.)
    shares = tlob_labelling_demo()["stationary_up"]
    ups = [shares[h]["tlob"] for h in (10, 50, 100, 200)]
    assert max(ups) - min(ups) < 0.02, ups


def test_tlob_up_share_still_rises_with_drift():
    # Honest guardrail, not a failure: on a drifting series the tlob
    # up-share still rises with h (0.659 -> 0.999, seed 0), because
    # drift accumulated over a horizon gap is real signal. The paper
    # balances classes through its theta choice (§6.3), not through the
    # labelling alone. If this test ever fails, someone changed the
    # labelling to force flatness — that would be fabrication.
    shares = tlob_labelling_demo()["drifting_up"]
    assert shares[200]["tlob"] > shares[10]["tlob"], shares


def test_flat_series_is_all_flat():
    mids = np.full(500, 100.0)
    lab = fi2010_labels(mids, 10)
    assert set(lab[lab >= 0].tolist()) == {FLAT}
    lab = spread_aware_labels(mids, np.full(500, 0.05), 10)
    assert set(lab[lab >= 0].tolist()) == {FLAT}


def test_strong_uptrend_is_up():
    mids = 100.0 * (1.001 ** np.arange(500))
    lab = fi2010_labels(mids, 10)
    assert label_shares(lab)["up"] > 0.9
    lab = spread_aware_labels(mids, np.full(500, 0.01), 10)
    assert label_shares(lab)["up"] > 0.9


def test_huge_spread_kills_the_trend():
    # Same uptrend, but the spread (0.5) is far bigger than the 10-step
    # move (~1.0 * ... actually ~0.1 per 10 steps at this slope is small):
    # a move that does not clear the spread must be labelled flat.
    # This is the paper's transaction-cost point, in miniature.
    mids = 100.0 * (1.0001 ** np.arange(500))
    lab = spread_aware_labels(mids, np.full(500, 5.0), 10)
    assert set(lab[lab >= 0].tolist()) == {FLAT}


def test_labelling_is_antisymmetric():
    # Exact construction: an up-trend and its slope-negated mirror must
    # swap up/down shares exactly.
    n = np.arange(500)
    mids_up = 100.0 * (1.001 ** n)
    mids_down = 100.0 * (0.999 ** n)
    up = label_shares(fi2010_labels(mids_up, 10))["up"]
    down = label_shares(fi2010_labels(mids_down, 10))["down"]
    assert abs(up - down) < 1e-9
    up = label_shares(spread_aware_labels(mids_up, np.full(500, 0.01), 10))["up"]
    down = label_shares(spread_aware_labels(mids_down, np.full(500, 0.01), 10))["down"]
    assert abs(up - down) < 1e-9


def test_tail_is_unlabelled_not_guessed():
    mids = np.full(100, 100.0)
    lab = fi2010_labels(mids, 10)
    assert (lab[-10:] == -1).all() and (lab[:-10] == FLAT).all()


def test_tlob_labels_zscored_median_theta_balances():
    # The z-scored adaptation (absolute threshold): on a symmetric
    # series the default median-|d| threshold must give ~25/50/25
    # shares (median splits |d| in half; up/down split the rest).
    # The mean-|d| rule instead gives ~72% flat on FI-2010 data --
    # measured 2026-11-19 -- so the median is the balance-preserving
    # reading of the paper's theta intent (§6.3).
    rng = np.random.default_rng(0)
    mids = np.cumsum(rng.normal(0, 1, size=20000))  # zero-drift walk
    lab, theta = tlob_labels_zscored(mids, horizon=10, k=10)
    s = label_shares(lab)
    assert abs(s["up"] - 0.25) < 0.03, s
    assert abs(s["down"] - 0.25) < 0.03, s
    assert abs(s["flat"] - 0.5) < 0.03, s
    assert theta > 0


def test_tlob_labels_zscored_matches_relative_on_level_series():
    # On a series with a nearly constant level (w- ≈ 100 ± 0.5), the
    # absolute variant with theta_abs = theta * level must agree with
    # the relative variant. Checks the window math is shared, not
    # re-derived. (Agreement is approximate, not exact: the relative
    # threshold varies with w-, so the bar is 0.99, not 1.0.)
    t = np.arange(5000)
    mids = 100.0 + 0.5 * np.sin(2 * np.pi * t / 50.0)
    rel = tlob_labels(mids, horizon=10, k=10, theta=0.002)
    # theta_abs = 0.002 * typical w- (~100) = 0.2
    abs_lab, _ = tlob_labels_zscored(mids, horizon=10, k=10, theta_abs=0.2)
    agree = np.mean(rel[rel >= 0] == abs_lab[rel >= 0])
    assert agree > 0.99, agree


def test_horizon_bias_demo_shows_growth():
    shares = horizon_bias_demo()
    assert shares[200] > shares[10], shares
    assert shares[10] > 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
