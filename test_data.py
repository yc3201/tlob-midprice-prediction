"""Tests for data.py (numpy only -- no torch needed).

Run:  python3 test_data.py
Covers: file shapes and label ranges, window/label alignment (the
label belongs to the window's FINAL time step), walk-forward folds
do not overlap and expand, and the label-mapping ordering check.
"""
import numpy as np

from data import (LOBWindows, class_shares, compute_tlob_labels, load_raw,
                  standard_split, verify_label_mapping, walk_forward_folds)

DATA_DIR = "data/FI_2010"


def test_load_raw_shapes_and_labels():
    tr_f, tr_y, te_f, te_y = load_raw(DATA_DIR, 100)
    assert tr_f.shape == (144, 254750), tr_f.shape
    assert te_f.shape == (144, 139587), te_f.shape
    for y in (tr_y, te_y):
        assert set(np.unique(y)) <= {0, 1, 2}
    # h=10 is flat-heavy, h=100 near-balanced (known FI-2010 pattern)
    _, y10, _, _ = load_raw(DATA_DIR, 10)
    assert class_shares(y10)[1] > 0.55
    assert class_shares(tr_y)[1] < 0.25


def test_window_label_is_final_step():
    tr_f, tr_y, _, _ = load_raw(DATA_DIR, 100)
    w = LOBWindows(tr_f, tr_y, seq_size=100, stride=7)
    for i in (0, 1, 100):
        x, y = w[i]
        assert x.shape == (100, 144) and x.dtype == np.float32
        # label at the window's final time step
        assert y == int(tr_y[i * 7 + 99])
    assert len(w) == (len(tr_y) - 100) // 7 + 1


def test_walk_forward_folds_expand_no_overlap():
    folds = walk_forward_folds(254750)
    prev_train_end = 0
    for name, tr, te in folds:
        assert tr.start == 0 and tr.stop == te.start  # train ends at test
        assert te.stop - te.start > 0
        assert tr.stop >= prev_train_end  # expanding
        prev_train_end = tr.stop
    tr, va = standard_split(254750)
    assert tr.stop == va.start and va.stop == 254750


def test_label_mapping_down_is_negative():
    tr_f, tr_y, _, _ = load_raw(DATA_DIR, 100)
    m = verify_label_mapping(tr_f, tr_y, 100)
    # up and flat have positive mean forward moves; down is negative.
    assert m[0] > 0 and m[1] > 0 and m[2] < 0, m


def test_compute_tlob_labels_applies_given_theta():
    # An explicit theta_abs is used as-is (train-fitted theta applied
    # to test), and a larger theta can only move labels toward flat.
    rng = np.random.default_rng(0)
    feats = np.zeros((3, 5000))
    feats[0] = np.cumsum(rng.normal(0, 1, size=5000))
    feats[2] = feats[0]  # mid_z = feats[0]
    lab_lo, th_lo, _ = compute_tlob_labels(feats, 100, theta_abs=0.001)
    lab_hi, th_hi, _ = compute_tlob_labels(feats, 100, theta_abs=1.0)
    assert th_lo == 0.001 and th_hi == 1.0
    assert np.mean(lab_hi[lab_hi >= 0] == 1) > np.mean(
        lab_lo[lab_lo >= 0] == 1)


def test_tlob_labels_mode_shares_sane():
    # Extension labels (median theta, train-fitted): classes must be
    # present on both splits, flat the largest but not collapsing,
    # and only a handful of unlabelled indices (first k / last h).
    _, tr_y, _, te_y = load_raw(DATA_DIR, 100, labels="tlob")
    for y in (tr_y, te_y):
        assert int(np.sum(y == -1)) < 100
        s = class_shares(y[y >= 0])
        assert 0.05 < s[0] < 0.4 and 0.05 < s[2] < 0.4, s
        assert 0.4 < s[1] < 0.85, s


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
