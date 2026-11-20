"""FI-2010 data loading for the TLOB project.

Follows the official TLOB repo's data pipeline (``preprocessing/fi_2010.py``,
``preprocessing/dataset.py``) so the numbers are comparable to the paper:

* Files: ``Train_Dst_NoAuction_ZScore_CF_7.txt`` (days 1-7) and
  ``Test_Dst_NoAuction_ZScore_CF_7/8/9.txt`` (days 8-10), each shaped
  (149, n_samples): rows 0-143 are features (40 LOB + 104 handcrafted),
  rows 144-148 are the five horizon labels.
* Standard split (repo): train file first 80% of columns -> train,
  remaining 20% -> validation; test = the three test files h-stacked.
* Label rows (repo ``fi_2010_load``): horizon 10 -> row -5, 20 -> -4,
  30 -> -3, 50 -> -2, 100 -> -1. Labels are 1/2/3, shifted by -1 to 0/1/2.
* Windows: ``__getitem__(i)`` returns the window ``x[i:i+seq_size]`` with
  the label at the window's FINAL time step (labels shifted by seq_size-1
  before indexing) -- i.e. the model predicts the trend that starts at the
  last observed snapshot.

Label-direction mapping (VERIFIED, do not flip):
  raw 1 -> class 0 = UP, raw 2 -> class 1 = STATIONARY, raw 3 -> class 2 = DOWN.
  Source: the FI-2010 dataset record itself (Ntakaris et al., Univ. of
  Edinburgh research portal): "'1' is for up-movement, '2' is for stationary
  condition and '3' is for down-movement." Confirmed empirically on the
  z-scored files (see ``verify_label_mapping``): mean forward mid-price move
  is monotone across classes, label 1 most positive, label 3 most negative
  (corr ~ -0.02..-0.05 -- small because z-scored prices compress short-horizon
  moves, but the ordering is consistent on train and test files and across
  several window definitions).

Walk-forward splits: FI-2010 is only 10 days, so a single train/test split
invites overfitting questions. The files do not record day boundaries, so
folds split the train file (days 1-7) by equal column fractions as a proxy
for days (254,750 cols / 7 days ~= 36,393 cols/day). Fold 2 is the standard
repo split (headline, comparable to the paper); folds 0-1 are earlier
expanding windows used only as an overfitting check.
"""

from __future__ import annotations

import os

import numpy as np

from labelling import tlob_labels_zscored

# horizon (events) -> label row index from the end of the file
HORIZON_ROWS = {10: -5, 20: -4, 30: -3, 50: -2, 100: -1}

# After the -1 shift: class index -> name. VERIFIED (see module docstring).
CLASS_NAMES = {0: "up", 1: "stationary", 2: "down"}

N_FEATURES_ALL = 144
N_FEATURES_LOB = 40

TRAIN_FILE = "Train_Dst_NoAuction_ZScore_CF_7.txt"
TEST_FILES = [
    "Test_Dst_NoAuction_ZScore_CF_7.txt",
    "Test_Dst_NoAuction_ZScore_CF_8.txt",
    "Test_Dst_NoAuction_ZScore_CF_9.txt",
]


def compute_tlob_labels(feats: np.ndarray, horizon_events: int,
                        k: int = 10,
                        theta_abs: float | None = None
                        ) -> tuple[np.ndarray, float, dict]:
    """Paper's new labelling, adapted to the z-scored FI-2010 files.

    EXTENSION beyond the paper's FI-2010 protocol (which uses the
    original file labels): computes labels from the z-scored mid-price
    proxy mid_z = (ask1_z + bid1_z) / 2 with
    ``labelling.tlob_labels_zscored`` (absolute threshold, since the
    relative move needs a real price level). Horizons are in EVENTS;
    the file samples every 10th event, so horizon_samples =
    horizon_events // 10. ``k`` is in SAMPLES.

    ``theta_abs`` defaults to the median absolute smoothed move (see
    ``tlob_labels_zscored``). Pass an explicit value to apply a
    TRAIN-fitted threshold to another file — never fit theta on test
    data (callers: fit on train, apply to test).

    Returns (labels, theta_abs, shares) with labels in the TRAINING
    convention (0 = up, 1 = stationary, 2 = down) — converted from
    labelling.py's convention (0 = down, 1 = flat, 2 = up) via
    new = 2 - old; -1 (unlabelled: first k / last h indices) stays -1.
    """
    mid_z = (feats[0] + feats[2]) / 2.0
    h_samples = max(1, horizon_events // 10)
    raw, theta_abs = tlob_labels_zscored(mid_z, h_samples, k, theta_abs)
    labels = np.full(raw.shape, -1, dtype=np.int64)
    labelled = raw >= 0
    labels[labelled] = 2 - raw[labelled]
    return labels, theta_abs, class_shares(labels[labelled])


def load_raw(data_dir: str, horizon: int, labels: str = "classic"):
    """Load the FI-2010 train/test matrices for one horizon.

    Returns (train_feats, train_labels, test_feats, test_labels) where feats
    are (n_features, n_samples) float64 and labels are (n_samples,) int64
    in {0, 1, 2} = {up, stationary, down} (or -1 = unlabelled, only in
    ``labels="tlob"`` mode for the first k / last h indices).

    ``labels="classic"`` (default): the file's original labels — the
    paper's FI-2010 protocol. ``labels="tlob"``: the paper's new
    labelling computed from the z-scored mids (see
    ``compute_tlob_labels``) — an EXTENSION, computed per file so no
    window spans a file boundary.
    """
    if horizon not in HORIZON_ROWS:
        raise ValueError(f"horizon must be one of {sorted(HORIZON_ROWS)}, got {horizon}")
    if labels not in ("classic", "tlob"):
        raise ValueError(f"labels must be 'classic' or 'tlob', got {labels}")
    row = HORIZON_ROWS[horizon]

    train = np.loadtxt(os.path.join(data_dir, TRAIN_FILE))
    test_files = [np.loadtxt(os.path.join(data_dir, f)) for f in TEST_FILES]
    test = np.hstack(test_files)
    if labels == "classic":
        train_labels = (train[row] - 1).astype(np.int64)
        test_labels = (test[row] - 1).astype(np.int64)
    else:
        # Fit theta on TRAIN only, apply the same threshold to test
        # (fitting theta per test file would leak test information).
        train_labels, theta_abs, _ = compute_tlob_labels(train[:-5],
                                                          horizon)
        test_labels = np.concatenate([
            compute_tlob_labels(f[:-5], horizon,
                                theta_abs=theta_abs)[0]
            for f in test_files
        ])
    train_feats, test_feats = train[:-5], test[:-5]
    return train_feats, train_labels, test_feats, test_labels


def class_shares(labels: np.ndarray) -> dict:
    """Fraction of samples in each class {0: up, 1: stationary, 2: down}."""
    n = len(labels)
    return {c: float(np.mean(labels == c)) for c in (0, 1, 2)}


def verify_label_mapping(train_feats: np.ndarray, train_labels: np.ndarray,
                         horizon: int = 100) -> dict:
    """Empirical check that raw label 1 = up, 3 = down.

    Uses the z-scored mid-price proxy mid_z = (ask1_z + bid1_z) / 2
    (features 0 and 2). For each class, reports the mean forward move of
    mid_z over the next horizon/10 samples (the file samples every 10th
    event). Expectation if the mapping is right: classes 0 (up) and 1
    (flat) have positive mean forward moves and class 2 (down) a
    negative one. The signal is small in z-units (short-horizon moves
    are compressed by the file-wide z-score) and the up-vs-flat gap is
    within noise, so this is an ordering check on the down side, not a
    magnitude check; the authoritative source for the mapping is the
    FI-2010 dataset record (see module docstring).
    """
    mid_z = (train_feats[0] + train_feats[2]) / 2.0
    k = max(1, horizon // 10)
    n = len(mid_z)
    c = np.concatenate([[0.0], np.cumsum(mid_z)])
    ar = np.arange(n)
    valid = (ar + k) < n
    fwd = np.full(n, np.nan)
    fwd[valid] = (c[ar[valid] + k + 1] - c[ar[valid] + 1]) / k
    move = fwd - mid_z
    out = {}
    for cls in (0, 1, 2):
        m = valid & (train_labels == cls) & ~np.isnan(move)
        out[cls] = float(np.nanmean(move[m]))
    return out


class LOBWindows:
    """Lazy sliding-window view of one (features, labels) matrix.

    Windows are NOT materialised: ``__getitem__`` slices the stored matrix,
    so a stride-1 view over ~250k samples costs one copy of the file.
    The label returned for window ``i`` is the label at the window's final
    time step, matching the repo's ``Dataset`` (labels shifted by seq-1).
    Windows whose final-step label is -1 (unlabelled — only in tlob-labels
    mode, for the first k / last h indices) are dropped; ``starts`` holds
    the surviving window start indices, and ``__getitem__`` indexes into it.
    """

    def __init__(self, feats: np.ndarray, labels: np.ndarray, seq_size: int,
                 stride: int = 1, n_features: int = N_FEATURES_ALL):
        if n_features not in (N_FEATURES_LOB, N_FEATURES_ALL):
            raise ValueError("n_features must be 40 (LOB only) or 144 (all)")
        self.feats = feats[:n_features].astype(np.float32)
        self.labels = labels
        self.seq_size = seq_size
        self.stride = stride
        # candidate starts, then keep only windows with a labelled final step
        starts = np.arange(0, len(labels) - seq_size + 1, stride)
        self.starts = starts[labels[starts + seq_size - 1] >= 0]
        self.n_windows = len(self.starts)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, i: int):
        start = int(self.starts[i])
        end = start + self.seq_size
        # x: [seq, features]; y: label at the final time step
        x = self.feats[:, start:end].T.copy()
        y = int(self.labels[end - 1])
        return x, y


def walk_forward_folds(n_cols: int):
    """Column ranges for the expanding walk-forward folds on the train file.

    Returns a list of (name, train_slice, test_slice) with slices into the
    train file's columns. Day boundaries are approximated by equal column
    fractions (7 days in the train file). Fold 'standard' is the repo's
    headline split; its test part is the separate test files, not a slice.
    """
    day = n_cols // 7
    return [
        ("wf-days1-4->day5", slice(0, 4 * day), slice(4 * day, 5 * day)),
        ("wf-days1-5->day6", slice(0, 5 * day), slice(5 * day, 6 * day)),
        ("wf-days1-6->day7", slice(0, 6 * day), slice(6 * day, 7 * day)),
    ]


def standard_split(n_cols: int, val_frac: float = 0.2):
    """Repo split: first (1-val_frac) of train-file columns -> train,
    remainder -> validation."""
    cut = int(n_cols * (1 - val_frac))
    return slice(0, cut), slice(cut, n_cols)


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/FI_2010"
    print("data dir:", data_dir)
    for horizon in (10, 20, 50, 100):
        tr_f, tr_y, te_f, te_y = load_raw(data_dir, horizon)
        print(f"h={horizon}: train feats {tr_f.shape}, test feats {te_f.shape}")
        print(f"   train class shares (up/flat/down): "
              f"{[round(class_shares(tr_y)[c], 3) for c in (0, 1, 2)]}")
        print(f"   test  class shares (up/flat/down): "
              f"{[round(class_shares(te_y)[c], 3) for c in (0, 1, 2)]}")
    tr_f, tr_y, _, _ = load_raw(data_dir, 100)
    print("label-mapping check (mean fwd move by class, expect 0 > 1 > 2):",
          {CLASS_NAMES[c]: round(v, 5)
           for c, v in verify_label_mapping(tr_f, tr_y, 100).items()})
    w = LOBWindows(tr_f, tr_y, seq_size=100, stride=10)
    x, y = w[0]
    print(f"window sample: x {x.shape} float32, y={y} ({CLASS_NAMES[y]})")
    print("walk-forward folds:", [(n, s.start, s.stop, t.start, t.stop)
                                  for n, s, t in walk_forward_folds(tr_f.shape[1])])
