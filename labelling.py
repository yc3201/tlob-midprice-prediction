"""Labelling for LOB mid-price trend prediction (TLOB project scaffold).

Labels are 0 = down, 1 = flat, 2 = up. Only indices with a full forward
window are labelled; the tail is dropped, never padded with a guess.

Two labellings live here, and they are NOT the same thing:

1. ``fi2010_labels`` — the prior standard (Tsantekidis et al. 2017, the
   labelling every FI-2010 baseline uses). Compare the mean mid-price
   over the next ``horizon`` events with the current mid-price; call it
   up/down when the move exceeds a fixed fraction ``alpha`` (default
   0.002, the value the FI-2010 literature uses). This is the baseline
   your comparison must beat, not the paper's contribution.

2. ``spread_aware_labels`` — an implementation of the *idea described
   in the TLOB abstract*: define a trend relative to the average spread,
   the primary transaction cost, instead of a fixed percentage. A move
   only counts as up if the forward mean mid-price clears the current
   mid-price by more than the average spread over the window. This is
   the paper's point about why classification accuracy does not convert
   into trading profit: most "up" moves are smaller than the spread.
   This function is my implementation of that described idea, not a
   transcription of the paper's formula.

3. ``tlob_labels`` — the paper's own labelling (Berti & Kasneci 2025,
   §4, eqs. 5–7), implemented after reading the paper. It dissociates
   the smoothing window ``k`` from the prediction horizon ``h``:
   ``w_+`` is the mean mid-price over the k+1 events ending at t+h,
   ``w_-`` the mean over the k+1 events ending at t, and
   ``l = (w_+ - w_-) / w_-``; up iff l > θ, down iff l < −θ, else flat.
   The old FI-2010 labelling ties the smoothing window to the horizon
   (k = h), so the amount of smoothing — and hence the label
   distribution — depends on the horizon you chose. With k fixed, the
   label statistic has the same smoothing at every horizon.

   What the new labelling does and does NOT do (measured, see
   ``tlob_labelling_demo``): on a stationary series (white noise around
   a fixed level) the tlob up-share is flat across horizons
   (0.007 → 0.010 for h = 10 → 200) while the old labelling's shares
   move with h — that is the bias it removes. On a DRIFTING series the
   tlob up-share still rises with h (0.659 → 0.999): drift accumulated
   over a horizon gap is real signal, not labelling bias, and no
   labelling that compares t with t+h can remove it. The paper handles
   that by choosing θ per horizon/dataset (for Tesla/Intel, θ = the mean
   percentage change, §6.3) to balance the classes. Do not claim the
   paper's labelling makes up-shares flat on a drifting series — the
   numbers say otherwise.

   Note the paper used the ORIGINAL FI-2010 labels for its FI-2010
   experiments and the new labelling only for Intel/Tesla (§4, end).
   So for the FI-2010 comparison, models are trained on the original
   labels; ``tlob_labels`` is the project's labelling contribution and
   the trading-check labelling.

Why the horizon bias matters — ``horizon_bias_demo`` below: with a
fixed ``alpha`` and a drifting series, the share of "up" labels grows
with the horizon, because the same drift accumulates over a longer
window. The label then depends on the horizon you chose, not only on
the market. A labelling that removes that dependence is what makes
results comparable across the paper's four horizons.

Conventions: ``mids`` is a 1-D array of mid-prices, ``spreads`` a 1-D
array of bid-ask spreads aligned with ``mids``. No file I/O here; the
data loader for FI-2010 / LOBSTER is TODO 2 in the README.
"""

from __future__ import annotations

import numpy as np

DOWN, FLAT, UP = 0, 1, 2


def _forward_mean(mids: np.ndarray, horizon: int) -> np.ndarray:
    """Mean mid-price over the next ``horizon`` events, per start index.

    Only starts with a FULL forward window are defined; the tail is NaN
    (callers turn it into "unlabelled"). A shrinking window near the end
    would silently change the horizon — the exact bias this module is
    about — so it is refused, not approximated.
    """
    n = len(mids)
    csum = np.concatenate([[0.0], np.cumsum(mids)])
    ends = np.arange(1, n + 1)  # forward window for start i is mids[i+1 : i+1+h]
    out = np.full(n, np.nan)
    valid = (ends + horizon) <= n
    out[valid] = (csum[ends[valid] + horizon] - csum[ends[valid]]) / horizon
    return out


def fi2010_labels(mids: np.ndarray, horizon: int, alpha: float = 0.002) -> np.ndarray:
    """Classic FI-2010 labelling: fixed percentage threshold ``alpha``."""
    mids = np.asarray(mids, dtype=float)
    fwd = _forward_mean(mids, horizon)
    labels = np.full(len(mids), -1, dtype=int)
    valid = ~np.isnan(fwd)
    move = (fwd - mids) / np.abs(mids)
    labels[valid & (move > alpha)] = UP
    labels[valid & (move < -alpha)] = DOWN
    labels[valid & (np.abs(move) <= alpha)] = FLAT
    return labels


def spread_aware_labels(mids: np.ndarray, spreads: np.ndarray,
                        horizon: int) -> np.ndarray:
    """Trend defined against the average spread (TLOB abstract's idea).

    Up iff forward mean mid-price exceeds current mid-price by more than
    the mean spread over the same forward window; down iff it falls by
    more. Otherwise flat — the move does not survive transaction costs.
    """
    mids = np.asarray(mids, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    fwd = _forward_mean(mids, horizon)
    fwd_spread = _forward_mean(spreads, horizon)
    labels = np.full(len(mids), -1, dtype=int)
    valid = ~np.isnan(fwd) & ~np.isnan(fwd_spread)
    diff = fwd - mids
    labels[valid & (diff > fwd_spread)] = UP
    labels[valid & (diff < -fwd_spread)] = DOWN
    labels[valid & (np.abs(diff) <= fwd_spread)] = FLAT
    return labels


def tlob_labels(mids: np.ndarray, horizon: int, k: int,
                theta: float = 0.002) -> np.ndarray:
    """TLOB labelling (Berti & Kasneci 2025, §4, eqs. 5–7).

    ``w_+(t,h,k)`` = mean of the ``k+1`` mid-prices ending at ``t+h``;
    ``w_-(t,h,k)`` = mean of the ``k+1`` mid-prices ending at ``t``;
    ``l(t,h,k) = (w_+ - w_-) / w_-``. Up iff ``l > theta``, down iff
    ``l < -theta``, else flat. The smoothing window ``k`` is FIXED and
    independent of the horizon ``h`` — that dissociation is the paper's
    point: in the classic labelling the smoothing window IS the horizon,
    so the label distribution depends on the horizon chosen.

    Valid indices are ``k <= t <= n-1-h`` (both windows must be full);
    everything else is -1 (unlabelled). ``theta`` is the paper's θ; for
    FI-2010 the paper kept the original labels with θ = 0.002, and for
    Tesla/Intel set θ to the mean percentage change to balance classes.
    """
    mids = np.asarray(mids, dtype=float)
    n = len(mids)
    if k < 0 or horizon < 0:
        raise ValueError("k and horizon must be non-negative")
    csum = np.concatenate([[0.0], np.cumsum(mids)])
    t = np.arange(n)
    valid = (t >= k) & (t + horizon <= n - 1)
    w_minus = np.full(n, np.nan)
    w_plus = np.full(n, np.nan)
    # w_-: mids[t-k : t+1];  w_+: mids[t+h-k : t+h+1]; each has k+1 terms.
    w_minus[valid] = (csum[t[valid] + 1] - csum[t[valid] - k]) / (k + 1)
    w_plus[valid] = (csum[t[valid] + horizon + 1]
                     - csum[t[valid] + horizon - k]) / (k + 1)
    l = (w_plus - w_minus) / w_minus
    labels = np.full(n, -1, dtype=int)
    labels[valid & (l > theta)] = UP
    labels[valid & (l < -theta)] = DOWN
    labels[valid & (np.abs(l) <= theta)] = FLAT
    return labels


def tlob_labels_zscored(mids: np.ndarray, horizon: int, k: int,
                         theta_abs: float | None = None) -> tuple[np.ndarray, float]:
    """TLOB labelling adapted to Z-SCORED mid-prices (extension, not the
    paper's literal formula).

    The paper's ``tlob_labels`` divides by ``w_-`` (eq. 7), which needs
    a real price level. The FI-2010 release files are z-scored, so
    ``w_-`` wanders around 0 and the relative move explodes. This
    variant keeps the paper's window construction (``w_+``/``w_-`` as
    means of k+1 mids, k dissociated from h) but thresholds the
    ABSOLUTE smoothed move ``d = w_+ - w_-`` (in z-units): up iff
    ``d > theta_abs``, down iff ``d < -theta_abs``, else flat.

    ``theta_abs`` defaults to the MEDIAN absolute smoothed move
    (``median|d|``) — chosen so the training classes are approximately
    balanced (up/flat/down ≈ 25/50/25), which is the paper's stated
    intent for θ (§6.3: θ = mean percentage change, chosen to balance
    classes). The literal mean|d| analogue yields 72% flat on train
    and 93% flat on test for FI-2010 z-scored data (measured
    2026-11-19), which would collapse training to flat-only
    predictions; median|d| is the balance-preserving reading of the
    paper's rule on this data. Returns (labels, theta_abs used).
    Labels use this module's convention: DOWN=0, FLAT=1, UP=2;
    -1 = unlabelled (same valid range as ``tlob_labels``).
    """
    mids = np.asarray(mids, dtype=float)
    n = len(mids)
    if k < 0 or horizon < 0:
        raise ValueError("k and horizon must be non-negative")
    csum = np.concatenate([[0.0], np.cumsum(mids)])
    t = np.arange(n)
    valid = (t >= k) & (t + horizon <= n - 1)
    w_minus = np.full(n, np.nan)
    w_plus = np.full(n, np.nan)
    w_minus[valid] = (csum[t[valid] + 1] - csum[t[valid] - k]) / (k + 1)
    w_plus[valid] = (csum[t[valid] + horizon + 1]
                     - csum[t[valid] + horizon - k]) / (k + 1)
    d = w_plus - w_minus
    if theta_abs is None:
        theta_abs = float(np.nanmedian(np.abs(d[valid])))
    labels = np.full(n, -1, dtype=int)
    labels[valid & (d > theta_abs)] = UP
    labels[valid & (d < -theta_abs)] = DOWN
    labels[valid & (np.abs(d) <= theta_abs)] = FLAT
    return labels, theta_abs


def label_shares(labels: np.ndarray) -> dict:
    """Share of each class among labelled indices."""
    lab = labels[labels >= 0]
    if len(lab) == 0:
        return {"down": 0.0, "flat": 0.0, "up": 0.0, "n": 0}
    return {
        "down": float(np.mean(lab == DOWN)),
        "flat": float(np.mean(lab == FLAT)),
        "up": float(np.mean(lab == UP)),
        "n": int(len(lab)),
    }


def horizon_bias_demo(seed: int = 0) -> dict:
    """Show the horizon bias on a synthetic drifting random walk.

    A mid-price with a small constant drift: with a fixed-alpha labelling
    the up-share grows with horizon even though the underlying process
    never changes. Returns the up-share per horizon.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.002, size=20000)  # drift + noise
    mids = 100.0 * np.exp(np.cumsum(steps))
    return {h: round(label_shares(fi2010_labels(mids, h))["up"], 3)
            for h in (10, 50, 100, 200)}


def tlob_labelling_demo(seed: int = 0) -> dict:
    """What the paper's labelling does and does not remove.

    Two synthetic series, horizons 10/50/100/200, k = 10, θ = 0.002:

    * drifting random walk (same as ``horizon_bias_demo``): the tlob
      up-share STILL rises with h — drift over a horizon gap is real
      signal, and the paper balances classes via its θ choice, not via
      the labelling alone. Anyone claiming flat up-shares here is wrong;
      the numbers are printed so the claim can be checked.
    * stationary white noise around a fixed level: the tlob up-share is
      flat across horizons (both smoothing windows have fixed length),
      while the classic labelling's shares move with h because its
      smoothing window IS the horizon. That horizon-independence of the
      label distribution is the bias the paper removes.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.002, size=20000)
    mids_drift = 100.0 * np.exp(np.cumsum(steps))
    rng = np.random.default_rng(seed)
    mids_wn = 100.0 + rng.normal(0, 0.2, size=20000)
    out = {"drifting_up": {}, "stationary_up": {}}
    for h in (10, 50, 100, 200):
        out["drifting_up"][h] = {
            "fi2010": round(label_shares(fi2010_labels(mids_drift, h))["up"], 3),
            "tlob": round(label_shares(tlob_labels(mids_drift, h, 10))["up"], 3),
        }
        out["stationary_up"][h] = {
            "fi2010": round(label_shares(fi2010_labels(mids_wn, h))["up"], 3),
            "tlob": round(label_shares(tlob_labels(mids_wn, h, 10))["up"], 3),
        }
    return out


if __name__ == "__main__":
    print("Horizon bias demo (fixed alpha=0.002, drifting random walk):")
    print("horizon -> up-share:", horizon_bias_demo())
    print("Same drift, same process — only the horizon changed.")
    print()
    print("Spread-aware labelling on the same walk (spread = 0.05):")
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0004, 0.002, size=20000)
    mids = 100.0 * np.exp(np.cumsum(steps))
    spreads = np.full_like(mids, 0.05)
    for h in (10, 50, 100):
        print(f"  horizon {h}: {label_shares(spread_aware_labels(mids, spreads, h))}")
    print()
    print("TLOB labelling (k=10, theta=0.002) vs classic, same walks:")
    for kind, table in tlob_labelling_demo().items():
        print(f"  {kind}:")
        for h, row in table.items():
            print(f"    horizon {h}: classic up={row['fi2010']}, tlob up={row['tlob']}")
    print("  Drifting walk: tlob up-share still rises — drift is signal,")
    print("  not bias. Stationary noise: tlob up-share is flat across")
    print("  horizons — that is the bias the paper's labelling removes.")
