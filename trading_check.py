"""Trading check: does the classifier's edge survive the spread?

Reads the test predictions saved by train.py
(``results/<tag>_test_preds.npz``: pred, y, ends, mid_z, spread_z) and
runs a simple signal-following backtest on the FI-2010 test set.

Rules (kept deliberately simple so the numbers are auditable):
* Signal at a window's end index e: predicted class 0 (up) -> long +1,
  class 2 (down) -> short -1, class 1 (flat) -> flat 0. Class mapping
  verified in data.py: 0 = up, 1 = stationary, 2 = down.
* Latency: a signal at e is entered at e + ``latency`` samples
  (default 5) -- you cannot trade on a signal at the instant it forms.
* Position is held until the OPPOSITE signal appears (flat = hold the
  current position, not an exit); exit at that signal's entry time and
  reverse into the new direction.
* SEGMENTATION (added after a verification catch): the test mid series
  is 5 stocks x 3 days concatenated, and the z-scored mid jumps by
  0.42-2.32z at the 14 stock/day boundaries (median step is 0.0).
  Positions are NEVER held across a boundary: the backtest runs
  separately inside each segment and results are summed. Without this,
  the top trades were pure boundary jumps -- e.g. DeepLOB h=20's
  apparent +10.9z net was 89% boundary jumps, and collapses once they
  are excluded. The oracle and buy-and-hold are segmented the same way.
* PnL per round trip = direction * (mid_z[exit] - mid_z[entry]).
* Cost: one full spread per round trip, where the spread proxy is the
  test-set mean of spread_z = |ask1_z - bid1_z|.

What is reported, per run: macro F1 (recomputed from pred/y), gross
PnL, net-of-spread PnL, number of round trips, win rate, mean PnL per
trade -- plus the same for an oracle (true labels as signals, the upper
bound on what the labelling can pay) and buy-and-hold. Plus the
spread-clearing fraction: the share of non-flat signals whose realized
forward move over the horizon is smaller than the average spread proxy
(the paper's Table 6 point in FI-2010 terms: most classified trends do
not survive the spread).

Units caveat (read before quoting any number): the FI-2010 release
files are Z-SCORED. mid_z and spread_z are in z-score units, not
currency, and the spread proxy is the difference of two separately
z-scored series -- a proxy, not the true spread. PnL is therefore in
z-score units per unit of position; the meaningful comparison is
gross vs net (does the edge survive costs?) and model vs oracle, not
the absolute PnL. A currency backtest would need the DecPre/raw
FI-2010 files, which the repo does not ship.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

# 0 = up, 1 = stationary, 2 = down (verified in data.py)
SIGNAL = {0: 1, 1: 0, 2: -1}

# A single-step |change| in the z-scored mid above this is a file
# concatenation boundary (stock/day change), not a market move: the
# median step is 0.0, the 99.9th percentile is 0.004, and the 14
# detected jumps are 0.42-2.32 (5 stocks x 3 test days = 15 segments).
JUMP_THRESH = 0.1


def find_segments(mid_z: np.ndarray, thresh: float = JUMP_THRESH):
    """Split the mid series at concatenation jumps.

    Returns [(start, end)] index ranges (end exclusive) with no jump
    inside any segment. A position must never be held across a segment
    boundary: the jump is a change of stock/day in the concatenated
    file, and counting it as PnL would credit/blame the model for a
    discontinuity it did not predict.
    """
    jumps = np.where(np.abs(np.diff(mid_z)) > thresh)[0]
    bounds = [0] + [int(j) + 1 for j in jumps] + [len(mid_z)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] > 1]


def load_run(path: str):
    d = np.load(path)
    return {k: d[k] for k in ("pred", "y", "ends", "mid_z", "spread_z")}


def backtest(signals: np.ndarray, times: np.ndarray, mid_z: np.ndarray,
             latency: int, spread_cost: float):
    """Signal-following backtest.

    signals: +1/0/-1 per signal time; times: sample indices of the
    signals (strictly increasing); mid_z: full mid-price proxy series.
    Returns a dict of gross/net PnL, trade count, win rate.
    """
    position = 0
    entry_price = None
    pnls_gross, pnls_net = [], []
    for sig, t in zip(signals, times):
        entry_t = t + latency
        if entry_t >= len(mid_z):
            break
        if position == 0:
            if sig == 0:
                continue  # no signal, stay flat
            position = int(sig)
            entry_price = mid_z[entry_t]
        elif sig == -position:
            # opposite signal: close and reverse at this entry time
            exit_price = mid_z[entry_t]
            gross = position * (exit_price - entry_price)
            pnls_gross.append(gross)
            pnls_net.append(gross - spread_cost)
            position = int(sig)
            entry_price = exit_price
        # same signal or flat: hold the position
    # close any open position at the last available price
    if position != 0 and entry_price is not None:
        gross = position * (mid_z[-1] - entry_price)
        pnls_gross.append(gross)
        pnls_net.append(gross - spread_cost)
    pnls_gross = np.array(pnls_gross)
    pnls_net = np.array(pnls_net)
    return {
        "round_trips": int(len(pnls_gross)),
        "wins": int(np.sum(pnls_net > 0)) if len(pnls_net) else 0,
        "gross_pnl_z": float(pnls_gross.sum()) if len(pnls_gross) else 0.0,
        "net_pnl_z": float(pnls_net.sum()) if len(pnls_net) else 0.0,
        "mean_gross_per_trade_z": float(pnls_gross.mean())
        if len(pnls_gross) else 0.0,
        "mean_net_per_trade_z": float(pnls_net.mean())
        if len(pnls_net) else 0.0,
        "win_rate_net": float(np.mean(pnls_net > 0))
        if len(pnls_net) else 0.0,
    }


def _aggregate(results):
    """Sum per-segment backtest dicts into one (win rate recomputed)."""
    n = sum(r["round_trips"] for r in results)
    wins = sum(r["wins"] for r in results)
    gross = sum(r["gross_pnl_z"] for r in results)
    net = sum(r["net_pnl_z"] for r in results)
    return {
        "round_trips": n,
        "wins": wins,
        "gross_pnl_z": gross,
        "net_pnl_z": net,
        "mean_gross_per_trade_z": gross / n if n else 0.0,
        "mean_net_per_trade_z": net / n if n else 0.0,
        "win_rate_net": wins / n if n else 0.0,
    }


def spread_clearing(pred: np.ndarray, ends: np.ndarray, mid_z: np.ndarray,
                    horizon: int, spread_cost: float, segments=None):
    """Share of non-flat signals whose realized forward move over the
    horizon is smaller than the spread proxy (i.e. would not pay).
    Windows spanning a segment boundary are skipped (the jump is not
    a market move)."""
    k = max(1, horizon // 10)  # file samples every 10th event
    seg_of = np.zeros(len(mid_z), dtype=int)
    if segments:
        for i, (s0, s1) in enumerate(segments):
            seg_of[s0:s1] = i
    realised, is_signal = [], []
    for p, e in zip(pred, ends):
        if p == 1 or e + k >= len(mid_z):
            continue
        if segments and seg_of[e] != seg_of[e + k]:
            continue
        move = abs(mid_z[e + k] - mid_z[e])
        realised.append(move)
        is_signal.append(True)
    realised = np.array(realised)
    if len(realised) == 0:
        return {"n_signals": 0, "frac_below_spread": float("nan"),
                "mean_move_z": float("nan")}
    return {
        "n_signals": int(len(realised)),
        "frac_below_spread": float(np.mean(realised < spread_cost)),
        "mean_move_z": float(realised.mean()),
    }


def backtest_hold(signals: np.ndarray, times: np.ndarray, mid_z: np.ndarray,
                  latency: int, hold: int, spread_cost: float):
    """Diagnostic variant: each signal opens a position entered at
    t + latency and closed exactly ``hold`` samples after entry (one
    trade per signal, signals arriving while open are skipped). This
    matches the trade's holding period to the label's horizon, unlike
    the flip-based rule whose holding period is however long the label
    stays unchanged. Comparing the two separates "no edge at all" from
    "edge exists but the flip rule wastes it"."""
    pnls_gross, pnls_net = [], []
    busy_until = -1
    for sig, t in zip(signals, times):
        if sig == 0 or t < busy_until:
            continue
        entry_t = t + latency
        exit_t = min(entry_t + hold, len(mid_z) - 1)
        if exit_t <= entry_t:
            break
        gross = int(sig) * (mid_z[exit_t] - mid_z[entry_t])
        pnls_gross.append(gross)
        pnls_net.append(gross - spread_cost)
        busy_until = exit_t
    pnls_gross = np.array(pnls_gross)
    pnls_net = np.array(pnls_net)
    return {
        "round_trips": int(len(pnls_gross)),
        "wins": int(np.sum(pnls_net > 0)) if len(pnls_net) else 0,
        "gross_pnl_z": float(pnls_gross.sum()) if len(pnls_gross) else 0.0,
        "net_pnl_z": float(pnls_net.sum()) if len(pnls_net) else 0.0,
        "mean_gross_per_trade_z": float(pnls_gross.mean())
        if len(pnls_gross) else 0.0,
        "mean_net_per_trade_z": float(pnls_net.mean())
        if len(pnls_net) else 0.0,
        "win_rate_net": float(np.mean(pnls_net > 0))
        if len(pnls_net) else 0.0,
    }


def check(path: str, horizon: int, latency: int = 5, hold: int | None = None):
    d = load_run(path)
    pred, y, ends, mid_z, spread_z = (d["pred"], d["y"], d["ends"],
                                       d["mid_z"], d["spread_z"])
    spread_cost = float(np.mean(spread_z))
    signals = np.array([SIGNAL[int(p)] for p in pred])
    oracle_signals = np.array([SIGNAL[int(t)] for t in y])
    segments = find_segments(mid_z)

    def per_segment(fn, sigs):
        """Run fn within each segment and aggregate (positions never
        cross a segment boundary)."""
        outs = []
        for s0, s1 in segments:
            m = (ends >= s0) & (ends < s1)
            if not np.any(m):
                continue
            outs.append(fn(sigs[m], ends[m] - s0, mid_z[s0:s1]))
        return _aggregate(outs) if outs else _aggregate([])

    model = per_segment(
        lambda s, t, m: backtest(s, t, m, latency, spread_cost), signals)
    oracle = per_segment(
        lambda s, t, m: backtest(s, t, m, latency, spread_cost),
        oracle_signals)
    # horizon-matched diagnostic (hold = horizon in samples)
    hold_n = hold if hold is not None else max(1, horizon // 10)
    model_hold = per_segment(
        lambda s, t, m: backtest_hold(s, t, m, latency, hold_n,
                                      spread_cost), signals)
    oracle_hold = per_segment(
        lambda s, t, m: backtest_hold(s, t, m, latency, hold_n,
                                      spread_cost), oracle_signals)
    # buy-and-hold: within-segment drift summed (a single B&H across
    # the concatenated file would itself capture the boundary jumps)
    bh_gross = float(sum(mid_z[s1 - 1] - mid_z[s0]
                         for s0, s1 in segments))
    clearing = spread_clearing(pred, ends, mid_z, horizon, spread_cost,
                               segments)
    return {
        "run": os.path.basename(path),
        "horizon": horizon,
        "latency_samples": latency,
        "spread_proxy_mean_z": spread_cost,
        "macro_f1": float(f1_score(y, pred, average="macro",
                                   zero_division=0)),
        "n_test_windows": int(len(pred)),
        "n_segments": len(segments),
        "model": model,
        "oracle_labels_as_signals": oracle,
        "model_hold_matched": model_hold,
        "oracle_hold_matched": oracle_hold,
        "hold_samples": hold_n,
        "buy_and_hold_gross_z": bh_gross,
        "spread_clearing": clearing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+",
                    help="results/<tag>_test_preds.npz files")
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--latency", type=int, default=5)
    ap.add_argument("--hold", type=int, default=None,
                    help="holding period in samples for the matched-hold "
                         "diagnostic (default: horizon // 10)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    for path in args.npz:
        r = check(path, args.horizon, args.latency, args.hold)
        results.append(r)
        m, o = r["model"], r["oracle_labels_as_signals"]
        mh, oh = r["model_hold_matched"], r["oracle_hold_matched"]
        print(f"{r['run']}: macro F1 {r['macro_f1']:.4f} | "
              f"spread proxy {r['spread_proxy_mean_z']:.5f} z | "
              f"flip-rule model: {m['round_trips']} trades, gross {m['gross_pnl_z']:+.3f}z, "
              f"net {m['net_pnl_z']:+.3f}z, win {m['win_rate_net']:.2f} | "
              f"flip-rule oracle net {o['net_pnl_z']:+.3f}z ({o['round_trips']} trades) | "
              f"hold-{r['hold_samples']} model net {mh['net_pnl_z']:+.3f}z "
              f"({mh['round_trips']} trades), oracle net {oh['net_pnl_z']:+.3f}z "
              f"({oh['round_trips']} trades) | "
              f"B&H {r['buy_and_hold_gross_z']:+.3f}z | "
              f"signals below spread: "
              f"{r['spread_clearing']['frac_below_spread']:.2f} "
              f"(n={r['spread_clearing']['n_signals']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {args.out}")


def _self_test():
    """Mechanics check on synthetic series (no real data)."""
    # 1. Steady rise, constant +1 signal: one long trade, gross = rise.
    mid = np.linspace(0.0, 1.0, 101)
    r = backtest(np.array([1, 1, 1]), np.array([0, 10, 20]), mid,
                 latency=5, spread_cost=0.0)
    assert r["round_trips"] == 1 and abs(r["gross_pnl_z"] - 0.95) < 1e-9, r
    # 2. Flat holds the position (no exit on flat).
    r = backtest(np.array([1, 0, 0, 1]), np.array([0, 10, 20, 30]), mid,
                 latency=5, spread_cost=0.0)
    assert r["round_trips"] == 1, r
    # 3. Opposite signal reverses: long up then short down.
    mid2 = np.concatenate([np.linspace(0.0, 1.0, 51),
                           np.linspace(1.0, 0.0, 50)])
    r = backtest(np.array([1, -1]), np.array([0, 50]), mid2,
                 latency=0, spread_cost=0.0)
    # long 0->1.0 (+1.0), then short 1.0->0.0 (+1.0)
    assert r["round_trips"] == 2 and abs(r["gross_pnl_z"] - 2.0) < 1e-9, r
    # 4. Spread cost is charged once per round trip.
    r = backtest(np.array([1, -1]), np.array([0, 50]), mid2,
                 latency=0, spread_cost=0.1)
    assert abs(r["net_pnl_z"] - (2.0 - 0.2)) < 1e-9, r
    # 5. A jump splits segments: a position is not carried across it.
    mid3 = np.concatenate([np.linspace(0.0, 1.0, 51),
                           np.linspace(3.0, 4.0, 50)])  # +2 jump at 51
    segs = find_segments(mid3)
    assert len(segs) == 2, segs
    assert segs[0] == (0, 51) and segs[1] == (51, 101), segs
    print("trading_check self-test: 5 checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
        raise SystemExit(0)
    main()
