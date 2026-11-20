# TLOB — mid-price trend prediction from LOB data

Status: **built and measured, 2026-11-20. All planned runs complete:**
10-epoch full recipe (TLOB vs DeepLOB vs MLPLOB), labelling extension,
walk-forward folds, trading checks. Yangfan picked TLOB on 2026-11-19
15:44 ET as the quant project for QR/trader roles; she also asked
(17:29 ET) for this to be kept as a backup/alternative project entry
for her resume — the resume-ready bullets are at the bottom, with
measured 10-epoch numbers.

## The paper (verified 2026-11-19 against arXiv + the repo)

- **TLOB: A Novel Transformer Model with Dual Attention for Price Trend
  Prediction with Limit Order Book Data**, Berti & Kasneci, arXiv
  2502.15757, submitted 12 Feb 2025 (v3, 7 May 2025).
- Dual attention: one attention over time, one over the LOB levels'
  spatial structure, so the model can focus on microstructure; the
  authors report it helps most at longer horizons and in volatile
  conditions.
- A **new labelling method that removes the horizon bias** (labelling
  §4, architecture §5). Implemented in `labelling.py` as `tlob_labels`
  exactly as defined (smoothed past/future mid-price means, window `k`
  dissociated from horizon `h`); the horizon-bias demo test passes.
  IMPORTANT correction (2026-11-19): the paper used the ORIGINAL
  FI-2010 labels for its FI-2010 experiments and the new labelling
  only for Intel/Tesla (§4, end). So training on classic FI-2010
  labels IS the paper's FI-2010 protocol, not a flaw in our
  comparison — the new-labelling comparison below is an EXTENSION
  beyond that protocol, not a fix. And the demo finding stands: on a
  DRIFTING series the tlob up-share still rises with h (drift is
  signal, not bias); it is flat across horizons only on stationary
  noise. Do not claim the new labelling makes up-shares flat on a
  drifting series.
- Evaluated on FI-2010, NASDAQ and Bitcoin, four horizons. Reported
  deltas, all from the arXiv abstract / repo README: **+3.7 F1 on
  average vs SoTA on FI-2010; +1.3 on Tesla, +7.7 on Intel, +1.1 on
  Bitcoin**. Predictability declined over time by **−6.68 F1** — the
  market-getting-more-efficient talking point.
- Transaction-cost experiment (abstract): defining trends against the
  **average spread** instead of a fixed move deteriorates performance —
  most classified trends do not survive the spread. That is the
  classification-vs-trading gap, and it is the interview story.
  Reproduced here on FI-2010 (see "Trading check").
- **Verified (corrected 2026-11-19):** the "F1 92.8%" figure IS in the
  paper — the arXiv HTML v1 abstract says "we evaluate it on the
  well-known FI-2010 benchmark (F1 of 92.8%)". That is the paper's
  reported h=100 number, not a measured result of this project. My
  own measured per-horizon numbers (below) are the primary evidence
  for what I reproduced — do not present 92.8% as my result.

## The official repo (verified 2026-11-19)

`LeonardoBerti00/TLOB` — public, MIT. Cloned as a REFERENCE at
`reference-TLOB/` (583 MB, includes FI-2010 zips + BTC checkpoints).
Layout: `config/ data/ models/ preprocessing/ utils/ main.py run.py
run_backtesting.py`. Model contract: input `[batch, seq_len,
num_features]` → output `[batch, 3]`. Built-in baselines: **TLOB,
MLPLOB, DeepLOB, BiNCTABL** — the DeepLOB comparison uses the repo's
own `models/deeplob.py`, imported as-is. FI-2010: hidden dim 40 with
LOB only, 144 with LOB + market features.

Standing rule, non-negotiable: **study the repo, do not copy its TLOB
implementation as her own work** — the dual-attention block in
`model.py` is reimplemented from the paper (§5.1–5.2, Appendix A);
the repo was read only for the data pipeline and the DeepLOB baseline.
Deliberately never read: `models/tlob.py`, `models/mlplob.py`,
`utils/attention.py`.

Cautions: FI-2010 is 5 Finnish stocks over 10 days — small, so any
generalisation claim needs the NASDAQ/LOBSTER check the paper itself
runs (not done here: LOBSTER data needs a subscription). The repo's
DeepLOB returns softmax probabilities, so it is trained here with
NLLLoss on log-probs (= cross-entropy on the logits); its Adam
eps=1 quirk is kept for fidelity (see `train.py` header).

## What is built here

- `labelling.py` — FI-2010 classic labelling (fixed α = 0.002),
  spread-aware labelling, the paper's `tlob_labels` (§4 eqs. 5–7),
  horizon-bias demo. Numpy only.
- `test_labelling.py` — 11 tests, all passing.
- `model.py` — my TLOB reimplementation: BilinearNorm (batch
  statistics over both axes, per BiNCTABL), temporal attention,
  spatial attention, MLPLOB block, 4 blocks, head on the final time
  step → 3 logits (0 = up, 1 = stationary, 2 = down — label mapping
  verified against the FI-2010 dataset record and empirically).
- `test_model.py` — 8 tests, all passing (shapes, attention masking
  invariance, BilinearNorm batch dependence, training step, eval
  determinism). Caught a real bug: BilinearNorm first used per-sample
  statistics; the paper's BiN pools over the batch — fixed.
- `data.py` — FI-2010 loader following the repo's `fi_2010_load`
  (train file 80/20, test = days 8–10, label rows −1..−5, −1 shift),
  lazy sliding windows, walk-forward folds. `data/FI_2010/` holds the
  unzipped files (train 144 × 254,750; test 144 × 139,587).
- `test_data.py` — 4 tests, all passing (shapes, label ranges,
  window/label alignment, fold structure, label-mapping check).
- `train.py` — TLOB vs DeepLOB training with macro F1 (the paper's
  metric), EMA (decay 0.999), LR halving on val-loss stall, best-val
  checkpointing; saves per-run JSON + test predictions `.npz`.
  `--labels tlob` trains on the paper's new labelling (extension);
  `--model mlplob` adds the repo's MLPLOB baseline. Best checkpoint
  + partial results are saved to disk after every improved epoch
  (`<tag>_best.pt`, `<tag>_partial.json`); `--eval-checkpoint` scores
  a killed run's best checkpoint without retraining.
- `trading_check.py` — signal-following backtest on the test
  predictions (enter at t+5 for latency, exit on the opposite signal,
  one full spread per round trip) + a horizon-matched hold diagnostic
  + oracle (true labels as signals) and buy-and-hold. Self-test: 5
  checks passing, including the segment-boundary check.
- `results/` — per-run JSONs, test-prediction `.npz` files,
  `trading_h*.json`, run logs (`run_log.txt`, `run_log2.txt`,
  `run_log3.txt`), `run_main_comparison.sh`, `run_remaining.sh`,
  `run_wf_conv.sh`.
- Environment: `~/workspace/.tlob-venv` (torch 2.15.0+cpu, sklearn
  1.9.1, numpy 2.5.5, pandas 3.1.1). First install attempt failed
  because the PyTorch index URL was applied to sklearn too (no such
  package there); fixed by installing torch from the PyTorch index
  and sklearn from PyPI.

## Measured results

Protocol (all runs): seq 100, 40 LOB features, stride 10 (window
subsampling — overlapping stride-1 windows are ~10x redundant),
batch 32, standard split (train days 1–7, 80/20 train/val; test days
8–10), 3 epochs unless noted. TLOB Adam lr 1e-4; DeepLOB Adam lr 0.01,
eps=1 (repo quirk). Macro F1 on the test set; per-class F1 order is
[up, stationary, down].

### All planned runs — COMPLETE (2026-11-20)

Yangfan's analysis (2026-11-19 ~18:47 ET): the 3-epoch comparison was
unfair, and the main cause is undertraining, not the model (TLOB h=100
val F1 rose every epoch and its best epoch was the last). The LR gap
is the paper's own recipe (repo config: TLOB lr 1e-4, DeepLOB lr 1e-2)
— do not "fix" the LR, fix the epochs. Data scale also matters: our
runs used ~39k windows vs the paper's ~394k sampled observations,
which penalizes the bigger transformer (579,739 params) more than
DeepLOB (143,907).

(a) Full recipe: TLOB AND DeepLOB at 10 epochs (repo max_epochs),
TLOB seq 128, DeepLOB seq 100, repo LRs, classic labels (the paper's
FI-2010 protocol), all 4 horizons. **Done** — table below.
(b) Labelling EXTENSION: TLOB, 10 epochs, seq 128, `--labels tlob`
(median-θ z-scored adaptation, train-fitted θ; measured train shares
25/50/25 at h=100 and h=50, test 19/65/16) vs the classic-labels runs
from (a), h=100 and h=50. **Done** — h=100 new 0.4350 vs classic
0.5464; h=50 new 0.4911 vs classic 0.5549.
(c) MLPLOB baseline (repo config: lr 3e-4; seq 100 for comparability,
repo default is 384), 10 epochs, h=10/20. **Done** — MLPLOB wins both
short horizons by a wide margin (h=10 0.7851, h=20 0.6625), matching
the paper's claim.
(d) Finalize: full per-horizon table, trading check, README results
section, 3 resume bullets with measured numbers only. **Done** — this
README.

Execution note: the runs were launched in ~2 h chunks (run_chunk*.sh,
resumable skip-if-exists) because the runtime kills long jobs
unpredictably (kills at ~15 min, ~20 min, ~35 min, ~36 min, ~40 min,
~80 min, ~2 h 15 m). `train.py` saves the best checkpoint + partial
results after every improved epoch, and `--eval-checkpoint` scores a
killed run's best checkpoint without retraining — every killed run was
recovered this way (checkpoint evals reproduced full-run results
exactly where both exist, e.g. DeepLOB h=20 0.31952...).

### 10-epoch full-recipe runs (complete — chunked, ~2 h per chunk)

Headline first: **TLOB h=100, 10 epochs, seq 128: test macro F1
0.5464** (per-class [0.50, 0.63, 0.51], acc 0.543), vs 0.2853 at
3 epochs. Val F1 climbed every epoch (0.18 → 0.29 → 0.29 → 0.34 →
0.47 → 0.51 → 0.52 → 0.52 → 0.53 → 0.53) and the best epoch was the
last — the model only learned the flat class properly after epoch 4.
The undertraining diagnosis is confirmed: 3 epochs was not a fair
test. Per-class F1 is now balanced (no more flat collapse).

DeepLOB h=100 10-epoch was killed by a runtime restart at epoch 6;
its best checkpoint (epoch 5, val 0.5866) scored **test F1 0.6264**
from disk, and the full rerun finished in chunk 1: **DeepLOB h=100
10ep test F1 0.7140** (per-class [0.67, 0.79, 0.68], acc 0.716) and
**DeepLOB h=50 10ep test F1 0.6336** ([0.53, 0.80, 0.57], acc 0.669).
So at 10 epochs DeepLOB is still ahead of TLOB at h=100 (0.714 vs
0.546) and h=50 — the fair-recipe comparison does NOT flip the
ranking on FI-2010 with 40 features. Caveat: the paper's TLOB uses
144 features (all_features=True); these runs use 40. A 144-feature
TLOB 10-epoch run is the remaining gap vs the paper's recipe (not yet
run — see "Not run").

TLOB h=100 10ep trading check: flip-rule 4,975 trades, gross −0.27z,
net −7.41z (win 0.17); hold-10 net −8.33z; oracle flip −4.69z; 81% of
signals below the spread. Even at F1 0.55 the classification edge
does not survive the spread — the paper's finding holds at 10 epochs
too. DeepLOB 10ep trading: h=100 — 3,851 trades, gross +0.64z, net
−4.89z (win 0.22), hold-10 net −6.95z; h=50 — 3,370 trades, gross
+0.17z, net −4.67z (win 0.23), hold-5 net −10.23z. DeepLOB's gross
PnL is slightly positive at both horizons (a real, small edge in
gross terms), but the spread still eats it net. Oracle flip net is
negative at every horizon (−4.69z / −4.02z), so even perfect labels
lose net of spread at these horizons.

DeepLOB h=20 10ep: test F1 0.3195 ([0.07, 0.71, 0.17], acc 0.551) —
identical to its 3-epoch result. Val F1 was flat at ~0.23 for all 10
epochs and the best checkpoint is epoch 0: at short horizons DeepLOB
does not improve with epochs, it just predicts flat. DeepLOB h=10
10ep (best checkpoint epoch 0, val 0.2933, scored from disk; epoch 9
not run — val flat at 0.2582 for epochs 1–8): test F1 0.3125
([0.03, 0.75, 0.16], acc 0.596), also identical to its 3-epoch
result. DeepLOB 10-epoch table is now complete: h=10 0.3125, h=20
0.3195, h=50 0.6336, h=100 0.7140.

TLOB h=50 10ep: test F1 0.5549 ([0.49, 0.70, 0.47], acc 0.585) vs
0.3114 at 3 epochs — again a large, balanced improvement. TLOB 10ep
so far: h=50 0.5549, h=100 0.5464 — flat across horizons, unlike
DeepLOB (0.31–0.32 short, 0.63–0.71 long).

DeepLOB trading at short horizons: h=20 — 672 trades, gross −0.02z,
net −0.98z (win 0.26); h=10 — 272 trades, gross −0.19z, net −0.58z
(win 0.30); 97% of signals below the spread at both. The flat-only
predictor trades rarely and loses little; the oracle still loses net
(−2.84z / −2.25z). TLOB h=50 10ep trading: 3,737 trades, gross
+0.09z, net −5.27z (win 0.21), hold-5 net −10.58z, oracle flip
−3.73z, 89% of signals below spread — same story as h=100.

TLOB h=20: first 10-epoch attempt killed at epoch 3 with val F1 flat
at 0.2310 (epochs 0–3) — the short-horizon flat pattern again, and
the epoch-3 checkpoint scored test F1 0.2592 (flat-only). But the
full rerun JUMPED late, exactly as h=100 did: **TLOB h=20 10ep test
F1 0.4369** ([0.27, 0.78, 0.27], acc 0.625) vs DeepLOB h=20 0.3195
and 3-epoch TLOB 0.2548. TLOB 10ep so far: h=20 0.4369, h=50 0.5549,
h=100 0.5464 — TLOB wins the short horizon and loses the long ones.

TLOB h=20 10ep trading: 1,436 trades, gross −0.17z, net −2.23z
(win 0.27), hold-2 net −4.36z, oracle flip −2.91z, 95% of signals
below the spread. TLOB h=10: first 10-epoch attempt killed at
epoch 4 (val 0.2579 flat epochs 0–3, 0.2629 at epoch 4; epoch-4
checkpoint scored test F1 0.2935, flat-only). Full rerun jumped late
again: **TLOB h=10 10ep test F1 0.5367** ([0.39, 0.85, 0.37], acc
0.737) vs DeepLOB h=10 0.3125 and 3-epoch TLOB 0.2756.

### Full-recipe 10-epoch comparison — COMPLETE (classic labels)

| horizon | TLOB (seq 128) | DeepLOB (seq 100) | MLPLOB (seq 100) | majority baseline |
|---|---|---|---|---|
| 10 | 0.5367 ([0.39, 0.85, 0.37]) | 0.3125 ([0.03, 0.75, 0.16]) | **0.7851** ([0.73, 0.91, 0.72]) | 0.276 |
| 20 | 0.4369 ([0.27, 0.78, 0.27]) | 0.3195 ([0.07, 0.71, 0.17]) | **0.6625** ([0.58, 0.83, 0.58]) | 0.255 |
| 50 | 0.5549 ([0.49, 0.70, 0.47]) | **0.6336** ([0.53, 0.80, 0.57]) | not run | 0.214 |
| 100 | 0.5464 ([0.50, 0.63, 0.51]) | **0.7140** ([0.67, 0.79, 0.68]) | not run | 0.171 |

Reading: **no single model wins everywhere.** MLPLOB dominates the
short horizons (0.79 / 0.66, vs TLOB 0.54 / 0.44 and DeepLOB 0.31 /
0.32) — the paper's claim that MLPLOB wins h=10/20 is confirmed,
and it is the strongest result in the project: per-class F1 is
balanced at h=10 ([0.73, 0.91, 0.72], acc 0.86). DeepLOB wins the
long horizons (0.63 / 0.71). TLOB is roughly flat across horizons
(0.44–0.55) and sits in the middle everywhere: it beats DeepLOB
short (+0.224 / +0.117) and loses long (−0.079 / −0.168). On the
4-horizon mean TLOB (0.5187) is ahead of DeepLOB (0.4949) by +0.024
— but that average hides the real story, which is the horizon
interaction. Both models beat the majority baseline at every horizon
at 10 epochs (DeepLOB h=10/20 only narrowly, via flat F1).

Every TLOB run showed the same shape: val F1 flat (near the
majority baseline) for the first 3–4 epochs, then a sharp jump as
up/down classes started being predicted — the 3-epoch comparison cut
every TLOB run off before the jump. Caveat: TLOB's best val epoch
was 9 (the last) at ALL four horizons, and DeepLOB's was also 9 at
h=50/100 — both models were still improving at 10 epochs, so the
ranking could shift with more training. DeepLOB h=10's best
checkpoint is epoch 0 (val never improved past 0.2933; epoch 9 not
run — see the h=10 note below).

TLOB h=10 10ep trading: 1,255 trades, gross −0.14z, net −1.94z
(win 0.27), hold-1 net −3.74z, oracle flip −2.52z, 97% of signals
below the spread.

### Labelling extension (b) — TLOB with the paper's new labelling

TLOB h=100 with `--labels tlob`: **test F1 0.4350** ([0.29, 0.76,
0.25], acc 0.622) vs classic-labels 0.5464. The epoch-0 checkpoint
scored 0.2624 (flat-only). But the label distributions differ, so
compare against each scheme's own majority baseline: new labels
0.4350 vs baseline 0.262 (+0.173); classic 0.5464 vs baseline 0.171
(+0.375). The new labelling did NOT improve F1 on FI-2010 — its
test set is more flat-dominated (65% flat vs 34%), so up/down are
rarer and harder (per-class up/down 0.29/0.25 vs classic 0.50/0.51).
The paper's new labelling was designed for Intel/Tesla to remove
horizon bias, not to raise FI-2010 F1; on FI-2010 the headline stays
on classic labels. Labelling extension (b) is COMPLETE: h=100 new
0.4350 vs classic 0.5464; h=50 new 0.4911 ([0.37, 0.78, 0.32], acc
0.651) vs classic 0.5549. Same pattern at both horizons — the new
labels lower F1 (more flat-dominated test set) but the trading
profile improves (see below).

Trading under the new labels is a different story: TLOB lab-tlob
h=100 — 1,274 trades (vs 4,975 classic), gross +0.90z, net −0.92z
(win 0.37, the best win rate of any run), hold-10 net −2.96z,
oracle flip −0.12z (nearly breakeven, vs −4.69z classic). TLOB
lab-tlob h=50 — 1,612 trades, gross +0.54z, net −1.77z (win 0.30),
hold-5 net −4.92z, oracle flip −1.75z. The new labels produce far
fewer, better-timed flips — the classification F1 is lower but the
trading profile is much less loss-making. That contrast (F1 vs net
PnL ranking the two label schemes oppositely) is itself the point of
the trading check.

All runs complete. MLPLOB h=10/20 (chunk 8b): **MLPLOB h=10 test F1
0.7851** ([0.73, 0.91, 0.72], acc 0.862) and **MLPLOB h=20 test F1
0.6625** ([0.58, 0.83, 0.58], acc 0.749) — the paper's claim that
MLPLOB wins the short horizons is confirmed; it is the strongest
result in the project. MLPLOB trading: h=10 — 1,580 trades, gross
+0.61z, net −1.65z (win 0.30); h=20 — 1,821 trades, gross +0.59z,
net −2.02z (win 0.28). The runtime killed detached/tracked jobs
unpredictably throughout (kills at ~15 min, ~20 min, ~35 min,
~36 min, ~40 min, ~80 min, ~2 h 15 m); every killed run was
recovered via per-epoch checkpoints + `--eval-checkpoint`.

### Per-horizon macro F1 (test), TLOB vs DeepLOB — UNDERTRAINED 3-epoch comparison (NOT the final result)

This table is the 3-epoch run only. It is NOT a fair test: both
models' validation F1 was still rising at epoch 3 and TLOB's best
epoch was the last one, so the models were cut off mid-climb (see
"Reading this honestly"). The fair comparison is the 10-epoch
full-recipe run (repo max_epochs, TLOB seq 128), now complete — see
the 10-epoch table above. Do not quote this table as the final result
anywhere.

| horizon | TLOB | DeepLOB | majority-class baseline* |
|--------:|-----:|--------:|-------------------------:|
| 10 | 0.276 | 0.313 | 0.276 |
| 20 | 0.255 | 0.320 | 0.255 |
| 50 | 0.311 | 0.358 | 0.214 |
| 100 | 0.285 | 0.353 | 0.171 |

\* Constant-flat predictor on the test set (flat share 0.71 / 0.62 /
0.47 / 0.34). Per-class F1: TLOB h=10 is [0.00, 0.83, 0.00] and h=20
is [0.00, 0.76, 0.00] — it collapsed to predicting flat only and did
not beat the baseline at all at short horizons. At h=50/100 TLOB
predicts up/down ([0.42, 0.15, 0.36] / [0.41, 0.00, 0.44]) but
almost never flat. DeepLOB beats the baseline at every horizon.

Reading this honestly: **DeepLOB wins at every horizon under this
budget, and the ranking is confounded by convergence.** Both models'
validation F1 was still rising at epoch 3 (TLOB h=100: 0.260 → 0.283
→ 0.311; DeepLOB h=100: 0.208 → 0.307 → 0.325), and TLOB's lr is
100x smaller than DeepLOB's, so 3 epochs is far less training for
TLOB. The paper trains 10 epochs. The 8-epoch convergence check
below tests whether the ranking flips with more training — do not
quote the 3-epoch table as "TLOB is worse than DeepLOB" without that
caveat.

### Trading check (segmented backtest, z-score units)

Rules: signal at a window's end, enter at +5 samples, exit on the
opposite signal (flat = hold), one full spread per round trip
(spread proxy = mean |ask1_z − bid1_z| = 0.00143z). PnL is in
z-score units (FI-2010 files are z-scored; no currency PnL is
possible from them). B&H = summed within-segment drift, +0.410z at
every horizon (same test set).

#### 10-epoch runs (the final numbers)

| h | model | F1 | flip trades | flip gross | flip net | win | hold net | oracle flip net | signals < spread |
|---|-------|---:|----:|---:|---:|---:|---:|---:|---:|
| 10 | MLPLOB | 0.785 | 1,580 | +0.61z | −1.65z | 0.30 | −4.58z | −2.25z | 95% |
| 10 | TLOB | 0.537 | 1,255 | −0.14z | −1.94z | 0.27 | −3.74z | −2.52z | 97% |
| 10 | DeepLOB | 0.313 | 272 | −0.19z | −0.58z | 0.30 | −3.87z | −2.25z | 97% |
| 20 | MLPLOB | 0.663 | 1,821 | +0.59z | −2.02z | 0.28 | −5.25z | −2.84z | 93% |
| 20 | TLOB | 0.437 | 1,436 | −0.17z | −2.23z | 0.27 | −4.36z | −2.91z | 95% |
| 20 | DeepLOB | 0.320 | 672 | −0.02z | −0.98z | 0.26 | −3.56z | −2.84z | 97% |
| 50 | DeepLOB | 0.634 | 3,370 | +0.17z | −4.67z | 0.23 | −10.23z | −4.02z | 87% |
| 50 | TLOB | 0.555 | 3,737 | +0.09z | −5.27z | 0.21 | −10.58z | −3.73z | 89% |
| 50 | TLOB (new labels) | 0.491 | 1,612 | +0.54z | −1.77z | 0.30 | −4.92z | −1.75z | 88% |
| 100 | DeepLOB | 0.714 | 3,851 | +0.64z | −4.89z | 0.22 | −6.95z | −4.55z | 79% |
| 100 | TLOB | 0.546 | 4,975 | −0.27z | −7.41z | 0.17 | −8.33z | −4.69z | 81% |
| 100 | TLOB (new labels) | 0.435 | 1,274 | +0.90z | −0.92z | 0.37 | −2.96z | −0.12z | 79% |

Findings at 10 epochs: **no model is net-profitable, and neither is
the oracle** — even perfect labels lose net at every horizon
(−0.12z to −4.69z flip-rule), so the spread, not the classifier, is
the binding constraint. But the ranking by net PnL is not the
ranking by F1: MLPLOB has the best F1 AND the only clearly positive
gross PnL at short horizons (+0.61z / +0.59z), yet still loses net
(−1.65z / −2.02z); DeepLOB h=10/20 "loses least" (−0.58z / −0.98z)
only because it predicts flat and barely trades (272 / 672 trades).
The new-labels TLOB has the smallest net loss of any run that
actually trades (h=100: −0.92z on 1,274 trades, win 0.37) because
its labels produce far fewer, better-timed flips. 79–97% of signals
are too small to clear the spread. This is the paper's
classification-vs-trading gap, reproduced at full training: a
0.79-F1 classifier still does not make money net of spread on
FI-2010.

#### 3-epoch runs (undertrained — for the record only)

| h | model | F1 | flip trades | flip gross | flip net | win | hold net | oracle flip net | signals < spread |
|---|-------|---:|----:|---:|---:|---:|---:|---:|---:|
| 100 | TLOB | 0.285 | 4,715 | +0.57z | −6.20z | 0.18 | −9.67z | −4.56z | 83% |
| 100 | DeepLOB | 0.353 | 3,537 | +0.42z | −4.65z | 0.19 | −9.61z | −4.56z | 83% |
| 50 | TLOB | 0.311 | 6,032 | −0.37z | −9.02z | 0.14 | −19.09z | −4.02z | 91% |
| 50 | DeepLOB | 0.358 | 3,078 | +0.06z | −4.36z | 0.19 | −15.38z | −4.02z | 91% |
| 20 | TLOB | 0.255 | 0 | 0 | 0 | — | 0 | −2.84z | — |
| 20 | DeepLOB | 0.320 | 672 | −0.02z | −0.98z | 0.26 | −3.56z | −2.84z | 97% |
| 10 | TLOB | 0.276 | 0 | 0 | 0 | — | 0 | −2.25z | — |
| 10 | DeepLOB | 0.313 | 272 | −0.19z | −0.58z | 0.30 | −3.87z | −2.25z | 97% |

"Hold net" is the horizon-matched diagnostic (hold = horizon in
samples: 10/5/2/1). "Signals < spread" = share of non-flat signals
whose realized forward move is smaller than the spread proxy.

Findings at 3 epochs (for the record): gross PnL ≈ 0 for both
models at every horizon (no tradeable edge at this training level);
net is negative everywhere because thousands of trades × the spread
overwhelms it. **Even the oracle (true labels as signals) loses net
at every horizon** (−2.25z to −4.56z flip-rule) — the paper's
finding, reproduced: most classified trends do not survive the spread
(83–97% of signals are too small to clear it). TLOB at h=10/20 makes
0 trades (it only predicts flat), so its net PnL is 0 by not trading
— not by skill. The 10-epoch findings above supersede this: gross
PnL turns slightly positive for the best models, but net stays
negative everywhere.

Method note — the boundary-jump catch: the first (unsegmented)
backtest credited DeepLOB h=20 with **+10.89z net**. Verification
showed the top 5 trades supplied 89% of that, and all 5 spanned
single-step mid-price jumps of 0.42–2.32z — the 14 stock/day
boundaries in the concatenated test file (median step is 0.0).
Those are file artifacts, not market moves. The backtest now splits
the series at jumps (|Δmid| > 0.1z) and never holds a position
across one; the +10.89z collapses to −0.98z. The segmented numbers
above are the ones to quote.

### Walk-forward folds (h=100, 3 epochs)

Expanding folds on the train file (days 1–7); the standard split
(test = days 8–10) is the fourth row. Per-class F1 is
[up, stationary, down].

| fold (train → test) | TLOB | DeepLOB |
|---|---|---|
| days 1–4 → day 5 | 0.304 ([0.37, 0.00, 0.54]) | 0.309 ([0.53, 0.00, 0.40]) |
| days 1–5 → day 6 | 0.308 ([0.46, 0.00, 0.46]) | 0.293 ([0.50, 0.00, 0.38]) |
| days 1–6 → day 7 | 0.299 ([0.48, 0.01, 0.41]) | **0.440** ([0.41, 0.47, 0.44]) |
| standard → days 8–10 | 0.285 ([0.41, 0.00, 0.44]) | 0.353 ([0.45, 0.11, 0.50]) |

Reading: TLOB is stable across folds (0.29–0.31) and still ignores
the flat class everywhere; DeepLOB is more variable (0.29–0.44),
and its day-7 fold is the only run in the whole project where either
model predicts flat with a non-trivial F1 (0.47). No dramatic
overfitting signal — the standard-split numbers sit inside the
walk-forward ranges — but with 3 epochs and one 10-day sample this is
a consistency check, not a generalisation proof.

### Convergence check (h=100, 8 epochs)

Not completed as a standalone pair: the TLOB 8-epoch run started
00:55:59 UTC and was killed by a runtime restart mid-run (no
checkpoint existed then); DeepLOB 8-epoch never started. It is
SUBSUMED by the 10-epoch full-recipe runs below, which answer the
same question (does the ranking flip with more training?) more
definitively. Reliability note: the runtime killed detached training
twice (once ~43 min in, once ~2 h in), so `train.py` now saves the
best checkpoint + partial results to disk after every improved epoch,
and `--eval-checkpoint` can score a killed run's best checkpoint
without retraining. The full-recipe script is resumable
(skip-if-exists) at the run level.

### Not run (stated limitations)

- TLOB's paper config (seq 128, 144 features incl. handcrafted) —
  the comparison above is seq 128 / 40 features for TLOB (seq 100 /
  40 for DeepLOB and MLPLOB), so it is fair across models, but it is
  not the paper's TLOB configuration (144 features). A 144-feature
  TLOB run is the one remaining gap vs the paper's recipe.
- Second dataset (NASDAQ/LOBSTER) — LOBSTER data needs a
  subscription; FI-2010 only.
- More than 10 epochs: TLOB's best val epoch was 9 (the last) at all
  four horizons, and DeepLOB's was also 9 at h=50/100 — both were
  still improving at 10 epochs, so the ranking could shift with
  longer training.
- DeepLOB h=10: best checkpoint is epoch 0 (val never improved past
  0.2933); epoch 9 was not run (killed at epoch 8) — the reported
  0.3125 is the best-checkpoint result through epoch 8, not a
  completed 10-epoch run. Same caveat does not apply to h=20 (full
  10 epochs completed, best epoch 0, 0.3195).
- MLPLOB at h=50/100 not run (plan only required h=10/20, where the
  paper says it wins).
- Walk-forward folds are 3-epoch only (0.29–0.31 TLOB / 0.29–0.44
  DeepLOB at h=100); a 10-epoch walk-forward was not run.

## Build plan — status

1. ~~Read the paper (§4 labelling, §5 architecture).~~ `tlob_labels`
   implemented exactly; demo test passes (up-share flat across
   horizons on stationary data, still rises with drift — documented
   in `labelling.py`).
2. ~~Data.~~ FI-2010 via the repo clone; 40- vs 144-feature shapes
   confirmed.
3. ~~Model.~~ Reimplemented from the paper; 8/8 tests pass.
4. ~~Comparison.~~ TLOB vs DeepLOB vs MLPLOB per-horizon F1 at 10
   epochs — done (table above). Second dataset not run (see
   limitations).
5. ~~Trading check.~~ Done at 3 and 10 epochs (tables above),
   including the boundary-jump correction. Walk-forward folds done
   at 3 epochs (h=100).
6. ~~Résumé bullets (below) — final, numbers from measured 10-epoch
   runs only.~~

## Résumé bullets — backup/alternative project entry (final)

Written to drop straight into the one-page resume's project section
(3 bullets, each a single focused point: what it is, why it matters,
method, measured impact). Numbers are from the measured 10-epoch
runs above only.

- "Reimplemented TLOB (dual-attention Transformer over LOB time and
  price levels, Berti & Kasneci 2025) from the paper for mid-price
  trend prediction on FI-2010 (394k snapshots, 40 LOB features);
  trained 10 epochs against DeepLOB (CNN+LSTM) and MLPLOB baselines
  — macro F1 0.44–0.55 across horizons 10–100, beating DeepLOB at
  short horizons (0.54 vs 0.31 at h=10) while DeepLOB wins long
  (0.71 vs 0.55 at h=100), and MLPLOB best at h=10 (0.79)."
- "Diagnosed the undertraining trap behind the inverted first
  comparison: at 3 epochs TLOB trailed DeepLOB at h=100 (0.29 vs
  0.35) because its validation F1 stays flat for 3–4 epochs before
  jumping; at 10 epochs TLOB nearly doubles to 0.55 — and
  walk-forward folds (train days 1–4/5/6 → test day 5/6/7) hold at
  0.30 vs the 0.29 standard split at the same 3-epoch budget, so the
  split is not inflating results."
- "Ran the spread-aware trading check that closes the project: even
  the best classifier (MLPLOB, F1 0.79) is net-negative after the
  spread (−1.65z on 1,580 trades), and even perfect oracle labels
  lose net at every horizon — while the paper's horizon-bias-free
  labelling cuts the h=100 net loss to −0.92z with a 0.37 win rate,
  showing classification gains do not survive transaction costs."

## Relation to the other quant-project options

- The AS simulator in `../as-market-making-simulator/` is the classical
  baseline for the **deep-RL market-making** option (arXiv 2305.15821),
  not the headline project — Yangfan rejected 2008 AS as the project
  basis on 2026-11-19.
- TradingAgents (Dec 2024) stays the third option: strongest LLM fit,
  weakest quant signal; only as a second project.
- GitHub: Yangfan asked for the finished project to be put on GitHub
  (account confirmed yc3201; planned repo yc3201/tlob-midprice-
  prediction, name not yet confirmed by her). **Do not publish until
  she confirms the repo name.**
