#!/bin/bash
# Full-recipe runs (Yangfan's analysis, 2026-11-19 ~18:47 ET).
# LAUNCH ONLY AFTER the current 8-epoch pair (run_wf_conv.sh) finishes.
# Resumable: any run whose results/<tag>.json already exists is skipped.
#
# (a) Full recipe: TLOB AND DeepLOB at 10 epochs (repo max_epochs),
#     TLOB seq 128 (repo), DeepLOB seq 100 (repo), repo LRs
#     (TLOB 1e-4, DeepLOB 1e-2), classic labels (the paper's FI-2010
#     protocol), all 4 horizons. This is the fair comparison.
# (b) Labelling EXTENSION: TLOB, 10 epochs, seq 128, --labels tlob
#     (paper's new labelling, median-theta z-scored adaptation) vs the
#     classic-labels runs from (a), h=100 and h=50. Headline FI-2010
#     comparison stays on original labels.
# (c) MLPLOB baseline (repo config: lr 3e-4; seq 100 for comparability,
#     repo default is 384 -- flagged in train.py), 10 epochs, h=10/20
#     (the short horizons where the paper says it wins).
#
# Time estimate (measured epoch times): TLOB ~7-8 min/epoch at seq 128,
# DeepLOB ~3 min/epoch, MLPLOB ~3-4 min/epoch. Total ~10-11 hours.
# Runs sequentially; each run appends to results/run_log4.txt.
cd ~/workspace/goals/land-ml-applied-scientist-role/files/tlob-midprice-prediction
PY=~/workspace/.tlob-venv/bin/python
{
echo "=== full-recipe runs started $(date -u) ==="
run() {  # run <tag> <train.py args...>
  local tag="$1"; shift
  if [ -f "results/${tag}.json" ]; then
    echo "=== SKIP $tag (already done) ==="
  else
    echo "=== $tag $(date -u +%H:%M:%S) ==="
    $PY -u train.py "$@"
  fi
}
echo "=== (a) full recipe: 10 epochs, classic labels ==="
for H in 100 50 20 10; do
  run "tlob_h${H}_seq128_f40_standard_10ep" --model tlob --horizon $H \
    --seq-size 128 --n-features 40 --stride 10 --epochs 10 --batch 32 \
    --split standard --out-prefix tlob_h${H}_seq128_f40_standard_10ep
  run "deeplob_h${H}_seq100_f40_standard_10ep" --model deeplob --horizon $H \
    --seq-size 100 --n-features 40 --stride 10 --epochs 10 --batch 32 \
    --split standard --out-prefix deeplob_h${H}_seq100_f40_standard_10ep
done
echo "=== (b) labelling extension: TLOB, new labels, 10 epochs ==="
for H in 100 50; do
  run "tlob_h${H}_seq128_f40_standard_10ep_lab-tlob" --model tlob \
    --horizon $H --labels tlob --seq-size 128 --n-features 40 \
    --stride 10 --epochs 10 --batch 32 --split standard \
    --out-prefix tlob_h${H}_seq128_f40_standard_10ep_lab-tlob
done
echo "=== (c) MLPLOB baseline: 10 epochs, short horizons ==="
for H in 10 20; do
  run "mlplob_h${H}_seq100_f40_standard_10ep" --model mlplob --horizon $H \
    --seq-size 100 --n-features 40 --stride 10 --epochs 10 --batch 32 \
    --split standard --out-prefix mlplob_h${H}_seq100_f40_standard_10ep
done
echo "=== ALL FULL-RECIPE RUNS DONE $(date -u) ==="
} > results/run_log4.txt 2>&1
