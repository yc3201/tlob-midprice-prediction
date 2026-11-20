#!/bin/bash
# Remaining comparison runs (protocol: seq 100, 40 feats, stride 10,
# batch 32, standard split) + an 8-epoch convergence check at h=100.
# Run 1-6: the six runs the first script did not finish (3 epochs, same
# protocol as the completed h=100 pair, for a consistent table).
# Run 7-8: h=100 at 8 epochs for BOTH models -- the 3-epoch val curves
# were still rising, so this tests whether the ranking flips with more
# training (TLOB lr 1e-4 converges slower than DeepLOB lr 0.01).
# Output goes ONLY to results/run_log2.txt (no inner tee: the earlier
# double-tee garbled run_log.txt). python -u so the log updates live.
cd ~/workspace/goals/land-ml-applied-scientist-role/files/tlob-midprice-prediction
PY=~/workspace/.tlob-venv/bin/python
{
echo "=== remaining runs started $(date -u) ==="
for H in 50 20 10; do
  for M in tlob deeplob; do
    echo "=== $M h=$H $(date -u +%H:%M:%S) ==="
    $PY -u train.py --model $M --horizon $H --seq-size 100 --n-features 40 \
      --stride 10 --epochs 3 --batch 32 --split standard
  done
done
echo "=== convergence check: h=100, 8 epochs ==="
for M in tlob deeplob; do
  echo "=== $M h=100 8ep $(date -u +%H:%M:%S) ==="
  $PY -u train.py --model $M --horizon 100 --seq-size 100 --n-features 40 \
    --stride 10 --epochs 8 --batch 32 --split standard \
    --out-prefix ${M}_h100_seq100_f40_standard_8ep
done
echo "=== ALL REMAINING RUNS DONE $(date -u) ==="
} > results/run_log2.txt 2>&1
