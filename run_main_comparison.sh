#!/bin/bash
# Main comparison: TLOB vs DeepLOB, horizons 10/20/50/100.
# Fair head-to-head: seq 100, 40 features, stride 10, 3 epochs, batch 32.
# Runs sequentially; each run appends to run_log.txt.
# Launched in the background; check progress with: tail -f results/run_log.txt
cd ~/workspace/goals/land-ml-applied-scientist-role/files/tlob-midprice-prediction
PY=~/workspace/.tlob-venv/bin/python
for H in 100 50 20 10; do
  for M in tlob deeplob; do
    echo "=== $M h=$H $(date -u +%H:%M:%S) ===" | tee -a results/run_log.txt
    $PY train.py --model $M --horizon $H --seq-size 100 --n-features 40 \
      --stride 10 --epochs 3 --batch 32 --split standard 2>&1 | tee -a results/run_log.txt
  done
done
echo "=== ALL MAIN RUNS DONE $(date -u) ===" | tee -a results/run_log.txt
