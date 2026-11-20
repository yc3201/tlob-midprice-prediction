#!/bin/bash
# Walk-forward folds + h=100 convergence check.
# Resumable: any run whose results/<tag>.json already exists is skipped,
# so a runtime restart loses at most the run in progress.
# 1) Walk-forward (h=100, 3 epochs, both models, 3 folds): the
#    overfitting check FI-2010's 10 days demand. Fold test slices are
#    days 5/6/7 of the train file; the standard split is already done.
# 2) Convergence check (h=100, 8 epochs, both models): the 3-epoch val
#    curves were still rising, so this tests whether the TLOB-vs-DeepLOB
#    ranking flips with more training (TLOB lr 1e-4 vs DeepLOB lr 0.01).
cd ~/workspace/goals/land-ml-applied-scientist-role/files/tlob-midprice-prediction
PY=~/workspace/.tlob-venv/bin/python
{
echo "=== wf + convergence runs started $(date -u) ==="
run() {  # run <tag> <train.py args...>
  local tag="$1"; shift
  if [ -f "results/${tag}.json" ]; then
    echo "=== SKIP $tag (already done) ==="
  else
    echo "=== $tag $(date -u +%H:%M:%S) ==="
    $PY -u train.py "$@"
  fi
}
for SPLIT in wf-days1-4-\>day5 wf-days1-5-\>day6 wf-days1-6-\>day7; do
  for M in tlob deeplob; do
    run "${M}_h100_seq100_f40_${SPLIT}" --model $M --horizon 100 \
      --seq-size 100 --n-features 40 --stride 10 --epochs 3 --batch 32 \
      --split "$SPLIT"
  done
done
echo "=== convergence check: h=100, 8 epochs ==="
for M in tlob deeplob; do
  run "${M}_h100_seq100_f40_standard_8ep" --model $M --horizon 100 \
    --seq-size 100 --n-features 40 --stride 10 --epochs 8 --batch 32 \
    --split standard --out-prefix ${M}_h100_seq100_f40_standard_8ep
done
echo "=== ALL WF + CONVERGENCE RUNS DONE $(date -u) ==="
} > results/run_log3.txt 2>&1
