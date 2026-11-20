# TLOB — mid-price trend prediction from LOB data

Reimplementation of TLOB (dual-attention Transformer, Berti & Kasneci 2025) for 3-way mid-price trend prediction (up / stationary / down) on FI-2010, benchmarked against DeepLOB and MLPLOB, with a spread-aware trading check on the predictions.

## Method

- **TLOB**, reimplemented from the paper: temporal attention over time plus spatial attention over the LOB levels' structure, 4 blocks, 579,739 parameters. Input is a window of LOB snapshots; output is 3 logits.
- **Baselines**: DeepLOB (CNN + LSTM, 143,907 parameters) and MLPLOB, both from the official TLOB repo.
- **Data**: FI-2010, 40 LOB features, horizons 10 / 20 / 50 / 100, classic labels (fixed α = 0.002). Standard split: train days 1–7 (80/20 train/val), test days 8–10. The paper's new labelling (§4) is also implemented and compared as an extension.
- **Training**: 10 epochs, Adam (TLOB lr 1e-4, DeepLOB lr 1e-2, MLPLOB lr 3e-4), EMA, macro F1 as the metric.

## Results — test macro F1, 10 epochs

Per-class F1 in brackets, order [up, stationary, down].

| horizon | TLOB (seq 128) | DeepLOB (seq 100) | MLPLOB (seq 100) | majority baseline |
|---|---|---|---|---|
| 10 | 0.537 ([0.39, 0.85, 0.37]) | 0.313 ([0.03, 0.75, 0.16]) | **0.785** ([0.73, 0.91, 0.72]) | 0.276 |
| 20 | 0.437 ([0.27, 0.78, 0.27]) | 0.320 ([0.07, 0.71, 0.17]) | **0.663** ([0.58, 0.83, 0.58]) | 0.255 |
| 50 | 0.555 ([0.49, 0.70, 0.47]) | **0.634** ([0.53, 0.80, 0.57]) | not run | 0.214 |
| 100 | 0.546 ([0.50, 0.63, 0.51]) | **0.714** ([0.67, 0.79, 0.68]) | not run | 0.171 |

No single model wins everywhere: MLPLOB dominates the short horizons, DeepLOB the long ones, and TLOB is roughly flat across horizons (0.44–0.55), beating DeepLOB short and losing long. This matches the paper's finding that MLPLOB wins h=10/20 on FI-2010.

## Trading check

Signal-following backtest on the test predictions: enter at t+5, exit on the opposite signal, one full spread per round trip. PnL in z-score units (FI-2010 files are z-scored). Oracle = true labels as signals.

| h | model | F1 | trades | gross | net | win | oracle net |
|---|-------|---:|---:|---:|---:|---:|---:|
| 10 | MLPLOB | 0.785 | 1,580 | +0.61z | −1.65z | 0.30 | −2.25z |
| 10 | TLOB | 0.537 | 1,255 | −0.14z | −1.94z | 0.27 | −2.52z |
| 10 | DeepLOB | 0.313 | 272 | −0.19z | −0.58z | 0.30 | −2.25z |
| 20 | MLPLOB | 0.663 | 1,821 | +0.59z | −2.02z | 0.28 | −2.84z |
| 20 | TLOB | 0.437 | 1,436 | −0.17z | −2.23z | 0.27 | −2.91z |
| 20 | DeepLOB | 0.320 | 672 | −0.02z | −0.98z | 0.26 | −2.84z |
| 50 | DeepLOB | 0.634 | 3,370 | +0.17z | −4.67z | 0.23 | −4.02z |
| 50 | TLOB | 0.555 | 3,737 | +0.09z | −5.27z | 0.21 | −3.73z |
| 50 | TLOB (new labels) | 0.491 | 1,612 | +0.54z | −1.77z | 0.30 | −1.75z |
| 100 | DeepLOB | 0.714 | 3,851 | +0.64z | −4.89z | 0.22 | −4.55z |
| 100 | TLOB | 0.546 | 4,975 | −0.27z | −7.41z | 0.17 | −4.69z |
| 100 | TLOB (new labels) | 0.435 | 1,274 | +0.90z | −0.92z | 0.37 | −0.12z |

Nothing is net-profitable, and neither is the oracle: even perfect labels lose net at every horizon, so the spread, not the classifier, is the binding constraint. MLPLOB has the only clearly positive gross PnL at short horizons (+0.61z / +0.59z) and still loses net. The new-labels TLOB has the smallest net loss of any run that actually trades (h=100: −0.92z on 1,274 trades, win 0.37).

## Labelling extension

TLOB with the paper's new labelling (§4) scores lower F1 than classic labels (h=100: 0.435 vs 0.546; h=50: 0.491 vs 0.555) because its test set is more flat-dominated (65% flat), so up/down are rarer. But its trading profile is much less loss-making (h=100 net −0.92z vs −7.41z, win 0.37 vs 0.17) — the labels produce far fewer, better-timed flips. F1 and net PnL rank the two label schemes oppositely.

## Reproduce

```bash
pip install torch scikit-learn numpy pandas
# FI-2010 data is not included. Get it from the official TLOB repo
# (LeonardoBerti00/TLOB) and place the unzipped files in data/FI_2010/

python train.py --model tlob --horizon 100 --epochs 10 --seq-size 128
python train.py --model deeplob --horizon 100 --epochs 10
python train.py --model mlplob --horizon 10 --epochs 10
python train.py --model tlob --horizon 100 --epochs 10 --labels tlob

python trading_check.py results/<predictions>.npz --horizon 100 --out results/trading_h100.json
```

Per-run metrics and test predictions land in `results/`. Tests:

```bash
python -m pytest test_labelling.py test_model.py test_data.py  # 13 + 8 + 6 tests
python trading_check.py --self-test  # 5 checks
```

## Structure

- `labelling.py` — classic FI-2010 labelling, spread-aware labelling, the paper's `tlob_labels`
- `model.py` — TLOB reimplementation (temporal + spatial attention, BilinearNorm)
- `data.py` — FI-2010 loader, sliding windows, walk-forward folds
- `train.py` — TLOB / DeepLOB / MLPLOB training with macro F1, EMA, best-val checkpointing
- `trading_check.py` — signal-following backtest on test predictions
- `results/` — per-run JSONs and test predictions

## Limitations

- FI-2010 only (5 Finnish stocks, 10 days). No second dataset.
- 40 LOB features. The paper's TLOB config uses 144 (LOB + handcrafted features).
- MLPLOB was run at h=10/20 only.

## Reference

- Berti, L. & Kasneci, G. *TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data.* arXiv:2502.15757, 2025.
- Official repo: [LeonardoBerti00/TLOB](https://github.com/LeonardoBerti00/TLOB) (MIT) — DeepLOB and MLPLOB baselines from there.
