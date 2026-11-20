"""Train TLOB (mine, model.py) vs DeepLOB (repo baseline) on FI-2010.

Usage (from the project folder, with the project venv):
    ~/workspace/.tlob-venv/bin/python train.py --model tlob --horizon 100
    ~/workspace/.tlob-venv/bin/python train.py --model deeplob --horizon 100

What is matched to the paper / official repo (reference-TLOB/):
* Data pipeline: data.py (repo's fi_2010_load split/labels; see data.py).
* Metric: macro F1 (sklearn), the paper's F1 -- repo engine.py reads it
  from classification_report's "macro avg" row.
* Loss: cross-entropy. DeepLOB's forward returns softmax probabilities,
  so its loss is NLLLoss on log(probs), which is exactly cross-entropy
  on the pre-softmax logits.
* Optimizers: TLOB Adam lr 1e-4 (paper App. A); DeepLOB Adam lr 0.01
  with eps=1 (the repo engine's configure_optimizers quirk -- kept for
  fidelity, flagged here so nobody "fixes" it silently).
* LR halved when val loss stalls (repo engine; here ReduceLROnPlateau,
  factor 0.5, patience 1).
* EMA of parameters (decay 0.999) used for validation/test, as in the
  repo engine.

What differs from the repo (documented, not hidden):
* No hyperparameter search; fixed configs below.
* Windows are subsampled with --stride (default 10) on CPU: overlapping
  stride-1 windows are ~10x redundant, and stride-10 keeps a full run
  to minutes. The stride is reported in every results file.
* The comparison default is seq 100 / 40 features for BOTH models
  (DeepLOB's config; its conv stack assumes 40 features). TLOB's
  paper config is seq 128 / 144 features (--seq-size 128 --n-features
  144) and is run as a separate row, not mixed into the head-to-head.
* --labels classic (default) trains on the file's original labels --
  the paper's FI-2010 protocol. --labels tlob trains on the paper's
  NEW labelling computed from the z-scored mids (data.py:
  compute_tlob_labels) -- an EXTENSION beyond that protocol, reported
  separately, never mixed into the headline table.
* --model mlplob adds the repo's MLPLOB baseline (num_layers 3,
  hidden_dim 40, lr 3e-4; seq_size follows the comparison window
  instead of the repo's 384 -- flagged in build_model).

Class labels (verified, see data.py): 0 = up, 1 = stationary, 2 = down.
A predicted 0 means "long", 2 means "short" in the trading check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reference-TLOB"))

from data import (LOBWindows, load_raw, standard_split, walk_forward_folds)
from model import TLOB

RESULTS_DIR = "results"


class WindowDataset(Dataset):
    """Thin torch adapter over data.LOBWindows (keeps data.py torch-free)."""

    def __init__(self, windows: LOBWindows):
        self.w = windows

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        x, y = self.w[i]
        return torch.from_numpy(x), y


class EMA:
    """Parameter EMA (repo engine: decay 0.999), applied at eval time."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(),
                                                     alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}


def build_model(name: str, seq_size: int, n_features: int) -> nn.Module:
    if name == "tlob":
        return TLOB(in_features=n_features, seq_len=seq_size)
    if name == "deeplob":
        if n_features != 40:
            raise ValueError("DeepLOB's conv stack assumes 40 features")
        from models.deeplob import DeepLOB  # repo baseline, imported as-is
        return DeepLOB()
    if name == "mlplob":
        # Repo baseline, imported as-is. Repo config: num_layers 3,
        # hidden_dim 40, lr 3e-4, seq_size 384; here seq_size follows the
        # comparison window (100/128) for comparability -- flagged, not
        # hidden. Returns logits (no softmax), so plain cross-entropy.
        from models.mlplob import MLPLOB  # noqa: PLC0415
        return MLPLOB(hidden_dim=40, num_layers=3, seq_size=seq_size,
                      num_features=n_features, dataset_type="FI_2010")
    raise ValueError(f"unknown model {name}")


def loss_for(model_name: str, out: torch.Tensor, y: torch.Tensor):
    if model_name == "deeplob":
        # Repo DeepLOB returns softmax probs; NLL on log-probs == CE.
        return nn.functional.nll_loss(
            torch.log(out.clamp_min(1e-12)), y)
    return nn.functional.cross_entropy(out, y)


def evaluate(model, loader, device, model_name, ema=None):
    model.eval()
    if ema is not None:
        ema.apply_to(model)
    preds, trues, losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            losses.append(loss_for(model_name, out, y).item())
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    if ema is not None:
        ema.restore(model)
    preds = np.array(preds)
    trues = np.array(trues)
    return {
        "macro_f1": float(f1_score(trues, preds, average="macro",
                                   zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(
            trues, preds, average=None, zero_division=0)],
        "accuracy": float(accuracy_score(trues, preds)),
        "loss": float(np.mean(losses)),
        "n": int(len(trues)),
    }


def predict_test(model, loader, device, model_name, ema=None):
    """Test predictions + window-end indices, for the trading check."""
    model.eval()
    if ema is not None:
        ema.apply_to(model)
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
    if ema is not None:
        ema.restore(model)
    return np.array(preds), np.array(trues)


def get_splits(args):
    """Return (train_windows, val_windows, test_windows, test_feats)
    for the requested split."""
    tr_f, tr_y, te_f, te_y = load_raw("data/FI_2010", args.horizon,
                                     labels=args.labels)
    tr_f = tr_f[:args.n_features]
    te_f = te_f[:args.n_features]
    if args.split == "standard":
        tr_slice, va_slice = standard_split(tr_f.shape[1])
        train_w = LOBWindows(tr_f[:, tr_slice], tr_y[tr_slice],
                             args.seq_size, args.stride, args.n_features)
        val_w = LOBWindows(tr_f[:, va_slice], tr_y[va_slice],
                           args.seq_size, args.stride, args.n_features)
        test_w = LOBWindows(te_f, te_y, args.seq_size, args.stride,
                            args.n_features)
    else:
        folds = {n: (s, t) for n, s, t in walk_forward_folds(tr_f.shape[1])}
        tr_slice, te_slice = folds[args.split]
        # last 10% of the train slice is the (small) validation set
        cut = int((te_slice.start - tr_slice.start) * 0.9) + tr_slice.start
        train_w = LOBWindows(tr_f[:, tr_slice.start:cut],
                             tr_y[tr_slice.start:cut],
                             args.seq_size, args.stride, args.n_features)
        val_w = LOBWindows(tr_f[:, cut:te_slice.start],
                           tr_y[cut:te_slice.start],
                           args.seq_size, args.stride, args.n_features)
        test_w = LOBWindows(tr_f[:, te_slice], tr_y[te_slice],
                            args.seq_size, args.stride, args.n_features)
        te_f, te_y = tr_f[:, te_slice], tr_y[te_slice]
    return train_w, val_w, test_w, te_f, te_y


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tlob", "deeplob", "mlplob"],
                    required=True)
    ap.add_argument("--horizon", type=int, choices=[10, 20, 50, 100],
                    required=True)
    ap.add_argument("--labels", choices=["classic", "tlob"],
                    default="classic",
                    help="classic = file's original labels (the paper's "
                         "FI-2010 protocol); tlob = paper's new labelling "
                         "computed from z-scored mids (EXTENSION, see "
                         "data.compute_tlob_labels)")
    ap.add_argument("--seq-size", type=int, default=100)
    ap.add_argument("--n-features", type=int, choices=[40, 144], default=40)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=10,
                    help="window stride (subsampling); 1 = every window")
    ap.add_argument("--split", default="standard",
                    choices=["standard", "wf-days1-4->day5",
                             "wf-days1-5->day6", "wf-days1-6->day7"])
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-4 (tlob) / 0.01 (deeplob)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--eval-checkpoint", default=None,
                    help="path to a <tag>_best.pt checkpoint: skip training, "
                         "evaluate that checkpoint on test and save the "
                         "predictions + JSON (recovery after a kill)")
    ARGS = ap.parse_args()

    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    lab = "" if ARGS.labels == "classic" else "_lab-tlob"
    tag = ARGS.out_prefix or (
        f"{ARGS.model}_h{ARGS.horizon}_seq{ARGS.seq_size}"
        f"_f{ARGS.n_features}_{ARGS.split}{lab}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lr = ARGS.lr if ARGS.lr is not None else (
        {"tlob": 1e-4, "deeplob": 0.01, "mlplob": 0.0003}[ARGS.model])
    train_w, val_w, test_w, te_f, te_y = get_splits(ARGS)
    print(f"model={ARGS.model} h={ARGS.horizon} labels={ARGS.labels} "
          f"seq={ARGS.seq_size} "
          f"feats={ARGS.n_features} stride={ARGS.stride} split={ARGS.split} "
          f"epochs={ARGS.epochs} lr={lr}")
    print(f"windows: train={len(train_w)} val={len(val_w)} test={len(test_w)}")
    train_loader = DataLoader(WindowDataset(train_w), batch_size=ARGS.batch,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(WindowDataset(val_w), batch_size=ARGS.batch,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(WindowDataset(test_w), batch_size=ARGS.batch,
                             shuffle=False, num_workers=0)

    model = build_model(ARGS.model, ARGS.seq_size, ARGS.n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params}")
    eps = 1.0 if ARGS.model == "deeplob" else 1e-8  # repo quirk, see header
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=1)
    ema = None if ARGS.no_ema else EMA(model, decay=0.999)

    if ARGS.eval_checkpoint:
        # Recovery mode: score a saved best checkpoint on test.
        ckpt = torch.load(ARGS.eval_checkpoint, map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt["state"])
        if ema is not None and ckpt.get("ema_shadow") is not None:
            ema.shadow = ckpt["ema_shadow"]
        test = evaluate(model, test_loader, device, ARGS.model, ema)
        print(f"CHECKPOINT TEST macro F1 {test['macro_f1']:.4f} "
              f"per-class {[round(v, 4) for v in test['per_class_f1']]} "
              f"acc {test['accuracy']:.4f}")
        _save_test_outputs(model, test_loader, device, test_w, te_f,
                           tag, test, ema, extra={
                               "evaluated_from_checkpoint":
                                   ARGS.eval_checkpoint,
                               "best_val_macro_f1": ckpt.get("best_val_f1"),
                               "best_epoch": ckpt.get("best_epoch"),
                               "epochs": ARGS.epochs})
        return

    best_val_f1, best_state, best_shadow, best_epoch = -1.0, None, None, -1
    val_history = []
    t0 = time.time()
    for epoch in range(ARGS.epochs):
        model.train()
        ep_losses, nb = [], 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_for(ARGS.model, model(x), y)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            ep_losses.append(loss.item())
            nb += 1
        val = evaluate(model, val_loader, device, ARGS.model, ema)
        sched.step(val["loss"])
        val_history.append({"epoch": epoch,
                            "train_loss": float(np.mean(ep_losses)),
                            "val_macro_f1": val["macro_f1"],
                            "val_loss": val["loss"]})
        print(f"epoch {epoch}: train_loss={np.mean(ep_losses):.4f} "
              f"val_macro_f1={val['macro_f1']:.4f} val_loss={val['loss']:.4f} "
              f"({time.time() - t0:.0f}s elapsed)")
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            # The val score was computed with EMA weights, so checkpoint
            # the EMA shadow (not the raw weights) for the test eval.
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            best_shadow = ({k: v.detach().clone()
                            for k, v in ema.shadow.items()}
                           if ema is not None else None)
            # Save to disk every time the best improves: a runtime kill
            # mid-run then loses at most the current epoch, and the
            # checkpoint can be scored with --eval-checkpoint.
            torch.save({"state": best_state, "ema_shadow": best_shadow,
                        "best_val_f1": best_val_f1,
                        "best_epoch": best_epoch},
                       os.path.join(RESULTS_DIR, f"{tag}_best.pt"))
            with open(os.path.join(RESULTS_DIR, f"{tag}_partial.json"),
                      "w") as f:
                json.dump({"tag": tag, "model": ARGS.model,
                           "horizon": ARGS.horizon,
                           "labels": ARGS.labels,
                           "seq_size": ARGS.seq_size,
                           "n_features": ARGS.n_features,
                           "split": ARGS.split,
                           "epochs_planned": ARGS.epochs,
                           "best_val_macro_f1": best_val_f1,
                           "best_epoch": best_epoch,
                           "val_history": val_history}, f, indent=2)
    print(f"best val macro F1 {best_val_f1:.4f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if ema is not None and best_shadow is not None:
            ema.shadow = best_shadow
    test = evaluate(model, test_loader, device, ARGS.model, ema)
    print(f"TEST macro F1 {test['macro_f1']:.4f} "
          f"per-class {[round(v, 4) for v in test['per_class_f1']]} "
          f"acc {test['accuracy']:.4f}")

    _save_test_outputs(model, test_loader, device, test_w, te_f, tag, test,
                       ema, extra={
                           "lr": lr, "adam_eps": eps,
                           "ema": ema is not None, "seed": ARGS.seed,
                           "n_params": n_params,
                           "train_windows": len(train_w),
                           "val_windows": len(val_w),
                           "best_val_macro_f1": best_val_f1,
                           "best_epoch": best_epoch,
                           "val_history": val_history,
                           "seconds": round(time.time() - t0, 1)})


def _save_test_outputs(model, test_loader, device, test_w, te_f, tag, test,
                       ema=None, extra=None):
    """Save test predictions (.npz for trading_check.py) + result JSON.

    ``ema`` must be the EMA object whose shadow holds the evaluated
    weights (or None): predict_test applies it, so the saved
    predictions match the reported test metrics.
    """
    preds, trues = predict_test(model, test_loader, device, ARGS.model, ema)
    # window-end indices in the test series (from the windows' actual starts)
    ends = np.array(test_w.starts) + ARGS.seq_size - 1
    mid_z = (te_f[0] + te_f[2]) / 2.0
    spread_z = np.abs(te_f[0] - te_f[2])
    np.savez(os.path.join(RESULTS_DIR, f"{tag}_test_preds.npz"),
             pred=preds, y=trues, ends=ends, mid_z=mid_z, spread_z=spread_z)
    result = {
        "model": ARGS.model, "horizon": ARGS.horizon,
        "labels": ARGS.labels,
        "seq_size": ARGS.seq_size, "n_features": ARGS.n_features,
        "stride": ARGS.stride, "split": ARGS.split,
        "epochs": ARGS.epochs, "batch": ARGS.batch,
        "test": test,
        "test_windows": len(test_w),
    }
    if extra:
        result.update(extra)
    with open(os.path.join(RESULTS_DIR, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {RESULTS_DIR}/{tag}.json + {tag}_test_preds.npz")


if __name__ == "__main__":
    main()
