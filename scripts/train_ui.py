"""Phase 8 / M2: train the screen-state classifier on Dataset B.

    python -m scripts.train_ui --epochs 30

Dataset B is the cheap half of perception: labels are one word per frame, and
`scripts/ingest.py` + a contact-sheet pass already produced them for every
captured frame. That makes this the first perception model that can actually
be trained on real game data.

The class distribution is heavily skewed (BASE_PREVIEW dominates because that
is what you sit on while searching), so the loss is class-weighted and the
report shows per-class recall rather than a single accuracy that a
majority-class predictor would ace.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ai.perception.dataset import UI_STATES, UIStateDataset
from ai.perception.ui import UIClassifier


def split_indices(labels: list, val_frac: float, seed: int = 0) -> tuple:
    """Stratified split: every state must appear in both halves when it can."""
    rng = np.random.default_rng(seed)
    by: dict = {}
    for i, y in enumerate(labels):
        by.setdefault(y, []).append(i)
    train, val = [], []
    for y, idx in by.items():
        idx = list(rng.permutation(idx))
        n_val = 1 if len(idx) < 4 else max(1, int(len(idx) * val_frac))
        val += idx[:n_val]
        train += idx[n_val:] or idx[:1]
    return train, val


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    n_correct, n_total = 0, 0
    per_class = {s: [0, 0] for s in UI_STATES}
    confusion: dict = {}
    for x, y in loader:
        pred = model(x.to(device)).argmax(-1).cpu()
        for p, t in zip(pred.tolist(), y.tolist()):
            per_class[UI_STATES[t]][1] += 1
            if p == t:
                per_class[UI_STATES[t]][0] += 1
                n_correct += 1
            else:
                confusion[f"{UI_STATES[t]}->{UI_STATES[p]}"] = \
                    confusion.get(f"{UI_STATES[t]}->{UI_STATES[p]}", 0) + 1
            n_total += 1
    model.train()
    return {
        "accuracy": n_correct / max(n_total, 1),
        "recall": {k: round(v[0] / v[1], 3) for k, v in per_class.items() if v[1]},
        "support": {k: v[1] for k, v in per_class.items() if v[1]},
        "confusions": confusion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=["data/raw/screenshots/coc_01"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--out", default="checkpoints/ui_classifier.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    full = UIStateDataset(args.sessions, img_size=args.img_size, augment=True)
    if not len(full):
        raise SystemExit("no labelled frames; run scripts/ingest.py and label states first")
    labels = [y for _, y in full.items]
    tr_idx, va_idx = split_indices(labels, args.val_frac)

    val_ds = UIStateDataset(args.sessions, img_size=args.img_size, augment=False)
    train_dl = DataLoader(Subset(full, tr_idx), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(Subset(val_ds, va_idx), batch_size=args.batch_size)

    counts = np.bincount(labels, minlength=len(UI_STATES)).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    weights = weights / weights[weights > 0].mean()
    print(f"{len(full)} frames  train {len(tr_idx)}  val {len(va_idx)}")
    print("class counts:", {UI_STATES[i]: int(c) for i, c in enumerate(counts) if c})

    device = torch.device(args.device)
    model = UIClassifier(width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    best, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        tot, n = 0.0, 0
        for x, y in train_dl:
            loss = lossf(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        row = {"epoch": epoch, "loss": round(tot / max(n, 1), 4)}
        if epoch % 5 == 0 or epoch == args.epochs:
            row.update(evaluate(model, val_dl, device))
            if row["accuracy"] >= best:
                best = row["accuracy"]
                pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "width": args.width,
                            "img_size": args.img_size, "states": UI_STATES,
                            "accuracy": best}, args.out)
        history.append(row)
        print(json.dumps(row), flush=True)

    print(f"\nbest val accuracy {best:.3f} -> {args.out}")
    pathlib.Path("experiments").mkdir(exist_ok=True)
    pathlib.Path("experiments/ui_classifier_history.json").write_text(
        json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
