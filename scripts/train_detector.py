"""Phase 5 / M3: train the object detector on Dataset A.

    python -m scripts.train_detector --data data/raw/screenshots --epochs 40

Reports per-class AP, not just mAP: the useful finding at this stage is
"walls and traps are bad, town halls are fine", which one aggregate number
hides.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai.perception.dataset import DetectionDataset, load_labels

from ai.perception.detector import Detector, decode, detector_loss, average_precision
from annotation.app import CLASSES

# Chrome, not buildings. The HUD boxes are projected onto every frame that
# shows the chrome, so they outnumber hand-drawn building boxes ~20:1.
UI_CLASSES = {"ATTACK_BUTTON", "NEXT_BUTTON", "END_BATTLE_BUTTON",
              "RETURN_BUTTON", "CONFIRM_BUTTON", "ARMY_CARD", "LOOT_PANEL",
              "HUD_GOLD", "HUD_ELIXIR", "HUD_DARK", "HUD_TIMER",
              "HUD_DESTRUCTION"}


def split(samples: list, val_frac: float = 0.15, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_val = max(int(len(samples) * val_frac), 1)
    val = [samples[i] for i in idx[:n_val]]
    train = [samples[i] for i in idx[n_val:]]
    return train, val


@torch.no_grad()
def evaluate(model, samples: list, img_size: int, device, threshold: float = 0.3,
             n_classes: int | None = None, names: list | None = None) -> dict:
    model.eval()
    ds = DetectionDataset(samples, img_size=img_size, augment=False,
                          n_classes=n_classes)
    preds, gts = [], []
    for i in range(len(ds)):
        x, *_ = ds[i]
        out = model(x.unsqueeze(0).to(device))
        preds.append(decode(out, threshold=threshold)[0])
        s = ds.samples[i]
        sx, sy = img_size / max(s.w, 1), img_size / max(s.h, 1)
        gts.append({"boxes": s.boxes * np.array([sx, sy, sx, sy], dtype=np.float32),
                    "labels": s.labels})
    model.train()
    return average_precision(preds, gts, n_classes=n_classes, names=names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to start from (two-stage: pretrain on "
                         "synthetic, then fine-tune on real frames only)")
    ap.add_argument("--train-extra", default=None,
                    help="extra labelled data added to TRAIN only (e.g. the "
                         "synthetic set). Validation stays on --data, so the "
                         "reported AP is always on real frames.")
    ap.add_argument("--coarse", action="store_true",
                    help="collapse the 47 classes into TOWN_HALL / AIR_DEFENSE "
                         "/ LOOT / DEFENSE / OTHER_BUILDING / WALL")
    ap.add_argument("--buildings-only", action="store_true",
                    help="drop HUD/UI boxes and frames that have only those")
    ap.add_argument("--data", default="data/raw/screenshots")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/detector_v1.pt")
    args = ap.parse_args()

    from ai.perception.coarse import COARSE_CLASSES
    samples = load_labels(args.data)
    if args.buildings_only:
        # The HUD labels are projected onto every frame that shows the chrome,
        # so they outnumber hand-drawn building boxes ~20:1 and take over both
        # the loss and the validation split -- the first run reported AP for
        # ARMY_CARD and HUD_TIMER and never saw a building at all.
        from ai.perception.dataset import CLASS_TO_ID
        ui = {CLASS_TO_ID[c] for c in UI_CLASSES if c in CLASS_TO_ID}
        kept = []
        for s_ in samples:
            m = ~np.isin(s_.labels, list(ui))
            if m.any():
                kept.append(type(s_)(s_.image, s_.boxes[m], s_.labels[m], s_.w, s_.h))
        print(f"buildings-only: {len(kept)} of {len(samples)} frames carry "
              f"building boxes ({sum(len(k.labels) for k in kept)} boxes)")
        samples = kept
    if not samples:
        raise SystemExit(
            f"no labelled images under {args.data}\n"
            "label some first:  python -m annotation.app <session_dir>"
        )
    if args.coarse:
        from ai.perception.coarse import COARSE, COARSE_CLASSES, COARSE_TO_ID
        from ai.perception.dataset import CLASS_TO_ID
        id_to_name = {v: k for k, v in CLASS_TO_ID.items()}
        remapped = []
        for s_ in samples:
            new, keep = [], []
            for j, lab in enumerate(s_.labels):
                grp = COARSE.get(id_to_name.get(int(lab), ""), None)
                if grp is not None:
                    new.append(COARSE_TO_ID[grp])
                    keep.append(j)
            if keep:
                remapped.append(type(s_)(s_.image, s_.boxes[keep],
                                         np.array(new, dtype=np.int64),
                                         s_.w, s_.h))
        samples = remapped
        import collections
        hist = collections.Counter(COARSE_CLASSES[int(l)]
                                   for s_ in samples for l in s_.labels)
        print(f"coarse: {len(COARSE_CLASSES)} classes -> {dict(hist)}")

    train, val = split(samples)
    if args.train_extra:
        extra = load_labels(args.train_extra)
        if args.coarse:
            from ai.perception.coarse import COARSE as _C, COARSE_TO_ID as _T
            from ai.perception.dataset import CLASS_TO_ID as _M
            _n = {v: k for k, v in _M.items()}
            conv = []
            for s_ in extra:
                nn_, kk = [], []
                for j, lab in enumerate(s_.labels):
                    g = _C.get(_n.get(int(lab), ""))
                    if g:
                        nn_.append(_T[g]); kk.append(j)
                if kk:
                    conv.append(type(s_)(s_.image, s_.boxes[kk],
                                         np.array(nn_, dtype=np.int64), s_.w, s_.h))
            extra = conv
        print(f"train-extra: +{len(extra)} frames "
              f"({sum(len(e.labels) for e in extra)} boxes), validation "
              f"stays on {len(val)} real frames")
        train = train + extra
    print(f"{len(samples)} labelled frames  ->  train {len(train)}  val {len(val)}")

    device = torch.device(args.device)
    n_classes = len(COARSE_CLASSES) if args.coarse else None
    names = COARSE_CLASSES if args.coarse else None
    model = Detector(width=args.width,
                     **({"n_classes": n_classes} if n_classes else {})).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if unexpected:
            raise SystemExit(f"checkpoint does not fit: unexpected {unexpected[:4]}")
        print(f"[init] loaded {args.init_from} "
              f"(mAP50 {ck.get('mAP50', float('nan')):.3f}), "
              f"{len(missing)} params left fresh")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dl = DataLoader(DetectionDataset(train, img_size=args.img_size, augment=True,
                                     n_classes=n_classes),
                    batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, hm, wh, off, mask, ind in dl:
            x, hm, wh, off = x.to(device), hm.to(device), wh.to(device), off.to(device)
            mask, ind = mask.to(device), ind.to(device)
            loss, parts = detector_loss(model(x), hm, wh, off, mask, ind)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()

        row = {"epoch": epoch, "loss": tot / max(n, 1), "secs": time.perf_counter() - t0}
        if epoch % 5 == 0 or epoch == args.epochs:
            aps = evaluate(model, val, args.img_size, device,
                           n_classes=n_classes, names=names)
            row["mAP50"] = aps["mAP"]
            row["per_class"] = {k: round(v, 3) for k, v in aps.items() if k != "mAP"}
            if aps["mAP"] > best:
                best = aps["mAP"]
                torch.save({"model": model.state_dict(),
                            "classes": names or CLASSES,
                            "width": args.width, "img_size": args.img_size,
                            "mAP50": best}, out_path)
        history.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)

    pathlib.Path("experiments").mkdir(exist_ok=True)
    pathlib.Path("experiments/detector_history.json").write_text(json.dumps(history, indent=2))
    print(f"best mAP@50 {best:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
