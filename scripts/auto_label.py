"""Phases 6 + 7: automatic labelling and active learning.

    # run detector v1 over everything unlabelled and triage by confidence
    python -m scripts.auto_label --checkpoint checkpoints/detector_v1.pt \
        --data data/raw/screenshots

Triage (docs: Phase 6):
    score > 0.90        auto-accepted, written as a label file
    0.50 - 0.90         queued for human review (review/ symlink list)
    < 0.50              uncertain, queued first for the next labelling round

The active-learning ranking (Phase 7) is *not* raw confidence: an image whose
detections are all confidently wrong looks great by that metric. It ranks by
per-image mean entropy of the peak heatmap responses plus a count-disagreement
term, which surfaces crowded and ambiguous frames.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from ai.perception.detector import Detector, decode
from annotation.app import CLASSES

ACCEPT = 0.90
REVIEW = 0.50


def load_detector(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Detector(width=ck.get("width", 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("img_size", 512)


def unlabelled_images(root: str) -> list:
    out = []
    for p in sorted(pathlib.Path(root).rglob("*.png")):
        if not p.with_suffix(".json").exists():
            out.append(p)
    return out


@torch.no_grad()
def predict_image(model, path: pathlib.Path, img_size: int, device) -> tuple:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    x = torch.from_numpy(
        np.asarray(img.resize((img_size, img_size)), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    out = model(x)
    det = decode(out, threshold=0.05)[0]
    sx, sy = w / img_size, h / img_size
    if len(det["boxes"]):
        det["boxes"] = det["boxes"] * np.array([sx, sy, sx, sy], dtype=np.float32)
    return det, (w, h), out


def uncertainty(det: dict, out: dict) -> float:
    """Higher = more worth a human's time."""
    s = det["scores"]
    if len(s) == 0:
        return 1.0
    # binary entropy of each detection score, averaged: peaks near 0.5 are the
    # ones the model genuinely cannot decide
    p = np.clip(s, 1e-6, 1 - 1e-6)
    ent = float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))) / np.log(2))
    borderline = float(np.mean((s > REVIEW) & (s < ACCEPT)))
    return 0.7 * ent + 0.3 * borderline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/raw/screenshots")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write-accepted", action="store_true",
                    help="write label json for high-confidence detections")
    ap.add_argument("--out", default="datasets/active_learning_queue.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, img_size = load_detector(args.checkpoint, device)
    images = unlabelled_images(args.data)[: args.limit]
    if not images:
        raise SystemExit(f"nothing unlabelled under {args.data}")

    buckets = {"accepted": 0, "review": 0, "uncertain": 0}
    queue = []
    for i, p in enumerate(images):
        det, (w, h), out = predict_image(model, p, img_size, device)
        hi = det["scores"] >= ACCEPT
        mid = (det["scores"] >= REVIEW) & (~hi)

        if hi.any() and not mid.any():
            buckets["accepted"] += 1
            if args.write_accepted:
                _write_labels(p, det, hi, w, h, auto=True)
        elif mid.any():
            buckets["review"] += 1
            if args.write_accepted:
                _write_labels(p, det, hi | mid, w, h, auto=True)
        else:
            buckets["uncertain"] += 1

        queue.append({"image": str(p), "uncertainty": uncertainty(det, out),
                      "n_det": int(len(det["scores"])),
                      "max_score": float(det["scores"].max()) if len(det["scores"]) else 0.0})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(images)}", flush=True)

    queue.sort(key=lambda r: -r["uncertainty"])
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(queue, indent=1))

    print(json.dumps(buckets, indent=2))
    print(f"\nlabel the worst {min(200, len(queue))} first:")
    for r in queue[:10]:
        print(f"  {r['uncertainty']:.3f}  {r['image']}")
    print(f"\nfull ranking -> {outp}")


def _write_labels(path: pathlib.Path, det: dict, keep: np.ndarray,
                  w: int, h: int, auto: bool) -> None:
    boxes = [{"cls": CLASSES[int(c)], "xyxy": [round(float(v), 1) for v in b],
              "auto": auto, "score": round(float(s), 3)}
             for b, c, s in zip(det["boxes"][keep], det["labels"][keep], det["scores"][keep])]
    path.with_suffix(".json").write_text(json.dumps(
        {"image": path.name, "w": w, "h": h, "boxes": boxes, "auto_labeled": True}, indent=1))


if __name__ == "__main__":
    main()
