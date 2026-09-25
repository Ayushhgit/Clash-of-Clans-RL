"""Template-matching pre-labeller.

The screenshots all come from one game at one zoom level, so a building looks
essentially identical everywhere it appears. That makes normalised
cross-correlation a good bootstrap for Dataset A: identify each structure once
by eye, crop it as a template, and let the matcher find every other instance.

This is the Phase 6 auto-labelling loop with a classical matcher standing in
for detector v1 -- it produces the seed labels that train detector v1, which
then replaces it.

    # 1. cut templates (coordinates come from annotation.assist grid)
    python -m annotation.template_match cut --image <png> --box 463,20,550,153 \
        --cls ARCHER_TOWER

    # 2. match every template against every screenshot
    python -m annotation.template_match run --session data/raw/screenshots/coc_01

    # 3. look at what it did
    python -m annotation.assist check data/raw/screenshots/coc_01/000000009.png

Matching is done on a Sobel-gradient image rather than raw pixels: CoC
buildings sit on grass whose colour shifts between bases, but their outlines
do not.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

TEMPLATE_DIR = pathlib.Path("datasets/templates")


# ------------------------------------------------------------------ features
def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def _sobel(a: np.ndarray) -> np.ndarray:
    """Gradient magnitude. Robust to the background colour changing."""
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def features(img: Image.Image) -> np.ndarray:
    return _sobel(_gray(img))


# ------------------------------------------------------------------ matching
def ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation via FFT. Returns a score map of valid
    top-left positions, values roughly in [-1, 1]."""
    ih, iw = image.shape
    th, tw = template.shape
    if th > ih or tw > iw:
        return np.zeros((0, 0), dtype=np.float32)

    t = template - template.mean()
    t_norm = np.sqrt((t * t).sum()) + 1e-8

    fh, fw = ih + th - 1, iw + tw - 1
    F_img = np.fft.rfft2(image, s=(fh, fw))
    F_tpl = np.fft.rfft2(t[::-1, ::-1], s=(fh, fw))
    corr = np.fft.irfft2(F_img * F_tpl, s=(fh, fw))
    corr = corr[th - 1:ih, tw - 1:iw]

    # local sum / sum-of-squares for the denominator (integral images)
    ones = np.ones((th, tw), dtype=np.float32)
    F_ones = np.fft.rfft2(ones[::-1, ::-1], s=(fh, fw))
    s1 = np.fft.irfft2(F_img * F_ones, s=(fh, fw))[th - 1:ih, tw - 1:iw]
    F_sq = np.fft.rfft2(image * image, s=(fh, fw))
    s2 = np.fft.irfft2(F_sq * F_ones, s=(fh, fw))[th - 1:ih, tw - 1:iw]

    n = th * tw
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    denom = np.sqrt(var) * t_norm + 1e-8
    return (corr / denom).astype(np.float32)


def nms(boxes: list, iou_thr: float = 0.25) -> list:
    """Greedy NMS across all classes: two structures cannot occupy one spot."""
    boxes = sorted(boxes, key=lambda b: -b["score"])
    keep: list = []
    for b in boxes:
        if all(_iou(b["xyxy"], k["xyxy"]) < iou_thr for k in keep):
            keep.append(b)
    return keep


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-9)


def match_image(path: pathlib.Path, templates: list, threshold: float = 0.55,
                max_per_template: int = 40) -> list:
    img = Image.open(path).convert("RGB")
    feat = features(img)
    found = []
    for t in templates:
        score_map = ncc(feat, t["feat"])
        if score_map.size == 0:
            continue
        th, tw = t["feat"].shape
        hits = np.argwhere(score_map >= threshold)
        if len(hits) > max_per_template * 20:      # pathological template
            order = np.argsort(-score_map[hits[:, 0], hits[:, 1]])[: max_per_template * 20]
            hits = hits[order]
        for y, x in hits:
            found.append({"cls": t["cls"], "score": float(score_map[y, x]),
                          "xyxy": [float(x), float(y), float(x + tw), float(y + th)]})
    return nms(found)


# ----------------------------------------------------------------- templates
def cut(image: pathlib.Path, box: tuple, cls: str, name: str | None = None) -> pathlib.Path:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(image).convert("RGB").crop(box)
    existing = len(list(TEMPLATE_DIR.glob(f"{cls}_*.png")))
    out = TEMPLATE_DIR / f"{cls}_{name or existing:02}.png" if name else \
        TEMPLATE_DIR / f"{cls}_{existing:02d}.png"
    img.save(out)
    return out


def load_templates(only: set | None = None) -> list:
    out = []
    for p in sorted(TEMPLATE_DIR.glob("*.png")):
        cls = p.stem.rsplit("_", 1)[0]
        if only and cls not in only:
            continue
        out.append({"cls": cls, "path": p, "feat": features(Image.open(p))})
    return out


# ---------------------------------------------------------------------- main
def run(session: pathlib.Path, threshold: float, only: set | None,
        merge: bool, states: set | None) -> dict:
    templates = load_templates(only)
    if not templates:
        raise SystemExit(f"no templates in {TEMPLATE_DIR}; cut some first")

    meta = {}
    mp = session / "meta.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                meta[d["path"]] = d

    counts: dict = {}
    for p in sorted(session.glob("*.png")):
        if states and meta.get(p.name, {}).get("screen_state") not in states:
            continue
        boxes = match_image(p, templates, threshold)

        lp = p.with_suffix(".json")
        old = json.loads(lp.read_text())["boxes"] if (merge and lp.exists()) else []
        # keep hand-drawn boxes AND the measured UI layout; only re-generate
        # boxes that this matcher produced on a previous run
        keep_old = [b for b in old if b.get("source") != "template"]
        boxes = [b for b in boxes
                 if all(_iou(b["xyxy"], o["xyxy"]) < 0.3 for o in keep_old)]

        payload = keep_old + [{"cls": b["cls"], "xyxy": [round(v, 1) for v in b["xyxy"]],
                               "auto": True, "source": "template",
                               "score": round(b["score"], 3)}
                              for b in boxes]
        img = Image.open(p)
        lp.write_text(json.dumps({"image": p.name, "w": img.width, "h": img.height,
                                  "boxes": payload}, indent=1))
        for b in payload:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
        print(f"{p.name}: {len(payload)} boxes", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("cut")
    c.add_argument("--image", required=True)
    c.add_argument("--box", required=True, help="x1,y1,x2,y2")
    c.add_argument("--cls", required=True)
    c.add_argument("--name", default=None)

    r = sub.add_parser("run")
    r.add_argument("--session", default="data/raw/screenshots/coc_01")
    r.add_argument("--threshold", type=float, default=0.55)
    r.add_argument("--only", nargs="*", default=None)
    r.add_argument("--states", nargs="*", default=None)
    r.add_argument("--no-merge", action="store_true")

    ls = sub.add_parser("list")
    args = ap.parse_args()

    if args.mode == "cut":
        box = tuple(int(v) for v in args.box.split(","))
        print(cut(pathlib.Path(args.image), box, args.cls, args.name))
    elif args.mode == "list":
        for t in load_templates():
            print(f"{t['cls']:<20} {t['feat'].shape}  {t['path'].name}")
    else:
        counts = run(pathlib.Path(args.session), args.threshold,
                     set(args.only) if args.only else None,
                     not args.no_merge,
                     set(args.states) if args.states else None)
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
