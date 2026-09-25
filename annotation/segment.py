"""Building-blob segmentation -- EXPERIMENTAL, DOES NOT WORK YET.

Kept because the three failures are worth knowing about before anyone tries
again; not wired into any workflow. See `not_grass` for what was tried.

Original idea: drawing boxes is the slow part of annotation and naming an
existing box is one keypress, so propose a box around every structure
automatically and leave only naming to a human.

Drawing boxes is the slow part of annotation; naming a box that already exists
is one keypress. So this does the slow part automatically and leaves the
naming to a human (or, later, to detector v1).

How it works: in this game the ground is green and the structures are not.
A "not grass" mask, cleaned up morphologically and split into connected
components, recovers one blob per building almost everywhere. Walls are
excluded by shape -- they form long thin diagonal chains rather than compact
blobs -- and by an explicit aspect/area filter.

    python -m annotation.segment run --session data/raw/screenshots/coc_01 \
        --states BASE_PREVIEW
    python -m annotation.assist check data/raw/screenshots/coc_01/000000022.png

Everything it emits is `UNLABELED_BUILDING`, `auto: true`, `source:
"segment"`, so a later pass can tell generated boxes from human ones.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PROPOSED_CLASS = "UNLABELED_BUILDING"


def local_std(a: np.ndarray, k: int = 9) -> np.ndarray:
    """Local standard deviation via box sums -- a cheap texture measure."""
    m = ndi.uniform_filter(a, size=k)
    m2 = ndi.uniform_filter(a * a, size=k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def not_grass(img: Image.Image, tex_thr: float = 0.055,
              hue_lo: float = 35.0, hue_hi: float = 160.0,
              sat_min: float = 0.18) -> np.ndarray:
    """True where a pixel is likely part of a structure.

    Two attempts failed before this one, and both failures are informative:

    * "green channel dominant" (RGB): the mown grass *inside* a base is a
      yellow-green with r ~= g, so the whole base interior read as structure.
    * "hue in a green band" (HSV): the interior ground is an olive/tan that
      falls outside any band that still excludes gold buildings.

    * "local contrast" (this one): the grass in this game is *itself* a noisy
      texture, so its local standard deviation is comparable to a building's
      and the mask covers 86-96% of the frame.

    A fourth attempt should probably not be another colour/texture heuristic.
    The promising direction is geometric: bases sit on an isometric tile grid,
    the wall chains expose that grid, and building footprints are whole
    numbers of tiles. Fit the grid, then snap proposals to it.
    """
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0

    textured = ndi.uniform_filter(local_std(g), size=11) > tex_thr
    plain_grass = (hue >= hue_lo) & (hue <= hue_hi) & (sat >= sat_min)
    return textured & ~plain_grass


def clean(mask: np.ndarray, close: int = 5, open_: int = 3) -> np.ndarray:
    m = ndi.binary_closing(mask, structure=np.ones((close, close)))
    m = ndi.binary_opening(m, structure=np.ones((open_, open_)))
    return ndi.binary_fill_holes(m)


def components(mask: np.ndarray, min_area: int = 500, max_area: int = 40_000,
               min_side: int = 18, max_side: int = 220,
               max_aspect: float = 3.2) -> list:
    lab, n = ndi.label(mask)
    out = []
    for sl, i in zip(ndi.find_objects(lab), range(1, n + 1)):
        if sl is None:
            continue
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        area = int((lab[sl] == i).sum())
        if not (min_area <= area <= max_area):
            continue
        if min(h, w) < min_side or max(h, w) > max_side:
            continue
        if max(h, w) / max(min(h, w), 1) > max_aspect:
            continue                      # long thin chain: a wall run
        fill = area / float(h * w)
        if fill < 0.35:                   # sparse, ragged: usually wall or trees
            continue
        out.append({"xyxy": [float(xs.start), float(ys.start),
                             float(xs.stop), float(ys.stop)],
                    "area": area, "fill": round(fill, 2)})
    return out


def propose(path: pathlib.Path, ui_boxes: list | None = None, **kw) -> list:
    img = Image.open(path).convert("RGB")
    mask = clean(not_grass(img))
    # never propose inside the HUD: it is already labelled exactly
    if ui_boxes:
        for b in ui_boxes:
            x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
            mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = False
    return components(mask, **kw)


def run(session: pathlib.Path, states: set | None, **kw) -> dict:
    meta = {}
    mp = session / "meta.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                meta[d["path"]] = d

    total, per_image = 0, {}
    for p in sorted(session.glob("*.png")):
        if states and meta.get(p.name, {}).get("screen_state") not in states:
            continue
        lp = p.with_suffix(".json")
        old = json.loads(lp.read_text())["boxes"] if lp.exists() else []
        keep = [b for b in old if b.get("source") != "segment"]

        props = propose(p, ui_boxes=keep, **kw)
        boxes = []
        for c in props:
            if all(_iou(c["xyxy"], o["xyxy"]) < 0.35 for o in keep):
                boxes.append({"cls": PROPOSED_CLASS,
                              "xyxy": [round(v, 1) for v in c["xyxy"]],
                              "auto": True, "source": "segment",
                              "area": c["area"], "fill": c["fill"]})
        img = Image.open(p)
        lp.write_text(json.dumps({"image": p.name, "w": img.width, "h": img.height,
                                  "boxes": keep + boxes}, indent=1))
        per_image[p.name] = len(boxes)
        total += len(boxes)
    return {"images": len(per_image), "proposals": total,
            "median_per_image": int(np.median(list(per_image.values()))) if per_image else 0}


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--session", default="data/raw/screenshots/coc_01")
    r.add_argument("--states", nargs="*", default=["BASE_PREVIEW"])
    r.add_argument("--min-area", type=int, default=500)
    r.add_argument("--max-area", type=int, default=40000)
    r.add_argument("--max-aspect", type=float, default=3.2)

    d = sub.add_parser("debug")
    d.add_argument("--image", required=True)
    d.add_argument("--out", default="data/raw/_mask.png")

    args = ap.parse_args()
    if args.mode == "debug":
        img = Image.open(args.image).convert("RGB")
        m = clean(not_grass(img))
        vis = np.asarray(img).copy()
        vis[m] = (vis[m] * 0.45 + np.array([255, 60, 60]) * 0.55).astype(np.uint8)
        Image.fromarray(vis).save(args.out)
        print(args.out, f"{m.mean() * 100:.1f}% of pixels flagged")
        return
    print(json.dumps(run(pathlib.Path(args.session),
                         set(args.states) if args.states else None,
                         min_area=args.min_area, max_area=args.max_area,
                         max_aspect=args.max_aspect), indent=2))


if __name__ == "__main__":
    main()
