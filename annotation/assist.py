"""Labelling aids: coordinate grids and label overlays.

Two tools, used as a loop:

    grid   -- render an image with a labelled pixel grid, so box corners can be
              read off directly instead of estimated
    check  -- render existing labels back onto the image, so a wrong box is
              obvious at a glance

    python -m annotation.assist grid  data/raw/screenshots/coc_01/000000009.png
    python -m annotation.assist grid  ...png --crop 600,300,1180,670 --zoom 2
    python -m annotation.assist check data/raw/screenshots/coc_01/000000009.png
    python -m annotation.assist stats data/raw/screenshots/coc_01
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

from PIL import Image, ImageDraw

from annotation.app import CLASSES, PALETTE


def grid(path: pathlib.Path, out: pathlib.Path, step: int = 50,
         crop: tuple | None = None, zoom: float = 1.0) -> pathlib.Path:
    img = Image.open(path).convert("RGB")
    ox, oy = 0, 0
    if crop:
        ox, oy = crop[0], crop[1]
        img = img.crop(crop)
    if zoom != 1.0:
        img = img.resize((int(img.width * zoom), int(img.height * zoom)), Image.LANCZOS)

    d = ImageDraw.Draw(img, "RGBA")
    for gx in range(0, img.width, int(step * zoom)):
        src_x = ox + int(gx / zoom)
        major = src_x % (step * 2) == 0
        d.line([(gx, 0), (gx, img.height)], fill=(255, 255, 255, 110 if major else 45), width=1)
        if major:
            d.text((gx + 2, 2), str(src_x), fill=(255, 255, 0))
    for gy in range(0, img.height, int(step * zoom)):
        src_y = oy + int(gy / zoom)
        major = src_y % (step * 2) == 0
        d.line([(0, gy), (img.width, gy)], fill=(255, 255, 255, 110 if major else 45), width=1)
        if major:
            d.text((2, gy + 2), str(src_y), fill=(0, 255, 255))
    img.save(out)
    return out


def check(path: pathlib.Path, out: pathlib.Path, zoom: float = 1.0) -> pathlib.Path:
    img = Image.open(path).convert("RGB")
    lp = path.with_suffix(".json")
    boxes = json.loads(lp.read_text())["boxes"] if lp.exists() else []
    d = ImageDraw.Draw(img, "RGBA")
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        col = PALETTE[CLASSES.index(b["cls"]) % len(PALETTE)] if b["cls"] in CLASSES else "#ffffff"
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        d.text((x1 + 2, max(y1 - 10, 0)), b["cls"][:14], fill=col)
    if zoom != 1.0:
        img = img.resize((int(img.width * zoom), int(img.height * zoom)), Image.LANCZOS)
    img.save(out)
    return out


def stats(session: pathlib.Path) -> dict:
    counts, labelled, total = Counter(), 0, 0
    for p in sorted(session.glob("*.png")):
        total += 1
        lp = p.with_suffix(".json")
        if not lp.exists():
            continue
        labelled += 1
        for b in json.loads(lp.read_text())["boxes"]:
            counts[b["cls"]] += 1
    return {"images": total, "labelled": labelled, "boxes": sum(counts.values()),
            "per_class": dict(counts.most_common())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grid", "check", "stats"])
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--crop", default=None, help="x1,y1,x2,y2")
    ap.add_argument("--zoom", type=float, default=1.0)
    args = ap.parse_args()

    p = pathlib.Path(args.path)
    if args.mode == "stats":
        print(json.dumps(stats(p), indent=2))
        return

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    out = pathlib.Path(args.out or f"data/raw/_{args.mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fn = grid if args.mode == "grid" else check
    kw = {"step": args.step, "crop": crop, "zoom": args.zoom} if args.mode == "grid" else {"zoom": args.zoom}
    print(fn(p, out, **kw))


if __name__ == "__main__":
    main()
