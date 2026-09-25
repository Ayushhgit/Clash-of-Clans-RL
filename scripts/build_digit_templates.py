"""Bootstrap the digit templates from numbers a human read off the screen.

    python -m scripts.build_digit_templates

The HUD font never changes, so a couple of labelled examples per digit is
enough. The examples below are frames from the captured sessions whose numbers
are legible to a person.

Two passes, and the order matters. A template cut from a clean segmentation is
exact. A template recovered by evenly splitting a blob of touching digits is
approximate, and mixing those in caused 0/5 confusion -- "3360" read back as
"33151". So harvest every clean example first, then fall back to splitting only
for digits that never appear on their own.
"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

from ai.perception.digits import (
    crop_frac,
    glyph_image,
    load_templates,
    read_number,
    save_template,
    segment,
)

# (frame path, crop box as viewport fractions, the number as a human reads it)
EXAMPLES = [
    ("data/raw/screenshots/live_loop_02/000000032.png", (0.44, 0.44, 0.60, 0.52), "412311"),
    ("data/raw/screenshots/live_loop_02/000000032.png", (0.44, 0.52, 0.60, 0.60), "411999"),
    ("data/raw/screenshots/live_loop_02/000000032.png", (0.44, 0.60, 0.60, 0.68), "3360"),
    ("data/raw/screenshots/coc_01/000000031.png",       (0.42, 0.42, 0.62, 0.52), "766403"),
    ("data/raw/screenshots/coc_01/000000031.png",       (0.40, 0.50, 0.58, 0.58), "708081"),
    ("data/raw/screenshots/coc_01/000000022.png",       (0.03, 0.11, 0.12, 0.16), "551731"),
    # loot panel on a live BASE_PREVIEW: well-separated digits, and the only
    # clean example of 5 and 8 in the captures
    ("data/raw/screenshots/live_loop_04/000000008.png", (0.06, 0.140, 0.17, 0.190), "687933"),
    ("data/raw/screenshots/live_loop_04/000000008.png", (0.06, 0.185, 0.17, 0.235), "579832"),
]
MAX_CLEAN_PER_DIGIT = 2


def split_to_count(boxes: list, want: int) -> list:
    """Split blobs of touching digits into `want` glyph boxes."""
    if not boxes or len(boxes) >= want:
        return boxes
    widths = np.array([b[2] - b[0] for b in boxes], dtype=np.float64)
    counts = np.maximum(np.round(widths / widths.sum() * want).astype(int), 1)
    while counts.sum() > want:
        counts[int(np.argmax(counts))] -= 1
    while counts.sum() < want:
        counts[int(np.argmax(widths / counts))] += 1

    out = []
    for (x1, y1, x2, y2), k in zip(boxes, counts):
        step = (x2 - x1) / k
        out += [(int(x1 + i * step), y1, int(x1 + (i + 1) * step), y2)
                for i in range(k)]
    return out


def harvest(counts: dict, clean_pass: bool) -> None:
    for path, box, text in EXAMPLES:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        crop = crop_frac(rgb, box)
        boxes = segment(crop)
        is_clean = len(boxes) == len(text)
        if clean_pass != is_clean:
            continue
        if not is_clean:
            boxes = split_to_count(boxes, len(text))
            if len(boxes) != len(text):
                print(f"skip  {p.name} {text!r}: {len(boxes)} glyphs for {len(text)} digits")
                continue

        taken = []
        for ch, b in zip(text, boxes):
            idx = counts.get(ch, 0)
            limit = MAX_CLEAN_PER_DIGIT if clean_pass else 1
            if idx >= limit:
                continue
            save_template(ch, glyph_image(crop, b), idx)
            counts[ch] = idx + 1
            taken.append(ch)
        tag = "clean" if clean_pass else "split"
        print(f"{tag:<5} {p.name} {text!r}  -> new templates: {''.join(taken) or '-'}")


def refine() -> None:
    """Re-cut every template by aligning the known string to the strip.

    The bootstrap templates are good enough to guide an alignment, and the
    alignment then gives exact cut points even where digits touch -- which is
    the only way to get a clean '5' and '8' out of these frames.
    """
    from ai.perception.digits import align_known, band_glyph

    templates = load_templates()
    counts: dict = {}
    for path, box, text in EXAMPLES:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        crop = crop_frac(np.asarray(Image.open(p).convert("RGB")), box)
        spans = align_known(crop, text, templates)
        if not spans or len(spans) != len(text):
            print(f"refine skip {p.name} {text!r}")
            continue
        for ch, span in zip(text, spans):
            g = band_glyph(crop, span)
            if g is None or g.shape[1] < 4:
                continue
            idx = counts.get(ch, 0)
            if idx >= MAX_CLEAN_PER_DIGIT:
                continue
            save_template(ch, g, idx)
            counts[ch] = idx + 1
        print(f"refine {p.name} {text!r}")


def main() -> None:
    counts: dict = {}
    harvest(counts, clean_pass=True)
    harvest(counts, clean_pass=False)
    # NB: a label-guided re-cut (`refine`) was tried here and made things
    # worse -- 5/6 exact fell to 3/6. Letting the alignment choose glyph widths
    # freely lets it drift, because a window resized to a template's width
    # matches almost anything. Left out deliberately.

    have = sorted(load_templates())
    missing = [d for d in "0123456789" if d not in have]
    print(f"\ntemplates for: {''.join(have) or '(none)'}")
    if missing:
        print(f"MISSING digits: {''.join(missing)} -- add an example containing them")

    print("\nre-read check:")
    ok = 0
    for path, box, text in EXAMPLES:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        got = read_number(crop_frac(rgb, box))
        good = str(got) == text
        ok += good
        print(f"  {p.name} expected {text:>8}  got {str(got):>8}  "
              f"{'OK' if good else 'MISMATCH'}")
    print(f"\n{ok}/{len(EXAMPLES)} exact")


if __name__ == "__main__":
    main()
