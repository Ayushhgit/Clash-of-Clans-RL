"""Generate exact UI labels for every frame, from a measured layout.

The HUD does not move between frames of the same screen state -- it is drawn
by the game at fixed positions -- so hand-measuring it once and projecting it
onto every frame is both faster and *more accurate* than detecting it. The
detector still gets trained on these boxes; it just does not have to discover
what a human can measure in five minutes.

Coordinates were read off a coordinate grid (`annotation.assist grid`) on the
reference frames named below, and are stored as fractions of frame size so
they survive the small (+-15 px) differences between captures.

    python -m scripts.label_ui --session data/raw/screenshots/coc_01
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

# reference frame these pixel measurements came from
REF_W, REF_H = 1183.0, 667.0

# class -> (x1, y1, x2, y2) in reference pixels
RECTS = {
    "ATTACK_BUTTON":     (18, 553, 120, 650),     # frame 0   (HOME)
    "RETURN_BUTTON":     (522, 554, 677, 622),    # frame 11  (RESULT)
    "NEXT_BUTTON":       (1018, 470, 1157, 549),  # frame 22  (BASE_PREVIEW)
    "END_BATTLE_BUTTON": (12, 515, 129, 546),
    "LOOT_PANEL":        (20, 62, 140, 168),
    "HUD_GOLD":          (1027, 24, 1170, 47),
    "HUD_ELIXIR":        (1027, 56, 1170, 78),
    "HUD_DARK":          (1060, 91, 1160, 110),
    "HUD_TIMER":         (520, 18, 650, 62),
}

# which of the above are drawn in which screen state
PER_STATE = {
    "HOME":           ["ATTACK_BUTTON", "HUD_GOLD", "HUD_ELIXIR", "HUD_DARK"],
    "BASE_PREVIEW":   ["NEXT_BUTTON", "END_BATTLE_BUTTON", "LOOT_PANEL",
                       "HUD_GOLD", "HUD_ELIXIR", "HUD_DARK", "HUD_TIMER"],
    "BATTLE":         ["END_BATTLE_BUTTON", "HUD_GOLD", "HUD_ELIXIR", "HUD_DARK",
                       "HUD_TIMER"],
    "BATTLE_CONFIRM": ["HUD_GOLD", "HUD_ELIXIR", "HUD_DARK"],
    "RESULT":         ["RETURN_BUTTON"],
}

# army bar: first slot origin, slot pitch and size, in reference pixels
CARD_X0, CARD_PITCH, CARD_W = 172.0, 72.8, 68.0
CARD_Y0, CARD_Y1 = 572.0, 665.0
CARD_SLOTS = 10
CARD_STATES = {"BASE_PREVIEW", "BATTLE"}


def scale_rect(rect, w: float, h: float) -> list:
    sx, sy = w / REF_W, h / REF_H
    x1, y1, x2, y2 = rect
    return [round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1)]


def card_is_occupied(img: Image.Image, box: list, sat_thr: float = 0.22) -> bool:
    """Empty army slots are a dashed outline over the map; real cards are
    saturated blue/purple artwork. Saturation separates them cleanly."""
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = np.asarray(img.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32) / 255.0
    if crop.size == 0:
        return False
    mx = crop.max(axis=2)
    mn = crop.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(sat.mean()) > sat_thr


def _colour_mask(img: Image.Image, hue_lo: float, hue_hi: float,
                 sat_min: float, val_min: float = 0.25) -> np.ndarray:
    a = np.asarray(img.convert("HSV"), dtype=np.float32)
    hue = a[..., 0] * (360.0 / 255.0)
    sat = a[..., 1] / 255.0
    val = a[..., 2] / 255.0
    return (hue >= hue_lo) & (hue <= hue_hi) & (sat >= sat_min) & (val >= val_min)


def find_anchor(img: Image.Image, state: str) -> tuple | None:
    """Locate the HUD and return the (dx, dy) offset from the reference layout.

    Not every capture is framed identically -- some are cropped or zoomed, so
    the HUD sits somewhere other than where it sits in the reference frame, and
    some show no HUD at all. Rather than assume, find a strongly-coloured
    anchor widget and measure how far it moved:

        BASE_PREVIEW -> the orange NEXT button (bottom right)
        BATTLE       -> the red END BATTLE button (bottom left)

    Returns None when the anchor is not found, which means "this frame does not
    show the HUD" and no UI boxes should be written for it.
    """
    w, h = img.size
    if state == "BATTLE":
        key, mask = "END_BATTLE_BUTTON", _colour_mask(img, 340, 370, 0.55)
        mask |= _colour_mask(img, 0, 12, 0.55)
    elif state == "BASE_PREVIEW":
        key, mask = "NEXT_BUTTON", _colour_mask(img, 20, 45, 0.55)
    else:
        return (0.0, 0.0)

    ref = scale_rect(RECTS[key], w, h)
    rw, rh = ref[2] - ref[0], ref[3] - ref[1]
    # search only where that widget could plausibly be: the same half of the
    # frame, with generous slack
    ys, xs = np.nonzero(mask)
    if len(xs) < 0.25 * rw * rh:
        return None
    # the anchor is the densest cluster; median is a robust centre estimate
    if key == "NEXT_BUTTON":
        sel = (xs > w * 0.6) & (ys > h * 0.55) & (ys < h * 0.95)
    else:
        sel = (xs < w * 0.35) & (ys > h * 0.6) & (ys < h * 0.95)
    if sel.sum() < 0.25 * rw * rh:
        return None
    cx, cy = float(np.median(xs[sel])), float(np.median(ys[sel]))
    ref_cx, ref_cy = (ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2
    dx, dy = cx - ref_cx, cy - ref_cy
    # Tolerance is deliberately tight. A loose search "finds" the button in any
    # orange building and stamps the whole HUD on top of the base -- verified
    # by rendering the result on cropped captures. If the anchor is not almost
    # exactly where the reference frame puts it, treat the HUD as absent.
    if abs(dx) > w * 0.04 or abs(dy) > h * 0.04:
        return None
    return (dx, dy)


def ui_boxes(img: Image.Image, state: str, offset: tuple = (0.0, 0.0)) -> list:
    w, h = img.size
    dx, dy = offset

    def shifted(rect):
        r = scale_rect(rect, w, h)
        return [round(r[0] + dx, 1), round(r[1] + dy, 1),
                round(r[2] + dx, 1), round(r[3] + dy, 1)]

    def on_screen(b):
        return b[0] > -4 and b[1] > -4 and b[2] < w + 4 and b[3] < h + 4

    out = []
    for cls in PER_STATE.get(state, []):
        box = shifted(RECTS[cls])
        if on_screen(box):
            out.append({"cls": cls, "xyxy": box, "auto": True, "source": "layout"})
    if state in CARD_STATES:
        sx, sy = w / REF_W, h / REF_H
        for i in range(CARD_SLOTS):
            x1 = (CARD_X0 + i * CARD_PITCH) * sx + dx
            box = [round(x1, 1), round(CARD_Y0 * sy + dy, 1),
                   round(x1 + CARD_W * sx, 1),
                   round(min(CARD_Y1 * sy + dy, h - 1), 1)]
            if not on_screen(box):
                continue
            if card_is_occupied(img, box):
                out.append({"cls": "ARMY_CARD", "xyxy": box, "auto": True,
                            "source": "layout"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="data/raw/screenshots/coc_01")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    session = pathlib.Path(args.session)
    meta = {}
    for line in (session / "meta.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            meta[d["path"]] = d

    ui_classes = set(RECTS) | {"ARMY_CARD"}
    counts: dict = {}
    skipped: list = []
    for p in sorted(session.glob("*.png")):
        state = meta.get(p.name, {}).get("screen_state", "UNKNOWN")
        img = Image.open(p)
        anchor = find_anchor(img, state)
        boxes = ui_boxes(img, state, anchor) if anchor is not None else []
        if not boxes and state in ("BASE_PREVIEW", "BATTLE"):
            skipped.append(p.name)

        lp = p.with_suffix(".json")
        old = json.loads(lp.read_text())["boxes"] if lp.exists() else []
        # drop earlier *generated* UI guesses; keep hand-drawn ones and all
        # non-UI boxes
        old = [b for b in old if not (b.get("auto") and b["cls"] in ui_classes)]

        if not args.dry_run:
            lp.write_text(json.dumps({"image": p.name, "w": img.width,
                                      "h": img.height, "boxes": old + boxes}, indent=1))
        for b in boxes:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    print(json.dumps(counts, indent=2))
    if skipped:
        print(f"\n{len(skipped)} frames have no HUD on screen (cropped/zoomed "
              f"captures); no UI boxes written for them:")
        print("  " + " ".join(skipped[:12]) + (" ..." if len(skipped) > 12 else ""))


if __name__ == "__main__":
    main()
