"""Find the game's buttons at whatever resolution the window happens to be.

`configs/geometry.json` stores button rectangles as fractions of the frame,
measured once. That works only for the resolution it was measured at: Clash of
Clans does not scale its HUD linearly with the window, so boxes measured at
1183x667 miss the buttons at 1670x949.

The buttons are, however, unmistakable blobs of saturated colour in known
corners of the screen:

    ATTACK  bottom-left    orange, tall
    NEXT    bottom-right   orange, wide
    RETURN  bottom-centre  green
    END     bottom-left    red, small

So find them by colour, constrained to the corner each one lives in. That is
resolution-independent and needs no training data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

@dataclass(frozen=True)
class Spec:
    hue: tuple                  # (lo, hi) degrees, may wrap through 0
    sat_min: float
    val_min: float
    box: tuple                  # search region, frame fractions (x0, y0, x1, y1)
    area: tuple = (0.0008, 0.05)   # min/max blob area as a fraction of the frame
    aspect: tuple = (0.5, 6.0)     # width / height
    # Buttons carry text and icons that punch holes in the colour blob: the
    # army screen's "Attack!" filled only 0.42 of its bounding box and a fixed
    # 0.45 threshold rejected it.
    fill_min: float = 0.35

# The green RETURN button and *grass* share a hue. The first version of this
# table had only a hue range, so on every BASE_PREVIEW frame the lawn matched
# and the state came back RESULT (35 of 89 frames). Buttons are distinguished
# from scenery by being small, bounded in aspect, and much more saturated.
SPECS = {
    "ATTACK_BUTTON":     Spec((18, 45), 0.55, 0.45, (0.00, 0.72, 0.20, 1.00),
                              area=(0.002, 0.05), aspect=(0.5, 2.0)),
    "NEXT_BUTTON":       Spec((18, 45), 0.55, 0.45, (0.80, 0.60, 1.00, 0.86),
                              area=(0.002, 0.05), aspect=(0.8, 4.0)),
    "RETURN_BUTTON":     Spec((85, 155), 0.55, 0.60, (0.30, 0.72, 0.70, 1.00),
                              area=(0.002, 0.03), aspect=(1.2, 5.0)),
    "END_BATTLE_BUTTON": Spec((345, 375), 0.45, 0.40, (0.00, 0.70, 0.22, 1.00),
                              area=(0.0008, 0.03), aspect=(1.2, 6.0)),
    "SHOP_BUTTON":       Spec((18, 45), 0.55, 0.45, (0.86, 0.86, 1.00, 1.00),
                              area=(0.002, 0.05), aspect=(0.5, 3.0)),
    # The multiplayer "Find a Match" button, left card of the attack menu. The
    # square orange icons on the HOME rail sit in the same band and the same
    # hue, so the aspect ratio is what separates them: this button is wide.
    "FIND_MATCH_BUTTON": Spec((18, 45), 0.55, 0.50, (0.01, 0.58, 0.27, 0.84),
                              area=(0.004, 0.05), aspect=(2.5, 9.0)),
    # Confirming a match opens the army screen, whose green "Attack!" actually
    # starts matchmaking. Same green as RETURN_BUTTON but bottom-right rather
    # than bottom-centre, so the search boxes keep them apart.
    "ARMY_ATTACK_BUTTON": Spec((85, 155), 0.50, 0.55, (0.70, 0.76, 1.00, 0.98),
                               area=(0.002, 0.04), aspect=(1.5, 6.0)),
}


@dataclass
class Button:
    name: str
    xyxy: tuple
    area: int
    score: float

    @property
    def centre(self) -> tuple:
        x1, y1, x2, y2 = self.xyxy
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _hsv(rgb: np.ndarray) -> tuple:
    a = rgb.astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    diff = mx - mn
    hue = np.zeros_like(mx)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = diff > 1e-6
    idx = m & (mx == r)
    hue[idx] = (60 * ((g - b)[idx] / diff[idx])) % 360
    idx = m & (mx == g)
    hue[idx] = 60 * ((b - r)[idx] / diff[idx]) + 120
    idx = m & (mx == b)
    hue[idx] = 60 * ((r - g)[idx] / diff[idx]) + 240
    sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return hue, sat, mx


def find_buttons(rgb: np.ndarray, names=None) -> dict:
    h, w = rgb.shape[:2]
    hue, sat, val = _hsv(rgb)
    out: dict = {}

    for name, spec in SPECS.items():
        if names and name not in names:
            continue
        h0, h1 = spec.hue
        x0, y0, x1, y1 = (int(spec.box[0] * w), int(spec.box[1] * h),
                          int(spec.box[2] * w), int(spec.box[3] * h))
        sub_hue = hue[y0:y1, x0:x1]
        sub = ((sub_hue >= h0) & (sub_hue <= h1) if h0 <= h1
               else ((sub_hue >= h0) | (sub_hue <= h1)))
        sub &= (sat[y0:y1, x0:x1] >= spec.sat_min) & (val[y0:y1, x0:x1] >= spec.val_min)
        sub = ndi.binary_closing(sub, structure=np.ones((5, 5)))

        lab, n = ndi.label(sub)
        if n == 0:
            continue
        best = None
        for sl, i in zip(ndi.find_objects(lab), range(1, n + 1)):
            ys, xs = sl
            area = int((lab[sl] == i).sum())
            frac = area / float(w * h)
            if not (spec.area[0] <= frac <= spec.area[1]):
                continue
            bh, bw = ys.stop - ys.start, xs.stop - xs.start
            if not (spec.aspect[0] <= bw / max(bh, 1) <= spec.aspect[1]):
                continue
            fill = area / float(bh * bw)
            if fill < spec.fill_min:
                continue
            if best is None or area > best[0]:
                best = (area, (x0 + xs.start, y0 + ys.start,
                               x0 + xs.stop, y0 + ys.stop), fill)
        if best:
            out[name] = Button(name, best[1], best[0], round(best[2], 2))
    return out


def find_army_slots(rgb: np.ndarray, min_cards: int = 2) -> list:
    """The army bar: one box per troop/hero/spell card.

    Splitting the bottom strip into "runs of saturated columns" does not work:
    each card contains dark artwork, so a single card breaks into two or three
    runs and the slot indices stop matching real cards. On a 9-card bar that
    produced 6 slots, and half the deploy clicks landed on gaps.

    Cards are evenly pitched, so instead: find the block of columns the bar
    occupies, recover the pitch by autocorrelating the column profile, and lay
    the cards out on that grid.
    """
    h, w = rgb.shape[:2]
    y0 = int(h * 0.86)
    _, sat, val = _hsv(rgb[y0:])
    score = ((sat > 0.30) & (val > 0.30)).mean(axis=0)

    # the bar block: widest region where cards are present at all
    on = score > 0.25
    runs, start = [], None
    for i, f in enumerate(on):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(on)))
    # merge runs separated by a gap narrower than a plausible card
    merged: list = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 40:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r) if False else (r[0], r[1]))
    if not merged:
        return []
    x0, x1 = max(merged, key=lambda r: r[1] - r[0])
    span = x1 - x0
    if span < 60:
        return []

    # pitch by autocorrelation of the (mean-removed) profile inside the block
    seg = score[x0:x1] - score[x0:x1].mean()
    best_lag, best_val = None, 0.0
    for lag in range(60, min(200, span // 2)):
        v = float(np.dot(seg[:-lag], seg[lag:]) / (span - lag))
        if v > best_val:
            best_lag, best_val = lag, v
    pitch = best_lag or 113
    n = max(int(round(span / pitch)), min_cards)
    pitch = span / n

    pad = pitch * 0.08
    boxes = [(int(x0 + i * pitch + pad), y0, int(x0 + (i + 1) * pitch - pad), h)
             for i in range(n)]

    # The block runs past the last card into the empty dashed slots, which pick
    # up enough colour from the map behind them to look occupied. Real cards are
    # solid saturated artwork; keep only those.
    keep = []
    for bx in boxes:
        crop = rgb[bx[1]:bx[3], bx[0]:bx[2]]
        if crop.size == 0:
            continue
        _, csat, cval = _hsv(crop)
        if float(((csat > 0.35) & (cval > 0.30)).mean()) > 0.45:
            keep.append(bx)
    return keep


# --------------------------------------------------------------- state rules
# Which buttons are on screen is a far more reliable signal than a CNN trained
# on 107 frames, and it does not care about resolution. Order matters: the
# first rule whose condition holds wins.
STATE_RULES = [
    ("ARMY",         lambda b: "ARMY_ATTACK_BUTTON" in b),
    ("ATTACK_MENU",  lambda b: "FIND_MATCH_BUTTON" in b),
    ("RESULT",       lambda b: "RETURN_BUTTON" in b),
    ("BASE_PREVIEW", lambda b: "NEXT_BUTTON" in b and "END_BATTLE_BUTTON" in b),
    ("BATTLE",       lambda b: "END_BATTLE_BUTTON" in b and "NEXT_BUTTON" not in b),
    ("HOME",         lambda b: "ATTACK_BUTTON" in b and "SHOP_BUTTON" in b),
]


def infer_state(rgb: np.ndarray, buttons: dict | None = None) -> tuple:
    """(state, buttons). Returns UNKNOWN when no rule matches."""
    b = buttons if buttons is not None else find_buttons(rgb)
    for name, rule in STATE_RULES:
        if rule(b):
            return name, b
    return "UNKNOWN", b
