"""Read the game's HUD numbers.

Loot totals, damage percentage and resource counters are all drawn in one bold
white font with a dark outline, at a size that only changes with the window.
That makes a template matcher both simpler and far more reliable here than a
learned OCR: segment the white glyphs, scale each to a fixed box, and compare
against one labelled example per digit.

Templates are bootstrapped once from frames whose numbers a human read off the
screen (`scripts/build_digit_templates.py`) and stored in
`datasets/digits/`.

    from ai.perception.digits import read_number
    read_number(rgb_crop)      # -> 412311  (or None)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

TEMPLATE_DIR = pathlib.Path("datasets/digits")
GLYPH_H = 28                     # every glyph is scaled to this height,
                                 # keeping its own width: '1' is much narrower
                                 # than '8', and that width is what lets the
                                 # reader walk along a strip of touching digits


def white_mask(rgb: np.ndarray, val_min: float = 0.72, sat_max: float = 0.30) -> np.ndarray:
    a = rgb.astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (mx >= val_min) & (sat <= sat_max)


def segment(rgb: np.ndarray, min_h_frac: float = 0.35,
            min_area: int = 25) -> list:
    """Split a number strip into glyph boxes, left to right.

    Filters by height rather than area: the digits in a row share a height,
    while specks of white from the background do not.
    """
    m = white_mask(rgb)
    m = ndi.binary_closing(m, structure=np.ones((3, 3)))
    lab, n = ndi.label(m)
    if n == 0:
        return []

    boxes = []
    for sl, i in zip(ndi.find_objects(lab), range(1, n + 1)):
        ys, xs = sl
        h = ys.stop - ys.start
        area = int((lab[sl] == i).sum())
        if area < min_area:
            continue
        boxes.append((xs.start, ys.start, xs.stop, ys.stop, h))
    if not boxes:
        return []

    tall = max(b[4] for b in boxes)
    boxes = [b for b in boxes if b[4] >= tall * min_h_frac]

    # Keep one text line. A generous crop catches the caption above the number
    # ("You got:") and specks of scenery; the digits of one number all share a
    # vertical centre, so group by that and keep the biggest group.
    if len(boxes) > 1:
        centres = np.array([(b[1] + b[3]) / 2 for b in boxes])
        best, best_n = None, 0
        for c in centres:
            near = np.abs(centres - c) <= tall * 0.4
            if near.sum() > best_n:
                best, best_n = near, int(near.sum())
        boxes = [b for b, k in zip(boxes, best) if k]

    boxes.sort(key=lambda b: b[0])
    boxes = [(b[0], b[1], b[2], b[3]) for b in boxes]

    # Adjacent digits sometimes touch and come back as one component -- "99"
    # merged in the very first test, so a six-digit number read as five glyphs.
    # A merged glyph is a multiple of the usual width, so split it evenly.
    if len(boxes) > 1:
        widths = sorted(b[2] - b[0] for b in boxes)
        med = widths[len(widths) // 2]
        split: list = []
        for x1, y1, x2, y2 in boxes:
            k = int(round((x2 - x1) / max(med, 1)))
            if k <= 1:
                split.append((x1, y1, x2, y2))
                continue
            step = (x2 - x1) / k
            split += [(int(x1 + i * step), y1, int(x1 + (i + 1) * step), y2)
                      for i in range(k)]
        boxes = split
    return boxes


def glyph_image(rgb: np.ndarray, box: tuple) -> np.ndarray:
    """Binary glyph, scaled to GLYPH_H with its aspect preserved."""
    x1, y1, x2, y2 = box
    m = white_mask(rgb[y1:y2, x1:x2]).astype(np.float32)
    h, w = m.shape
    if h == 0 or w == 0:
        return np.zeros((GLYPH_H, 1), dtype=np.float32)
    tw = max(int(round(w * GLYPH_H / h)), 1)
    img = Image.fromarray((m * 255).astype(np.uint8)).resize((tw, GLYPH_H))
    return np.asarray(img, dtype=np.float32) / 255.0


def text_band(mask: np.ndarray) -> tuple:
    """(top, bottom) of the line holding the number.

    A generous crop usually contains a caption too ("You got:"). Taking every
    white row as one band merged both lines and squashed the digits when
    scaled, so scores never rose above ~0.5. Split into runs of white rows and
    keep the one carrying the most ink.
    """
    rows = mask.any(axis=1)
    bands, start = [], None
    for i, f in enumerate(rows):
        if f and start is None:
            start = i
        elif not f and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(rows)))
    if not bands:
        return (0, mask.shape[0])
    return max(bands, key=lambda b: int(mask[b[0]:b[1]].sum()))


def read_strip(rgb: np.ndarray, templates: dict, min_score: float = 0.72) -> str:
    """Read a number from a strip of (possibly touching) digits.

    Two earlier attempts failed on real frames:

    * splitting connected components -- "708 081" is two blobs, not six glyphs;
    * walking templates greedily left to right -- one narrow mismatch (a '1'
      template is half the width of an '8') consumes part of a digit and every
      later position is shifted, so "3360" came back "33113".

    So parse the whole strip with dynamic programming: choose the sequence of
    templates whose total match score is highest. A wrong narrow match now has
    to pay for the mess it leaves behind, which is exactly the information the
    greedy version threw away.
    """
    m = white_mask(rgb)
    m = ndi.binary_closing(m, structure=np.ones((3, 3)))
    top, bot = text_band(m)
    band = m[top:bot].astype(np.float32)
    if band.shape[0] < 6 or band.shape[1] < 4:
        return ""
    W = max(int(round(band.shape[1] * GLYPH_H / band.shape[0])), 1)
    band = np.asarray(
        Image.fromarray((band * 255).astype(np.uint8)).resize((W, GLYPH_H)),
        dtype=np.float32) / 255.0

    ink = band.sum(axis=0) > 0
    NEG = -1e9
    best = np.full(W + 1, NEG, dtype=np.float64)
    best[0] = 0.0
    parent: dict = {}

    for x in range(W):
        if best[x] <= NEG / 2:
            continue
        # skipping a blank column is free; skipping ink is penalised
        nxt = x + 1
        cost = 0.0 if not ink[x] else -0.35
        if best[x] + cost > best[nxt]:
            best[nxt] = best[x] + cost
            parent[nxt] = (x, None)
        if not ink[x]:
            continue
        for digit, arrs in templates.items():
            for t in arrs:
                tw = t.shape[1]
                end = x + tw
                if end > W + 2:
                    continue
                win = band[:, x:min(end, W)]
                if win.shape[1] < tw:
                    win = np.pad(win, ((0, 0), (0, tw - win.shape[1])))
                score = 1.0 - float(np.abs(win - t).mean())
                if score < min_score:
                    continue
                # width-weighted so a wide correct digit beats two narrow wrong
                # ones covering the same span
                gain = (score - min_score) * tw
                e = min(end, W)
                if best[x] + gain > best[e]:
                    best[e] = best[x] + gain
                    parent[e] = (x, digit)

    if best[W] <= NEG / 2:
        return ""
    out, x = [], W
    while x > 0:
        px, digit = parent.get(x, (x - 1, None))
        if digit is not None:
            out.append(digit)
        x = px
    return "".join(reversed(out))


def normalised_band(rgb: np.ndarray) -> np.ndarray | None:
    """The number's text line, scaled to GLYPH_H, as a binary array."""
    m = white_mask(rgb)
    m = ndi.binary_closing(m, structure=np.ones((3, 3)))
    top, bot = text_band(m)
    band = m[top:bot].astype(np.float32)
    if band.shape[0] < 6 or band.shape[1] < 4:
        return None
    W = max(int(round(band.shape[1] * GLYPH_H / band.shape[0])), 1)
    return np.asarray(
        Image.fromarray((band * 255).astype(np.uint8)).resize((W, GLYPH_H)),
        dtype=np.float32) / 255.0


def align_known(rgb: np.ndarray, text: str, templates: dict,
                width_range: tuple = (8, 32)) -> list | None:
    """Cut a labelled strip into one glyph per character.

    Even-splitting a blob of touching digits gives templates that are close but
    wrong -- enough to make a '5' read back as a '6'. Here the answer is known,
    so the cut points can be *solved for*: find the boundaries that maximise the
    total match of the known digit sequence, using current templates as a
    guide, and fall back to even spacing only where a digit has no template yet.

    Returns a list of (x0, x1) column pairs in the normalised band.
    """
    band = normalised_band(rgb)
    if band is None:
        return None
    W = band.shape[1]
    n = len(text)
    NEG = -1e9
    # dp[i][x] = best score having placed i digits, consuming x columns
    dp = np.full((n + 1, W + 1), NEG, dtype=np.float64)
    back = np.zeros((n + 1, W + 1), dtype=np.int32)
    dp[0, 0] = 0.0
    ink = band.sum(axis=0) > 0

    for i in range(n):
        ch = text[i]
        arrs = templates.get(ch, [])
        for x in range(W):
            if dp[i, x] <= NEG / 2:
                continue
            if not ink[x]:                      # skip gaps between groups
                if dp[i, x] > dp[i, x + 1]:
                    dp[i, x + 1] = dp[i, x]
                    back[i, x + 1] = x
                continue
            lo, hi = width_range
            for wdt in range(lo, min(hi, W - x) + 1):
                win = band[:, x:x + wdt]
                if arrs:
                    score = max(
                        1.0 - float(np.abs(
                            np.asarray(Image.fromarray((win * 255).astype(np.uint8))
                                       .resize((t.shape[1], GLYPH_H)),
                                       dtype=np.float32) / 255.0 - t).mean())
                        for t in arrs)
                else:
                    score = 0.5                 # no template yet: stay neutral
                v = dp[i, x] + score
                if v > dp[i + 1, x + wdt]:
                    dp[i + 1, x + wdt] = v
                    back[i + 1, x + wdt] = x

    end = int(np.argmax(dp[n]))
    if dp[n, end] <= NEG / 2:
        return None
    cuts, x = [], end
    for i in range(n, 0, -1):
        px = int(back[i, x])
        cuts.append((px, x))
        x = px
    return list(reversed(cuts))


def band_glyph(rgb: np.ndarray, span: tuple) -> np.ndarray | None:
    band = normalised_band(rgb)
    if band is None:
        return None
    x0, x1 = span
    if x1 <= x0:
        return None
    return band[:, x0:x1]


def load_templates() -> dict:
    if not TEMPLATE_DIR.exists():
        return {}
    out: dict = {}
    for p in sorted(TEMPLATE_DIR.glob("*.npy")):
        out[p.stem.split("_")[0]] = out.get(p.stem.split("_")[0], [])
        out[p.stem.split("_")[0]].append(np.load(p))
    return out


def save_template(digit: str, arr: np.ndarray, idx: int = 0) -> pathlib.Path:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    p = TEMPLATE_DIR / f"{digit}_{idx:02d}.npy"
    np.save(p, arr)
    return p


def classify(glyph: np.ndarray, templates: dict) -> tuple:
    """(digit, score). Score is 1 - mean absolute difference."""
    best, best_score = None, -1.0
    for digit, arrs in templates.items():
        for t in arrs:
            score = 1.0 - float(np.abs(glyph - t).mean())
            if score > best_score:
                best, best_score = digit, score
    return best, best_score


def read_number(rgb: np.ndarray, templates: dict | None = None,
                min_score: float = 0.72) -> int | None:
    """Read a strip containing one number. Returns None if unreadable."""
    templates = templates if templates is not None else load_templates()
    if not templates:
        return None
    text = read_strip(rgb, templates, min_score)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


# --------------------------------------------------------------- HUD regions
# Fractions of the game viewport. The result screen is a fixed layout, so these
# hold at any window size.
RESULT_ROWS = {
    "gold":   (0.44, 0.44, 0.60, 0.52),
    "elixir": (0.44, 0.52, 0.60, 0.60),
    "dark":   (0.44, 0.60, 0.60, 0.68),
}
BATTLE_DAMAGE = (0.86, 0.86, 1.00, 0.95)


def crop_frac(rgb: np.ndarray, box: tuple) -> np.ndarray:
    h, w = rgb.shape[:2]
    return rgb[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]


def read_result_loot(rgb: np.ndarray, templates: dict | None = None) -> dict:
    """Gold/elixir/dark from a RESULT screen."""
    templates = templates if templates is not None else load_templates()
    return {name: read_number(crop_frac(rgb, box), templates)
            for name, box in RESULT_ROWS.items()}


# ------------------------------------------------- finding the loot numbers
# Fixed fractional boxes do not survive a layout change. The boxes calibrated
# on one capture clipped the digits in half on another -- the crop still looked
# right to the eye, but `text_band` then picked a 21-row slice through the
# middle of 40-row glyphs and "133 905" was read as 7114114.
#
# The resource icons are a far better anchor: they are large, saturated, and
# sit immediately right of their number at every resolution.
RESOURCE_HUES = {
    "gold":   (35.0, 62.0, 0.55, 0.60),     # (hue_lo, hue_hi, sat_min, val_min)
    "elixir": (275.0, 315.0, 0.55, 0.60),
    "dark":   (255.0, 295.0, 0.25, 0.12),   # dark elixir: purple but nearly black
}


def find_loot_rows(rgb: np.ndarray, region: tuple = (0.30, 0.30, 0.95, 0.85)) -> dict:
    """Locate each loot number by finding its resource icon.

    Returns {resource: (x1, y1, x2, y2)} for the strip holding the digits,
    which is to the *left* of the icon and shares its vertical extent. Only
    resources whose icon was actually found appear in the result.
    """
    from ai.perception.buttons import _hsv

    h, w = rgb.shape[:2]
    x0, y0 = int(w * region[0]), int(h * region[1])
    x1, y1 = int(w * region[2]), int(h * region[3])
    sub = rgb[y0:y1, x0:x1]
    hue, sat, val = _hsv(sub)

    out: dict = {}
    for name, (lo, hi, s_min, v_min) in RESOURCE_HUES.items():
        m = (hue >= lo) & (hue <= hi) & (sat >= s_min) & (val >= v_min)
        if name == "dark":
            m &= val <= 0.45                 # dark elixir is a *dark* purple
        if m.sum() < 50:
            continue
        lab, n = ndi.label(m)
        if not n:
            continue
        # the icon is a compact blob, not the largest patch of that hue in the
        # scene -- score by how square and how filled each candidate is
        best, best_score = None, 0.0
        for sl in ndi.find_objects(lab):
            ih, iw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if ih < 12 or iw < 12:
                continue
            fill = float(m[sl].mean())
            aspect = min(ih, iw) / max(ih, iw)
            score = fill * aspect * min(ih, iw)
            if score > best_score:
                best, best_score = sl, score
        if best is None:
            continue
        iy0, iy1 = best[0].start, best[0].stop
        ix0 = best[1].start
        pad = int((iy1 - iy0) * 0.30)        # glyphs are taller than the icon
        strip_w = int((iy1 - iy0) * 5.5)     # room for six digits and a space
        out[name] = (max(x0 + ix0 - strip_w, 0), max(y0 + iy0 - pad, 0),
                     x0 + ix0 - 2, min(y0 + iy1 + pad, h))

    # The loot icons form a vertical stack, so any "icon" far off that column
    # is something else in the scene wearing the same hue -- a purple roof read
    # as dark elixir on a frame that had no dark row at all.
    if len(out) > 1:
        xs = sorted(b[2] for b in out.values())
        median_x = xs[len(xs) // 2]
        tol = w * 0.06
        out = {k: b for k, b in out.items() if abs(b[2] - median_x) <= tol}
    return out


# No single raid yields this much of one resource. A reading above it is a
# misread, not a haul: before the icon anchoring, a clipped crop turned
# "133 905" into 7114114 and the bandit recorded a perfect score for it.
MAX_RAID_LOOT = 2_000_000


def read_loot_by_icon(rgb: np.ndarray, templates: dict | None = None) -> dict:
    """Loot numbers, located by their resource icons rather than by fractions.

    Implausible values come back as None, which callers already treat as
    "unreadable" and refuse to score.
    """
    templates = templates if templates is not None else load_templates()
    out = {}
    for name, b in find_loot_rows(rgb).items():
        n = read_number(rgb[b[1]:b[3], b[0]:b[2]], templates)
        out[name] = n if (n is not None and 0 <= n <= MAX_RAID_LOOT) else None
    return out
