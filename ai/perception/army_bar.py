"""Read the army bar: how many units each card holds, and which are heroes.

The live agent used to tap every card a fixed number of times. That is wrong in
both directions at once -- with 4 balloons, 8 electro dragons and 3 heroes, a
fixed 3 taps under-deploys the balloons, strands 5 dragons, and taps each hero
card three times when it holds exactly one hero.

Over-tapping a hero card is not harmless. Once the hero is down its card turns
into the *ability* button, so the extra taps fire the ability immediately, at
the drop point, with nothing to rage or heal. The old comment in `play_live`
claimed over-clicking was safe because exhausted cards ignore clicks; that is
true for troops and false for heroes.

What the bar looks like:

    +-----------+     count badge, top-right, white "xN" on a dark outline
    |       x4  |     -- troops and spells have one, heroes never do
    |  artwork  |
    | [8]     1 |     level badge bottom-left, hotkey bottom-right
    +-----------+

So "has a count badge" is also the hero test, and it needs no OCR to answer.
"""
from __future__ import annotations

import numpy as np

from ai.perception.digits import load_templates, read_number, white_mask

# Badge box as a fraction of the card. The right edge runs slightly *past* the
# card box because `find_army_slots` insets each box by 8% of the pitch, which
# clips the badge on the last card.
BADGE = (0.34, 0.02, 1.10, 0.24)

# A badge is dense white text; hero artwork bleeds through as thin highlights.
# Raw ink separates them only barely (hero 0.098 vs troop 0.128 on the same
# bar), but ink *density inside the text band* separates them cleanly:
#
#     heroes  0.02  0.06  0.13
#     troops  0.19  0.21  0.22  0.23  0.24  0.26
#
# Measured on one labelled 9-card bar, so treat it as a good default rather
# than a validated constant -- `--army` overrides it for live play.
FILL_MIN = 0.16
INK_MIN = 0.03

# No army card holds more than this. A parse above it is a misread, not a
# reading -- "5" came back as 116 before badge-specific templates existed.
MAX_COUNT = 60


def badge_crop(rgb: np.ndarray, slot: tuple) -> np.ndarray:
    x1, y1, x2, y2 = slot
    cw, ch = x2 - x1, y2 - y1
    h, w = rgb.shape[:2]
    return rgb[y1 + int(ch * BADGE[1]): y1 + int(ch * BADGE[3]),
               max(x1 + int(cw * BADGE[0]), 0): min(x1 + int(cw * BADGE[2]), w)]


def _glyph_runs(mask: np.ndarray) -> list:
    """Split a text mask into glyphs at blank columns.

    `digits.segment` merges these: in "x4" the x and the 4 are 10 and 11 pixels
    wide with a single empty column between them, which its connected-component
    pass treats as one blob. Splitting on the blank column directly is both
    simpler and exactly right for a two-or-three glyph badge.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return []
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    cols = mask[r0:r1].any(axis=0)
    runs, start = [], None
    for i, on in enumerate(cols):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, r0, i, r1))
            start = None
    if start is not None:
        runs.append((start, r0, len(cols), r1))
    return [b for b in runs if b[2] - b[0] >= 3]


def read_badge(rgb: np.ndarray, slot: tuple,
               templates: dict | None = None) -> tuple:
    """(has_badge, count). Count is None when a badge is there but unreadable.

    These must stay separate. "No badge" means hero, which means exactly one
    tap; "badge I could not read" means an unknown number of troops. Collapsing
    the second into the first is how a misread turns into taps on a hero card,
    which fires its ability -- the exact failure this module exists to prevent.

    The leading glyph is the "x". Rather than give a digit classifier a
    non-digit template, the crop is advanced past it and the number is read by
    the same dynamic-programming parser used for the HUD.
    """
    templates = templates if templates is not None else load_templates()
    crop = badge_crop(rgb, slot)
    if crop.size == 0:
        return (False, None)
    mask = white_mask(crop)
    if float(mask.mean()) < INK_MIN:
        return (False, None)              # hero: artwork bleed only
    band = mask[mask.any(axis=1)]
    if band.size == 0 or float(band.mean()) < FILL_MIN:
        return (False, None)

    glyphs = _glyph_runs(mask)
    if not glyphs:
        return (False, None)
    if len(glyphs) >= 2:
        x_start = glyphs[1][0]
    else:
        # x and digit ran together with no blank column; the x is a fixed-width
        # prefix taking a little under half of a two-glyph blob
        gx0, _, gx1, _ = glyphs[0]
        if gx1 - gx0 < 12:
            return (True, None)
        x_start = gx0 + (gx1 - gx0) * 45 // 100
    x_end = glyphs[-1][2]
    n = read_number(crop[:, max(x_start - 1, 0):x_end + 1], templates)
    return (True, n if n and 0 < n <= MAX_COUNT else None)


def read_army(rgb: np.ndarray, slots: list, templates: dict | None = None) -> list:
    """One dict per card: {"count", "hero", "readable"}.

    A hero card carries no badge, which means exactly one unit and exactly one
    tap -- that part needs no OCR and is reliable. A troop card whose number
    could not be read is returned with `readable=False` and a count of None;
    the caller decides what to do, and `plan_taps` refuses to guess.
    """
    templates = templates if templates is not None else load_templates()
    out = []
    for s in slots:
        has_badge, n = read_badge(rgb, s, templates)
        if not has_badge:
            out.append({"count": 1, "hero": True, "spell": False, "readable": True})
        else:
            # a badge alone cannot tell a spell from a troop -- both carry one.
            # Only --army can, so the reader never marks a card as a spell.
            out.append({"count": n, "hero": False, "spell": False,
                        "readable": n is not None})
    return out


def describe(army: list, keys: list | None = None) -> str:
    parts = []
    for i, a in enumerate(army):
        k = keys[i] if keys and i < len(keys) else str(i)
        if a["hero"]:
            parts.append(f"{k}:hero")
        elif not a["readable"]:
            parts.append(f"{k}:x?")
        else:
            parts.append(f"{k}:{'spell x' if a.get('spell') else 'x'}{a['count']}")
    return " ".join(parts)


def parse_army_spec(spec: str, n_cards: int | None = None) -> list:
    """Parse `--army "4,8,H,H,H"` into the same shape `read_army` returns.

    Explicit counts are the authoritative input for live play. Badge OCR is
    good enough to cross-check with and not good enough to fire hero abilities
    on: the hero test is calibrated on a single labelled bar, and a hero card
    misread as troops gets tapped repeatedly, which triggers its ability at the
    drop point.
    """
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        low = tok.lower()
        if low in ("h", "hero"):
            out.append({"count": 1, "hero": True, "spell": False, "readable": True})
            continue
        spell = low.startswith("s")
        num = low[1:] if spell else low
        if not num.isdigit() or not 0 < int(num) <= MAX_COUNT:
            raise ValueError(
                f"bad army entry {tok!r}: expected a count 1-{MAX_COUNT}, "
                f"'H' for a hero, or 'S<n>' for n spells")
        out.append({"count": int(num), "hero": False, "spell": spell,
                    "readable": True})
    if n_cards is not None and len(out) != n_cards:
        raise ValueError(f"--army lists {len(out)} cards but {n_cards} keys are in use")
    return out


def plan_taps(army: list) -> list:
    """Taps per card. Refuses to guess an unreadable count.

    Returning 0 for an unreadable card loses those units for the raid, which is
    the cheap failure. The expensive one is guessing high on a card that turns
    out to be a hero.
    """
    return [0 if not a["readable"] else int(a["count"]) for a in army]
