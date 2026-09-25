"""Turn a pattern, an army and a set of learned choices into a timed attack.

The old loop tapped every card as fast as it could. That makes two decisions
unrepresentable:

* **when the heroes go in.** Sending heroes with the first wave is a different
  attack from sending them once the troops have opened a compartment, and the
  bar cannot express the difference if everything lands at t=0.
* **where the spells go.** A spell is worth something only on top of the
  fighting, and the fighting is not where the troops landed -- it is wherever
  they have walked to by the time the spell comes down.

So deployment becomes a schedule: (time, hotkey, point) steps, executed in
order with real waits between them.

## The model of the game

There is no object detector, so the agent cannot see its own troops. It does
not have to guess blindly either, because troop movement in this game is
almost entirely predictable: units land on the deploy ring, acquire the
nearest building and walk inward. So the fighting at time t is on the line
from where the army landed toward the middle of the base, and how far along
depends on how long you waited.

`advance()` is that model. How far to lead is not hard-coded: it is paired
with the delay into a few physically coherent tactics ("cast early and
shallow", "cast late and deep") and the bandit finds which pays. Learning the
lead separately from the delay would let it pick combinations that aim at
empty ground.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai.policies import deploy_patterns as dp

# ---------------------------------------------------------------- decisions
# Each is one dimension of the factored bandit. Arms are named tactics, not
# raw numbers, so the learned table reads as advice a person could follow.
HERO_TIMING = {
    "with_troops": 0.0,     # heroes tank from the start
    "after_funnel": 6.0,    # let the troops open a compartment first
    "late": 14.0,           # heroes clean up behind a committed army
}
HERO_SPOT = {
    "same": (0.0, 1.00),    # (angle offset, radius scale) on the deploy ring
    "ahead": (0.0, 0.92),   # slightly deeper, so heroes lead the push
    "flank": (0.55, 1.00),  # off to one side of the troops
}
# (delay after the troops land, how far along the attack axis to aim)
SPELL_TIMING = {
    "early_shallow": (4.0, 0.20),
    "mid": (8.0, 0.42),
    "late_deep": (14.0, 0.62),
}

DIMENSIONS = {
    "pattern": list(dp.PATTERNS),
    "hero_timing": list(HERO_TIMING),
    "hero_spot": list(HERO_SPOT),
    "spell_timing": list(SPELL_TIMING),
}

# How long the troop taps are spread over. Not zero: a single instant burst is
# both unlike a human and pointless, since the units still walk in together.
TROOP_WINDOW = 2.5

# Floor on the gap between two taps, matching `play_live.TAP_DELAY`. A big army
# cannot be squeezed into TROOP_WINDOW -- 30 troops in 2.5s is 83ms apart --
# so the window stretches instead. Left fixed, every later step would inherit
# the slippage and the hero and spell times would drift out of the plan.
#
# 0.35s, not something faster: the game silently drops taps that arrive too
# close together. A raid at 0.12s spacing deployed 2 of 4 wall breakers and 3
# of 8 dragons, while its 2s-spaced spells all landed.
MIN_TAP_GAP = 0.35


def troop_window(n_troop: int) -> float:
    return max(TROOP_WINDOW, n_troop * MIN_TAP_GAP)


@dataclass
class Step:
    t: float                 # seconds from the start of the attack
    key: str                 # army hotkey to press
    point: tuple             # pixel coordinates to tap
    label: str

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return f"Step(t={self.t:.1f}s {self.key} {self.point} {self.label})"


def advance(point: tuple, frac: float, w: int, h: int) -> tuple:
    """Where a unit that landed at `point` has walked to, `frac` of the way in.

    The model of the game: units acquire the nearest building and head for the
    middle, so the fighting moves along the line from the drop point toward the
    centre of the base.
    """
    cx, cy = dp.CENTRE[0] * w, dp.CENTRE[1] * h
    x = point[0] + (cx - point[0]) * frac
    y = point[1] + (cy - point[1]) * frac
    return (int(x), int(min(y, h * 0.83)))


def hero_point(plan: dp.Plan, w: int, h: int, spot: str,
               rng: np.random.Generator | None = None) -> tuple:
    rng = rng or np.random.default_rng()
    d_angle, scale = HERO_SPOT.get(spot, HERO_SPOT["same"])
    shifted = dp.Plan(plan.name, plan.angle + d_angle, plan.spread, plan.jitter)
    fx, fy = dp._ellipse(shifted.angle, scale)
    x = min(max(fx + float(rng.normal(0, 0.012)), 0.04), 0.96)
    y = min(max(fy + float(rng.normal(0, 0.012)), 0.05), 0.83)
    return (int(x * w), int(y * h))


def build_schedule(plan: dp.Plan, w: int, h: int, army: list, keys: list,
                   choices: dict, rng: np.random.Generator | None = None) -> list:
    """The full attack, as timed steps.

    `army[i]` is {"count", "hero", "spell", "readable"} for the card on
    `keys[i]`. Cards whose count could not be read are skipped rather than
    guessed -- see `ai.perception.army_bar`.
    """
    rng = rng or np.random.default_rng()
    taps = [0 if not a.get("readable", True) else int(a.get("count") or 0)
            for a in army]
    is_hero = [bool(a.get("hero")) for a in army]
    is_spell = [bool(a.get("spell")) for a in army]

    # troops only: heroes get one shared point, spells are aimed at the fight
    troop_taps = [0 if (hero or spell) else n
                  for n, hero, spell in zip(taps, is_hero, is_spell)]
    drops = dp.army_points(plan, w, h, troop_taps, [False] * len(taps), rng)

    steps: list = []
    n_troop = sum(troop_taps) or 1
    window = troop_window(n_troop)
    i = 0
    for key, pts in zip(keys, drops):
        for pt in pts:
            steps.append(Step(window * i / n_troop, key, pt, "troop"))
            i += 1

    troop_pts = [s.point for s in steps]
    centroid = (int(np.mean([p[0] for p in troop_pts])),
                int(np.mean([p[1] for p in troop_pts]))) if troop_pts else \
        hero_point(plan, w, h, "same", rng)

    t_hero = HERO_TIMING.get(choices.get("hero_timing", "with_troops"), 0.0)
    hp = hero_point(plan, w, h, choices.get("hero_spot", "same"), rng)
    dropped = 0
    for key, n, hero in zip(keys, taps, is_hero):
        if hero and n:
            # stagger by hero ordinal, not card index: offsetting by position
            # in the bar meant "with_troops" put the first hero in at 0.8s
            # purely because it sat in the third slot
            steps.append(Step(t_hero + 0.4 * dropped, key, hp, "hero"))
            dropped += 1

    delay, lead = SPELL_TIMING.get(choices.get("spell_timing", "mid"),
                                   SPELL_TIMING["mid"])
    cast = 0
    for key, n, spell in zip(keys, taps, is_spell):
        if not (spell and n):
            continue
        for _ in range(n):
            # stagger across *all* spells, not per card: two different spell
            # cards each starting their own count landed together on one tile.
            # Each later cast aims slightly deeper, following the fight in.
            t = window + delay + 2.0 * cast
            frac = min(lead + 0.06 * cast, 0.85)
            steps.append(Step(t, key, advance(centroid, frac, w, h), "spell"))
            cast += 1

    steps.sort(key=lambda s: s.t)
    return steps


def describe(steps: list) -> str:
    by = {}
    for s in steps:
        by.setdefault(s.label, []).append(s)
    parts = []
    for label in ("troop", "hero", "spell"):
        g = by.get(label)
        if g:
            parts.append(f"{label} x{len(g)} @ {g[0].t:.1f}-{g[-1].t:.1f}s")
    return "; ".join(parts)
