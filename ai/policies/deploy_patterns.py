"""Deployment patterns for live attacks.

The first live loop dropped every card on the same evenly-spaced ring. That is
the weakest plan in the game: troops land alone on all four sides, walk into
whatever is nearest, and die piecemeal. Real attacks pick *one* approach and
commit to it.

Each pattern here is a plan a human would recognise, and each is randomised
internally -- which side, how wide, how much jitter -- so two raids never look
the same. One plan is sampled per attack and *all* cards follow it, which is
what makes the attack coherent rather than merely random.

Coordinates are fractions of the battle viewport, converted to pixels by
`points()`. The map is drawn as a diamond, so the usable deploy band is an
ellipse a little inside the frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# The base sits in the middle of the viewport; the deployable ground is the
# ring outside it. cy is above centre because the army bar eats the bottom.
CENTRE = (0.50, 0.46)
RADIUS = (0.34, 0.30)


@dataclass
class Plan:
    """A sampled attack plan: which pattern, and the parameters drawn for it."""

    name: str
    angle: float = 0.0          # main approach direction, radians
    spread: float = 0.6         # how much of the perimeter to use, radians
    jitter: float = 0.03        # positional noise, fraction of the frame
    params: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (f"{self.name}(angle={math.degrees(self.angle):.0f}deg, "
                f"spread={math.degrees(self.spread):.0f}deg)")


def _ellipse(angle: float, scale: float = 1.0) -> tuple:
    return (CENTRE[0] + RADIUS[0] * scale * math.cos(angle),
            CENTRE[1] + RADIUS[1] * scale * math.sin(angle))


# ------------------------------------------------------------------ patterns
def _spread_angles(plan: Plan, n: int) -> list:
    if n == 1:
        return [plan.angle]
    half = plan.spread / 2
    return [plan.angle - half + plan.spread * i / (n - 1) for i in range(n)]


def pattern_side(plan: Plan, n: int) -> list:
    """Everything along one side. The default competent attack."""
    return [_ellipse(a) for a in _spread_angles(plan, n)]


def pattern_spearhead(plan: Plan, n: int) -> list:
    """Tight cluster on one point, a few stragglers behind it.

    Concentration is what gets units through a compartment: they arrive
    together, so defences split damage instead of killing them one at a time.
    """
    pts = []
    for i in range(n):
        a = plan.angle + np.random.uniform(-0.12, 0.12)
        scale = 1.0 + 0.06 * (i % 3)          # slight depth stagger
        pts.append(_ellipse(a, scale))
    return pts


def pattern_two_prong(plan: Plan, n: int) -> list:
    """Two groups on opposite sides: splits defensive attention."""
    a1, a2 = plan.angle, plan.angle + math.pi
    out = []
    for i in range(n):
        base = a1 if i % 2 == 0 else a2
        out.append(_ellipse(base + np.random.uniform(-0.2, 0.2)))
    return out


def pattern_quadrant(plan: Plan, n: int) -> list:
    """Wrap one corner: two adjacent edges, so troops converge inward."""
    return [_ellipse(a) for a in _spread_angles(
        Plan(plan.name, plan.angle, max(plan.spread, 1.4)), n)]


def pattern_line(plan: Plan, n: int) -> list:
    """A wide even line across one face -- what you want for air.

    Spreads splash damage instead of feeding a single splash defence.
    """
    return [_ellipse(a) for a in _spread_angles(
        Plan(plan.name, plan.angle, max(plan.spread, 2.0)), n)]


def pattern_ring(plan: Plan, n: int) -> list:
    """The original: all the way round. Kept as a deliberate baseline."""
    return [_ellipse(2 * math.pi * i / max(n, 1)) for i in range(n)]


PATTERNS = {
    "side": pattern_side,
    "spearhead": pattern_spearhead,
    "two_prong": pattern_two_prong,
    "quadrant": pattern_quadrant,
    "line": pattern_line,
    "ring": pattern_ring,
}

# Sampling weights. `ring` is included only so the old behaviour stays
# measurable against the rest; it is not expected to be good.
WEIGHTS = {"side": 3.0, "spearhead": 3.0, "two_prong": 2.0,
           "quadrant": 2.0, "line": 2.0, "ring": 0.5}


def sample_plan(rng: np.random.Generator | None = None,
                name: str | None = None) -> Plan:
    """Draw a plan for one attack."""
    rng = rng or np.random.default_rng()
    if name in (None, "random"):
        names = list(PATTERNS)
        w = np.array([WEIGHTS[n] for n in names], dtype=np.float64)
        name = str(rng.choice(names, p=w / w.sum()))
    return Plan(
        name=name,
        angle=float(rng.uniform(0, 2 * math.pi)),
        spread=float(rng.uniform(0.35, 1.1)),
        jitter=float(rng.uniform(0.015, 0.045)),
    )


def points(plan: Plan, w: int, h: int, n: int,
           rng: np.random.Generator | None = None) -> list:
    """Pixel deploy points for `n` drops under `plan`."""
    rng = rng or np.random.default_rng()
    raw = PATTERNS[plan.name](plan, n)
    out = []
    for fx, fy in raw:
        jx = float(rng.normal(0, plan.jitter))
        jy = float(rng.normal(0, plan.jitter))
        # never below 0.83: the army bar starts around 0.86 of the viewport and
        # a stray drop there re-selects a card instead of deploying
        x = min(max(fx + jx, 0.04), 0.96)
        y = min(max(fy + jy, 0.05), 0.83)
        out.append((int(x * w), int(y * h)))
    return out


def hero_point(plan: Plan, w: int, h: int,
               rng: np.random.Generator | None = None) -> tuple:
    """The single spot every hero goes to.

    Heroes are one unit each and they tank for the army, so they belong
    together at the head of the attack -- spreading them around the ring sends
    three separate lone units into three separate defences.
    """
    rng = rng or np.random.default_rng()
    fx, fy = _ellipse(plan.angle, 0.98)
    x = min(max(fx + float(rng.normal(0, 0.01)), 0.04), 0.96)
    y = min(max(fy + float(rng.normal(0, 0.01)), 0.05), 0.83)
    return (int(x * w), int(y * h))


def army_points(plan: Plan, w: int, h: int, counts: list, heroes: list | None = None,
                rng: np.random.Generator | None = None) -> list:
    """Drop points for a whole army, one list per card.

    `counts[i]` is how many units card i actually holds; `heroes[i]` marks it
    as a hero card.

    Two bugs this exists to kill. Calling `points()` once per card returned the
    *same* coordinates every time -- the patterns are deterministic given a
    plan, so five cards landed on one shape, and `ring` with four drops is
    literally a square. And spreading N points per card rather than N across
    the army made the shape depend on the card count instead of the army size.

    Here the whole army gets one shape, dealt round-robin so each card's units
    are spread along it rather than bunched at one end.
    """
    rng = rng or np.random.default_rng()
    heroes = heroes or [False] * len(counts)
    n_troop = sum(c for c, hero in zip(counts, heroes) if not hero)
    pool = points(plan, w, h, n_troop, rng) if n_troop else []

    # deal round-robin: card 0 takes pool[0], pool[k], pool[2k]... so every
    # card covers the whole shape and no two cards overlap exactly
    troop_cards = [i for i, hero in enumerate(heroes) if not hero]
    out: list = [[] for _ in counts]
    cursor = 0
    remaining = {i: counts[i] for i in troop_cards}
    while cursor < len(pool) and any(remaining.values()):
        for i in troop_cards:
            if remaining[i] and cursor < len(pool):
                out[i].append(pool[cursor])
                remaining[i] -= 1
                cursor += 1

    hp = hero_point(plan, w, h, rng)
    for i, hero in enumerate(heroes):
        if hero:
            out[i] = [hp] * counts[i]
    return out
