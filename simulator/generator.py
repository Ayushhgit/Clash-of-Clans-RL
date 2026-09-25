"""Procedural base generation (Phase 11).

Five difficulty levels matching the curriculum in docs/ROADMAP.md. Every base
is a pure function of (level, seed), so train/test splits are just disjoint
seed ranges -- see `split_seeds`. That is what stops the policy memorising
layouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import BaseLayout
from .entities import Cls

LEVELS = ["L1", "L2", "L3", "L4", "L5"]


@dataclass(frozen=True)
class LevelSpec:
    n_defenses: tuple            # (min, max)
    n_storages: tuple
    n_resources: tuple
    n_obstacles: tuple
    compartments: int            # grid subdivisions of the core
    wall_gap_prob: float         # chance a wall tile is left out (a "gap")
    core_frac: float             # core box size as a fraction of the map
    defense_pool: tuple
    adversarial: bool = False


SPECS = {
    "L1": LevelSpec((2, 2), (1, 2), (2, 4), (0, 2), 1, 0.35, 0.28,
                    (Cls.CANNON, Cls.ARCHER_TOWER)),
    "L2": LevelSpec((4, 6), (2, 3), (4, 7), (2, 5), 2, 0.25, 0.36,
                    (Cls.CANNON, Cls.ARCHER_TOWER, Cls.MORTAR)),
    "L3": LevelSpec((7, 10), (3, 4), (6, 10), (4, 8), 3, 0.18, 0.44,
                    (Cls.CANNON, Cls.ARCHER_TOWER, Cls.MORTAR, Cls.WIZARD_TOWER,
                     Cls.AIR_DEFENSE)),
    "L4": LevelSpec((11, 16), (4, 6), (8, 14), (6, 12), 4, 0.12, 0.52,
                    (Cls.CANNON, Cls.ARCHER_TOWER, Cls.MORTAR, Cls.WIZARD_TOWER,
                     Cls.AIR_DEFENSE, Cls.TESLA, Cls.XBOW)),
    "L5": LevelSpec((14, 20), (5, 7), (8, 14), (8, 16), 4, 0.08, 0.58,
                    (Cls.MORTAR, Cls.WIZARD_TOWER, Cls.AIR_DEFENSE, Cls.TESLA,
                     Cls.XBOW, Cls.INFERNO), adversarial=True),
}


def split_seeds(n_train: int = 200_000, n_eval: int = 2_000) -> tuple:
    """Disjoint seed ranges. Never evaluate on a training seed."""
    train = (0, n_train)
    evaluation = (10_000_000, 10_000_000 + n_eval)
    return train, evaluation


class BaseGenerator:
    def __init__(self, map_size: float = 44.0):
        self.map_size = map_size

    # ------------------------------------------------------------------ api
    def generate(self, level: str = "L1", seed: int | None = None) -> BaseLayout:
        spec = SPECS[level]
        rng = np.random.default_rng(seed)
        m = self.map_size
        core = spec.core_frac * m
        cx = cy = m / 2.0
        lo, hi = cx - core / 2.0, cx + core / 2.0

        cls: list[int] = []
        # Occupancy is kept in preallocated numpy arrays so a placement test is
        # one vectorised distance check instead of a python loop over entities.
        cap = 512
        ox = np.empty(cap, dtype=np.float32)
        oy = np.empty(cap, dtype=np.float32)
        orad = np.empty(cap, dtype=np.float32)
        n = 0

        def place(c: Cls, x: float, y: float, r: float = 1.6) -> bool:
            nonlocal n
            if n >= cap:
                return False
            if n:
                d2 = (x - ox[:n]) ** 2 + (y - oy[:n]) ** 2
                if (d2 < (r + orad[:n]) ** 2).any():
                    return False
            cls.append(int(c))
            ox[n], oy[n], orad[n] = x, y, r
            n += 1
            return True

        def try_place(c: Cls, region: tuple, r: float = 1.6, tries: int = 24) -> bool:
            nonlocal n
            x0, y0, x1, y1 = region
            cx_ = rng.uniform(x0, x1, size=tries).astype(np.float32)
            cy_ = rng.uniform(y0, y1, size=tries).astype(np.float32)
            if n:
                d2 = (cx_[:, None] - ox[None, :n]) ** 2 + (cy_[:, None] - oy[None, :n]) ** 2
                ok = ~(d2 < (r + orad[None, :n]) ** 2).any(axis=1)
                hit = np.nonzero(ok)[0]
                if hit.size == 0:
                    return False
                k = int(hit[0])
            else:
                k = 0
            cls.append(int(c))
            ox[n], oy[n], orad[n] = cx_[k], cy_[k], r
            n += 1
            return True

        # --- town hall ---------------------------------------------------
        if spec.adversarial:
            th_x = rng.uniform(lo + 2, hi - 2)
            th_y = rng.uniform(lo + 2, hi - 2)
        else:
            th_x, th_y = cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2)
        place(Cls.TOWN_HALL, th_x, th_y, 2.4)

        # --- walls: compartment grid over the core -----------------------
        wx, wy = self._walls(spec, rng, lo, hi)
        if wx.size:
            keep = ((wx - ox[:n, None]) ** 2 + (wy - oy[:n, None]) ** 2
                    >= (orad[:n, None] + 0.45) ** 2).all(axis=0)
            wx, wy = wx[keep], wy[keep]
            k = min(wx.size, cap - n)
            ox[n:n + k], oy[n:n + k], orad[n:n + k] = wx[:k], wy[:k], 0.45
            cls.extend([int(Cls.WALL)] * k)
            n += k

        core_region = (lo + 1.0, lo + 1.0, hi - 1.0, hi - 1.0)
        outer_region = (2.0, 2.0, m - 2.0, m - 2.0)

        # --- storages: inside, they are the loot the agent wants ---------
        for _ in range(int(rng.integers(*_incl(spec.n_storages)))):
            try_place(Cls.STORAGE, core_region, 1.8)

        # --- defenses ----------------------------------------------------
        n_def = int(rng.integers(*_incl(spec.n_defenses)))
        for i in range(n_def):
            c = Cls(int(rng.choice([int(d) for d in spec.defense_pool])))
            if spec.adversarial and i < n_def // 2:
                # adversarial: push half the defenses toward the perimeter so
                # they cover the deploy ring instead of hiding in the core
                region = self._ring_region(rng, m, lo, hi)
            else:
                region = core_region
            if not try_place(c, region, 1.8):
                try_place(c, outer_region, 1.8)

        # --- resource buildings and obstacles: filler / pathing noise ----
        for _ in range(int(rng.integers(*_incl(spec.n_resources)))):
            region = core_region if rng.random() < 0.5 else outer_region
            try_place(Cls.RESOURCE_BUILDING, region, 1.7)
        for _ in range(int(rng.integers(*_incl(spec.n_obstacles)))):
            try_place(Cls.OBSTACLE, outer_region, 1.0)

        return BaseLayout(
            cls=np.asarray(cls, dtype=np.int32),
            x=ox[:n].copy(),
            y=oy[:n].copy(),
            seed=-1 if seed is None else int(seed),
            difficulty=level,
        )

    # ---------------------------------------------------------------- walls
    def _walls(self, spec: LevelSpec, rng, lo: float, hi: float):
        """Axis-aligned wall segments forming compartments^2 cells, with gaps.

        Gaps are what make a base attackable: a fully sealed compartment grid
        forces every ground unit to chew through walls, which is both boring
        and unrepresentative of real bases.
        """
        edges = np.linspace(lo, hi, spec.compartments + 1)
        t = np.arange(lo, hi + 1e-6, 1.0, dtype=np.float32)
        ee, tt = np.meshgrid(edges.astype(np.float32), t, indexing="ij")
        vx, vy = ee.ravel(), tt.ravel()          # vertical lines
        hx, hy = tt.ravel(), ee.ravel()          # horizontal lines
        wx = np.concatenate([vx, hx])
        wy = np.concatenate([vy, hy])
        keep = rng.random(wx.size) >= spec.wall_gap_prob
        return wx[keep], wy[keep]

    def _ring_region(self, rng, m: float, lo: float, hi: float) -> tuple:
        """A band just outside the core, where perimeter defenses live."""
        pad = 3.0
        side = int(rng.integers(0, 4))
        if side == 0:
            return (lo - pad, hi, hi + pad, hi + pad)
        if side == 1:
            return (lo - pad, lo - pad, hi + pad, lo)
        if side == 2:
            return (lo - pad, lo - pad, lo, hi + pad)
        return (hi, lo - pad, hi + pad, hi + pad)


def _incl(bounds: tuple) -> tuple:
    """numpy integers() is half-open; make the spec ranges inclusive."""
    return (bounds[0], bounds[1] + 1)


def describe(base: BaseLayout) -> dict:
    """Cheap summary used by the base-selection agent (Phase 17) and logging."""
    counts = {}
    for c in base.cls:
        counts[Cls(int(c)).name] = counts.get(Cls(int(c)).name, 0) + 1
    from .entities import BUILDINGS, DEFENSE_CLASSES

    dset = {int(d) for d in DEFENSE_CLASSES}
    return {
        "n_entities": len(base),
        "n_defenses": int(sum(1 for c in base.cls if int(c) in dset)),
        "n_walls": int((base.cls == int(Cls.WALL)).sum()),
        "total_loot": float(sum(BUILDINGS[Cls(int(c))].loot for c in base.cls)),
        "total_hp": float(sum(BUILDINGS[Cls(int(c))].hp for c in base.cls)),
        "difficulty": base.difficulty,
        "counts": counts,
    }
