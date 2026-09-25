"""Entity class tables for the fast raid simulator.

Every number here is a *modelling choice*, not a measurement from the target
game. Phase 1 (black-box analysis) replaces these with fitted values; keeping
them in one table is what makes that swap cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Cls(IntEnum):
    """Entity classes. 0 is reserved for padding in observation tensors."""

    PAD = 0
    # --- buildings -------------------------------------------------------
    TOWN_HALL = 1
    STORAGE = 2
    RESOURCE_BUILDING = 3
    WALL = 4
    OBSTACLE = 5
    # --- defenses --------------------------------------------------------
    CANNON = 6
    ARCHER_TOWER = 7
    MORTAR = 8
    WIZARD_TOWER = 9
    AIR_DEFENSE = 10
    TESLA = 11
    INFERNO = 12
    XBOW = 13
    # --- units -----------------------------------------------------------
    TANK = 14
    DPS = 15
    WALL_BREAKER = 16
    RANGED = 17
    AIR = 18
    SPLASH = 19


NUM_CLASSES = len(Cls)

UNIT_CLASSES = [Cls.TANK, Cls.DPS, Cls.WALL_BREAKER, Cls.RANGED, Cls.AIR, Cls.SPLASH]
NUM_ARCHETYPES = len(UNIT_CLASSES)

DEFENSE_CLASSES = [
    Cls.CANNON,
    Cls.ARCHER_TOWER,
    Cls.MORTAR,
    Cls.WIZARD_TOWER,
    Cls.AIR_DEFENSE,
    Cls.TESLA,
    Cls.INFERNO,
    Cls.XBOW,
]

# Buildings that count toward destruction %. Walls and obstacles deliberately
# excluded -- see docs/REWARD_SPEC.md section 5 (anti reward-hacking).
SCORING_CLASSES = {Cls.TOWN_HALL, Cls.STORAGE, Cls.RESOURCE_BUILDING} | set(DEFENSE_CLASSES)


@dataclass(frozen=True)
class BuildingStats:
    hp: float
    radius: float          # footprint radius in tiles
    loot: float = 0.0      # loot units released on destruction
    dps: float = 0.0
    rng: float = 0.0       # attack range in tiles
    min_rng: float = 0.0   # mortars cannot hit adjacent units
    hits_air: bool = False
    hits_ground: bool = True
    splash: float = 0.0    # splash radius, 0 = single target
    ramp: float = 0.0      # dps multiplier gained per second on the same target


# Balance target (measured by scripts/balance_report.py):
#   random agent  L1 ~30% win, L5 ~0%
#   heuristic     L1 ~85% win, L5 ~15%
# If these drift, retune here rather than in the reward.
BUILDINGS: dict[Cls, BuildingStats] = {
    Cls.TOWN_HALL:         BuildingStats(hp=3500, radius=2.0, loot=200.0),
    Cls.STORAGE:           BuildingStats(hp=1800, radius=1.5, loot=1000.0),
    Cls.RESOURCE_BUILDING: BuildingStats(hp=900,  radius=1.5, loot=250.0),
    Cls.WALL:              BuildingStats(hp=900,  radius=0.5),
    Cls.OBSTACLE:          BuildingStats(hp=200,  radius=0.5),

    Cls.CANNON:       BuildingStats(hp=1200, radius=1.5, dps=110, rng=9.0),
    Cls.ARCHER_TOWER: BuildingStats(hp=1000, radius=1.5, dps=90,  rng=10.0, hits_air=True),
    Cls.MORTAR:       BuildingStats(hp=900,  radius=1.5, dps=70,  rng=11.0, min_rng=4.0, splash=1.5),
    Cls.WIZARD_TOWER: BuildingStats(hp=1100, radius=1.5, dps=100, rng=7.0,  hits_air=True, splash=1.2),
    Cls.AIR_DEFENSE:  BuildingStats(hp=1300, radius=1.5, dps=260, rng=10.0, hits_air=True, hits_ground=False),
    Cls.TESLA:        BuildingStats(hp=900,  radius=1.0, dps=150, rng=7.0,  hits_air=True),
    Cls.INFERNO:      BuildingStats(hp=1600, radius=1.5, dps=70,  rng=9.0,  hits_air=True, ramp=0.9),
    Cls.XBOW:         BuildingStats(hp=1800, radius=1.5, dps=170, rng=11.0),
}


@dataclass(frozen=True)
class UnitStats:
    hp: float
    dps: float
    speed: float        # tiles / second
    rng: float          # attack range in tiles
    is_air: bool = False
    prefers_walls: bool = False
    wall_mult: float = 1.0   # damage multiplier against walls
    splash: float = 0.0
    housing: int = 1


UNITS: dict[Cls, UnitStats] = {
    Cls.TANK:         UnitStats(hp=3000, dps=45,  speed=1.2, rng=1.0, housing=5),
    Cls.DPS:          UnitStats(hp=420,  dps=95,  speed=2.4, rng=1.0, housing=1),
    Cls.WALL_BREAKER: UnitStats(hp=120,  dps=40,  speed=3.2, rng=1.0, prefers_walls=True, wall_mult=40.0, housing=2),
    Cls.RANGED:       UnitStats(hp=300,  dps=60,  speed=1.8, rng=4.5, housing=2),
    Cls.AIR:          UnitStats(hp=900,  dps=70,  speed=2.6, rng=2.5, is_air=True, housing=4),
    Cls.SPLASH:       UnitStats(hp=700,  dps=50,  speed=1.6, rng=3.0, splash=1.5, housing=4),
}

# Fastest-lookup helper: maps a class id to its "generic" range/dps so the
# DETECTED observation mode can fill token slots 7-8 from a class prior alone.
CLASS_PRIOR_RNG = {int(c): (BUILDINGS[c].rng if c in BUILDINGS else UNITS[c].rng) for c in list(BUILDINGS) + list(UNITS)}
CLASS_PRIOR_DPS = {int(c): (BUILDINGS[c].dps if c in BUILDINGS else UNITS[c].dps) for c in list(BUILDINGS) + list(UNITS)}

MAX_RNG = 12.0
MAX_DPS = 300.0


# --------------------------------------------------------------- real stats
# `scripts/import_coc_data.py` writes simulator/game_data.py from the real
# game's published numbers. Turning this on replaces every table above.
#
# It is off by default on purpose. The real numbers are on a completely
# different scale (a max wall has 9 000 hp, a barbarian 230; armies are sized
# by housing space, not unit count), so switching them in without also
# resizing the army and re-running scripts/baselines.py produces a game no
# agent can win. Flip it, then re-tune, then update the balance targets above.
USE_REAL_STATS = False

if USE_REAL_STATS:  # pragma: no cover - opt-in path
    try:
        from . import game_data as _gd

        for _name, _row in _gd.BUILDINGS.items():
            _c = Cls[_name]
            _cur = BUILDINGS[_c]
            BUILDINGS[_c] = BuildingStats(
                hp=_row["hp"], radius=_row["radius"], loot=_cur.loot,
                dps=_row["dps"], rng=_row["rng"], min_rng=_row["min_rng"],
                hits_air=_row["hits_air"], hits_ground=_row["hits_ground"],
                splash=_row["splash"], ramp=_cur.ramp)
        for _name, _row in _gd.UNITS.items():
            _c = Cls[_name]
            _cur = UNITS[_c]
            UNITS[_c] = UnitStats(
                hp=_row["hp"], dps=_row["dps"], speed=_row["speed"],
                rng=_row["rng"], is_air=_row["is_air"],
                prefers_walls=_cur.prefers_walls, wall_mult=_cur.wall_mult,
                splash=_row["splash"], housing=_row["housing"])
        MAX_DPS = max(s.dps for s in BUILDINGS.values()) or MAX_DPS
        MAX_RNG = max(s.rng for s in BUILDINGS.values()) or MAX_RNG
    except ImportError:
        raise SystemExit(
            "USE_REAL_STATS is on but simulator/game_data.py is missing; run "
            "python -m scripts.import_coc_data --town-hall 13 --write")
