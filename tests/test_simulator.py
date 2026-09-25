import numpy as np
import pytest

from simulator.core import Army, BaseLayout, RaidSim, SimConfig
from simulator.entities import Cls
from simulator.generator import BaseGenerator, LEVELS, describe, split_seeds


def make_base(pairs):
    cls = np.array([int(c) for c, _, _ in pairs], dtype=np.int32)
    x = np.array([px for _, px, _ in pairs], dtype=np.float32)
    y = np.array([py for _, _, py in pairs], dtype=np.float32)
    return BaseLayout(cls=cls, x=x, y=y)


def test_unit_kills_undefended_building():
    base = make_base([(Cls.TOWN_HALL, 22, 22)])
    sim = RaidSim(base, Army({Cls.DPS: 1}))
    assert sim.deploy(Cls.DPS, 15, 22)
    for _ in range(1200):
        sim.tick()
        if sim.done:
            break
    assert sim.town_hall_down
    assert sim.destruction == 1.0
    assert sim.win


def test_defense_kills_unit():
    # one xbow, one fragile unit, nothing else: the unit must die
    base = make_base([(Cls.TOWN_HALL, 40, 40), (Cls.XBOW, 22, 22)])
    sim = RaidSim(base, Army({Cls.WALL_BREAKER: 1}))
    sim.deploy(Cls.WALL_BREAKER, 20, 22)
    for _ in range(300):
        sim.tick()
    assert sim.alive_unit_count() == 0


def test_air_unit_ignores_air_immune_defense():
    base = make_base([(Cls.TOWN_HALL, 22, 22), (Cls.CANNON, 22, 26)])
    sim = RaidSim(base, Army({Cls.AIR: 1}))
    sim.deploy(Cls.AIR, 22, 14)
    for _ in range(900):
        sim.tick()
        if sim.done:
            break
    # cannon cannot hit air, so the air unit must survive and win
    assert sim.alive_unit_count() == 1
    assert sim.win


def test_air_defense_cannot_hit_ground():
    base = make_base([(Cls.TOWN_HALL, 30, 22), (Cls.AIR_DEFENSE, 22, 22)])
    sim = RaidSim(base, Army({Cls.TANK: 1}))
    sim.deploy(Cls.TANK, 18, 22)
    for _ in range(400):
        sim.tick()
    assert sim.alive_unit_count() == 1


def test_walls_do_not_count_toward_destruction():
    base = make_base([(Cls.TOWN_HALL, 22, 22), (Cls.WALL, 20, 22), (Cls.WALL, 20, 23)])
    sim = RaidSim(base, Army({Cls.DPS: 1}))
    assert sim.n_scoring == 1
    sim.deploy(Cls.DPS, 15, 22)
    for _ in range(1500):
        sim.tick()
        if sim.done:
            break
    assert sim.destruction == 1.0   # walls destroyed or not, only the TH counts


def test_wall_blocks_ground_unit():
    """A ground unit behind a wall must attack the wall, not walk through it."""
    wall = [(Cls.WALL, 18.0, 22.0 + dy) for dy in np.arange(-3, 3, 1.0)]
    base = make_base([(Cls.TOWN_HALL, 22, 22), *wall])
    sim = RaidSim(base, Army({Cls.DPS: 1}))
    sim.deploy(Cls.DPS, 14, 22)
    for _ in range(60):
        sim.tick()
    tgt = int(sim.u_target[0])
    assert sim.b_iswall[tgt], "unit should be hitting the wall, not the town hall"
    assert sim.u_x[0] < 18.0, "unit must not have passed through the wall"


def test_wall_breaker_prefers_walls():
    base = make_base([(Cls.TOWN_HALL, 22, 22), (Cls.WALL, 26, 22)])
    sim = RaidSim(base, Army({Cls.WALL_BREAKER: 1}))
    sim.deploy(Cls.WALL_BREAKER, 30, 22)
    sim.tick()
    assert sim.b_iswall[int(sim.u_target[0])]


def test_deploy_cooldown_and_army_limit():
    base = make_base([(Cls.TOWN_HALL, 22, 22)])
    sim = RaidSim(base, Army({Cls.DPS: 2}), SimConfig(deploy_cooldown=1.0))
    assert sim.deploy(Cls.DPS, 10, 10)
    assert not sim.deploy(Cls.DPS, 10, 10), "cooldown must block a second deploy"
    for _ in range(11):
        sim.tick()
    assert sim.deploy(Cls.DPS, 10, 10)
    for _ in range(11):
        sim.tick()
    assert not sim.deploy(Cls.DPS, 10, 10), "army is empty"


def test_loot_accumulates_only_on_destruction():
    base = make_base([(Cls.STORAGE, 22, 22)])
    sim = RaidSim(base, Army({Cls.DPS: 3}))
    sim.deploy(Cls.DPS, 18, 22)
    sim.tick()
    assert sim.loot == 0.0
    for _ in range(600):
        sim.tick()
        if sim.done:
            break
    assert sim.loot_frac == pytest.approx(1.0, rel=1e-3)


def test_battle_times_out():
    base = make_base([(Cls.TOWN_HALL, 22, 22), (Cls.XBOW, 22, 24)])
    sim = RaidSim(base, Army({}), SimConfig(t_max=5.0))
    for _ in range(60):
        sim.tick()
    assert sim.done and not sim.win


@pytest.mark.parametrize("level", LEVELS)
def test_generator_is_deterministic_and_valid(level):
    g = BaseGenerator()
    a, b = g.generate(level, seed=42), g.generate(level, seed=42)
    assert np.array_equal(a.cls, b.cls) and np.allclose(a.x, b.x)
    c = g.generate(level, seed=43)
    assert not (len(a) == len(c) and np.array_equal(a.cls, c.cls) and np.allclose(a.x, c.x))
    assert (a.cls == int(Cls.TOWN_HALL)).sum() == 1
    assert (a.x >= 0).all() and (a.x <= 44).all()
    assert describe(a)["n_defenses"] >= 1


def test_difficulty_is_monotonic_in_defenses():
    g = BaseGenerator()
    counts = [describe(g.generate(lv, seed=5))["n_defenses"] for lv in LEVELS]
    assert counts == sorted(counts)


def test_train_and_eval_seeds_are_disjoint():
    (t0, t1), (e0, e1) = split_seeds()
    assert t1 <= e0, "eval seeds must never overlap training seeds"
