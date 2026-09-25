"""Tests for the live-game stack: digits, deploy patterns, strategy learning.

None of these need the game running -- they use recorded frames and synthetic
rewards, so they run in CI.
"""
import json
import pathlib

import numpy as np
import pytest

from ai.perception.digits import (
    crop_frac,
    load_templates,
    read_number,
    text_band,
    white_mask,
)
from ai.policies import deploy_patterns as dp
from ai.policies.online_strategy import ArmStats, StrategyLearner, loot_reward

FRAMES = pathlib.Path("data/raw/screenshots")


# ------------------------------------------------------------------- digits
def _labelled():
    from scripts.build_digit_templates import EXAMPLES

    return [(pathlib.Path(p), box, text) for p, box, text in EXAMPLES
            if pathlib.Path(p).exists()]


@pytest.mark.skipif(not load_templates(), reason="digit templates not built")
def test_reads_the_labelled_numbers():
    """The reader must reproduce numbers a human read off the same frames."""
    from PIL import Image

    cases = _labelled()
    if not cases:
        pytest.skip("no captured frames")
    templates = load_templates()
    ok = 0
    for path, box, text in cases:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        got = read_number(crop_frac(rgb, box), templates)
        ok += str(got) == text
    # one low-resolution crop is known to misread a digit; everything else must
    # be exact
    assert ok >= len(cases) - 1, f"only {ok}/{len(cases)} numbers read correctly"


def test_text_band_picks_the_number_not_the_caption():
    """A crop containing 'You got:' above the number must yield the number's
    line -- merging both lines squashed the glyphs and broke matching."""
    mask = np.zeros((40, 60), dtype=bool)
    mask[2:8, 5:40] = True            # caption: thin
    mask[15:35, 5:50] = True          # number: tall and heavy
    top, bot = text_band(mask)
    assert (top, bot) == (15, 35)


def test_white_mask_ignores_coloured_ui():
    rgb = np.zeros((10, 30, 3), dtype=np.uint8)
    rgb[:, :10] = (255, 255, 255)     # white digits
    rgb[:, 10:20] = (250, 200, 40)    # gold coin
    rgb[:, 20:] = (30, 90, 40)        # grass
    m = white_mask(rgb)
    assert m[:, :10].all()
    assert not m[:, 10:].any()


# ----------------------------------------------------------------- patterns
@pytest.mark.parametrize("name", list(dp.PATTERNS))
def test_pattern_points_stay_on_the_map(name):
    rng = np.random.default_rng(0)
    plan = dp.sample_plan(rng, name)
    pts = dp.points(plan, 1600, 900, 40, rng)
    assert len(pts) == 40
    for x, y in pts:
        assert 0 <= x < 1600
        # never in the army bar: a drop there re-selects a card
        assert 0 <= y <= 900 * 0.84


def test_patterns_differ_from_each_other():
    rng = np.random.default_rng(1)
    spans = {}
    for name in dp.PATTERNS:
        plan = dp.Plan(name=name, angle=0.0, spread=0.6, jitter=0.0)
        pts = np.array(dp.points(plan, 1000, 1000, 24, rng), dtype=float)
        spans[name] = pts.std(axis=0).sum()
    # a ring spreads much more widely than a spearhead
    assert spans["ring"] > spans["spearhead"] * 2


def test_sampled_plans_vary_between_attacks():
    rng = np.random.default_rng(3)
    plans = [dp.sample_plan(rng) for _ in range(30)]
    assert len({p.name for p in plans}) > 1
    assert len({round(p.angle, 3) for p in plans}) > 25


# ------------------------------------------------------------------ learning
def test_learner_finds_the_better_arm():
    rng = np.random.default_rng(0)
    learner = StrategyLearner(list(dp.PATTERNS))
    true = {"side": 0.62, "spearhead": 0.55, "two_prong": 0.45,
            "quadrant": 0.42, "line": 0.40, "ring": 0.25}
    for _ in range(120):
        arm = learner.select(rng)
        learner.update(arm, float(np.clip(rng.normal(true[arm], 0.12), 0, 1.5)))
    assert learner.best() == "side"
    recent = [h["arm"] for h in learner.history[-30:]]
    assert recent.count("side") >= 20, "should exploit the best arm once found"


def test_learner_explores_before_committing():
    rng = np.random.default_rng(0)
    learner = StrategyLearner(list(dp.PATTERNS))
    for _ in range(30):
        arm = learner.select(rng)
        learner.update(arm, float(rng.uniform(0, 1)))
    tried = {a for a in learner.arms if learner.stats[a].n}
    assert len(tried) >= 4, "a bandit that never explores cannot rank arms"


def test_learner_survives_a_round_trip(tmp_path):
    learner = StrategyLearner(list(dp.PATTERNS))
    learner.update("side", 0.7)
    learner.update("ring", 0.1)
    path = tmp_path / "stats.json"
    learner.save(path)

    back = StrategyLearner.load(list(dp.PATTERNS), path)
    assert back.stats["side"].n == 1
    assert back.stats["side"].mean == pytest.approx(0.7)
    assert back.best() == "side"


def test_reward_is_a_fraction_of_available_loot():
    """Raw loot mostly measures how rich the base was, not how good the attack
    was, so the reward is normalised by what was on offer."""
    gained = {"gold": 400_000, "elixir": 400_000, "dark": 3_000}
    rich = loot_reward(gained, {"gold": 800_000, "elixir": 800_000, "dark": 6_000})
    poor = loot_reward(gained, {"gold": 420_000, "elixir": 420_000, "dark": 3_100})
    assert poor > rich
    assert 0.0 <= rich <= 1.5


def test_reward_falls_back_when_available_loot_unreadable():
    r = loot_reward({"gold": 500_000, "elixir": None, "dark": 0}, None)
    assert r is not None and 0.0 < r <= 1.5


def test_zero_loot_is_zero_reward():
    assert loot_reward({"gold": 0, "elixir": 0, "dark": 0}, {"gold": 100}) == 0.0


def test_arm_stats_track_mean_and_variance():
    s = ArmStats()
    for v in (0.2, 0.4, 0.6):
        s.update(v)
    assert s.n == 3
    assert s.mean == pytest.approx(0.4)
    assert s.var == pytest.approx(0.04)


# --------------------------------------------------------- army bar / deploys
def _bar_frame():
    """A real battle frame: 2 troop cards, 3 heroes, 4 spells."""
    import pathlib

    import numpy as np
    from PIL import Image

    p = pathlib.Path("data/raw/screenshots/coc_01/000000024.png")
    if not p.exists():
        pytest.skip("capture not available")
    return np.asarray(Image.open(p).convert("RGB"))


def test_hero_cards_are_identified_by_having_no_count_badge():
    """The safety-critical half of reading the bar. A hero card becomes the
    *ability* button once the hero is down, so a spare tap fires the ability at
    the drop point -- mistaking a hero for troops is not a small error."""
    from ai.perception.army_bar import read_army
    from ai.perception.buttons import find_army_slots

    rgb = _bar_frame()
    army = read_army(rgb, find_army_slots(rgb))
    assert [a["hero"] for a in army] == [False, False, True, True, True,
                                         False, False, False, False]


def test_unreadable_count_is_not_reported_as_a_hero():
    """"No badge" means one tap; "badge I could not read" means an unknown
    number of troops. Collapsing the two is how a misread becomes taps into a
    hero ability."""
    from ai.perception.army_bar import plan_taps

    army = [{"count": None, "hero": False, "readable": False},
            {"count": 1, "hero": True, "readable": True}]
    assert plan_taps(army) == [0, 1], "an unreadable card must be skipped, not guessed"


def test_army_spec_overrides_and_rejects_nonsense():
    from ai.perception.army_bar import parse_army_spec, plan_taps

    army = parse_army_spec("4,8,H,H,H")
    assert plan_taps(army) == [4, 8, 1, 1, 1]
    assert [a["hero"] for a in army] == [False, False, True, True, True]
    with pytest.raises(ValueError):
        parse_army_spec("4,8,H", n_cards=5)
    with pytest.raises(ValueError):
        parse_army_spec("4,banana")


def test_each_card_taps_exactly_what_it_holds():
    """Regression: a fixed tap count under-deployed the big cards and
    over-tapped the heroes. With 4 balloons, 8 dragons and 3 heroes, three taps
    per card stranded 5 dragons and tapped each hero three times."""
    import numpy as np

    import ai.policies.deploy_patterns as dp
    from ai.perception.army_bar import parse_army_spec, plan_taps

    army = parse_army_spec("4,8,H,H,H")
    taps = plan_taps(army)
    drops = dp.army_points(dp.Plan("side", angle=0.3), 1000, 1000, taps,
                           [a["hero"] for a in army], np.random.default_rng(0))
    assert [len(d) for d in drops] == [4, 8, 1, 1, 1]
    assert sum(len(d) for d in drops) == 15


def test_every_hero_goes_to_the_same_point():
    """Heroes tank for the army. Spreading three of them around the ring sends
    three lone units into three separate defences."""
    import numpy as np

    import ai.policies.deploy_patterns as dp

    drops = dp.army_points(dp.Plan("ring", angle=1.0), 1000, 1000,
                           [6, 1, 1, 1], [False, True, True, True],
                           np.random.default_rng(0))
    heroes = [d[0] for d in drops[1:]]
    assert len(set(heroes)) == 1, f"heroes landed on {len(set(heroes))} spots"


def test_cards_do_not_all_land_on_the_same_shape():
    """Regression: `points()` was called once per card with the same plan, and
    the patterns are deterministic, so every card got identical coordinates --
    and `ring` with four drops is literally a square."""
    import numpy as np

    import ai.policies.deploy_patterns as dp

    plan = dp.Plan("ring", angle=0.4, spread=0.6, jitter=0.0)
    rng = np.random.default_rng(0)
    per_card = [dp.points(plan, 1000, 1000, 4, rng) for _ in range(3)]
    assert per_card[0] == per_card[1] == per_card[2], "precondition: the old bug"

    drops = dp.army_points(plan, 1000, 1000, [4, 4, 4],
                           [False] * 3, np.random.default_rng(0))
    assert drops[0] != drops[1] and drops[1] != drops[2]
    flat = [p for d in drops for p in d]
    assert len(set(flat)) == len(flat), "drop points must not repeat"


# ------------------------------------------------- learning the whole attack
def test_factored_learner_finds_a_good_combination():
    """One raid teaches every dimension at once. The full cross-product is 486
    arms and each arm costs a three-minute raid, so dimensions are learned
    independently and the reward is shared."""
    import numpy as np

    from ai.policies.online_strategy import FactoredLearner

    dims = {"pattern": ["side", "ring"], "hero_timing": ["with_troops", "late"]}
    learner = FactoredLearner(dims=dims)
    rng = np.random.default_rng(0)
    for _ in range(60):
        c = learner.select(rng)
        good = (c["pattern"] == "side" and c["hero_timing"] == "late")
        learner.update(c, (0.8 if good else 0.2) + rng.normal(0, 0.05))
    assert learner.best() == {"pattern": "side", "hero_timing": "late"}


def test_new_dimensions_do_not_reset_what_was_learned(tmp_path):
    from ai.policies.online_strategy import FactoredLearner

    path = tmp_path / "attack.json"
    a = FactoredLearner(dims={"pattern": ["side", "ring"]})
    for _ in range(5):
        a.update({"pattern": "side"}, 0.9)
    a.save(path)

    b = FactoredLearner.load({"pattern": ["side", "ring"],
                              "spell_timing": ["mid", "late_deep"]}, path)
    assert b.stats["pattern"]["side"].n == 5
    assert b.stats["pattern"]["side"].mean == pytest.approx(0.9)
    assert set(b.dims["spell_timing"]) == {"mid", "late_deep"}


def test_unreadable_loot_is_not_scored_as_zero():
    """A failed OCR and an empty raid both returned 0.0, so an unreadable
    result screen taught the bandit that the tactic scored nothing."""
    from ai.policies.online_strategy import loot_reward

    assert loot_reward({"gold": None, "elixir": None, "dark": None},
                       {"gold": 500_000}) is None
    assert loot_reward({"gold": 0, "elixir": 0, "dark": 0}, {"gold": 500_000}) == 0.0


# ------------------------------------------------------- the timed schedule
def _army_and_keys():
    from ai.perception.army_bar import parse_army_spec

    keys = ["1", "2", "q", "w", "e", "a", "s"]
    return parse_army_spec("4,8,H,H,H,S2,S5"), keys


def test_schedule_deploys_troops_then_heroes_then_spells():
    """Hero timing and spell delay are only real if the steps are spread over
    time -- an instant burst cannot express either."""
    import numpy as np

    from ai.policies import attack_plan as ap, deploy_patterns as dp

    army, keys = _army_and_keys()
    steps = ap.build_schedule(dp.Plan("side", angle=0.3), 1000, 1000, army, keys,
                              {"hero_timing": "after_funnel", "spell_timing": "mid"},
                              np.random.default_rng(0))
    t = {lab: [s.t for s in steps if s.label == lab]
         for lab in ("troop", "hero", "spell")}
    assert len(t["troop"]) == 12 and len(t["hero"]) == 3 and len(t["spell"]) == 7
    assert max(t["troop"]) < min(t["hero"]) < min(t["spell"])
    assert [s.t for s in steps] == sorted(s.t for s in steps)


def test_spells_land_ahead_of_the_drop_ring_not_on_it():
    """A spell is worth something on top of the fighting, and the fighting is
    not where the troops landed -- it is where they have walked to."""
    import numpy as np

    from ai.policies import attack_plan as ap, deploy_patterns as dp

    army, keys = _army_and_keys()
    steps = ap.build_schedule(dp.Plan("side", angle=0.3), 1000, 1000, army, keys,
                              {"spell_timing": "mid"}, np.random.default_rng(0))
    centre = (dp.CENTRE[0] * 1000, dp.CENTRE[1] * 1000)
    troops = [s.point for s in steps if s.label == "troop"]
    spells = [s.point for s in steps if s.label == "spell"]

    def d(p):
        return ((p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2) ** 0.5

    assert max(d(p) for p in spells) < min(d(p) for p in troops), \
        "every spell must land closer to the base centre than any troop drop"


def test_spells_do_not_stack_on_one_tile():
    """Regression: two spell cards each started their own stagger count, so
    both first casts landed at the same time on the same point."""
    import numpy as np

    from ai.policies import attack_plan as ap, deploy_patterns as dp

    army, keys = _army_and_keys()
    steps = ap.build_schedule(dp.Plan("side", angle=0.3), 1000, 1000, army, keys,
                              {"spell_timing": "mid"}, np.random.default_rng(0))
    spells = [s for s in steps if s.label == "spell"]
    assert len({s.point for s in spells}) == len(spells)
    assert len({round(s.t, 2) for s in spells}) == len(spells)


def test_later_spell_timing_aims_deeper():
    """The model of the game: troops walk inward, so a later cast must lead
    further. If it did not, the delay would be aiming at empty ground."""
    import numpy as np

    from ai.policies import attack_plan as ap, deploy_patterns as dp

    army, keys = _army_and_keys()
    plan = dp.Plan("side", angle=0.3)
    centre = (dp.CENTRE[0] * 1000, dp.CENTRE[1] * 1000)

    def first_spell(timing):
        steps = ap.build_schedule(plan, 1000, 1000, army, keys,
                                  {"spell_timing": timing}, np.random.default_rng(0))
        s = [x for x in steps if x.label == "spell"][0]
        return s.t, ((s.point[0] - centre[0]) ** 2 + (s.point[1] - centre[1]) ** 2) ** 0.5

    t_early, d_early = first_spell("early_shallow")
    t_late, d_late = first_spell("late_deep")
    assert t_late > t_early
    assert d_late < d_early, "a later cast must land deeper, following the fight in"


def test_heroes_still_share_one_point_under_the_schedule():
    import numpy as np

    from ai.policies import attack_plan as ap, deploy_patterns as dp

    army, keys = _army_and_keys()
    steps = ap.build_schedule(dp.Plan("ring", angle=1.0), 1000, 1000, army, keys,
                              {"hero_spot": "flank"}, np.random.default_rng(0))
    pts = {s.point for s in steps if s.label == "hero"}
    assert len(pts) == 1
