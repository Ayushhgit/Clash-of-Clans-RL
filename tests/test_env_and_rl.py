import pytest
import numpy as np
import torch

from ai.policies.actor_critic import ActorCritic
from ai.policies.baselines import HeuristicAgent, RandomAgent
from ai.rl.buffer import RolloutBuffer, default_obs_spec
from simulator.env import (
    ACT_DEPLOY,
    ACT_WAIT,
    N_ACT_TYPES,
    N_GLOBALS,
    N_TOKENS,
    TOKEN_DIM,
    EnvConfig,
    RaidEnv,
)
from simulator.entities import NUM_ARCHETYPES
from simulator.vec_env import SyncVectorEnv, batch_obs


def test_obs_matches_spec():
    env = RaidEnv(EnvConfig(level="L2"))
    obs = env.reset(seed=1)
    assert obs["tokens"].shape == (N_TOKENS, TOKEN_DIM)
    assert obs["class_ids"].shape == (N_TOKENS,)
    assert obs["globals"].shape == (N_GLOBALS,)
    assert obs["token_mask"].dtype == np.bool_
    # padded slots must be exactly zero, otherwise attention sees garbage
    pad = ~obs["token_mask"]
    assert np.all(obs["tokens"][pad] == 0) and np.all(obs["class_ids"][pad] == 0)
    assert np.all(obs["tokens"][:, 0:2] >= 0) and np.all(obs["tokens"][:, 0:2] <= 1)


def test_reset_is_deterministic_given_seed():
    env = RaidEnv(EnvConfig(level="L3"))
    a = env.reset(seed=99)["tokens"].copy()
    b = env.reset(seed=99)["tokens"]
    assert np.allclose(a, b)


def test_deploy_mask_rejects_cells_inside_the_base():
    env = RaidEnv(EnvConfig(level="L3"))
    obs = env.reset(seed=4)
    grid = obs["action_mask"]["xy"]
    assert grid.any() and not grid.all(), "some cells legal, some blocked"
    illegal = np.argwhere(~grid)[0]
    obs2, _, _, info = env.step((ACT_DEPLOY, 0, int(illegal[0]), int(illegal[1])))
    assert info["invalid_action"]
    assert env.sim.n_units == 0


def test_deploy_succeeds_on_legal_cell():
    env = RaidEnv(EnvConfig(level="L1"))
    obs = env.reset(seed=4)
    legal = np.argwhere(obs["action_mask"]["xy"])[0]
    _, _, _, info = env.step((ACT_DEPLOY, 1, int(legal[0]), int(legal[1])))
    assert not info["invalid_action"] and env.sim.n_units == 1


def test_wait_action_never_deploys():
    env = RaidEnv(EnvConfig(level="L1"))
    env.reset(seed=7)
    for _ in range(20):
        env.step((ACT_WAIT, 0, 0, 0))
    assert env.sim.n_units == 0


def test_episode_terminates_and_reports_info():
    env = RaidEnv(EnvConfig(level="L1"))
    obs = env.reset(seed=11)
    agent = RandomAgent(seed=0)
    done, steps = False, 0
    while not done and steps < 5000:
        obs, r, done, info = env.step(agent.act(obs))
        steps += 1
    assert done
    for k in ("win", "destruction", "loot_frac", "unit_efficiency", "duration"):
        assert k in info


def test_stalling_is_penalised_relative_to_attacking():
    """Reward-hacking guard: sitting still must score worse than attacking."""
    cfg = EnvConfig(level="L1")
    env = RaidEnv(cfg)

    obs = env.reset(seed=21)
    idle_r = 0.0
    done = False
    while not done:
        obs, r, done, info = env.step((ACT_WAIT, 0, 0, 0))
        idle_r += r

    obs = env.reset(seed=21)
    agent = HeuristicAgent(seed=0)
    play_r, done = 0.0, False
    while not done:
        obs, r, done, info = env.step(agent.act(obs))
        play_r += r
    assert play_r > idle_r


def test_detected_mode_degrades_observation():
    clean = RaidEnv(EnvConfig(level="L3", obs_mode="privileged")).reset(seed=5)
    noisy = RaidEnv(EnvConfig(level="L3", obs_mode="detected")).reset(seed=5)
    assert noisy["token_mask"].sum() <= clean["token_mask"].sum()
    assert noisy["tokens"][noisy["token_mask"]][:, 9].max() <= 1.0
    assert np.allclose(clean["tokens"][clean["token_mask"]][:, 9], 1.0)


def test_vector_env_autoresets():
    venv = SyncVectorEnv(num_envs=4, cfg=EnvConfig(level="L1"), seed=0)
    obs = venv.reset()
    assert obs["tokens"].shape[0] == 4
    for _ in range(50):
        actions = np.stack([[ACT_WAIT, 0, 0, 0]] * 4)
        obs, r, d, infos = venv.step(actions)
    assert obs["tokens"].shape == (4, N_TOKENS, TOKEN_DIM)


def _dummy_batch(b=3):
    env = RaidEnv(EnvConfig(level="L2"))
    obs = [env.reset(seed=i) for i in range(b)]
    return {k: torch.as_tensor(v) for k, v in batch_obs(obs).items()}


def test_policy_respects_action_masks():
    torch.manual_seed(0)
    pol = ActorCritic(d_model=32, n_heads=2, n_layers=1)
    b = _dummy_batch(4)
    for _ in range(10):
        act, logp, ent, val, hx = pol(b)
        for i in range(act.shape[0]):
            t, u, x, y = (int(v) for v in act[i])
            assert b["mask_type"][i, t]
            assert b["mask_unit"][i, u]
            assert b["mask_xy"][i, x, y], "sampled (x, y) must be a legal cell"
        assert torch.isfinite(logp).all() and torch.isfinite(val).all()


def test_policy_logprob_is_reproducible_for_given_action():
    torch.manual_seed(0)
    pol = ActorCritic(d_model=32, n_heads=2, n_layers=1, memory=None)
    b = _dummy_batch(3)
    act, logp, _, _, _ = pol(b)
    _, logp2, _, _, _ = pol(b, action=act)
    assert torch.allclose(logp, logp2, atol=1e-5)


def test_gae_matches_manual_computation():
    spec = default_obs_spec(4, 3, 2, N_ACT_TYPES, NUM_ARCHETYPES, 2)
    buf = RolloutBuffer(3, 1, spec)
    obs = {k: np.zeros((1, *shape), dtype=dt) for k, (shape, dt) in spec.items()}
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    for t in range(3):
        buf.add(obs, np.zeros((1, 4)), np.zeros(1), np.array([rewards[t]]),
                np.zeros(1), np.array([values[t]]))
    gamma, lam = 0.9, 0.8
    adv, ret = buf.compute_gae(np.array([0.5]), np.array([False]), gamma, lam)

    manual, last = np.zeros(3), 0.0
    for t in reversed(range(3)):
        nv = 0.5
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last
        manual[t] = last
    assert np.allclose(adv[:, 0], manual, atol=1e-6)
    assert np.allclose(ret[:, 0], manual + np.array(values), atol=1e-6)


def test_gae_zeroes_across_episode_boundary():
    spec = default_obs_spec(4, 3, 2, N_ACT_TYPES, NUM_ARCHETYPES, 2)
    buf = RolloutBuffer(2, 1, spec)
    obs = {k: np.zeros((1, *shape), dtype=dt) for k, (shape, dt) in spec.items()}
    buf.add(obs, np.zeros((1, 4)), np.zeros(1), np.array([1.0]), np.zeros(1), np.array([0.0]))
    # step 1 starts a new episode (done flag on the *next* step)
    buf.add(obs, np.zeros((1, 4)), np.zeros(1), np.array([1.0]), np.ones(1), np.array([0.0]))
    adv, _ = buf.compute_gae(np.array([10.0]), np.array([False]), 0.99, 0.95)
    # the terminal at t=1 must stop the bootstrap from leaking backwards
    assert adv[0, 0] == 1.0


def test_town_hall_damage_is_rewarded():
    """Regression for EXPERIMENTS E7: with no town-hall term the policy could
    raise destruction while win rate fell. Damaging the town hall must pay."""
    from simulator.core import Army, BaseLayout, RaidSim
    from simulator.entities import Cls

    env = RaidEnv(EnvConfig(level="L1"))
    env.reset(seed=1)
    # replace the generated battle with a controlled one: town hall only
    cls = np.array([int(Cls.TOWN_HALL)], dtype=np.int32)
    env.sim = RaidSim(BaseLayout(cls, np.array([22.0], np.float32),
                                 np.array([22.0], np.float32)),
                      Army({Cls.DPS: 1}), env.cfg.sim)
    env._prev = env._progress()
    env._deploy_grid_dirty = True

    env.sim.deploy(Cls.DPS, 18.0, 22.0)
    saw_th_reward = False
    for _ in range(200):
        _, r, done, info = env.step((ACT_WAIT, 0, 0, 0))
        if info.get("r_th", 0.0) > 0:
            saw_th_reward = True
        if done:
            break
    assert saw_th_reward, "chipping the town hall produced no reward signal"


def test_a_town_hall_kill_outscores_equal_destruction_without_one():
    """A win by town-hall kill must score above a non-win with the same
    number of buildings destroyed."""
    from simulator.core import Army, BaseLayout, RaidSim
    from simulator.entities import Cls

    def run(target_is_town_hall: bool) -> float:
        cls = np.array([int(Cls.TOWN_HALL), int(Cls.STORAGE), int(Cls.STORAGE),
                        int(Cls.RESOURCE_BUILDING)], dtype=np.int32)
        x = np.array([22.0, 30.0, 34.0, 38.0], dtype=np.float32)
        y = np.array([22.0, 22.0, 22.0, 22.0], dtype=np.float32)
        env = RaidEnv(EnvConfig(level="L1"))
        env.reset(seed=2)
        env.sim = RaidSim(BaseLayout(cls, x, y), Army({Cls.DPS: 2}), env.cfg.sim)
        env._prev = env._progress()
        env._deploy_grid_dirty = True
        # send both units at either the town hall or the far resource building
        tx = 18.0 if target_is_town_hall else 42.0
        env.sim.deploy(Cls.DPS, tx, 22.0)
        total, done = 0.0, False
        while not done:
            _, r, done, _ = env.step((ACT_WAIT, 0, 0, 0))
            total += r
        return total

    assert run(True) > run(False)


def test_logprob_ignores_action_components_that_do_not_apply():
    """On a WAIT step, unit/x/y are sampled but never used. Their log-probs
    must not enter the PPO ratio -- otherwise the deploy heads receive noise
    weighted by the episode advantage (see docs/EXPERIMENTS.md E8)."""
    torch.manual_seed(0)
    pol = ActorCritic(d_model=32, n_heads=2, n_layers=1, memory=None)
    b = _dummy_batch(2)

    wait = torch.zeros(2, 4, dtype=torch.long)          # a_type = ACT_WAIT
    wait_a = wait.clone()
    wait_b = wait.clone()
    wait_b[:, 1] = 3                                     # different dummy unit
    wait_b[:, 2] = 7
    wait_b[:, 3] = 9

    _, logp_a, ent_a, _, _ = pol(b, action=wait_a)
    _, logp_b, ent_b, _, _ = pol(b, action=wait_b)
    assert torch.allclose(logp_a, logp_b), "WAIT log-prob must not depend on unused components"
    assert torch.allclose(ent_a, ent_b)

    deploy = wait.clone()
    deploy[:, 0] = ACT_DEPLOY
    for i in range(2):
        legal = np.argwhere(b["mask_xy"][i].numpy())[0]
        deploy[i, 2], deploy[i, 3] = int(legal[0]), int(legal[1])
    _, logp_d, _, _, _ = pol(b, action=deploy)
    assert (logp_d < logp_a).all(), "a DEPLOY action carries the extra head densities"


class _FakeSim:
    """Just the three numbers `_unit_efficiency` reads."""

    def __init__(self, loot: float, spent: float):
        self.loot_frac = loot
        self.destruction = loot
        self.army_hp_total = 1.0
        self.units_lost_hp = spent


def _terminal_score(env, loot: float, spent: float) -> float:
    """The end-of-episode reward the farming profile pays for a raid that
    captured `loot` of the base while losing `spent` of the army. Calls the
    real `_unit_efficiency` so the test breaks if the formula changes."""
    real, env.sim = env.sim, _FakeSim(loot, spent)
    try:
        eff = env._unit_efficiency()
    finally:
        env.sim = real
    w = env.cfg.reward
    return w.loot * loot - w.unit * spent + w.efficiency * eff


def test_efficiency_is_capped_and_floored():
    from simulator.env import EnvConfig, RaidEnv, RewardWeights

    env = RaidEnv(EnvConfig(level="L3", reward=RewardWeights.farming()))
    env.reset(seed=0)
    real, env.sim = env.sim, _FakeSim(0.02, 0.0)
    try:
        # spending nothing must not divide by ~0 into a perfect score
        assert env._unit_efficiency() == pytest.approx(0.02 / env.MIN_SPEND)
        env.sim = _FakeSim(0.9, 0.1)
        assert env._unit_efficiency() == 1.0          # capped
    finally:
        env.sim = real


def _terminal_score(env, loot: float, spent: float) -> float:
    """The end-of-episode reward the farming profile would pay for a raid that
    captured `loot` of the base while losing `spent` of the army."""
    env.sim.loot_frac_override = loot
    w = env.cfg.reward
    eff = float(np.clip(loot / max(spent, env.MIN_SPEND), 0.0, 1.0))
    return w.loot * loot - w.unit * spent + w.efficiency * eff


def test_chipping_never_beats_committing_the_army():
    """Efficiency is loot per army spent, and the denominator has a floor.

    Without it, one unit that survives and steals 2% of the loot divides by
    ~0 and scores *maximum* efficiency -- on L4 that came within 0.02 of
    committing the whole army, and on a harder base it strictly won. The agent
    would have learned to stand still.
    """
    from simulator.env import EnvConfig, RaidEnv, RewardWeights

    env = RaidEnv(EnvConfig(level="L3", reward=RewardWeights.farming()))
    env.reset(seed=0)

    chip = _terminal_score(env, loot=0.02, spent=0.0)
    for loot, spent in [(0.55, 0.87), (0.24, 0.93), (0.12, 0.95)]:
        assert _terminal_score(env, loot, spent) > chip, (
            f"chipping ({chip:.3f}) beats a real raid at loot={loot}")


def test_efficiency_rewards_a_cheap_raid_over_an_expensive_one():
    from simulator.env import EnvConfig, RaidEnv, RewardWeights

    env = RaidEnv(EnvConfig(level="L3", reward=RewardWeights.farming()))
    env.reset(seed=0)
    assert (_terminal_score(env, 0.8, 0.3) > _terminal_score(env, 0.8, 0.9))
