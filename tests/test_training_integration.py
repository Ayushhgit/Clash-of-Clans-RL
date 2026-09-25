"""Slow-ish integration tests: they run real (tiny) training loops.

These are the tests that would have caught the two bugs that actually
happened during development: a PPO update that never moved the policy, and a
BC accuracy metric that was dominated by WAIT steps.
"""
import pytest
import numpy as np
import torch

from ai.imitation.bc import bc_to_ppo
from ai.planning.base_selection import HeuristicSelector
from ai.policies.baselines import HeuristicAgent
from ai.policies.hierarchical import HierarchicalAgent, RaidSession, SessionConfig
from ai.rl.ppo import PPO, PPOConfig
from evaluation.failure_analysis import replay_seed
from simulator.core import SimConfig
from simulator.env import EnvConfig, RaidEnv
from simulator.vec_env import SyncVectorEnv


def _tiny_cfg(tmp_path, **kw):
    return PPOConfig(total_steps=2048, num_envs=4, rollout_steps=16, epochs=2,
                     minibatches=2, d_model=32, n_layers=1,
                     run_dir=str(tmp_path / "ppo"), log_every=1000,
                     save_every=1000, **kw)


def test_ppo_update_actually_moves_the_policy(tmp_path):
    """Regression: with too few gradient steps the KL was ~1e-4 and nothing
    learned. An update must change the parameters measurably."""
    trainer = PPO(_tiny_cfg(tmp_path))
    before = [p.detach().clone() for p in trainer.policy.parameters()]

    obs = trainer.envs.reset()
    hx = trainer.policy.initial_state(trainer.cfg.num_envs, trainer.device)
    done = np.zeros(trainer.cfg.num_envs, dtype=bool)
    obs, hx, done = trainer.collect(obs, hx, done)
    stats = trainer.update(obs, hx, done)

    after = list(trainer.policy.parameters())
    delta = max(float((a - b).abs().max()) for a, b in zip(after, before))
    assert delta > 0, "the optimiser step did nothing"
    assert np.isfinite(stats["pg_loss"]) and np.isfinite(stats["v_loss"])
    assert stats["entropy"] > 0


def test_ppo_train_loop_runs_and_checkpoints(tmp_path):
    trainer = PPO(_tiny_cfg(tmp_path, curriculum=False))
    trainer.train()
    ck = tmp_path / "ppo" / "ppo_final.pt"
    assert ck.exists()

    from ai.policies.neural_agent import NeuralAgent

    agent = NeuralAgent(checkpoint=str(ck))
    env = RaidEnv(EnvConfig(level="L1"))
    obs = env.reset(seed=5)
    agent.reset()
    for _ in range(10):
        a = agent.act(obs)
        assert obs["action_mask"]["type"][a[0]]
        obs, _, done, _ = env.step(a)
        if done:
            break


def test_bc_checkpoint_initialises_ppo(tmp_path):
    """M13: actor weights transfer, critic is deliberately reset."""
    trainer = PPO(_tiny_cfg(tmp_path))
    bc_path = tmp_path / "bc.pt"
    torch.save({"model": trainer.policy.state_dict(),
                "cfg": {"d_model": 32, "n_heads": 4, "n_layers": 1, "memory": "gru"}},
               bc_path)

    fresh = PPO(_tiny_cfg(tmp_path / "b"))
    v_before = [p.detach().clone() for p in fresh.policy.head_v.parameters()]
    bc_to_ppo(str(bc_path), fresh)

    for a, b in zip(fresh.policy.head_type.parameters(), trainer.policy.head_type.parameters()):
        assert torch.allclose(a, b), "actor head must be loaded"
    unchanged = all(torch.allclose(a, b) for a, b in
                    zip(fresh.policy.head_v.parameters(), v_before))
    assert unchanged, "critic must stay freshly initialised"


def _ep(level: str, win: float = 0.0, r: float = 0.0, loot: float = 0.0) -> dict:
    return {"level": level, "win": win, "r": r, "loot": loot}


def test_reported_metrics_ignore_episodes_from_the_previous_level(tmp_path):
    """A rolling window that straddles a promotion reports the old level's
    easy returns as if they were the new level's, which reads exactly like a
    regression -- it is why an earlier run looked like it was decaying when
    L3 was in fact still improving."""
    trainer = PPO(_tiny_cfg(tmp_path, level="L4"))
    trainer.recent = ([_ep("L3", win=1.0, r=3.0, loot=0.9) for _ in range(50)]
                      + [_ep("L4", win=0.0, r=0.5, loot=0.2) for _ in range(10)])
    w = trainer._window(100)
    assert len(w) == 10
    assert all(e["level"] == "L4" for e in w)
    assert max(e["r"] for e in w) == 0.5


def test_promotion_needs_a_full_window_on_the_new_level(tmp_path):
    """Right after a promotion there is no evidence about the new level yet,
    so the old level's wins must not carry the agent straight through it."""
    trainer = PPO(_tiny_cfg(tmp_path, level="L2", promote_window=5,
                            promote_win_rate=0.5))
    trainer.recent = [_ep("L1", win=1.0) for _ in range(20)]
    trainer._maybe_promote()
    assert trainer.level == "L2"


def test_curriculum_promotes_on_a_high_win_rate(tmp_path):
    trainer = PPO(_tiny_cfg(tmp_path, level="L1", promote_window=5,
                            promote_win_rate=0.5))
    trainer.recent = [_ep("L1", win=1.0) for _ in range(5)]
    trainer._maybe_promote()
    assert trainer.level == "L2" and trainer.envs.level == "L2"


def test_curriculum_does_not_promote_on_a_low_win_rate(tmp_path):
    trainer = PPO(_tiny_cfg(tmp_path, level="L1", promote_window=5,
                            promote_win_rate=0.9))
    trainer.recent = [_ep("L1", win=0.0) for _ in range(5)]
    trainer._maybe_promote()
    assert trainer.level == "L1"


def test_vector_env_level_switch_takes_effect_on_reset():
    venv = SyncVectorEnv(num_envs=2, cfg=EnvConfig(level="L1"), seed=0)
    venv.reset()
    venv.set_level("L3")
    obs = venv.reset()
    assert obs["tokens"].shape[0] == 2
    assert venv.envs[0].episode_level == "L3"


# ------------------------------------------------------- domain randomisation
def test_randomisation_changes_stats_but_stays_bounded():
    plain = RaidEnv(EnvConfig(level="L2"))
    plain.reset(seed=11)
    rand = RaidEnv(EnvConfig(level="L2", sim=SimConfig(randomize=True, stat_jitter=0.2)))
    rand.reset(seed=11)

    ratio = rand.sim.b_maxhp / plain.sim.b_maxhp
    assert not np.allclose(ratio, 1.0), "randomisation must actually perturb stats"
    assert ratio.min() >= 0.79 and ratio.max() <= 1.21, "jitter must respect its bound"


def test_randomised_env_is_still_seed_reproducible():
    cfg = EnvConfig(level="L2", sim=SimConfig(randomize=True))
    a = RaidEnv(cfg)
    a.reset(seed=7)
    b = RaidEnv(cfg)
    b.reset(seed=7)
    assert np.allclose(a.sim.b_maxhp, b.sim.b_maxhp)
    assert np.allclose(a.sim.b_dps, b.sim.b_dps)


# ------------------------------------------------------------- hierarchical
def test_raid_session_reports_strategic_metrics():
    sess = RaidSession(HeuristicAgent(seed=0),
                       SessionConfig(level="L2", budget=4, max_attacks=2))
    out = sess.run(HeuristicSelector(), n_sessions=1)
    assert out["attacks"] + out["skips"] > 0
    assert 0.0 <= out["skip_rate"] <= 1.0
    assert "reward_per_session" in out


def test_hierarchical_agent_exposes_both_interfaces():
    from simulator.generator import BaseGenerator

    agent = HierarchicalAgent(HeuristicSelector(), HeuristicAgent(seed=0))
    base = BaseGenerator().generate("L3", seed=1)
    assert agent.decide(base) in ("ATTACK", "NEXT")

    env = RaidEnv(EnvConfig(level="L3"))
    obs = env.reset(seed=1)
    agent.reset()
    a = agent.act(obs)
    assert obs["action_mask"]["type"][a[0]]
    assert "decision" in agent.explain(base)


# ------------------------------------------------------------------ replay
def test_logged_actions_replay_to_the_same_outcome():
    """Failure analysis is worthless if an episode cannot be reproduced."""
    env = RaidEnv(EnvConfig(level="L2"))
    obs = env.reset(seed=10_000_042, level="L2")
    agent = HeuristicAgent(seed=0)
    actions, done, info = [], False, {}
    while not done:
        a = agent.act(obs)
        actions.append(a)
        obs, _, done, info = env.step(a)

    replayed = replay_seed(10_000_042, "L2", actions)
    assert replayed["destruction"] == info["destruction"]
    assert replayed["win"] == info["win"]


def test_curriculum_can_gate_on_loot_instead_of_wins(tmp_path):
    """The farming profile pays nothing for a win, so gating promotion on win
    rate asks the agent to clear a bar it is not being trained to clear."""
    cfg = _tiny_cfg(tmp_path, level="L1", promote_window=5, promote_win_rate=0.7)
    cfg.promote_metric, cfg.promote_threshold = "loot", 0.5
    trainer = PPO(cfg)
    trainer.recent = [_ep("L1", win=0.0, loot=0.8) for _ in range(5)]
    trainer._maybe_promote()
    assert trainer.level == "L2", "high loot with zero wins must still promote"

    cfg2 = _tiny_cfg(tmp_path / "c", level="L1", promote_window=5)
    cfg2.promote_metric, cfg2.promote_threshold = "loot", 0.5
    t2 = PPO(cfg2)
    t2.recent = [_ep("L1", win=1.0, loot=0.2) for _ in range(5)]
    t2._maybe_promote()
    assert t2.level == "L1", "wins must not promote when loot is the metric"


def test_level_mix_is_spread_across_envs_not_sampled():
    """Round-robin, so every rollout contains every level at a fixed ratio.
    Sampling leaves whole updates with no hard base in them and the gradient
    then swings with the draw."""
    venv = SyncVectorEnv(num_envs=6, cfg=EnvConfig(level="L1"), seed=0)
    venv.set_level(["L2", "L3", "L5"])
    venv.reset()
    assert [e.episode_level for e in venv.envs] == ["L2", "L3", "L5"] * 2


def test_level_mix_disables_the_curriculum(tmp_path):
    cfg = _tiny_cfg(tmp_path, level="L1")
    cfg.levels, cfg.curriculum = ["L2", "L3"], True
    trainer = PPO(cfg)
    assert trainer.level == "mix" and not trainer.cfg.curriculum


def test_mixed_window_keeps_every_level(tmp_path):
    """Under a curriculum the window filters to the current level; on a mix
    every level *is* the training distribution, so filtering would empty it."""
    cfg = _tiny_cfg(tmp_path / "m", level="L1")
    cfg.levels = ["L2", "L3"]
    trainer = PPO(cfg)
    trainer.recent = [_ep("L2", loot=0.8), _ep("L3", loot=0.2)]
    assert len(trainer._window(100)) == 2


def _wait_only_batch(n: int = 8):
    """A batch of pure WAIT steps whose dummy unit label is masked illegal --
    exactly the shape of a late-episode batch with the army spent."""
    from simulator.env import N_ACT_TYPES, XY_BINS
    from simulator.entities import UNIT_CLASSES

    n_unit, n_tok = len(UNIT_CLASSES), 8
    obs = {
        "tokens": torch.zeros(n, n_tok, 16),
        "class_ids": torch.zeros(n, n_tok, dtype=torch.long),
        "token_mask": torch.ones(n, n_tok, dtype=torch.bool),
        "globals": torch.zeros(n, 12),
        "mask_type": torch.zeros(n, N_ACT_TYPES, dtype=torch.bool),
        "mask_unit": torch.zeros(n, n_unit, dtype=torch.bool),
        "mask_xy": torch.ones(n, XY_BINS, XY_BINS, dtype=torch.bool),
    }
    obs["mask_type"][:, 0] = True        # only WAIT is legal
    obs["mask_unit"][:, -1] = True       # unit 0 -- the dummy label -- is not
    action = torch.zeros(n, 4, dtype=torch.long)
    return obs, action


def test_bc_loss_is_finite_when_a_batch_contains_no_deploys():
    """The unit/x/y heads take no decision on a WAIT step, so their loss must
    be zero -- not the cross-entropy of a dummy label against a -1e9 masked
    logit, which is ~1e9 and was being backpropagated."""
    from ai.imitation.bc import bc_loss
    from ai.policies.actor_critic import ActorCritic
    from simulator.env import N_TOKENS

    obs, action = _wait_only_batch()
    policy = ActorCritic(d_model=32, n_layers=1, memory=None)
    obs["tokens"] = torch.zeros(obs["tokens"].shape[0], obs["tokens"].shape[1],
                                policy.token_dim) if hasattr(policy, "token_dim") \
        else obs["tokens"]
    total, parts = bc_loss(policy, obs, action)
    assert torch.isfinite(total) and total.item() < 100, parts
    for k in ("loss_unit", "loss_x", "loss_y"):
        assert parts[k] == 0.0, f"{k} should be 0 with no deploys, got {parts[k]}"


def test_bc_type_weight_counteracts_the_wait_majority():
    """94% of demonstration steps are WAIT. Unweighted, the head reaches the
    base rate by predicting WAIT unconditionally and stops learning -- it did,
    at exactly 0.9386, the WAIT fraction."""
    from ai.imitation.bc import type_class_weights

    w = type_class_weights([80362, 5280, 0, 0])
    assert w[1] > w[0], "the rare DEPLOY class must be weighted up"
    assert w[1] / w[0] == pytest.approx(80362 / 5280, rel=1e-4)
    assert w[2] == 0.0 and w[3] == 0.0, "unseen classes must not be infinite"
    assert torch.isfinite(w).all()


def test_bc_type_weight_changes_the_loss_on_a_mixed_batch():
    """A weighted cross-entropy is a weighted *mean*, so it only bites when
    both classes are present -- which is the case that matters."""
    from ai.imitation.bc import bc_loss
    from ai.policies.actor_critic import ActorCritic

    obs, action = _wait_only_batch(n=4)
    obs["mask_type"][:, 1] = True
    obs["mask_unit"][:] = True
    action[2:, 0] = 1                       # half the batch deploys
    policy = ActorCritic(d_model=32, n_layers=1, memory=None)
    plain, _ = bc_loss(policy, obs, action)
    weighted, _ = bc_loss(policy, obs, action,
                          type_weight=torch.tensor([0.2, 5.0, 0.0, 0.0]))
    assert abs(weighted.item() - plain.item()) > 1e-6


def test_critic_warmup_freezes_the_actor_but_not_the_value_head(tmp_path):
    """A BC warm start hands PPO a good actor and a random critic. Letting the
    actor move on those first noisy advantages is what evaporates the warm
    start, so during warm-up only the value head may change."""
    cfg = _tiny_cfg(tmp_path, level="L1")
    cfg.critic_warmup_updates = 5
    trainer = PPO(cfg)
    before = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}

    trainer._set_actor_trainable(False)
    obs = trainer.envs.reset()
    hx = trainer.policy.initial_state(cfg.num_envs, trainer.device)
    done = np.zeros(cfg.num_envs, dtype=bool)
    obs, hx, done = trainer.collect(obs, hx, done)
    stats = trainer.update(obs, hx, done)

    assert stats["warmup"] == 1.0
    after = trainer.policy.state_dict()
    moved = {k for k in before if not torch.allclose(before[k], after[k])}
    assert moved, "the value head should have been fitted"
    assert all(k.startswith("head_v.") for k in moved), (
        f"non-critic parameters moved during warm-up: {sorted(moved)[:5]}")


def test_actor_trains_again_once_warmup_is_over(tmp_path):
    cfg = _tiny_cfg(tmp_path / "w2", level="L1")
    cfg.critic_warmup_updates = 1
    trainer = PPO(cfg)
    trainer.updates = 5                     # past the warm-up
    trainer._set_actor_trainable(True)
    before = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}

    obs = trainer.envs.reset()
    hx = trainer.policy.initial_state(cfg.num_envs, trainer.device)
    done = np.zeros(cfg.num_envs, dtype=bool)
    obs, hx, done = trainer.collect(obs, hx, done)
    stats = trainer.update(obs, hx, done)

    assert stats["warmup"] == 0.0
    after = trainer.policy.state_dict()
    moved = {k for k in before if not torch.allclose(before[k], after[k])}
    assert any(k.startswith("head_type.") for k in moved), "actor stayed frozen"


def test_resuming_a_level_mix_restores_the_mix_not_the_label(tmp_path):
    """Checkpoints record the level as "mix", which is a label for a set of
    levels rather than a level the base generator knows."""
    cfg = _tiny_cfg(tmp_path, level="L1")
    cfg.levels = ["L2", "L4"]
    trainer = PPO(cfg)
    ck = tmp_path / "mix.pt"
    trainer.save(ck)

    cfg2 = _tiny_cfg(tmp_path / "r", level="L1")
    cfg2.levels, cfg2.resume = ["L2", "L4"], str(ck)
    resumed = PPO(cfg2)
    assert resumed.level == "mix"
    assert resumed.envs.levels == ["L2", "L4"]
    resumed.envs.reset()          # would raise if "mix" reached the generator


def test_advantages_are_whitened_per_level_on_a_mix(tmp_path):
    """Returns differ ~10x across the mix, and advantage magnitude follows
    reward magnitude. Pooling them hands most of the gradient to the easiest
    level: over 256k steps L2 went 0.89 -> 0.95 while L3 went 0.68 -> 0.52 and
    L5 0.28 -> 0.13, ending below the BC it started from."""
    cfg = _tiny_cfg(tmp_path, level="L1")
    cfg.levels, cfg.eval_every = ["L2", "L5"], 0
    trainer = PPO(cfg)
    assert list(trainer.env_level) == [0, 1, 0, 1]

    rng = np.random.default_rng(0)
    adv = np.zeros((4, cfg.num_envs), dtype=np.float32)
    adv[:, 0::2] = rng.normal(3.0, 5.0, (4, 2))       # the easy level
    adv[:, 1::2] = rng.normal(0.1, 0.2, (4, 2))       # the hard one
    out = trainer._normalise_per_level(adv)

    for cols in (slice(0, None, 2), slice(1, None, 2)):
        assert abs(float(out[:, cols].mean())) < 1e-4
        assert abs(float(out[:, cols].std()) - 1.0) < 1e-3


def test_single_level_runs_keep_minibatch_normalisation(tmp_path):
    """Per-level whitening applies only to a mix; a single-level run must keep
    the standard per-minibatch normalisation it was tuned with."""
    cfg = _tiny_cfg(tmp_path / "s", level="L2")
    trainer = PPO(cfg)
    assert cfg.levels is None
    assert list(trainer.env_level) == [0] * cfg.num_envs
