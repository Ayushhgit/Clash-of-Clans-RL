"""M8-M10: train the tactical policy with PPO.

    python -m scripts.train_ppo --config configs/ppo_small.yaml
    python -m scripts.train_ppo --total-steps 50000 --run-dir experiments/smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from ai.rl.ppo import PPO, PPOConfig
from simulator.env import EnvConfig, RewardWeights
from simulator.core import SimConfig


def load_config(path: str | None, overrides: dict) -> PPOConfig:
    raw: dict = {}
    if path:
        text = pathlib.Path(path).read_text()
        try:
            import yaml  # optional dependency

            raw = yaml.safe_load(text) or {}
        except ImportError:
            raw = json.loads(text)

    env_raw = raw.pop("env", {}) or {}
    sim_raw = env_raw.pop("sim", {}) or {}
    rew_raw = env_raw.pop("reward", {}) or {}
    env = EnvConfig(sim=SimConfig(**sim_raw), reward=RewardWeights(**rew_raw), **env_raw)

    cfg = PPOConfig(env=env, **raw)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--level", default=None)
    ap.add_argument("--memory", default=None, choices=["gru", "none"])
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--bc-init", default=None,
                    help="behavioural-cloning checkpoint to warm-start the "
                         "actor from (the critic stays fresh)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from (keeps step count and level)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    overrides = {
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "level": args.level,
        "memory": None if args.memory is None else (None if args.memory == "none" else "gru"),
        "run_dir": args.run_dir,
        "resume": args.resume,
        "device": args.device,
        "seed": args.seed,
    }
    if args.memory == "none":
        overrides["memory"] = "__none__"      # distinguish "unset" from "None"
    cfg = load_config(args.config, {k: v for k, v in overrides.items() if k != "memory"})
    if args.memory is not None:
        cfg.memory = None if args.memory == "none" else "gru"
    if args.no_curriculum:
        cfg.curriculum = False
    if args.level:
        # An explicit --level means a single level, so the config's `levels`
        # mix must be cleared too. Without this the run silently trains on the
        # mix while claiming to be a single-level run -- which would have made
        # the DQN/PPO comparison meaningless.
        cfg.env.level = args.level
        cfg.level = args.level
        cfg.levels = None

    print(json.dumps({k: str(v) for k, v in dataclasses.asdict(cfg).items()
                      if k not in ("env",)}, indent=2))
    trainer = PPO(cfg)
    if args.bc_init:
        # after PPO() so a --resume checkpoint, if any, wins: resuming a run
        # must not silently reset the actor to its warm start
        if cfg.resume:
            print("[bc->ppo] skipped: --resume takes precedence")
        else:
            from ai.imitation.bc import bc_to_ppo

            bc_to_ppo(args.bc_init, trainer)
    trainer.train()


if __name__ == "__main__":
    main()
