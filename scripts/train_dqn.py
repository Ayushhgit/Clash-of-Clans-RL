"""Train the branching DQN on the farming objective.

    python -m scripts.train_dqn --total-steps 300000 --level L3
"""
from __future__ import annotations

import argparse
import json

from ai.rl.dqn import DQN, DQNConfig
from simulator.env import EnvConfig, RewardWeights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--level", default="L3")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--run-dir", default="experiments/dqn")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = EnvConfig(level=args.level, reward=RewardWeights.farming())
    cfg = DQNConfig(total_steps=args.total_steps, num_envs=args.num_envs,
                    level=args.level, levels=args.levels, lr=args.lr,
                    seed=args.seed, run_dir=args.run_dir, env=env)
    print(json.dumps({k: str(v) for k, v in vars(cfg).items() if k != "env"}, indent=1))
    DQN(cfg).train()


if __name__ == "__main__":
    main()
