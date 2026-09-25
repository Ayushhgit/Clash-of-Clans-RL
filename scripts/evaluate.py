"""Phase 23: the benchmark table.

    python -m scripts.evaluate --episodes 30 \
        --checkpoint experiments/ppo_v1/ppo_latest.pt

Compares every agent that exists on every difficulty level, on evaluation
seeds that were never trained on, and writes both JSON and a markdown table.
"""
from __future__ import annotations

import argparse
import pathlib

from ai.policies.baselines import HeuristicAgent, RandomAgent
from evaluation.benchmark import benchmark, save_json, to_table
from simulator.env import EnvConfig
from simulator.generator import LEVELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    ap.add_argument("--checkpoint", default=None, help="PPO or BC checkpoint")
    ap.add_argument("--bc-checkpoint", default=None)
    ap.add_argument("--obs-mode", default="privileged", choices=["privileged", "detected"])
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--metric", default="win_rate",
                    choices=["win_rate", "mean_loot", "mean_destruction",
                             "mean_reward", "unit_efficiency"],
                    help="which number the comparison table shows")
    ap.add_argument("--out", default="experiments/benchmark.json")
    args = ap.parse_args()

    agents = {"random": RandomAgent(seed=0), "heuristic": HeuristicAgent(seed=0)}

    if args.bc_checkpoint and pathlib.Path(args.bc_checkpoint).exists():
        from ai.policies.neural_agent import NeuralAgent

        agents["bc"] = NeuralAgent(checkpoint=args.bc_checkpoint,
                                   deterministic=args.deterministic)
    if args.checkpoint and pathlib.Path(args.checkpoint).exists():
        from ai.policies.neural_agent import NeuralAgent

        agents["ppo"] = NeuralAgent(checkpoint=args.checkpoint,
                                    deterministic=args.deterministic)
    elif args.checkpoint:
        print(f"[warn] no checkpoint at {args.checkpoint}; skipping the learned agent")

    metrics = benchmark(agents, levels=args.levels, episodes=args.episodes,
                        cfg_factory=lambda: EnvConfig(obs_mode=args.obs_mode))
    save_json(metrics, args.out)

    table = to_table(metrics, metric=args.metric)
    print()
    print(f"{args.metric}, {args.episodes} unseen bases per cell, "
          f"obs_mode={args.obs_mode}")
    print(table)

    md = pathlib.Path(args.out).with_suffix(".md")
    md.write_text(f"# Benchmark ({args.obs_mode})\n\n{table}\n\n"
                  + "\n".join(m.row() for m in metrics) + "\n")
    print(f"\nwrote {args.out} and {md}")


if __name__ == "__main__":
    main()
