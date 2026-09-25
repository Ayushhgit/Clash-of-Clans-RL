"""Render one episode as a GIF, and plot the training curve.

    python -m scripts.render_episode --agent heuristic --level L3
    python -m scripts.render_episode --plot experiments/ppo_v1/log.jsonl
"""
from __future__ import annotations

import argparse

from ai.policies.baselines import HeuristicAgent, RandomAgent
from visualization.render import plot_training, record_episode


def build_agent(name: str, checkpoint: str | None):
    if name == "random":
        return RandomAgent(seed=0)
    if name == "heuristic":
        return HeuristicAgent(seed=0)
    from ai.policies.neural_agent import NeuralAgent

    return NeuralAgent(checkpoint=checkpoint)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="heuristic", choices=["random", "heuristic", "neural"])
    ap.add_argument("--checkpoint", default="experiments/ppo_v1/ppo_latest.pt")
    ap.add_argument("--level", default="L3")
    ap.add_argument("--seed", type=int, default=10_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None, help="path to a PPO log.jsonl")
    args = ap.parse_args()

    if args.plot:
        print(plot_training(args.plot))
        return
    out = args.out or f"experiments/episode_{args.agent}_{args.level}.gif"
    print(record_episode(build_agent(args.agent, args.checkpoint),
                         level=args.level, seed=args.seed, out=out))


if __name__ == "__main__":
    main()
