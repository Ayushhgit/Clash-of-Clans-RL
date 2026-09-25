"""Phase 25: run an agent, classify every failure, print the dashboard.

    python -m scripts.failure_report --agent heuristic --level L4 --episodes 40
"""
from __future__ import annotations

import argparse

from ai.policies.baselines import HeuristicAgent, RandomAgent
from evaluation.failure_analysis import analyse_run


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
    ap.add_argument("--level", default="L4")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--obs-mode", default="privileged", choices=["privileged", "detected"])
    ap.add_argument("--train-seeds", action="store_true")
    args = ap.parse_args()

    log = analyse_run(build_agent(args.agent, args.checkpoint), level=args.level,
                      episodes=args.episodes, obs_mode=args.obs_mode,
                      train_seeds=args.train_seeds)
    print(log.report())


if __name__ == "__main__":
    main()
