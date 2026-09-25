"""M6 + M7: measure the random and heuristic baselines on every level.

    python -m scripts.baselines --episodes 30
"""
from __future__ import annotations

import argparse

from ai.policies.baselines import HeuristicAgent, RandomAgent
from evaluation.benchmark import benchmark, save_json, to_table
from simulator.generator import LEVELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    ap.add_argument("--out", default="experiments/baselines.json")
    args = ap.parse_args()

    agents = {"random": RandomAgent(seed=0), "heuristic": HeuristicAgent(seed=0)}
    metrics = benchmark(agents, levels=args.levels, episodes=args.episodes)
    save_json(metrics, args.out)
    print()
    print(to_table(metrics))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
