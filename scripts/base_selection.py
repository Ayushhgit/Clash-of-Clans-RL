"""Phase 17: train and evaluate the ATTACK-or-NEXT agent.

    python -m scripts.base_selection --episodes 200 --level L3

The label ("was attacking this base worth it?") is a property of the *battle
policy*, so pass the same agent you intend to deploy.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from ai.planning.base_selection import (
    HeuristicSelector,
    collect_outcomes,
    evaluate_selector,
    fit,
)
from ai.policies.baselines import HeuristicAgent, RandomAgent


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
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--data", default="datasets/base_selection.json")
    ap.add_argument("--reuse", action="store_true", help="skip collection")
    args = ap.parse_args()

    if args.reuse and pathlib.Path(args.data).exists():
        rows = json.loads(pathlib.Path(args.data).read_text())
    else:
        agent = build_agent(args.agent, args.checkpoint)
        rows = collect_outcomes(agent, level=args.level, episodes=args.episodes,
                                out=args.data)
    n_train = int(len(rows) * 0.8)
    train, test = rows[:n_train], rows[n_train:]
    print(f"{len(rows)} bases  ->  train {len(train)}  test {len(test)}")

    model = fit(train)
    print("\nheuristic selector:", json.dumps(evaluate_selector(HeuristicSelector(), test), indent=2))
    print("learned selector:  ", json.dumps(evaluate_selector(model, test), indent=2))


if __name__ == "__main__":
    main()
