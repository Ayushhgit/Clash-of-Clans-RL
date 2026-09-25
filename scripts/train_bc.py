"""M11-M13: generate demonstrations, run behavioural cloning, evaluate the
cloned policy against the demonstrator.

    python -m scripts.train_bc --demos 150 --level L2 --epochs 5
"""
from __future__ import annotations

import argparse

from ai.imitation.bc import train_bc
from ai.imitation.demos import DemoDataset, generate_simulator_demos
from ai.policies.baselines import HeuristicAgent
from ai.policies.neural_agent import NeuralAgent
from evaluation.benchmark import run_episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=150, help="episodes to record (0 = reuse)")
    ap.add_argument("--level", default="L2")
    ap.add_argument("--levels", nargs="*", default=None,
                    help="record a mix of levels instead of just --level")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--root", default="datasets/demos")
    ap.add_argument("--out", default="checkpoints/bc.pt")
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    if args.demos:
        n = generate_simulator_demos(args.demos, level=args.level,
                                     root=args.root, levels=args.levels)
        print(f"recorded {n} demonstration episodes -> {args.root}")
    print(DemoDataset(args.root).action_stats())

    train_bc(root=args.root, epochs=args.epochs, batch_size=args.batch_size, out=args.out)

    print("\n--- cloned policy vs demonstrator ---")
    demo = run_episodes(HeuristicAgent(seed=1), level=args.level,
                        episodes=args.eval_episodes, name="heuristic")
    clone = run_episodes(NeuralAgent(checkpoint=args.out), level=args.level,
                         episodes=args.eval_episodes, name="bc")
    print(demo.row())
    print(clone.row())


if __name__ == "__main__":
    main()
