"""Phase 22 (simulated half): what does perception noise cost?

Runs the same agent on the same bases with privileged observations and with
`detected` observations (missed entities, position jitter, class-prior stats
instead of true HP/DPS). The delta is the part of the sim-to-game gap that is
attributable to perception rather than to control latency or mechanics error.

    python -m scripts.obs_gap --agent heuristic --episodes 25
"""
from __future__ import annotations

import argparse
import json

from ai.policies.baselines import HeuristicAgent, RandomAgent
from evaluation.benchmark import run_episodes
from simulator.env import EnvConfig


def build_agent(name: str, checkpoint: str | None):
    if name == "random":
        return RandomAgent(seed=0)
    if name == "heuristic":
        return HeuristicAgent(seed=0)
    from ai.policies.neural_agent import NeuralAgent

    return NeuralAgent(checkpoint=checkpoint)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--checkpoint", default="experiments/ppo_v2/ppo_latest.pt")
    ap.add_argument("--levels", nargs="*", default=["L2", "L3", "L4"])
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=0.004)
    ap.add_argument("--out", default="experiments/obs_gap.json")
    args = ap.parse_args()

    rows = []
    for lv in args.levels:
        priv = run_episodes(build_agent(args.agent, args.checkpoint), level=lv,
                            episodes=args.episodes, name=f"{args.agent}/privileged",
                            cfg=EnvConfig(level=lv, obs_mode="privileged"))
        det = run_episodes(build_agent(args.agent, args.checkpoint), level=lv,
                           episodes=args.episodes, name=f"{args.agent}/detected",
                           cfg=EnvConfig(level=lv, obs_mode="detected",
                                         detect_dropout=args.dropout,
                                         detect_jitter=args.jitter))
        print(priv.row())
        print(det.row())
        rows.append({"level": lv,
                     "win_privileged": priv.win_rate, "win_detected": det.win_rate,
                     "delta_win": det.win_rate - priv.win_rate,
                     "dest_privileged": priv.mean_destruction,
                     "dest_detected": det.mean_destruction})

    print("\nperception cost (detected - privileged):")
    for r in rows:
        print(f"  {r['level']}  win {r['delta_win']:+.3f}   "
              f"destruction {r['dest_detected'] - r['dest_privileged']:+.3f}")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
