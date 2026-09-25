"""Phase 24: ablation studies.

Trains the same policy with one component removed at a time and evaluates all
variants on the same unseen bases.

    python -m scripts.ablation --steps 200000 --episodes 30

Variants
    full            transformer spatial encoder + GRU memory + curriculum + BC init
    no_memory       memory=None                     (Phase 18 claim)
    mem_stack       frame stacking instead of a GRU (Phase 18 comparison)
    mem_transformer causal attention over the window (Phase 18 comparison)
    no_curriculum   trains directly on the target level (Phase 14 claim)
    no_spatial      n_layers=0, i.e. DeepSets pooling, no attention (Phase 9 claim)
    no_imitation    skips the BC initialisation      (Phase 16 claim)

Each variant is a separate training run, so this is the expensive script in
the repo. `--steps` is per variant.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from ai.policies.neural_agent import NeuralAgent
from ai.rl.ppo import PPO, PPOConfig
from evaluation.benchmark import run_episodes, to_table
from simulator.env import EnvConfig

VARIANTS = {
    "full":          {},
    "no_memory":     {"memory": None},
    "mem_stack":     {"memory": "stack"},
    "mem_transformer": {"memory": "transformer"},
    "no_curriculum": {"curriculum": False},
    "no_spatial":    {"n_layers": 0},
    "no_imitation":  {"bc_init": False},
}


def build_cfg(name: str, steps: int, level: str, seed: int, root: str) -> PPOConfig:
    over = dict(VARIANTS[name])
    over.pop("bc_init", None)
    cfg = PPOConfig(total_steps=steps, level=level, seed=seed,
                    run_dir=f"{root}/{name}", env=EnvConfig(level=level), **over)
    if name == "no_curriculum":
        cfg.level = level
        cfg.env.level = level
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--level", default="L1", help="starting level (curriculum) ")
    ap.add_argument("--eval-levels", nargs="*", default=["L2", "L3", "L4"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--bc-checkpoint", default="checkpoints/bc.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="experiments/ablation")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    metrics, summary = [], {}

    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        cfg = build_cfg(name, args.steps, args.level, args.seed, args.root)
        trainer = PPO(cfg)

        wants_bc = VARIANTS[name].get("bc_init", True)
        if wants_bc and pathlib.Path(args.bc_checkpoint).exists():
            from ai.imitation.bc import bc_to_ppo

            try:
                bc_to_ppo(args.bc_checkpoint, trainer)
            except Exception as e:                      # shape mismatch on ablated nets
                print(f"[{name}] BC init skipped: {e}")
        trainer.train()

        ck = pathlib.Path(cfg.run_dir) / "ppo_final.pt"
        agent = NeuralAgent(checkpoint=str(ck))
        for lv in args.eval_levels:
            m = run_episodes(agent, level=lv, episodes=args.episodes, name=name)
            metrics.append(m)
            print(m.row(), flush=True)
        summary[name] = {m.level: m.win_rate for m in metrics if m.agent == name}

    table = to_table(metrics)
    (root / "ablation.md").write_text(f"# Ablation ({args.steps} steps per variant)\n\n{table}\n")
    (root / "ablation.json").write_text(json.dumps(summary, indent=2))
    print("\n" + table)

    if "full" in summary:
        print("\ndelta vs full (win rate):")
        for name, per_level in summary.items():
            if name == "full":
                continue
            d = {lv: round(per_level[lv] - summary["full"][lv], 3) for lv in per_level}
            print(f"  {name:<15} {d}")


if __name__ == "__main__":
    main()
