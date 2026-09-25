"""Throughput profiler: where the training loop actually spends its time.

    python -m scripts.profile_ppo --sweep

Used to pick the CPU defaults in configs/ppo_small.yaml.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ai.rl.ppo import PPO, PPOConfig


def measure(num_envs: int, rollout: int, layers: int, epochs: int,
            minibatches: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    cfg = PPOConfig(num_envs=num_envs, rollout_steps=rollout, n_layers=layers,
                    epochs=epochs, minibatches=minibatches, total_steps=1,
                    target_kl=None, run_dir="experiments/_profile")
    t = PPO(cfg)
    obs = t.envs.reset()
    hx = t.policy.initial_state(num_envs, t.device)
    done = np.zeros(num_envs, dtype=bool)

    t0 = time.perf_counter()
    obs, hx, done = t.collect(obs, hx, done)
    t1 = time.perf_counter()
    t.update(obs, hx, done)
    t2 = time.perf_counter()
    n = num_envs * rollout
    return {"envs": num_envs, "T": rollout, "layers": layers, "epochs": epochs,
            "mb": minibatches, "threads": threads,
            "collect_s": round(t1 - t0, 2), "update_s": round(t2 - t1, 2),
            "sps": round(n / (t2 - t0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    if not args.sweep:
        print(measure(args.envs, args.rollout, args.layers, args.epochs,
                      args.minibatches, args.threads))
        return
    for envs, rollout in ((16, 64), (32, 32), (64, 32), (64, 16)):
        for layers in (1, 2):
            print(measure(envs, rollout, layers, args.epochs, args.minibatches,
                          args.threads), flush=True)


if __name__ == "__main__":
    main()
