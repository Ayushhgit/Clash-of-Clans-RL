"""Demonstration dataset (Phase 15 / M11) + behavioural cloning (Phase 16 / M12).

Two sources produce the same record format, so BC code does not care which
one it is training on:

  * `simulator`  -- a scripted/heuristic policy playing the fast sim. Free,
                    unlimited, and useful for validating the whole BC path
                    before spending a human evening on real demonstrations.
  * `human`      -- recorded screen + input from the real game
                    (`scripts/record_demos.py`).

Record format (one .npz per episode):
    tokens (T, N, D)  class_ids (T, N)  token_mask (T, N)  globals (T, G)
    mask_type (T, A)  mask_unit (T, U)  mask_xy (T, X, Y)
    actions (T, 4)    rewards (T,)      meta.json alongside
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

KEYS = ("tokens", "class_ids", "token_mask", "globals",
        "mask_type", "mask_unit", "mask_xy")


class DemoWriter:
    def __init__(self, root: str = "datasets/demos", source: str = "simulator"):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.n = len(list(self.root.glob("*.npz")))

    def write(self, frames: list, actions: list, rewards: list, meta: dict) -> pathlib.Path:
        """frames: list of observation dicts (as RaidEnv emits)."""
        arrs = {k: np.stack([_get(f, k) for f in frames]) for k in KEYS}
        arrs["actions"] = np.asarray(actions, dtype=np.int64)
        arrs["rewards"] = np.asarray(rewards, dtype=np.float32)
        path = self.root / f"{self.source}_{self.n:06d}.npz"
        np.savez_compressed(path, **arrs)
        path.with_suffix(".json").write_text(json.dumps({"source": self.source, **meta}))
        self.n += 1
        return path


def _get(frame: dict, key: str):
    if key in frame:
        return frame[key]
    return frame["action_mask"][key.replace("mask_", "")]


class DemoDataset(Dataset):
    """Flat (state, action) pairs. Episode boundaries are irrelevant for BC."""

    def __init__(self, root: str = "datasets/demos", sources: tuple | None = None,
                 max_episodes: int | None = None):
        files = sorted(pathlib.Path(root).glob("*.npz"))
        if sources:
            files = [f for f in files if f.name.split("_")[0] in sources]
        if max_episodes:
            files = files[:max_episodes]
        if not files:
            raise SystemExit(f"no demonstrations in {root}")

        self.chunks = []
        self.index = []
        for fi, f in enumerate(files):
            d = np.load(f)
            self.chunks.append({k: d[k] for k in (*KEYS, "actions")})
            self.index += [(fi, t) for t in range(len(d["actions"]))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fi, t = self.index[i]
        c = self.chunks[fi]
        obs = {k: torch.as_tensor(c[k][t]) for k in KEYS}
        return obs, torch.as_tensor(c["actions"][t])

    def action_stats(self) -> dict:
        acts = np.concatenate([c["actions"] for c in self.chunks])
        return {"n": len(acts),
                "type_hist": np.bincount(acts[:, 0], minlength=4).tolist(),
                "unit_hist": np.bincount(acts[:, 1], minlength=6).tolist()}


def collate(batch: list) -> tuple:
    obs = {k: torch.stack([b[0][k] for b in batch]) for k in KEYS}
    act = torch.stack([b[1] for b in batch])
    return obs, act


def generate_simulator_demos(n_episodes: int = 200, level: str = "L2",
                             root: str = "datasets/demos", seed0: int = 500_000,
                             levels: list | None = None) -> int:
    """Records the heuristic agent playing, giving BC a dataset without a human.

    These are demonstrations of a *known* policy, so BC quality is measurable:
    a correctly implemented BC run should reproduce the heuristic's win rate.

    `levels` records a *mix* of difficulties, round-robin. A clone trained on
    one level only knows that level's base density, so it is a poor warm start
    for a curriculum that will walk through five of them.
    """
    from ai.policies.baselines import HeuristicAgent
    from simulator.env import EnvConfig, RaidEnv

    pool = list(levels) if levels else [level]
    env = RaidEnv(EnvConfig(level=pool[0]))
    agent = HeuristicAgent(seed=0)
    writer = DemoWriter(root=root, source="simulator")

    for ep in range(n_episodes):
        level = pool[ep % len(pool)]
        obs = env.reset(seed=seed0 + ep, level=level)
        agent.reset()
        frames, actions, rewards = [], [], []
        done = False
        while not done:
            a = agent.act(obs)
            frames.append(obs)
            actions.append(a)
            obs, r, done, info = env.step(a)
            rewards.append(r)
        writer.write(frames, actions, rewards,
                     {"level": level, "seed": seed0 + ep,
                      "win": bool(info["win"]),
                      "destruction": float(info["destruction"])})
    return n_episodes
