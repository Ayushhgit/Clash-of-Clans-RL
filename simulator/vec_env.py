"""Synchronous vector env.

The simulator is numpy-bound and each step is small, so the win here is not
parallelism but batching: one forward pass of the policy for N envs instead of
N forward passes. That is worth ~10x on CPU.
"""
from __future__ import annotations

import numpy as np

from .env import EnvConfig, RaidEnv


class SyncVectorEnv:
    def __init__(self, num_envs: int = 8, cfg: EnvConfig | None = None,
                 seed: int = 0, level: str | None = None):
        self.num_envs = num_envs
        self.envs = [RaidEnv(cfg if cfg is None else _copy_cfg(cfg)) for _ in range(num_envs)]
        self.rng = np.random.default_rng(seed)
        self.set_level(level or (cfg.level if cfg else "L1"))
        self.episode_returns = np.zeros(num_envs, dtype=np.float64)
        self.episode_lens = np.zeros(num_envs, dtype=np.int64)

    # ------------------------------------------------------------------ api
    def set_level(self, level) -> None:
        """Set the level, or a *mix* of levels, for the next episodes.

        A mix is spread across the envs round-robin rather than sampled, so
        every level is represented in every rollout at a fixed ratio -- a
        sampled mix leaves whole updates with no L5 in them at all, and the
        gradient then swings with the draw.

        Takes effect at the next episode boundary.
        """
        self.levels = [level] if isinstance(level, str) else list(level)
        self.level = self.levels[0] if len(self.levels) == 1 else "mix"

    def _level_for(self, i: int) -> str:
        return self.levels[i % len(self.levels)]

    def reset(self) -> dict:
        obs = [e.reset(seed=self._seed(), level=self._level_for(i))
               for i, e in enumerate(self.envs)]
        self.episode_returns[:] = 0.0
        self.episode_lens[:] = 0
        return batch_obs(obs)

    def step(self, actions: np.ndarray) -> tuple:
        obs, rews, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, d, info = env.step(actions[i])
            self.episode_returns[i] += r
            self.episode_lens[i] += 1
            if d:
                info = dict(info)
                info["episode"] = {"r": float(self.episode_returns[i]),
                                   "l": int(self.episode_lens[i])}
                self.episode_returns[i] = 0.0
                self.episode_lens[i] = 0
                o = env.reset(seed=self._seed(), level=self._level_for(i))
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return (batch_obs(obs), np.asarray(rews, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    def _seed(self) -> int:
        lo, hi = self.envs[0].cfg.seed_range
        return int(self.rng.integers(lo, hi))


def batch_obs(obs_list: list) -> dict:
    """Stack per-env observations, flattening the nested action mask."""
    out = {
        "tokens": np.stack([o["tokens"] for o in obs_list]),
        "class_ids": np.stack([o["class_ids"] for o in obs_list]),
        "token_mask": np.stack([o["token_mask"] for o in obs_list]),
        "globals": np.stack([o["globals"] for o in obs_list]),
        "mask_type": np.stack([o["action_mask"]["type"] for o in obs_list]),
        "mask_unit": np.stack([o["action_mask"]["unit"] for o in obs_list]),
        "mask_xy": np.stack([o["action_mask"]["xy"] for o in obs_list]),
    }
    return out


def _copy_cfg(cfg: EnvConfig) -> EnvConfig:
    import copy

    return copy.deepcopy(cfg)
