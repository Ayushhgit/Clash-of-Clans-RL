"""Rollout buffer + GAE (Phase 13, hand-implemented).

Stores a fixed (T, N) block of transitions for N parallel envs, then computes
generalised advantage estimates. Recurrent state is stored per timestep so a
GRU policy can be replayed exactly during the update.
"""
from __future__ import annotations

import numpy as np
import torch

OBS_KEYS = ("tokens", "class_ids", "token_mask", "globals",
            "mask_type", "mask_unit", "mask_xy")


class RolloutBuffer:
    def __init__(self, steps: int, num_envs: int, obs_spec: dict,
                 action_dim: int = 4, hidden_size: int = 0, device: str = "cpu"):
        self.T, self.N = steps, num_envs
        self.device = device
        self.hidden_size = hidden_size

        self.obs = {
            k: np.zeros((steps, num_envs, *shape), dtype=dtype)
            for k, (shape, dtype) in obs_spec.items()
        }
        self.actions = np.zeros((steps, num_envs, action_dim), dtype=np.int64)
        self.logprobs = np.zeros((steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((steps, num_envs), dtype=np.float32)
        self.values = np.zeros((steps, num_envs), dtype=np.float32)
        self.hidden = (np.zeros((steps, num_envs, hidden_size), dtype=np.float32)
                       if hidden_size else None)
        self.ptr = 0

    def add(self, obs: dict, action, logprob, reward, done, value, hidden=None) -> None:
        t = self.ptr
        for k in self.obs:
            self.obs[k][t] = obs[k]
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        if self.hidden is not None and hidden is not None:
            self.hidden[t] = hidden
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.T

    def reset(self) -> None:
        self.ptr = 0

    # -------------------------------------------------------------- returns
    def compute_gae(self, last_value: np.ndarray, last_done: np.ndarray,
                    gamma: float = 0.99, lam: float = 0.95) -> tuple:
        """Standard truncated GAE.

        adv_t = delta_t + (gamma * lam) * (1 - done_{t+1}) * adv_{t+1}
        with delta_t = r_t + gamma * (1 - done_{t+1}) * V_{t+1} - V_t
        """
        T, N = self.T, self.N
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - last_done.astype(np.float32)
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + self.values
        return adv, returns

    # ------------------------------------------------------------- batching
    def flat_tensors(self, adv: np.ndarray, returns: np.ndarray) -> dict:
        d = self.device
        flat = {k: torch.as_tensor(v.reshape(self.T * self.N, *v.shape[2:]), device=d)
                for k, v in self.obs.items()}
        flat["actions"] = torch.as_tensor(self.actions.reshape(-1, self.actions.shape[-1]), device=d)
        flat["logprobs"] = torch.as_tensor(self.logprobs.reshape(-1), device=d)
        flat["values"] = torch.as_tensor(self.values.reshape(-1), device=d)
        flat["advantages"] = torch.as_tensor(adv.reshape(-1), device=d)
        flat["returns"] = torch.as_tensor(returns.reshape(-1), device=d)
        if self.hidden is not None:
            flat["hidden"] = torch.as_tensor(self.hidden.reshape(self.T * self.N, -1), device=d)
        return flat


def default_obs_spec(n_tokens: int, token_dim: int, n_globals: int,
                     n_types: int, n_units: int, xy_bins: int) -> dict:
    return {
        "tokens": ((n_tokens, token_dim), np.float32),
        "class_ids": ((n_tokens,), np.int64),
        "token_mask": ((n_tokens,), np.bool_),
        "globals": ((n_globals,), np.float32),
        "mask_type": ((n_types,), np.bool_),
        "mask_unit": ((n_units,), np.bool_),
        "mask_xy": ((xy_bins, xy_bins), np.bool_),
    }
