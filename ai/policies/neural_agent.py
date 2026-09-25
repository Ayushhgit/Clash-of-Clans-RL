"""Adapter so a trained ActorCritic satisfies the same `act(obs)` interface as
the random and heuristic baselines, and can therefore go through the exact
same benchmark harness."""
from __future__ import annotations

import torch

from ai.policies.actor_critic import ActorCritic
from simulator.vec_env import batch_obs


# Checkpoints written before the Phase 18 memory refactor stored the GRU at
# `gru.*`; it now lives at `memory.cell.*`. Loading those silently with
# strict=False would leave the memory randomly initialised, which looks like a
# trained policy that suddenly plays badly -- so remap, then load strictly.
_LEGACY_KEYS = {"gru.": "memory.cell."}


def _migrate(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        for old, new in _LEGACY_KEYS.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def load_policy(path: str, device: str = "cpu") -> ActorCritic:
    """Loads either a PPO checkpoint or a BC checkpoint -- both store the same
    ActorCritic state dict plus a cfg block."""
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    pol = ActorCritic(d_model=c["d_model"], n_heads=c.get("n_heads", 4),
                      n_layers=c["n_layers"], memory=c["memory"]).to(device)
    missing, unexpected = pol.load_state_dict(_migrate(ck["model"]), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {path} does not match the current ActorCritic; "
            f"missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]}"
        )
    pol.eval()
    return pol


class NeuralAgent:
    def __init__(self, checkpoint: str | None = None, policy=None,
                 device: str = "cpu", deterministic: bool = False):
        assert checkpoint or policy, "pass a checkpoint path or a policy"
        self.device = torch.device(device)
        self.policy = policy if policy is not None else load_policy(checkpoint, device)
        self.policy.eval()
        self.deterministic = deterministic
        self.hx = None

    def reset(self) -> None:
        self.hx = self.policy.initial_state(1, self.device)

    @torch.no_grad()
    def act(self, obs: dict):
        b = {k: torch.as_tensor(v, device=self.device) for k, v in batch_obs([obs]).items()}
        if self.hx is None:
            self.hx = self.policy.initial_state(1, self.device)
        action, _, _, _, self.hx = self.policy(b, self.hx, deterministic=self.deterministic)
        return tuple(int(v) for v in action[0].cpu().numpy())
