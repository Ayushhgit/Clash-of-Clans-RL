"""Factored actor-critic over the ACTION_SPEC action space.

The action is (type, unit, x, y). x and y are sampled autoregressively --
y is conditioned on the chosen x -- so the 32x32 deploy mask can be applied
exactly. Sampling x and y independently would put probability mass on illegal
(x, y) pairs that the mask cannot express, and the policy would spend early
training learning to avoid them.

Memory (Phase 18) sits between encoder and heads and is swappable:
`None`/"none", "stack" (frame stacking in feature space), "gru", or
"transformer". `memory=None` is the "- memory" ablation row.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.memory.temporal import TemporalMemory
from ai.spatial.encoder import SpatialEncoder
from simulator.env import ACT_DEPLOY, N_ACT_TYPES, XY_BINS
from simulator.entities import NUM_ARCHETYPES

NEG = -1e9


class ActorCritic(nn.Module):
    def __init__(self, d_model: int = 96, n_heads: int = 4, n_layers: int = 2,
                 memory: str | None = "gru", hidden: int = 96, trunk: int = 192):
        super().__init__()
        self.encoder = SpatialEncoder(d_model=d_model, n_heads=n_heads,
                                      n_layers=n_layers)
        # Phase 18: none | stack | gru | transformer, all behind one interface
        self.memory_kind = memory or "none"
        self.memory = TemporalMemory(memory, d_model, hidden=hidden)
        self.hidden_size = self.memory.state_size
        feat = self.memory.out_dim

        self.trunk = nn.Sequential(nn.Linear(feat, trunk), nn.GELU())
        self.head_type = _head(trunk, N_ACT_TYPES)
        self.head_unit = _head(trunk, NUM_ARCHETYPES)
        self.head_x = _head(trunk, XY_BINS)
        self.head_y = _head(trunk + XY_BINS, XY_BINS)   # conditioned on x
        self.head_v = nn.Sequential(nn.Linear(trunk, 128), nn.GELU(), nn.Linear(128, 1))
        _orthogonal_init(self)

    # ------------------------------------------------------------------ core
    def features(self, obs: dict, hx: torch.Tensor | None = None) -> tuple:
        state, _ = self.encoder(obs["tokens"], obs["class_ids"],
                                obs["token_mask"], obs["globals"])
        state, hx = self.memory(state, hx)
        return self.trunk(state), hx

    def value(self, obs: dict, hx: torch.Tensor | None = None) -> torch.Tensor:
        z, _ = self.features(obs, hx)
        return self.head_v(z).squeeze(-1)

    # --------------------------------------------------------------- actions
    def forward(self, obs: dict, hx=None, action=None, deterministic: bool = False):
        """Returns (action, logprob, entropy, value, hx)."""
        z, hx = self.features(obs, hx)
        mt, mu, mxy = obs["mask_type"], obs["mask_unit"], obs["mask_xy"]

        lt = self.head_type(z).masked_fill(~mt, NEG)
        lu = self.head_unit(z).masked_fill(~mu, NEG)

        row_ok = mxy.any(dim=2)                       # (B, X) any legal y in row
        lx = self.head_x(z).masked_fill(~row_ok, NEG)

        d_t, d_u, d_x = (torch.distributions.Categorical(logits=l) for l in (lt, lu, lx))
        if action is None:
            a_t = d_t.probs.argmax(-1) if deterministic else d_t.sample()
            a_u = d_u.probs.argmax(-1) if deterministic else d_u.sample()
            a_x = d_x.probs.argmax(-1) if deterministic else d_x.sample()
        else:
            a_t, a_u, a_x = action[:, 0], action[:, 1], action[:, 2]

        x_onehot = F.one_hot(a_x, XY_BINS).float()
        ly = self.head_y(torch.cat([z, x_onehot], dim=-1))
        col_ok = mxy[torch.arange(mxy.shape[0], device=mxy.device), a_x]   # (B, Y)
        ly = ly.masked_fill(~col_ok, NEG)
        d_y = torch.distributions.Categorical(logits=ly)
        if action is None:
            a_y = d_y.probs.argmax(-1) if deterministic else d_y.sample()
        else:
            a_y = action[:, 3]

        # The unit/x/y components only exist when the action type is DEPLOY.
        # On a WAIT step they are sampled but never reach the environment, so
        # including their log-probs in the PPO ratio feeds those heads pure
        # noise scaled by the episode's advantage -- and ~80% of steps are
        # WAIT once the army is spent. Conditioning the density on the action
        # type is what makes the factorisation correct:
        #     p(a) = p(type) * [p(unit) p(x) p(y)]^[type == DEPLOY]
        deploy = (a_t == ACT_DEPLOY).float()
        logp = d_t.log_prob(a_t) + deploy * (
            d_u.log_prob(a_u) + d_x.log_prob(a_x) + d_y.log_prob(a_y)
        )
        entropy = d_t.entropy() + deploy * (
            d_u.entropy() + d_x.entropy() + d_y.entropy()
        )
        act = torch.stack([a_t, a_u, a_x, a_y], dim=-1)
        return act, logp, entropy, self.head_v(z).squeeze(-1), hx

    def initial_state(self, batch: int, device) -> torch.Tensor | None:
        return self.memory.initial_state(batch, device)


def _head(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, out_dim))


def _orthogonal_init(module: nn.Module) -> None:
    """Orthogonal init with small-gain policy heads: the standard PPO recipe.
    Large initial logits make the first updates enormous and kill entropy."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
            nn.init.zeros_(m.bias)
    for head in (module.head_type, module.head_unit, module.head_x, module.head_y):
        last = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
    last_v = [m for m in module.head_v.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.orthogonal_(last_v.weight, gain=1.0)
