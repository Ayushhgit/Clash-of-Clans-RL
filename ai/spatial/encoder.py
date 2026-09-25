"""Spatial (set) encoder -- Phase 9.

Detections are an unordered set, not an image, so the encoder is a small
transformer over entity tokens with a padding mask. Self-attention is what
lets the policy reason about relations ("this storage sits behind that air
defense") instead of scoring detections independently.

Ablation hook: `pool="mean"` with `n_layers=0` degrades this to a
DeepSets-style encoder, which is exactly the "- spatial encoder" row of the
Phase 24 ablation table.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from simulator.entities import NUM_CLASSES
from simulator.env import N_GLOBALS, TOKEN_DIM


class SpatialEncoder(nn.Module):
    def __init__(self, d_model: int = 96, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 192, dropout: float = 0.0, n_globals: int = N_GLOBALS):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.class_emb = nn.Embedding(NUM_CLASSES, d_model, padding_idx=0)
        self.feat = nn.Sequential(
            nn.Linear(TOKEN_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            _Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # a learned CLS token aggregates the set; masked mean is concatenated
        # because CLS alone is noisy early in training
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        self.global_mlp = nn.Sequential(
            nn.Linear(n_globals, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.out = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

    def forward(self, tokens: torch.Tensor, class_ids: torch.Tensor,
                token_mask: torch.Tensor, globals_: torch.Tensor) -> tuple:
        """tokens (B,N,D) -> (state (B,d_model), per_token (B,N,d_model))."""
        h = self.feat(tokens) + self.class_emb(class_ids)
        b = h.shape[0]
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)

        pad = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=h.device),
                         token_mask], dim=1)
        key_padding = ~pad                      # True == ignore

        for layer in self.layers:
            h = layer(h, key_padding)
        h = self.norm(h)

        cls_out, tok_out = h[:, 0], h[:, 1:]
        m = token_mask.unsqueeze(-1).float()
        mean = (tok_out * m).sum(1) / m.sum(1).clamp(min=1.0)
        state = self.out(torch.cat([cls_out, mean, self.global_mlp(globals_)], dim=-1))
        return state, tok_out


class _Block(nn.Module):
    """Pre-norm transformer block. Pre-norm because RL gradients are noisy and
    post-norm blocks need a warmup schedule to stay stable."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding, need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))
