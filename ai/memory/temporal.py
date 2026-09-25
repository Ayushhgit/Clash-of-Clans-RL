"""Temporal memory (Phase 18).

Four interchangeable modes, so the ablation in the brief -- no memory vs
frame stacking vs GRU vs transformer -- is a config string, not four
codebases:

    none         the encoder output goes straight to the policy heads
    stack        the last K encoder outputs, concatenated (frame stacking,
                 done in feature space rather than pixel space)
    gru          a GRU cell carried across steps
    transformer  causal self-attention over the last K encoder outputs

All four expose the same interface, and all four keep their state in a single
flat tensor `hx` so PPO's rollout buffer stores it without special cases:

    hx: (B, state_size)     state_size == 0 for `none`
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMemory(nn.Module):
    def __init__(self, kind: str | None, d_model: int, hidden: int = 96,
                 window: int = 4, n_heads: int = 4):
        super().__init__()
        self.kind = kind or "none"
        self.d_model = d_model
        self.window = window

        if self.kind == "none":
            self.state_size = 0
            self.out_dim = d_model
        elif self.kind == "stack":
            self.state_size = window * d_model
            self.out_dim = window * d_model
        elif self.kind == "gru":
            self.cell = nn.GRUCell(d_model, hidden)
            self.state_size = hidden
            self.out_dim = hidden
        elif self.kind == "transformer":
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.state_size = window * d_model
            self.out_dim = d_model
        else:
            raise ValueError(f"unknown memory kind: {kind}")

    def initial_state(self, batch: int, device) -> torch.Tensor | None:
        if self.state_size == 0:
            return None
        return torch.zeros(batch, self.state_size, device=device)

    def forward(self, x: torch.Tensor, hx: torch.Tensor | None) -> tuple:
        """x: (B, d_model) encoder output. Returns (features, new_hx)."""
        if self.kind == "none":
            return x, None

        if hx is None:
            hx = self.initial_state(x.shape[0], x.device)

        if self.kind == "gru":
            h = self.cell(x, hx)
            return h, h

        # rolling window: drop the oldest slot, append the newest
        buf = hx.view(x.shape[0], self.window, self.d_model)
        buf = torch.cat([buf[:, 1:], x.unsqueeze(1)], dim=1)
        new_hx = buf.reshape(x.shape[0], -1)

        if self.kind == "stack":
            return new_hx, new_hx

        # transformer: attend from the newest step over the window
        q = self.norm(x).unsqueeze(1)
        kv = self.norm(buf)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        return (x + a.squeeze(1)), new_hx
