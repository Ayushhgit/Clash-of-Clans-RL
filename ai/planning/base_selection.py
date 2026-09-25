"""Base-selection agent (Phase 17): ATTACK or NEXT?

A separate, much smaller MDP than the battle: one decision per base, reward
= realised loot minus the cost of searching. The interesting property is that
it must learn *expected value* -- a rich base it cannot beat is worth less
than pressing NEXT.

Two implementations:
  * `HeuristicSelector`  -- loot-per-defense ratio with a hard difficulty cut.
  * `ValueSelector`      -- a small MLP trained on outcomes actually observed
                            by the battle policy (`fit`), so it learns the
                            battle policy's own competence, not a generic one.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from simulator.entities import BUILDINGS, DEFENSE_CLASSES, Cls

FEATURES = ["n_entities", "n_defenses", "n_walls", "loot_norm", "hp_norm",
            "defense_density", "loot_per_defense", "th_centrality", "loot_depth"]
N_FEATURES = len(FEATURES)


@dataclass
class SelectionConfig:
    search_cost: float = 0.02
    fail_penalty: float = 0.5
    attack_cost: float = 0.05


def base_features(base, map_size: float = 44.0) -> np.ndarray:
    """Everything computable from a BASE_PREVIEW screen (detections only)."""
    dset = {int(d) for d in DEFENSE_CLASSES}
    cls = base.cls
    n = max(len(cls), 1)
    n_def = int(sum(1 for c in cls if int(c) in dset))
    n_wall = int((cls == int(Cls.WALL)).sum())
    loot = float(sum(BUILDINGS[Cls(int(c))].loot for c in cls))
    hp = float(sum(BUILDINGS[Cls(int(c))].hp for c in cls))

    th = cls == int(Cls.TOWN_HALL)
    if th.any():
        cx, cy = float(base.x[th][0]), float(base.y[th][0])
        centrality = 1.0 - (abs(cx - map_size / 2) + abs(cy - map_size / 2)) / map_size
    else:
        centrality = 0.0

    # How deep inside the base the loot sits: distance from the nearest map
    # edge to each storage, averaged. (The previous "compartments" feature was
    # n_walls/60 clipped to 2, which saturates on every base above L2 and is
    # therefore constant -- a constant feature contributes nothing and made the
    # correlation table report NaN.)
    loot_mask = np.isin(cls, [int(Cls.STORAGE), int(Cls.TOWN_HALL)])
    if loot_mask.any():
        lx, ly = base.x[loot_mask], base.y[loot_mask]
        edge = np.minimum(np.minimum(lx, map_size - lx), np.minimum(ly, map_size - ly))
        loot_depth = float(edge.mean() / (map_size / 2))
    else:
        loot_depth = 0.0
    return np.array([
        n / 200.0,
        n_def / 20.0,
        n_wall / 200.0,
        loot / 10_000.0,
        hp / 100_000.0,
        n_def / n,
        loot / max(n_def, 1) / 1000.0,
        centrality,
        loot_depth,
    ], dtype=np.float32)


class HeuristicSelector:
    """Attack when loot per defense clears a threshold. The baseline to beat."""

    def __init__(self, min_loot_per_defense: float = 0.35, max_defenses: float = 0.6):
        self.min_ratio = min_loot_per_defense
        self.max_def = max_defenses

    def decide(self, base) -> str:
        f = base_features(base)
        ratio, defenses = f[6], f[1]
        return "ATTACK" if (ratio >= self.min_ratio and defenses <= self.max_def) else "NEXT"


class ValueSelector(nn.Module):
    """Predicts E[loot | attack] and P(win | attack) for the *current* battle
    policy, then compares against the cost of searching."""

    def __init__(self, hidden: int = 64, cfg: SelectionConfig | None = None):
        super().__init__()
        self.cfg = cfg or SelectionConfig()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_loot = nn.Linear(hidden, 1)
        self.head_win = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.net(x)
        return self.head_loot(h).squeeze(-1), self.head_win(h).squeeze(-1)

    @torch.no_grad()
    def decide(self, base) -> str:
        x = torch.as_tensor(base_features(base)).unsqueeze(0)
        loot, win_logit = self(x)
        p_win = torch.sigmoid(win_logit).item()
        ev = loot.item() - self.cfg.attack_cost - self.cfg.fail_penalty * (1 - p_win)
        return "ATTACK" if ev > -self.cfg.search_cost else "NEXT"

    def expected_value(self, base) -> dict:
        x = torch.as_tensor(base_features(base)).unsqueeze(0)
        with torch.no_grad():
            loot, win_logit = self(x)
        p_win = torch.sigmoid(win_logit).item()
        return {"loot": loot.item(), "p_win": p_win,
                "ev": loot.item() - self.cfg.attack_cost - self.cfg.fail_penalty * (1 - p_win)}


def collect_outcomes(agent, level: str = "L3", episodes: int = 200,
                     seed0: int = 700_000, out: str = "datasets/base_selection.json") -> list:
    """Play `episodes` bases with `agent`; record (features, loot, win).

    This is the dataset the ValueSelector needs and it must be regenerated
    whenever the battle policy changes -- the label is a property of the
    policy, not of the base.
    """
    from evaluation.benchmark import RaidEnv
    from simulator.env import EnvConfig

    env = RaidEnv(EnvConfig(level=level))
    rows = []
    for i in range(episodes):
        obs = env.reset(seed=seed0 + i, level=level)
        if hasattr(agent, "reset"):
            agent.reset()
        f = base_features(env.sim.base)
        done = False
        while not done:
            obs, r, done, info = env.step(agent.act(obs))
        rows.append({"features": f.tolist(), "loot": info["loot_frac"],
                     "win": float(info["win"]), "level": level,
                     "seed": seed0 + i})
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))
    return rows


def fit(rows: list, epochs: int = 200, lr: float = 1e-3,
        out: str = "checkpoints/base_selector.pt") -> ValueSelector:
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    y_loot = torch.tensor([r["loot"] for r in rows], dtype=torch.float32)
    y_win = torch.tensor([r["win"] for r in rows], dtype=torch.float32)

    model = ValueSelector()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        loot, win_logit = model(x)
        loss = nn.functional.mse_loss(loot, y_loot) + bce(win_logit, y_win)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
    return model


def evaluate_selector(selector, rows: list, cfg: SelectionConfig | None = None) -> dict:
    """Expected reward per decision, versus always-attack and always-next."""
    cfg = cfg or SelectionConfig()
    got, always, never, correct = 0.0, 0.0, 0.0, 0
    for r in rows:
        loot, win = r["loot"], r["win"]
        attack_r = loot - cfg.attack_cost - cfg.fail_penalty * (1 - win)
        next_r = -cfg.search_cost

        f = np.asarray(r["features"], dtype=np.float32)
        choice = _decide_from_features(selector, f)
        got += attack_r if choice == "ATTACK" else next_r
        correct += int((choice == "ATTACK") == (attack_r >= next_r))
        always += attack_r
        never += next_r
    n = max(len(rows), 1)
    return {"policy": got / n, "always_attack": always / n, "always_next": never / n,
            "decision_accuracy": correct / n, "n": n}


def _decide_from_features(selector, f: np.ndarray) -> str:
    if isinstance(selector, ValueSelector):
        with torch.no_grad():
            loot, win_logit = selector(torch.as_tensor(f).unsqueeze(0))
        p_win = torch.sigmoid(win_logit).item()
        c = selector.cfg
        ev = loot.item() - c.attack_cost - c.fail_penalty * (1 - p_win)
        return "ATTACK" if ev > -c.search_cost else "NEXT"
    ratio, defenses = f[6], f[1]
    return "ATTACK" if (ratio >= selector.min_ratio and defenses <= selector.max_def) else "NEXT"
