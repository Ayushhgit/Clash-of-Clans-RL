"""Hierarchical agent (Phase 19 / M15).

Three layers, each with its own decision rate:

    STRATEGIC   once per base      ATTACK or NEXT, and which objective
    TACTICAL    every battle step  which unit, where, when
    CONTROL     every action       mouse/keyboard (see ai/control/)

The layers are deliberately separate objects rather than one network: the
strategic decision is a one-step bandit-like problem with a handful of
features, and the tactical one is a long-horizon MDP over 64 entity tokens.
Training them jointly would mean back-propagating a battle's worth of credit
into a single ATTACK/NEXT logit.

`RaidSession` is the environment that makes the strategic layer measurable:
one episode is a *sequence* of bases under a decision budget, so skipping a
base has a real cost and attacking a hopeless one has a real penalty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai.planning.base_selection import SelectionConfig, base_features
from simulator.env import EnvConfig, RaidEnv


@dataclass
class SessionConfig:
    level: str = "L3"
    budget: int = 20                 # NEXT presses + attacks allowed per session
    max_attacks: int = 5             # army refills between attacks; session ends after this
    selection: SelectionConfig = None

    def __post_init__(self):
        if self.selection is None:
            self.selection = SelectionConfig()


class RaidSession:
    """HOME -> SEARCH -> BASE_PREVIEW -> {NEXT | ATTACK} -> ... episode."""

    def __init__(self, tactical_agent, cfg: SessionConfig | None = None,
                 env_cfg: EnvConfig | None = None):
        self.cfg = cfg or SessionConfig()
        self.tactical = tactical_agent
        self.env = RaidEnv(env_cfg or EnvConfig(level=self.cfg.level))
        self.rng = np.random.default_rng(0)

    def run(self, selector, seed0: int = 10_000_000, n_sessions: int = 10) -> dict:
        """Returns aggregate strategic-layer metrics."""
        totals = {"reward": 0.0, "loot": 0.0, "attacks": 0, "skips": 0,
                  "wins": 0, "sessions": n_sessions}
        seed = seed0
        for _ in range(n_sessions):
            attacks = 0
            for _ in range(self.cfg.budget):
                if attacks >= self.cfg.max_attacks:
                    break
                base = self.env.gen.generate(self.cfg.level, seed=seed)
                seed += 1

                if _decide(selector, base) == "NEXT":
                    totals["reward"] -= self.cfg.selection.search_cost
                    totals["skips"] += 1
                    continue

                info = self._battle(base.seed)
                attacks += 1
                totals["attacks"] += 1
                totals["wins"] += int(info["win"])
                totals["loot"] += info["loot_frac"]
                totals["reward"] += (info["loot_frac"]
                                     - self.cfg.selection.attack_cost
                                     - self.cfg.selection.fail_penalty * (1 - info["win"]))
        totals["reward_per_session"] = totals["reward"] / max(n_sessions, 1)
        totals["win_rate"] = totals["wins"] / max(totals["attacks"], 1)
        totals["skip_rate"] = totals["skips"] / max(totals["skips"] + totals["attacks"], 1)
        return totals

    def _battle(self, seed: int) -> dict:
        obs = self.env.reset(seed=seed, level=self.cfg.level)
        if hasattr(self.tactical, "reset"):
            self.tactical.reset()
        done, info = False, {}
        while not done:
            obs, _, done, info = self.env.step(self.tactical.act(obs))
        return info


def _decide(selector, base) -> str:
    if selector is None:
        return "ATTACK"
    return selector.decide(base)


class HierarchicalAgent:
    """Convenience wrapper: strategic selector + tactical policy in one object.

    Satisfies the strategic interface (`decide(base)`) and the tactical one
    (`act(obs)`), so it drops into both `RaidSession` and `LiveGameEnv`.
    """

    def __init__(self, selector, tactical):
        self.selector = selector
        self.tactical = tactical

    def decide(self, base) -> str:
        return _decide(self.selector, base)

    def reset(self) -> None:
        if hasattr(self.tactical, "reset"):
            self.tactical.reset()

    def act(self, obs):
        return self.tactical.act(obs)

    def explain(self, base) -> dict:
        """What the strategic layer thought -- for the failure dashboard."""
        out = {"features": base_features(base).tolist(), "decision": self.decide(base)}
        if hasattr(self.selector, "expected_value"):
            out.update(self.selector.expected_value(base))
        return out
