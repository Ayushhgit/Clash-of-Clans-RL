"""Learn which deployment pattern pays, from real raids.

The agent has a handful of attack patterns and one number that says how well a
raid went (loot, read off the result screen). That is a bandit, not an RL
problem: no state to carry between raids, a small discrete action set, and an
expensive noisy reward. Thompson sampling is the right tool -- it explores in
proportion to how uncertain each arm still is, so a pattern that looked bad
once is not written off, and a pattern that keeps winning gets used more.

Reward is loot *captured as a fraction of loot available*, when both can be
read. Raw loot would mostly measure how rich the base was, and the pattern had
nothing to do with that.

Statistics persist to `experiments/strategy_stats.json`, so learning carries
across sessions.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field

import numpy as np

DEFAULT_PATH = pathlib.Path("experiments/strategy_stats.json")


@dataclass
class ArmStats:
    """Normal-inverse-gamma posterior over one pattern's mean reward."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0            # sum of squared deviations, for the variance
    total_reward: float = 0.0

    def update(self, reward: float) -> None:
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (reward - self.mean)
        self.total_reward += reward

    @property
    def var(self) -> float:
        return self.m2 / max(self.n - 1, 1) if self.n > 1 else 0.25


@dataclass
class StrategyLearner:
    arms: list
    prior_mean: float = 0.35        # a mediocre raid, before any evidence
    prior_strength: float = 2.0     # how many raids the prior is worth
    stats: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def __post_init__(self):
        for a in self.arms:
            self.stats.setdefault(a, ArmStats())

    # ------------------------------------------------------------- selection
    def select(self, rng: np.random.Generator | None = None) -> str:
        """Thompson sample: draw a plausible mean for each arm, take the best."""
        rng = rng or np.random.default_rng()
        best, best_draw = self.arms[0], -1e9
        for a in self.arms:
            s = self.stats[a]
            n = s.n + self.prior_strength
            mean = (s.mean * s.n + self.prior_mean * self.prior_strength) / n
            # posterior std of the mean; wide while an arm is barely tried
            sd = float(np.sqrt(max(s.var, 0.02) / n))
            draw = float(rng.normal(mean, sd))
            if draw > best_draw:
                best, best_draw = a, draw
        return best

    def update(self, arm: str, reward: float, meta: dict | None = None) -> None:
        if arm not in self.stats:
            self.stats[arm] = ArmStats()
            if arm not in self.arms:
                self.arms.append(arm)
        self.stats[arm].update(float(reward))
        self.history.append({"arm": arm, "reward": round(float(reward), 4),
                             **(meta or {})})

    # --------------------------------------------------------------- reports
    def table(self) -> str:
        rows = ["  pattern         raids    mean     best-so-far"]
        order = sorted(self.arms, key=lambda a: -self.stats[a].mean)
        for a in order:
            s = self.stats[a]
            rows.append(f"  {a:<14} {s.n:>5}   {s.mean:6.3f}   "
                        f"{'*' if a == order[0] and s.n else ''}")
        return "\n".join(rows)

    def best(self) -> str:
        tried = [a for a in self.arms if self.stats[a].n]
        if not tried:
            return self.arms[0]
        return max(tried, key=lambda a: self.stats[a].mean)

    # ------------------------------------------------------------ persistence
    def save(self, path: pathlib.Path = DEFAULT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "arms": self.arms,
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
            "stats": {k: asdict(v) for k, v in self.stats.items()},
            "history": self.history[-500:],
        }, indent=1))

    @classmethod
    def load(cls, arms: list, path: pathlib.Path = DEFAULT_PATH) -> "StrategyLearner":
        if not path.exists():
            return cls(arms=list(arms))
        d = json.loads(path.read_text())
        learner = cls(arms=list(dict.fromkeys(list(d.get("arms", [])) + list(arms))),
                      prior_mean=d.get("prior_mean", 0.35),
                      prior_strength=d.get("prior_strength", 2.0),
                      history=d.get("history", []))
        for k, v in d.get("stats", {}).items():
            learner.stats[k] = ArmStats(**v)
        return learner


def loot_reward(gained: dict, available: dict | None) -> float | None:
    """Loot captured, as a fraction of what the base held.

    Returns None when *nothing* could be read. That distinction matters: a
    failed OCR and a genuinely empty raid both used to come back as 0.0, so an
    unreadable result screen was recorded as "this tactic scored zero" and the
    bandit learned from a measurement it never made.

    Falls back to a log-scaled absolute number when the available loot could
    not be read, so a raid still teaches something.
    """
    nums = [v for v in (gained or {}).values() if isinstance(v, (int, float))]
    if not nums:
        return None                      # read nothing; do not score the raid
    got = sum(nums)
    if not got:
        return 0.0
    if available:
        have = sum(v for v in available.values() if isinstance(v, (int, float)))
        if have and have > 0:
            return float(min(got / have, 1.5))
    return float(min(np.log10(max(got, 1)) / 6.0, 1.5))


# --------------------------------------------------------------------------
# Learning more than one thing per raid
# --------------------------------------------------------------------------
@dataclass
class FactoredLearner:
    """Thompson sampling over several independent decisions at once.

    A raid is not one choice. It is a pattern, *and* when the heroes go in,
    *and* where the spells land. `StrategyLearner` learns only the first, so
    everything else stayed on a fixed default and never improved.

    Modelling the full cross-product is not an option: six patterns x three
    hero timings x three hero spots x three spell fractions x three delays is
    486 arms, and one arm costs a three-minute raid. Instead each dimension
    keeps its own posterior and all of them are updated with the same raid
    reward. That treats the dimensions as independent when they are not -- a
    late hero suits some patterns better than others -- so it finds a good
    combination rather than provably the best one. With raids this expensive
    that is the right trade, and it is honest about what it does: `table()`
    reports each dimension separately.
    """

    dims: dict = field(default_factory=dict)          # name -> [arm, ...]
    prior_mean: float = 0.35
    prior_strength: float = 2.0
    stats: dict = field(default_factory=dict)         # name -> {arm: ArmStats}
    history: list = field(default_factory=list)

    def __post_init__(self):
        for name, arms in self.dims.items():
            self.stats.setdefault(name, {})
            for a in arms:
                self.stats[name].setdefault(a, ArmStats())

    # ------------------------------------------------------------- selection
    def _draw(self, name: str, arm: str, rng) -> float:
        s = self.stats[name][arm]
        n = s.n + self.prior_strength
        mean = (s.mean * s.n + self.prior_mean * self.prior_strength) / n
        sd = float(np.sqrt(max(s.var, 0.02) / n))
        return float(rng.normal(mean, sd))

    def select(self, rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng()
        return {name: max(arms, key=lambda a: self._draw(name, a, rng))
                for name, arms in self.dims.items()}

    def update(self, choice: dict, reward: float, meta: dict | None = None) -> None:
        for name, arm in choice.items():
            if name not in self.stats:
                self.stats[name] = {}
                self.dims.setdefault(name, [])
            if arm not in self.stats[name]:
                self.stats[name][arm] = ArmStats()
                if arm not in self.dims[name]:
                    self.dims[name].append(arm)
            self.stats[name][arm].update(float(reward))
        self.history.append({"choice": dict(choice),
                             "reward": round(float(reward), 4), **(meta or {})})

    # --------------------------------------------------------------- reports
    def best(self) -> dict:
        out = {}
        for name, arms in self.dims.items():
            tried = [a for a in arms if self.stats[name][a].n]
            out[name] = (max(tried, key=lambda a: self.stats[name][a].mean)
                         if tried else (arms[0] if arms else None))
        return out

    def table(self) -> str:
        rows = [f"  raids: {len(self.history)}"]
        for name, arms in self.dims.items():
            order = sorted(arms, key=lambda a: -self.stats[name][a].mean)
            rows.append(f"  {name}:")
            for a in order:
                s = self.stats[name][a]
                star = " *" if a == order[0] and s.n else ""
                rows.append(f"    {a:<16} n={s.n:<4} mean={s.mean:6.3f}{star}")
        return "\n".join(rows)

    # ------------------------------------------------------------ persistence
    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "dims": self.dims,
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
            "stats": {n: {a: asdict(s) for a, s in d.items()}
                      for n, d in self.stats.items()},
            "history": self.history[-500:],
        }, indent=1))

    @classmethod
    def load(cls, dims: dict, path: pathlib.Path) -> "FactoredLearner":
        if not path.exists():
            return cls(dims={k: list(v) for k, v in dims.items()})
        d = json.loads(path.read_text())
        merged = {k: list(v) for k, v in d.get("dims", {}).items()}
        for k, v in dims.items():          # new dimensions join old stats
            merged[k] = list(dict.fromkeys(list(merged.get(k, [])) + list(v)))
        learner = cls(dims=merged,
                      prior_mean=d.get("prior_mean", 0.35),
                      prior_strength=d.get("prior_strength", 2.0),
                      history=d.get("history", []))
        for n, arms in d.get("stats", {}).items():
            for a, v in arms.items():
                learner.stats.setdefault(n, {})[a] = ArmStats(**v)
        return learner
