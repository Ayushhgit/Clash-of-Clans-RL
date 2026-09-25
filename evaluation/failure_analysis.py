"""Failure analysis (Phase 25).

Every failed episode is recorded with enough state to replay it, then
classified into one of the seven failure modes from the brief. The classifier
is a set of explicit rules over measurable quantities -- deliberately not a
model, so a disagreement is a bug in a rule you can read.

    PERCEPTION      the agent acted on entities that were not there (or missed
                    ones that were) -- only detectable in DETECTED mode
    PLANNING        attacked a base whose expected value was negative
    TACTICAL        good entry, bad unit usage: army spent, low destruction
    CONTROL         actions rejected by the environment (invalid deploys)
    EXPLORATION     policy never deployed a meaningful fraction of its army
    REWARD_HACKING  high shaped reward, low destruction
    GENERALIZATION  fails on eval seeds while succeeding on train seeds
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter
from dataclasses import asdict, dataclass, field


@dataclass
class EpisodeRecord:
    seed: int
    level: str
    win: bool
    destruction: float
    loot_frac: float
    duration: float
    units_deployed: int
    army_size: int
    invalid_actions: int
    steps: int
    shaped_return: float
    obs_mode: str = "privileged"
    detector_recall: float | None = None      # DETECTED mode only
    train_seed: bool = False
    train_reference: float | None = None   # win rate of the same agent on train seeds
    actions: list = field(default_factory=list)
    notes: str = ""

    @property
    def deploy_frac(self) -> float:
        return self.units_deployed / max(self.army_size, 1)

    @property
    def invalid_rate(self) -> float:
        return self.invalid_actions / max(self.steps, 1)


def classify(rec: EpisodeRecord) -> str:
    """First matching rule wins; order encodes which cause is more fundamental."""
    if rec.obs_mode == "detected" and rec.detector_recall is not None and rec.detector_recall < 0.7:
        return "PERCEPTION_FAILURE"
    if rec.invalid_rate > 0.25:
        return "CONTROL_FAILURE"
    if rec.deploy_frac < 0.3:
        return "EXPLORATION_FAILURE"
    if rec.shaped_return > 1.0 and rec.destruction < 0.3:
        return "REWARD_HACKING"
    # only claim a generalisation failure when the same agent demonstrably
    # succeeds on training seeds -- otherwise the agent is simply weak, and
    # calling that "generalisation" hides the real cause
    if (not rec.train_seed and rec.destruction < 0.3
            and rec.train_reference is not None and rec.train_reference >= 0.6):
        return "GENERALIZATION_FAILURE"
    if rec.deploy_frac > 0.8 and rec.destruction < 0.5:
        return "TACTICAL_FAILURE"
    return "PLANNING_FAILURE"


class FailureLog:
    def __init__(self, root: str = "experiments/failures"):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list = []

    def add(self, rec: EpisodeRecord) -> str:
        kind = classify(rec)
        self.records.append((kind, rec))
        with (self.root / "failures.jsonl").open("a") as f:
            f.write(json.dumps({"kind": kind, **asdict(rec)}) + "\n")
        return kind

    def summary(self) -> dict:
        counts = Counter(k for k, _ in self.records)
        total = max(len(self.records), 1)
        return {"n": len(self.records),
                "by_kind": dict(counts),
                "share": {k: round(v / total, 3) for k, v in counts.items()}}

    def worst_seeds(self, k: int = 10) -> list:
        return [r.seed for _, r in sorted(self.records, key=lambda t: t[1].destruction)[:k]]

    def report(self) -> str:
        s = self.summary()
        lines = [f"failures: {s['n']}", ""]
        for kind, n in sorted(s["by_kind"].items(), key=lambda t: -t[1]):
            bar = "#" * int(40 * n / max(s["n"], 1))
            lines.append(f"{kind:<24} {n:4d}  {bar}")
        lines += ["", f"worst seeds: {self.worst_seeds()}"]
        return "\n".join(lines)


def analyse_run(agent, level: str = "L3", episodes: int = 50, obs_mode: str = "privileged",
                train_seeds: bool = False, root: str = "experiments/failures",
                train_reference: float | None = None,
                measure_reference: bool = True) -> FailureLog:
    """Run episodes, log every loss, classify it.

    `train_reference` is the agent's win rate on *training* seeds. Without it
    a loss on an unseen base cannot be attributed to generalisation, so it is
    measured first (on a small sample) unless supplied or disabled.
    """
    from simulator.env import DEFAULT_ARMY, EnvConfig, RaidEnv
    from simulator.generator import split_seeds

    (t0, _), (e0, _) = split_seeds()
    seed0 = t0 if train_seeds else e0
    army_size = sum(DEFAULT_ARMY.values())

    if train_reference is None and measure_reference and not train_seeds:
        from evaluation.benchmark import run_episodes

        ref = run_episodes(agent, level=level, episodes=max(episodes // 2, 5),
                           cfg=EnvConfig(level=level, obs_mode=obs_mode),
                           seeds=range(t0, t0 + max(episodes // 2, 5)), name="ref")
        train_reference = ref.win_rate

    env = RaidEnv(EnvConfig(level=level, obs_mode=obs_mode))
    log = FailureLog(root)
    for i in range(episodes):
        obs = env.reset(seed=seed0 + i, level=level)
        if hasattr(agent, "reset"):
            agent.reset()
        done, ret, invalid, actions = False, 0.0, 0, []
        while not done:
            a = agent.act(obs)
            actions.append([int(v) for v in a])
            obs, r, done, info = env.step(a)
            ret += r
            invalid += int(info.get("invalid_action", False))
        if info["win"]:
            continue
        log.add(EpisodeRecord(
            seed=seed0 + i, level=level, win=info["win"],
            destruction=info["destruction"], loot_frac=info["loot_frac"],
            duration=info["duration"], units_deployed=info["units_deployed"],
            army_size=army_size, invalid_actions=invalid, steps=info["steps"],
            shaped_return=ret, obs_mode=obs_mode, train_seed=train_seeds,
            train_reference=train_reference,
            actions=actions[:200],
        ))
    return log


def replay_seed(seed: int, level: str, actions: list, obs_mode: str = "privileged") -> dict:
    """Deterministically re-run a logged episode -- the basis of any bug hunt."""
    from simulator.env import EnvConfig, RaidEnv

    env = RaidEnv(EnvConfig(level=level, obs_mode=obs_mode))
    env.reset(seed=seed, level=level)
    info = {}
    for a in actions:
        _, _, done, info = env.step(a)
        if done:
            break
    return info
