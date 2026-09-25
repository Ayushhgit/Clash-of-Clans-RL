"""Benchmark harness (Phase 23).

One function every agent is measured with, so the numbers in the report are
comparable by construction. Metrics follow docs/ROADMAP.md M18.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from simulator.env import EnvConfig, RaidEnv
from simulator.generator import LEVELS, split_seeds


@dataclass
class Metrics:
    agent: str
    level: str
    episodes: int
    win_rate: float
    mean_reward: float
    mean_destruction: float
    mean_loot: float
    th_rate: float
    unit_efficiency: float
    mean_duration: float
    failed_attacks: float
    invalid_action_rate: float
    steps_per_sec: float
    inference_ms: float

    def row(self) -> str:
        return (f"{self.agent:<12} {self.level:<3} "
                f"win {self.win_rate:5.2f}  dest {self.mean_destruction:5.3f}  "
                f"loot {self.mean_loot:5.3f}  R {self.mean_reward:7.3f}  "
                f"eff {self.unit_efficiency:5.3f}  t {self.mean_duration:6.1f}s  "
                f"{self.inference_ms:5.2f} ms/act")


def run_episodes(agent, level: str = "L1", episodes: int = 50,
                 cfg: EnvConfig | None = None, seeds=None,
                 name: str = "agent", record=None) -> Metrics:
    cfg = cfg or EnvConfig()
    cfg.level = level
    env = RaidEnv(cfg)
    if seeds is None:
        lo, hi = split_seeds()[1]
        seeds = range(lo, lo + episodes)
    seeds = list(seeds)[:episodes]

    wins, rewards, dests, loots, ths, effs, durs, invalids = ([] for _ in range(8))
    n_steps = 0
    act_time = 0.0
    t0 = time.perf_counter()

    for seed in seeds:
        obs = env.reset(seed=seed, level=level)
        if hasattr(agent, "reset"):
            agent.reset()
        done, total_r, inv = False, 0.0, 0
        traj = []
        while not done:
            ta = time.perf_counter()
            action = agent.act(obs)
            act_time += time.perf_counter() - ta
            nxt, r, done, info = env.step(action)
            if record is not None:
                traj.append((obs, action, r))
            obs = nxt
            total_r += r
            inv += int(info.get("invalid_action", False))
            n_steps += 1
        wins.append(info["win"])
        rewards.append(total_r)
        dests.append(info["destruction"])
        loots.append(info["loot_frac"])
        ths.append(info["town_hall_down"])
        effs.append(info["unit_efficiency"])
        durs.append(info["duration"])
        invalids.append(inv / max(info["steps"], 1))
        if record is not None:
            record.append({"seed": seed, "info": info, "traj": traj})

    wall = time.perf_counter() - t0
    return Metrics(
        agent=name, level=level, episodes=len(seeds),
        win_rate=float(np.mean(wins)),
        mean_reward=float(np.mean(rewards)),
        mean_destruction=float(np.mean(dests)),
        mean_loot=float(np.mean(loots)),
        th_rate=float(np.mean(ths)),
        unit_efficiency=float(np.mean(effs)),
        mean_duration=float(np.mean(durs)),
        failed_attacks=float(1.0 - np.mean(wins)),
        invalid_action_rate=float(np.mean(invalids)),
        steps_per_sec=n_steps / max(wall, 1e-9),
        inference_ms=1000.0 * act_time / max(n_steps, 1),
    )


def benchmark(agents: dict, levels=None, episodes: int = 30,
              cfg_factory=None) -> list:
    """Cross product of agents x levels. Returns a list of Metrics."""
    levels = levels or LEVELS
    out = []
    for name, agent in agents.items():
        for lv in levels:
            cfg = cfg_factory() if cfg_factory else EnvConfig()
            m = run_episodes(agent, level=lv, episodes=episodes, cfg=cfg, name=name)
            out.append(m)
            print(m.row(), flush=True)
    return out


def to_table(metrics: list, metric: str = "win_rate") -> str:
    """Markdown comparison table, the Phase 23 headline artifact.

    `metric` picks the column: "win_rate" for the destruction objective,
    "mean_loot" for farming -- where a win is worth nothing and a table of
    win rates ranks the agents on something nobody is optimising.
    """
    agents = sorted({m.agent for m in metrics}, key=lambda a: [m.agent for m in metrics].index(a))
    levels = sorted({m.level for m in metrics})
    lut = {(m.agent, m.level): m for m in metrics}
    head = "| agent | " + " | ".join(levels) + " |"
    sep = "|---" * (len(levels) + 1) + "|"
    rows = []
    for a in agents:
        cells = []
        for lv in levels:
            m = lut.get((a, lv))
            if m is None:
                cells.append("-")
            elif metric == "win_rate":
                cells.append(f"{m.win_rate*100:.0f}%")
            else:
                cells.append(f"{getattr(m, metric):.3f}")
        rows.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def save_json(metrics: list, path: str) -> None:
    import json
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(m) for m in metrics], indent=2))
