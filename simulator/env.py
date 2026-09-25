"""RaidEnv -- the RL interface over the fast simulator.

Implements docs/OBSERVATION_SPEC.md, docs/ACTION_SPEC.md and
docs/REWARD_SPEC.md exactly. `game/live_env.py` must implement the same
`reset`/`step` signature so Phase 22 (sim-to-game transfer) is a one-line swap.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .core import Army, RaidSim, SimConfig
from .entities import (
    CLASS_PRIOR_DPS,
    CLASS_PRIOR_RNG,
    MAX_DPS,
    MAX_RNG,
    NUM_ARCHETYPES,
    UNIT_CLASSES,
    Cls,
)
from .generator import BaseGenerator

# 64 entity slots: units + defenses + scoring buildings always fit; only
# surplus walls are truncated (they are the lowest-priority tokens). Halving
# this from 128 cut PPO update cost ~4x on CPU with no measured win-rate loss.
N_TOKENS = 64
TOKEN_DIM = 16
N_GLOBALS = 12
XY_BINS = 32
N_ACT_TYPES = 4          # WAIT, DEPLOY_UNIT, USE_ABILITY, SELECT_UNIT

ACT_WAIT, ACT_DEPLOY, ACT_ABILITY, ACT_SELECT = range(N_ACT_TYPES)


@dataclass
class RewardWeights:
    """See docs/REWARD_SPEC.md. `th` and the raised `win` weight are the fix
    for the misalignment measured in docs/EXPERIMENTS.md E7: maximising
    destruction % alone pulled the policy away from town-hall kills, which are
    a win condition on their own, so win rate fell while destruction held."""

    loot: float = 1.0
    dest: float = 0.8
    th: float = 0.5          # progress toward destroying the town hall
    unit: float = 0.2
    time: float = 0.05
    win: float = 3.0
    perfect: float = 1.0
    fail: float = 1.0
    efficiency: float = 0.5
    # What the terminal efficiency bonus divides by army spent: "destruction"
    # for a win-seeking agent, "loot" for a farming one.
    efficiency_metric: str = "destruction"

    @staticmethod
    def farming() -> "RewardWeights":
        """Loot per raid. Stars are worth nothing.

        Destruction keeps a small weight only because it is a dense proxy that
        correlates with reaching storages; the loot term is what actually
        decides behaviour. Army spent is charged so the agent prefers cheap
        raids on exposed collectors over full clears that cost the whole camp.
        """
        return RewardWeights(loot=3.0, dest=0.1, th=0.0, unit=0.2, time=0.05,
                             win=0.0, perfect=0.0, fail=0.0, efficiency=0.5,
                             efficiency_metric="loot")


@dataclass
class EnvConfig:
    level: str = "L1"
    seed_range: tuple = (0, 200_000)
    sim: SimConfig = field(default_factory=SimConfig)
    # 10 Hz sim, 2.5 Hz policy. The real game is driven at 5 Hz (ACTION_SPEC
    # section 3); training uses a coarser control rate because it halves the
    # steps per episode, and the policy is rate-agnostic (globals[0] carries
    # elapsed time). Phase 22 measures whether the rate change costs anything.
    ticks_per_step: int = 4
    army_scale: float = 1.0
    obs_mode: str = "privileged"     # or "detected"
    detect_dropout: float = 0.05     # DETECTED mode: missed-entity rate
    detect_jitter: float = 0.004     # DETECTED mode: position noise (map frac)
    reward: RewardWeights = field(default_factory=RewardWeights)
    deploy_clearance: float = 2.0    # min tiles from any building to deploy


# Army is housing-limited, like the real game: the agent cannot brute force a
# base by out-massing it, so unit choice and placement actually matter.
DEFAULT_ARMY = {
    Cls.TANK: 2,
    Cls.DPS: 8,
    Cls.WALL_BREAKER: 4,
    Cls.RANGED: 4,
    Cls.AIR: 2,
    Cls.SPLASH: 2,
}


class RaidEnv:
    """One battle per episode. Base is resampled from the seed range on reset."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.gen = BaseGenerator(map_size=self.cfg.sim.map_size)
        self.rng = np.random.default_rng()
        self.sim: RaidSim | None = None
        self._xy_centers = (np.arange(XY_BINS) + 0.5) / XY_BINS * self.cfg.sim.map_size

    # --------------------------------------------------------------- gym api
    def reset(self, seed: int | None = None, level: str | None = None) -> dict:
        cfg = self.cfg
        if seed is None:
            seed = int(self.rng.integers(*cfg.seed_range))
        self.rng = np.random.default_rng(seed)
        lvl = level or cfg.level
        base = self.gen.generate(lvl, seed=seed)
        army = Army({c: max(1, int(round(k * cfg.army_scale))) for c, k in DEFAULT_ARMY.items()})
        sim_cfg = replace(cfg.sim, seed=seed) if cfg.sim.randomize else cfg.sim
        self.sim = RaidSim(base, army, sim_cfg, rng=np.random.default_rng(seed + 1))
        self.episode_seed, self.episode_level = seed, lvl

        self._prev = self._progress()
        self._deploy_grid_dirty = True
        self.steps = 0
        return self._obs()

    def step(self, action) -> tuple:
        assert self.sim is not None, "call reset() first"
        sim, cfg = self.sim, self.cfg
        a_type, a_unit, a_x, a_y = (int(v) for v in action)

        info: dict = {"invalid_action": False}
        if a_type == ACT_DEPLOY:
            cls = UNIT_CLASSES[a_unit % NUM_ARCHETYPES]
            x, y = self._bin_to_map(a_x, a_y)
            ok = self._valid_cell(a_x, a_y) and sim.deploy(cls, x, y)
            info["invalid_action"] = not ok

        alive_before = int(sim.b_alive.sum())
        for _ in range(cfg.ticks_per_step):
            sim.tick()
        self.steps += 1
        # the deploy grid only depends on living buildings, so it is stale
        # exactly when something was destroyed
        if int(sim.b_alive.sum()) != alive_before:
            self._deploy_grid_dirty = True

        reward, terms = self._reward()
        done = sim.done
        info.update(terms)
        if done:
            info.update(self._episode_info())
        return self._obs(), reward, done, info

    # --------------------------------------------------------------- reward
    def _progress(self) -> dict:
        s = self.sim
        return {
            "loot": s.loot_frac,
            "dest": s.destruction,
            "th": 1.0 - s.th_hp_frac,
            "unit_lost": s.units_lost_hp / s.army_hp_total,
        }

    def _reward(self) -> tuple:
        s, w, cfg = self.sim, self.cfg.reward, self.cfg
        cur = self._progress()
        d_loot = cur["loot"] - self._prev["loot"]
        d_dest = cur["dest"] - self._prev["dest"]
        d_th = cur["th"] - self._prev["th"]
        d_unit = cur["unit_lost"] - self._prev["unit_lost"]
        dt = cfg.sim.dt * cfg.ticks_per_step

        r = (w.loot * d_loot + w.dest * d_dest + w.th * d_th - w.unit * d_unit
             - w.time * dt / cfg.sim.t_max)
        self._prev = cur

        terms = {"r_loot": w.loot * d_loot, "r_dest": w.dest * d_dest,
                 "r_th": w.th * d_th, "r_unit": -w.unit * d_unit}
        if s.done:
            eff = self._unit_efficiency()
            term = (w.win * float(s.win)
                    + w.perfect * float(s.destruction >= 0.999)
                    - w.fail * float(s.destruction < cfg.sim.win_destruction)
                    + w.efficiency * eff)
            r += term
            terms["r_terminal"] = term
        return float(r), terms

    # Below this the agent is not attacking, it is chipping. The floor exists
    # to stop a degenerate strategy: with a denominator floored at ~0, landing
    # one unit that survives and steals 2% of the loot scores *maximum*
    # efficiency. On L4 that scored within 0.02 of committing the whole army
    # and on L5 it strictly won, so the curriculum was walking into a local
    # optimum of doing almost nothing.
    MIN_SPEND = 0.2

    def _unit_efficiency(self) -> float:
        s = self.sim
        spent = max(s.units_lost_hp / s.army_hp_total, self.MIN_SPEND)
        gained = s.loot_frac if self.cfg.reward.efficiency_metric == "loot" else s.destruction
        return float(np.clip(gained / spent, 0.0, 1.0))

    def _episode_info(self) -> dict:
        s = self.sim
        return {
            "win": bool(s.win),
            "destruction": float(s.destruction),
            "loot_frac": float(s.loot_frac),
            "town_hall_down": bool(s.town_hall_down),
            "duration": float(s.t),
            "units_deployed": int(s.n_units),
            "unit_efficiency": self._unit_efficiency(),
            "seed": int(self.episode_seed),
            "level": self.episode_level,
            "steps": int(self.steps),
        }

    # ---------------------------------------------------------- observation
    def _obs(self) -> dict:
        s, cfg = self.sim, self.cfg
        m = cfg.sim.map_size
        detected = cfg.obs_mode == "detected"

        b_idx = np.nonzero(s.b_alive)[0]
        n = s.n_units
        u_idx = np.nonzero(s.u_alive[:n])[0] if n else np.zeros(0, dtype=np.int64)

        # priority: own units > defenses > scoring buildings > walls/obstacles
        prio = np.where(s.b_isdef[b_idx], 1, np.where(s.b_scores[b_idx], 2, 3))
        order = np.argsort(prio, kind="stable")
        b_idx = b_idx[order]

        room = N_TOKENS - u_idx.size
        b_idx = b_idx[:max(room, 0)]

        tokens = np.zeros((N_TOKENS, TOKEN_DIM), dtype=np.float32)
        class_ids = np.zeros(N_TOKENS, dtype=np.int64)
        mask = np.zeros(N_TOKENS, dtype=bool)

        k = 0
        if u_idx.size:
            k = self._fill_units(tokens, class_ids, mask, u_idx, m, detected)
        if b_idx.size:
            self._fill_buildings(tokens, class_ids, mask, b_idx, m, detected, k)

        return {
            "tokens": tokens,
            "class_ids": class_ids,
            "token_mask": mask,
            "globals": self._globals(),
            "action_mask": self.action_mask(),
        }

    def _fill_units(self, tokens, class_ids, mask, u_idx, m, detected) -> int:
        s = self.sim
        k = u_idx.size
        sl = slice(0, k)
        tokens[sl, 0] = s.u_x[u_idx] / m
        tokens[sl, 1] = s.u_y[u_idx] / m
        tokens[sl, 2] = 0.03
        tokens[sl, 3] = 0.03
        tokens[sl, 4] = 1.0 if detected else s.u_hp[u_idx] / np.maximum(s.u_maxhp[u_idx], 1.0)
        tokens[sl, 5] = 1.0
        tokens[sl, 6] = 1.0                       # is_mine
        tokens[sl, 7] = s.u_rng[u_idx] / MAX_RNG
        tokens[sl, 8] = s.u_dps[u_idx] / MAX_DPS
        tokens[sl, 9] = 1.0
        class_ids[sl] = s.u_cls[u_idx]
        mask[sl] = True
        if detected:
            self._corrupt(tokens, class_ids, mask, 0, k)
        return k

    def _fill_buildings(self, tokens, class_ids, mask, b_idx, m, detected, k) -> None:
        s = self.sim
        j = min(k + b_idx.size, N_TOKENS)
        b_idx = b_idx[: j - k]
        sl = slice(k, j)
        tokens[sl, 0] = s.b_x[b_idx] / m
        tokens[sl, 1] = s.b_y[b_idx] / m
        tokens[sl, 2] = 2.0 * s.b_r[b_idx] / m
        tokens[sl, 3] = 2.0 * s.b_r[b_idx] / m
        if detected:
            tokens[sl, 4] = 1.0
            tokens[sl, 7] = [CLASS_PRIOR_RNG[int(c)] / MAX_RNG for c in s.b_cls[b_idx]]
            tokens[sl, 8] = [CLASS_PRIOR_DPS[int(c)] / MAX_DPS for c in s.b_cls[b_idx]]
        else:
            tokens[sl, 4] = s.b_hp[b_idx] / np.maximum(s.b_maxhp[b_idx], 1.0)
            tokens[sl, 7] = s.b_rng[b_idx] / MAX_RNG
            tokens[sl, 8] = s.b_dps[b_idx] / MAX_DPS
        tokens[sl, 5] = 1.0
        tokens[sl, 6] = 0.0
        tokens[sl, 9] = 1.0
        class_ids[sl] = s.b_cls[b_idx]
        mask[sl] = True
        if detected:
            self._corrupt(tokens, class_ids, mask, k, j)

    def _corrupt(self, tokens, class_ids, mask, lo: int, hi: int) -> None:
        """Simulate detector noise so DETECTED-mode training matches Phase 21."""
        cfg = self.cfg
        span = hi - lo
        if span <= 0:
            return
        drop = self.rng.random(span) < cfg.detect_dropout
        idx = np.arange(lo, hi)
        mask[idx[drop]] = False
        class_ids[idx[drop]] = 0
        tokens[idx[drop]] = 0.0
        jitter = self.rng.normal(0.0, cfg.detect_jitter, size=(span, 2)).astype(np.float32)
        tokens[lo:hi, 0:2] = np.clip(tokens[lo:hi, 0:2] + jitter, 0.0, 1.0)
        tokens[lo:hi, 9] = np.clip(self.rng.normal(0.85, 0.1, size=span), 0.2, 1.0)
        tokens[idx[drop], 9] = 0.0

    def _globals(self) -> np.ndarray:
        s, cfg = self.sim, self.cfg
        g = np.zeros(N_GLOBALS, dtype=np.float32)
        g[0] = s.t / cfg.sim.t_max
        g[1] = s.destruction
        g[2] = float(s.town_hall_down)
        for i, c in enumerate(UNIT_CLASSES):
            g[3 + i] = s.army.counts.get(c, 0) / max(DEFAULT_ARMY[c], 1)
        n = s.n_units
        g[9] = (s.alive_unit_count() / n) if n else 0.0
        g[10] = s.loot_frac
        g[11] = 1.0 - s.units_lost_hp / s.army_hp_total
        return g

    # --------------------------------------------------------- action masks
    def action_mask(self) -> dict:
        s = self.sim
        unit_mask = np.array(
            [s.army.counts.get(c, 0) > 0 and s.n_units < s.MAX_UNITS for c in UNIT_CLASSES],
            dtype=bool,
        )
        xy = self._deploy_grid()
        can_deploy = bool(unit_mask.any() and xy.any() and s.can_deploy_any())
        type_mask = np.zeros(N_ACT_TYPES, dtype=bool)
        type_mask[ACT_WAIT] = True
        type_mask[ACT_DEPLOY] = can_deploy
        if not unit_mask.any():
            unit_mask[:] = True          # keep the categorical well-defined
        if not xy.any():
            xy = np.ones_like(xy)
        return {"type": type_mask, "unit": unit_mask, "xy": xy}

    def _deploy_grid(self) -> np.ndarray:
        """32x32 boolean grid: True where a unit may be placed.

        A cell is valid when it is on the map and at least `deploy_clearance`
        tiles away from every living building -- the standard "cannot deploy
        inside the base" rule.
        """
        if not self._deploy_grid_dirty and hasattr(self, "_grid_cache"):
            return self._grid_cache
        s = self.sim
        gx, gy = np.meshgrid(self._xy_centers, self._xy_centers, indexing="ij")
        b = np.nonzero(s.b_alive)[0]
        if b.size:
            d2 = ((gx[..., None] - s.b_x[b]) ** 2 + (gy[..., None] - s.b_y[b]) ** 2)
            clear = (d2 > (s.b_r[b] + self.cfg.deploy_clearance) ** 2).all(axis=-1)
        else:
            clear = np.ones_like(gx, dtype=bool)
        self._grid_cache = clear
        self._deploy_grid_dirty = False
        return clear

    def _valid_cell(self, ax: int, ay: int) -> bool:
        return bool(self._deploy_grid()[ax % XY_BINS, ay % XY_BINS])

    def _bin_to_map(self, ax: int, ay: int) -> tuple:
        return float(self._xy_centers[ax % XY_BINS]), float(self._xy_centers[ay % XY_BINS])

    # -------------------------------------------------------------- helpers
    @staticmethod
    def obs_shapes() -> dict:
        return {
            "tokens": (N_TOKENS, TOKEN_DIM),
            "class_ids": (N_TOKENS,),
            "token_mask": (N_TOKENS,),
            "globals": (N_GLOBALS,),
        }

    @staticmethod
    def action_dims() -> tuple:
        return (N_ACT_TYPES, NUM_ARCHETYPES, XY_BINS, XY_BINS)
