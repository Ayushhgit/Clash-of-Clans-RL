"""Non-learned baselines (M6, M7).

Both read ONLY the observation dict, never the simulator, so the exact same
code runs against detected (noisy) observations and, later, the live game.
That is what makes them honest baselines in the Phase 23 benchmark table.
"""
from __future__ import annotations

import numpy as np

from simulator.entities import Cls
from simulator.env import ACT_DEPLOY, ACT_WAIT, XY_BINS

_STORAGE = int(Cls.STORAGE)
_TOWN_HALL = int(Cls.TOWN_HALL)
_COLLECTOR = int(Cls.RESOURCE_BUILDING)
# loot carried by each class, from simulator.entities.BUILDINGS
_LOOT_VALUE = {_STORAGE: 1000.0, _COLLECTOR: 250.0, _TOWN_HALL: 200.0}
_DEFENSES = {int(c) for c in (Cls.CANNON, Cls.ARCHER_TOWER, Cls.MORTAR,
                              Cls.WIZARD_TOWER, Cls.AIR_DEFENSE, Cls.TESLA,
                              Cls.INFERNO, Cls.XBOW)}
_AIR_DEFENSE = int(Cls.AIR_DEFENSE)


class RandomAgent:
    """Uniform over the legal action set. The floor of the benchmark."""

    def __init__(self, seed: int = 0, deploy_prob: float = 0.9):
        self.rng = np.random.default_rng(seed)
        self.deploy_prob = deploy_prob

    def reset(self) -> None:
        pass

    def act(self, obs: dict):
        m = obs["action_mask"]
        if not m["type"][ACT_DEPLOY] or self.rng.random() > self.deploy_prob:
            return (ACT_WAIT, 0, 0, 0)
        unit = int(self.rng.choice(np.nonzero(m["unit"])[0]))
        cells = np.nonzero(m["xy"].ravel())[0]
        c = int(self.rng.choice(cells))
        return (ACT_DEPLOY, unit, c // XY_BINS, c % XY_BINS)


class HeuristicAgent:
    """A hand-written attack plan, the target every learned policy must beat.

    Plan:
      1. pick the approach point that maximises loot value reachable per unit
         of defensive coverage;
      2. send the TANK first so defenses lock onto it;
      3. WALL_BREAKERs next, at the same point, to open the compartment;
      4. then damage units in a tight cluster behind the tank;
      5. hold AIR back until air defenses are likely dead (measured, not timed).
    """

    ORDER = [Cls.TANK, Cls.WALL_BREAKER, Cls.DPS, Cls.SPLASH, Cls.RANGED, Cls.AIR]

    def __init__(self, seed: int = 0, cluster_jitter: float = 1.5,
                 deploy_every: int = 2):
        self.rng = np.random.default_rng(seed)
        self.cluster_jitter = cluster_jitter
        self.deploy_every = deploy_every
        self.reset()

    def reset(self) -> None:
        self.entry = None          # (bin_x, bin_y)
        self.step_i = 0
        self.unit_order = list(self.ORDER)

    # ------------------------------------------------------------------ api
    def act(self, obs: dict):
        self.step_i += 1
        m = obs["action_mask"]
        if not m["type"][ACT_DEPLOY] or self.step_i % self.deploy_every:
            return (ACT_WAIT, 0, 0, 0)

        ents = _entities(obs)
        if self.entry is None or not m["xy"][self.entry]:
            self.entry = self._choose_entry(m["xy"], ents)

        unit_idx = self._choose_unit(m["unit"], ents)
        if unit_idx is None:
            return (ACT_WAIT, 0, 0, 0)

        bx, by = self._jitter(self.entry, m["xy"])
        return (ACT_DEPLOY, unit_idx, bx, by)

    # -------------------------------------------------------------- planning
    def _choose_entry(self, xy_mask: np.ndarray, ents: dict):
        """Score every legal cell: close to loot, far from defensive fire."""
        gx, gy = np.meshgrid(
            (np.arange(XY_BINS) + 0.5) / XY_BINS,
            (np.arange(XY_BINS) + 0.5) / XY_BINS,
            indexing="ij",
        )
        loot = ents["loot_xy"]
        dfs = ents["def_xy"]

        if loot.size:
            d_loot = np.sqrt(((gx[..., None] - loot[:, 0]) ** 2
                              + (gy[..., None] - loot[:, 1]) ** 2)).min(axis=-1)
        else:
            d_loot = np.full(gx.shape, 0.5, dtype=np.float32)

        if dfs.size:
            d_def = np.sqrt(((gx[..., None] - dfs[:, 0]) ** 2
                             + (gy[..., None] - dfs[:, 1]) ** 2))
            # coverage = how many defenses can already reach this cell
            coverage = (d_def < ents["def_rng"][None, None, :]).sum(axis=-1)
        else:
            coverage = np.zeros(gx.shape, dtype=np.float32)

        score = -d_loot - 0.08 * coverage
        score = np.where(xy_mask, score, -np.inf)
        flat = int(np.argmax(score))
        return (flat // XY_BINS, flat % XY_BINS)

    def _choose_unit(self, unit_mask: np.ndarray, ents: dict):
        from simulator.entities import UNIT_CLASSES

        air_alive = ents["air_def_alive"]
        for c in self.unit_order:
            i = UNIT_CLASSES.index(c)
            if not unit_mask[i]:
                continue
            if c is Cls.AIR and air_alive:
                continue          # do not feed air units to live air defense
            return i
        # everything preferred is unavailable: take anything legal
        legal = np.nonzero(unit_mask)[0]
        return int(legal[0]) if legal.size else None

    def _jitter(self, entry, xy_mask):
        bx, by = entry
        for _ in range(8):
            jx = int(np.clip(bx + self.rng.integers(-2, 3), 0, XY_BINS - 1))
            jy = int(np.clip(by + self.rng.integers(-2, 3), 0, XY_BINS - 1))
            if xy_mask[jx, jy]:
                return jx, jy
        return bx, by


def _entities(obs: dict) -> dict:
    """Pull the few facts the heuristics need out of the token tensor."""
    mask = obs["token_mask"]
    cls = obs["class_ids"][mask]
    tok = obs["tokens"][mask]
    mine = tok[:, 6] > 0.5
    enemy = ~mine

    # Collectors were missing here, which mattered: they are the shallow,
    # lightly-defended loot a farmer actually wants, and leaving them out made
    # the farming agent chase storages in the core and score *below* the
    # generic heuristic.
    is_loot = enemy & np.isin(cls, [_STORAGE, _COLLECTOR, _TOWN_HALL])
    is_def = enemy & np.isin(cls, list(_DEFENSES))
    return {
        "loot_xy": tok[is_loot][:, 0:2],
        "loot_value": np.array([_LOOT_VALUE.get(int(c), 0.0) for c in cls[is_loot]],
                               dtype=np.float32),
        "def_xy": tok[is_def][:, 0:2],
        "def_rng": tok[is_def][:, 7] * 12.0 / 44.0,   # token range -> map fraction
        "air_def_alive": bool((cls[is_def] == _AIR_DEFENSE).any()) if is_def.any() else False,
    }


class FarmingAgent:
    """Loot per raid, not stars.

    The heuristic above plans a single funnel and commits the whole army to
    breaking into the core -- the right idea if you want 50% destruction. A
    farmer wants the opposite: hit the cheapest loot first, spend as few units
    as possible, and never pay for a compartment it does not need to enter.

    So: rank loot buildings by value discounted by how defended and how deep
    they are, then drop a couple of units next to the best one at a time.
    """

    ORDER = [Cls.DPS, Cls.RANGED, Cls.SPLASH, Cls.TANK, Cls.AIR, Cls.WALL_BREAKER]

    def __init__(self, seed: int = 0, units_per_target: int = 4,
                 deploy_every: int = 2, coverage_penalty: float = 0.35):
        self.rng = np.random.default_rng(seed)
        self.units_per_target = units_per_target
        self.deploy_every = deploy_every
        self.coverage_penalty = coverage_penalty
        self.reset()

    def reset(self) -> None:
        self.step_i = 0
        self.sent_at_target = 0
        self.target = None
        self._done_targets = []

    def act(self, obs: dict):
        self.step_i += 1
        m = obs["action_mask"]
        if not m["type"][ACT_DEPLOY] or self.step_i % self.deploy_every:
            return (ACT_WAIT, 0, 0, 0)

        ents = _entities(obs)
        loot = ents["loot_xy"]
        if loot.size == 0:
            return (ACT_WAIT, 0, 0, 0)

        if self.target is None or self.sent_at_target >= self.units_per_target:
            self.target = self._pick_target(ents)
            self.sent_at_target = 0
            if self.target is None:
                return (ACT_WAIT, 0, 0, 0)

        cell = self._cell_near(self.target, m["xy"])
        if cell is None:
            self.target = None
            return (ACT_WAIT, 0, 0, 0)

        unit = self._pick_unit(m["unit"])
        if unit is None:
            return (ACT_WAIT, 0, 0, 0)
        self.sent_at_target += 1
        return (ACT_DEPLOY, unit, cell[0], cell[1])

    # -------------------------------------------------------------- planning
    def _pick_target(self, ents: dict):
        """Best loot building: valuable, shallow, and not well covered."""
        loot = ents["loot_xy"]
        dfs = ents["def_xy"]
        rng = ents["def_rng"]
        if loot.size == 0:
            return None

        centre = np.array([0.5, 0.5], dtype=np.float32)
        depth = np.linalg.norm(loot - centre, axis=1)      # bigger = closer to edge

        if dfs.size:
            d = np.sqrt(((loot[:, None, :] - dfs[None, :, :]) ** 2).sum(-1))
            coverage = (d < rng[None, :]).sum(axis=1).astype(np.float32)
        else:
            coverage = np.zeros(len(loot), dtype=np.float32)

        value = ents.get("loot_value")
        if value is None or len(value) != len(loot):
            value = np.ones(len(loot), dtype=np.float32)
        # value per unit of risk: a fat storage is worth walking past two
        # cannons for, a collector is not
        score = np.log1p(value) + 2.0 * depth - self.coverage_penalty * coverage
        self._done_targets = getattr(self, "_done_targets", [])
        for t in self._done_targets:                       # do not re-pick
            score[np.linalg.norm(loot - t, axis=1) < 0.02] = -1e9
        best = int(np.argmax(score))
        self._done_targets.append(loot[best])
        return loot[best]

    def _cell_near(self, target, xy_mask: np.ndarray):
        legal = np.argwhere(xy_mask)
        if legal.size == 0:
            return None
        centres = (legal + 0.5) / XY_BINS
        d = np.linalg.norm(centres - target[None, :], axis=1)
        return tuple(int(v) for v in legal[int(np.argmin(d))])

    def _pick_unit(self, unit_mask: np.ndarray):
        from simulator.entities import UNIT_CLASSES

        for c in self.ORDER:
            i = UNIT_CLASSES.index(c)
            if unit_mask[i]:
                return i
        legal = np.nonzero(unit_mask)[0]
        return int(legal[0]) if legal.size else None


class EpsilonGreedyAgent:
    """The heuristic, but with probability `epsilon` it acts at random instead.

    This is the third point on the exploration axis, and it exists to make a
    specific comparison honest: DQN and PPO both explore, so "does exploration
    help here?" needs an answer that is not confounded by also changing the
    learning algorithm. This agent changes *only* the exploration.

    epsilon = 0 is exactly `HeuristicAgent`; epsilon = 1 is exactly
    `RandomAgent`. Anything in between says how much undirected noise a good
    scripted policy can absorb before it degrades -- and in a game where the
    deploy point is the only thing you control, the expectation is that it
    degrades quickly.
    """

    def __init__(self, epsilon: float = 0.1, seed: int = 0):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.greedy = HeuristicAgent(seed=seed)
        self.random = RandomAgent(seed=seed + 1)

    def reset(self) -> None:
        self.greedy.reset()
        if hasattr(self.random, "reset"):
            self.random.reset()

    def act(self, obs: dict):
        if self.rng.random() < self.epsilon:
            return self.random.act(obs)
        return self.greedy.act(obs)
