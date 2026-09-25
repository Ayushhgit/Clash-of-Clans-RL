"""Fast headless raid simulator (Phase 10 / M5).

Structure-of-arrays numpy. No rendering, no assets, no game engine. One tick
is a handful of vectorised numpy ops over (units x buildings) matrices, so a
full 180 s battle costs milliseconds rather than the three real-time minutes
the actual game needs.

Mechanics implemented (docs/GAME_SPEC.md section 3.3):
  movement, targeting, wall blocking, range, dps, splash, ramping dps,
  air/ground validity, destruction accounting, loot, victory/defeat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .entities import BUILDINGS, SCORING_CLASSES, UNITS, Cls

_EPS = 1e-8


@dataclass
class SimConfig:
    map_size: float = 44.0
    dt: float = 0.1                 # sim tick, seconds
    t_max: float = 180.0
    win_destruction: float = 0.5
    deploy_cooldown: float = 0.2    # seconds between two deployments
    wall_block_width: float = 0.9   # how close a wall must be to a path to block it
    wall_lookahead: float = 3.5     # how far ahead a ground unit checks for walls
    unit_splash_mult: float = 0.5   # collateral fraction for unit splash
    def_splash_mult: float = 0.6    # collateral fraction for defense splash
    # Domain randomisation (Phase 20). Perturbs the *mechanics*, not just the
    # pixels: the real game's numbers are estimates, so a policy that only
    # works at the estimated values will not transfer.
    randomize: bool = False
    stat_jitter: float = 0.12       # +/- fraction on hp, dps and range
    seed: int | None = None


@dataclass
class BaseLayout:
    """A base is just parallel arrays of class ids and positions."""

    cls: np.ndarray   # int32[B]
    x: np.ndarray     # float32[B]
    y: np.ndarray     # float32[B]
    seed: int = -1
    difficulty: str = "unknown"

    def __len__(self) -> int:
        return int(self.cls.shape[0])


@dataclass
class Army:
    """Remaining reserve per archetype."""

    counts: dict = field(default_factory=dict)

    def total(self) -> int:
        return int(sum(self.counts.values()))

    def copy(self) -> "Army":
        return Army(dict(self.counts))


class RaidSim:
    """One battle. Create, deploy() units, tick() until done."""

    MAX_UNITS = 80

    def __init__(self, base: BaseLayout, army: Army, cfg: SimConfig | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg or SimConfig()
        self.base = base
        self.army0 = army.copy()
        self.rng = rng or np.random.default_rng(self.cfg.seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        base = self.base
        b = len(base)

        self.b_cls = base.cls.astype(np.int32)
        self.b_x = base.x.astype(np.float32).copy()
        self.b_y = base.y.astype(np.float32).copy()

        stats = [BUILDINGS[Cls(int(c))] for c in self.b_cls]

        def f32(vals):
            return np.asarray(vals, dtype=np.float32)

        self.b_r = f32([s.radius for s in stats])
        self.b_maxhp = f32([s.hp for s in stats])
        self.b_hp = self.b_maxhp.copy()
        self.b_dps = f32([s.dps for s in stats])
        self.b_rng = f32([s.rng for s in stats])
        self.b_minrng = f32([s.min_rng for s in stats])
        self.b_splash = f32([s.splash for s in stats])
        self.b_ramp = f32([s.ramp for s in stats])
        self.b_loot = f32([s.loot for s in stats])
        self.b_air = np.asarray([s.hits_air for s in stats], dtype=bool)
        self.b_ground = np.asarray([s.hits_ground for s in stats], dtype=bool)
        self.b_alive = np.ones(b, dtype=bool)
        self.b_isdef = self.b_dps > 0
        self.b_iswall = self.b_cls == int(Cls.WALL)
        self.b_scores = np.asarray(
            [Cls(int(c)) in SCORING_CLASSES for c in self.b_cls], dtype=bool
        )

        if self.cfg.randomize:
            j = self.cfg.stat_jitter
            self.b_maxhp *= self._jit(b, j)
            self.b_hp = self.b_maxhp.copy()
            self.b_dps *= self._jit(b, j)
            self.b_rng *= self._jit(b, j * 0.5)

        self.b_target = np.full(b, -1, dtype=np.int32)   # which unit a defense shoots
        self.b_ramp_t = np.zeros(b, dtype=np.float32)

        n = self.MAX_UNITS

        def z():
            return np.zeros(n, dtype=np.float32)

        self.u_cls = np.zeros(n, dtype=np.int32)
        self.u_x, self.u_y = z(), z()
        self.u_hp, self.u_maxhp = z(), z()
        self.u_dps, self.u_speed, self.u_rng = z(), z(), z()
        self.u_wallmult, self.u_splash = z(), z()
        self.u_air = np.zeros(n, dtype=bool)
        self.u_prefwall = np.zeros(n, dtype=bool)
        self.u_alive = np.zeros(n, dtype=bool)
        self.u_target = np.full(n, -1, dtype=np.int32)
        self.n_units = 0

        self.army = self.army0.copy()
        self.t = 0.0
        self.loot = 0.0
        self.done = False
        self.last_deploy_t = -1e9
        self.deployed_hp = 0.0        # total max hp ever committed
        self.n_scoring = max(int(self.b_scores.sum()), 1)
        self.total_loot = float(self.b_loot.sum()) + _EPS
        self.events = []

    # ------------------------------------------------------------- properties
    @property
    def destroyed_scoring(self) -> int:
        return int((self.b_scores & ~self.b_alive).sum())

    @property
    def destruction(self) -> float:
        return self.destroyed_scoring / self.n_scoring

    @property
    def town_hall_down(self) -> bool:
        th = self.b_cls == int(Cls.TOWN_HALL)
        return bool(th.any() and not self.b_alive[th].any())

    @property
    def th_hp_frac(self) -> float:
        """Remaining town-hall health, 1.0 = untouched, 0.0 = destroyed.

        Exposed because destroying it is a win condition in its own right, and
        a reward built only on destruction % gives no gradient toward it.
        """
        th = self.b_cls == int(Cls.TOWN_HALL)
        if not th.any():
            return 0.0
        return float((self.b_hp[th] / np.maximum(self.b_maxhp[th], 1.0)).mean())

    @property
    def loot_frac(self) -> float:
        return self.loot / self.total_loot

    @property
    def army_hp_total(self) -> float:
        reserve = sum(UNITS[c].hp * k for c, k in self.army.counts.items())
        return self.deployed_hp + reserve + _EPS

    @property
    def units_lost_hp(self) -> float:
        n = self.n_units
        alive_hp = float(self.u_hp[:n][self.u_alive[:n]].sum())
        return self.deployed_hp - alive_hp

    @property
    def win(self) -> bool:
        return self.destruction >= self.cfg.win_destruction or self.town_hall_down

    def alive_unit_count(self) -> int:
        return int(self.u_alive[: self.n_units].sum())

    # ----------------------------------------------------------------- deploy
    def can_deploy(self, cls: Cls) -> bool:
        if self.done or self.n_units >= self.MAX_UNITS:
            return False
        if self.army.counts.get(cls, 0) <= 0:
            return False
        return (self.t - self.last_deploy_t) >= self.cfg.deploy_cooldown

    def can_deploy_any(self) -> bool:
        """True when at least one deployment is legal right now (cooldown, army)."""
        if self.done or self.n_units >= self.MAX_UNITS or self.army.total() <= 0:
            return False
        return (self.t - self.last_deploy_t) >= self.cfg.deploy_cooldown

    def deploy(self, cls: Cls, x: float, y: float) -> bool:
        """Place one unit of cls at map coords (x, y). Returns success."""
        if not self.can_deploy(cls):
            return False
        s = UNITS[cls]
        i = self.n_units
        j = self.cfg.stat_jitter if self.cfg.randomize else 0.0
        m_hp, m_dps, m_spd = (float(self._jit(1, j)[0]) for _ in range(3))
        self.u_cls[i] = int(cls)
        self.u_x[i], self.u_y[i] = float(x), float(y)
        self.u_hp[i] = self.u_maxhp[i] = s.hp * m_hp
        self.u_dps[i], self.u_speed[i], self.u_rng[i] = s.dps * m_dps, s.speed * m_spd, s.rng
        self.u_wallmult[i], self.u_splash[i] = s.wall_mult, s.splash
        self.u_air[i], self.u_prefwall[i] = s.is_air, s.prefers_walls
        self.u_alive[i] = True
        self.u_target[i] = -1
        self.n_units += 1
        self.army.counts[cls] -= 1
        self.deployed_hp += float(self.u_maxhp[i])
        self.last_deploy_t = self.t
        return True

    def _jit(self, n: int, j: float) -> np.ndarray:
        """Multiplicative jitter in [1-j, 1+j]. j == 0 is exactly 1.0."""
        if j <= 0.0:
            return np.ones(n, dtype=np.float32)
        return self.rng.uniform(1.0 - j, 1.0 + j, size=n).astype(np.float32)

    # ------------------------------------------------------------------- tick
    def tick(self, dt: float | None = None) -> None:
        if self.done:
            return
        dt = self.cfg.dt if dt is None else dt
        self.t += dt

        n = self.n_units
        if n and self.u_alive[:n].any():
            self._units_act(dt, n)
            self._defenses_act(dt, n)
            self._resolve_deaths()

        self._check_end()

    # ------------------------------------------------------------- unit logic
    def _units_act(self, dt: float, n: int) -> None:
        idx = np.nonzero(self.u_alive[:n])[0]
        balive = np.nonzero(self.b_alive)[0]
        if balive.size == 0:
            return

        ux, uy = self.u_x[idx], self.u_y[idx]
        dx = self.b_x[balive][None, :] - ux[:, None]
        dy = self.b_y[balive][None, :] - uy[:, None]
        d2 = dx * dx + dy * dy                            # (U, B)

        iswall = self.b_iswall[balive]
        big = np.float32(1e12)
        # wall breakers score walls first, everyone else scores non-walls first
        pref_wall = self.u_prefwall[idx][:, None]
        is_wall_col = np.broadcast_to(iswall[None, :], d2.shape)
        penalty = np.where(pref_wall, ~is_wall_col, is_wall_col).astype(np.float32) * big
        score = d2 + penalty
        tgt_local = np.argmin(score, axis=1)
        # if every candidate was penalised (e.g. only walls left) fall back to
        # the raw nearest building
        exhausted = score[np.arange(idx.size), tgt_local] >= big
        if exhausted.any():
            tgt_local[exhausted] = np.argmin(d2[exhausted], axis=1)
        tgt = balive[tgt_local]

        ground = ~self.u_air[idx]
        if ground.any() and iswall.any():
            tgt = self._apply_wall_block(idx, ground, ux, uy, tgt, balive, iswall)

        self.u_target[idx] = tgt

        tx, ty = self.b_x[tgt], self.b_y[tgt]
        vx, vy = tx - ux, ty - uy
        dist = np.sqrt(vx * vx + vy * vy) + _EPS
        reach = self.u_rng[idx] + self.b_r[tgt]
        moving = dist > reach

        if moving.any():
            step = np.minimum(self.u_speed[idx] * dt, np.maximum(dist - reach, 0.0))
            step = np.where(moving, step, 0.0)
            self.u_x[idx] = ux + vx / dist * step
            self.u_y[idx] = uy + vy / dist * step

        atk = ~moving
        if atk.any():
            a = np.nonzero(atk)[0]
            hitting = tgt[a]
            mult = np.where(self.b_iswall[hitting], self.u_wallmult[idx[a]], 1.0)
            dmg = self.u_dps[idx[a]] * mult * dt
            np.add.at(self.b_hp, hitting, -dmg)

            sp = self.u_splash[idx[a]]
            for k in np.nonzero(sp > 0)[0]:
                j = hitting[k]
                rad = sp[k] + self.b_r
                near = self.b_alive & (
                    (self.b_x - self.b_x[j]) ** 2 + (self.b_y - self.b_y[j]) ** 2 <= rad * rad
                )
                near[j] = False
                if near.any():
                    self.b_hp[near] -= dmg[k] * self.cfg.unit_splash_mult

    def _apply_wall_block(self, idx, ground, ux, uy, tgt, balive, iswall):
        """A ground unit whose straight path crosses a wall attacks that wall."""
        cfg = self.cfg
        wall_glob = balive[np.nonzero(iswall)[0]]
        wx, wy = self.b_x[wall_glob], self.b_y[wall_glob]

        gi = np.nonzero(ground)[0]
        tx, ty = self.b_x[tgt[gi]], self.b_y[tgt[gi]]
        px, py = ux[gi], uy[gi]
        dirx, diry = tx - px, ty - py
        norm = np.sqrt(dirx * dirx + diry * diry) + _EPS
        dirx, diry = dirx / norm, diry / norm

        rx = wx[None, :] - px[:, None]
        ry = wy[None, :] - py[:, None]
        along = rx * dirx[:, None] + ry * diry[:, None]      # projection on path
        perp = np.abs(rx * diry[:, None] - ry * dirx[:, None])

        blocking = (
            (along > 0.0)
            & (along < np.minimum(norm, cfg.wall_lookahead)[:, None])
            & (perp < cfg.wall_block_width)
        )
        any_block = blocking.any(axis=1)
        if any_block.any():
            along_masked = np.where(blocking, along, np.float32(1e12))
            nearest = np.argmin(along_masked, axis=1)
            rows = np.nonzero(any_block)[0]
            tgt = tgt.copy()
            tgt[gi[rows]] = wall_glob[nearest[rows]]
        return tgt

    # ---------------------------------------------------------- defense logic
    def _defenses_act(self, dt: float, n: int) -> None:
        d_idx = np.nonzero(self.b_alive & self.b_isdef)[0]
        u_idx = np.nonzero(self.u_alive[:n])[0]
        if d_idx.size == 0 or u_idx.size == 0:
            self.b_target[:] = -1
            return

        dx = self.u_x[u_idx][None, :] - self.b_x[d_idx][:, None]
        dy = self.u_y[u_idx][None, :] - self.b_y[d_idx][:, None]
        dist = np.sqrt(dx * dx + dy * dy)                     # (D, U)

        u_is_air = self.u_air[u_idx][None, :]
        can_hit = np.where(u_is_air, self.b_air[d_idx][:, None], self.b_ground[d_idx][:, None])
        in_ring = (dist <= self.b_rng[d_idx][:, None]) & (dist >= self.b_minrng[d_idx][:, None])
        valid = can_hit & in_ring

        has = valid.any(axis=1)
        idle = d_idx[~has]
        if idle.size:
            self.b_target[idle] = -1
            self.b_ramp_t[idle] = 0.0
        if not has.any():
            return

        masked = np.where(valid, dist, np.float32(1e12))
        pick = np.argmin(masked, axis=1)
        rows = np.nonzero(has)[0]
        dg = d_idx[rows]
        tgt_u = u_idx[pick[rows]]

        same = self.b_target[dg] == tgt_u
        self.b_ramp_t[dg] = np.where(same, self.b_ramp_t[dg] + dt, 0.0)
        self.b_target[dg] = tgt_u

        dmg = self.b_dps[dg] * (1.0 + self.b_ramp[dg] * self.b_ramp_t[dg]) * dt
        np.add.at(self.u_hp, tgt_u, -dmg)

        sp = self.b_splash[dg]
        for k in np.nonzero(sp > 0)[0]:
            j = tgt_u[k]
            rad = sp[k]
            near = self.u_alive[:n] & (
                (self.u_x[:n] - self.u_x[j]) ** 2 + (self.u_y[:n] - self.u_y[j]) ** 2 <= rad * rad
            )
            near[j] = False
            if near.any():
                self.u_hp[:n][near] -= dmg[k] * self.cfg.def_splash_mult

    # ------------------------------------------------------------- accounting
    def _resolve_deaths(self) -> None:
        dead_b = self.b_alive & (self.b_hp <= 0.0)
        if dead_b.any():
            self.b_alive[dead_b] = False
            self.b_hp[dead_b] = 0.0
            self.loot += float(self.b_loot[dead_b].sum())
            if (self.b_cls[dead_b] == int(Cls.TOWN_HALL)).any():
                self.events.append((self.t, "TOWN_HALL_DOWN"))
            self.b_target[dead_b] = -1

        n = self.n_units
        dead_u = self.u_alive[:n] & (self.u_hp[:n] <= 0.0)
        if dead_u.any():
            self.u_alive[:n][dead_u] = False
            self.u_hp[:n][dead_u] = 0.0

    def _check_end(self) -> None:
        no_units = self.alive_unit_count() == 0 and self.army.total() == 0
        no_buildings = not self.b_alive[self.b_scores].any()
        if self.t >= self.cfg.t_max or no_units or no_buildings:
            self.done = True

    # ------------------------------------------------------------------ debug
    def summary(self) -> dict:
        return {
            "t": round(self.t, 2),
            "destruction": round(self.destruction, 3),
            "loot_frac": round(self.loot_frac, 3),
            "town_hall_down": self.town_hall_down,
            "win": self.win,
            "units_deployed": self.n_units,
            "units_alive": self.alive_unit_count(),
            "army_left": self.army.total(),
        }
