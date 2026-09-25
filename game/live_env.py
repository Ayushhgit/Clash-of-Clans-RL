"""LiveGameEnv (M17): the real game behind the same interface as RaidEnv.

    screenshot -> Perceiver -> obs -> policy -> action -> ActionDecoder -> game

Everything the simulator knows for free (reward, destruction %, termination)
must be *read off the screen* here, which is exactly why the sim-to-game gap
(Phase 22) exists. Where a quantity cannot be read, this env says so rather
than inventing one.

Safety: the loop only clicks inside the configured game window rectangle, it
stops on any unexpected screen state, and `dry_run=True` (the default) never
sends real input.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ai.control.capture import ScreenCapture, SessionRecorder
from ai.control.controller import ActionDecoder, DesktopController, NullController, ScreenGeometry
from ai.perception.pipeline import Perceiver
from simulator.env import XY_BINS


@dataclass
class LiveConfig:
    geometry: ScreenGeometry
    region: tuple | None = None          # capture rect; None = full monitor
    dry_run: bool = True                 # never send input unless explicitly disabled
    max_steps: int = 600
    step_period: float = 0.4             # seconds; matches ticks_per_step=4 at 10 Hz
    record: bool = True
    record_root: str = "data/raw/screenshots"
    stop_states: tuple = ("HOME", "RESULT", "UNKNOWN")
    settle_after_click: float = 0.15
    hud_timeout_frames: int = 3


class LiveGameEnv:
    """`reset()` assumes the game is already on BASE_PREVIEW and presses ATTACK."""

    def __init__(self, perceiver: Perceiver, cfg: LiveConfig):
        self.p = perceiver
        self.cfg = cfg
        self.capture = ScreenCapture(region=cfg.region)
        self.controller = (NullController() if cfg.dry_run
                           else DesktopController(dry_run=False))
        self.decoder = ActionDecoder(cfg.geometry, self.controller)
        self.recorder = SessionRecorder(root=cfg.record_root) if cfg.record else None
        self.steps = 0
        self.last_obs: dict | None = None
        self.prev_destruction = 0.0
        self.prev_loot = 0.0

    # --------------------------------------------------------------- gym api
    def reset(self, seed: int | None = None, level: str | None = None) -> dict:
        obs = self.observe()
        if obs["screen_state"] == "BASE_PREVIEW":
            self.decoder.press("ATTACK_BUTTON")
            time.sleep(1.0)
            obs = self.observe()
        self.steps = 0
        self.prev_destruction = 0.0
        self.prev_loot = 0.0
        return obs

    def step(self, action) -> tuple:
        t0 = time.perf_counter()
        info = self.decoder.execute(action, xy_bins=XY_BINS)
        time.sleep(self.cfg.settle_after_click)

        obs = self.observe()
        reward, terms = self.reward_from_hud(obs)
        self.steps += 1

        done = (obs["screen_state"] in self.cfg.stop_states
                or self.steps >= self.cfg.max_steps)
        info.update(terms)
        info["screen_state"] = obs["screen_state"]
        info["screen_conf"] = obs["screen_conf"]
        if done:
            info.update(self.episode_info(obs))

        # keep the real-time cadence the policy was trained at
        lag = self.cfg.step_period - (time.perf_counter() - t0)
        if lag > 0:
            time.sleep(lag)
        else:
            info["over_budget_ms"] = -lag * 1000.0
        return obs, reward, done, info

    # --------------------------------------------------------------- helpers
    def observe(self) -> dict:
        rgb = self.capture.grab()
        obs = self.p.perceive(rgb)
        if self.recorder is not None:
            self.recorder.add(rgb, obs["screen_state"])
        self.last_obs = obs
        return obs

    def reward_from_hud(self, obs: dict) -> tuple:
        """Only the shaped terms are computable live: destruction and loot come
        from OCR, unit loss does not. The missing term is reported, not faked."""
        hud = obs.get("hud", {})
        dest = hud.get("destruction")
        dest = 0.0 if dest is None else float(np.clip(dest / 100.0, 0, 1))
        gold = hud.get("gold") or 0
        loot = float(gold) / 1e6

        d_dest = max(dest - self.prev_destruction, 0.0)
        d_loot = max(loot - self.prev_loot, 0.0)
        self.prev_destruction, self.prev_loot = dest, loot
        r = d_dest + d_loot
        return r, {"r_dest": d_dest, "r_loot": d_loot,
                   "r_unit": None,  # not observable from pixels
                   "destruction": dest}

    def episode_info(self, obs: dict) -> dict:
        hud = obs.get("hud", {})
        dest = hud.get("destruction")
        dest = 0.0 if dest is None else dest / 100.0
        return {"win": dest >= 0.5, "destruction": float(dest),
                "loot_frac": float(self.prev_loot),
                "town_hall_down": None, "duration": self.steps * self.cfg.step_period,
                "unit_efficiency": float("nan"), "steps": self.steps,
                "seed": -1, "level": "live"}

    # ------------------------------------------------------- strategic loop
    def raid_loop(self, battle_policy, selector=None, n_battles: int = 1) -> list:
        """HOME -> SEARCH -> BASE_PREVIEW -> {NEXT | ATTACK} -> BATTLE -> RESULT.

        `selector.decide(obs)` returns "ATTACK" or "NEXT" (Phase 17). With no
        selector, every base is attacked.
        """
        results = []
        for _ in range(n_battles):
            obs = self.observe()
            while obs["screen_state"] != "BASE_PREVIEW":
                if obs["screen_state"] == "HOME":
                    self.decoder.press("ATTACK_BUTTON")
                elif obs["screen_state"] == "RESULT":
                    self.decoder.press("RETURN_BUTTON")
                time.sleep(1.0)
                obs = self.observe()

            if selector is not None and selector.decide(obs) == "NEXT":
                self.decoder.press("NEXT_BUTTON")
                continue

            obs = self.reset()
            if hasattr(battle_policy, "reset"):
                battle_policy.reset()
            done = False
            while not done:
                obs, r, done, info = self.step(battle_policy.act(obs))
            results.append(info)
        return results
