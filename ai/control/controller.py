"""Action decoder + OS-level controller (Phase 19 bottom layer, M1).

Turns an ACTION_SPEC action into mouse/keyboard events with human-like
timing. Two implementations share one interface:

  * `NullController` -- logs intents, moves nothing. Used in tests and dry runs.
  * `DesktopController` -- real input via pynput.

Timing is jittered on purpose: fixed-interval input is both easy for a game to
reject and a distribution mismatch against the human demonstration data.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from simulator.env import ACT_ABILITY, ACT_DEPLOY, ACT_SELECT, ACT_WAIT
from simulator.entities import UNIT_CLASSES


@dataclass
class ScreenGeometry:
    """Maps normalised battle-map coords to screen pixels.

    `battle_rect` is the on-screen quad the map occupies; `army_slots` are the
    unit-card rectangles at the bottom of the battle UI. Both are measured once
    per resolution in Phase 1 and stored in configs/geometry.yaml.
    """

    battle_rect: tuple             # (left, top, width, height)
    army_slots: list               # list of (left, top, width, height), one per archetype
    buttons: dict                  # name -> (left, top, width, height)

    def map_to_screen(self, x: float, y: float) -> tuple:
        left, top, w, h = self.battle_rect
        return (int(left + x * w), int(top + y * h))

    def slot_center(self, unit_idx: int) -> tuple:
        left, top, w, h = self.army_slots[unit_idx]
        return (int(left + w / 2), int(top + h / 2))

    def button_center(self, name: str) -> tuple:
        left, top, w, h = self.buttons[name]
        return (int(left + w / 2), int(top + h / 2))


class NullController:
    """Records intents instead of performing them."""

    def __init__(self):
        self.log: list = []

    def move(self, x: int, y: int) -> None:
        self.log.append(("move", x, y))

    def click(self, x: int, y: int) -> None:
        self.log.append(("click", x, y))

    def key(self, k: str) -> None:
        self.log.append(("key", k))

    def wait(self, ms: float) -> None:
        self.log.append(("wait", ms))


class DesktopController:
    """Real mouse/keyboard. `pynput` is imported lazily."""

    def __init__(self, move_duration: tuple = (0.04, 0.12),
                 click_hold: tuple = (0.04, 0.09), jitter_px: int = 2,
                 dry_run: bool = False):
        self.move_duration = move_duration
        self.click_hold = click_hold
        self.jitter_px = jitter_px
        self.dry_run = dry_run
        self._mouse = None
        self._kb = None

    def _devices(self):
        if self._mouse is None:
            try:
                from pynput.keyboard import Controller as KB
                from pynput.mouse import Button, Controller as MC
            except ImportError as e:  # pragma: no cover - depends on host
                raise RuntimeError("input control needs `pynput`: pip install pynput") from e
            self._mouse, self._kb, self._button = MC(), KB(), Button
        return self._mouse, self._kb

    def move(self, x: int, y: int) -> None:
        x += random.randint(-self.jitter_px, self.jitter_px)
        y += random.randint(-self.jitter_px, self.jitter_px)
        if self.dry_run:
            return
        mouse, _ = self._devices()
        sx, sy = mouse.position
        steps = max(int(random.uniform(*self.move_duration) * 120), 2)
        for i in range(1, steps + 1):
            f = _ease(i / steps)
            mouse.position = (int(sx + (x - sx) * f), int(sy + (y - sy) * f))
            time.sleep(random.uniform(0.002, 0.006))

    def click(self, x: int, y: int) -> None:
        self.move(x, y)
        if self.dry_run:
            return
        mouse, _ = self._devices()
        mouse.press(self._button.left)
        time.sleep(random.uniform(*self.click_hold))
        mouse.release(self._button.left)

    def key(self, k: str) -> None:
        if self.dry_run:
            return
        _, kb = self._devices()
        kb.press(k)
        time.sleep(random.uniform(0.03, 0.08))
        kb.release(k)

    def wait(self, ms: float) -> None:
        time.sleep(ms / 1000.0)


def _ease(t: float) -> float:
    """Ease-in-out so the cursor accelerates and decelerates like a hand."""
    return t * t * (3 - 2 * t)


class ActionDecoder:
    """ACTION_SPEC action tuple -> controller primitives."""

    def __init__(self, geometry: ScreenGeometry, controller, deploy_cooldown_ms: float = 200.0):
        self.geo = geometry
        self.ctl = controller
        self.deploy_cooldown_ms = deploy_cooldown_ms
        self.selected_unit: int | None = None

    def execute(self, action, xy_bins: int = 32) -> dict:
        a_type, a_unit, a_x, a_y = (int(v) for v in action)
        if a_type == ACT_WAIT:
            self.ctl.wait(50)
            return {"did": "wait"}

        if a_type in (ACT_SELECT, ACT_DEPLOY):
            if self.selected_unit != a_unit:
                self.ctl.click(*self.geo.slot_center(a_unit))
                self.selected_unit = a_unit
            if a_type == ACT_SELECT:
                return {"did": "select", "unit": UNIT_CLASSES[a_unit].name}

            nx = (a_x + 0.5) / xy_bins
            ny = (a_y + 0.5) / xy_bins
            self.ctl.click(*self.geo.map_to_screen(nx, ny))
            self.ctl.wait(self.deploy_cooldown_ms)
            return {"did": "deploy", "unit": UNIT_CLASSES[a_unit].name, "xy": (nx, ny)}

        if a_type == ACT_ABILITY:
            self.ctl.key("f")
            return {"did": "ability"}
        return {"did": "noop"}

    # ----------------------------------------------------------- strategic
    def press(self, button: str) -> dict:
        """NEXT / ATTACK / CONFIRM / RETURN / END_BATTLE."""
        self.ctl.click(*self.geo.button_center(button))
        return {"did": "press", "button": button}
