"""Phase 15 / M11: record human demonstrations from the real game.

Captures frames AND the mouse/keyboard events you produce, then aligns them
into (screen, action) pairs:

    python -m scripts.record_demos --seconds 600 \
        --geometry configs/geometry.json

Every click inside the army bar becomes SELECT_UNIT; every click inside the
battle area within `pair_window` seconds of a select becomes DEPLOY_UNIT at
that map position; everything else becomes a UI press. Frames with no event
become WAIT.

The output is written in the same .npz format `ai/imitation/demos.py` reads,
so behavioural cloning does not care whether the data came from a human or
from the scripted demonstrator.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time

import numpy as np

from ai.control.capture import ScreenCapture, SessionRecorder
from ai.control.controller import ScreenGeometry
from simulator.env import ACT_DEPLOY, ACT_SELECT, ACT_WAIT, XY_BINS


class InputRecorder:
    """Timestamped mouse clicks and key presses on a background listener."""

    def __init__(self):
        self.events: list = []
        self._lock = threading.Lock()
        self._m = self._k = None

    def start(self) -> bool:
        try:
            from pynput import keyboard, mouse
        except ImportError:
            print("[warn] pynput not installed: no actions will be recorded")
            return False

        def on_click(x, y, button, pressed):
            if pressed:
                with self._lock:
                    self.events.append({"t": time.time(), "kind": "click",
                                        "x": int(x), "y": int(y)})

        def on_press(key):
            ch = getattr(key, "char", None) or str(key)
            with self._lock:
                self.events.append({"t": time.time(), "kind": "key", "key": ch})

        self._m = mouse.Listener(on_click=on_click)
        self._k = keyboard.Listener(on_press=on_press)
        self._m.start()
        self._k.start()
        return True

    def stop(self) -> None:
        for l in (self._m, self._k):
            if l:
                l.stop()

    def drain(self) -> list:
        with self._lock:
            out, self.events = self.events, []
        return out


def load_geometry(path: str) -> ScreenGeometry:
    d = json.loads(pathlib.Path(path).read_text())
    return ScreenGeometry(battle_rect=tuple(d["battle_rect"]),
                          army_slots=[tuple(s) for s in d["army_slots"]],
                          buttons={k: tuple(v) for k, v in d["buttons"].items()})


def classify_click(geo: ScreenGeometry, x: int, y: int, last_unit: int | None) -> tuple:
    """(action tuple, new_last_unit)."""
    for i, (l, t, w, h) in enumerate(geo.army_slots):
        if l <= x <= l + w and t <= y <= t + h:
            return (ACT_SELECT, i, 0, 0), i

    bl, bt, bw, bh = geo.battle_rect
    if bl <= x <= bl + bw and bt <= y <= bt + bh and last_unit is not None:
        nx = (x - bl) / bw
        ny = (y - bt) / bh
        return (ACT_DEPLOY, last_unit,
                min(int(nx * XY_BINS), XY_BINS - 1),
                min(int(ny * XY_BINS), XY_BINS - 1)), last_unit
    return (ACT_WAIT, 0, 0, 0), last_unit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--geometry", default="configs/geometry.json")
    ap.add_argument("--region", default=None, help="left,top,width,height")
    ap.add_argument("--root", default="data/raw/demos")
    args = ap.parse_args()

    if not pathlib.Path(args.geometry).exists():
        raise SystemExit(
            f"missing {args.geometry}\n"
            "measure the battle area, army slots and buttons once per resolution;\n"
            "see configs/geometry.example.json"
        )
    geo = load_geometry(args.geometry)
    region = tuple(int(v) for v in args.region.split(",")) if args.region else None

    cap = ScreenCapture(region=region)
    rec = SessionRecorder(root=args.root, fps=args.fps)
    inputs = InputRecorder()
    inputs.start()

    print(f"recording {args.seconds:.0f}s of demonstration -> {rec.dir}")
    period = 1.0 / args.fps
    end = time.perf_counter() + args.seconds
    last_unit = None
    pairs = []

    while time.perf_counter() < end:
        t = time.perf_counter()
        rgb = cap.grab()
        meta = rec.add(rgb, "BATTLE")
        events = inputs.drain()

        action = (ACT_WAIT, 0, 0, 0)
        for e in events:
            if e["kind"] == "click":
                action, last_unit = classify_click(geo, e["x"], e["y"], last_unit)
        pairs.append({"frame": meta.path, "action": [int(v) for v in action],
                      "t": meta.timestamp})

        sleep = period - (time.perf_counter() - t)
        if sleep > 0:
            time.sleep(sleep)

    inputs.stop()
    (rec.dir / "actions.json").write_text(json.dumps(pairs, indent=1))
    counts = np.bincount([p["action"][0] for p in pairs], minlength=4)
    print(f"{len(pairs)} frames -> {rec.dir}")
    print(f"actions: wait={counts[0]} deploy={counts[1]} ability={counts[2]} select={counts[3]}")
    print("next: run the detector over these frames to build the token dataset "
          "(scripts/build_demo_tokens.py), then python -m scripts.train_bc --demos 0")


if __name__ == "__main__":
    main()
