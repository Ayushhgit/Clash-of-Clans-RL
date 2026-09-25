"""Phase 2 / Phase 15: capture gameplay, optionally with human actions.

Screens only (Dataset A/B raw material):
    python -m scripts.record_session --seconds 300 --fps 4

Screens + your mouse/keyboard (Dataset D, behavioural cloning):
    python -m scripts.record_demos --seconds 600

While recording, press 1-5 to stamp the current screen state into meta.jsonl:
    1 HOME   2 SEARCH   3 BASE_PREVIEW   4 BATTLE   5 RESULT

Requires `mss` (capture) and, for hotkey labelling, `pynput`.
"""
from __future__ import annotations

import argparse
import time

from ai.control.capture import ScreenCapture, SessionRecorder

HOTKEYS = {"1": "HOME", "2": "SEARCH", "3": "BASE_PREVIEW", "4": "BATTLE", "5": "RESULT"}


class StateStamp:
    """Holds the state set by the last hotkey press."""

    def __init__(self, initial: str = "UNKNOWN"):
        self.state = initial
        self.listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            print("[warn] pynput not installed: frames will be labelled UNKNOWN")
            return

        def on_press(key):
            ch = getattr(key, "char", None)
            if ch in HOTKEYS:
                self.state = HOTKEYS[ch]
                print(f"  state -> {self.state}", flush=True)

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def stop(self) -> None:
        if self.listener:
            self.listener.stop()


def parse_region(s: str | None):
    if not s:
        return None
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise SystemExit("--region takes left,top,width,height")
    return tuple(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--region", default=None, help="left,top,width,height")
    ap.add_argument("--root", default="data/raw/screenshots")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cap = ScreenCapture(region=parse_region(args.region))
    rec = SessionRecorder(root=args.root, name=args.name, fps=args.fps)
    stamp = StateStamp()
    stamp.start()

    print(f"recording {args.seconds:.0f}s at {args.fps} fps -> {rec.dir}")
    print("hotkeys: " + "  ".join(f"{k}={v}" for k, v in HOTKEYS.items()))
    t0 = time.perf_counter()
    n = rec.record(cap, args.seconds, state_fn=lambda rgb: stamp.state)
    stamp.stop()

    print(f"{n} frames in {time.perf_counter() - t0:.1f}s -> {rec.dir}")
    print("next: python -m annotation.app " + str(rec.dir))


if __name__ == "__main__":
    main()
