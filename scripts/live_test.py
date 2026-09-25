"""Watch the real game and report what the agent perceives. Sends no input.

    python -m scripts.live_test --seconds 20

This is the honest end-to-end check of the perception path against the live
game: capture the window, run whatever models exist, and print what the agent
would see and what it *would* click. `--dry-run` is the default and cannot be
turned off from here on purpose -- driving the game is `game/live_env.py`.

What runs today:
  * screen-state classifier   (trained, checkpoints/ui_classifier.pt)
  * UI geometry               (measured, configs/geometry.json)
  * object detector           (NOT trained -- reports 0 entities, loudly)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from ai.control.capture import ScreenCapture, SessionRecorder, WindowCapture
from ai.perception.ui import UIClassifier


def load_ui(path: str, device: str = "cpu"):
    p = pathlib.Path(path)
    if not p.exists():
        return None, None
    ck = torch.load(p, map_location=device, weights_only=False)
    m = UIClassifier(width=ck.get("width", 24)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def find_window(match: str = "Clash of Clans"):
    """Locate the game window's client rect via PowerShell."""
    import subprocess

    ps = r"""
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WL {
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
}
"@
$p = Get-Process | Where-Object { $_.MainWindowTitle -like '*MATCH*' } | Select-Object -First 1
if ($p) {
  $h = $p.MainWindowHandle
  $cr = New-Object WL+RECT; [void][WL]::GetClientRect($h, [ref]$cr)
  $pt = New-Object WL+POINT; [void][WL]::ClientToScreen($h, [ref]$pt)
  "$($pt.X) $($pt.Y) $($cr.R) $($cr.B)"
}
""".replace("MATCH", match)
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        nums = [int(v) for v in out.split()]
        return tuple(nums) if len(nums) == 4 and nums[2] > 100 else None
    except Exception:
        return None


def scaled_geometry(geo: dict, w: int, h: int) -> dict:
    """configs/geometry.json stores (left, top, width, height); convert to
    (x1, y1, x2, y2) in this capture's pixels while scaling."""
    rw, rh = geo["reference_size"]
    sx, sy = w / rw, h / rh

    def s(r):
        left, top, bw, bh = r
        return [int(left * sx), int(top * sy),
                int((left + bw) * sx), int((top + bh) * sy)]
    return {"buttons": {k: s(v) for k, v in geo["buttons"].items()},
            "army_slots": [s(v) for v in geo["army_slots"]],
            "hud": {k: s(v) for k, v in geo["hud"].items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--window", default="Clash of Clans")
    ap.add_argument("--region", default=None, help="left,top,width,height (overrides window search)")
    ap.add_argument("--ui-checkpoint", default="checkpoints/ui_classifier.pt")
    ap.add_argument("--detector", default="checkpoints/detector_v1.pt")
    ap.add_argument("--record", default=None, help="session name to save frames under")
    args = ap.parse_args()

    # Capture the window's own pixels by default. Screen-region capture records
    # whatever is in front, which in the first live run meant four frames of
    # terminal that the classifier confidently labelled a RESULT screen.
    if args.region:
        region = tuple(int(v) for v in args.region.split(","))
        cap = ScreenCapture(region=region)
        frame_w, frame_h = region[2], region[3]
        print(f"capturing screen region {frame_w}x{frame_h} at ({region[0]},{region[1]})")
    else:
        cap = WindowCapture(args.window)
        probe = cap.grab()
        frame_h, frame_w = probe.shape[:2]
        vx0, vy0, vx1, vy1 = cap.viewport()
        print(f"capturing window {args.window!r}: game viewport {frame_w}x{frame_h} "
              f"(chrome cropped: {vx0},{vy0}-{vx1},{vy1})")

    ui, ui_ck = load_ui(args.ui_checkpoint)
    print("screen-state classifier:",
          f"loaded (val acc {ui_ck.get('accuracy', 0):.2f})" if ui else "MISSING - run scripts.train_ui")
    det_exists = pathlib.Path(args.detector).exists()
    print("object detector:",
          "loaded" if det_exists else "NOT TRAINED - the agent sees zero entities")

    geo = None
    gp = pathlib.Path("configs/geometry.json")
    if gp.exists():
        geo = scaled_geometry(json.loads(gp.read_text()), frame_w, frame_h)
        print(f"geometry: {len(geo['buttons'])} buttons, {len(geo['army_slots'])} army slots")
    print()

    rec = SessionRecorder(name=args.record, fps=args.fps, downscale=None) if args.record else None

    period = 1.0 / args.fps
    end = time.perf_counter() + args.seconds
    counts: dict = {}
    while time.perf_counter() < end:
        t = time.perf_counter()
        rgb = cap.grab()

        state, conf = "UNKNOWN", 0.0
        if ui is not None:
            from PIL import Image

            x = np.asarray(Image.fromarray(rgb).resize((224, 224)), dtype=np.float32) / 255.0
            xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            names, confs = ui.predict(xt)
            state, conf = names[0], confs[0]
        counts[state] = counts.get(state, 0) + 1

        would = "-"
        if geo:
            if state == "HOME":
                would = f"click ATTACK_BUTTON at {_centre(geo['buttons']['ATTACK_BUTTON'])}"
            elif state == "BASE_PREVIEW":
                would = f"decide NEXT/ATTACK; NEXT at {_centre(geo['buttons']['NEXT_BUTTON'])}"
            elif state == "RESULT":
                would = f"click RETURN_BUTTON at {_centre(geo['buttons']['RETURN_BUTTON'])}"
            elif state == "BATTLE":
                would = "deploy units (needs detector; none loaded)"
        print(f"{time.strftime('%H:%M:%S')}  {state:<15} conf {conf:4.2f}   entities 0   would: {would}",
              flush=True)

        if rec:
            rec.add(rgb, state)
        sleep = period - (time.perf_counter() - t)
        if sleep > 0:
            time.sleep(sleep)

    print("\nseen:", json.dumps(counts))
    if rec:
        print("frames saved to", rec.dir)
    print("\nNO INPUT WAS SENT.")


def _centre(box) -> tuple:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


if __name__ == "__main__":
    main()
