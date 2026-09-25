"""Screen capture (Phase 2 / M1).

Captures a window region at a target FPS, stamps every frame with metadata,
and writes a session directory that the annotation tool and the dataset
builders read directly.

    data/raw/screenshots/<session>/000000123.png
    data/raw/screenshots/<session>/meta.jsonl

`mss` is imported lazily so the RL half of the repo runs on a machine with no
screen (CI, a training box) without pulling in a display dependency.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass

import numpy as np

SCREEN_STATES = ["HOME", "SEARCH", "ARMY", "BASE_PREVIEW", "BATTLE",
                 "BATTLE_CONFIRM", "RESULT", "DIALOG", "UNKNOWN"]


@dataclass
class FrameMeta:
    frame_id: int
    timestamp: float          # seconds since session start
    wall_time: float          # unix time, for aligning with input logs
    screen_state: str = "UNKNOWN"
    path: str = ""


class ScreenCapture:
    """Grabs a fixed region. region = (left, top, width, height) in pixels."""

    def __init__(self, region: tuple | None = None, monitor: int = 1):
        self.region = region
        self.monitor = monitor
        self._sct = None

    def _grab_fn(self):
        if self._sct is None:
            try:
                import mss
            except ImportError as e:  # pragma: no cover - depends on host
                raise RuntimeError(
                    "screen capture needs `mss`: pip install mss"
                ) from e
            self._sct = mss.mss()
        return self._sct

    def region_dict(self) -> dict:
        sct = self._grab_fn()
        if self.region is None:
            return sct.monitors[self.monitor]
        left, top, width, height = self.region
        return {"left": left, "top": top, "width": width, "height": height}

    def grab(self) -> np.ndarray:
        """Returns an RGB uint8 array (H, W, 3)."""
        sct = self._grab_fn()
        raw = sct.grab(self.region_dict())
        arr = np.asarray(raw)             # BGRA
        return arr[:, :, [2, 1, 0]].copy()


class WindowCapture:
    """Capture a specific window's own pixels, even when it is not on top.

    `ScreenCapture` grabs a screen *rectangle*, which silently captures
    whatever window happens to be in front -- during the first live test it
    recorded the terminal instead of the game for four frames straight, and the
    classifier dutifully labelled the terminal a RESULT screen. Asking Windows
    to render the target window into a bitmap (PrintWindow) removes that whole
    class of error, and it does not steal focus.

    Windows only. Falls back to ScreenCapture elsewhere.
    """

    def __init__(self, title_match: str, client_only: bool = True,
                 crop_chrome: bool = True):
        self.title_match = title_match
        self.client_only = client_only
        # Google Play Games wraps the game in its own chrome: a title bar and a
        # left icon rail. Those are not the game, and leaving them in shifts
        # every coordinate the geometry file depends on.
        self.crop_chrome = crop_chrome
        self._hwnd = None
        self._viewport = None
        self._sct = None
        enable_dpi_awareness()

    # ------------------------------------------------------------- win32
    def _find(self):
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if self.title_match.lower() in buf.value.lower():
                    found.append(hwnd)
            return True

        user32.EnumWindows(cb, 0)
        if not found:
            raise RuntimeError(f"no visible window matching {self.title_match!r}")
        return found[0]

    def _client_rect(self) -> tuple:
        """(left, top, width, height) of the client area in PHYSICAL pixels."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        if self._hwnd is None or not user32.IsWindow(self._hwnd):
            self._hwnd = self._find()
        cr = wintypes.RECT()
        user32.GetClientRect(self._hwnd, ctypes.byref(cr))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(self._hwnd, ctypes.byref(pt))
        return (pt.x, pt.y, cr.right, cr.bottom)

    def is_foreground(self) -> bool:
        import ctypes

        user32 = ctypes.windll.user32
        if self._hwnd is None:
            self._hwnd = self._find()
        return bool(user32.GetForegroundWindow() == self._hwnd)

    def grab(self) -> np.ndarray:
        """Grab the game's pixels.

        PrintWindow would let this work while the window is occluded, but on a
        DPI-scaled display it renders into a logical-sized bitmap and returns
        the top-left crop of the real frame (measured: 1191x668 of a 1787x1003
        window), losing the game's bottom UI entirely. So this grabs the screen
        region instead -- which means the window has to be visible. Check
        `is_foreground()` if that matters.
        """
        import mss

        left, top, w, h = self._client_rect()
        if self._sct is None:
            self._sct = mss.mss()
        raw = np.asarray(self._sct.grab({"left": left, "top": top,
                                         "width": w, "height": h}))
        arr = raw[:, :, [2, 1, 0]]

        if self.crop_chrome:
            if self._viewport is None:
                self._viewport = find_viewport(arr)
            x0, y0, x1, y1 = self._viewport
            arr = arr[y0:y1, x0:x1]
        return arr.copy()

    def viewport(self) -> tuple:
        """(x0, y0, x1, y1) of the game area inside the client area."""
        if self._viewport is None:
            self.grab()
        return self._viewport

    def client_origin(self) -> tuple:
        """Screen coords of the game viewport's top-left, for the controller.

        Includes the chrome offset, so a click computed in viewport pixels maps
        straight to a screen position by adding this.
        """
        left, top, _, _ = self._client_rect()
        if self.crop_chrome:
            if self._viewport is None:
                self.grab()
            return (left + self._viewport[0], top + self._viewport[1])
        return (left, top)


def enable_dpi_awareness() -> None:
    """Make this process see physical pixels.

    Windows lies to DPI-unaware processes: on a 150%-scaled display it reports
    a 1920x1080 screen as 1280x720 and renders PrintWindow into a bitmap sized
    in those logical units -- so the capture came back as the top-left crop of
    the real frame, with the game's bottom UI missing. Declaring per-monitor
    DPI awareness makes every rect and bitmap physical.
    """
    import ctypes

    # Each call returns falsy on failure rather than raising, so check the
    # result before giving up on the next one -- an early `return` here left
    # the process DPI-unaware and silently re-introduced the cropped capture.
    try:                                   # Windows 10 1703+
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4))           # PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.GetSystemMetrics(0) > 1280:
            return
    except Exception:
        pass
    try:                                   # Windows 8.1+
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def find_viewport(arr: np.ndarray, max_chrome: float = 0.25) -> tuple:
    """Strip the launcher's chrome (title bar + icon rail) from a window grab.

    An earlier version took the longest run of *bright* rows and columns. That
    works on the village screen and fails on menus, whose dark brown panels
    look like chrome -- during a live run it cut the bottom off the frame and
    hid the button the agent was about to click.

    Chrome is better identified by being *uniform*: the rail and title bar are
    a flat dark grey, while game content -- even dark menu art -- varies a lot
    down a column. So measure per-column and per-row standard deviation and
    trim only the leading/trailing bands that are both flat and dark.
    """
    g = arr.mean(axis=2)
    h, w = g.shape

    def trim(axis: int, limit: int) -> tuple:
        std = g.std(axis=axis)
        mean = g.mean(axis=axis)
        flat = (std < 12.0) & (mean < 60.0)
        n = len(flat)
        lo = 0
        while lo < min(limit, n) and flat[lo]:
            lo += 1
        hi = n
        while hi > max(n - limit, 0) and flat[hi - 1]:
            hi -= 1
        return lo, hi

    x0, x1 = trim(0, int(w * max_chrome))
    y0, y1 = trim(1, int(h * max_chrome))
    return (x0, y0, x1, y1)


class SessionRecorder:
    """Writes frames + meta.jsonl. Also the sink for demonstration recording."""

    def __init__(self, root: str = "data/raw/screenshots", name: str | None = None,
                 fps: float = 5.0, downscale: tuple | None = (1280, 720),
                 save_frames: bool = True):
        self.name = name or time.strftime("session_%Y%m%d_%H%M%S")
        self.dir = pathlib.Path(root) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.jsonl"
        self.fps = fps
        self.downscale = downscale
        self.save_frames = save_frames
        self.frame_id = 0
        self.t0 = time.perf_counter()

    def add(self, rgb: np.ndarray, screen_state: str = "UNKNOWN") -> FrameMeta:
        rel = f"{self.frame_id:09d}.png"
        meta = FrameMeta(
            frame_id=self.frame_id,
            timestamp=time.perf_counter() - self.t0,
            wall_time=time.time(),
            screen_state=screen_state,
            path=rel,
        )
        if self.save_frames:
            _write_png(self.dir / rel, self._resize(rgb))
        with self.meta_path.open("a") as f:
            f.write(json.dumps(asdict(meta)) + "\n")
        self.frame_id += 1
        return meta

    def _resize(self, rgb: np.ndarray) -> np.ndarray:
        if self.downscale is None:
            return rgb
        w, h = self.downscale
        if rgb.shape[1] == w and rgb.shape[0] == h:
            return rgb
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))

    def record(self, capture: ScreenCapture, seconds: float,
               state_fn=None, on_frame=None) -> int:
        """Capture for `seconds` at self.fps. `state_fn(rgb) -> str` optionally
        labels each frame (a UI classifier once M2 exists, a hotkey before)."""
        period = 1.0 / self.fps
        end = time.perf_counter() + seconds
        n = 0
        while time.perf_counter() < end:
            t = time.perf_counter()
            rgb = capture.grab()
            state = state_fn(rgb) if state_fn else "UNKNOWN"
            meta = self.add(rgb, state)
            if on_frame:
                on_frame(rgb, meta)
            n += 1
            sleep = period - (time.perf_counter() - t)
            if sleep > 0:
                time.sleep(sleep)
        return n


def _write_png(path: pathlib.Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(path)


def load_session(session_dir: str) -> list:
    """Read meta.jsonl back as a list of dicts."""
    p = pathlib.Path(session_dir) / "meta.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
