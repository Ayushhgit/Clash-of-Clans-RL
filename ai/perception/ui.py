"""UI perception (Phase 8): screen-state classifier + resource OCR.

Kept separate from the object detector on purpose. The two tasks have
different failure modes and different data: mixing them means a bad battle
frame can corrupt the screen-state prediction that decides whether the agent
is even allowed to click ATTACK.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.perception.dataset import UI_STATES


class UIClassifier(nn.Module):
    """Tiny CNN over a downscaled frame. Screen states differ globally
    (layout, colour, presence of a timer bar), so 224px is plenty."""

    def __init__(self, n_states: int = len(UI_STATES), width: int = 24):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            _blk(3, w), _blk(w, w * 2), _blk(w * 2, w * 4), _blk(w * 4, w * 8),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(w * 8, 128), nn.SiLU(), nn.Linear(128, n_states))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple:
        p = F.softmax(self(x), dim=-1)
        conf, idx = p.max(dim=-1)
        return [UI_STATES[i] for i in idx.tolist()], conf.tolist()


def _blk(cin: int, cout: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 2, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class DigitOCR(nn.Module):
    """Fixed-width digit classifier over a cropped number strip.

    Game HUD numbers are a single font at a fixed size, so per-character
    classification on segmented glyphs beats a general OCR engine, is ~40x
    faster, and has no external dependency. Segmentation is a vertical
    projection profile -- reliable because the HUD background is flat.
    """

    CHARS = "0123456789,.KMkm "

    def __init__(self, h: int = 24, w: int = 16):
        super().__init__()
        self.h, self.w = h, w
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (h // 4) * (w // 4), 128), nn.SiLU(),
            nn.Linear(128, len(self.CHARS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def read(self, strip) -> str:
        """strip: (H, W) float32 grayscale crop of one number."""
        glyphs = segment_glyphs(strip)
        if not glyphs:
            return ""
        batch = torch.stack([
            torch.from_numpy(_resize_glyph(g, self.h, self.w)).unsqueeze(0) for g in glyphs
        ])
        pred = self(batch).argmax(-1).tolist()
        return "".join(self.CHARS[i] for i in pred).strip()


def segment_glyphs(strip, min_width: int = 2, gap: int = 1) -> list:
    """Split a number strip into glyph images using a vertical ink profile."""
    import numpy as np

    a = np.asarray(strip, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    ink = (a > a.mean()).sum(axis=0) if a.mean() < 0.5 else (a < a.mean()).sum(axis=0)
    on = ink > 0
    out, start, run_gap = [], None, 0
    for i, v in enumerate(on):
        if v and start is None:
            start = i
            run_gap = 0
        elif not v and start is not None:
            run_gap += 1
            if run_gap > gap:
                if i - run_gap - start >= min_width:
                    out.append(a[:, start:i - run_gap])
                start = None
    if start is not None and len(on) - start >= min_width:
        out.append(a[:, start:])
    return out


def _resize_glyph(g, h: int, w: int):
    import numpy as np
    from PIL import Image

    im = Image.fromarray((np.asarray(g) * 255).astype("uint8")).resize((w, h))
    return (np.asarray(im, dtype=np.float32) / 255.0)


def parse_number(text: str) -> int | None:
    """'1,250,000' -> 1250000 ; '1.2M' -> 1200000 ; junk -> None."""
    t = text.replace(",", "").replace(" ", "").strip()
    if not t:
        return None
    mult = 1
    if t[-1] in "kK":
        mult, t = 1_000, t[:-1]
    elif t[-1] in "mM":
        mult, t = 1_000_000, t[:-1]
    try:
        return int(float(t) * mult)
    except ValueError:
        return None
