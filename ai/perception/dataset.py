"""Dataset A (boxes) and Dataset B (screen states) loaders.

Reads exactly what `annotation/app.py` writes, so there is no intermediate
conversion step to get out of sync.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from annotation.app import CLASSES

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
# BATTLE_CONFIRM (the "End Battle?" dialog) and ARMY (the army overlay) are
# separate states, not noise: each needs a different action from the agent,
# and both were in the very first capture batch.
UI_STATES = ["HOME", "SEARCH", "ARMY", "BASE_PREVIEW", "BATTLE",
             "BATTLE_CONFIRM", "RESULT", "DIALOG"]
UI_TO_ID = {s: i for i, s in enumerate(UI_STATES)}


@dataclass
class Sample:
    image: pathlib.Path
    boxes: np.ndarray      # (K, 4) xyxy in original pixels
    labels: np.ndarray     # (K,) class ids
    w: int
    h: int


def load_labels(root: str) -> list:
    """Every <image>.json under `root`, recursively."""
    out = []
    for jp in sorted(pathlib.Path(root).rglob("*.json")):
        if jp.name == "meta.jsonl":
            continue
        try:
            d = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        if "boxes" not in d:
            continue
        img = jp.with_suffix(".png")
        if not img.exists():
            continue
        boxes = np.array([b["xyxy"] for b in d["boxes"]], dtype=np.float32).reshape(-1, 4)
        labels = np.array([CLASS_TO_ID.get(b["cls"], -1) for b in d["boxes"]], dtype=np.int64)
        keep = labels >= 0
        out.append(Sample(img, boxes[keep], labels[keep], d.get("w", 0), d.get("h", 0)))
    return out


class DetectionDataset(Dataset):
    """Images + boxes, rendered into CenterNet targets.

    Domain randomisation (Phase 20) lives here, not in the model: brightness,
    contrast, hue jitter, gaussian noise, and random resized crops. Geometric
    augmentation is applied to boxes as well.
    """

    def __init__(self, samples: list, img_size: int = 512, stride: int = 4,
                 augment: bool = True, n_classes: int | None = None):
        self.samples = samples
        self.img_size = img_size
        self.stride = stride
        self.augment = augment
        self.n_classes = n_classes or len(CLASSES)
        self.out = img_size // stride

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image

        s = self.samples[i]
        img = Image.open(s.image).convert("RGB")
        w0, h0 = img.size
        boxes = s.boxes.copy()
        labels = s.labels.copy()

        if self.augment:
            img, boxes = _random_crop(img, boxes)
            w0, h0 = img.size
            img = _color_jitter(img)

        img = img.resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment:
            arr = np.clip(arr + np.random.normal(0, 0.02, arr.shape).astype(np.float32), 0, 1)
        x = torch.from_numpy(arr).permute(2, 0, 1)

        if boxes.size:
            sx, sy = self.img_size / max(w0, 1), self.img_size / max(h0, 1)
            boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
        hm, wh, off, mask, ind = _centernet_targets(
            boxes, labels, self.out, self.stride, self.n_classes
        )
        return x, hm, wh, off, mask, ind


def _centernet_targets(boxes, labels, out: int, stride: int, n_classes: int):
    """Anchor-free targets: a gaussian peak per object centre, plus size and
    sub-pixel offset regression at that peak."""
    hm = np.zeros((n_classes, out, out), dtype=np.float32)
    wh = np.zeros((128, 2), dtype=np.float32)
    off = np.zeros((128, 2), dtype=np.float32)
    mask = np.zeros(128, dtype=np.float32)
    ind = np.zeros(128, dtype=np.int64)

    for k, (b, c) in enumerate(zip(boxes[:128], labels[:128])):
        x1, y1, x2, y2 = b / stride
        bw, bh = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < out and 0 <= iy < out):
            continue
        radius = max(int(_gaussian_radius(bh, bw)), 1)
        _draw_gaussian(hm[c], ix, iy, radius)
        wh[k] = (bw, bh)
        off[k] = (cx - ix, cy - iy)
        ind[k] = iy * out + ix
        mask[k] = 1.0

    to_t = torch.from_numpy
    return to_t(hm), to_t(wh), to_t(off), to_t(mask), to_t(ind)


def _gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    a = 1
    b = h + w
    c = w * h * (1 - min_overlap) / (1 + min_overlap)
    return max((b - np.sqrt(max(b ** 2 - 4 * a * c, 0))) / 2, 1.0)


def _draw_gaussian(hm: np.ndarray, cx: int, cy: int, radius: int) -> None:
    d = 2 * radius + 1
    sigma = d / 6.0
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)

    h, w = hm.shape
    l, r = min(cx, radius), min(w - cx, radius + 1)
    t, b = min(cy, radius), min(h - cy, radius + 1)
    np.maximum(hm[cy - t:cy + b, cx - l:cx + r],
               g[radius - t:radius + b, radius - l:radius + r],
               out=hm[cy - t:cy + b, cx - l:cx + r])


def _random_crop(img, boxes: np.ndarray, scale=(0.7, 1.0)):
    import random

    w, h = img.size
    s = random.uniform(*scale)
    nw, nh = int(w * s), int(h * s)
    x0 = random.randint(0, w - nw)
    y0 = random.randint(0, h - nh)
    img = img.crop((x0, y0, x0 + nw, y0 + nh))
    if boxes.size:
        boxes = boxes - np.array([x0, y0, x0, y0], dtype=np.float32)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, nw)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, nh)
    return img, boxes


def _color_jitter(img):
    import random

    from PIL import ImageEnhance

    for enh, rng in ((ImageEnhance.Brightness, (0.7, 1.3)),
                     (ImageEnhance.Contrast, (0.7, 1.3)),
                     (ImageEnhance.Color, (0.6, 1.4))):
        img = enh(img).enhance(random.uniform(*rng))
    return img


class UIStateDataset(Dataset):
    """Dataset B: frames labelled with a screen state, read from meta.jsonl."""

    def __init__(self, session_dirs: list, img_size: int = 224, augment: bool = True):
        from ai.control.capture import load_session

        self.items = []
        for d in session_dirs:
            root = pathlib.Path(d)
            for m in load_session(d):
                if m.get("screen_state") in UI_TO_ID:
                    self.items.append((root / m["path"], UI_TO_ID[m["screen_state"]]))
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        path, y = self.items[i]
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        if self.augment:
            img = _color_jitter(img)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, y
