"""Object detector (Phase 5 / M3): a compact anchor-free CenterNet.

Why not YOLO-from-a-package: the detection targets here are small, dense and
axis-aligned, the class list is fixed, and the whole point of the exercise is
to own the pipeline. CenterNet is ~200 lines, has no anchor tuning, no NMS
(peak extraction instead), and trains fine on 1-2k labelled frames.

    heatmap  (C, H/4, W/4)   per-class object centre
    size     (2, H/4, W/4)   box width/height at the centre
    offset   (2, H/4, W/4)   sub-pixel centre correction
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from annotation.app import CLASSES


class Detector(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), width: int = 32):
        super().__init__()
        w = width
        self.stem = _conv(3, w, stride=2)                 # /2
        self.d1 = nn.Sequential(_conv(w, w * 2, stride=2), _conv(w * 2, w * 2))     # /4
        self.d2 = nn.Sequential(_conv(w * 2, w * 4, stride=2), _conv(w * 4, w * 4))  # /8
        self.d3 = nn.Sequential(_conv(w * 4, w * 8, stride=2), _conv(w * 8, w * 8))  # /16

        self.up2 = _conv(w * 8, w * 4)
        self.up1 = _conv(w * 4, w * 2)

        self.head_hm = _head(w * 2, n_classes, bias=-2.19)   # prior p ~ 0.1
        self.head_wh = _head(w * 2, 2)
        self.head_off = _head(w * 2, 2)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        f1 = self.d1(x)          # /4
        f2 = self.d2(f1)         # /8
        f3 = self.d3(f2)         # /16
        u2 = self.up2(F.interpolate(f3, scale_factor=2, mode="nearest")) + f2
        u1 = self.up1(F.interpolate(u2, scale_factor=2, mode="nearest")) + f1
        return {"hm": self.head_hm(u1), "wh": self.head_wh(u1), "off": self.head_off(u1)}


def _conv(cin: int, cout: int, k: int = 3, stride: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


def _head(cin: int, cout: int, bias: float | None = None) -> nn.Module:
    conv = nn.Conv2d(cin, cout, 1)
    if bias is not None:
        nn.init.constant_(conv.bias, bias)
    return nn.Sequential(nn.Conv2d(cin, cin, 3, 1, 1), nn.SiLU(inplace=True), conv)


# --------------------------------------------------------------------- loss
def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CornerNet/CenterNet penalty-reduced focal loss on the heatmap."""
    pred = torch.clamp(pred.sigmoid(), 1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def _gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, K, C) at flat indices `ind`."""
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w).permute(0, 2, 1)
    idx = ind.unsqueeze(-1).expand(-1, -1, c)
    return feat.gather(1, idx)


def reg_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    p = _gather(pred, ind)
    m = mask.unsqueeze(-1).expand_as(p)
    return (torch.abs(p * m - target * m).sum()) / (m.sum() + 1e-4)


def detector_loss(out: dict, hm, wh, off, mask, ind,
                  w_hm: float = 1.0, w_wh: float = 0.1, w_off: float = 1.0) -> tuple:
    l_hm = focal_loss(out["hm"], hm)
    l_wh = reg_l1_loss(out["wh"], wh, mask, ind)
    l_off = reg_l1_loss(out["off"], off, mask, ind)
    total = w_hm * l_hm + w_wh * l_wh + w_off * l_off
    return total, {"hm": l_hm.item(), "wh": l_wh.item(), "off": l_off.item()}


# ------------------------------------------------------------------- decode
@torch.no_grad()
def decode(out: dict, k: int = 128, threshold: float = 0.3, stride: int = 4) -> list:
    """Peaks -> boxes. 3x3 max-pool NMS, which is all CenterNet needs."""
    hm = out["hm"].sigmoid()
    keep = (F.max_pool2d(hm, 3, 1, 1) == hm).float()
    hm = hm * keep
    b, c, h, w = hm.shape

    scores, idx = hm.view(b, -1).topk(k)
    cls = idx // (h * w)
    pix = idx % (h * w)
    ys = (pix // w).float()
    xs = (pix % w).float()

    off = _gather(out["off"], pix)
    wh = _gather(out["wh"], pix)
    xs = xs + off[..., 0]
    ys = ys + off[..., 1]
    x1 = (xs - wh[..., 0] / 2) * stride
    y1 = (ys - wh[..., 1] / 2) * stride
    x2 = (xs + wh[..., 0] / 2) * stride
    y2 = (ys + wh[..., 1] / 2) * stride

    results = []
    for i in range(b):
        m = scores[i] >= threshold
        results.append({
            "boxes": torch.stack([x1[i][m], y1[i][m], x2[i][m], y2[i][m]], dim=-1).cpu().numpy(),
            "labels": cls[i][m].cpu().numpy(),
            "scores": scores[i][m].cpu().numpy(),
        })
    return results


# ---------------------------------------------------------------- evaluation
def box_iou(a, b):
    import numpy as np

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(preds: list, gts: list, iou_thr: float = 0.5,
                      n_classes: int | None = None,
                      names: list | None = None) -> dict:
    """Per-class AP plus mAP. Deliberately reports per class: the useful
    signal in Phase 5 is 'walls are bad', not one aggregate number.

    `names` overrides the class vocabulary, so a coarse run reports LOOT and
    AIR_DEFENSE rather than indices into the 47-class list.
    """
    names = names or CLASSES
    n_classes = n_classes if n_classes is not None else len(names)
    import numpy as np

    aps = {}
    for c in range(n_classes):
        scored, n_gt = [], 0
        for p, g in zip(preds, gts):
            pb = p["boxes"][p["labels"] == c]
            ps = p["scores"][p["labels"] == c]
            gb = g["boxes"][g["labels"] == c]
            n_gt += len(gb)
            if len(pb) == 0:
                continue
            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]
            ious = box_iou(pb, gb)
            taken = set()
            for i in range(len(pb)):
                j = int(np.argmax(ious[i])) if ious.shape[1] else -1
                hit = j >= 0 and ious[i, j] >= iou_thr and j not in taken
                if hit:
                    taken.add(j)
                scored.append((ps[i], float(hit)))
        if n_gt == 0:
            continue
        if not scored:
            aps[names[c]] = 0.0
            continue
        scored.sort(key=lambda t: -t[0])
        tp = np.cumsum([s[1] for s in scored])
        fp = np.cumsum([1 - s[1] for s in scored])
        rec = tp / n_gt
        prec = tp / np.maximum(tp + fp, 1e-9)
        # 101-point interpolated AP (COCO style)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = prec[rec >= t].max() if (rec >= t).any() else 0.0
            ap += p / 101
        aps[names[c]] = float(ap)
    aps["mAP"] = float(sum(v for k, v in aps.items() if k != "mAP") / max(len(aps), 1))
    return aps
