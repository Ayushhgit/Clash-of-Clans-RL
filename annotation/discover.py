"""Unsupervised sprite discovery for bulk pre-labelling.

Template matching needs one template per building type, and I have to name each
one. Discovery flips it around: find every building-shaped thing in every
screenshot, group the ones that look alike, and ask for a name *once per
group*. One naming pass then labels hundreds of boxes.

    python -m annotation.discover find   --session data/raw/screenshots/coc_01
    #   -> datasets/discovery/clusters.json + a contact sheet of exemplars
    python -m annotation.discover apply  --map datasets/discovery/names.json
    #   -> writes <image>.json label files

Pipeline
    objectness   gradient energy, minus grass; buildings light up
    peaks        non-maximum suppression over the objectness map
    patches      fixed-size crop per peak, scale-normalised
    descriptor   24x24 gradient thumbnail, L2 normalised
    clustering   greedy cosine agglomeration (no k to choose)

The screenshots are all one game at one art style, so "looks alike" is a
strong signal: clusters come out close to one-per-building-type.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

DISCOVERY = pathlib.Path("datasets/discovery")


# ---------------------------------------------------------------- objectness
def gradient(a: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """Separable running-mean blur via cumulative sums."""
    pad = k // 2
    p = np.pad(a, ((pad, pad), (pad, pad)), mode="edge")
    c = np.cumsum(p, axis=0)
    a1 = (c[k:, :] - c[:-k, :]) / k
    c = np.cumsum(a1, axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def objectness(img: Image.Image) -> np.ndarray:
    """High where structures are, low on grass.

    Grass is textured too, so edge energy alone is not enough. Buildings are
    also *less green* and more saturated in the greys/browns, so the score
    mixes gradient energy with a "not grass" colour term.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    edges = box_blur(gradient(g), 9)
    r, gr, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    grassiness = np.clip(gr - (r + b) / 2, 0, 1)      # green dominance
    not_grass = box_blur(1.0 - grassiness * 4.0, 15).clip(0, 1)
    return edges * not_grass


def peaks(score: np.ndarray, min_dist: int = 42, top: int = 70,
          margin: int = 20) -> list:
    """Greedy non-maximum suppression over the objectness map."""
    s = score.copy()
    s[:margin, :] = s[-margin:, :] = 0
    s[:, :margin] = s[:, -margin:] = 0
    out = []
    for _ in range(top):
        idx = int(np.argmax(s))
        y, x = divmod(idx, s.shape[1])
        if s[y, x] <= 0:
            break
        out.append((x, y, float(s[y, x])))
        y0, y1 = max(0, y - min_dist), min(s.shape[0], y + min_dist)
        x0, x1 = max(0, x - min_dist), min(s.shape[1], x + min_dist)
        s[y0:y1, x0:x1] = 0
    return out


# --------------------------------------------------------------- descriptors
PATCH = 96
DESC = 24


def descriptor(img: Image.Image, x: int, y: int, size: int = PATCH) -> np.ndarray | None:
    half = size // 2
    if x - half < 0 or y - half < 0 or x + half > img.width or y + half > img.height:
        return None
    crop = img.crop((x - half, y - half, x + half, y + half)).convert("L").resize((DESC, DESC))
    a = np.asarray(crop, dtype=np.float32) / 255.0
    d = gradient(a).ravel()
    d = d - d.mean()
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else None


def cluster(descs: np.ndarray, threshold: float = 0.62) -> np.ndarray:
    """Greedy agglomeration by cosine similarity. Returns a label per row."""
    n = len(descs)
    labels = np.full(n, -1, dtype=np.int64)
    centres: list = []
    order = np.arange(n)
    for i in order:
        if not centres:
            centres.append(descs[i].copy())
            labels[i] = 0
            continue
        sims = np.array([float(descs[i] @ c) for c in centres])
        j = int(np.argmax(sims))
        if sims[j] >= threshold:
            labels[i] = j
            centres[j] = 0.9 * centres[j] + 0.1 * descs[i]     # slow drift
            centres[j] /= np.linalg.norm(centres[j]) + 1e-9
        else:
            centres.append(descs[i].copy())
            labels[i] = len(centres) - 1
    return labels


# --------------------------------------------------------------------- find
def find(session: pathlib.Path, top: int, threshold: float, states: set | None,
         patch: int = PATCH) -> dict:
    meta = {}
    mp = session / "meta.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                meta[d["path"]] = d

    items, descs = [], []
    for p in sorted(session.glob("*.png")):
        if states and meta.get(p.name, {}).get("screen_state") not in states:
            continue
        img = Image.open(p).convert("RGB")
        for x, y, s in peaks(objectness(img), top=top):
            d = descriptor(img, x, y, patch)
            if d is None:
                continue
            items.append({"image": p.name, "x": x, "y": y, "score": round(s, 4)})
            descs.append(d)

    if not items:
        raise SystemExit("no candidates found")
    DISCOVERY.mkdir(parents=True, exist_ok=True)
    D = np.stack(descs)
    np.save(DISCOVERY / "descriptors.npy", D)     # cached so recluster is instant
    labels = cluster(D, threshold)
    for it, l in zip(items, labels):
        it["cluster"] = int(l)

    DISCOVERY.mkdir(parents=True, exist_ok=True)
    payload = {"session": str(session), "patch": patch, "threshold": threshold,
               "items": items, "n_clusters": int(labels.max()) + 1}
    (DISCOVERY / "clusters.json").write_text(json.dumps(payload))
    sheet(session, payload)
    return payload


def sheet(session: pathlib.Path, payload: dict, per_cluster: int = 3,
          out: pathlib.Path | None = None, max_clusters: int | None = 60) -> pathlib.Path:
    """Contact sheet: a few members of every cluster, biggest clusters first."""
    from PIL import ImageDraw

    items = payload["items"]
    patch = payload["patch"]
    by: dict = {}
    for it in items:
        by.setdefault(it["cluster"], []).append(it)
    order = sorted(by, key=lambda c: -len(by[c]))
    if max_clusters:
        order = order[:max_clusters]

    cell, cols = 104, 12
    rows = (len(order) * per_cluster + cols - 1) // cols
    img = Image.new("RGB", (cell * cols, (cell + 14) * rows), (18, 18, 20))
    d = ImageDraw.Draw(img)
    cache: dict = {}
    k = 0
    for c in order:
        for it in by[c][:per_cluster]:
            src = cache.setdefault(it["image"], Image.open(session / it["image"]).convert("RGB"))
            half = patch // 2
            crop = src.crop((it["x"] - half, it["y"] - half, it["x"] + half, it["y"] + half))
            crop = crop.resize((cell - 8, cell - 8))
            cx, cy = (k % cols) * cell, (k // cols) * (cell + 14)
            img.paste(crop, (cx + 4, cy + 14))
            d.text((cx + 3, cy + 2), f"c{c} n={len(by[c])}", fill=(255, 255, 0))
            k += 1
    out = out or DISCOVERY / "clusters_sheet.png"
    img.save(out)
    return out


# -------------------------------------------------------------------- apply
def apply(names: dict, session: pathlib.Path | None = None,
          patch_scale: float = 0.85) -> dict:
    """Turn named clusters into label files. Clusters mapped to null are dropped."""
    payload = json.loads((DISCOVERY / "clusters.json").read_text())
    session = session or pathlib.Path(payload["session"])
    patch = payload["patch"]
    half = int(patch * patch_scale) // 2

    per_image: dict = {}
    for it in payload["items"]:
        cls = names.get(str(it["cluster"]))
        if not cls:
            continue
        per_image.setdefault(it["image"], []).append(
            {"cls": cls,
             "xyxy": [float(it["x"] - half), float(it["y"] - half),
                      float(it["x"] + half), float(it["y"] + half)],
             "auto": True, "score": it["score"], "cluster": it["cluster"]})

    counts: dict = {}
    for name, boxes in per_image.items():
        p = session / name
        lp = p.with_suffix(".json")
        old = json.loads(lp.read_text())["boxes"] if lp.exists() else []
        # keep everything already there (hand-drawn and template-matched);
        # only add discovered boxes that do not land on an existing one
        boxes = [b for b in boxes if all(_iou(b["xyxy"], o["xyxy"]) < 0.35 for o in old)]
        img = Image.open(p)
        lp.write_text(json.dumps({"image": name, "w": img.width, "h": img.height,
                                  "boxes": old + boxes}, indent=1))
        for b in old + boxes:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    return counts


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("find")
    f.add_argument("--session", default="data/raw/screenshots/coc_01")
    f.add_argument("--top", type=int, default=70)
    f.add_argument("--threshold", type=float, default=0.62)
    f.add_argument("--patch", type=int, default=PATCH)
    f.add_argument("--states", nargs="*", default=None)

    rc = sub.add_parser("recluster")
    rc.add_argument("--threshold", type=float, required=True)

    a = sub.add_parser("apply")
    a.add_argument("--map", required=True, help="json: {cluster_id: CLASS or null}")
    a.add_argument("--session", default=None)

    args = ap.parse_args()
    if args.mode == "recluster":
        payload = json.loads((DISCOVERY / "clusters.json").read_text())
        D = np.load(DISCOVERY / "descriptors.npy")
        labels = cluster(D, args.threshold)
        for it, l in zip(payload["items"], labels):
            it["cluster"] = int(l)
        payload["threshold"] = args.threshold
        payload["n_clusters"] = int(labels.max()) + 1
        (DISCOVERY / "clusters.json").write_text(json.dumps(payload))
        sheet(pathlib.Path(payload["session"]), payload)
        by: dict = {}
        for it in payload["items"]:
            by[it["cluster"]] = by.get(it["cluster"], 0) + 1
        big = sorted(by.items(), key=lambda t: -t[1])
        singles = sum(1 for v in by.values() if v == 1)
        print(f"threshold {args.threshold}: {payload['n_clusters']} clusters, "
              f"{singles} singletons, largest {big[:12]}")
        return
    if args.mode == "find":
        p = find(pathlib.Path(args.session), args.top, args.threshold,
                 set(args.states) if args.states else None, args.patch)
        by: dict = {}
        for it in p["items"]:
            by[it["cluster"]] = by.get(it["cluster"], 0) + 1
        print(f"{len(p['items'])} candidates -> {p['n_clusters']} clusters")
        print("largest:", sorted(by.items(), key=lambda t: -t[1])[:15])
        print(f"sheet: {DISCOVERY / 'clusters_sheet.png'}")
    else:
        names = json.loads(pathlib.Path(args.map).read_text())
        counts = apply(names, pathlib.Path(args.session) if args.session else None)
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
