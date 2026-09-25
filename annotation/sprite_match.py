"""Label buildings by matching against the game's own sprites.

STATUS: EXPERIMENTAL, DOES NOT WORK YET (see the note at the bottom of this
docstring). The sprite *index* is useful on its own -- 603 official renders
across 31 building types -- but the matcher built on it misclassifies.

Cutting templates out of one screenshot only finds buildings that look like
that screenshot's -- useless across 30 different players' bases, where every
structure is a different level with a different skin. The
`clash-of-clans-data` package ships the official render of *every building at
every level*, which is exactly the missing piece: match against those instead.

    python -m annotation.sprite_match index          # build the sprite index
    python -m annotation.sprite_match calibrate --image <png>
    python -m annotation.sprite_match run --session data/raw/screenshots/coc_01

Method: propose candidate locations (objectness peaks), then classify each
candidate by masked cosine similarity between its gradient descriptor and
every sprite's. The sprite's alpha channel is the mask, so the transparent
background never contributes -- only the building's own shape does.

What went wrong: with masked cosine similarity, sprites with large opaque
masks (Army Camp, Eagle Artillery) win almost every comparison regardless of
content -- on a real base frame, 25 of 26 detections came back as one of those
two, and raising the threshold to 0.40 produced zero detections instead of
better ones. The similarity is dominated by mask area, not by shape. Fixing it
needs per-sprite score normalisation (calibrate each sprite's score
distribution on negatives) and colour, not just gradient. Left for later; the
detector-in-the-loop path in docs/LABELING.md is the cheaper route.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

PKG = pathlib.Path("datasets/coc_data/package")
INDEX = pathlib.Path("datasets/sprites/index.json")
DESC = 40                      # descriptor grid
DEFAULT_PX_PER_TILE = 24.0

# sprite directory -> annotation class. Anything not listed is skipped.
DIR_TO_CLASS = {
    "cannon": "CANNON", "archer-tower": "ARCHER_TOWER", "mortar": "MORTAR",
    "wizard-tower": "WIZARD_TOWER", "air-defense": "AIR_DEFENSE",
    "air-sweeper": "AIR_SWEEPER", "hidden-tesla": "TESLA",
    "bomb-tower": "BOMB_TOWER", "inferno-tower": "INFERNO", "x-bow": "XBOW",
    "eagle-artillery": "EAGLE_ARTILLERY", "scattershot": "SCATTERSHOT",
    "monolith": "MONOLITH", "spell-tower": "SPELL_TOWER",
    "multi-archer-tower": "ARCHER_TOWER", "ricochet-cannon": "CANNON",
    "super-wizard-tower": "WIZARD_TOWER", "firespitter": "DEFENSE_OTHER",
    "multi-gear-tower": "DEFENSE_OTHER", "revenge-tower": "DEFENSE_OTHER",
    "builders-hut": "BUILDER_HUT", "crafting-station": "OTHER_BUILDING",

    "gold-storage": "GOLD_STORAGE", "elixir-storage": "ELIXIR_STORAGE",
    "dark-elixir-storage": "DARK_STORAGE", "gold-mine": "GOLD_MINE",
    "elixir-collector": "ELIXIR_COLLECTOR", "dark-elixir-drill": "DARK_DRILL",
    "clan-castle": "CLAN_CASTLE",

    "army-camp": "ARMY_CAMP", "barracks": "BARRACKS",
    "dark-barracks": "BARRACKS", "laboratory": "LABORATORY",
    "spell-factory": "SPELL_FACTORY", "dark-spell-factory": "SPELL_FACTORY",
    "workshop": "WORKSHOP", "pet-house": "OTHER_BUILDING",
    "hero-hall": "OTHER_BUILDING", "blacksmith": "OTHER_BUILDING",
    "hero-banner": "OTHER_BUILDING",

    "town-hall": "TOWN_HALL",
    "wall": "WALL",
}

# footprint in tiles, for turning a sprite into an expected pixel size
TILES = {
    "TOWN_HALL": 4, "CLAN_CASTLE": 3, "EAGLE_ARTILLERY": 3, "SCATTERSHOT": 3,
    "INFERNO": 2, "XBOW": 3, "AIR_DEFENSE": 3, "WIZARD_TOWER": 3,
    "ARCHER_TOWER": 3, "CANNON": 3, "MORTAR": 3, "TESLA": 2, "BOMB_TOWER": 3,
    "AIR_SWEEPER": 3, "MONOLITH": 3, "SPELL_TOWER": 3, "DEFENSE_OTHER": 3,
    "GOLD_STORAGE": 3, "ELIXIR_STORAGE": 3, "DARK_STORAGE": 3,
    "GOLD_MINE": 3, "ELIXIR_COLLECTOR": 3, "DARK_DRILL": 3,
    "ARMY_CAMP": 4, "BARRACKS": 3, "LABORATORY": 3, "SPELL_FACTORY": 3,
    "WORKSHOP": 4, "BUILDER_HUT": 2, "OTHER_BUILDING": 3, "WALL": 1,
}


# ------------------------------------------------------------------ features
def gradient(a: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def sprite_descriptor(path: pathlib.Path) -> tuple | None:
    """(descriptor, mask) for a sprite, cropped to its opaque bounding box."""
    im = Image.open(path).convert("RGBA")
    alpha = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(alpha > 24)
    if len(ys) < 50:
        return None
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    g = np.asarray(im.convert("L").resize((DESC, DESC)), dtype=np.float32) / 255.0
    m = np.asarray(im.split()[-1].resize((DESC, DESC)), dtype=np.float32) / 255.0
    d = gradient(g) * m
    d = d - d[m > 0.2].mean() if (m > 0.2).any() else d
    d = d * m
    n = np.linalg.norm(d)
    return (d / n, m) if n > 1e-6 else None


def patch_descriptor(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray | None:
    half = size // 2
    h, w = img.shape
    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
        return None
    crop = img[y - half:y + half, x - half:x + half]
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((DESC, DESC)),
                       dtype=np.float32) / 255.0
    return gradient(small)


# ------------------------------------------------------------------- index
def build_index() -> list:
    if not PKG.exists():
        raise SystemExit(f"{PKG} not found; see scripts/import_coc_data.py")
    entries = []
    for png in sorted((PKG / "images" / "home").rglob("*.png")):
        parts = png.parts
        try:
            key = next(p for p in parts if p in DIR_TO_CLASS)
        except StopIteration:
            continue
        if "geared-up" in str(png) or "super" in png.parts[-2]:
            continue                       # rare skins; skip for now
        cls = DIR_TO_CLASS[key]
        entries.append({"cls": cls, "path": str(png), "tiles": TILES.get(cls, 3)})
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(entries))
    return entries


def load_index(levels_per_class: int | None = 6, classes: set | None = None) -> list:
    entries = json.loads(INDEX.read_text()) if INDEX.exists() else build_index()
    if classes:
        entries = [e for e in entries if e["cls"] in classes]
    if levels_per_class:
        by: dict = {}
        for e in entries:
            by.setdefault(e["cls"], []).append(e)
        entries = []
        for cls, group in by.items():
            # highest levels first: real bases at TH11+ use the top ones
            group.sort(key=lambda e: e["path"])
            entries.extend(group[-levels_per_class:])
    out = []
    for e in entries:
        d = sprite_descriptor(pathlib.Path(e["path"]))
        if d is not None:
            out.append({**e, "desc": d[0], "mask": d[1]})
    return out


# ------------------------------------------------------------------ matching
def candidates(img: Image.Image, top: int = 90, min_dist: int = 30) -> list:
    from annotation.discover import objectness, peaks

    return peaks(objectness(img), min_dist=min_dist, top=top)


def classify(gray: np.ndarray, cand: list, sprites: list, px_per_tile: float,
             threshold: float) -> list:
    S = np.stack([s["desc"].ravel() for s in sprites])
    masks = np.stack([s["mask"].ravel() for s in sprites])
    sizes = sorted({max(int(round(s["tiles"] * px_per_tile)), 24) for s in sprites})

    out = []
    for x, y, _score in cand:
        best = None
        for size in sizes:
            d = patch_descriptor(gray, x, y, size)
            if d is None:
                continue
            v = d.ravel()
            # mask each comparison by that sprite's silhouette
            vm = v[None, :] * masks
            vm = vm - vm.mean(axis=1, keepdims=True) * masks
            norms = np.linalg.norm(vm, axis=1) + 1e-8
            sims = (vm * S).sum(axis=1) / norms
            j = int(np.argmax(sims))
            if best is None or sims[j] > best[0]:
                best = (float(sims[j]), j, size)
        if best is None or best[0] < threshold:
            continue
        score, j, size = best
        half = size // 2
        out.append({"cls": sprites[j]["cls"], "score": round(score, 3),
                    "xyxy": [float(x - half), float(y - half),
                             float(x + half), float(y + half)],
                    "sprite": pathlib.Path(sprites[j]["path"]).name})
    return out


def match_image(path: pathlib.Path, sprites: list, px_per_tile: float,
                threshold: float, top: int = 90) -> list:
    img = Image.open(path).convert("RGB")
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return classify(gray, candidates(img, top=top), sprites, px_per_tile, threshold)


def calibrate(path: pathlib.Path, sprites: list,
              grid=(18, 20, 22, 24, 26, 28, 30)) -> dict:
    """Pick pixels-per-tile by seeing which value maximises match quality."""
    img = Image.open(path).convert("RGB")
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    cand = candidates(img, top=60)
    scores = {}
    for ppt in grid:
        hits = classify(gray, cand, sprites, ppt, threshold=0.0)
        s = sorted((h["score"] for h in hits), reverse=True)[:25]
        scores[ppt] = round(float(np.mean(s)) if s else 0.0, 4)
    best = max(scores, key=scores.get)
    return {"scores": scores, "best_px_per_tile": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("index")

    c = sub.add_parser("calibrate")
    c.add_argument("--image", required=True)
    c.add_argument("--levels", type=int, default=4)

    r = sub.add_parser("run")
    r.add_argument("--session", default="data/raw/screenshots/coc_01")
    r.add_argument("--states", nargs="*", default=["BASE_PREVIEW"])
    r.add_argument("--px-per-tile", type=float, default=DEFAULT_PX_PER_TILE)
    r.add_argument("--threshold", type=float, default=0.35)
    r.add_argument("--levels", type=int, default=4)
    r.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()
    if args.mode == "index":
        e = build_index()
        by: dict = {}
        for x in e:
            by[x["cls"]] = by.get(x["cls"], 0) + 1
        print(f"{len(e)} sprites over {len(by)} classes")
        print(json.dumps(dict(sorted(by.items(), key=lambda t: -t[1])), indent=1))
        return

    sprites = load_index(levels_per_class=args.levels)
    print(f"{len(sprites)} sprite templates loaded", flush=True)

    if args.mode == "calibrate":
        print(json.dumps(calibrate(pathlib.Path(args.image), sprites), indent=2))
        return

    session = pathlib.Path(args.session)
    meta = {}
    for line in (session / "meta.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            meta[d["path"]] = d

    counts: dict = {}
    done = 0
    for p in sorted(session.glob("*.png")):
        if args.states and meta.get(p.name, {}).get("screen_state") not in set(args.states):
            continue
        if args.limit and done >= args.limit:
            break
        boxes = match_image(p, sprites, args.px_per_tile, args.threshold)
        lp = p.with_suffix(".json")
        old = json.loads(lp.read_text())["boxes"] if lp.exists() else []
        keep = [b for b in old if b.get("source") != "sprite"]
        new = [{"cls": b["cls"], "xyxy": [round(v, 1) for v in b["xyxy"]],
                "auto": True, "source": "sprite", "score": b["score"],
                "sprite": b["sprite"]} for b in boxes]
        img = Image.open(p)
        lp.write_text(json.dumps({"image": p.name, "w": img.width, "h": img.height,
                                  "boxes": keep + new}, indent=1))
        for b in new:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1
        done += 1
        print(f"{p.name}: {len(new)} boxes", flush=True)
    print(json.dumps(dict(sorted(counts.items(), key=lambda t: -t[1])), indent=2))


if __name__ == "__main__":
    main()
