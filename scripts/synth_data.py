"""Synthesise labelled base images by compositing official building renders.

    python -m scripts.synth_data --n 600 --out data/synth

Why this exists: a CenterNet trained from scratch on 33 hand-labelled frames
learns walls (AP 0.75 -- numerous and uniform) and nothing else (every building
class 0.0). The gap is examples per class, and hand labelling supplies them at
roughly one frame per half hour. There is no pretrained backbone available in
this environment (`torchvision` is not installed), so the usual shortcut is
closed too.

But the appearance of every building at every level is *already known*:
`clash-of-clans-data` ships 2207 official renders with alpha. Pasting them onto
real grass at plausible positions and scales produces unlimited, perfectly
labelled data. Cut-and-paste synthesis is a well-worn answer to exactly this
shape of problem -- known object appearance, unknown layout.

What it does not fix, and what the real frames are still needed for:

* the renders are clean and unoccluded; real buildings sit behind walls, are
  overlapped by their neighbours, and carry level badges and clan flags;
* real frames have shadows baked into the terrain and a global colour grade.

So synthetic data is for *pretraining* and the hand-labelled frames remain the
validation set. If AP on real frames does not move, the synthesis is not
transferring and the honest thing is to say so rather than report the
synthetic number.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

import numpy as np
from PIL import Image

from annotation.app import CLASSES

SPRITE_ROOT = pathlib.Path("datasets/coc_data/package/images/home")

# sprite directory -> the *fine* class name from `annotation.app.CLASSES`.
#
# It must be the fine name, not the coarse group: labels are read back through
# `CLASS_TO_ID`, which only knows the 47-class vocabulary, and anything outside
# it is silently dropped. Writing "LOOT"/"DEFENSE" here cost every storage and
# every defence in the synthetic set -- 500 frames that looked fine on disk and
# loaded as walls and town halls only. `--coarse` does the collapsing.
SPRITE_GROUPS = {
    "town-hall": "TOWN_HALL",
    "defenses/air-defense": "AIR_DEFENSE",
    "defenses/air-sweeper": "AIR_SWEEPER",
    "resource-buildings/gold-storage": "GOLD_STORAGE",
    "resource-buildings/elixir-storage": "ELIXIR_STORAGE",
    "resource-buildings/dark-elixir-storage": "DARK_STORAGE",
    "resource-buildings/gold-mine": "GOLD_MINE",
    "resource-buildings/elixir-collector": "ELIXIR_COLLECTOR",
    "resource-buildings/dark-elixir-drill": "DARK_DRILL",
    "defenses/archer-tower": "ARCHER_TOWER",
    "defenses/cannon": "CANNON",
    "defenses/mortar": "MORTAR",
    "defenses/wizard-tower": "WIZARD_TOWER",
    "defenses/hidden-tesla": "TESLA",
    "defenses/bomb-tower": "BOMB_TOWER",
    "defenses/inferno-tower": "INFERNO",
    "defenses/x-bow": "XBOW",
    "defenses/eagle-artillery": "EAGLE_ARTILLERY",
    "defenses/scattershot": "SCATTERSHOT",
    "defenses/monolith": "MONOLITH",
    "defenses/spell-tower": "SPELL_TOWER",
    "other/clan-castle": "CLAN_CASTLE",
    "army-buildings/army-camp": "ARMY_CAMP",
    "army-buildings/barracks": "BARRACKS",
    "army-buildings/laboratory": "LABORATORY",
    "army-buildings/spell-factory": "SPELL_FACTORY",
    "army-buildings/workshop": "WORKSHOP",
    "defenses/wall": "WALL",
}

# Walls are laid in runs, not scattered singly, and they are small.
WALL_RUN = (4, 14)


def load_sprites() -> dict:
    """{coarse class: [RGBA image, ...]}, preferring higher levels."""
    out: dict = {c: [] for c in set(SPRITE_GROUPS.values())}
    for sub, group in SPRITE_GROUPS.items():
        d = SPRITE_ROOT / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.png")):
            try:
                im = Image.open(p).convert("RGBA")
            except Exception:
                continue
            if im.size[0] < 24 or im.size[1] < 24:
                continue
            out[group].append(im)
    return {k: v for k, v in out.items() if v}


def grass_tiles(frames: list, n: int = 400, size: int = 128,
                rng: random.Random | None = None) -> list:
    """Patches of real terrain, sampled from anywhere in real frames.

    Synthetic backgrounds have to be real grass, not a flat colour: the
    detector would otherwise learn "building = anything textured". Taking only
    the four corners of each frame yielded 5 usable tiles, which tiled the
    background visibly; random crops with a looser test give hundreds.
    """
    rng = rng or random.Random(0)
    tiles = []
    for p in frames:
        try:
            rgb = np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            continue
        h, w = rgb.shape[:2]
        if h < size + 2 or w < size + 2:
            continue
        for _ in range(24):
            y = rng.randint(0, h - size - 1)
            x = rng.randint(0, w - size - 1)
            t = rgb[y:y + size, x:x + size]
            m = t.mean(axis=(0, 1))
            g = t.max(axis=2)
            # green-dominant and not too busy, and free of chrome: a looser
            # test pulled in the black letterbox at the frame edge and white
            # HUD numerals, which would have taught the detector that "text on
            # black" is terrain.
            dark = float((g < 40).mean())
            bright = float((t.min(axis=2) > 225).mean())
            if (m[1] > m[2] + 18 and m[1] > 80 and t.std() < 52
                    and dark < 0.02 and bright < 0.005):
                tiles.append(t)
            if len(tiles) >= n:
                return tiles
    return tiles


def make_background(tiles: list, w: int, h: int, rng: random.Random) -> Image.Image:
    bg = Image.new("RGB", (w, h))
    if not tiles:
        bg.paste((92, 128, 56), (0, 0, w, h))
        return bg
    ts = tiles[0].shape[0]
    # start off-grid and step irregularly so the tiling seams do not line up
    # into a lattice the detector could key on
    y = -rng.randint(0, ts - 1)
    while y < h:
        x = -rng.randint(0, ts - 1)
        while x < w:
            t = Image.fromarray(rng.choice(tiles))
            if rng.random() < 0.5:
                t = t.transpose(Image.FLIP_LEFT_RIGHT)
            bg.paste(t, (x, y))
            x += ts
        y += ts
    return bg


# Matched to the real frames rather than guessed. Measured over 813 real
# hand-labelled boxes, a building spans 0.042 / 0.058 / 0.121 of frame width at
# p10 / p50 / p90. The first synthesis produced 0.073 / 0.125 / 0.255 -- 2.2x
# too large -- so at a 320px training size real buildings were ~13px while the
# model had been pretrained on ~40px objects. Aspect matches too: real captures
# are ~1.76:1, not 4:3.
SPRITE_SCALE = (0.13, 0.34)


def compose(sprites: dict, tiles: list, rng: random.Random,
            w: int = 840, h: int = 480) -> tuple:
    img = make_background(tiles, w, h, rng)
    boxes: list = []
    placed: list = []
    # a few wall runs first, so buildings can sit on top of them as they do
    # in a real base
    if "WALL" in sprites:
        for _ in range(rng.randint(2, 5)):
            seg = rng.choice(sprites["WALL"])
            sc = rng.uniform(0.12, 0.24)
            sw, sh = max(int(seg.size[0] * sc), 8), max(int(seg.size[1] * sc), 8)
            seg = seg.resize((sw, sh), Image.LANCZOS)
            x0, y0 = rng.randint(0, max(w - sw, 1)), rng.randint(0, max(h - sh, 1))
            dx, dy = rng.choice([(sw // 2, sh // 4), (-sw // 2, sh // 4)])
            for k in range(rng.randint(*WALL_RUN)):
                x, y = x0 + dx * k, y0 + dy * k
                if not (0 <= x <= w - sw and 0 <= y <= h - sh):
                    break
                img.paste(seg, (x, y), seg)
                a = np.asarray(seg.split()[-1])
                ys, xs = np.nonzero(a > 32)
                if ys.size:
                    boxes.append({"cls": "WALL",
                                  "xyxy": [x + int(xs.min()), y + int(ys.min()),
                                           x + int(xs.max()), y + int(ys.max())]})

    # Sample the *coarse* group first, then a fine class within it. Choosing
    # uniformly over the 27 fine classes gave 128 town halls against 2304
    # defences across 400 frames, because "defence" spans a dozen sprite dirs
    # and "town hall" spans one.
    from ai.perception.coarse import COARSE
    by_group: dict = {}
    for fine in sprites:
        if fine == "WALL":
            continue
        by_group.setdefault(COARSE.get(fine, "OTHER_BUILDING"), []).append(fine)
    coarse_groups = sorted(by_group)

    n = rng.randint(22, 46)          # a real base holds ~40 buildings
    for _ in range(n):
        g = rng.choice(by_group[rng.choice(coarse_groups)])
        sp = rng.choice(sprites[g])
        scale = rng.uniform(*SPRITE_SCALE)
        sw = max(int(sp.size[0] * scale), 16)
        sh = max(int(sp.size[1] * scale), 16)
        s = sp.resize((sw, sh), Image.LANCZOS)
        for _try in range(12):
            x = rng.randint(0, max(w - sw, 1))
            y = rng.randint(0, max(h - sh, 1))
            # allow slight overlap (real bases are packed) but not burial
            if all(not (x < px + pw * 0.6 and px < x + sw * 0.6 and
                        y < py + ph * 0.6 and py < y + sh * 0.6)
                   for px, py, pw, ph in placed):
                break
        else:
            continue
        img.paste(s, (x, y), s)
        placed.append((x, y, sw, sh))
        # tighten the box to the opaque pixels, not the padded canvas
        a = np.asarray(s.split()[-1])
        ys, xs = np.nonzero(a > 32)
        if ys.size == 0:
            continue
        boxes.append({"cls": g, "xyxy": [x + int(xs.min()), y + int(ys.min()),
                                         x + int(xs.max()), y + int(ys.max())]})
    return img, boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sprites = load_sprites()
    if not sprites:
        raise SystemExit(f"no sprites under {SPRITE_ROOT}")
    print({k: len(v) for k, v in sprites.items()})

    frames = sorted(pathlib.Path("data/raw/screenshots").rglob("*.png"))
    rng0 = random.Random(args.seed)
    rng0.shuffle(frames)
    tiles = grass_tiles(frames[:120], rng=rng0)
    print(f"{len(tiles)} grass tiles from real frames")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for i in range(args.n):
        img, boxes = compose(sprites, tiles, rng)
        stem = out / f"synth_{i:05d}"
        img.save(stem.with_suffix(".png"))
        stem.with_suffix(".json").write_text(json.dumps(
            {"image": stem.name + ".png", "w": img.size[0], "h": img.size[1],
             "boxes": boxes}, indent=1))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)
    print(f"wrote {args.n} synthetic frames to {out}")


if __name__ == "__main__":
    main()
