# LABELING -- status of the real-game dataset

## Current state (hand-labelled base frames)

| | |
|---|---|
| distinct frames in the corpus | 182 of 683 (rest are near-duplicates) |
| frames worth labelling for buildings | 76 |
| **hand-labelled so far** | **5** |
| building boxes | 159 (33 of them legacy WALL boxes) |

The five were labelled by reading the frame at 2x with a coordinate grid and
**verified by rendering the boxes back over the image** -- every box landed on
the intended building. The rule used throughout: a building whose type is not
certain gets `DEFENSE_OTHER` or `OTHER_BUILDING`, never a guess. 14 boxes are
generic for that reason.

Training on this is not yet meaningful. `--buildings-only` (new) drops the HUD
boxes, which outnumber hand-drawn ones ~20:1 and otherwise take over both the
loss and the validation split -- the first run happily reported AP for
`ARMY_CARD` and never saw a building. With the filter on, 40 epochs gives
mAP\@50 0.505 **on WALL alone**, because the 4-frame validation split contains
no other class. That number says nothing about whether the detector can find a
storage or an air defence.

Measured colour separability, from the hand labels:

| class | hue (p25/p50/p75) | separable by colour? |
|---|---|---|
| ELIXIR_STORAGE | 275 / 290 / 300 | **yes** -- tight and isolated |
| GOLD_STORAGE | 36 / 42 / 58 | no |
| AIR_DEFENSE | 40 / 57 / 62 | no -- overlaps gold |
| TOWN_HALL | 26 / 44 / 55 | no -- overlaps gold |

So the magenta storages can be found without a detector; everything else
shares the gold/olive band and cannot. That is *why* the four automated
approaches below failed, now measured rather than assumed.

**Estimated remaining effort: ~70 frames.** A human with the GUI is far faster
at this than a model estimating coordinates from a rendered image.


Session: `data/raw/screenshots/coc_01` -- 107 frames of Clash of Clans,
~1180x670, captured by hand.

## What is done

| item | count | how | trust |
|---|---|---|---|
| screen-state labels (Dataset B) | 107 / 107 | read by eye from contact sheets | high |
| UI boxes (Dataset A, HUD) | ~690 across 56 frames | measured once per state, projected onto every frame that shows the HUD | high |
| building boxes (Dataset A) | 24, on one frame | template matching, verified on an overlay | high |
| building boxes, everything else | 0 | -- | -- |

Screen states: `BASE_PREVIEW` 89, `HOME` 7, `BATTLE` 4, `RESULT` 3,
`SEARCH` 1, `ARMY` 1, `BATTLE_CONFIRM` 1, `DIALOG` 1.

`ARMY` (army overlay), `BATTLE_CONFIRM` (the "End Battle?" dialog) and
`DIALOG` (the star-bonus popup) were added to the state vocabulary because all
three appear in the captures and each needs a different action from the agent.

## UI boxes are measured, not detected

The HUD is drawn at fixed positions, so `scripts/label_ui.py` stores one
measured rectangle per element (as a fraction of frame size, read off a
coordinate grid) and projects it onto every frame of that state. Faster and
more accurate than detecting it, and it doubles as the source of
`configs/geometry.json`, which the controller needs in order to click.

Covered: `ATTACK_BUTTON`, `NEXT_BUTTON`, `END_BATTLE_BUTTON`,
`RETURN_BUTTON`, `LOOT_PANEL`, `HUD_GOLD`, `HUD_ELIXIR`, `HUD_DARK`,
`HUD_TIMER`, and each occupied `ARMY_CARD` slot (occupancy from saturation --
empty slots are a dashed outline over the map).

**51 frames get no UI boxes** because they are cropped or zoomed captures with
the HUD out of frame. The script detects this by looking for the anchor widget
where it belongs and skipping the frame when it is not there. An earlier
version searched for the anchor anywhere in the frame; it "found" the orange
NEXT button inside orange *buildings* and stamped a full fake HUD across the
base. Rendering the result is what caught it -- hence the tight tolerance now.

> **For future captures: grab the whole game window, not a crop.** Cropped
> frames still train the object detector, but they cannot contribute UI labels
> and they break the fixed-layout shortcut.

## What is NOT done: buildings, and why

Four automated approaches were tried. One works within a single base; none
work across bases.

1. **Template matching** (`annotation/template_match.py`) -- normalised
   cross-correlation on gradients. Localises precisely (verified on an
   overlay) but only matches buildings that look like the template. Every
   preview is a different player's base with different levels and skins, so
   templates cut from one base find ~24 boxes there and ~0 elsewhere.
2. **Unsupervised sprite clustering** (`annotation/discover.py`) -- cluster
   candidate patches, name each cluster once. 1564 candidates produced
   488-1062 clusters, mostly singletons: the descriptor is sensitive to where
   the candidate point lands on the building. Output was ~30% wrong, so it was
   purged rather than shipped.
3. **Blob segmentation** (`annotation/segment.py`) -- "everything that is not
   grass". Failed three ways: green-dominance in RGB flags the whole base
   interior (mown grass is yellow-green); a hue band cannot exclude gold
   buildings while including olive ground; local contrast fails because this
   game's grass is itself a noisy texture. 86-96% of the frame ends up flagged.
4. **Official sprite matching** (`annotation/sprite_match.py`) -- match against
   the real renders from `clash-of-clans-data` (603 sprites, 31 types, every
   level). This *should* be the answer to the level/skin problem. As
   implemented it is not: with masked cosine similarity the sprites with the
   largest opaque masks (Army Camp, Eagle Artillery) win nearly every
   comparison -- 25 of 26 detections on a real frame -- and raising the
   threshold gives zero detections rather than better ones. It needs per-sprite
   score normalisation against negatives, plus colour. Left experimental.

Honest summary: **cross-base building labelling did not fall to the classical
tricks.** It needs a human seed pass or a detector, and the detector needs the
seed pass first -- exactly the bootstrap Phase 4 of the plan assumes.

## Recommended next step (the cheap path)

Do **not** hand-label all 89 previews. Label a seed, train, correct:

```bash
# 1. label ~8-10 BASE_PREVIEW frames by hand (~15 min each)
python -m annotation.app data/raw/screenshots/coc_01

# 2. train detector v1 (the UI boxes are already free training signal)
python -m scripts.train_detector --data data/raw/screenshots --epochs 40

# 3. let it label the rest, ranked by uncertainty
python -m scripts.auto_label --checkpoint checkpoints/detector_v1.pt \
    --data data/raw/screenshots --write-accepted

# 4. correct the worst, retrain (the Phase 7 loop)
```

Effort: a TH13 base has ~50 buildings plus ~250 wall segments. Skip walls at
first -- they are the known-hard class, and what the agent mainly needs from
them is where the compartments are. At ~4 s per box, a 10-frame seed is well
under an hour.

## Real game data is now wired in

`npm pack clash-of-clans-data` gives the wiki's stats as JSON plus every
building's official render. Two uses:

* `python -m scripts.import_coc_data --town-hall 13 --write` writes
  `simulator/game_data.py` with **real** hitpoints, DPS, range, target type and
  footprint size. `simulator/entities.py` has `USE_REAL_STATS = False`; flip it
  to use them. It is off by default because the real numbers are on a very
  different scale (a max wall has 9 000 hp, a barbarian 230; armies are sized
  by housing space, not unit count), so switching over also means resizing the
  army and re-running `scripts/baselines.py` to re-establish the balance
  targets.
* `python -m annotation.sprite_match index` builds an index of 603 official
  building renders -- useful raw material even though the matcher built on it
  does not work yet.

## Tooling built for this

| tool | what it does |
|---|---|
| `scripts/ingest.py` | drop screenshots in `data/raw/inbox`, get a session with `meta.jsonl` |
| `annotation/app.py` | the labelling GUI; 47 classes, keyboard-driven |
| `annotation/assist.py` | `grid` (coordinate overlay for reading exact boxes), `check` (render labels onto the image), `stats` |
| `scripts/label_ui.py` | project the measured HUD layout onto every frame that shows it |
| `scripts/import_coc_data.py` | real stats -> `simulator/game_data.py` |
| `annotation/template_match.py` | cut a template, find its twins (same base only) |
| `annotation/discover.py` | cluster candidate patches, name a cluster once |
| `annotation/sprite_match.py` | official-sprite index; matcher experimental |
| `annotation/segment.py` | experimental, does not work yet |

## Class list

47 classes, extended from the generic starter list to the real game: Eagle
Artillery, Scattershot, Monolith, Spell Tower, Clan Castle, hero altars, three
storage types and three collector types, plus `DEFENSE_OTHER` (a defence whose
exact type is not certain from the frame) and `UNLABELED_BUILDING` (a box with
no name yet). Every one maps to a simulator class in
`ai/perception/pipeline.py`, or is declared UI chrome; a test enforces that
nothing falls between the two.
