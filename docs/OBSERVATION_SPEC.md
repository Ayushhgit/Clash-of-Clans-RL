# OBSERVATION_SPEC

Three observation modes exist. They share one schema so a policy trained on
one can be evaluated on another.

```
PRIVILEGED  (Phase 12)   simulator ground truth  ── training only
DETECTED    (Phase 21)   detector output          ── the deployment mode
PIXEL       (Phase 21)   raw RGB                  ── input to the detector
```

## 1. Canonical observation

```python
Observation = {
  "screen_state": int,           # index into ScreenState enum
  "tokens":       float32[N, D], # entity tokens, zero-padded
  "token_mask":   bool[N],       # True where a token is valid
  "globals":      float32[G],    # scalar context
}
```

`N = 128` (max entities kept, nearest-to-centroid truncation), `D = 16`, `G = 12`.

## 2. Entity token layout (`D = 16`)

| idx | field | range | notes |
|---|---|---|---|
| 0 | `x` | [0,1] | normalised by map width |
| 1 | `y` | [0,1] | normalised by map height |
| 2 | `w` | [0,1] | bbox width (detector) / footprint (sim) |
| 3 | `h` | [0,1] | bbox height |
| 4 | `hp_frac` | [0,1] | 1.0 if unknown (DETECTED mode) |
| 5 | `is_alive` | {0,1} | |
| 6 | `is_mine` | {0,1} | 1 for own units |
| 7 | `range_norm` | [0,1] | 0 if unknown |
| 8 | `dps_norm` | [0,1] | 0 if unknown |
| 9 | `confidence` | [0,1] | 1.0 in PRIVILEGED mode |
| 10..15 | `class_embed` | — | 6-dim learned class id is passed separately; slots reserved |

Class identity is **not** packed into the token as a one-hot. It is passed as
an `int64[N]` `class_ids` array and embedded by the spatial encoder, so the
vocabulary can grow without changing `D`.

```python
Observation["class_ids"]: int64[N]   # 0 = PAD
```

## 3. Globals (`G = 12`)

| idx | field |
|---|---|
| 0 | `time_frac` = elapsed / T_MAX |
| 1 | `destruction_pct` |
| 2 | `town_hall_down` |
| 3..8 | remaining army count per archetype, normalised |
| 9 | `units_alive_frac` |
| 10 | `loot_collected_norm` |
| 11 | `deploy_budget_frac` |

## 4. Mode differences

| Field | PRIVILEGED | DETECTED |
|---|---|---|
| `hp_frac` | exact | 1.0 (or memory estimate) |
| `range/dps` | exact | class prior lookup |
| hidden `TESLA` | present | absent until triggered |
| `confidence` | 1.0 | detector score |
| false positives | none | possible |
| missed entities | none | possible |

Phase 22 measures the performance gap caused exactly by this table.

## 5. Frame / pixel observation

```python
Frame = {
  "frame_id": int, "timestamp": float, "screen_state": str,
  "rgb": uint8[H, W, 3],   # native capture, downscaled to 1280x720 for storage
}
```

Stored as `data/raw/screenshots/{session}/{frame_id:08d}.png` plus one
`meta.jsonl` per session with one JSON object per frame.

## 6. Normalisation

All coordinates are normalised to the **battle map**, not the screen, once
camera position is known. Until camera estimation exists (Phase 9), screen
coordinates are used and the camera offset is treated as observation noise —
domain randomisation (Phase 20) covers this deliberately.
