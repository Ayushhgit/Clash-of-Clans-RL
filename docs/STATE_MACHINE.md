# STATE_MACHINE (Phase 1)

Filled in by black-box observation. Each transition must be backed by recorded
frames in `data/raw/screenshots/` before it is trusted.

| # | from | trigger | to | screen evidence | verified |
|---|---|---|---|---|---|
| 1 | `HOME` | click ATTACK | `SEARCH` | spinner + "Searching" text | ☐ |
| 2 | `SEARCH` | auto | `BASE_PREVIEW` | loot panel + NEXT/ATTACK buttons | ☐ |
| 3 | `BASE_PREVIEW` | click NEXT | `SEARCH` | spinner returns | ☐ |
| 4 | `BASE_PREVIEW` | click ATTACK | `BATTLE` | army bar + timer appear | ☐ |
| 5 | `BATTLE` | timer 0 / all units dead | `RESULT` | stars + loot summary | ☐ |
| 6 | `BATTLE` | click END + CONFIRM | `RESULT` | confirm dialog | ☐ |
| 7 | `RESULT` | click RETURN | `HOME` | village view | ☐ |

## Distinguishing features per screen (for the UI classifier)

| state | reliable cue |
|---|---|
| `HOME` | own village layout, shop/menu rail |
| `SEARCH` | dimmed background + progress spinner |
| `BASE_PREVIEW` | `NEXT` and `ATTACK` buttons both present |
| `BATTLE` | countdown timer + army deployment bar |
| `RESULT` | star row + loot totals + `RETURN` button |

## Ambiguous / trap states to enumerate

- loading between SEARCH and BASE_PREVIEW (looks like SEARCH)
- BATTLE with the end-battle confirm overlay (looks like BATTLE)
- RESULT while the star animation plays (OCR unreliable → wait 1.0 s)
- disconnect / maintenance modal (must map to a `RECOVER` action)

## Recording protocol

1. `python scripts/record_session.py --label-hotkeys`
2. Play 20 full loops HOME → RESULT → HOME.
3. Press `1..5` during capture to stamp the screen state into `meta.jsonl`.
4. `python scripts/build_dataset_b.py` turns the session into Dataset B.
