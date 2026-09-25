# ACTION_SPEC

## 1. Two-level action space

The agent is hierarchical (Phase 19). Each level has its own action space.

### 1.1 Strategic (once per BASE_PREVIEW, and every K steps in BATTLE)

| id | action | valid in |
|---|---|---|
| 0 | `NEXT` | BASE_PREVIEW |
| 1 | `ATTACK` | BASE_PREVIEW |
| 2 | `CHOOSE_ENTRY(side)` | BASE_PREVIEW → BATTLE |
| 3 | `CHOOSE_OBJECTIVE(obj)` | BATTLE |
| 4 | `END_BATTLE` | BATTLE |

`side ∈ {N,E,S,W}`, `obj ∈ {LOOT, TOWN_HALL, DESTRUCTION}`.

### 1.2 Tactical (per battle tick)

Factored action, sampled jointly:

```python
a = (a_type, a_unit, a_x, a_y)
```

| component | type | size | meaning |
|---|---|---|---|
| `a_type` | categorical | 4 | `WAIT`, `DEPLOY_UNIT`, `USE_ABILITY`, `SELECT_UNIT` |
| `a_unit` | categorical | 6 | archetype index (TANK/DPS/WALL_BREAKER/RANGED/AIR/SPLASH) |
| `a_x` | categorical | 32 | x bin over the deploy ring |
| `a_y` | categorical | 32 | y bin over the deploy ring |

Total nominal size `4 x 6 x 32 x 32 = 24576`, but the components are
independent heads, so the policy outputs `4 + 6 + 32 + 32 = 74` logits.

`a_unit`, `a_x`, `a_y` are ignored when `a_type == WAIT`.

### 1.3 Masking

An action mask is produced by the environment every step:

```python
mask = {
  "type":  bool[4],   # DEPLOY invalid when army empty
  "unit":  bool[6],   # per-archetype remaining count > 0
  "xy":    bool[32,32]# valid deploy cells (outside walls, on map, not blocked)
}
```

Invalid logits are set to `-1e9` before the softmax. Never rely on the policy
learning validity from reward alone.

## 2. Controller / action decoder (Phase 19 bottom layer)

Maps a tactical action to OS-level events:

```
DEPLOY_UNIT(unit=DPS, x=.22, y=.71)
  → click(army_slot_rect[DPS].center)      # SELECT
  → click(map_to_screen(.22, .71))         # DEPLOY
  → sleep(deploy_cooldown)
```

| primitive | implementation |
|---|---|
| `MOUSE_MOVE(x,y)` | absolute move, then a small jitter |
| `CLICK` | down, `U(40,90) ms`, up |
| `DRAG` | for camera pan |
| `KEY_PRESS(k)` | army hotkeys where the game supports them |
| `WAIT(ms)` | no-op tick |

Human-like timing is required so the game does not reject inputs and so
behavioural cloning data and agent data share a distribution.

## 3. Timing contract

* Tactical policy runs at **5 Hz**; the simulator tick is 10 Hz, so one policy
  step = 2 sim ticks.
* Real-game latency budget: perception ≤ 60 ms, policy ≤ 15 ms, control ≤ 30 ms.
* Exceeding budget is logged as a `CONTROL FAILURE` (Phase 25).
