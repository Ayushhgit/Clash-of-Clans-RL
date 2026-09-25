# GAME_SPEC

Black-box specification of the target game as the agent must model it.

## 1. Genre assumption

Base-raid strategy game (Clash-of-Clans-like). The player searches for an
opponent base, deploys a fixed army around its perimeter, and units fight
autonomously until a timer expires or the base is destroyed.

This spec is written so the *simulator* (`simulator/`) reproduces only the
mechanics the agent needs. Anything the agent cannot observe or influence is
deliberately omitted.

## 2. Global state machine

```
HOME ──► SEARCH ──► BASE_PREVIEW ──┬── NEXT ──► SEARCH
                                   └── ATTACK ─► BATTLE ──┬── VICTORY ─► RESULT ─► HOME
                                                          └── DEFEAT  ─► RESULT ─► HOME
```

See `docs/STATE_MACHINE.md` for transitions, triggers and screen evidence.

## 3. Entities

### 3.1 Buildings (static, destructible)

| Class | Role | Notes |
|---|---|---|
| `TOWN_HALL` | objective | destroying it is a primary objective |
| `STORAGE` | loot | holds gold/elixir, contributes to loot reward |
| `RESOURCE_BUILDING` | loot | slow trickle, smaller loot value |
| `WALL` | blocker | high HP, blocks ground pathing, no attack |
| `OBSTACLE` | blocker | low HP decoration, blocks placement |

### 3.2 Defenses (static, destructible, hostile)

| Class | Range | Targets | Notes |
|---|---|---|---|
| `CANNON` | short | ground | single target |
| `ARCHER_TOWER` | medium | ground+air | single target |
| `MORTAR` | long, has min-range | ground | splash |
| `WIZARD_TOWER` | medium | ground+air | splash |
| `AIR_DEFENSE` | long | air only | high DPS |
| `TESLA` | short | ground+air | hidden until triggered |
| `INFERNO` | long | ground+air | ramping DPS |
| `XBOW` | long | ground (configurable) | high DPS |

### 3.3 Units (player-controlled deployment, autonomous behaviour)

Unit archetypes the agent must reason about:

| Archetype | HP | DPS | Speed | Target preference | Move |
|---|---|---|---|---|---|
| `TANK` | very high | low | slow | nearest building | ground |
| `DPS` | low | high | fast | nearest building | ground |
| `WALL_BREAKER` | very low | huge vs walls | fast | nearest wall | ground |
| `RANGED` | low | medium | medium | nearest building | ground |
| `AIR` | medium | medium | fast | nearest building | air (ignores walls) |
| `SPLASH` | medium | medium (aoe) | medium | nearest building | ground |

Behaviour rules assumed (validated in Phase 1 by observation):

1. A unit walks toward its current target at `speed` tiles/s.
2. When `dist <= range`, it stops and attacks at `dps` HP/s.
3. When its target dies, it re-acquires using its target preference.
4. Ground units that are blocked by walls either break the wall (if wall DPS
   ratio is favourable) or path around it.
5. Air units ignore walls entirely.
6. Defenses acquire the nearest valid unit inside range and fire continuously.

### 3.4 Resources / meta

* `gold`, `elixir` — loot, read via OCR from `RESULT` and `BASE_PREVIEW`.
* `army` — a fixed composition available at the start of a battle.
* `search_cost` — pressing `NEXT` costs a small amount of gold/time.

## 4. Battle rules

* Battle length: `T_MAX = 180 s` (configurable per game).
* Deployment: only outside the base's placement-blocked region (perimeter).
* Victory condition: `destruction_pct >= 50` **or** `TOWN_HALL` destroyed.
* Perfect: `destruction_pct == 100`.
* Battle ends early when all deployed units are dead **and** no units remain
  in the army reserve.
* Loot is granted proportionally to storages destroyed.

## 5. Observability

The agent sees **only an RGB screenshot** in the final system (Phase 21).
Everything in §3 must therefore be recovered by perception:

| Fact | Source |
|---|---|
| screen state | UI classifier |
| buttons | UI detector |
| gold/elixir/timer/destruction % | OCR |
| buildings/defenses/walls/units | object detector |
| unit HP, defense levels | *not directly observable* — must be inferred |

Non-observable quantities (exact HP, exact DPS, hidden Tesla positions before
trigger) are treated as latent state and handled by the memory module
(Phase 18), never by privileged access at test time.

## 6. Open questions to resolve in Phase 1

- [ ] Exact `T_MAX` for the target title.
- [ ] Whether deployment is restricted to a ring or allowed anywhere outside walls.
- [ ] Does the game pause on menu overlay?
- [ ] Action latency between click and in-game effect (needed for Phase 22).
