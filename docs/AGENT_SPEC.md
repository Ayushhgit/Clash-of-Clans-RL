# AGENT_SPEC

## 1. Loop

```
SCREEN → PERCEPTION → SPATIAL → MEMORY → STRATEGIC → TACTICAL → CONTROL → GAME
```

## 2. Modules

| module | package | input | output | trained by |
|---|---|---|---|---|
| UI classifier | `ai/perception/ui_classifier.py` | RGB | screen state | supervised (Dataset B) |
| UI detector | `ai/perception/ui_detector.py` | RGB | button boxes | supervised (Dataset B) |
| OCR | `ai/perception/ocr.py` | RGB crops | numbers | supervised (Dataset C) |
| Object detector | `ai/perception/detector.py` | RGB | boxes+classes | supervised (Dataset A) |
| Spatial encoder | `ai/spatial/encoder.py` | tokens | set embedding | end-to-end w/ policy |
| Memory | `ai/memory/` | embedding seq | recurrent state | end-to-end w/ policy |
| Strategic policy | `ai/policies/strategic.py` | state | NEXT/ATTACK/objective | PPO |
| Tactical policy | `ai/policies/tactical.py` | state | factored action | PPO (+BC init) |
| Controller | `ai/control/` | action | mouse/keyboard | hand-written |

## 3. Interfaces (frozen — changing these breaks every downstream phase)

```python
class Perceiver(Protocol):
    def perceive(self, rgb: np.ndarray) -> Observation: ...

class Policy(Protocol):
    def act(self, obs: Observation, state: Any) -> tuple[Action, Any]: ...
    def value(self, obs: Observation, state: Any) -> float: ...

class Env(Protocol):
    def reset(self, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]: ...
```

`simulator.env.RaidEnv` and `game.live_env.LiveGameEnv` implement the same
`Env` protocol. That is what makes Phase 22 sim-to-game transfer a one-line
swap.

## 4. Training stages

| stage | env | obs mode | policy init | algo |
|---|---|---|---|---|
| M6 | sim | privileged | — | random |
| M7 | sim | privileged | — | heuristic |
| M8–M10 | sim | privileged | scratch | PPO + curriculum |
| M11–M13 | sim + demos | privileged | BC | BC → PPO |
| M14–M15 | sim | privileged | prev | PPO + GRU + hierarchy |
| M16 | sim (rendered) | detected | prev | PPO fine-tune |
| M17 | real game | detected | prev | eval only (+ optional fine-tune) |

### 4.1 Training the farming objective

The loot objective is trained differently from the destruction objective, for
reasons that are recorded in `docs/EXPERIMENTS.md` (E12-E16):

```bash
# 1. demonstrations across the difficulty range, not one level
python -m scripts.train_bc --demos 240 --levels L2 L3 L4 L5     --root datasets/demos_farm --out checkpoints/bc_farm.pt

# 2. warm-started PPO on a level mix
python -m scripts.train_ppo --config configs/ppo_farm_mix.yaml     --bc-init checkpoints/bc_farm.pt
```

| knob | why it exists |
|---|---|
| `levels: [L2..L5]` | replaces the curriculum. A promotion gate needs a bar, and achievable loot differs 2.5x between L3 (~0.65) and L4 (~0.26), so no single bar works. A mix also matches deployment, where bases are not sorted. |
| `promote_metric` | if you *do* use a curriculum here, gate it on loot -- the farming reward pays 0 for a win, so a win-rate gate asks the agent to clear a bar it is not being trained to clear. |
| `critic_warmup_updates` | BC leaves the value head random. Acting on those first advantages is what evaporates a warm start, so everything but `head_v` is frozen until the critic has fitted the real reward. |
| `eval_every` | held-out bases at a fixed level. Training-window numbers move with the level distribution and cannot answer "better than 100k steps ago?". |

Read `ent_live`, not `entropy`. Once the army is spent WAIT is the only legal
action, so ~90% of steps are masked to a single choice and contribute exactly
0; the plain mean reads ~0.1 on a policy that is exploring perfectly well.

## 5. Non-goals

* No pixel-to-action end-to-end CNN policy trained from scratch on the real
  game — sample cost is prohibitive.
* No reverse engineering, memory reading, or packet interception of the target
  game. Perception is screen-only; control is OS-level input only.
* No account automation beyond the attack loop under test.
