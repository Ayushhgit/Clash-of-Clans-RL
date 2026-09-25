# autonomous-strategy-agent

A vision -> spatial reasoning -> hierarchical RL agent for a base-raid strategy
game, built to run from pixels only.

```
SCREEN -> PERCEPTION -> SPATIAL -> MEMORY -> STRATEGIC -> TACTICAL -> CONTROL -> GAME
                                                                                  |
                                                                               REWARD
```

The project is three learning problems kept deliberately separate:

| | question | package |
|---|---|---|
| perception | what am I looking at? | `ai/perception/` |
| representation | where is everything, and how does it relate? | `ai/spatial/`, `ai/memory/` |
| decision | what should I do? | `ai/policies/`, `ai/planning/`, `ai/rl/` |

## Status

Runs today, no game needed: the fast simulator, procedural base generation,
random/heuristic baselines, a from-scratch PPO with curriculum learning,
behavioural cloning, base selection, failure analysis, rendering, and the
benchmark harness. 59 tests pass.

Waiting on captured frames from the target game: detector training, UI
classifier training, OCR training, and the live-game loop. Their code, losses,
metrics and training scripts are written and unit-tested against synthetic
inputs -- see `docs/ROADMAP.md` for the exact five commands that unblock them.

Measured results live in `docs/EXPERIMENTS.md`.

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=.                       # Windows: $env:PYTHONPATH="."

pytest -q                                 # 59 tests

python -m scripts.baselines --episodes 30                 # M6 + M7 baselines
python -m scripts.train_ppo --config configs/ppo_small.yaml   # M8-M10
python -m scripts.evaluate --checkpoint experiments/ppo_v1/ppo_latest.pt
python -m scripts.render_episode --agent heuristic --level L3 # writes a GIF
```

Other entry points:

```bash
python -m scripts.train_bc --demos 150 --level L2      # M11-M12
python -m scripts.base_selection --level L4            # Phase 17
python -m scripts.failure_report --agent heuristic --level L4   # Phase 25
python -m scripts.ablation --steps 200000              # Phase 24 (expensive)
python -m scripts.profile_ppo --sweep                  # throughput
```

With the game running:

```bash
pip install mss pynput
python -m scripts.record_session --seconds 600 --fps 4 # Phase 2
python -m annotation.app data/raw/screenshots/<session>  # Phase 3-4
python -m scripts.train_detector --data data/raw/screenshots   # Phase 5
python -m scripts.auto_label --checkpoint checkpoints/detector_v1.pt  # Phase 6-7
python -m scripts.record_demos --seconds 600           # Phase 15
```

## Read the specs first

`docs/` is the contract the rest of the code implements. Changing an interface
there means changing every downstream phase, so they are treated as frozen.

| file | contents |
|---|---|
| `GAME_SPEC.md` | entities, mechanics, victory conditions, what is observable |
| `OBSERVATION_SPEC.md` | the token schema, and how PRIVILEGED/DETECTED differ |
| `ACTION_SPEC.md` | the factored action space, masking rules, timing contract |
| `REWARD_SPEC.md` | shaped + terminal reward, and the reward-hacking watchlist |
| `AGENT_SPEC.md` | module interfaces and the training-stage table |
| `STATE_MACHINE.md` | the UI state machine to verify during Phase 1 |
| `ROADMAP.md` | milestone status and what unblocks the blocked ones |
| `EXPERIMENTS.md` | every measured result, with the command that produced it |

## Layout

| dir | contents |
|---|---|
| `simulator/` | headless raid simulator, procedural generator, RL env, vector env |
| `ai/spatial/` | set-transformer entity encoder |
| `ai/policies/` | random, heuristic, and the factored actor-critic |
| `ai/rl/` | rollout buffer, GAE, PPO (implemented from scratch) |
| `ai/imitation/` | demonstration recording, behavioural cloning, BC -> PPO |
| `ai/planning/` | base-selection (ATTACK vs NEXT) agent |
| `ai/perception/` | detector, UI classifier, OCR, pixels -> tokens pipeline |
| `ai/control/` | screen capture, input synthesis, action decoder |
| `annotation/` | the bounding-box labelling tool (Tkinter, no extra deps) |
| `game/` | the live-game environment behind the simulator's interface |
| `evaluation/` | benchmark harness, failure analysis |
| `visualization/` | battle renderer, training-curve plotter |
| `experiments/` | run outputs, one directory per run |

## Design decisions worth knowing

* **The simulator is the RL environment, never the real game.** A 180 s battle
  costs ~0.4 s headless. Sim-to-game transfer is measured (Phase 22), not
  assumed.
* **Walls do not count toward destruction %.** Otherwise the optimal policy is
  to chew walls forever -- see the reward-hacking table in `REWARD_SPEC.md`.
* **x and y are sampled autoregressively.** The 32x32 deploy mask cannot be
  expressed as a product of independent x and y marginals.
* **Train and evaluation base seeds are disjoint by construction**
  (`simulator.generator.split_seeds`), so a memorised layout cannot inflate a
  benchmark number.
* **The same `act(obs)` interface** is used by random, heuristic, BC and PPO
  agents, in privileged, detected and live modes. Swapping any of them is a
  one-line change, which is what makes the ablations honest.
