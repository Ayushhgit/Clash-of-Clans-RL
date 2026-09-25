# ROADMAP -- build order

Status legend: `done` = implemented and tested here; `running` = training in
progress; `blocked: data` = code complete, waiting on captured frames from the
target game; `blocked: deps` = needs a package this machine does not have.

| M | deliverable | depends on | status |
|---|---|---|---|
| M1 | screen capture + mouse/keyboard controller | - | code done; `blocked: deps` (`mss`, `pynput`) |
| M2 | screenshot -> UI state | M1, Dataset B | model + trainer done; `blocked: data` |
| M3 | screenshot -> object detection | Dataset A | detector, loss, decode, AP eval done; `blocked: data` |
| M4 | detections -> spatial tokens | M3 | done (`ai/perception/pipeline.py`) |
| M5 | fast simulator | specs | done |
| M6 | random agent in sim | M5 | done |
| M7 | heuristic agent in sim | M5 | done |
| M8 | custom PPO | M5 | done |
| M9 | PPO beats simple bases | M8 | done -- 100% on L2, beats random everywhere |
| M10 | PPO on procedural bases | M9 | done at 600k steps; matches the heuristic on L2/L4, behind it on L3/L5 (see E10) |
| M11 | human demonstrations | M1 | recorder done; scripted demos stand in |
| M12 | behavioural cloning | M11 | done (62% win vs 100% demonstrator) |
| M13 | BC -> PPO | M12 | done (`ai.imitation.bc.bc_to_ppo`) |
| M14 | temporal memory + ablation | M13 | GRU in policy; ablation runner ready |
| M15 | hierarchical policy | M14 | tactical + base-selection layers exist, not yet joint |
| M16 | pixel perception + policy | M3, M15 | path complete; `blocked: data` |
| M17 | pixel -> real game | M16, M1 | `game/live_env.py`, dry-run default |
| M18 | unseen-base evaluation | M17 | done in sim (disjoint seed ranges) |
| M19 | ablation studies | M18 | `scripts/ablation.py` ready |
| M20 | final report | M19 | `docs/EXPERIMENTS.md` accumulating |

**Rule:** no milestone starts before the previous one has a passing test in
`tests/` and a recorded number in `docs/EXPERIMENTS.md`.

## What "blocked: data" actually needs

1. Install the capture stack: `pip install mss pynput` (and a numpy-2
   compatible `opencv-python` if you want video export).
2. `python -m scripts.record_session --seconds 600 --fps 4` while playing,
   pressing 1-5 to stamp screen states.
3. `python -m annotation.app data/raw/screenshots/<session>` and label
   1 000-2 000 frames (Phase 4).
4. `python -m scripts.train_detector --data data/raw/screenshots`
5. `python -m scripts.auto_label --checkpoint checkpoints/detector_v1.pt`
   then relabel the top of the uncertainty queue and retrain (Phases 6-7).

Everything downstream of that -- tokens, policy, control -- already runs.
