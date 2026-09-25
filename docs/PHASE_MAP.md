# PHASE_MAP -- the 25-phase plan, mapped to code

| phase | what it asks for | where it lives | state |
|---|---|---|---|
| 0 | formal specs | `docs/*_SPEC.md` | done |
| 1 | black-box analysis, UI state machine | `docs/STATE_MACHINE.md` | template, needs play sessions |
| 2 | screen capture | `ai/control/capture.py`, `scripts/record_session.py` | done, needs `mss` |
| 3 | annotation tool | `annotation/app.py` | done (Tkinter, no extra deps) |
| 4 | first labelled dataset | `ai/perception/dataset.py` loaders | needs human labelling |
| 5 | train detector #1 | `ai/perception/detector.py`, `scripts/train_detector.py` | done, needs data |
| 6 | automatic labelling | `scripts/auto_label.py` | done, needs a detector |
| 7 | active learning | `scripts/auto_label.py` (uncertainty queue) | done |
| 8 | UI perception + OCR | `ai/perception/ui.py` | done, needs data |
| 9 | spatial representation | `ai/spatial/encoder.py` | done |
| 10 | fast simulator | `simulator/core.py` | done |
| 11 | procedural maps | `simulator/generator.py` | done, 5 levels |
| 12 | privileged RL agent | `EnvConfig.obs_mode="privileged"` | done |
| 13 | PPO from scratch | `ai/rl/ppo.py`, `ai/rl/buffer.py` | done |
| 14 | curriculum learning | `PPO._maybe_promote` | done |
| 15 | human demonstrations | `scripts/record_demos.py` | done, needs `pynput` + game |
| 16 | behavioural cloning | `ai/imitation/bc.py` | done |
| 17 | base-selection agent | `ai/planning/base_selection.py` | done |
| 18 | temporal memory | `ActorCritic(memory="gru")` | done, ablation pending |
| 19 | hierarchical RL | `ai/policies/hierarchical.py` | done (session env + wrapper) |
| 20 | domain randomisation | `SimConfig.randomize`, `DetectionDataset` augment, `obs_mode="detected"` | done |
| 21 | pixel-only agent | `ai/perception/pipeline.py` -> same policy | path complete, needs detector |
| 22 | sim-to-game transfer | `game/live_env.py` + `obs_mode` comparison | harness done, needs game |
| 23 | evaluation benchmark | `evaluation/benchmark.py`, `scripts/evaluate.py` | done |
| 24 | ablation studies | `scripts/ablation.py` | runner done |
| 25 | failure analysis | `evaluation/failure_analysis.py` | done |

## Three deliberate deviations from the plan

**1. Policy runs at 2.5 Hz in training, 5 Hz in the game.**
The plan specifies 5 Hz. Halving the control rate halves the steps per
episode, which roughly doubles the number of complete episodes a CPU-bound
run can see. The policy is rate-agnostic (elapsed time is an input), and the
gap is measurable via `EnvConfig.ticks_per_step`. See `ACTION_SPEC.md` s3.

**2. x and y are sampled autoregressively, not independently.**
The plan's action space lists x and y as separate components. Sampled
independently, the policy can produce (x, y) pairs the 32x32 deploy mask
forbids, and no masking of the marginals can prevent it. `head_y` is
conditioned on the sampled x, so every sampled action is legal by
construction.

**3. The detector is CenterNet, not YOLO.**
Anchor-free, ~200 lines, no anchor tuning and no NMS -- peak extraction
instead. The targets here are small, dense and axis-aligned, which is the
regime CenterNet handles well, and owning the loss (`focal_loss`,
`reg_l1_loss`) is closer to the spirit of the exercise than calling into a
detection package.

## What is genuinely not done yet

* Joint training of the strategic and tactical layers. They compose
  (`HierarchicalAgent`) and are evaluated together (`RaidSession`), but the
  selector is trained on the tactical policy's *outcomes*, not through it.
* A trap class in the simulator. `TRAP` exists in the annotation vocabulary
  and maps to `OBSTACLE`; hidden-trap mechanics are not simulated.
* Camera control. The agent assumes the whole base is visible; panning is in
  `ACTION_SPEC` as a control primitive but no policy uses it.
* Video recording of live sessions (`data/raw/videos/` is unused until a
  numpy-2-compatible OpenCV is installed).
