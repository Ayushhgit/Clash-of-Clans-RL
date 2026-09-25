# EXPERIMENTS

Every number here was produced by a command in this repo, with fixed seeds.
Negative results are kept -- they are the parts that were actually informative.

## E1 -- Simulator and training throughput (M5, M8)

`python -m scripts.profile_ppo --sweep`

| config | collect | update | env-steps/s |
|---|---|---|---|
| 16 envs x T64, 2 layers, 4 ep / 4 mb | 4.3 s | 2.8 s | 144 |
| 32 envs x T32, 1 layer, 2 ep / 2 mb | 3.1 s | 2.7 s | 176 |
| 64 envs x T32, 1 layer, 2 ep / 2 mb | 5.7 s | 4.2 s | 208 |
| 64 envs x T32, 1 layer, 4 ep / 4 mb | 5.7 s | 12.6 s | ~110 |

Raw simulator, no policy: ~2 400 ticks/s on a small base; ~1 200 env-steps/s
on L1, ~500 on L5. A 180 s battle costs ~0.4 s of wall clock -- roughly 450x
faster than real time, which is the entire reason the RL loop does not touch
the real game.

Two model-size passes were needed to get there: 128 tokens with a 128-wide
2-layer encoder ran at 17 steps/s; 64 tokens with a 96-wide 1-layer encoder
runs at ~110-200. Token truncation drops surplus *walls* only, since tokens
are priority-ordered (own units > defenses > scoring buildings > walls).

The final config trades throughput for gradient steps: at 2 epochs /
2 minibatches the approximate KL per update was ~1e-4 and the policy was
effectively frozen. 4 ep / 4 mb costs ~2x wall clock and moves the policy
(KL ~3e-3 to 6e-3).

## E2 -- Difficulty calibration (M6, M7)

`python -m scripts.baselines --episodes 12`  (win rate, unseen seeds)

| agent | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| random | 100% | 83% | 50% | 8% | 8% |
| heuristic | 100% | 92% | 92% | 17% | 25% |

The gap opens at L3, which is where a learned policy has room to show skill.
The first balance pass had random winning everywhere: defense DPS was ~3x too
low and the army ~2x too large. Current targets are recorded in the balance
comment in `simulator/entities.py`.

L5 scoring above L4 is not a bug: L5 places the town hall off-centre, which
makes a town-hall snipe possible (a win at ~11% destruction). That is a real
strategy in this game family, and the reason Phase 17 exists.

## E3 -- Behavioural cloning (M12)

`python -m scripts.train_bc --demos 40 --level L2 --epochs 2`

40 demonstration episodes = 14 348 state-action pairs, 94% of them WAIT.

| | demonstrator | clone |
|---|---|---|
| win rate L2 | 1.00 | 0.62 |
| destruction | 0.79 | 0.47 |
| loot | 0.88 | 0.77 |

With 120 episodes / 8 epochs, per-head accuracy **on deploy steps only**:
type 0.94, unit 1.00, x 0.09, y 0.20 (chance = 0.03). Unit choice clones
almost immediately; position is the hard part, because the demonstrator picks
one entry cell per base, making x/y a base-conditional target.

Metric bug worth recording: unit/x/y accuracy must be masked to deploy steps.
Averaged over all steps, a policy that always waits scores 94%.

## E4 -- Base selection (Phase 17)

`python -m scripts.base_selection --agent heuristic --level L3 --episodes 60`

On L3 (heuristic battle policy wins 92%), always-attack is near optimal and
the learned selector correctly collapses to it: 0.429 vs 0.429 reward per
decision, 0.83 decision accuracy, always-NEXT -0.020.

L4 is the interesting case, run as full raid sessions
(`ai/policies/hierarchical.py`, 6 sessions x 15 decisions, 4 attacks max):

| strategy | reward/session | win rate among attacks | skip rate |
|---|---|---|---|
| attack everything | **+0.062** | 0.25 | 0.00 |
| learned value selector | -0.135 | 0.25 | 0.50 |
| heuristic selector | -0.343 | 0.25 | 0.70 |

**Both selectors lose to attacking everything, and neither raises the win rate
among the bases it does attack.** Held-out AUC of P(win) from the 9 base
features is 0.677 and predicted-vs-actual loot correlation is 0.50, so the
features carry some signal, but not enough to beat the cost of skipping when
the base value is roughly break-even (mean loot 0.42, attack cost 0.05,
fail penalty 0.5 x 0.74).

Feature correlations with winning (120 L4 bases):

| feature | corr |
|---|---|
| th_centrality | +0.216 |
| n_defenses | -0.206 |
| defense_density | -0.196 |
| hp_norm | -0.117 |
| loot_depth | +0.101 |

One feature (`compartments = n_walls/60`, clipped at 2) was *constant* on every
base above L2 and contributed nothing; it is replaced by `loot_depth`, the
mean distance from the map edge to the loot. The honest conclusion: aggregate
counts are the wrong representation for this decision, and the selector should
consume the spatial encoder's embedding of the base instead.

## E5 -- Failure analysis (Phase 25)

`python -m scripts.failure_report --agent heuristic --level L4 --episodes 10`

The first classifier attributed nearly every loss to GENERALIZATION_FAILURE,
because its rule was "lost on an unseen seed" -- unfalsifiable for a weak
agent. The rule now requires a *measured* train-seed win rate >= 0.6 for the
same agent before the label is applied.

## E6 -- Domain randomisation (Phase 20)

`SimConfig(randomize=True, stat_jitter=0.12)` perturbs building HP, DPS and
range and unit HP, DPS and speed per episode.

| | win | destruction | loot |
|---|---|---|---|
| fixed stats, L3 | 0.80 | 0.372 | 0.573 |
| randomised stats, L3 | 0.80 | 0.407 | 0.633 |

The heuristic is insensitive to +/-12% stat noise, which is the point: it
gives a policy trained under randomisation a chance to transfer to a game
whose true numbers we only estimated.

## E6b -- Perception noise: how much does `detected` cost? (Phase 22, sim half)

`python -m scripts.obs_gap --agent heuristic --episodes 15`, then a noise
sweep on L3 with n=40 (binomial SE ~ 0.079):

| dropout / jitter | win | destruction | loot |
|---|---|---|---|
| privileged (0 / 0) | 0.750 | 0.423 | 0.622 |
| 0.10 / 0.008 | 0.500 | 0.403 | 0.585 |
| 0.25 / 0.016 | 0.650 | 0.428 | 0.623 |
| 0.50 / 0.030 | 0.600 | 0.413 | 0.593 |

Win rate drops by roughly 0.1-0.25 under detector noise, but the effect is
**not monotone in the noise level** at n=40, so this sample size does not
support a precise number -- only "there is a cost, it is smaller than the
sampling noise between rows". Destruction and loot barely move at all.

That is expected for *this* agent: the heuristic consumes only coarse
statistics (loot centroid, defense coverage), so dropping one detection in
four changes little. A learned policy reading per-token features should be
more sensitive, which makes this the right experiment to repeat on the trained
policy -- and it is the reason `obs_mode` is an env config rather than a
separate code path.

## E7 -- Reward misalignment (the main negative result)

`experiments/ppo_v1_misaligned_reward/log.jsonl`, reward
`loot 1.0 / dest 1.0 / unit 0.3 / time 0.05 / win 2.0`, no town-hall term:

| step | level | win rate | return | entropy |
|---|---|---|---|---|
| 61k | L3 | 0.692 | 3.78 | 8.02 |
| 71k | L3 | 0.727 | 3.32 | 8.07 |
| 82k | L3 | 0.716 | 2.79 | 7.94 |
| 92k | L3 | 0.639 | 2.13 | 7.87 |
| 102k | L3 | 0.570 | 1.74 | 7.84 |
| 112k | L3 | 0.470 | 1.19 | 7.51 |

Curriculum promotion worked (L1 at 20k, L2 -> L3 at ~60k with a 97% L2 win
rate), then win rate fell monotonically for 50k steps.

Diagnosis, at 112k on unseen L3 seeds: win 0.20, **destruction 0.388**,
army committed **22/22 units**, 95% of steps are WAIT (that is just the
2.5 Hz deploy cadence). So the policy was not stalling and not under-deploying
-- its destruction matched the heuristic's 0.372 almost exactly. What it did
not do was kill town halls, and the heuristic's 0.92 win rate on L3 comes
mostly from town-hall kills.

The shaped reward had no gradient toward the town hall, so optimising it
pulled the policy toward spreading damage, which is precisely the behaviour
that prevents a town-hall kill. Destruction and win rate decoupled.

Fix: add `w_th * d(1 - town_hall_hp_frac)`, lower `w_dest` to 0.8, lower
`w_unit` to 0.2, raise `w_win` to 3.0. Rerun is `experiments/ppo_v2`.

## E8 -- The decay was mostly an action-space bug, not the reward

Adding the town-hall term (v2) reduced the decay but did not remove it:

| step | v1 win (L3) | v2 win (L3) |
|---|---|---|
| 61k | 0.692 | 0.688 |
| 71k | 0.727 | 0.778 |
| 82k | 0.716 | 0.770 |
| 92k | 0.639 | 0.680 |
| 102k | 0.570 | 0.580 |

A shaped-reward problem should not produce a *monotone* decline in the
policy's own return, which is what both runs showed. The action space is where
it came from.

The action is `(type, unit, x, y)`. On a `WAIT` step -- and once the 22-unit
army is spent, essentially every remaining step of the episode is `WAIT`,
about 80% of all steps -- `unit`, `x` and `y` are still sampled but never
reach the environment. Their log-probs were nevertheless included in the PPO
ratio, so the deploy heads received a gradient on every WAIT step: pure noise,
scaled by that episode's advantage, from four fifths of the data. The heads
that decide *where to place units* were being trained mostly on steps where
placement did not happen.

The fix is to condition the density on the action type, which is what the
factorisation actually means:

    p(a) = p(type) * [p(unit) p(x) p(y)]^[type == DEPLOY]

`ai/policies/actor_critic.py` now multiplies the unit/x/y log-probs and
entropies by the deploy indicator; `tests/test_env_and_rl.py::
test_logprob_ignores_action_components_that_do_not_apply` locks it in. The BC
loss already masked these components (E3), which is what made the discrepancy
visible.

Rerun: `experiments/ppo_v3`.

Early v3 trace (training-seed win rate over the last 100 episodes):

| step | level | win | return |
|---|---|---|---|
| 41k | L2 | 0.974 | 6.45 |
| 61k | L3 | 0.774 | 5.19 |
| 71k | L3 | 0.769 | 4.62 |
| 82k | L3 | 0.767 | 4.13 |

Compared with v2 at the same steps (0.688 / 0.778 / 0.770 then falling to
0.580 by 102k), v3 holds. Entropy is no longer comparable across versions:
it now only counts the heads that are actually used, so it reads ~0.3-0.8
instead of ~8.

## E9 -- Unseen-base evaluation (M18)

Same v3 checkpoint (~82k steps), 25 episodes each, L3:

| seeds | win | destruction | loot |
|---|---|---|---|
| training range (0-200k) | 0.56 | 0.464 | 0.528 |
| evaluation range (10M+) | 0.52 | 0.475 | 0.545 |

**No generalisation gap.** Each training seed is visited roughly once
(~800 distinct bases in 82k steps), so there is nothing to memorise, and the
procedural generator's train/eval split behaves as designed.

It also shows the training-time metric reads high: the same policy logs
~0.77 during rollouts but measures 0.52-0.56 when evaluated. The rollout
number is a trailing 100-episode average over a policy that is still changing,
and `ppo_latest.pt` lags the live weights by up to 20 updates, so benchmark
numbers must come from `scripts/evaluate.py`, never from the training log.

## Current standing (v3 at ~82k steps, 20-25 unseen bases per cell)

| agent | L2 | L3 | L4 |
|---|---|---|---|
| random | 95% | 55% | 10% |
| heuristic | 100% | 85% | 20% |
| PPO v3 @82k | 80% | 35-52% | 10% |

PPO is between random and heuristic on L3 and has not yet overtaken either.
82k steps is roughly 800 episodes of experience for a policy that has to learn
placement over a 32x32 grid; the run is 600k steps and CPU-bound at ~130
steps/s. The honest statement today is "the training loop is correct and
stable after E7/E8; it is not yet trained to competitive strength".

## E10 -- PPO v3, final (600k steps)

`python -m scripts.evaluate --episodes 40 --levels L2 L3 L4 L5     --checkpoint experiments/ppo_v3/ppo_final.pt`

Win rate, 40 unseen bases per cell, privileged observations:

| agent | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| random | 85% | 38% | 2% | 5% |
| heuristic | 100% | 70% | 12% | 22% |
| PPO v3 | 100% | 62% | 12% | 10% |

Curriculum: L1 -> L2 (20k) -> L3 (60k) -> L4 (~300k). On L4 the win rate rose
from 0.09 at 380k to 0.18 at 594k, so it was still improving when the budget
ran out.

**PPO destroys more than the heuristic on every single level, and still wins
less on L3 and L5:**

| level | destruction (heuristic / PPO) | win rate (heuristic / PPO) |
|---|---|---|
| L2 | 0.824 / **0.871** | 1.00 / 1.00 |
| L3 | 0.422 / **0.502** | **0.70** / 0.62 |
| L4 | 0.228 / **0.268** | 0.12 / 0.12 |
| L5 | 0.160 / **0.184** | **0.22** / 0.10 |

This is the same decoupling as E7, surviving the `w_th` fix: a win needs
*either* 50% destruction *or* the town hall, and the heuristic goes for the
town hall while PPO spreads damage. The town-hall term narrowed the gap
(v1 collapsed to 0.20 on L3; v3 holds 0.62) but did not close it. L5 is the
clearest case -- the level where a town-hall snipe wins at ~11% destruction is
exactly where PPO's higher destruction converts worst.

Next thing to try is not more steps: it is making the win condition visible in
the reward at the right granularity, e.g. shaping on *distance of the nearest
unit to the town hall* rather than only on town-hall damage, so the policy gets
gradient before it lands a hit.

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| PPO + BC init | | | | | |
| PPO (detected obs) | | | | | |

## E11 -- The objective changed to loot, and concentration still wins

Stars are worth nothing; the goal is loot per raid
(`RewardWeights.farming()`). Scored on loot, 30 unseen bases per cell:

| agent | L3 loot | L4 loot |
|---|---|---|
| random | 0.443 | 0.273 |
| heuristic (funnel, whole army at one entry) | **0.610** | **0.364** |
| farming agent, 3 units per target | 0.508 | 0.331 |
| farming agent, 5 units per target | 0.564 | 0.352 |
| farming agent, 8 units per target | 0.472 | 0.312 |

The purpose-built farming agent -- rank loot buildings by value discounted by
depth and defensive coverage, send a few units at a time -- **loses to the
generic heuristic on its own metric**, at every split tested.

The reason is a mechanic, not a bug: units auto-acquire the nearest building
after landing, so the deploy point is the only thing a policy controls.
Splitting the army across several loot targets means each group crosses
defended ground alone and dies before finishing a building; concentrating it
at one entry gets units inside, where they clear storages as a side effect of
advancing. "Snipe the exposed collector" only pays when collectors are cheap
relative to the army, and at these stats they are not (collector 250 loot vs
storage 1000).

First version of the farming agent was worse still (0.529 on L3) because
`_entities` only exposed storages and town halls, so it never even considered
collectors.

## E12 -- The "decaying" farming run was not decaying

`ppo_farm` looked like it was falling apart: return 2.99 -> 0.77 over 570k
steps. It was not. The metric window was a plain rolling tail over the last
100 episodes while the curriculum moved underneath it, so every promotion
mixed the *old* level's easy returns into the *new* level's average and the
mixture decayed as the old episodes aged out.

Split by level, the same log reads the opposite way:

| level | first (contaminated) | pure | trend |
|---|---|---|---|
| L3 | 3.29 | 1.35 -> 1.86 | improving, win 0.45 -> 0.68 |
| L4 | 1.84 | 0.82 -> 0.77 | flat, i.e. L4's actual difficulty |

Fixed by tagging every finished episode with the level it was played on and
filtering the window to the current level (`PPO._window`). A rolling average
across a distribution shift is not a measurement.

The real finding underneath was different and worse: after 573k steps PPO
*loses to the hand-written heuristic on PPO's own objective* -- L3 return 1.62
vs 1.96, loot 0.547 vs 0.654. Under-optimised, not mis-optimised.

## E13 -- Efficiency reward paid the agent to do nothing

`_unit_efficiency` was `clip(loot / max(spent, 1e-3), 0, 2) / 2`. Landing one
unit that survives and steals 2% of the loot divides by ~0 and scores
*maximum* efficiency. Scored against actually committing the army:

| level | full commit | one-unit chip |
|---|---|---|
| L3 | 1.574 | 0.510 |
| L4 | 0.533 | 0.510 |
| L5 (est.) | 0.152 | **0.510 -- chipping wins** |

So the curriculum was walking the agent into a local optimum of standing
still, and it would have arrived on L5. Fixed by flooring the denominator at
`MIN_SPEND = 0.2` -- below a fifth of the army you are not attacking. Chipping
now scores 0.06, and the genuinely ideal farm (loot 0.8 for 0.2 of the army)
scores highest at 2.81. Pinned by
`test_chipping_never_beats_committing_the_army`.

## E14 -- Entropy near zero was masking, not collapse

Logged entropy sat at 0.07-0.3 for the whole run, which reads as a collapsed
policy. It was an artifact: once the army is spent, WAIT is the only legal
action, the type distribution is masked to one choice and contributes exactly
0. ~95% of steps are like that, so the mean is ~0 no matter how the policy
explores.

`ent_live` -- entropy over steps where a deploy is still legal -- reads 3.5-5
nats on the same policy. Exploration was never the problem. Reported as NaN
rather than 0 when no minibatch held a deployable step, because "could not
measure" is a different claim from "there was none".

## E15 -- Behavioural cloning was being fed a 1e9 loss

Training BC on 240 mixed-level heuristic raids logged `loss_unit` of 75380
and then 150761, while validation loss for the same head was 0.0000.

`bc_loss` conditions the unit/x/y losses on steps that actually deployed --
but only inside `if deploy.sum() > 0`. A batch with *no* deploys in it (common
late in an episode, once the army is spent) fell through to the unconditional
cross-entropy, where the recorded unit label is a dummy `0` and the logit at
index 0 is `-1e9` because that unit is illegal. Loss ~1e9, backpropagated,
gradient-clipped to norm 1 -- so that step's entire gradient was garbage.

The same batch also showed `acc_type = 0.9386`, which is exactly the WAIT
fraction of the dataset (80362/85642). The action-type head had learned
nothing beyond the majority class; a warm start that never deploys is not a
warm start.

Two fixes, both pinned by tests:

* no-deploy batches contribute exactly 0 for unit/x/y, not the dummy-label CE
  (`test_bc_loss_is_finite_when_a_batch_contains_no_deploys`);
* inverse-frequency class weights on the type head
  (`type_class_weights`), mean-normalised so the loss scale does not move.

This is the same class of bug as E8: a factored action space where components
that did not apply were still contributing to the objective.

## E16 -- The BC warm start had no critic

`bc.py`'s module docstring claimed the value head was trained on discounted
demonstration returns. It was not -- `bc_loss` computes four action losses and
no value loss, and `DemoDataset` never even loaded the recorded rewards.
`bc_to_ppo` then stripped `head_v` citing a training that never happened.

Fitting the critic on demonstrations would be wrong anyway: those returns are
under the demos' reward, and the farming profile weights loot 3.0 and a win
0.0. Instead `PPOConfig.critic_warmup_updates` fits the value head against the
*real* reward for the first N updates with every other parameter frozen, then
unfreezes the actor. The freeze covers the shared trunk too, so the encoder
cannot drift while the critic is still noise.

## E17 -- A level mix trained the policy to specialise on the easy level

Warm-started from BC and trained on `[L2, L3, L4, L5]` for 256k steps, the
held-out eval barely moved (0.531, 0.478, 0.504, 0.554, 0.468 -- mean 0.507,
std 0.032). Split by level it was not flat at all:

| level | first eval | last eval |
|---|---|---|
| L2 | 0.89 | **0.95** |
| L3 | 0.68 | 0.52 |
| L4 | 0.27 | 0.26 |
| L5 | 0.28 | 0.13 |

It got better at the level that was already solved and worse at every other
one, ending *below* the behavioural cloning it started from (BC L3 0.627 /
L4 0.332 against PPO 0.499 / 0.267).

Cause: `norm_adv` whitens each minibatch as a single pool. Returns differ about
tenfold across the mix (L2 ~3.5, L5 ~0.2) and advantage magnitude follows
reward magnitude, so the easy level supplied most of the gradient in every
update. Introduced by the switch from a curriculum to a mix in E12 -- the
curriculum never had two levels in one batch, so the bug had nowhere to show.

Fixed by whitening advantages **within each level** before flattening
(`PPO._normalise_per_level`). Single-level runs keep per-minibatch
normalisation so their tuning is untouched. L2 was also dropped from the mix:
at 0.95 eval loot it is solved, and it was consuming a quarter of the envs.

Method note: the first read of "PPO is below BC" came from a 20-episode
benchmark, which is not enough to rank agents here -- the *same* heuristic
measured 0.651 and 0.512 on two different 20-episode draws. The finding stands
on the per-level eval trend, which is systematic rather than noise-shaped, not
on that benchmark.

## E18 -- The loot baseline, measured properly

60 episodes per cell on evaluation seeds, `--metric mean_loot`:

| agent | L3 | L4 | L5 |
|---|---|---|---|
| random | 0.420 | 0.220 | 0.129 |
| heuristic | 0.589 | 0.328 | **0.269** |
| bc (cloned from the heuristic) | 0.582 | **0.364** | 0.216 |

BC lands on its demonstrator on L3, beats it on L4, and trails it on L5, so
the cloning is doing more than copying. It costs 35ms per action against the
heuristic's 0.09ms, which does not matter at 2.5Hz.

**Episode count matters more than it looks.** Earlier 20-episode draws put the
heuristic at 0.651 and at 0.512 on L3, and BC at 0.627 -- against 0.589 and
0.582 here. Twenty episodes cannot rank these agents; conclusions in this file
that rest on 20-episode benchmarks should be re-read with that in mind.

This is the bar any learned policy has to clear to be worth deploying.

## Open questions

- [ ] Does the realigned reward hold win rate as destruction rises?
- [ ] GRU vs frame stacking vs transformer vs none
      (`scripts/ablation.py --variants full no_memory mem_stack mem_transformer`).
- [ ] Win-rate cost of `obs_mode` privileged -> detected at fixed weights:
      the simulated half of the Phase 22 gap.
- [ ] Does curriculum promotion beat training directly on L3?
- [ ] Would a spatial-embedding base selector beat always-attack on L4?
