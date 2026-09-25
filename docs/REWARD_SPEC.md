# REWARD_SPEC

## 0. The objective changed

The brief listed five objectives. The actual goal is **loot per raid** --
stars, town halls and destruction percentage are worth nothing except as
means to loot. `RewardWeights.farming()` encodes that:

```python
loot 3.0   dest 0.1   th 0.0   unit 0.2   time 0.05
win 0.0    perfect 0.0   fail 0.0   efficiency 0.5 (measured on loot)
```

`dest` keeps a small weight only as a dense proxy that correlates with
reaching storages; `unit` charges the army so a cheap partial raid beats an
expensive full clear. Everything below describes the original win-seeking
profile, which is still the default in `RewardWeights()` and still used by the
win-rate benchmark.

## 1. Objectives (original brief)

1. maximise loot
2. maximise destruction
3. minimise unit loss
4. complete objective (town hall / 50%)
5. avoid failed attacks

## 2. Shaped per-step reward (battle)

```
r_t =  w_loot   * d(loot_frac)
     + w_dest   * d(destruction_frac)
     + w_th     * d(1 - town_hall_hp_frac)
     - w_unit   * d(unit_hp_lost_frac)
     - w_time   * dt / T_MAX
```

Defaults (`configs/ppo_small.yaml`, `simulator.env.RewardWeights`):

| weight | value | rationale |
|---|---|---|
| `w_loot` | 1.0 | loot fraction is in [0,1] |
| `w_dest` | 0.8 | destruction fraction is in [0,1] |
| `w_th` | 0.5 | **see below** -- destroying the town hall wins outright, so it needs its own gradient |
| `w_unit` | 0.2 | losing the whole army costs 0.2 |
| `w_time` | 0.05 | mild pressure, prevents stalling |

### Why `w_th` exists (measured, not assumed)

The first version had no town-hall term and `w_dest = 1.0`. Over 110k steps of
PPO on L3, destruction stayed flat (~0.39) while **win rate fell from 0.73 to
0.20**: the policy was spreading its army to farm destruction increments, and
spreading is exactly what stops a town-hall kill. Since a town-hall kill is a
win condition on its own, a reward with no gradient toward it is misaligned
with the objective it is supposed to encode. `docs/EXPERIMENTS.md` E7 has the
trace.

All deltas are **potential-based differences of normalised quantities**, so
per-episode shaped return is bounded and comparable across base sizes.

## 3. Terminal reward

```
R_T =  w_win   * win
     + w_perf  * (destruction == 1.0)
     - w_fail  * (destruction < 0.5)
     + w_eff   * unit_efficiency
```

| weight | value |
|---|---|
| `w_win` | 3.0 |
| `w_perf` | 1.0 |
| `w_fail` | 1.0 |
| `w_eff` | 0.5 |

`unit_efficiency = destruction_frac / max(army_hp_spent_frac, eps)`, clipped to
[0, 2] then rescaled to [0,1].

## 4. Base-selection reward (Phase 17)

Separate MDP, one decision per base:

```
NEXT   →  r = -search_cost                (default 0.02)
ATTACK →  r = loot_gained - attack_cost - fail_penalty * (not win)
```

The agent must learn expected value: attacking a rich base it cannot beat must
score worse than pressing `NEXT`.

## 5. Reward hacking watchlist

Explicitly monitored in Phase 25. Known exploits of this reward:

| exploit | symptom | mitigation |
|---|---|---|
| deploy nothing, farm `w_time` sign error | zero destruction, full timer | `w_time` is strictly a cost |
| snipe one storage, quit | high loot/low destruction | terminal `w_fail` |
| dump army on walls | destruction rises, loot doesn't | walls excluded from destruction % |
| stall to avoid unit loss | low destruction, army unspent | time cost + terminal fail; measured: the trained policy deploys 22/22 units, so this exploit did not appear |
| spread damage to farm destruction increments | destruction flat, win rate falls | `w_th` gives explicit credit for town-hall progress (E7) |

**Walls and obstacles do not count toward `destruction_pct`.** This is a
deliberate anti-hacking decision, mirroring the real game.

## 6. Curriculum-aware normalisation

Reward is *not* rescaled per curriculum level. A harder level genuinely yields
less reward; that is the signal the promotion criterion uses
(`mean_win_rate >= 0.7` over 200 eval episodes → promote).
