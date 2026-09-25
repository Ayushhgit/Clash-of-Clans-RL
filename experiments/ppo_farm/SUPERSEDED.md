# Superseded -- do not resume or benchmark this run

Trained under the pre-fix `_unit_efficiency`, whose denominator was floored at
`1e-3`. That paid maximum efficiency for landing one unit that survives and
steals ~2% of the loot: 0.510 against 0.533 for committing the whole army on
L4, and a strict win on L5 (see `docs/EXPERIMENTS.md` E13). Whatever this
checkpoint learned, it learned against that incentive.

It was also curriculum-gated on win rate while the farming reward pays 0.0 for
a win, so it stalled on L4 at 0.14 wins with loot still improving, and could
never have reached L5 (E12).

Replacement: `experiments/ppo_farm_mix`, warm-started from
`checkpoints/bc_farm.pt` and trained on a level mix.

    python -m scripts.train_ppo --config configs/ppo_farm_mix.yaml \
        --bc-init checkpoints/bc_farm.pt
