"""Behavioural cloning (Phase 16 / M12) and the BC -> PPO handoff (M13).

BC trains the *same* ActorCritic used by PPO, so `bc_to_ppo` is a state-dict
load rather than a distillation step.

The value head is deliberately *not* trained here. Demonstration returns are
returns under whatever reward the demos were recorded with, which is not
necessarily the reward PPO will optimise -- the farming profile weights loot
3.0 and a win 0.0, so a critic fitted on default-reward demos would be
confidently wrong. The random critic that PPO then starts with is handled on
the PPO side by `critic_warmup_updates`, which fits the value head against the
real reward with the actor frozen before letting the actor move. That is the
fix for the classic "BC init evaporates" failure.
"""
from __future__ import annotations

import json
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ai.imitation.demos import DemoDataset, collate
from ai.policies.actor_critic import ActorCritic


def action_logits(policy: ActorCritic, obs: dict, action: torch.Tensor) -> tuple:
    """Re-derive the four head logits for a given (obs, action) pair."""
    z, _ = policy.features(obs, None)
    lt = policy.head_type(z).masked_fill(~obs["mask_type"], -1e9)
    lu = policy.head_unit(z).masked_fill(~obs["mask_unit"], -1e9)
    row_ok = obs["mask_xy"].any(dim=2)
    lx = policy.head_x(z).masked_fill(~row_ok, -1e9)
    x_onehot = F.one_hot(action[:, 2], lx.shape[-1]).float()
    ly = policy.head_y(torch.cat([z, x_onehot], dim=-1))
    col_ok = obs["mask_xy"][torch.arange(obs["mask_xy"].shape[0]), action[:, 2]]
    ly = ly.masked_fill(~col_ok, -1e9)
    return (lt, lu, lx, ly), z


def type_class_weights(hist) -> torch.Tensor:
    """Inverse-frequency weights for the action-type head.

    ~94% of demonstration steps are WAIT, so an unweighted head hits the base
    rate by predicting WAIT unconditionally and stops -- observed at exactly
    0.9386 accuracy, the WAIT fraction. Weights are mean-normalised over the
    classes that occur so the overall loss scale does not move, and classes
    with no examples get 0 rather than infinity.
    """
    h = torch.as_tensor(hist, dtype=torch.float32)
    seen = h > 0
    w = torch.where(seen, h.clamp(min=1).reciprocal(), torch.zeros_like(h))
    return w / w[seen].mean()


def bc_loss(policy: ActorCritic, obs: dict, action: torch.Tensor,
            weights: tuple = (1.0, 1.0, 1.0, 1.0),
            type_weight: torch.Tensor | None = None) -> tuple:
    (lt, lu, lx, ly), _ = action_logits(policy, obs, action)

    # `type_weight` counteracts the class imbalance in the demonstrations:
    # ~94% of steps are WAIT, so an unweighted head reaches the base rate by
    # predicting WAIT unconditionally and stops there -- which it did, at
    # exactly 0.9386 accuracy. A warm start that never deploys is useless.
    losses = [F.cross_entropy(lt, action[:, 0], weight=type_weight)]

    # unit/x/y only exist on steps where the demonstration actually deployed.
    # On a WAIT step the recorded unit/x/y are dummy zeros, and the logit at
    # index 0 is -1e9 whenever that unit is illegal (army exhausted) -- so the
    # unconditional cross-entropy is ~1e9. A batch with no deploys in it used
    # to fall through to exactly that and backpropagate it, which is where
    # `loss_unit` of 150761 came from. Zero is the correct loss for "this head
    # took no decision in this batch".
    deploy = (action[:, 0] == 1).float()
    n_dep = deploy.sum()
    for i, logits in enumerate((lu, lx, ly), start=1):
        if n_dep > 0:
            per = F.cross_entropy(logits, action[:, i], reduction="none")
            losses.append((per * deploy).sum() / n_dep)
        else:
            losses.append(logits.sum() * 0.0)
    total = sum(w * l for w, l in zip(weights, losses))
    with torch.no_grad():
        # accuracy for unit/x/y is only meaningful on steps that actually
        # deployed -- ~94% of demonstration steps are WAIT, and averaging over
        # them makes a policy that never deploys look excellent
        n_dep = n_dep.clamp(min=1.0)
        accs = [(lt.argmax(-1) == action[:, 0]).float().mean().item()]
        for i, l in enumerate((lu, lx, ly), start=1):
            hit = (l.argmax(-1) == action[:, i]).float()
            accs.append(((hit * deploy).sum() / n_dep).item())
    return total, {"loss_type": losses[0].item(), "loss_unit": losses[1].item(),
                   "loss_x": losses[2].item(), "loss_y": losses[3].item(),
                   "acc_type": accs[0], "acc_unit": accs[1],
                   "acc_x": accs[2], "acc_y": accs[3]}


def train_bc(root: str = "datasets/demos", epochs: int = 5, batch_size: int = 128,
             lr: float = 3e-4, d_model: int = 96, n_layers: int = 1,
             memory: str | None = "gru", device: str = "cpu",
             out: str = "checkpoints/bc.pt", val_frac: float = 0.1,
             log_path: str | None = "experiments/bc_log.jsonl") -> dict:
    ds = DemoDataset(root)
    n_val = max(int(len(ds) * val_frac), 1)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate)

    dev = torch.device(device)
    hist = ds.action_stats()["type_hist"]
    type_weight = type_class_weights(hist).to(dev)
    print(f"[bc] action-type counts {hist} -> class weights "
          f"{[round(v, 3) for v in type_weight.tolist()]}", flush=True)

    policy = ActorCritic(d_model=d_model, n_layers=n_layers, memory=memory).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        policy.train()
        agg, n = {}, 0
        for obs, act in dl:
            obs = {k: v.to(dev) for k, v in obs.items()}
            act = act.to(dev)
            loss, parts = bc_loss(policy, obs, act, type_weight=type_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

        val = evaluate_bc(policy, vl, dev, type_weight=type_weight)
        row = {"epoch": epoch, **{k: round(v / max(n, 1), 4) for k, v in agg.items()},
               **{f"val_{k}": round(v, 4) for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(),
                "cfg": {"d_model": d_model, "n_heads": 4, "n_layers": n_layers,
                        "memory": memory},
                "history": history}, out)
    if log_path:
        p = pathlib.Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(r) for r in history))
    return {"checkpoint": out, "history": history}


@torch.no_grad()
def evaluate_bc(policy: ActorCritic, loader, device, type_weight=None) -> dict:
    policy.eval()
    agg, n = {}, 0
    for obs, act in loader:
        obs = {k: v.to(device) for k, v in obs.items()}
        _, parts = bc_loss(policy, obs, act.to(device), type_weight=type_weight)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def bc_to_ppo(bc_checkpoint: str, ppo_trainer) -> None:
    """M13: initialise a PPO run from a BC checkpoint.

    The policy heads are loaded as-is. The value head is *not*: BC never
    trains it, and demonstration returns are under the demos' reward rather
    than PPO's. Pair this with `PPOConfig.critic_warmup_updates` so the fresh
    critic is fitted against the real reward before the warm actor is allowed
    to move.
    """
    from ai.policies.neural_agent import _migrate

    ck = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in _migrate(ck["model"]).items() if not k.startswith("head_v.")}
    missing, unexpected = ppo_trainer.policy.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in BC checkpoint: {unexpected}"
    print(f"[bc->ppo] loaded actor weights, reinitialised critic "
          f"({len(missing)} value-head params left fresh)")
