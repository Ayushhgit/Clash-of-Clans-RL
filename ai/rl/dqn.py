"""Branching DQN over the factored (type, unit, x, y) action space.

Why branching rather than plain DQN: the action space is
4 x 6 x 16 x 16 = 6144 combinations. A single Q head over all of them needs an
example of every combination to learn anything, and most are illegal on any
given step. Branching Dueling Q-Networks (Tavakoli et al., 2018) keep one Q
head per action *dimension* and share a state value, so the network learns
4 + 6 + 16 + 16 = 42 outputs instead of 6144, and the argmax factorises.

The composite value is the mean of the branch values, which is what makes the
bootstrap consistent: each branch is regressed onto the same TD target, so a
branch cannot chase a return the others cannot deliver.

Two deliberate differences from the PPO setup, both of which matter when
reading the comparison:

* **No recurrence.** PPO uses a GRU; a replay buffer of independent
  transitions cannot carry hidden state without sequence replay. DQN here is
  feed-forward, so it cannot represent "how long since I deployed".
* **Off-policy.** Transitions are replayed many times, which is the reason DQN
  is usually more sample-efficient -- and the reason it is more fragile when
  the masking changes what is legal from step to step.

Masks are applied to the Q values, not only at action selection: an illegal
action must never be the argmax used to form a target either.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ai.spatial.encoder import SpatialEncoder
from ai.memory.temporal import TemporalMemory
from simulator.env import ACT_DEPLOY, EnvConfig, N_ACT_TYPES, XY_BINS
from simulator.entities import NUM_ARCHETYPES
from simulator.vec_env import SyncVectorEnv, batch_obs

NEG = -1e9


@dataclass
class DQNConfig:
    total_steps: int = 300_000
    num_envs: int = 16
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 5_000
    train_every: int = 4
    target_sync: int = 2_000
    gamma: float = 0.995
    lr: float = 3e-4
    max_grad_norm: float = 5.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4      # fraction of training spent annealing
    double: bool = True
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 1
    device: str = "cpu"
    torch_threads: int = 10
    seed: int = 0
    level: str = "L3"
    levels: list | None = None
    run_dir: str = "experiments/dqn"
    log_every: int = 20
    eval_every: int = 100
    eval_episodes: int = 16
    env: EnvConfig = field(default_factory=EnvConfig)


class BranchQ(nn.Module):
    """Shared encoder, one duelling Q head per action dimension."""

    def __init__(self, d_model: int = 96, n_heads: int = 4, n_layers: int = 1,
                 trunk: int = 192):
        super().__init__()
        self.encoder = SpatialEncoder(d_model=d_model, n_heads=n_heads,
                                      n_layers=n_layers)
        self.memory = TemporalMemory(None, d_model)
        self.trunk = nn.Sequential(nn.Linear(self.memory.out_dim, trunk), nn.GELU())
        self.value = nn.Sequential(nn.Linear(trunk, 128), nn.GELU(), nn.Linear(128, 1))
        # named attributes, not a ModuleDict: "type" collides with
        # nn.Module.type() and add_module refuses it
        self.adv_type = _head(trunk, N_ACT_TYPES)
        self.adv_unit = _head(trunk, NUM_ARCHETYPES)
        self.adv_x = _head(trunk, XY_BINS)
        self.adv_y = _head(trunk, XY_BINS)

    def forward(self, obs: dict) -> dict:
        state, _ = self.encoder(obs["tokens"], obs["class_ids"],
                                obs["token_mask"], obs["globals"])
        state, _ = self.memory(state, None)
        z = self.trunk(state)
        v = self.value(z)
        out = {}
        for k, head in (("type", self.adv_type), ("unit", self.adv_unit),
                        ("x", self.adv_x), ("y", self.adv_y)):
            a = head(z)
            # duelling: advantages are zero-mean within a branch, so the shared
            # V carries the level of the return and the branch carries only
            # the relative merit of its own choices
            out[k] = v + a - a.mean(dim=-1, keepdim=True)
        return out


def _head(inp: int, out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(inp, 128), nn.GELU(), nn.Linear(128, out))


def masked(q: dict, obs: dict) -> dict:
    """Illegal actions get -inf, in both selection and target formation."""
    row_ok = obs["mask_xy"].any(dim=2)
    col_ok = obs["mask_xy"].any(dim=1)
    return {
        "type": q["type"].masked_fill(~obs["mask_type"], NEG),
        "unit": q["unit"].masked_fill(~obs["mask_unit"], NEG),
        "x": q["x"].masked_fill(~row_ok, NEG),
        "y": q["y"].masked_fill(~col_ok, NEG),
    }


class ReplayBuffer:
    def __init__(self, size: int, obs_spec: dict, device):
        self.size = size
        self.device = device
        self.n = 0
        self.i = 0
        self.obs = {k: np.zeros((size, *shape), dtype=dt)
                    for k, (shape, dt) in obs_spec.items()}
        self.next_obs = {k: np.zeros((size, *shape), dtype=dt)
                         for k, (shape, dt) in obs_spec.items()}
        self.actions = np.zeros((size, 4), dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done) -> None:
        n = len(reward)
        idx = (self.i + np.arange(n)) % self.size
        for k in self.obs:
            self.obs[k][idx] = obs[k]
            self.next_obs[k][idx] = next_obs[k]
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.i = (self.i + n) % self.size
        self.n = min(self.n + n, self.size)

    def sample(self, batch: int, rng) -> dict:
        idx = rng.integers(0, self.n, size=batch)
        t = lambda a: torch.as_tensor(a, device=self.device)
        return {
            "obs": {k: t(v[idx]) for k, v in self.obs.items()},
            "next_obs": {k: t(v[idx]) for k, v in self.next_obs.items()},
            "actions": t(self.actions[idx]),
            "rewards": t(self.rewards[idx]),
            "dones": t(self.dones[idx]),
        }


class DQN:
    def __init__(self, cfg: DQNConfig):
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        cfg.env.level = cfg.level
        self.envs = SyncVectorEnv(cfg.num_envs, cfg=cfg.env, seed=cfg.seed,
                                  level=cfg.levels or cfg.level)
        self.q = BranchQ(cfg.d_model, cfg.n_heads, cfg.n_layers).to(self.device)
        self.target = BranchQ(cfg.d_model, cfg.n_heads, cfg.n_layers).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.AdamW(self.q.parameters(), lr=cfg.lr, weight_decay=1e-4)
        self.rng = np.random.default_rng(cfg.seed)

        probe = self.envs.reset()
        spec = {k: (v.shape[1:], v.dtype) for k, v in probe.items()}
        self.buf = ReplayBuffer(cfg.buffer_size, spec, self.device)
        self.obs = probe
        self.global_step = 0
        self.recent: list = []
        self.run_dir = pathlib.Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(_jsonable(asdict(cfg)), indent=2))
        self.log_path = self.run_dir / "log.jsonl"

    # ------------------------------------------------------------ acting
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(self.global_step / max(c.total_steps * c.eps_decay_frac, 1), 1.0)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, obs: dict, eps: float) -> np.ndarray:
        t = {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}
        q = masked(self.q(t), t)
        greedy = np.stack([q[k].argmax(-1).cpu().numpy()
                           for k in ("type", "unit", "x", "y")], axis=-1)
        if eps <= 0:
            return greedy
        # epsilon-greedy per env, sampling only *legal* random actions -- a
        # uniform draw over all 6144 would spend nearly every exploration step
        # on an action the environment rejects
        explore = self.rng.random(len(greedy)) < eps
        if explore.any():
            rnd = self.random_legal(obs, explore)
            greedy[explore] = rnd
        return greedy

    def random_legal(self, obs: dict, which: np.ndarray) -> np.ndarray:
        out = []
        for i in np.flatnonzero(which):
            mt = np.flatnonzero(obs["mask_type"][i])
            a_t = int(self.rng.choice(mt)) if mt.size else 0
            mu = np.flatnonzero(obs["mask_unit"][i])
            a_u = int(self.rng.choice(mu)) if mu.size else 0
            legal = np.argwhere(obs["mask_xy"][i])
            if legal.size:
                a_x, a_y = (int(v) for v in legal[self.rng.integers(len(legal))])
            else:
                a_x = a_y = 0
            out.append((a_t, a_u, a_x, a_y))
        return np.array(out, dtype=np.int64)

    # ------------------------------------------------------------ learning
    def learn(self) -> dict:
        c = self.cfg
        b = self.buf.sample(c.batch_size, self.rng)
        keys = ("type", "unit", "x", "y")

        with torch.no_grad():
            nq_t = masked(self.target(b["next_obs"]), b["next_obs"])
            if c.double:
                nq_o = masked(self.q(b["next_obs"]), b["next_obs"])
                nxt = torch.stack([
                    nq_t[k].gather(1, nq_o[k].argmax(-1, keepdim=True)).squeeze(1)
                    for k in keys], dim=0).mean(0)
            else:
                nxt = torch.stack([nq_t[k].max(-1).values for k in keys], dim=0).mean(0)
            # a masked-out branch can be all -inf; clamp so the target stays finite
            nxt = torch.nan_to_num(nxt, neginf=0.0).clamp(min=-50.0, max=50.0)
            target = b["rewards"] + c.gamma * (1.0 - b["dones"]) * nxt

        q = masked(self.q(b["obs"]), b["obs"])
        loss = 0.0
        for j, k in enumerate(keys):
            taken = q[k].gather(1, b["actions"][:, j:j + 1]).squeeze(1)
            loss = loss + nn.functional.smooth_l1_loss(taken, target)
        loss = loss / len(keys)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.max_grad_norm)
        self.opt.step()
        return {"loss": float(loss.detach()), "q_mean": float(target.mean())}

    # ---------------------------------------------------------------- loop
    def train(self) -> None:
        c = self.cfg
        t0 = time.perf_counter()
        stats = {"loss": 0.0, "q_mean": 0.0}
        n_upd = 0
        while self.global_step < c.total_steps:
            eps = self.epsilon()
            a = self.act(self.obs, eps)
            nxt, r, done, infos = self.envs.step(a)
            self.buf.add(self.obs, a, r, nxt, done.astype(np.float32))
            self.obs = nxt
            self.global_step += c.num_envs
            for info in infos:
                if "episode" in info:
                    self.recent.append({"r": float(info["episode"]["r"]),
                                        "loot": float(info.get("loot_frac", 0.0)),
                                        "win": float(info.get("win", False))})
            del self.recent[:-500]

            if (self.buf.n >= c.learning_starts
                    and (self.global_step // c.num_envs) % c.train_every == 0):
                s = self.learn()
                stats = {k: stats[k] + s[k] for k in stats}
                n_upd += 1
                if n_upd % max(c.target_sync // c.num_envs, 1) == 0:
                    self.target.load_state_dict(self.q.state_dict())

            steps_done = self.global_step // c.num_envs
            if steps_done % c.log_every == 0 and n_upd:
                w = self.recent[-100:]
                mean = lambda k: float(np.mean([e[k] for e in w])) if w else float("nan")
                sps = self.global_step / max(time.perf_counter() - t0, 1e-9)
                row = {"step": self.global_step, "eps": round(eps, 3),
                       "loss": stats["loss"] / n_upd, "q_mean": stats["q_mean"] / n_upd,
                       "return": mean("r"), "loot": mean("loot"), "win": mean("win"),
                       "episodes": len(w), "sps": round(sps, 1)}
                with self.log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"step {self.global_step:8d} eps {eps:.2f} "
                      f"loss {row['loss']:7.4f} R {row['return']:7.3f} "
                      f"loot {row['loot']:5.3f} win {row['win']:4.2f} "
                      f"{sps:5.0f} sps", flush=True)
                stats = {k: 0.0 for k in stats}
                n_upd = 0
            if steps_done % c.eval_every == 0:
                self.save(self.run_dir / "dqn_latest.pt")
        self.save(self.run_dir / "dqn_final.pt")

    def save(self, path) -> None:
        torch.save({"model": self.q.state_dict(), "step": self.global_step,
                    "cfg": {"d_model": self.cfg.d_model, "n_heads": self.cfg.n_heads,
                            "n_layers": self.cfg.n_layers}}, path)


class DQNAgent:
    """Greedy policy from a trained BranchQ, for the benchmark harness."""

    def __init__(self, checkpoint: str, epsilon: float = 0.0, seed: int = 0,
                 device: str = "cpu"):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        c = ck.get("cfg", {})
        self.q = BranchQ(c.get("d_model", 96), c.get("n_heads", 4),
                         c.get("n_layers", 1)).to(device)
        self.q.load_state_dict(ck["model"])
        self.q.eval()
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: dict):
        t = {k: torch.as_tensor(v, device=self.device).unsqueeze(0)
             for k, v in batch_obs([obs]).items()}
        q = masked(self.q(t), t)
        a = [int(q[k].argmax(-1)) for k in ("type", "unit", "x", "y")]
        if self.epsilon and self.rng.random() < self.epsilon:
            m = obs["action_mask"]
            mt = np.flatnonzero(m["type"])
            a[0] = int(self.rng.choice(mt)) if mt.size else 0
            mu = np.flatnonzero(m["unit"])
            a[1] = int(self.rng.choice(mu)) if mu.size else 0
            legal = np.argwhere(m["xy"])
            if legal.size:
                a[2], a[3] = (int(v) for v in legal[self.rng.integers(len(legal))])
        return tuple(a)


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (int, float, str, bool)) or d is None:
        return d
    return str(d)
