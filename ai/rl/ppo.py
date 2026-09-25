"""PPO, implemented from scratch (Phase 13).

Everything the brief asks for is explicit here rather than hidden in a
library: rollout collection, GAE, advantage normalisation, the clipped
surrogate, clipped value loss, entropy bonus, gradient clipping, LR
annealing, KL early-stopping, checkpointing and evaluation hooks.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ai.policies.actor_critic import ACT_DEPLOY, ActorCritic
from ai.rl.buffer import RolloutBuffer, default_obs_spec
from simulator.env import (
    N_ACT_TYPES,
    N_GLOBALS,
    N_TOKENS,
    TOKEN_DIM,
    XY_BINS,
    EnvConfig,
)
from simulator.entities import NUM_ARCHETYPES
from simulator.vec_env import SyncVectorEnv


@dataclass
class PPOConfig:
    total_steps: int = 400_000
    num_envs: int = 64          # batching, not parallelism: one forward pass for N envs
    rollout_steps: int = 32
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    norm_adv: bool = True
    # Updates spent fitting the critic before the actor is allowed to move.
    # A BC warm start hands PPO a good actor and a *random* critic, so the
    # first advantages are noise and the clipped objective happily spends its
    # whole trust region acting on them -- the classic "BC init evaporates"
    # failure. Everything but the value head is frozen during warm-up, so the
    # shared trunk cannot drift either.
    critic_warmup_updates: int = 0
    device: str = "cpu"
    torch_threads: int = 10
    seed: int = 0
    # model
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 1
    memory: str | None = "gru"   # none | stack | gru | transformer
    # curriculum
    level: str = "L1"
    # Train on a fixed mix of levels instead of a curriculum. With a BC warm
    # start the curriculum's bootstrapping job is already done, and a mix
    # matches deployment, where base difficulty is not sorted for you. It also
    # sidesteps an unsolvable gate: a fixed loot bar cannot promote through
    # levels whose achievable loot differs by 2.5x (L3 ~0.65, L4 ~0.26).
    levels: list | None = None
    curriculum: bool = True
    promote_win_rate: float = 0.7
    promote_window: int = 100
    # Gate the curriculum on the metric actually being optimised. The farming
    # profile sets win=0 in the reward, so gating on win rate asks the agent
    # to clear a bar it is not being paid to clear -- L4 sat at 0.14 wins with
    # loot still improving, and L5 was simply unreachable.
    promote_metric: str = "win"          # "win" or "loot"
    promote_threshold: float | None = None   # defaults to promote_win_rate
    eval_every: int = 25          # updates between held-out evaluations
    eval_episodes: int = 24       # unseen bases per evaluation
    # bookkeeping
    resume: str | None = None   # checkpoint to continue from
    run_dir: str = "experiments/ppo_run"
    log_every: int = 1
    save_every: int = 20
    env: EnvConfig = field(default_factory=EnvConfig)


class PPO:
    def __init__(self, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        if self.cfg.device == "cpu" and self.cfg.torch_threads:
            torch.set_num_threads(self.cfg.torch_threads)

        c = self.cfg
        c.env.level = c.level
        self.envs = SyncVectorEnv(c.num_envs, cfg=c.env, seed=c.seed, level=c.level)
        self.device = torch.device(c.device)

        self.policy = ActorCritic(d_model=c.d_model, n_heads=c.n_heads,
                                  n_layers=c.n_layers, memory=c.memory).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=c.lr, eps=1e-5)

        spec = default_obs_spec(N_TOKENS, TOKEN_DIM, N_GLOBALS,
                                N_ACT_TYPES, NUM_ARCHETYPES, XY_BINS)
        self.buf = RolloutBuffer(c.rollout_steps, c.num_envs, spec,
                                 hidden_size=self.policy.hidden_size,
                                 device=c.device)

        self.run_dir = pathlib.Path(c.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(_jsonable(asdict(c)), indent=2))
        self.log_path = self.run_dir / "log.jsonl"

        self.global_step = 0
        self.updates = 0
        self.level = c.level
        # one dict per finished episode, tagged with the level it was played
        # on. Tagging matters: a plain rolling window straddles a curriculum
        # promotion and reports the old level's easy returns as if they were
        # the new level's, which reads exactly like a regression.
        self.recent: list = []
        # env i always plays levels[i % len(levels)] (SyncVectorEnv deals them
        # round-robin), so the level of every sample is known without tagging
        self.env_level = (np.arange(c.num_envs) % len(c.levels) if c.levels
                          else np.zeros(c.num_envs, dtype=np.int64))
        if c.levels:
            self.level = "mix"
            self.envs.set_level(c.levels)
            if c.curriculum:
                print("[curriculum] disabled: training on a level mix "
                      f"{c.levels}", flush=True)
                c.curriculum = False

        if c.resume:
            self._resume(c.resume)

    def _resume(self, path: str) -> None:
        """Continue a stopped run: weights, optimiser state, step count and
        curriculum level all carry over, so `total_steps` stays the budget for
        the *whole* run rather than for the continuation."""
        from ai.policies.neural_agent import _migrate

        ck = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.policy.load_state_dict(_migrate(ck["model"]),
                                                          strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint {path} does not fit the current model; "
                f"missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]}")
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        self.global_step = int(ck.get("step", 0))
        self.level = ck.get("level", self.level)
        # "mix" is a label for a set of levels, not a level the generator
        # knows: restore the set from the config rather than passing it on
        self.envs.set_level(self.cfg.levels if self.level == "mix"
                            else self.level)
        print(f"[resume] {path}: {self.global_step} steps done, level {self.level}",
              flush=True)

    # ------------------------------------------------------------- rollouts
    def collect(self, obs: dict, hx, done) -> tuple:
        self.buf.reset()
        pol = self.policy
        for _ in range(self.cfg.rollout_steps):
            t_obs = _to_torch(obs, self.device)
            with torch.no_grad():
                hx_in = hx
                action, logp, _, value, hx_out = pol(t_obs, hx_in)
            a = action.cpu().numpy()
            nxt, reward, next_done, infos = self.envs.step(a)

            self.buf.add(obs, a, logp.cpu().numpy(), reward, done,
                         value.cpu().numpy(),
                         hx_in.cpu().numpy() if hx_in is not None else None)

            # a finished episode must not leak recurrent state into the next one
            if hx_out is not None:
                keep = torch.as_tensor(~next_done, device=self.device).float().unsqueeze(-1)
                hx_out = hx_out * keep
            hx, obs, done = hx_out, nxt, next_done
            self.global_step += self.cfg.num_envs

            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    self.recent.append({
                        "level": info.get("level", self.level),
                        "r": float(ep["r"]),
                        "win": float(info.get("win", False)),
                        "loot": float(info.get("loot_frac", 0.0)),
                    })
            del self.recent[:-2000]
        return obs, hx, done

    # --------------------------------------------------------------- update
    def _normalise_per_level(self, adv: np.ndarray) -> np.ndarray:
        """Whiten advantages within each level, not across the whole batch.

        Returns differ about tenfold between the easiest and hardest levels in
        the mix (L2 ~3.5, L5 ~0.2), and advantage magnitude follows reward
        magnitude. Normalising the batch as one pool therefore hands most of
        the gradient to whichever level happens to pay the most, and the policy
        specialises on the easy one: over 256k steps L2 went 0.89 -> 0.95 while
        L3 went 0.68 -> 0.52 and L5 0.28 -> 0.13, ending *below* the behavioural
        cloning it started from.

        Per-level whitening makes every level contribute a comparable gradient,
        which is the whole point of training on a mix.
        """
        out = np.empty_like(adv)
        for lid in np.unique(self.env_level):
            cols = np.flatnonzero(self.env_level == lid)
            g = adv[:, cols]
            out[:, cols] = (g - g.mean()) / (g.std() + 1e-8)
        return out

    def _set_actor_trainable(self, on: bool) -> None:
        for name, p in self.policy.named_parameters():
            p.requires_grad_(on or name.startswith("head_v."))

    def update(self, obs: dict, hx, done) -> dict:
        c = self.cfg
        warmup = self.updates < c.critic_warmup_updates
        with torch.no_grad():
            last_value = self.policy.value(_to_torch(obs, self.device), hx).cpu().numpy()
        adv, returns = self.buf.compute_gae(last_value, done, c.gamma, c.gae_lambda)
        per_level = bool(c.levels) and c.norm_adv
        if per_level:
            adv = self._normalise_per_level(adv)
        data = self.buf.flat_tensors(adv, returns)

        n = c.rollout_steps * c.num_envs
        mb_size = n // c.minibatches
        idx = np.arange(n)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                 "kl": 0.0, "clipfrac": 0.0, "n_mb": 0}
        ent_live_sum, ent_live_n = 0.0, 0
        stop = False

        for _ in range(c.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = torch.as_tensor(idx[start:start + mb_size], device=self.device)
                obs_mb = {k: data[k][mb] for k in
                          ("tokens", "class_ids", "token_mask", "globals",
                           "mask_type", "mask_unit", "mask_xy")}
                hx_mb = data["hidden"][mb] if "hidden" in data else None

                _, newlogp, entropy, newvalue, _ = self.policy(
                    obs_mb, hx_mb, action=data["actions"][mb]
                )
                logratio = newlogp - data["logprobs"][mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > c.clip_coef).float().mean().item()

                mb_adv = data["advantages"][mb]
                if c.norm_adv and not per_level:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - c.clip_coef, 1 + c.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                ret_mb, val_mb = data["returns"][mb], data["values"][mb]
                if c.clip_vloss:
                    v_unclipped = (newvalue - ret_mb) ** 2
                    v_clipped = val_mb + torch.clamp(newvalue - val_mb,
                                                     -c.clip_coef, c.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - ret_mb) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret_mb) ** 2).mean()

                ent = entropy.mean()
                loss = (c.vf_coef * v_loss if warmup
                        else pg_loss - c.ent_coef * ent + c.vf_coef * v_loss)

                # Plain mean entropy is dominated by steps where the army is
                # spent and WAIT is the only legal action -- masked to a single
                # choice, those contribute exactly 0 and drag the average to
                # ~0 no matter how the policy is exploring. `ent_live` is the
                # entropy on steps where a deploy is still possible, which is
                # the only place exploration means anything.
                with torch.no_grad():
                    live = obs_mb["mask_type"][:, ACT_DEPLOY]
                    ent_live = (entropy[live].mean().item() if live.any()
                                else float("nan"))

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.opt.step()

                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                if ent_live == ent_live:              # not NaN
                    ent_live_sum += ent_live
                    ent_live_n += 1
                stats["kl"] += approx_kl
                stats["clipfrac"] += clipfrac
                stats["n_mb"] += 1

            if (not warmup and c.target_kl is not None
                    and stats["kl"] / max(stats["n_mb"], 1) > c.target_kl):
                stop = True
                break

        k = max(stats.pop("n_mb"), 1)
        out = {key: v / k for key, v in stats.items()}
        # NaN, not 0, when no minibatch held a deployable step: "we could not
        # measure exploration here" is a different claim from "there was none"
        out["ent_live"] = (ent_live_sum / ent_live_n if ent_live_n
                           else float("nan"))
        out["early_stop"] = float(stop)
        out["warmup"] = float(warmup)
        return out

    # ----------------------------------------------------------------- loop
    def train(self) -> None:
        c = self.cfg
        obs = self.envs.reset()
        hx = self.policy.initial_state(c.num_envs, self.device)
        done = np.zeros(c.num_envs, dtype=bool)
        per_update = c.rollout_steps * c.num_envs
        n_updates = max(c.total_steps // per_update, 1)
        start_update = self.global_step // per_update
        if start_update >= n_updates:
            print(f"[resume] already at {self.global_step} steps of "
                  f"{c.total_steps}; nothing to do")
            return
        t0 = time.perf_counter()
        self._step0 = self.global_step

        for update in range(start_update + 1, n_updates + 1):
            if c.anneal_lr:
                frac = 1.0 - (update - 1.0) / n_updates
                for g in self.opt.param_groups:
                    g["lr"] = frac * c.lr

            obs, hx, done = self.collect(obs, hx, done)
            was_warm = self.updates < c.critic_warmup_updates
            self._set_actor_trainable(not was_warm)
            stats = self.update(obs, hx, done)
            if was_warm and update >= c.critic_warmup_updates:
                self._set_actor_trainable(True)
                print(f"[warmup] critic fitted over {c.critic_warmup_updates} "
                      f"updates; actor unfrozen", flush=True)
            self.updates = update

            if c.curriculum:
                self._maybe_promote()

            if c.eval_every and update % c.eval_every == 0:
                ev = self.evaluate()
                stats.update(ev)
                print(f"[eval] {ev['eval_level']} loot {ev['eval_loot']:.3f} "
                      f"win {ev['eval_win']:.2f} over {ev['eval_episodes']} "
                      f"unseen bases", flush=True)
                self._log(stats, t0)
            elif update % c.log_every == 0:
                self._log(stats, t0)
            if update % c.save_every == 0 or update == n_updates:
                self.save(self.run_dir / "ppo_latest.pt")
        self.save(self.run_dir / "ppo_final.pt")

    # ------------------------------------------------------------ held-out
    @torch.no_grad()
    def evaluate(self, episodes: int | None = None, level: str | None = None) -> dict:
        """Play unseen bases at a fixed level and report loot.

        The training window moves with the curriculum, so it cannot answer
        "is the policy better than it was 100k steps ago?" -- a promotion
        changes the question being asked. This runs the same fixed level on
        seeds the trainer never sees, so its numbers are comparable across the
        whole run.
        """
        from copy import deepcopy

        from simulator.env import RaidEnv
        from simulator.generator import split_seeds
        from simulator.vec_env import batch_obs

        c = self.cfg
        if level is None and self.level == "mix":
            out = {}
            for lv in c.levels:
                r = self.evaluate(episodes=episodes, level=lv)
                out[f"eval_loot_{lv}"] = r["eval_loot"]
                out[f"eval_win_{lv}"] = r["eval_win"]
            out["eval_loot"] = float(np.mean([out[f"eval_loot_{lv}"] for lv in c.levels]))
            out["eval_win"] = float(np.mean([out[f"eval_win_{lv}"] for lv in c.levels]))
            out["eval_level"] = "mix"
            out["eval_episodes"] = (episodes or c.eval_episodes) * len(c.levels)
            return out
        level = level or self.level
        n = episodes or c.eval_episodes
        _, ev = split_seeds()

        cfg = deepcopy(c.env)
        cfg.seed_range = ev
        env = RaidEnv(cfg)
        self.policy.eval()
        loot, wins, rets = [], [], []
        for i in range(n):
            obs = env.reset(seed=ev[0] + i, level=level)
            hx = self.policy.initial_state(1, self.device)
            done, R = False, 0.0
            while not done:
                action, _, _, _, hx = self.policy(
                    _to_torch(batch_obs([obs]), self.device), hx)
                obs, r, done, info = env.step(action[0].cpu().numpy())
                R += r
            loot.append(info["loot_frac"])
            wins.append(float(info["win"]))
            rets.append(R)
        self.policy.train()
        return {"eval_loot": float(np.mean(loot)), "eval_win": float(np.mean(wins)),
                "eval_return": float(np.mean(rets)), "eval_level": level,
                "eval_episodes": n}

    # ---------------------------------------------------------- curriculum
    def _maybe_promote(self) -> None:
        from simulator.generator import LEVELS

        c = self.cfg
        w = self._window(c.promote_window)
        if len(w) < c.promote_window:
            return
        if self.level == "mix":
            return                       # nothing to promote through
        key = "loot" if c.promote_metric == "loot" else "win"
        bar = c.promote_threshold if c.promote_threshold is not None else c.promote_win_rate
        wr = float(np.mean([e[key] for e in w]))
        i = LEVELS.index(self.level)
        if wr >= bar and i + 1 < len(LEVELS):
            self.level = LEVELS[i + 1]
            self.envs.set_level(self.level)
            self.recent.clear()
            print(f"[curriculum] promoted to {self.level} "
                  f"({key} {wr:.2f} >= {bar:.2f})", flush=True)

    # ------------------------------------------------------------- logging
    def _window(self, n: int = 100) -> list:
        """The last `n` episodes played *on the current level*.

        On a level mix every level is in the distribution being trained on, so
        the whole tail is the right window; under a curriculum it is not, and
        filtering is what stops a promotion from looking like a regression.
        """
        if self.level == "mix":
            return self.recent[-n:]
        return [e for e in self.recent if e["level"] == self.level][-n:]

    def _log(self, stats: dict, t0: float) -> None:
        w = self._window(100)
        mean = lambda k: float(np.mean([e[k] for e in w])) if w else float("nan")
        wr, ret, loot = mean("win"), mean("r"), mean("loot")
        sps = (self.global_step - getattr(self, "_step0", 0)) / max(
            time.perf_counter() - t0, 1e-9)
        row = {"update": self.updates, "step": self.global_step, "level": self.level,
               "win_rate": wr, "return": ret, "loot": loot,
               "episodes": len(w), "sps": round(sps, 1), **stats}
        if self.level == "mix":
            # a mixed window hides which level is dragging the mean
            for lv in sorted({e["level"] for e in w}):
                sub = [e["loot"] for e in w if e["level"] == lv]
                row[f"loot_{lv}"] = float(np.mean(sub))
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"u{self.updates:4d} step {self.global_step:8d} {self.level} "
              f"R {ret:7.3f} loot {loot:5.3f} win {wr:5.2f} ent {stats['ent_live']:6.3f} "
              f"kl {stats['kl']:.4f} v {stats['v_loss']:.4f} {sps:6.0f} sps", flush=True)

    # ---------------------------------------------------------- checkpoints
    def save(self, path) -> None:
        torch.save({"model": self.policy.state_dict(),
                    "opt": self.opt.state_dict(),
                    "cfg": _jsonable(asdict(self.cfg)),
                    "step": self.global_step,
                    "level": self.level}, path)

    @staticmethod
    def load_policy(path, device: str = "cpu") -> ActorCritic:
        ck = torch.load(path, map_location=device, weights_only=False)
        c = ck["cfg"]
        pol = ActorCritic(d_model=c["d_model"], n_heads=c["n_heads"],
                          n_layers=c["n_layers"], memory=c["memory"]).to(device)
        pol.load_state_dict(ck["model"])
        pol.eval()
        return pol


def _to_torch(obs: dict, device) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (str, int, float, bool)) or d is None:
        return d
    return str(d)
