"""Battle rendering for debugging and demos.

Pure PIL, no game assets: buildings are coloured squares, units are dots,
defense range is a faint circle. It is not pretty, it is *diagnostic* -- if
the agent dumps its army outside every defense ring, one frame shows it.

    python -m scripts.render_episode --agent heuristic --level L3 --seed 10000000
"""
from __future__ import annotations

import pathlib

import numpy as np

from simulator.entities import Cls

COLORS = {
    int(Cls.TOWN_HALL): (230, 190, 60),
    int(Cls.STORAGE): (210, 130, 40),
    int(Cls.RESOURCE_BUILDING): (150, 120, 80),
    int(Cls.WALL): (110, 110, 110),
    int(Cls.OBSTACLE): (70, 90, 70),
    int(Cls.CANNON): (200, 70, 70),
    int(Cls.ARCHER_TOWER): (200, 110, 60),
    int(Cls.MORTAR): (170, 60, 120),
    int(Cls.WIZARD_TOWER): (120, 80, 220),
    int(Cls.AIR_DEFENSE): (80, 160, 220),
    int(Cls.TESLA): (240, 230, 90),
    int(Cls.INFERNO): (240, 90, 40),
    int(Cls.XBOW): (90, 200, 160),
}
UNIT_COLORS = {
    int(Cls.TANK): (255, 255, 255),
    int(Cls.DPS): (255, 90, 90),
    int(Cls.WALL_BREAKER): (255, 200, 0),
    int(Cls.RANGED): (120, 255, 120),
    int(Cls.AIR): (120, 200, 255),
    int(Cls.SPLASH): (255, 120, 255),
}
BG = (24, 28, 24)


def render(sim, px: int = 640, show_ranges: bool = True):
    from PIL import Image, ImageDraw

    m = sim.cfg.map_size
    img = Image.new("RGB", (px, px), BG)
    d = ImageDraw.Draw(img, "RGBA")
    s = px / m

    if show_ranges:
        for i in np.nonzero(sim.b_alive & sim.b_isdef)[0]:
            r = sim.b_rng[i] * s
            cx, cy = sim.b_x[i] * s, sim.b_y[i] * s
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 60, 60, 40))

    for i in range(len(sim.b_cls)):
        c = COLORS.get(int(sim.b_cls[i]), (90, 90, 90))
        r = sim.b_r[i] * s
        cx, cy = sim.b_x[i] * s, sim.b_y[i] * s
        if sim.b_alive[i]:
            hp = sim.b_hp[i] / max(sim.b_maxhp[i], 1)
            col = tuple(int(v * (0.35 + 0.65 * hp)) for v in c)
        else:
            col = (45, 45, 45)
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)

    for i in range(sim.n_units):
        if not sim.u_alive[i]:
            continue
        c = UNIT_COLORS.get(int(sim.u_cls[i]), (255, 255, 255))
        cx, cy = sim.u_x[i] * s, sim.u_y[i] * s
        r = 3
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)

    d.text((6, 6), f"t={sim.t:5.1f}  dest={sim.destruction:.2f}  "
                   f"loot={sim.loot_frac:.2f}  units={sim.alive_unit_count()}",
           fill=(230, 230, 230))
    return img


def record_episode(agent, level: str = "L3", seed: int = 10_000_000,
                   out: str = "experiments/episode.gif", every: int = 2,
                   px: int = 640, obs_mode: str = "privileged") -> dict:
    """Play one episode and write an animated GIF of it."""
    from simulator.env import EnvConfig, RaidEnv

    env = RaidEnv(EnvConfig(level=level, obs_mode=obs_mode))
    obs = env.reset(seed=seed, level=level)
    if hasattr(agent, "reset"):
        agent.reset()

    frames = [render(env.sim, px)]
    done, i = False, 0
    while not done:
        obs, r, done, info = env.step(agent.act(obs))
        i += 1
        if i % every == 0 or done:
            frames.append(render(env.sim, px))

    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=80, loop=0)
    return {"frames": len(frames), "path": str(p), **{k: info[k] for k in
            ("win", "destruction", "loot_frac", "duration")}}


def plot_training(log_path: str = "experiments/ppo_v1/log.jsonl",
                  out: str = "experiments/training_curve.png") -> str:
    """Win rate / return / entropy against steps, straight from log.jsonl."""
    import json

    from PIL import Image, ImageDraw

    rows = [json.loads(l) for l in pathlib.Path(log_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("return") == r.get("return")]   # drop NaN
    if not rows:
        return "no data yet"

    w, h, pad = 900, 300, 40
    img = Image.new("RGB", (w, h), (18, 18, 20))
    d = ImageDraw.Draw(img)
    steps = [r["step"] for r in rows]
    x0, x1 = min(steps), max(steps) or 1

    def series(key, color, lo=None, hi=None):
        vals = [r.get(key, 0.0) for r in rows]
        lo_ = min(vals) if lo is None else lo
        hi_ = max(vals) if hi is None else hi
        rng = (hi_ - lo_) or 1.0
        pts = [(pad + (s - x0) / max(x1 - x0, 1) * (w - 2 * pad),
                h - pad - (v - lo_) / rng * (h - 2 * pad))
               for s, v in zip(steps, vals)]
        d.line(pts, fill=color, width=2)

    series("return", (120, 200, 255))
    series("win_rate", (120, 255, 140), 0.0, 1.0)
    series("entropy", (255, 180, 100))
    d.text((pad, 8), "blue=return  green=win_rate  orange=entropy", fill=(200, 200, 200))
    d.text((pad, h - 24), f"steps {x0} -> {x1}", fill=(200, 200, 200))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
