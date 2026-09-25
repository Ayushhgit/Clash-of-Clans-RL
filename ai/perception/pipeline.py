"""Perception pipeline (M4): RGB frame -> the same observation dict RaidEnv emits.

    frame -> UI classifier -> screen state
          -> object detector -> boxes/classes/scores
          -> OCR            -> gold / elixir / timer / destruction %
          -> tokeniser      -> tokens, class_ids, mask, globals

Because the output schema is identical to `simulator.env.RaidEnv._obs`, a
policy trained in the simulator can be evaluated on real frames without
retraining -- that comparison IS Phase 22.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from annotation.app import CLASSES
from simulator.entities import CLASS_PRIOR_DPS, CLASS_PRIOR_RNG, MAX_DPS, MAX_RNG, Cls
from simulator.env import N_GLOBALS, N_TOKENS, TOKEN_DIM, XY_BINS

# Perception class -> simulator class. The detector vocabulary is the real
# game's (47 classes); the simulator's is 13. Several real structures collapse
# onto one simulated one, chosen by *behaviour*, not by name:
#   air sweeper      -> air defense      (air-only)
#   bomb tower       -> wizard tower     (splash, ground)
#   eagle artillery  -> mortar           (long range, splash, has a minimum range)
#   scattershot      -> wizard tower     (splash)
#   monolith         -> inferno          (single target, ramping damage)
#   clan castle      -> archer tower     (ranged defenders come out of it)
#   hero altar       -> cannon           (a hero defends the spot it sits on)
#   any support building, and an unnamed box, -> resource building
#     (destructible, counts toward destruction, does not shoot back)
NAME_TO_SIM = {
    "TOWN_HALL": Cls.TOWN_HALL,
    "CLAN_CASTLE": Cls.ARCHER_TOWER,

    "CANNON": Cls.CANNON,
    "ARCHER_TOWER": Cls.ARCHER_TOWER,
    "MORTAR": Cls.MORTAR,
    "WIZARD_TOWER": Cls.WIZARD_TOWER,
    "AIR_DEFENSE": Cls.AIR_DEFENSE,
    "AIR_SWEEPER": Cls.AIR_DEFENSE,
    "TESLA": Cls.TESLA,
    "BOMB_TOWER": Cls.WIZARD_TOWER,
    "INFERNO": Cls.INFERNO,
    "XBOW": Cls.XBOW,
    "EAGLE_ARTILLERY": Cls.MORTAR,
    "SCATTERSHOT": Cls.WIZARD_TOWER,
    "MONOLITH": Cls.INFERNO,
    "SPELL_TOWER": Cls.WIZARD_TOWER,
    "DEFENSE_OTHER": Cls.CANNON,

    "GOLD_STORAGE": Cls.STORAGE,
    "ELIXIR_STORAGE": Cls.STORAGE,
    "DARK_STORAGE": Cls.STORAGE,
    "GOLD_MINE": Cls.RESOURCE_BUILDING,
    "ELIXIR_COLLECTOR": Cls.RESOURCE_BUILDING,
    "DARK_DRILL": Cls.RESOURCE_BUILDING,

    "ARMY_CAMP": Cls.RESOURCE_BUILDING,
    "BARRACKS": Cls.RESOURCE_BUILDING,
    "LABORATORY": Cls.RESOURCE_BUILDING,
    "SPELL_FACTORY": Cls.RESOURCE_BUILDING,
    "WORKSHOP": Cls.RESOURCE_BUILDING,
    "BUILDER_HUT": Cls.RESOURCE_BUILDING,
    "OTHER_BUILDING": Cls.RESOURCE_BUILDING,
    "UNLABELED_BUILDING": Cls.RESOURCE_BUILDING,
    "HERO_ALTAR": Cls.CANNON,

    "WALL": Cls.WALL,
    "TRAP": Cls.OBSTACLE,
    "OBSTACLE": Cls.OBSTACLE,
    "TROOP": Cls.DPS,
}

# Drawn by the game, not part of the battlefield: never becomes an entity token.
UI_CLASS_NAMES = {
    "ATTACK_BUTTON", "NEXT_BUTTON", "CONFIRM_BUTTON", "RETURN_BUTTON",
    "END_BATTLE_BUTTON", "ARMY_CARD", "LOOT_PANEL",
    "HUD_GOLD", "HUD_ELIXIR", "HUD_DARK", "HUD_TIMER", "HUD_DESTRUCTION",
}

# token priority mirrors RaidEnv: own units, then defenses, then scoring
# buildings, then walls/obstacles
_PRIORITY = {int(Cls.DPS): 0}
for _c in (Cls.CANNON, Cls.ARCHER_TOWER, Cls.MORTAR, Cls.WIZARD_TOWER,
           Cls.AIR_DEFENSE, Cls.TESLA, Cls.INFERNO, Cls.XBOW):
    _PRIORITY[int(_c)] = 1
for _c in (Cls.TOWN_HALL, Cls.STORAGE, Cls.RESOURCE_BUILDING):
    _PRIORITY[int(_c)] = 2
for _c in (Cls.WALL, Cls.OBSTACLE):
    _PRIORITY[int(_c)] = 3


@dataclass
class HUDRegions:
    """Pixel rectangles (x1, y1, x2, y2) in the captured frame, measured once
    per resolution during Phase 1 and stored in configs/geometry.yaml."""

    gold: tuple | None = None
    elixir: tuple | None = None
    timer: tuple | None = None
    destruction: tuple | None = None
    battle_area: tuple | None = None      # the map viewport


@dataclass
class PerceptionConfig:
    img_size: int = 512
    score_threshold: float = 0.3
    max_detections: int = 128
    hud: HUDRegions = field(default_factory=HUDRegions)
    army_size: int = 22            # for normalising the remaining-army globals
    t_max: float = 180.0


class Perceiver:
    """Wraps the three perception models behind one `perceive(rgb)` call."""

    def __init__(self, detector=None, ui_classifier=None, ocr=None,
                 cfg: PerceptionConfig | None = None, device: str = "cpu"):
        self.detector = detector
        self.ui = ui_classifier
        self.ocr = ocr
        self.cfg = cfg or PerceptionConfig()
        self.device = torch.device(device)

    # ------------------------------------------------------------------ api
    @torch.no_grad()
    def perceive(self, rgb: np.ndarray, deploy_mask: np.ndarray | None = None) -> dict:
        state, conf = self.screen_state(rgb)
        dets = self.detect(rgb)
        hud = self.read_hud(rgb)
        obs = self.tokenise(dets, rgb.shape[1], rgb.shape[0], hud)
        obs["screen_state"] = state
        obs["screen_conf"] = conf
        obs["detections"] = dets
        obs["hud"] = hud
        obs["action_mask"] = self.action_mask(dets, rgb, deploy_mask)
        return obs

    # ------------------------------------------------------------ submodels
    def screen_state(self, rgb: np.ndarray) -> tuple:
        if self.ui is None:
            return "UNKNOWN", 0.0
        x = _to_tensor(rgb, 224).to(self.device)
        names, confs = self.ui.predict(x)
        return names[0], float(confs[0])

    def detect(self, rgb: np.ndarray) -> dict:
        if self.detector is None:
            return {"boxes": np.zeros((0, 4), np.float32),
                    "labels": np.zeros(0, np.int64),
                    "scores": np.zeros(0, np.float32)}
        from ai.perception.detector import decode

        x = _to_tensor(rgb, self.cfg.img_size).to(self.device)
        out = self.detector(x)
        det = decode(out, k=self.cfg.max_detections,
                     threshold=self.cfg.score_threshold)[0]
        # decode works in resized-image pixels; scale back to frame pixels
        sx = rgb.shape[1] / self.cfg.img_size
        sy = rgb.shape[0] / self.cfg.img_size
        if len(det["boxes"]):
            det["boxes"] = det["boxes"] * np.array([sx, sy, sx, sy], dtype=np.float32)
        return det

    def read_hud(self, rgb: np.ndarray) -> dict:
        from ai.perception.ui import parse_number

        hud: dict = {}
        if self.ocr is None:
            return hud
        for name in ("gold", "elixir", "timer", "destruction"):
            rect = getattr(self.cfg.hud, name)
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            crop = rgb[y1:y2, x1:x2].astype(np.float32) / 255.0
            hud[name] = parse_number(self.ocr.read(crop))
        return hud

    # ------------------------------------------------------------ tokeniser
    def tokenise(self, det: dict, frame_w: int, frame_h: int, hud: dict) -> dict:
        """Detections -> the OBSERVATION_SPEC token tensor (DETECTED mode)."""
        area = self.cfg.hud.battle_area or (0, 0, frame_w, frame_h)
        ax1, ay1, ax2, ay2 = area
        aw, ah = max(ax2 - ax1, 1), max(ay2 - ay1, 1)

        tokens = np.zeros((N_TOKENS, TOKEN_DIM), dtype=np.float32)
        class_ids = np.zeros(N_TOKENS, dtype=np.int64)
        mask = np.zeros(N_TOKENS, dtype=bool)

        rows = []
        for box, lab, score in zip(det["boxes"], det["labels"], det["scores"]):
            name = CLASSES[int(lab)] if int(lab) < len(CLASSES) else ""
            if name in UI_CLASS_NAMES or name not in NAME_TO_SIM:
                continue
            sim_cls = int(NAME_TO_SIM[name])
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2 - ax1) / aw
            cy = ((y1 + y2) / 2 - ay1) / ah
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                continue
            rows.append((_PRIORITY.get(sim_cls, 3), sim_cls, cx, cy,
                         (x2 - x1) / aw, (y2 - y1) / ah, float(score)))

        rows.sort(key=lambda r: (r[0], -r[6]))
        for i, (_, sim_cls, cx, cy, w, h, score) in enumerate(rows[:N_TOKENS]):
            is_unit = sim_cls == int(Cls.DPS)
            tokens[i, 0], tokens[i, 1] = cx, cy
            tokens[i, 2], tokens[i, 3] = w, h
            tokens[i, 4] = 1.0                       # hp unknown from pixels
            tokens[i, 5] = 1.0
            tokens[i, 6] = 1.0 if is_unit else 0.0
            tokens[i, 7] = CLASS_PRIOR_RNG.get(sim_cls, 0.0) / MAX_RNG
            tokens[i, 8] = CLASS_PRIOR_DPS.get(sim_cls, 0.0) / MAX_DPS
            tokens[i, 9] = score
            class_ids[i] = sim_cls
            mask[i] = True

        return {"tokens": tokens, "class_ids": class_ids, "token_mask": mask,
                "globals": self._globals(hud, mask, class_ids)}

    def _globals(self, hud: dict, mask: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        g = np.zeros(N_GLOBALS, dtype=np.float32)
        timer = hud.get("timer")
        if timer is not None:
            g[0] = float(np.clip(1.0 - timer / self.cfg.t_max, 0.0, 1.0))
        dest = hud.get("destruction")
        if dest is not None:
            g[1] = float(np.clip(dest / 100.0, 0.0, 1.0))
        g[2] = float(not (class_ids[mask] == int(Cls.TOWN_HALL)).any())
        # remaining-army slots (3..8) come from the army bar, filled by the
        # live env which reads the unit-card counters; unknown here
        g[9] = float((class_ids[mask] == int(Cls.DPS)).sum()) / max(self.cfg.army_size, 1)
        return g

    # ---------------------------------------------------------- action mask
    def action_mask(self, det: dict, rgb: np.ndarray,
                    deploy_mask: np.ndarray | None) -> dict:
        """Deploy grid derived from detections: a cell is legal when it is far
        enough from every detected building. Mirrors RaidEnv._deploy_grid, but
        with detected -- therefore incomplete -- geometry."""
        from simulator.entities import NUM_ARCHETYPES
        from simulator.env import N_ACT_TYPES

        if deploy_mask is None:
            area = self.cfg.hud.battle_area or (0, 0, rgb.shape[1], rgb.shape[0])
            ax1, ay1, ax2, ay2 = area
            aw, ah = max(ax2 - ax1, 1), max(ay2 - ay1, 1)
            gx, gy = np.meshgrid((np.arange(XY_BINS) + 0.5) / XY_BINS,
                                 (np.arange(XY_BINS) + 0.5) / XY_BINS, indexing="ij")
            deploy_mask = np.ones((XY_BINS, XY_BINS), dtype=bool)
            for box, lab in zip(det["boxes"], det["labels"]):
                name = CLASSES[int(lab)] if int(lab) < len(CLASSES) else ""
                if name in UI_CLASS_NAMES or name == "TROOP":
                    continue
                x1, y1, x2, y2 = box
                cx, cy = ((x1 + x2) / 2 - ax1) / aw, ((y1 + y2) / 2 - ay1) / ah
                r = max((x2 - x1) / aw, (y2 - y1) / ah) / 2 + 2.0 / 44.0
                deploy_mask &= ((gx - cx) ** 2 + (gy - cy) ** 2) > r * r
        if not deploy_mask.any():
            deploy_mask = np.ones((XY_BINS, XY_BINS), dtype=bool)
        return {"type": np.ones(N_ACT_TYPES, dtype=bool),
                "unit": np.ones(NUM_ARCHETYPES, dtype=bool),
                "xy": deploy_mask}


def _to_tensor(rgb: np.ndarray, size: int) -> torch.Tensor:
    from PIL import Image

    img = Image.fromarray(rgb).resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
