"""Bounding-box annotation tool (Phase 3).

Tkinter + Pillow only -- no OpenCV, no browser, no server. Runs on the same
machine that captured the frames.

    python -m annotation.app data/raw/screenshots/session_20260828_120000

Keys
    left drag   draw a box            n / right   next image
    click box   select                p / left    previous image
    1..9,0      set class of selection  d / Del   delete selection
    [ / ]       cycle class of selection  s       save now (autosaves on move)
    u           undo last box         q          quit
    a           toggle auto-advance after save

Labels are written next to the image as `<name>.json`:

    {"image": "000000123.png", "w": 1280, "h": 720,
     "boxes": [{"cls": "CANNON", "xyxy": [x1, y1, x2, y2]}, ...]}

`scripts/export_dataset.py` converts these to the training formats.
"""
from __future__ import annotations

import json
import pathlib
import sys

CLASSES = [
    # --- key structures ---------------------------------------------------
    "TOWN_HALL", "CLAN_CASTLE",
    # --- defenses ---------------------------------------------------------
    "CANNON", "ARCHER_TOWER", "MORTAR", "WIZARD_TOWER", "AIR_DEFENSE",
    "AIR_SWEEPER", "TESLA", "BOMB_TOWER", "INFERNO", "XBOW",
    "EAGLE_ARTILLERY", "SCATTERSHOT", "MONOLITH", "SPELL_TOWER",
    # a defensive structure whose exact type is not certain from the frame;
    # still worth a box -- the agent mainly needs "there is a threat here"
    "DEFENSE_OTHER",
    # --- resources --------------------------------------------------------
    "GOLD_STORAGE", "ELIXIR_STORAGE", "DARK_STORAGE",
    "GOLD_MINE", "ELIXIR_COLLECTOR", "DARK_DRILL",
    # --- support buildings ------------------------------------------------
    "ARMY_CAMP", "BARRACKS", "LABORATORY", "SPELL_FACTORY", "WORKSHOP",
    "BUILDER_HUT", "OTHER_BUILDING",
    # --- heroes -----------------------------------------------------------
    "HERO_ALTAR",
    # --- field ------------------------------------------------------------
    "WALL", "TRAP", "OBSTACLE", "TROOP",
    # a box drawn automatically around a structure whose type nobody has
    # named yet -- the annotation tool's job is to reclassify these
    "UNLABELED_BUILDING",
    # --- UI ---------------------------------------------------------------
    "ATTACK_BUTTON", "NEXT_BUTTON", "END_BATTLE_BUTTON", "RETURN_BUTTON",
    "CONFIRM_BUTTON", "ARMY_CARD", "LOOT_PANEL",
    "HUD_GOLD", "HUD_ELIXIR", "HUD_DARK", "HUD_TIMER", "HUD_DESTRUCTION",
]

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#a9a9a9", "#ff4d4d", "#00b894", "#fdcb6e", "#0984e3",
    "#e17055", "#6c5ce7", "#00cec9", "#d63031", "#55efc4", "#fab1a0",
    "#74b9ff", "#ffeaa7", "#b2bec3", "#dfe6e9", "#e84393", "#2d3436",
    "#00ff7f", "#ff69b4", "#7fffd4", "#ffa500", "#adff2f", "#dda0dd",
]

MIN_BOX = 4          # pixels; smaller drags are treated as clicks


class Annotator:
    def __init__(self, image_dir: str, max_width: int = 1500, max_height: int = 850):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.dir = pathlib.Path(image_dir)
        self.paths = sorted([p for p in self.dir.iterdir()
                             if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if not self.paths:
            raise SystemExit(f"no images in {self.dir}")
        self.max_width, self.max_height = max_width, max_height

        self.i = 0
        self.cls_i = 0
        self.boxes: list = []          # [{"cls": str, "xyxy": [..]}] in image coords
        self.selected: int | None = None
        self.auto_advance = True
        self.drag = None

        self.root = tk.Tk()
        self.root.title("annotator")
        self.canvas = tk.Canvas(self.root, cursor="crosshair", bg="#111")
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(self.root, anchor="w", font=("Consolas", 10))
        self.status.pack(fill="x")

        self._bind()
        self.load(0)

    # ------------------------------------------------------------- bindings
    def _bind(self) -> None:
        c, r = self.canvas, self.root
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        r.bind("<Key>", self.on_key)
        r.bind("<Right>", lambda e: self.step(1))
        r.bind("<Left>", lambda e: self.step(-1))
        r.bind("<Delete>", lambda e: self.delete_selected())

    # ---------------------------------------------------------------- image
    def label_path(self, i: int | None = None) -> pathlib.Path:
        p = self.paths[self.i if i is None else i]
        return p.with_suffix(".json")

    def load(self, i: int) -> None:
        self.i = i % len(self.paths)
        p = self.paths[self.i]
        img = self.Image.open(p).convert("RGB")
        self.img_w, self.img_h = img.size
        self.scale = min(self.max_width / self.img_w, self.max_height / self.img_h, 1.0)
        disp = img.resize((int(self.img_w * self.scale), int(self.img_h * self.scale)))
        self.photo = self.ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.width, height=disp.height)

        lp = self.label_path()
        self.boxes = json.loads(lp.read_text())["boxes"] if lp.exists() else []
        self.selected = None
        self.redraw()

    def step(self, d: int) -> None:
        self.save()
        self.load(self.i + d)

    # ----------------------------------------------------------------- draw
    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=self.photo)
        for j, b in enumerate(self.boxes):
            x1, y1, x2, y2 = [v * self.scale for v in b["xyxy"]]
            col = PALETTE[CLASSES.index(b["cls"]) % len(PALETTE)] if b["cls"] in CLASSES else "#fff"
            w = 3 if j == self.selected else 1
            c.create_rectangle(x1, y1, x2, y2, outline=col, width=w)
            c.create_text(x1 + 2, y1 - 8, anchor="w", text=b["cls"], fill=col,
                          font=("Consolas", 8))
        labeled = sum(1 for p in self.paths if p.with_suffix(".json").exists())
        self.status.config(
            text=(f"[{self.i + 1}/{len(self.paths)}] {self.paths[self.i].name}  "
                  f"boxes={len(self.boxes)}  class={CLASSES[self.cls_i]}  "
                  f"labeled={labeled}  auto_advance={self.auto_advance}")
        )

    # ---------------------------------------------------------------- mouse
    def on_press(self, e) -> None:
        self.drag = (e.x, e.y, e.x, e.y)

    def on_drag(self, e) -> None:
        if self.drag:
            x0, y0, _, _ = self.drag
            self.drag = (x0, y0, e.x, e.y)
            self.redraw()
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="#fff", dash=(3, 2))

    def on_release(self, e) -> None:
        if not self.drag:
            return
        x0, y0, x1, y1 = self.drag
        self.drag = None
        if abs(x1 - x0) < MIN_BOX or abs(y1 - y0) < MIN_BOX:
            self.select_at(e.x, e.y)
        else:
            xyxy = [min(x0, x1) / self.scale, min(y0, y1) / self.scale,
                    max(x0, x1) / self.scale, max(y0, y1) / self.scale]
            self.boxes.append({"cls": CLASSES[self.cls_i], "xyxy": [round(v, 1) for v in xyxy]})
            self.selected = len(self.boxes) - 1
        self.redraw()

    def select_at(self, sx: int, sy: int) -> None:
        x, y = sx / self.scale, sy / self.scale
        best, best_area = None, None
        for j, b in enumerate(self.boxes):
            x1, y1, x2, y2 = b["xyxy"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:   # innermost box wins
                    best, best_area = j, area
        self.selected = best

    # ------------------------------------------------------------------ keys
    def on_key(self, e) -> None:
        k = e.keysym.lower()
        if k == "q":
            self.save()
            self.root.destroy()
        elif k in ("n",):
            self.step(1)
        elif k in ("p",):
            self.step(-1)
        elif k == "s":
            self.save()
        elif k == "u" and self.boxes:
            self.boxes.pop()
            self.selected = None
        elif k == "d":
            self.delete_selected()
        elif k == "a":
            self.auto_advance = not self.auto_advance
        elif k in ("bracketleft", "bracketright"):
            d = 1 if k == "bracketright" else -1
            self.cls_i = (self.cls_i + d) % len(CLASSES)
            if self.selected is not None:
                self.boxes[self.selected]["cls"] = CLASSES[self.cls_i]
        elif k.isdigit():
            idx = (int(k) - 1) % 10
            if idx < len(CLASSES):
                self.cls_i = idx
                if self.selected is not None:
                    self.boxes[self.selected]["cls"] = CLASSES[idx]
        self.redraw()

    def delete_selected(self) -> None:
        if self.selected is not None and 0 <= self.selected < len(self.boxes):
            self.boxes.pop(self.selected)
            self.selected = None
        self.redraw()

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        if not self.boxes:
            return
        payload = {"image": self.paths[self.i].name, "w": self.img_w, "h": self.img_h,
                   "boxes": self.boxes}
        self.label_path().write_text(json.dumps(payload, indent=1))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    Annotator(sys.argv[1]).run()


if __name__ == "__main__":
    main()
