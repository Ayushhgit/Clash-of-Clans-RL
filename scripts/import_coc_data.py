"""Import real Clash of Clans stats from the `clash-of-clans-data` package.

Phase 1 of the plan says the simulator's numbers are placeholders to be
replaced by measurements. This does that in one step: the npm package
`clash-of-clans-data` ships the wiki's stats as structured JSON, so hitpoints,
DPS, range, target type and footprint size come from the real game instead of
from my guesses.

    npm pack clash-of-clans-data
    tar -xzf clash-of-clans-data-*.tgz -C datasets/coc_data
    python -m scripts.import_coc_data --town-hall 13 --write

Writes `simulator/game_data.py`, a generated table keyed by the simulator's
own class enum. `simulator/entities.py` picks it up when
`USE_REAL_STATS = True`, so the swap is one flag and the old hand-tuned
numbers stay in git history.

Mapping choices (real game -> simulator) mirror ai/perception/pipeline.py:
several real structures collapse onto one simulated one by behaviour.
"""
from __future__ import annotations

import argparse
import json
import pathlib

PKG = pathlib.Path("datasets/coc_data/package")

# simulator class name -> (json path relative to data/, mode used for stats)
SOURCES = {
    "TOWN_HALL":         ("home/town-hall/town-hall.json", "normal"),
    "CANNON":            ("home/defenses/cannon.json", "normal"),
    "ARCHER_TOWER":      ("home/defenses/archer-tower.json", "normal"),
    "MORTAR":            ("home/defenses/mortar.json", "normal"),
    "WIZARD_TOWER":      ("home/defenses/wizard-tower.json", "normal"),
    "AIR_DEFENSE":       ("home/defenses/air-defense.json", "normal"),
    "TESLA":             ("home/defenses/hidden-tesla.json", "normal"),
    "INFERNO":           ("home/defenses/inferno-tower.json", "normal"),
    "XBOW":              ("home/defenses/x-bow.json", "normal"),
    "STORAGE":           ("home/resource-buildings/gold-storage.json", None),
    "RESOURCE_BUILDING": ("home/resource-buildings/gold-mine.json", None),
    "WALL":              ("home/walls/wall.json", None),
}

UNIT_SOURCES = {
    "TANK":         "home/troops/golem.json",
    "DPS":          "home/troops/barbarian.json",
    "WALL_BREAKER": "home/troops/wall-breaker.json",
    "RANGED":       "home/troops/archer.json",
    "AIR":          "home/troops/dragon.json",
    "SPLASH":       "home/troops/wizard.json",
}


def load(rel: str) -> dict | None:
    p = PKG / "data" / rel
    if not p.exists():
        # walls live in a differently-shaped file in some versions
        alt = list((PKG / "data").rglob(pathlib.Path(rel).name))
        if not alt:
            return None
        p = alt[0]
    return json.loads(p.read_text(encoding="utf-8"))


def pick_level(entry: dict, town_hall: int) -> dict | None:
    """Highest level of this building unlocked at `town_hall`."""
    levels = entry.get("levels") or []
    ok = [lv for lv in levels
          if (lv.get("townHallRequired") or lv.get("townHallLevelRequired") or 1) <= town_hall]
    return (ok or levels)[-1] if (ok or levels) else None


def tiles(entry: dict) -> float:
    size = entry.get("size") or "3x3"
    try:
        w, h = (float(v) for v in str(size).lower().split("x"))
        return max(w, h)
    except ValueError:
        return 3.0


def dps_of(level: dict, mode: str | None) -> float:
    stats = level.get("stats") or {}
    if mode and mode in stats:
        s = stats[mode]
    elif stats:
        s = next(iter(stats.values()))
    else:
        s = {}
    return float(s.get("dps") or s.get("damagePerSecond") or 0.0)


def building_row(entry: dict, level: dict, mode: str | None) -> dict:
    modes = entry.get("modes") or {}
    m = modes.get(mode or "normal") or (next(iter(modes.values())) if modes else {})
    target = (entry.get("targetType") or "ground").lower()
    return {
        "hp": float(level.get("hitpoints") or 0),
        "radius": tiles(entry) / 2.0,
        "dps": dps_of(level, mode),
        "rng": float(m.get("range") or 0.0),
        "min_rng": float(m.get("minRange") or 0.0),
        "hits_air": target in ("air", "air & ground", "air and ground", "both"),
        "hits_ground": target in ("ground", "air & ground", "air and ground", "both"),
        "splash": 1.5 if (m.get("damageType") or "").lower() == "splash" else 0.0,
        "level": level.get("level"),
        "name": entry.get("name"),
    }


# The dataset records what a troop can *target*, not whether it flies, so the
# flying units are listed explicitly.
AIR_TROOPS = {
    "dragon", "baby-dragon", "balloon", "minion", "lava-hound", "healer",
    "electro-dragon", "dragon-rider", "flying-fortress", "super-dragon",
    "electro-owl", "phoenix", "inferno-dragon",
}


def unit_row(entry: dict, level: dict) -> dict:
    return {
        "hp": float(level.get("hitpoints") or 0),
        "dps": dps_of(level, "normal"),
        # movementSpeed is in the game's internal units; /12 puts barbarians at
        # ~1.3 tiles/s, which matches observed footage closely enough for a
        # simplified simulator. Refine in Phase 22 if the gap matters.
        "speed": float(entry.get("movementSpeed") or 16) / 12.0,
        "rng": float(entry.get("range") or 1.0),
        "is_air": entry.get("id") in AIR_TROOPS,
        "splash": 1.5 if (entry.get("damageType") or "").lower() == "splash" else 0.0,
        "housing": int(entry.get("housingSpace") or 1),
        "level": level.get("level"),
        "name": entry.get("name"),
    }


def build(town_hall: int) -> dict:
    if not PKG.exists():
        raise SystemExit(
            f"{PKG} not found.\n"
            "  npm pack clash-of-clans-data\n"
            "  mkdir -p datasets/coc_data && tar -xzf clash-of-clans-data-*.tgz "
            "-C datasets/coc_data"
        )
    buildings, units, missing = {}, {}, []
    for cls, (rel, mode) in SOURCES.items():
        entry = load(rel)
        if not entry:
            missing.append(rel)
            continue
        lv = pick_level(entry, town_hall)
        if lv:
            buildings[cls] = building_row(entry, lv, mode)
    for cls, rel in UNIT_SOURCES.items():
        entry = load(rel)
        if not entry:
            missing.append(rel)
            continue
        lv = pick_level(entry, town_hall)
        if lv:
            units[cls] = unit_row(entry, lv)
    return {"town_hall": town_hall, "buildings": buildings, "units": units,
            "missing": missing}


def emit(data: dict, out: pathlib.Path) -> None:
    lines = [
        '"""Real Clash of Clans stats, generated by scripts/import_coc_data.py.',
        "",
        "Do not edit by hand -- rerun the importer. Source: the npm package",
        "`clash-of-clans-data`, which mirrors the Clash of Clans wiki.",
        f"Town hall level used: {data['town_hall']} (highest level unlocked at that TH).",
        '"""',
        "",
        f"TOWN_HALL_LEVEL = {data['town_hall']}",
        "",
        "BUILDINGS = {",
    ]
    for cls, row in data["buildings"].items():
        lines.append(f"    {cls!r}: {row!r},")
    lines += ["}", "", "UNITS = {"]
    for cls, row in data["units"].items():
        lines.append(f"    {cls!r}: {row!r},")
    lines += ["}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--town-hall", type=int, default=13)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="simulator/game_data.py")
    args = ap.parse_args()

    data = build(args.town_hall)
    for cls, row in data["buildings"].items():
        print(f"{cls:<18} {row['name']:<16} lv{row['level']:<3} hp {row['hp']:>6.0f}  "
              f"dps {row['dps']:>5.0f}  rng {row['rng']:>4.1f}  "
              f"air {int(row['hits_air'])} ground {int(row['hits_ground'])}  "
              f"splash {row['splash']}")
    print()
    for cls, row in data["units"].items():
        print(f"{cls:<18} {row['name']:<16} lv{row['level']:<3} hp {row['hp']:>6.0f}  "
              f"dps {row['dps']:>5.0f}  speed {row['speed']:.2f}  rng {row['rng']:.1f}  "
              f"air {int(row['is_air'])}  "
              f"housing {row['housing']}")
    if data["missing"]:
        print("\nmissing sources:", data["missing"])
    if args.write:
        emit(data, pathlib.Path(args.out))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
