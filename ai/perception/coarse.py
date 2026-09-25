"""Collapse the 47-class label set into the few groups the agent acts on.

A CenterNet trained from scratch needs far more examples per class than hand
labelling can supply: at 14 frames and ~300 boxes, every fine-grained building
class sat at 0.0 AP while WALL -- the only class with enough instances --
reached 0.38.

Splitting hairs between a Cannon and an Archer Tower also buys the agent
nothing. Deployment needs three questions answered:

  * where is the loot?          -> LOOT
  * where can it shoot me?      -> DEFENSE, and AIR_DEFENSE separately,
                                   because the army is dragons
  * where is the Town Hall?     -> TOWN_HALL

So the fine labels stay on disk (they are strictly more information, and a
later model can use them) and are collapsed at load time.
"""
from __future__ import annotations

COARSE = {
    "TOWN_HALL": "TOWN_HALL",
    # air defence is its own group: it is the only thing that matters when the
    # army is electro dragons, and lumping it with ground defence hides it
    "AIR_DEFENSE": "AIR_DEFENSE",
    "AIR_SWEEPER": "AIR_DEFENSE",
    # anything that holds or produces loot
    "GOLD_STORAGE": "LOOT", "ELIXIR_STORAGE": "LOOT", "DARK_STORAGE": "LOOT",
    "GOLD_MINE": "LOOT", "ELIXIR_COLLECTOR": "LOOT", "DARK_DRILL": "LOOT",
    # anything that shoots
    "CANNON": "DEFENSE", "ARCHER_TOWER": "DEFENSE", "MORTAR": "DEFENSE",
    "WIZARD_TOWER": "DEFENSE", "TESLA": "DEFENSE", "BOMB_TOWER": "DEFENSE",
    "INFERNO": "DEFENSE", "XBOW": "DEFENSE", "EAGLE_ARTILLERY": "DEFENSE",
    "SCATTERSHOT": "DEFENSE", "MONOLITH": "DEFENSE", "SPELL_TOWER": "DEFENSE",
    "DEFENSE_OTHER": "DEFENSE", "CLAN_CASTLE": "DEFENSE",
    "HERO_ALTAR": "DEFENSE",
    # everything else that occupies ground
    "ARMY_CAMP": "OTHER_BUILDING", "BARRACKS": "OTHER_BUILDING",
    "LABORATORY": "OTHER_BUILDING", "SPELL_FACTORY": "OTHER_BUILDING",
    "WORKSHOP": "OTHER_BUILDING", "BUILDER_HUT": "OTHER_BUILDING",
    "OTHER_BUILDING": "OTHER_BUILDING", "UNLABELED_BUILDING": "OTHER_BUILDING",
    "WALL": "WALL",
}

COARSE_CLASSES = ["TOWN_HALL", "AIR_DEFENSE", "LOOT", "DEFENSE",
                  "OTHER_BUILDING", "WALL"]
COARSE_TO_ID = {c: i for i, c in enumerate(COARSE_CLASSES)}
