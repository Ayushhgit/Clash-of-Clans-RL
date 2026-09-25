"""Drive the real game: search bases, attack, return home, repeat.

    python -m scripts.play_live --battles 1              # dry run, clicks printed
    python -m scripts.play_live --battles 1 --live       # actually clicks

WHAT THIS IS, PRECISELY
-----------------------
The *strategic* loop is real: state comes from `ai.perception.buttons` (colour
blobs in known corners, so it works at any window size), and every click lands
on a button that was detected in this frame, not on a coordinate measured
months ago at a different resolution.

The *tactical* part is not the trained policy. The PPO agent chooses where to
deploy from a list of detected buildings, and there is no trained detector yet,
so it would be choosing blind. Instead the attack is built from a randomised
pattern plus a set of tactics the agent is learning across raids -- which
approach shape, when the heroes go in, and when and how far ahead the spells
land (`ai.policies.attack_plan`). Do not read the results as "the RL agent's
win rate"; they are a learned-bandit attack's.

Deployment is a *schedule*, not a burst: the waits are what make hero timing
and spell delay mean anything. When a detector exists, `attack_plan` is the
module to replace -- it is the only place that guesses where the fighting is.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import math
import time

import numpy as np

from ai.control.capture import SessionRecorder, WindowCapture
from ai.control.controller import DesktopController, NullController
from ai.perception.army_bar import describe, parse_army_spec, read_army
from ai.perception.buttons import find_army_slots, find_buttons, infer_state
from ai.perception.digits import (RESULT_ROWS, crop_frac, load_templates,
                                  read_loot_by_icon, read_number)
from ai.policies import deploy_patterns as dp
from ai.policies.attack_plan import DIMENSIONS, build_schedule
from ai.policies.attack_plan import describe as schedule_summary
from ai.policies.online_strategy import FactoredLearner, loot_reward


# Google Play Games prints a hotkey on every army card (1 2 Q W E A S D F).
# Selecting a card by keypress is exact, so the fragile part of deployment --
# locating each card in the bar by eye -- disappears. The first live run
# detected 6 cards on a 9-card bar and half the deploys hit gaps.
ARMY_HOTKEYS = ["1", "2", "q", "w", "e", "a", "s", "d", "f"]


# Spell cards must not be deployed on the ground like troops -- they belong on
# top of the fighting. Until that is handled, only the troop and hero cards are
# used. Clash orders the bar troops -> heroes -> spells, so the first few keys
# are the ones we want.
TROOP_KEYS_DEFAULT = "1,2,q,w,e"

LEARNER_PATH = pathlib.Path("experiments/attack_stats.json")

# Settle time after a deploy tap. Smaller than `step_delay`, which exists so UI
# buttons have time to redraw -- but not much smaller, because the game drops
# taps that arrive too fast. At 0.10s a raid deploying 4 wall breakers and 8
# dragons landed only 2 and 3 of them ("Troops expended: x2, x3" on the result
# screen), while the spells in the same raid -- spaced 2s apart -- all landed.
# That is the measurement that set this value.
TAP_DELAY = 0.35


class LivePlayer:
    def __init__(self, window: str, live: bool, record: str | None,
                 step_delay: float = 0.6, strategy: str = "random",
                 troop_keys: str = TROOP_KEYS_DEFAULT, seed: int | None = None,
                 army: str | None = None):
        self.cap = WindowCapture(window)
        self.ctl = DesktopController(dry_run=not live) if live else NullController()
        self.live = live
        self.step_delay = step_delay
        self.origin = self.cap.client_origin()
        self.rec = SessionRecorder(name=record, downscale=None) if record else None
        self.log: list = []
        self.strategy = strategy
        self.troop_keys = [k.strip() for k in troop_keys.split(",") if k.strip()]
        self.rng = np.random.default_rng(seed)
        self.army_spec = (parse_army_spec(army, len(self.troop_keys))
                          if army else None)
        # learning across raids: not just which pattern, but when the heroes
        # go in and where the spells land. New dimensions merge into an
        # existing stats file rather than resetting what has been learned.
        self.learner = FactoredLearner.load(DIMENSIONS, LEARNER_PATH)
        self.digits = load_templates()
        if not self.digits:
            print("[warn] no digit templates; loot cannot be read and nothing "
                  "will be learned. Run scripts/build_digit_templates.py")

    # ----------------------------------------------------------------- utils
    def observe(self) -> tuple:
        rgb = self.cap.grab()
        state, buttons = infer_state(rgb)
        if self.rec is not None:
            self.rec.add(rgb, state)
        return rgb, state, buttons

    def click(self, xy: tuple, what: str, delay: float | None = None) -> None:
        """xy is in viewport pixels; the controller needs screen pixels.

        `delay` overrides the settle time. UI buttons need the default ~0.6s
        for the game to redraw; deploy taps must not, or the pause after each
        one dominates the schedule -- at 0.6s a 12-troop wave takes 7.2s and
        every later step slips past the time it was planned for.
        """
        sx, sy = self.origin[0] + xy[0], self.origin[1] + xy[1]
        self.log.append({"what": what, "viewport": xy, "screen": (sx, sy)})
        print(f"    click {what:<26} viewport {xy}  screen ({sx},{sy})"
              f"{'' if self.live else '   [DRY RUN]'}", flush=True)
        if self.live:
            self.ctl.click(sx, sy)
        time.sleep(self.step_delay if delay is None else delay)

    def read_loot(self, rgb) -> dict:
        """Loot numbers from whatever screen this is (result rows)."""
        if not self.digits:
            return {}
        # Anchored on the resource icons, not on fixed fractions of the frame.
        # The fractional boxes were calibrated on one capture and clipped the
        # digits in half on another, which read "133 905" as 7114114 and taught
        # the bandit that a defeat was a perfect raid.
        by_icon = read_loot_by_icon(rgb, self.digits)
        if any(v is not None for v in by_icon.values()):
            return by_icon
        return {name: read_number(crop_frac(rgb, box), self.digits)
                for name, box in RESULT_ROWS.items()}

    def read_available(self, rgb) -> dict:
        """The 'Available Loot' panel shown on BASE_PREVIEW."""
        if not self.digits:
            return {}
        rows = {"gold": (0.06, 0.140, 0.17, 0.190),
                "elixir": (0.06, 0.185, 0.17, 0.235),
                "dark": (0.06, 0.230, 0.17, 0.280)}
        return {name: read_number(crop_frac(rgb, box), self.digits)
                for name, box in rows.items()}

    def press_key(self, key: str, what: str, delay: float | None = None) -> None:
        self.log.append({"what": what, "key": key})
        print(f"    key   {what}{'' if self.live else '   [DRY RUN]'}", flush=True)
        if self.live:
            self.ctl.key(key)
        time.sleep(self.step_delay * 0.5 if delay is None else delay)

    def settle(self, timeout: float = 90.0) -> tuple:
        """Get back to a state the loop knows how to act from.

        Between raids the game shows loading screens and award popups that no
        rule matches, and the village itself takes a while to redraw. The first
        multi-raid run died here: one UNKNOWN frame at the top of battle 2 and
        the whole session stopped. So poll, clear whatever is dismissable, and
        only give up after a real timeout.
        """
        end = time.time() + timeout
        last = None
        while time.time() < end:
            rgb, state, buttons = self.observe()
            if state in ("HOME", "ATTACK_MENU", "ARMY", "BASE_PREVIEW"):
                return rgb, state, buttons
            if state == "RESULT" and "RETURN_BUTTON" in buttons:
                self.click(buttons["RETURN_BUTTON"].centre, "RETURN_BUTTON")
                time.sleep(2.0)
            elif state != last:
                print(f"    ...settling, seeing {state}", flush=True)
                last = state
            time.sleep(1.5)
        return rgb, state, buttons

    def wait_for(self, states: set, timeout: float = 30.0) -> tuple:
        end = time.time() + timeout
        last = None
        while time.time() < end:
            rgb, state, buttons = self.observe()
            if state in states:
                return rgb, state, buttons
            if state != last:
                print(f"    ...waiting for {sorted(states)}, seeing {state}", flush=True)
                last = state
            time.sleep(1.0)
        return rgb, state, buttons

    def run_schedule(self, steps: list) -> None:
        """Execute a timed attack.

        Waits are real: hero timing and spell delay only mean something if the
        troops have actually had those seconds to walk in. The card is
        re-selected whenever it changes, because deploying the last unit of a
        card makes the game advance the selection on its own.
        """
        t0 = time.perf_counter()
        current = None
        late = 0.0
        for i, st in enumerate(steps):
            wait = st.t - (time.perf_counter() - t0)
            if wait > 0:
                if self.live:
                    time.sleep(wait)
                else:
                    print(f"    (wait {wait:.1f}s)", flush=True)
            else:
                late = max(late, -wait)
            if st.key != current:
                self.press_key(st.key, f"select {st.key}", delay=TAP_DELAY)
                current = st.key
            self.click(st.point, f"{st.label} {i + 1}/{len(steps)} @t={st.t:.1f}s",
                       delay=TAP_DELAY)
        if late > 1.0:
            print(f"  ! schedule ran {late:.1f}s behind: taps are slower than "
                  f"the plan assumes", flush=True)

    def resolve_army(self, rgb, slots: list) -> list:
        """What each card holds. `--army` wins over the badge reader.

        The hero test (no count badge) is reliable; the *counts* are not -- on
        the one labelled bar available, badge OCR read 2 of 6 counts correctly
        using the HUD digit templates, which are cut from larger, lighter text.
        So counts come from `--army` when given, and cards that neither source
        can resolve are skipped rather than guessed.
        """
        if self.army_spec is not None:
            return list(self.army_spec)
        return read_army(rgb, slots[:len(self.troop_keys)], self.digits)

    # ------------------------------------------------------------------ loop
    def play(self, battles: int, max_next: int = 3) -> list:
        results = []
        for b in range(1, battles + 1):
            print(f"\n=== battle {b}/{battles} ===", flush=True)
            # settle() rather than observe(): between raids the game shows
            # loading screens, award popups and a village that has not finished
            # redrawing, none of which match a state rule. A single observe()
            # here sees UNKNOWN and abandons the battle, which is exactly what
            # happened on the raid after the first one. settle() was written
            # for this and was never actually called from anywhere.
            rgb, state, buttons = self.settle()
            print(f"  state: {state}  buttons: {sorted(buttons)}", flush=True)

            if state == "RESULT":
                if "RETURN_BUTTON" in buttons:
                    self.click(buttons["RETURN_BUTTON"].centre, "RETURN_BUTTON")
                rgb, state, buttons = self.wait_for({"HOME"})

            if state == "HOME":
                if "ATTACK_BUTTON" not in buttons:
                    print("  ! no attack button found; skipping battle")
                    continue
                self.click(buttons["ATTACK_BUTTON"].centre, "ATTACK_BUTTON")
                rgb, state, buttons = self.wait_for({"ATTACK_MENU", "BASE_PREVIEW"},
                                                    timeout=20)

            if state == "ATTACK_MENU":
                if "FIND_MATCH_BUTTON" not in buttons:
                    print("  ! attack menu open but no Find-a-Match button; skipping")
                    continue
                self.click(buttons["FIND_MATCH_BUTTON"].centre, "FIND_MATCH_BUTTON")
                rgb, state, buttons = self.wait_for({"ARMY", "BASE_PREVIEW"}, timeout=30)

            if state == "ARMY":
                # this build confirms the army before searching
                self.click(buttons["ARMY_ATTACK_BUTTON"].centre, "ARMY_ATTACK_BUTTON")
                # matchmaking takes a few seconds and shows a spinner
                rgb, state, buttons = self.wait_for({"BASE_PREVIEW"}, timeout=60)
                if state != "BASE_PREVIEW":
                    print(f"  ! matchmaking did not reach BASE_PREVIEW (saw {state}); skipping")
                    continue

            skipped = 0
            while state == "BASE_PREVIEW" and skipped < max_next:
                print(f"  base found (skipped {skipped})", flush=True)
                break        # base-selection agent would decide NEXT here

            if state != "BASE_PREVIEW":
                print(f"  ! expected BASE_PREVIEW, got {state}; skipping battle")
                continue

            h, w = rgb.shape[:2]
            slots = find_army_slots(rgb)
            # Every decision the agent makes this raid, drawn together: which
            # pattern, when the heroes go in, where they go, and when and how
            # far ahead the spells land. Thompson sampling, so a tactic that
            # lost once keeps a shrinking share of the raids rather than being
            # dropped after one bad base.
            choices = self.learner.select(self.rng)
            if self.strategy != "random":
                choices["pattern"] = self.strategy
            plan = dp.sample_plan(self.rng, choices["pattern"])
            available = self.read_available(rgb)
            if any(available.values()):
                print(f"  available loot: {available}", flush=True)

            army = self.resolve_army(rgb, slots)
            print(f"  army: {describe(army, self.troop_keys)}  "
                  f"({len(slots)} cards seen)", flush=True)
            unread = [k for k, a in zip(self.troop_keys, army) if not a["readable"]]
            if unread:
                print(f"  ! unreadable card counts on {unread}: skipping those "
                      f"cards. Pass --army to deploy them.", flush=True)

            steps = build_schedule(plan, w, h, army, self.troop_keys,
                                   choices, self.rng)
            print(f"  plan: {plan.describe()}", flush=True)
            print(f"  choices: {choices}", flush=True)
            print(f"  schedule: {schedule_summary(steps)}", flush=True)
            self.run_schedule(steps)

            rgb, state, buttons = self.wait_for({"RESULT"}, timeout=200)
            print(f"  battle ended in state {state}", flush=True)

            gained = self.read_loot(rgb) if state == "RESULT" else {}
            reward = loot_reward(gained, available)
            # Only score a raid whose plan was actually carried out. If most of
            # the army never left the bar because its counts could not be read,
            # a low reward says nothing about the tactics that were chosen, and
            # feeding it to the bandit teaches the wrong lesson.
            planned = sum(1 for a in army if a["readable"])
            coverage = planned / max(len(army), 1)
            if reward is not None and state == "RESULT" and coverage < 0.6:
                print(f"  ! only {planned}/{len(army)} cards were deployable; "
                      f"not scoring this raid", flush=True)
                reward = None
            if reward is not None and state == "RESULT":
                self.learner.update(choices, reward,
                                    {"gained": gained, "available": available,
                                     "angle_deg": round(math.degrees(plan.angle))})
                self.learner.save(LEARNER_PATH)
                print(f"  loot: {gained}   reward {reward:.3f}", flush=True)
                print(self.learner.table(), flush=True)

            results.append({"battle": b, "end_state": state,
                            "plan": plan.name, "angle_deg": round(math.degrees(plan.angle)),
                            "spread_deg": round(math.degrees(plan.spread)),
                            "gained": gained, "reward": reward})
            if state == "RESULT" and "RETURN_BUTTON" in buttons:
                self.click(buttons["RETURN_BUTTON"].centre, "RETURN_BUTTON")
                self.wait_for({"HOME"}, timeout=90)
        return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="Clash of Clans")
    ap.add_argument("--battles", type=int, default=1)
    ap.add_argument("--units", type=int, default=None,
                    help="deprecated and ignored: taps now come from what each "
                         "card actually holds (--army or the badge reader)")
    ap.add_argument("--live", action="store_true",
                    help="actually send clicks (default is a dry run)")
    ap.add_argument("--record", default=None, help="session name for captured frames")
    ap.add_argument("--strategy", default="random",
                    choices=["random", *dp.PATTERNS], help="deployment pattern")
    ap.add_argument("--army", default=None,
                    help="units per card, matching --troop-keys, e.g. "
                         "\"4,8,H,H,H\" for 4 balloons, 8 dragons and three "
                         "heroes. 'H' means a hero: one unit, one tap. This is "
                         "authoritative -- badge OCR reads heroes reliably but "
                         "counts poorly.")
    ap.add_argument("--troop-keys", default=TROOP_KEYS_DEFAULT,
                    help="army hotkeys to use, in bar order. Extend it to "
                         "include spell cards (e.g. 1,2,q,w,e,a,s) and mark "
                         "them in --army as S<n>.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    try:
        player = LivePlayer(args.window, args.live, args.record,
                            strategy=args.strategy, troop_keys=args.troop_keys,
                            seed=args.seed, army=args.army)
    except RuntimeError as e:
        raise SystemExit(
            f"{e}\n\nStart the game (window title must contain "
            f"{args.window!r}) and leave it visible, then rerun.")
    print(f"mode: {'LIVE - sending real clicks' if args.live else 'DRY RUN - no input sent'}")
    print(f"viewport origin on screen: {player.origin}")

    results = player.play(args.battles)
    print("\nresults:", json.dumps(results))
    print(f"{len(player.log)} clicks {'sent' if args.live else 'planned'}")
    if player.rec:
        print("frames saved to", player.rec.dir)
    out = pathlib.Path("experiments/live_clicks.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(player.log, indent=1))


if __name__ == "__main__":
    main()
