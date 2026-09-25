# LIVE_AGENT -- playing the real game

What runs against Clash of Clans today, what it learns from, and what is still
blind.

```
window capture -> button detection -> state machine -> attack plan
                                                     -> hotkey deploys
                                                     -> loot OCR -> learner
```

## Running it

```bash
python -m scripts.play_live --battles 5              # dry run, prints every click
python -m scripts.play_live --battles 5 --live       # actually plays
python -m scripts.play_live --battles 5 --live --strategy side   # force one pattern
```

`--live` is opt-in. Without it nothing is sent to the game, which is the right
default while changing anything.

## Capture

`ai.control.capture.WindowCapture` grabs the game window by title, crops the
Google Play Games chrome (title bar and icon rail), and returns just the game
viewport. Three bugs had to be fixed to get a correct frame, and all three are
easy to reintroduce:

| symptom | cause |
|---|---|
| captured the terminal instead of the game | screen-*region* capture grabs whatever is on top |
| frame came back black (mean pixel 4.5 vs 94.6) | `PrintWindow` needs `PW_RENDERFULLCONTENT` on GPU-composited windows |
| bottom of the game missing, buttons off-screen | process was DPI-unaware: a 1920x1080 screen reports as 1280x720 and the bitmap is the top-left crop |

`find_viewport` trims the launcher chrome by *uniformity* rather than
brightness -- an earlier brightness rule cut the frame in half on menu screens,
whose dark panels look like chrome.

## Perception

**Buttons, not a classifier.** `ai.perception.buttons` finds each button as a
saturated colour blob inside the corner it lives in, with area/aspect/fill
constraints, and infers the screen state from which buttons are present. This
is resolution-independent and needs no training data.

The CNN screen classifier (`scripts/train_ui.py`, 79% held-out on 107 frames)
is *not* used live: it read HOME correctly at 1095x614 and called it
BASE_PREVIEW on all 12 frames at 1670x949. Rules scored 81% on the same
HUD-visible frames with no training at all, and do not care about resolution.

Constraints matter more than they look: the green RETURN button and *grass*
share a hue, so a hue-only rule reported RESULT on 35 of 89 preview frames.
Area and aspect bounds are what separate a button from a lawn.

**Numbers.** `ai.perception.digits` reads loot and HUD figures by template
matching. The font is fixed, so a couple of labelled examples per digit is
enough (`scripts/build_digit_templates.py`). Reading is a dynamic-programming
parse over the strip, which was necessary:

* connected components fail -- "708 081" is two blobs, not six glyphs;
* greedy left-to-right matching fails -- a '1' template is half the width of an
  '8', so one narrow mismatch shifts everything after it ("3360" -> "33113").

Current accuracy: 7 of 8 labelled numbers exact. The miss is a 105x34 crop of
the smallest text in the set.

## Acting

Cards are selected by **keyboard hotkey** (`1 2 Q W E A S D F`, printed on each
card), not by clicking a detected card box. The first version detected 6 cards
on a 9-card bar and half the deploys landed in gaps -- only 2 heroes and 2
dragons ever came out. Hotkeys are exact.

**Each card is tapped exactly as many times as it holds.** An earlier version
tapped every card a fixed number of times, which is wrong in both directions
at once: with 4 balloons, 8 electro dragons and 3 heroes, three taps per card
under-deployed the balloons, stranded 5 dragons, and tapped each hero card
three times when it holds one hero.

That last one is the expensive mistake. Once a hero is down its card becomes
the **ability** button, so the spare taps fire the ability immediately, at the
drop point, with nothing to rage or heal. An older comment here claimed
over-clicking was safe because exhausted cards ignore clicks -- true for
troops, false for heroes.

Where the counts come from:

| source | reliability |
|---|---|
| `--army "4,8,H,H,H"` | exact; authoritative when given |
| badge reader, hero/troop | reliable -- heroes are the cards with no `xN` badge |
| badge reader, the number | **poor** -- 2 of 6 counts on the one labelled bar |

`ai.perception.army_bar` reads the top-right `xN` badge. Splitting hero from
troop works because heroes simply have no badge, and ink *density inside the
text band* separates a badge from artwork bleed cleanly (heroes 0.02-0.13,
troops 0.19-0.26) where raw ink does not (0.098 vs 0.128).

The numbers themselves are read with the HUD digit templates, which are cut
from larger, lighter text, and they misread the small bold badge font -- a "5"
came back as 116. So a card whose count cannot be read is **skipped**, not
guessed: losing those units costs one raid, guessing high on a hero fires its
ability. Pass `--army` to deploy them.

## Attack plans

`ai.policies.deploy_patterns` holds six patterns -- `side`, `spearhead`,
`two_prong`, `quadrant`, `line`, `ring` -- each randomised in approach angle,
spread and jitter. **One plan is sampled per attack and every card follows
it**; that is what makes an attack coherent rather than merely random. The
original uniform ring is kept as `ring` so it stays measurable, and it is
expected to be the worst.

`army_points()` generates the drops for the **whole army at once**, then deals
them round-robin to the cards. Calling `points()` once per card instead
returned the *same* coordinates every time -- the patterns are deterministic
given a plan -- so five cards stacked on one shape, and `ring` with four drops
is literally a square. That is what "it deploys in the same square every time"
looked like from the outside.

Heroes are the exception: they all go to **one** point at the head of the
attack. Three heroes spread around the ring is three lone units walking into
three separate defences.

## Learning

`ai.policies.online_strategy.FactoredLearner` runs Thompson sampling over
**four** decisions at once, all updated from the one reward a raid produces:

| dimension | arms |
|---|---|
| `pattern` | side, spearhead, two_prong, quadrant, line, ring |
| `hero_timing` | with_troops (0s), after_funnel (6s), late (14s) |
| `hero_spot` | same, ahead, flank |
| `spell_timing` | early_shallow (4s, lead 0.20), mid (8s, 0.42), late_deep (14s, 0.62) |

The full cross-product is 162 combinations and each one costs a three-minute
raid, so the dimensions are learned **independently** and share the reward.
That treats them as separable when they are not -- a late hero suits some
patterns better than others -- so it converges on a good combination rather
than provably the best one. With raids this expensive that is the right trade,
and `table()` reports each dimension separately so the assumption is visible.

* **Reward** = loot captured / loot available, both read by OCR. Raw loot would
  mostly measure how rich the base was, which the tactics had nothing to do
  with.
* **Unreadable is not zero.** A failed OCR and a genuinely empty raid both
  returned 0.0, so an unreadable result screen was recorded as "this tactic
  scored nothing" and the bandit learned from a measurement it never made.
  `loot_reward` now returns None and the raid is skipped.
* **Persistence**: `experiments/attack_stats.json`. Adding a new dimension
  merges into the existing file rather than resetting what has been learned.
* **Why a bandit and not RL**: no state carries between raids, the action set
  is small and discrete, and each reward costs a three-minute raid. Thompson
  sampling explores in proportion to remaining uncertainty, so a tactic that
  lost once is not discarded.

Expect it to need a few dozen raids before the table means much: 15 arms, one
sample each per raid, on a reward that varies a lot with the base drawn.

## Spells, without a detector

A spell is worth something only on top of the fighting, and the fighting is not
where the troops landed. There is no object detector, so the agent cannot see
its own troops -- but it does not have to guess blindly, because movement here
is predictable: units land on the ring, acquire the nearest building and walk
inward. So the fight at time t is on the line from the drop points toward the
base centre, and `attack_plan.advance()` is that model.

How far to lead is **paired with the delay** into coherent tactics rather than
learned as a separate dimension. Learned separately, the bandit could pick a
long delay with a short lead and spend its spells on empty ground.

Deployment is therefore a *schedule*, not a burst:

```
troop x12 @ 0.0-2.3s; hero x3 @ 6.8-7.6s; spell x7 @ 10.5-22.5s
```

Deploy taps use `TAP_DELAY` (0.10s), not the 0.6s `step_delay` that UI buttons
need to redraw. At 0.6s a 12-troop wave takes 7.2s and the schedule slips past
its own hero time before the first hero lands; `run_schedule` warns when it
falls behind.

## What is still blind
## What is still blind

Deployment *positions* come from the pattern, not from the base. There is no
object detector, so the agent cannot see where the storages are, which
compartment is weak, or where the air defences sit. That is the next real gain
and it needs the labelling seed described in `docs/LABELING.md`.

Until then, the honest description of this agent is: **a competent macro loop
with a randomised-but-coherent attack, learning which attack shape pays.**
