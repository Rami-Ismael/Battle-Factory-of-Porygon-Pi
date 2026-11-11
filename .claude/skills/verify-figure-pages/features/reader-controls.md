# Reader controls

A reader scrolls to a step and watches it play, changes a control and sees every number that depends on it move, and taps a marked word to open its definition. A journey file replays those actions on one page and checks what the page shows after each.

## Sub-features

The ids of `journeys/wrong-mega-stone-validation-loss.json`, for `artifacts/wrong-mega-stone-validation-loss/index.html`:

- `draw-hides`: scrolling to step 1 hides 62 of the team's 84 fields.
- `block-replays`: scrolling to step 3 shows the model's own guess, cost 0.32, then the legal Pokémon construction filter blocks 9 rows and the cost becomes 9,997.
- `swap-item`: choosing Focus Sash moves the cost to 0.12, Whimsicott's share to 88.6%, the team's loss to 0.9 and the validation loss to 1.681.
- `swap-back`: choosing Glimmoranite restores 9,997, 2,292, 574.1 and 1.787.
- `definition`: tapping the word "draw" opens its definition in place, and tapping it again closes it.
- `journey-errors`: nothing errored along the way.

## How to get to it (user POV)

- Scroll to "Step 1" and to "Step 3".
- Under "Whimsicott holds", choose the `Focus Sash` button, then the `Glimmoranite` button.
- Tap the dotted word `draw` in step 1.

## Driving it with check_page.py

Preconditions:

- The doctor passes and the page has been built.

- **Replay.** Run the journey. Run `uv run .claude/skills/verify-figure-pages/check_page.py journey .claude/skills/verify-figure-pages/journeys/wrong-mega-stone-validation-loss.json --out "$RUN/journey"`. It prints 20 `PASS` lines for the five ids above, `PASS journey-errors`, then `RESULT pass | 20 expectations, 0 failed`.
- **Look.** Open `step3-model-alone.png`, `step3-after-tables.png`, `focus-sash.png` and `definition-open.png` in `$RUN/journey/`. The first shows cost 0.32 and no hatched row; the second shows nine hatched rows and 9,997.
- **Write a journey for another page.** Copy the file and replace its steps. A step is one of:
  - `{"scroll": "#css"}` brings an element to the middle of the screen.
  - `{"click": {"role": "button", "name": "Focus Sash"}}` clicks by accessible name. Add `"exact": true` for a whole-name match. A CSS string also works.
  - `{"press": "ArrowRight"}` presses a key.
  - `{"expect": "#css", "text": "1.681", "within": 3000}` waits up to `within` ms for that exact text. `"contains"` matches part of it, `"count"` the number of matches, and `"attr"` with `"value"` an attribute.
  - `{"shot": "name"}` saves `name.png`.
  
  Give each step an `"id"` that names the reader's action. Then add the page to the table in `README.md`.

## Gotchas

- Numbers roll for about half a second and bars move for as long. Give every `expect` after an action a `within`, or it reads a number mid-roll.
- The 0.32 state of step 3 lasts 0.9 seconds. Its `expect` must directly follow the scroll, with `within` at 800 or less.
- A scroll-triggered step plays once per page load. To see it again, load the page again.
- A click by accessible name matches part of the name. `"name": "draw"` without `"exact": true` can match another button.
- The journey's numbers are the page's real data. When `probe_worst_check.json` changes, update the journey from the new page; a journey edited to pass proves nothing.
