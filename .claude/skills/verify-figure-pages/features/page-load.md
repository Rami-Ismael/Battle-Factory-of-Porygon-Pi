# Page load

A reader opens a figure page and scrolls to the end. The page draws every image and chart, fits the screen without sideways scroll, and logs no error, on a desktop and a phone, in light and dark, in every state its address can ask for.

## Sub-features

- `load-errors`: no console error, page error, failed request or HTTP status of 400 or more.
- `load-overflow`: the page is no wider than the screen.
- `load-images`: every image decoded.
- `load-charts`: every ECharts container holds a drawn chart, and the chart library loaded when the page asks for it.

## How to get to it (user POV)

- Open the `.html` file in a desktop browser, in light or dark mode.
- Open it on a phone.
- Open it with a state in the address, such as `index.html?item=sash`.

## Driving it with check_page.py

Preconditions:

- The doctor passes.
- A page with a `build.py` has been built.

- **Open and scroll.** Load the page in every state. Run `uv run .claude/skills/verify-figure-pages/check_page.py load artifacts/wrong-mega-stone-validation-loss/index.html --state "" --state "item=sash" --out "$RUN/load"`. It prints one `PASS` line per capture, 8 for two states, each reading `errors 0, sideways scroll 0, broken images 0, charts 1/1`, then `RESULT pass`.
- **Open a still source.** Load a storyboard at its design width. Run `uv run .claude/skills/verify-figure-pages/check_page.py load artifacts/wrong-mega-stone-validation-loss/storyboard.html --viewport 800x900 --out "$RUN/load-still"`. Two `PASS` lines with `charts 0/0`, then `RESULT pass`.
- **Look.** Open one light and one dark screenshot per page from the `--out` folder. The page shows its last step with bars and numbers filled in, since the check scrolls the whole page before it captures.

## Gotchas

- A storyboard is a fixed board, 760 px wide, made to be captured as an image. At 390 px it reports `load-overflow` with 370 px of sideways scroll. Check it with `--viewport 800x900`; its phone reader sees the image, covered in `still-images.md`.
- Chart pages load ECharts from `cdn.jsdelivr.net`. Offline, `load-charts` fails with "the chart library did not load". Report it as not verified offline, not as a broken page.
- The check scrolls like a reader, so content that appears on scroll is captured in its final state. A page whose content appears only after a click needs a journey to prove it.
- `?static=1` on the wrong-Mega-Stone page turns motion off. The load check does not need it, and passing it as a state hides a broken animation.
- `smallest text` in the output is a measurement, not a check. Type size is judged by `verify-typography`.
