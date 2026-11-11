# Figure page verification map

The maintained list of every way a reader meets a figure page in this repo, and the check that proves each one works. Read this index, then the matching feature file.

| Feature file | What the reader does | Check ids |
|---|---|---|
| [page-load.md](page-load.md) | Opens the page and scrolls to the end, on a desktop or a phone, in light or dark | `load-errors`, `load-overflow`, `load-images`, `load-charts` |
| [reader-controls.md](reader-controls.md) | Watches a step play, changes a control, taps a marked word | the ids in the page's journey file, `journey-errors` |
| [built-pages.md](built-pages.md) | Opens the built page, which must match its template and data | `build-fresh` |
| [still-images.md](still-images.md) | Sees the figure as an image in a GitHub issue | `still-fresh`, `still-published` |
| [chat-frame.md](chat-frame.md) | Reads the page inline in a T3 Code thread | `frame-console`, `frame-layout` |

## Baseline

- Run from the repo root, on the commit or working tree being verified.
- Run the doctor in `../SKILL.md` first.
- Build a page that has a `build.py` before driving it.

## Driving conventions

- Everything a browser must draw runs through `check_page.py`. It serves the repo itself, so a page is always named by its repo path.
- A journey acts by accessible name for buttons and reads results by the page's own ids.
- A journey waits for a result up to its step's `within` limit. It never sleeps a fixed time.
- Reach a page state through the address, as `--state` does, or through the control a reader would use, in a journey.

## Proof and skip reporting

- Load proof is `load.json` plus one light and one dark screenshot per changed page, looked at.
- Journey proof is `journey.json`, which pairs each action with what the page showed, plus the journey's screenshots.
- A feature that was not run is reported as not run, with its check ids and the reason. A passing load never stands in for a journey.
- Publishing is outside every check here: pushing an image or editing an issue needs the owner's ask.

## Feature entry contract

Each feature file starts with an H1 and one paragraph on what the reader does, then four H2 sections in this order: `Sub-features`, `How to get to it (user POV)`, `Driving it with check_page.py` (or the named tool), `Gotchas`.

## Pages with a journey

| Page | Journey file |
|---|---|
| `artifacts/wrong-mega-stone-validation-loss/index.html` | `../journeys/wrong-mega-stone-validation-loss.json` |

A figure page missing from this table has load proof only. Add its journey and its row when its controls are next changed.
