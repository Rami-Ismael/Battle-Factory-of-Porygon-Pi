---
name: verify-figure-pages
description: "Verify a figure page in Rami's Pokémon counter-search repo in a real browser: a static HTML explainer, chart page or storyboard loads in light and dark at desktop and phone width, its controls change the numbers they should, and its built page and still images are current. Use after building or changing a figure page, and before committing it or reporting it done."
---

# Verify figure pages

A figure page is a static `.html` file a reader opens: an explainer under `artifacts/`, `blog/` or `notes/Blog/visuals/`, usually built by a `build.py` from a template and data. This skill drives one the way a reader does and records proof.

It starts a local file server on a random port and one headless Google Chrome, both closed on exit. It runs no model, battle or paid service, and two runs can go side by side.

Run every command from the repo root. [`features/README.md`](features/README.md) maps each way a reader meets a figure page to the check that proves it. Read the matching feature file before reporting on that feature.

## 1. Doctor

```
uv run .claude/skills/verify-figure-pages/check_page.py --doctor
```

Done when both lines read `PASS`: the command runs inside the repo, and the installed Google Chrome starts headless. Note the commit and the count of uncommitted paths it prints: that is the tree being verified.

## 2. Pick the evidence folder

```
RUN="verify-evidence/figure-pages/$(date +%Y-%m-%d-%H%M)"
```

`verify-evidence/` is git-ignored and inside the workspace, so its files can be linked in a reply.

## 3. Build the page

A page with a `build.py` beside it is built first, so the browser sees what the template and data produce today:

```
python3 artifacts/wrong-mega-stone-validation-loss/build.py
```

Done when the build prints its summary line. [`features/built-pages.md`](features/built-pages.md) holds the check that the built page was already current.

## 4. Load it in every state

```
uv run .claude/skills/verify-figure-pages/check_page.py load PAGE.html --state "" --state "item=sash" --out "$RUN/load"
```

One capture per page, state, width and theme: 1440x900 and 390x844, light and dark. `--state` is a query string the page reads. Leave it out for a page with one state.

Done when it prints `RESULT pass` and you have opened one light and one dark screenshot from `$RUN/load/` for each changed page and seen the page drawn as intended. A `FAIL` line names its check id: `load-errors`, `load-overflow`, `load-images` or `load-charts`. [`features/page-load.md`](features/page-load.md) explains each.

## 5. Replay the reader's actions

```
uv run .claude/skills/verify-figure-pages/check_page.py journey .claude/skills/verify-figure-pages/journeys/wrong-mega-stone-validation-loss.json --out "$RUN/journey"
```

A journey file lists a reader's actions on one page and what the page must show after each. A page whose controls or scroll-triggered steps changed needs its journey run, and a page with none needs one written: [`features/reader-controls.md`](features/reader-controls.md) gives the step format.

Done when it prints `RESULT pass`, every id in the journey has a `PASS` line, and you have opened the screenshots it saved.

## 6. Cover the other ways in

A reader also meets a figure as a still image in a GitHub issue and inline in a T3 Code thread. When the change touches a storyboard, a published image or the page's layout, run [`features/still-images.md`](features/still-images.md) and [`features/chat-frame.md`](features/chat-frame.md).

Done when every feature file the change touches has been run or reported as not run, with the reason.

## 7. Report

State the result per check id, the commit and uncommitted count from the doctor, and the paths to `$RUN/load/load.json`, `$RUN/journey/journey.json` and the screenshots you looked at. Name anything not run and why.

## Cleanup

`check_page.py` closes its server and browser before it exits. Nothing else is started. Keep `$RUN`: it is the evidence.

## Helpers

`check_page.py` reads the repo and writes only to `--out`. Exit code 1 on any `FAIL`.

- `--doctor`: the two checks of step 1.
- `load PAGE... [--state QUERY]... [--viewport WxH]... --out DIR`: writes `load.json` and one full-page screenshot per capture.
- `journey FILE.json --out DIR`: writes `journey.json` and the journey's screenshots.
- `same URL=FILE...`: a published copy has the same bytes as a local file.

## Neighbouring checks

- Where a Pokémon sprite came from: `verify-sprites`.
- Type size, line length and hierarchy: `verify-typography`.
- Layout and interaction breaks that have shipped before: `design-check`.
- A chart script's words and data points: `code/scripts/test_chart_ports.py`.
