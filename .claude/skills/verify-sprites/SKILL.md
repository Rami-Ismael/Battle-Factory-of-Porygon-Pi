---
name: verify-sprites
description: "Verify that every Pokémon sprite in Rami's Pokémon counter-search repo comes from the PokeAPI sprites repository, in the shared sprite folder, embedded in pages, and hotlinked. Use after changing the sprite folder, building or adding a page that shows Pokémon sprites, or syncing notes from the Obsidian vault."
---

# Verify sprites

Pokémon sprites come from the [PokeAPI sprites repository](https://github.com/PokeAPI/sprites) (owner decision 2026-11-09). Pokémon Showdown art is the regression this skill catches. Item icons (Smogon) and trainer sprites (Showdown) are outside it.

The surface is static pages plus one shared sprite folder, `notes/Blog/visuals/search-loop/sprites/`. Files there are named by Showdown id (`charizardmegay.png`); `pokeapi_ids.json` beside them maps each id to its PokeAPI file. Pages reach sprites three ways: embedded as base64, loaded from the folder by relative path, or hotlinked from `raw.githubusercontent.com/PokeAPI/sprites`.

Run every command from the repo root. [`features/README.md`](features/README.md) maps each way a page reaches sprites to the check that proves it; read the matching feature file before reporting on that feature.

## 1. Doctor

```
node .claude/skills/verify-sprites/render_pages.cjs --doctor
```

Done when both lines read `PASS`: the repo root holds the sprite folder, and Playwright loads. On a Playwright `FAIL`, set `PLAYWRIGHT_DIR` to a folder containing the `playwright` package and rerun. The render uses the system Chrome (`channel: 'chrome'`).

## 2. Pick the evidence folder

```
RUN="verify-evidence/sprites/$(date +%Y-%m-%d-%H%M)"
```

`verify-evidence/` is git-ignored and inside the workspace, so its files can be linked in a reply.

## 3. Check the bytes

```
python3 .claude/skills/verify-sprites/check_sprites.py --out "$RUN"
```

Downloads each store sprite from PokeAPI (285 small files, about 10 seconds) and scans every tracked file. Done when it prints `RESULT pass` with no `skipped` list. `--offline` skips the downloads and prints them as skipped; a run with skips is reported as partial, with the skipped check ids named.

Each `FAIL` line names its check id and the offending files. The check ids are explained in the feature files.

## 4. Render the pages

```
node .claude/skills/verify-sprites/render_pages.cjs "$RUN/render"
```

Renders eight pages that between them cover all three ways of reaching sprites, in light and dark. To render other pages, append their repo-relative paths; always append a page this task changed or built. Done when it prints `RESULT pass` and you have opened at least one light and one dark screenshot from `$RUN/render/` for each changed page and seen the sprites drawn in their slots.

A page fails when it has a broken image, a failed request, a page error, a request to a Showdown sprite URL, or shows no sprite at all. A page that shows sprites only after a click needs an entry in `PAGE_DRIVES` at the top of `render_pages.cjs`.

## 5. Report

State the result per check id, the commit checked, and the path to `$RUN/report.json`, `$RUN/render/render.json` and the screenshots you looked at. Name anything skipped and why.

## Cleanup

`render_pages.cjs` starts its own local server on a random port and closes it, along with the browser, before it exits. Nothing else is started. Keep `$RUN`: it is the evidence.

## Helpers

- `check_sprites.py` reads only. `--out DIR` writes `report.json`; `--offline` skips network checks. Exit code 1 on any failure.
- `render_pages.cjs <out-dir> [page ...]` writes `render.json` and screenshots. Exit code 1 on any failure.
- `showdown_sha256.json` holds the hash of every Showdown sprite the repo carried before the move. It is how `check_sprites.py` recognises old art; leave it as it is.
