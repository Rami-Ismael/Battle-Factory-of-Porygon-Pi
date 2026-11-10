# Hotlinked pages

Three pages hold no sprite bytes. They build a PokeAPI URL in the browser from a species lookup and load the picture when opened, so only a render with network access proves they work.

## Sub-features

- `hotlink-urls`: no tracked file contains a `pokemonshowdown.com/sprites` URL, apart from the trainer sprites in `brand/logo/build-video/sprites/sources.json`.
- `hotlink-default`: `artifacts/jev-request-anatomy.html` and `notes/Blog/visuals/team-lineage/team-lineage.html` load default front sprites (`sprites/pokemon/<id>.png`).
- `hotlink-home`: `notes/Blog/visuals/looped-diffusion/index.html` loads the larger HOME renders (`sprites/pokemon/other/home/<id>.png`).

## How to get to it (user POV)

- Open each of the three pages in a browser with network access. The team's sprites appear on load.

## Driving it with check_sprites.py and render_pages.cjs

Preconditions: network access to `raw.githubusercontent.com`.

- **Search for Showdown URLs.** Run `python3 .claude/skills/verify-sprites/check_sprites.py --out "$RUN"`. `hotlink-urls` reads `PASS`.
- **Load the pages.** Run `node .claude/skills/verify-sprites/render_pages.cjs "$RUN/render" artifacts/jev-request-anatomy.html notes/Blog/visuals/looped-diffusion/index.html notes/Blog/visuals/team-lineage/team-lineage.html`. Each line reports `PokeAPI requests` above 0 (6, 6 and 17 on 2026-11-09), `failed requests 0` and `broken 0`.
- **Look.** Open the screenshots: every team slot shows a Pokémon.
- **Proof.** `$RUN/render/render.json` records the request counts; the screenshots show the result.

## Gotchas

- PokeAPI names files by number, so each page carries an inline lookup from its species names to PokeAPI file stems. A species missing from that lookup requests `undefined.png`, which the render reports as a failed request.
- The lookup in `team-lineage.html` is written by `sprite()` in `notes/Blog/visuals/team-lineage/build-data.py`, which reads `pokeapi_ids.json`. Keep `page.tmpl.html`, the built page and the builder in step.
- An offline render fails these pages. Report them as not verified; the byte checks say nothing about them.
- A new page should embed store sprites instead of hotlinking, so it works offline and falls under `embedded-known`.
