# Shared sprite folder

`notes/Blog/visuals/search-loop/sprites/` holds one 96×96 PNG per species, named by Showdown id. Some pages load these files directly by relative path; every build script that embeds sprites reads them from here. If this folder holds the wrong art, every page built afterwards inherits it.

## Sub-features

- `store-records`: each PNG has a record in `sources.json` whose `source` is a PokeAPI URL and whose `sha256` matches the file.
- `store-live`: each PNG is byte-identical to what that PokeAPI URL serves today.
- `store-mapping`: `pokeapi_ids.json` has an entry for each PNG, so `pokeapi_sprites.py` can return its URL.

## How to get to it (user POV)

- Open `notes/Blog/visuals/search-loop/index.html` and move the `Team reveal progress` slider to 6: six sprites load from `./sprites/`.
- Open `notes/Blog/visuals/teams-differ-prototype/index.html`: the team rows load from the folder on page load.
- `notes/Blog/visuals/smoothness/` and `notes/Blog/visuals/preview-game-prototype/` read the folder through their `data.js`.

## Driving it with check_sprites.py and render_pages.cjs

Preconditions: network access to `raw.githubusercontent.com` for `store-live`.

- **Check the folder.** Run `python3 .claude/skills/verify-sprites/check_sprites.py --out "$RUN"`. The three `store-*` lines read `PASS`.
- **Load it in a page.** Run `node .claude/skills/verify-sprites/render_pages.cjs "$RUN/render" notes/Blog/visuals/search-loop/index.html notes/Blog/visuals/teams-differ-prototype/index.html`. Each line reports `loaded as files` above 0 and `broken 0`.
- **Proof.** `$RUN/report.json` and the four screenshots in `$RUN/render/`, with the six sprites visible in the search-loop team cards.

## Gotchas

- `tools/sync.py` copies the Obsidian vault into `notes/`. While the vault copy of this folder still holds Showdown art, a sync run with `--force` or `--allow` puts it back. Run this skill after any sync; `store-live` and `store-records` go red.
- Add a species with `python3 notes/Blog/visuals/search-loop/sprites/pokeapi_sprites.py --rebuild <showdown-id>`, then fetch with `notes/Blog/visuals/search-loop/model/fetch_sprites.py`. A hand-copied PNG fails `store-records`.
- Cosmetic forms (Vivillon patterns, Alcremie flavours) have a PokeAPI file stem such as `666-fancy`, not a number.
- `store-live` fails if PokeAPI redraws a sprite upstream. That failure names the file and is fixed by refetching, not by editing the record.
- The search-loop page shows no sprite until the slider moves; `PAGE_DRIVES` in `render_pages.cjs` moves it.
