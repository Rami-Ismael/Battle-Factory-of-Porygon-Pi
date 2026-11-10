# Sprite verification map

The maintained list of every way a reader meets a Pokémon sprite in this repo, and the check that proves each one shows PokeAPI art. Read this index, then the matching feature file.

| Feature file | What the reader sees | Check ids |
|---|---|---|
| [sprite-store.md](sprite-store.md) | Pages that load PNG files from the shared sprite folder | `store-records`, `store-live`, `store-mapping` |
| [embedded-sprites.md](embedded-sprites.md) | Pages, data files and drawings with sprites embedded as base64, and folders holding copied PNGs | `embedded-stale`, `embedded-known`, `logo-porygon` |
| [hotlinked-pages.md](hotlinked-pages.md) | Pages that load sprites from PokeAPI's GitHub URL when opened | `hotlink-urls`, render |

## Baseline

- Run from the repo root, on the commit or working tree being verified.
- Run the doctor in `../SKILL.md` first.
- The checks read tracked files (`git ls-files`). A new page counts once it is `git add`ed.

## Driving conventions

- Byte checks run through `check_sprites.py`; anything a browser must draw runs through `render_pages.cjs`.
- A byte check proves where a sprite came from. Only a render proves the page draws it. Report the two separately.
- Drive a page by accessible name (`getByLabel`, `getByRole`), as `PAGE_DRIVES` does.

## Proof and skip reporting

- Byte proof is `report.json`: the commit, each check id, its failures.
- Render proof is `render.json` plus the light and dark screenshot of each page, looked at.
- A check that printed `SKIPPED` is reported as skipped, with its id. An offline run never counts as a pass for `store-live`.
- A page left out of the render is reported as checked by bytes only.

## Feature entry contract

Each feature file starts with an H1 and one paragraph on what the reader sees, then four H2 sections in this order: `Sub-features`, `How to get to it (user POV)`, `Driving it with check_sprites.py and render_pages.cjs`, `Gotchas`.

## Out of scope

Item icons (`notes/Blog/visuals/item-sprites/`, from smogon/sprites), trainer sprites (`brand/logo/build-video/sprites/`, from Showdown), the brand pixel art in `brand/logo/sprite/`, and pictures with sprites baked in (preview PNGs, GIFs, the logo videos).
