# Embedded and copied sprites

Most figures and blog pages carry their sprites inside the file as base64 PNGs, so they open with no network. About 90 tracked HTML, JS, JSON, SVG and Excalidraw files do this, and four folders hold plain copies of store PNGs. The reader sees whatever bytes were embedded when the page was last built.

## Sub-features

- `embedded-stale`: no tracked file embeds, and no tracked PNG is a copy of, a Showdown sprite the repo used to hold.
- `embedded-known`: every embedded 96×96 image is byte-identical to a current store sprite.
- `logo-porygon`: `brand/logo/build-video/sprites/porygon.png`, embedded in the three logo pages, equals PokeAPI's Porygon.

## How to get to it (user POV)

- Open a blog page: `blog/mask-lab/index.html`, `blog/kingambit-finder/index.html`, `blog/best-team-against-the-meta/index.html`, `blog/meta-vs-random-teams/index.html`.
- Open a figure under `notes/Blog/visuals/`, for example `team-map-path/team-map-path.html` or `diversity-methods/diversity-methods.html`.
- Copied PNGs sit in `notes/Blog/visuals/ablation-plan/sprites/`, `team-interactions/`, `variable-types/` and `notes/artifacts/heuristic-generation-lab-20261021-7ce4/assets/`.

## Driving it with check_sprites.py and render_pages.cjs

Preconditions: the page under test is tracked or staged.

- **Scan every tracked file.** Run `python3 .claude/skills/verify-sprites/check_sprites.py --out "$RUN"`. `embedded-stale` and `embedded-known` read `PASS` and the line states how many embedded images were scanned (4,419 on 2026-11-09).
- **Render the changed page.** Run `node .claude/skills/verify-sprites/render_pages.cjs "$RUN/render" <page.html>`. The line reports `embedded` above 0 and `broken 0`.
- **Look.** Open the light and dark screenshots. Sprites sit inside their slots with hard pixel edges and no visible box.
- **Proof.** `$RUN/report.json`, `$RUN/render/render.json`, and the screenshots.

## Gotchas

- `embedded-known` fails on a sprite that was resized or re-encoded before embedding, because its bytes match no store file. Embed the store file unchanged and size it in CSS.
- Mega Evolutions that had no Showdown sprite were embedded as their base form. Those pages pass both checks while still showing the base form; `barbaraclemega`, `dragalgemega`, `eelektrossmega`, `pyroarmega`, `raichumegax`, `raichumegay` and `staraptormega` now have their own store file, and a rebuilt page picks it up.
- `blog/mask-lab/rebuild.py` keeps sprites already in the page and adds missing ones only under a 4,000-byte cap. Twenty store files are larger than that.
- Item icons are embedded beside the sprites. They are smaller than 96×96 and the scan leaves them alone.
- Several build scripts read data outside the repo (`~/vgc-data`, `/tmp/vgc-pilot`). Verify the page they wrote; a missing input is a reason to report the page as not rebuilt, never to skip the scan.
