# Review: Nine tries, one π

Output: `../build-video/logo-nine-tries.mp4`, 1080 × 1350, 30 fps, 31 s, H.264 + AAC. Storyline A from `STORYLINES.md`, picked by the owner on 2026-11-08; built to `SPEC.md`.

## Creative calls

- Every design on screen is a real file: the nine construction boards (centre 1440 px crop), `icon-512.png`, `favicon-32.png`, `favicon-16.png` and `mark-color.svg` from `brand/logo/`.
- Reasons appear only where the record has one: 04 "reads 77", 04c and 04d "read as bunting". 01 and 02 only dim; 03, 05 and 06 share one line ("Facets beat tiles, hexagon and bird").
- Registry pieces: `parallax-zoom` drives 04b taking the frame; the strike line follows the `strikethrough-replace` recipe.
- Sound: seven bundled effects (Pixabay licence, `assets/sfx/CREDITS.md`), one per visible action, takes rotated. No music: HeyGen was not signed in.
- Storylines not built: B (an endless zoom from a browser tab into the build script) and C (giant kinetic type, match cuts on the π).

## Measured

- `check`: lint, runtime, layout, motion, contrast pass. Kept warnings: `timeline_track_too_dense` (titles), `duplicate_media_discovery_risk` (the 16 px favicon in the size row and in the tab), `line_unread` on the end card's tab title (it carries over from the previous 2.7 s shot, so it is read before the cut).
- Sound, measured not heard: −14.6 LUFS integrated, true peak −6.5 dB, no clipping; bass hit peaks at 27.92 s as the mark lands.
- Fixed P0s: the opaque 512 px icon showed as a white square larger than the card; the browser tab was too small to read; the tab title sat too high; a 4 s static opening.

## Not checked

Playback at full speed on a phone; the sound by ear.

## Weakest shot

The size row (20.8 to 24.3 s): the three icons sit small in a lot of empty ground. With more time, push in on the 16 px favicon so its pixels fill a third of the frame.
