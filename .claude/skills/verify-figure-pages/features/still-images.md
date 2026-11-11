# Still images

A reader sees the figure as an image in a GitHub issue, in the light or the dark theme. The image must be what the storyboard page draws today, and the copy GitHub serves must be that same file.

## Sub-features

- `still-fresh`: rendering the storyboard again reproduces the committed PNGs.
- `still-published`: the image an issue shows has the same bytes as the committed PNG.

## How to get to it (user POV)

- Open https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/258 in GitHub's light theme: the second figure is `storyboard-light.png`.
- Open it in the dark theme: the same figure is `storyboard-dark.png`, served through a `<picture>` tag.

## Driving it with render_storyboard.py and check_page.py

Preconditions:

- `storyboard.html` has been built.
- `still-published` needs network access and the `gh` CLI signed in.

- **Render again.** Run `uv run artifacts/wrong-mega-stone-validation-loss/render_storyboard.py`. It prints one line per theme with `760 x` the board's height, `broken images 0` and `page errors []`.
- **Compare with the commit.** Run `git status --porcelain -- artifacts/wrong-mega-stone-validation-loss/storyboard-light.png artifacts/wrong-mega-stone-validation-loss/storyboard-dark.png`. No output is `PASS still-fresh`.
- **Find the published copies.** Run `gh issue view 258 --repo Rami-Ismael/Battle-Factory-of-Porygon-Pi --json body --jq .body | grep -oE 'https://gist[^"]+02_one_wrong_mega_stone[^"]*\.png' | sort -u`. It prints two addresses: one ending `02_one_wrong_mega_stone.png`, one ending `02_one_wrong_mega_stone_dark.png`.
- **Compare with the published copies.** Run `uv run .claude/skills/verify-figure-pages/check_page.py same "<light address>=artifacts/wrong-mega-stone-validation-loss/storyboard-light.png" "<dark address>=artifacts/wrong-mega-stone-validation-loss/storyboard-dark.png"`. Two `PASS still-published` lines, then `RESULT pass`.
- **Look.** Open both PNGs. Each shows six frames ending with the eight teams and three bars.

## Gotchas

- A Chrome update can change a few pixels of text, so `still-fresh` fails with nothing wrong. Open the old and new image side by side. If they read the same, restore the committed PNGs with `git checkout -- <the two PNGs>` and report `still-fresh` as changed by the browser, not by the page.
- A `still-fresh` failure after a real page change means the published image is now out of date. Publishing a new one pushes to the figure gist and edits the issue, so it waits for the owner's ask.
- The issue pins each image to a gist commit in its address. A new push changes nothing a reader sees until the issue body carries the new address.
- `check_page.py same` splits each argument at its last `=`. Quote the whole pair.
