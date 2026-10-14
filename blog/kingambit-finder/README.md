# Kingambit-free team finder

Search the 549 Regulation M-B teams that do not bring Kingambit, then open a matchup matrix of the top results against the 49 top-placing teams. Served at `https://rami-ismael.github.io/Battle-Factory-of-Porygon-Pi/blog/kingambit-finder/`.

- `index.html` is the page: one self-contained file, data and sprites included. Open it directly or serve the folder.
- `template.html` is the page source; `build.py` fills in the data and writes `index.html`.
- Rebuild with `python3 build.py` from this folder. It reads the matchup database at `~/vgc-data/matchup_regmb.sqlite` and the counter grid in the code folder, so it can only run on the author's laptop.
- Add `?clean` to the address to hide the bottom view switcher for a screenshot.

Win rate is wins ÷ battles against the top-placing teams, with the behaviour-cloning policy on both sides. Most matrix cells are about 4 battles, so read the row totals, not single cells. Tests: `python3 -m unittest discover -s tests`.
