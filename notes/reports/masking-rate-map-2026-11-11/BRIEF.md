# Brief for a storyboard maker

You are responsible for the storyboard of ONE GitHub issue. The reader is the project's owner, who said: "I am not reading
a huge chunk of text." He understands a picture faster than prose and stops at any word nobody defined.

## What you write

Two files in your folder `notes/reports/masking-rate-map-2026-11-11/issue-N/`:

1. `spec.json`: the figure, in the format `build.py` draws. Copy the shape of `issue-293/spec.json`, the worked example.
   Write it with a python heredoc that builds a dict and calls `json.dump`, so the JSON is always valid.
2. `storyboard.md`: for each frame, three bullets (Sentence, then Expected and Surprise, or Measured), then a `## Cut`
   list of what you left out and why. Under 200 words.

Also save the issue text first, it is your source of numbers:
`gh issue view N --repo Rami-Ismael/Battle-Factory-of-Porygon-Pi --json title,body,comments --jq '[.title,.body,(.comments[].body)]|join("\n\n")' > notes/reports/masking-rate-map-2026-11-11/issue-N/source.txt`

## Rules

- 3 or 4 frames. One idea per frame. Order them as you would explain the issue aloud.
- `headline` is the one sentence the reader should leave with, at most 14 words.
- `setup` is 2 or 3 short sentences in plain words, at most 60 words, with no project word left undefined.
- `terms`: at most 5 rows, each a word the figure uses and its meaning in at most 20 words.
- A frame about something not yet run has `takes`, `expected` and `surprise`. A frame about something already measured
  has `takes` and `measured`. Each of these is one sentence of at most 22 words.
- Every number you write must appear in `source.txt` or in `code/docs/masking-rate-rows.md`. Invent nothing. If you are not
  sure of a fact, leave it out.
- Frame kinds: flow, bars, lines, grid, compare, squares, checklist. Their data shapes are in the docstring at the top of
  `build.py` (read its first 35 lines). Keep labels short: a box label at most 4 words, a bar label at most 3 words.
- Row names, exactly: "Per-field, uncorrected t" (the control), "Per-field, corrected t", "Linear, species-first",
  "Linear, random order". In drawings the control is tone "plain", Linear, species-first is "c1", Linear, random order "c2".
- Words: say "masking rate", never "noise schedule" or "scheduler". Say "Stat Points" and "Stat Alignment". Say "the
  shuffled-only table". Write model and method names in full. No em dashes, no parentheses, no abbreviations.

## Facts that are true today, 2026-11-11

- The question of the map: our 84-field model differs from two papers, MD4 and LLaDA. Do the papers' settings do better?
- The model fills a team of six Pokémon as 84 fields. In training some fields are hidden and the model learns to fill them.
  The masking rate is the chance that a field is hidden.
- Per-field masking rate, ours: species are almost never hidden, other fields more often. Linear masking rate, the papers':
  every field is hidden with the same chance.
- Four rows are compared, named above. The first two share one model, the last two share another.
- Both models train on the shuffled-only table, 214,598 teams, about 6.7 hours each. The code reads that table by default.
- Every battled row is sampled at guidance 2, 4 and 8. Three battled rows make nine grids, 1.95 million battles, about 6.9 hours.
- Per-field, corrected t is battled only if a check called the t probe reaches 0.01 on the new per-field model. It read
  0.005 on today's default model and 0.008 on a model trained on the shuffled-only table.
- The main measure is the ask-versus-got error: how far the win rate the teams get is from the win rate asked for. Lower is better.
- Ticket states: Departures closed, Corrected t closed, Pre-register closed, the code ticket is next, then train, then
  sample and battle, then results and diagnosis.

## Finish

Run `python3 notes/reports/masking-rate-map-2026-11-11/build.py issue-N check` from the repo root until it prints
`issue-N: OK`. Fix every number it lists as unknown, or remove it. Edit only files inside your own folder.
Final message, under 150 words: your headline, each frame's sentence, and anything you were unsure about.
