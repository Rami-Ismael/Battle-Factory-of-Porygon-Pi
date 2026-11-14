# Preflight

Before a checkpoint's full grid is battled, preflight checks it is fit to: generated teams survive a round trip through the paste format, they are not copies of training teams, the ask changes what is sampled, and known teams score where they should through the real scorer.

## Sub-features

- `pf-round-trip` decodes 20 sampled teams to pastes and back.
- `pf-copies` and `pf-separation` count sampled teams identical to training teams.
- `pf-condition` checks that more than half of teams change between ask 0.1 and 0.7 under one seed.
- `pf-controls` battles 6 top-50 teams (win between 0.35 and 0.65) and 6 uniform legal teams (win under 0.12).

## How to get to it (user POV)

- `AG84_SMOKE=1 AG84_MODEL=<model> MATCHUP_DB=<private db> python ask_grid84.py preflight`, from `code/src`.

## Driving it with verify.sh

Preconditions:

- The baseline in `README.md` holds. Preflight may share a `RUN` with the smoke grid; run it after the smoke drive ends.

- **Run it.** `nohup verify.sh drive preflight "$RUN" > "$RUN/drive.out" 2>&1 &`. `RUN/preflight.log` prints the checks as JSON, then 20 sample pastes, then `AG84_PREFLIGHT_DONE PASS` or `FAIL`.
- **Read the verdict.** `drive.out` repeats the `AG84_PREFLIGHT_DONE` line. A FAIL names the failing check in the JSON's `passed` fields.
- **Read the teams.** Look at the 20 pastes in `preflight.log`: six Pokémon each, moves and items that belong to them.

## Gotchas

- The controls battle 12 teams × 49 opponents = 588 battles, three times the smoke grid.
- `controls()` refuses to run without `MATCHUP_DB`, and it rewrites the `top50` set in that database.
- `/Users/ramiismael/vgc-data/teams/hps_reg_mb_100k.jsonl` supplies the uniform legal teams; preflight fails without it.
