# Smoke grid

One checkpoint samples teams at ask 12/24, the teams battle the 49 distinct top-50 meta teams, and the report gives their win rate and the ask-versus-got error. This is the full path every ask-grid result rides on, at the smallest size.

## Sub-features

- `smoke-gen` draws until 4 teams pass Showdown and are not held-out teams, at most 64 tries.
- `smoke-battle` plays each team once against each of the 49 opponents, slots shuffled, behaviour-cloning policy on both sides.
- `smoke-report` writes the ask table and the ask-versus-got error.

## How to get to it (user POV)

- `AG84_SMOKE=1 AG84_MODEL=<model> MATCHUP_DB=<private db> python ask_grid84.py gen`, then `battle`, then `report`, from `code/src`.

## Driving it with verify.sh

Preconditions:

- The baseline in `README.md` holds, and no `stat_points_84_<model>_smoke.json` sits in `~/vgc-data/results`.

- **Run the journey.** `nohup verify.sh drive smoke "$RUN" > "$RUN/drive.out" 2>&1 &`. Each step writes `RUN/<step>.log`; `gen.log` shows `kept 4`, `battle.log` shows `win`, `report.log` ends with `AG84_REPORT_DONE`.
- **Read the checks.** `drive.out` prints one PASS or FAIL per check: one cell at 12/24, 4 teams, every team scored, 196 battles in the cell, the private database grown by 196 battles, the error reported. It ends with `SMOKE PASS` or `SMOKE FAIL`.
- **Clean up.** `verify.sh cleanup "$RUN"`; `RUN/smoke_results.json` and the logs remain.

## Gotchas

- Smoke mode writes one fixed file, `~/vgc-data/results/stat_points_84_<model>_smoke.json`, so two smoke runs on one model collide. `verify.sh` refuses to start when the file exists and moves it into `RUN` at cleanup.
- `gen` skips cells already in that file, so a stale file makes a run "pass" on old teams. Doctor fails on a stale file for this reason.
- 4 teams at one ask give a noisy win rate and an error from one ask. The smoke proves the pipeline runs, not where the model sits on the line.
