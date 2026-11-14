---
name: verify-ask-grid84
description: "Prove the 84-field ask grid works end to end: sample teams at an ask, battle them on the laptop in a private database, report the ask-versus-got error, and keep the evidence. Use before reporting a change to ask_grid84, diffusion84, matchup_db, ask_line or the error analyses as working, or when asked to verify the ask grid or a checkpoint."
---

# Verify the 84-field ask grid

The surface is a command-line pipeline, `code/src/ask_grid84.py`: `gen` samples teams at each ask, `battle` scores them against the 49 distinct top-50 meta teams, `report` prints the ask-versus-got error. Every command here goes through `verify.sh` beside this file. It runs `ask_grid84.py` in smoke mode (`AG84_SMOKE=1`: ask 12/24, seed 1, 4 teams) against a private database, so the shared `~/vgc-data/matchup_regmb.sqlite` is never written.

`AG84_MODEL` picks the checkpoint, default `gen84_all_ep21`; any name in `ask_grid84.MODELS` works. Battles run on this laptop only, never on Modal.

## 1. Doctor

```
.claude/skills/verify-ask-grid84/verify.sh doctor
```

Done when it prints `DOCTOR PASS`. A `FAIL` names the missing piece: after a restart `/tmp/vgc-pilot` needs its symlinks rebuilt, and a leftover `stat_points_84_<model>_smoke.json` in `~/vgc-data/results` means another run owns the smoke slot or a run skipped cleanup. Read the `INFO` job list: battles share the CPU with any running queue, so a busy laptop makes the run slower, not wrong.

## 2. Launch

```
.claude/skills/verify-ask-grid84/verify.sh launch
```

It makes `~/vgc-data/verify/ask-grid84/<time>-<model>/` (the run folder, `RUN`), seeds a private database there with the VGCPastes pool and the top-50 set, and prints `RUN=<folder>`. Done when it prints that line; `seed.log` ends with `top50 opponents: 49`.

## 3. Drive

Pick the journey from [`features/README.md`](features/README.md) and follow its file. Run drives detached, because an app restart kills child processes:

```
nohup .claude/skills/verify-ask-grid84/verify.sh drive smoke "$RUN" > "$RUN/drive.out" 2>&1 &
```

Done when `drive.out` ends with the journey's PASS or FAIL line. A FAIL is a finding to report with its log; leave the checks as they are.

## 4. Evidence

Everything stays in `RUN`: one `<step>.log` and `<step>.exit` per `ask_grid84.py` call, `smoke_results.json` (teams, scores, report), `db_before.json` and `db_after.json` (battle counts in the private database), and `drive.out` with the checks. A proof cites the PASS lines and the numbers from `drive.out`, plus the log of any step that failed. The fast battle runner also writes its own folder under `~/vgc-data/battle-evaluations/` named with the battle step's process ID, which is in `battle.pid`.

## 5. Cleanup

```
.claude/skills/verify-ask-grid84/verify.sh cleanup "$RUN"
```

It stops only the processes this run started (by the PIDs in `RUN`), moves the shared smoke results file into `RUN`, and deletes the private database. Done when it lists `RUN` and the evidence files are still there. Run it after every attempt, failed ones too.
