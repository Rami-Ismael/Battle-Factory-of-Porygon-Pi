# Error comparison

Saved arms (remasking rules, temperature, nucleus) are judged by the ask-versus-got error against no remasking, with a bootstrap interval over teams and opponents. Reads saved battles only.

## Sub-features

- `err-unit` checks the error and the paired difference on worked examples.
- `err-reproduce` reruns `remask84_error_analysis.py` and matches the saved numbers.

## How to get to it (user POV)

- `python scripts/test_ask_error_compare.py`, from `code/`.
- `python remask84_error_analysis.py <output.json>`, from `code/src`.

## Driving it with verify.sh

Preconditions:

- Doctor passes. No private database is needed: the analysis opens the shared database read-only.

- **Unit check.** `cd code && /tmp/vgc-pilot/.venv/bin/python scripts/test_ask_error_compare.py`. Exit code 0 and `OK`.
- **Reproduce.** `cd code/src && nohup /tmp/vgc-pilot/.venv/bin/python remask84_error_analysis.py "$RUN/error_analysis.json" > "$RUN/error_analysis.log" 2>&1 &`. The log ends with `ERROR_ANALYSIS_DONE`; every arm's contract line shows `"missing": 0`.
- **Match.** Compare `RUN/error_analysis.json` with `~/vgc-data/results/remask84_error_analysis.json` key by key; the bootstrap is seeded, so every number matches. No remasking without 12/24 is 0.0252, matching issue #90.

## Gotchas

- A run takes about 100 minutes: each team's policy lookup scans the whole `cell` table (22.8 million rows on 2026-11-06), slower while another job writes to it.
- Writing the output over `~/vgc-data/results/remask84_error_analysis.json` destroys the reference; always pass a path in `RUN`.
