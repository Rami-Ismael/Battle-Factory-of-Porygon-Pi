# Fast evaluator integrated into the main behaviour-cloning pipeline

13 October 2026. The main [Mask Lab battle entry point](mask_lab_battle.py) now defaults to **six persistent workers, each with its own private Showdown server and 24 concurrent battles**. All **33 tests passed**. The accepted benchmark measured **1.73× whole-stage throughput**, including startup, and **3.28× throughput in later saved batches**.

## Run it

From `team_completion_pipeline`, the normal command now uses the optimized evaluator:

```sh
caffeinate -i /tmp/vgc-pilot/.venv/bin/python mask_lab_battle.py
```

Add `--limit 6` for a small run. The defaults are equivalent to `--backend fast --workers 6 --concurrency 24`. The existing candidate input, private database, panel schema, held-out-team protection, and scoring conventions remain supported. The generic random-policy smoke pipeline is unchanged.

**Cache behavior:** optimized results have a separate versioned cache identity. Existing legacy results are preserved; the first fast run computes its missing cells. Subsequent identical runs reuse the fast cache, and `--per-cell` top-ups run only the deficit. This avoids mixing results from different execution configurations.

## What changed

The new [runner](fast_battle_runner.py) connects the validated fast observation encoder and actor inference to the existing `matchup_db.ensure` interface. Models and servers remain alive across saved batches. Database reads and writes stay on the caller thread; worker threads only submit simulation requests and collect results.

Chunks are assigned by scheduled battle count, with large chunks first. This fixes the imbalance caused by repeatedly assigning 24/24/1-battle chunks round-robin. All six workers handled 196 battles each in the first accepted fast trial.

Every completed battle is checked against its expected candidate and opponent. Missing, duplicated, unfinished, or misattributed results fail before the current database batch is committed. Earlier committed batches remain resumable. Shutdown closes owned processes and communication pipes. Fully cached invocations start no workers.

Each invocation preserves configuration and source fingerprints, requests, individual results, startup records, and timings. Panel metadata points to that evidence. Fingerprints include the behaviour-cloning (BC) checkpoint, relevant code, numerical packages, BC source/data, configuration, and compiled simulator. [Usage and test commands](README.md#optimized-bc-battles-for-mask-lab).

## Measured speed

The comparison used **24 real legal Mask Lab candidates against all 49 meta teams**, with two saved batches of 12 candidates per trial: **1,176 battles per trial**. It compared the previous three-worker/concurrency-50 evaluator with the integrated six-worker/concurrency-24 evaluator.

| Measurement | Legacy throughput | Integrated throughput | Paired speedup, exploratory 95% interval |
|---|---:|---:|---:|
| Whole battle stage, including startup | 22.35 battles/sec | 38.23 battles/sec | **1.73× (1.52–1.88×)** |
| Later saved batch, after the fast pool is loaded | 23.93 battles/sec | 76.51 battles/sec | **3.28× (2.79–3.60×)** |

Throughput columns use aggregate counts and elapsed time. Speedups normalize within each paired block and then take the geometric mean, so they need not equal the ratio of the aggregate columns. The later-batch comparison still includes the legacy evaluator's model reloads; it is not an isolated engine benchmark. These gains must not be multiplied by the earlier lab's speedup estimates.

There were **six attempted comparison blocks**, each ordered legacy–fast–fast–legacy or fast–legacy–legacy–fast (ABBA/BAAB). One entire block was rejected after a roughly 471-second wall-clock discontinuity, and a replacement block was collected. The final estimate uses **five valid blocks, 20 trials, and 23,520 battles**; 28,224 battles were completed across all six attempted blocks. Ordinary slow trials were retained. The two earlier interrupted development runs are preserved but excluded entirely.

The timer includes evaluator provenance/setup, model/server startup, deficit planning, simulations, database writes, score calculation, and cleanup. Candidate generation/validation and copying the fixture database are excluded. Both paths use isolated servers with the same compiled simulator configuration; the legacy runner code is unchanged.

Background CPU demand varied substantially, with median estimated external usage around four CPU cores. Interleaved comparisons and block-level resampling reduce confounding but cannot eliminate it. Live trajectories are stochastic, not trace-paired, and legacy output does not provide turn counts. This is a measured improvement for this workload and machine, not a guarantee for every workload. Very short cold runs can still be dominated by loading six policies.

## Verification

- **13 unit/database-contract tests:** defaults, compatibility flags, chunk coverage and balancing, reordered outcomes, win/loss/tie attribution, invalid/ambiguous teams, cache identity and top-ups, worker reuse, failed-worker atomicity, held-out guards, startup cleanup, and relative paths.
- **2 real integration tests:** actual six-worker BC simulations, reuse across top-ups, all six workers exercised, database counts, process/pipe closure, the main CLI, complete score panels, and a cached rerun that launches no workers.
- **16 existing pipeline regression/integration tests:** all still pass.
- **2 benchmark-analysis tests:** paired normalization under changing load, and exclusion of clock-invalid or incomplete blocks.

The final audit verified archived source hashes and identical workloads across the main and replacement runs. It reconciled every fast battle's win/loss/tie with the SQLite cells, checked all 864 logged fast chunks across attempted runs, and confirmed one consistent optimized configuration.

Evidence: [combined audited analysis](evidence/fast-integration-analysis.json), [31-test suite log](evidence/fast-integration-003/integration-tests.log), [two analysis-test results](evidence/fast-integration-003/benchmark-analysis-tests.log), [benchmark implementation](benchmark_fast_pipeline.py), [audit implementation](analyze_fast_pipeline.py). Full raw trials, per-battle logs, source snapshots, and fixture databases remain in the original local workspace under `team_completion_pipeline/runs/fast-integration-003` and `fast-integration-004`; these large runtime artifacts are not included in the PR. Re-running the full audit requires those directories.

All verification and benchmark database writes used isolated fixtures. The existing production/private matrix and current score panels were not overwritten by these tests. The integration is ready for the next normal pipeline invocation.

For the pull request, the integration was placed in the current repository's `notes/` layout and all 33 tests were rerun successfully against main ([packaged integration test log](evidence/pr-integration-tests.log)). Seven supporting harness/encoder tests also passed.
