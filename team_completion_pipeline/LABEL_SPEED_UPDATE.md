# Label generation speed update — October 13, 2026

The remaining-label job now uses request chunks of up to 96 battles with six workers and 24 concurrent battles per worker. This separates the amount of scheduled work from the concurrency limit, avoiding repeated draining of tiny request groups. It retains the BC checkpoint, simulator, policy preprocessing, and scoring protocol.

Two interleaved ABBA/BAAB blocks compared chunk sizes 24 and 96 on the same 36 queued candidates and 49 opponents. All eight trials completed and attributed 1,764 battles apiece. Block speedups were **1.136×** and **1.035×**, for a geometric mean **1.085×**. This is an exploratory estimate from two blocks, not a guaranteed production speedup. Full live trajectories are stochastic; this was not a bitwise-paired replay. Startup was excluded because the production pool persists.

A separate eight-trial comparison used the same 48 candidates for six versus eight active workers, with chunks fixed at 96. Each trial completed 2,352 battles. Eight workers delivered **0.888×** and **0.995×** the six-worker throughput by block, geometric mean **0.940×**. Six workers remain the default. The inactive workers in the six-worker benchmark arm stayed loaded but received no requests.

The old run's last ten measured batches averaged only 5.7% aggregate worker idle time, making reassignment a lower-priority opportunity. Earlier validated concurrency sweeps had already found no reliable benefit from raising concurrency above 24.

Sixteen unit/database-contract tests pass, including a new test verifying that a 50-battle request stays intact while concurrency remains 24, exact battle attribution, separate cache identities for chunk-size changes, and rejection of incompatible policy/encoder/runtime changes.

The production run was paused with **14,112 completed labels preserved**, then resumed with the new schedule. Earlier rows retain their original policy identity. New results use a separate execution/cache identity, and no new label combines cells from both identities. The live chart continues to refresh every 30 seconds.

Evidence:

- `runs/label-speed-20261013/summary.json`: chunk comparison and individual timings.
- `runs/label-speed-20261013/workers-summary.json`: worker comparison and individual timings.
- The corresponding evaluator directories contain raw per-battle attribution, timings, and policy/source fingerprints.
