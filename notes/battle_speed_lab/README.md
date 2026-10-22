# Battle speed lab

Isolated experiments for the 13 October 2026 throughput investigation. Production policies, runtimes, servers, and matchup databases are inputs only.

Start with the [decision report](DECISIONS.md) for recommendations and the [full results](RESULTS.md) for measurements, uncertainty, and prototype limitations.

Run with `/tmp/vgc-pilot/.venv/bin/python battle_speed_lab/run.py --help`.

The controller creates a private Showdown runtime, launches only its own servers on unused loopback ports, exports a fixed legal workload from a read-only database, and records append-only trial results. The runtime copy uses the pinned compiled Showdown code with separate configuration/logs and shared read-only dependencies. It stops only process groups it created.

Timing trials alternate baseline and treatment in randomized ABBA/BAAB blocks. Individual trials are not independent replicates: analysis resamples whole blocks. CPU load and memory snapshots accompany timings. Background load is measured, not assumed absent. Live-server stochastic battles are not bitwise paired; turns/second is reported alongside battles/second, and changes to randomness or policy semantics must be evaluated separately.

`scope.json` retains the evidence and limitations for all 19 ideas. Experimental coverage includes partial prototypes; it does not claim full native policy colocation, a general array-based VGC engine, or measured multi-host scaling.

## Reusable fast evaluator

`evaluate_pool.py` packages the validated sparse encoder, actor-only inference, shared within-worker batching and one numerical thread into a persistent worker/server pool. It evaluates candidates from the existing legal top50 set against the other teams in that set, preserving a fixed candidate within every chunk. The default is four worker/server groups with 24 in-flight battles each. Its dedicated two-worker smoke check completed 16 battles across four chunks, exercising reuse of both workers; the separate performance comparisons are in RESULTS.md.

Run from this project directory with the existing pilot environment:

```sh
/tmp/vgc-pilot/.venv/bin/python battle_speed_lab/evaluate_pool.py --list-candidates

caffeinate -i /tmp/vgc-pilot/.venv/bin/python battle_speed_lab/evaluate_pool.py \
  --candidate-ids 31 77 --battles-per-candidate 96 --workers 4
```

For throughput-focused evaluation, change `--workers 4` to `--workers 6`. Keep concurrency at its default **24**: the final 48,000-battle optimized-pool comparison found no established benefit from 50 and slower throughput at 8. Six groups delivered the highest observed throughput in the worker-count investigation; four uses fewer resources. Absolute rates vary with workload and background activity.

Each run writes an immutable workload snapshot, policy/source fingerprints, append-only completed chunk records and a final candidate summary under `battle_speed_lab/runs/evaluation-...`. It reads the source matchup database without modifying it. The original BC checkpoint and pinned Showdown dependencies must still be available at the paths recorded in this lab. `--run-id` chooses a fresh output directory; existing directories are rejected.

Candidate summaries report win scores over the requested opponent schedule. They are not high-confidence team rankings. Simulator randomness is fresh; job-level policy seeds do not promise reproducible full traces. Interrupted runs preserve completed chunks and an incomplete-run record. This runner is an integration artifact, separate from the paired performance harness and its claims.
