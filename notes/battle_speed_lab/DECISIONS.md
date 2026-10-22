# More VGC battles: what the experiments support

**Main-pipeline follow-up:** the BC battle pipeline now uses six persistent workers with concurrency 24. Its separate integration benchmark measured **1.73× whole-stage throughput** and **3.28× later-batch throughput**, with 33 passing tests. See the [integration report](../team_completion_pipeline/FAST_EVALUATOR_INTEGRATION.md) for commands, cache behavior, and measured uncertainty. The results below describe the earlier isolated lab investigation.

The biggest useful change was reducing Python observation-building work, then running persistent worker/server groups across cores. The original batching-deadline idea did not help in the measured workload.

Several battles can be in progress on one core, overlapping waits and batching policy requests. They still share that core's computation capacity. Independent Python processes can use different cores; your existing system already does this. More processes and more simultaneous battles per process are separate controls.

## Use now

| Change | Measured result | Evidence |
|---|---|---|
| Fast observation encoder and inference stack | **2.74×** the unchanged upstream path; 95% block-bootstrap interval 2.66–2.81× | [Eight-block confirmation](runs/bc-confirm-001/summary.json) |
| Four persistent optimized worker/server groups | **3.04×** one optimized group; 2.87–3.18× | [Pool comparison](runs/optimized-pool-001/analysis-v2.json) |
| Six persistent optimized worker/server groups | **3.61×** one optimized group; 3.29–3.96×; observed aggregate 75.2 battles/sec | [Pool comparison](runs/optimized-pool-001/analysis-v2.json) |

These ratios have different baselines. Do not multiply them or treat the observed rates as guarantees under another workload or background load.

The [ready-to-run evaluator](evaluate_pool.py) packages the fast encoder, actor-only inference, within-worker batching, one numerical thread per worker, and persistent private servers. Use **six groups with 24 in-flight battles each** when prioritizing throughput on this laptop; four is the lower-resource default. [Commands and output details](README.md) include both options. It reads the existing team database and writes separate result artifacts. Production files were not replaced.

The final **48,000-battle optimized concurrency sweep** compared 8 and 50 with 24 at both worker counts. Eight was slower: 0.863× with four workers and 0.837× with six. Fifty did not establish a gain: 0.942× and 0.995×, with intervals spanning 1. Keep **24**. This supersedes the initial single-worker concurrency result when configuring the optimized pool. [Full paired evidence](runs/optimized-concurrency-001/analysis-v2.json).

## Interesting experiments that did not earn a default setting

| Idea | Result and decision |
|---|---|
| Shorter batch deadlines | 0, 0.5, 1 and 2 ms all lost throughput in their tested settings. Keep the original batching behavior. |
| Shared inference service across processes | Slower at two and four workers. Keep local inference for this CPU setup. |
| Exact policy-output cache | Zero hits in 23,930 requests in the isolated comparison; no established gain. |
| Direct simulator adapter | 0.95× whole-job throughput, interval spanning 1. Current implementation does not establish a benefit. |
| Exported actor beside Showdown | Numerical/action checks and trace checks passed, but partial colocation measured 0.94×, interval spanning 1. Python preprocessing and sampling remain; full round-trip removal is untested. |
| Dynamic work queue | No clear benefit for these similarly sized chunks. Durable local HTTP queue cost about 4% versus static assignment; multi-machine scaling was not measured. |
| Opening/checkpoint library | 1.21× for repeated identical seeded continuations. All 48 fresh seeds missed; repeated continuations are not new independent evidence. |
| Cross-team prefix reuse | No exact state matches among 64 nature mutations. Similar visible boards frequently hid different future traces. |
| Tiny apprentice policy | Only 22.6% joint greedy agreement; longer battles erased faster decisions. Reject this first model. |
| Common random seeds | Helped 12 of 28 candidate comparisons and hurt the others. No universal variance reduction. |
| Specialized array engine | 2.36× over the same restricted scalar rules, validated against Showdown. None of the 49 real teams qualify. This is a research prototype, not a general VGC speedup. |

Adaptive racing, opponent allocation, and surrogate screening completed both finite-corpus experiments and a live equal-time comparison. Across 60 fifteen-second trials, all three quality-difference intervals included zero. None has demonstrated better team selection for the same time budget.

## How much to trust the numbers

Timing treatments alternate in randomized ABBA/BAAB blocks; uncertainty resamples whole blocks. The harness records external CPU activity and excludes entire blocks with detected clock discontinuities. Timing experiments run sequentially. This reduces background-load bias but cannot eliminate all interference or guarantee performance on other teams.

Correctness checks include exact observation equality and seeded trace equality where the path supports it. Live-server experiments are stochastic and are not exactly paired by simulator RNG. Changes to policy or supported mechanics have their own explicit limitations. Held-out team rankings remain uncertain; faster sampling is not evidence that a chosen team is truly strongest.

The [evidence audit](EVIDENCE-AUDIT.json) passed accounting and archived-hash checks for **21 timing runs**, plus the live allocation experiment. An early component screen lacks a complete source archive; later confirmation runs preserve source snapshots. The [full report](RESULTS.md) links raw trials, validation, negative results, and scope limitations. These experiments are creative engineering trials, not claims of research novelty.
