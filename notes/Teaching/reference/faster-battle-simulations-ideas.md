# How to run many more VGC battles

Research and code inspection: 13 October 2026.

**Experimental follow-up:** the proposals below are preserved as the original hypothesis list. Subsequent implementations and measurements are in the [decision report](../../battle_speed_lab/DECISIONS.md), [full results](../../battle_speed_lab/RESULTS.md), and [usable evaluator instructions](../../battle_speed_lab/README.md). Several initial bets, including shorter batching deadlines, did not improve measured throughput.

**My first bet is to tune the inference batcher, keep workers alive, and balance work across independent simulator processes.** Your project already uses multiple Python processes and Showdown servers. “Python only uses one core” does not describe the whole system.

These are proposals grounded in the inspected code, not measured speedups. I inspected the vault pipeline, its linked pilot repository, and its pinned VGC-Bench runtime. I did not change battle execution or run a new end-to-end benchmark. The more unusual proposals below are experimental combinations worth testing; I am not claiming they are new to the research literature.

## Can several battles run on one core?

Yes, several can be **in progress** on one core. While battle A waits for a policy action or a message, battle B can proceed. But if they all need CPU calculation, they share the same core's capacity. Adding tasks does not create more compute.

For illustration, if a battle requires 10 milliseconds of actual CPU work, one core's ideal ceiling is 100 battles/second. If running it alone also incurs 40 milliseconds of waiting, serial execution only reaches 20/second. Overlapping battles can recover some of that gap; it cannot exceed the CPU ceiling without reducing work. These numbers are illustrative, not measurements of your system.

| Technique | What it helps | What it does not do |
|---|---|---|
| Async tasks on one core | Overlap waiting and collect policy requests into batches | Execute CPU-bound Python tasks simultaneously |
| Multiple processes | Run independent Python interpreters and simulations across cores | Make one shared, saturated simulator faster |
| Native numerical libraries | Execute tensor operations outside ordinary Python bytecode; may use their own threads | Automatically choose the right thread count |
| Free-threaded Python | Potentially parallelize Python threads in a compatible build | Automatically accelerate your existing Node/PyTorch system |
| Less work per battle | Improve throughput even on a single core | Preserve correctness unless the removed work is actually unnecessary |

The ordinary CPython GIL restricts simultaneous Python bytecode within one interpreter, not all processes on your computer. Async CPU work still blocks its event-loop thread. See [Python threading](https://docs.python.org/3.14/library/threading.html), [multiprocessing](https://docs.python.org/3/library/multiprocessing.html), and [asyncio development guidance](https://docs.python.org/3/library/asyncio-dev.html).

Your computer reports an Apple M4 Pro, 12 physical/logical CPUs, and 24 GiB RAM. That does not imply that 12 Python workers plus 12 Node processes plus numerical-library threads is optimal. They compete for the same resources.

## What the code actually does

There are two different execution paths. Improvements to one do not automatically improve the other.

| Evidence | Implication |
|---|---|
| [Mask Lab driver](../../team_completion_pipeline/mask_lab_battle.py) defaults to three servers and submits batches of 24 teams | There is already a multicore design; server count is a tunable parameter |
| [Pilot matchup driver](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/matchup_db.py:284) launches Python subprocesses over available servers, with concurrency defaulting to 50 | More async concurrency and more worker processes are different controls |
| The same driver sorts jobs by schedule length and assigns them round-robin | Allocation uses battle counts, not observed battle duration, and cannot rebalance a running shard |
| [Pair worker](/tmp/vgc-pilot/src/pair_shard.py:71) loads the policy once per subprocess, then calls `asyncio.run` for every team job | New submitted batches reload the policy; each team job recreates event-loop/player setup |
| [Pinned batch policy](/Users/ramiismael/.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/src/policy_player.py:580) sleeps 0.005 seconds when its queue empties while gathering a batch | A measurable tradeoff between batch size and action latency already exists |
| Each `BatchPolicyPlayer` owns its own request queue, while both players receive the policy object | Requests from the two sides are not automatically combined into one inference batch |
| [Completion pipeline](../../team_completion_pipeline/pipeline.py) invokes its JSON adapter through `subprocess.run` | Its bundled direct-simulation route can repeatedly pay startup costs; it is separate from the BC server route |
| The matchup database already evaluates only missing cells | Caching is already implemented; improvement means verifying and extending it, not suggesting it from scratch |

The old [pool comment](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/pool.py:1) records 19.4 battles/second for four workers sharing one server, 38.3 for four workers with separate servers, and 63.2 for seven workers at concurrency 50. This is historical documentation without a freshly verified benchmark here. It suggests investigating simulator contention, but neither proves today's bottleneck nor guarantees those rates on the current format and policy. Showdown can have subprocess machinery; inspect the full process tree rather than equating every server with exactly one CPU consumer.

## The practical experiments I would run first

### 1. Replace repeated batch waiting with a deadline

The current batch loop waits 5 ms on an empty queue. If another request arrives, it resets its `just_slept` flag, so another empty queue can cause another wait. This is not a strict 5 ms deadline from the first request.

Test immediate draining, 0.5 ms, 1 ms, 2 ms, and the current behavior. An alternative is: start a timer when the first request arrives; flush when the batch is full **or** that deadline expires. Record actual queue age because an event loop cannot promise precise sub-millisecond scheduling.

On CPU, a smaller batch with less waiting might win. On an accelerator, a larger batch might win. Do not assume zero waiting is best: it can produce many inefficient tiny forward passes. Measure completed battles/second, realized batch size, inference time, and p95 request wait together.

### 2. Keep a warm worker service

Keep the policy, event loop, and simulator connection alive across submitted work. Send small jobs to an already running worker instead of starting a new Python process for each shard of each batch.

Measure startup-to-first-action time separately from steady-state throughput. Persistent workers matter most when job batches are short. They need explicit cleanup of completed battles and pending tasks; otherwise memory grows indefinitely.

### 3. Tune the whole machine, not just the Python count

Start with 1, 2, 4, and 6 independent simulator/worker groups. For the best two counts, test 8, 24, and 50 in-flight battles per worker. Expand only if throughput is still rising. Keep other experiments off the same servers during the comparison.

Start numerical inference with one CPU thread per worker, then compare two. Excess numerical threads can compete with other Python and Node workers. Set thread configuration before inference starts; [PyTorch documents this timing requirement](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html) and discusses [thread oversubscription](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html). This is a test configuration, not a claim that one thread always wins.

### 4. Give idle workers more work immediately

Replace static whole-run shards with a shared queue of chunks. A worker finishing short battles should pick up the next chunk while another finishes a stall matchup.

First try assigning each incoming job to the worker with the smallest estimated outstanding work. Then test smaller chunks if long jobs still dominate. Very small chunks can lose the benefit of batching and increase bookkeeping. Stable battle IDs must survive reassignment and retries.

### 5. Share an inference queue across both sides

Because both sides use the same fixed policy, first combine their ready observations within one worker. Preserve player perspective, action masks, and request-to-response IDs. This is a smaller experiment than a machine-wide inference server.

If it helps, test a shared inference service across workers. That may increase useful batch sizes and reduce model duplication, but adds transport overhead and a possible central bottleneck. Test CPU first; benchmark Apple GPU inference only after measuring actual model size and batch sizes.

### 6. Profile observation construction, not only the neural network

The batcher embeds each battle and builds its action mask before queueing inference. If this dominates Python CPU, faster matrix multiplication will have little effect.

Cache immutable species/move features; reuse buffers; separate static team features from changing HP, boosts, weather, and field state. Measure allocation and parsing costs before optimizing. Cache an observation component only when its dependencies are explicit.

## More adventurous ideas

| Idea | Why it could help | First experiment and main catch |
|---|---|---|
| **A battle engine without the server** | Remove server routing, WebSocket traffic, and some protocol overhead | Run a small sample through direct Showdown `BattleStream`. The difficult part is preserving the exact observations, masks, preview logic, and policy semantics currently provided by poke-env |
| **A policy living beside the simulator** | Avoid Python↔Node round trips for every decision | Export the frozen policy and reproduce preprocessing in the simulator process. Compare masks, action probabilities, and seeded traces. Numerical or preprocessing differences can change outcomes |
| **A battle-opening library** | Repeated matchups may revisit identical opening states or decisions | First measure exact-state hit rates. Only reuse a prefix if full simulator state, policy-visible history, policy state, and RNG state match. A similar-looking board is insufficient |
| **A cabinet of cached policy outputs** | Repeated observations could avoid repeated inference | Cache action distributions keyed by the complete policy input and model version, then sample afresh. Caching the sampled action would alter a stochastic policy |
| **A simulator specialized to the fixed opponent pool** | A finite format/team pool may exercise a small subset of mechanics | Record mechanic coverage, implement a small fast subset, and route unsupported battles to Showdown. Team mutations, abilities, and rare interactions make exactness difficult |
| **Thousands of battles in arrays** | Vectorize the same operation over many battle states, potentially on a GPU | Start with a deliberately restricted mechanic set. Pokémon's branching rules, variable battle lengths, and event ordering make this a major simulator research project |
| **A cheaper apprentice policy** | A smaller policy could make decisions faster | Distill the BC policy; use the apprentice for screening and the original for finalist evaluation. This changes the evaluator and may change which teams are favored |
| **Let a cheap predictor reject obviously weak teams** | Spend actual battles on plausible candidates | Maintain random audits and uncertainty-driven exploration. Keep surrogate predictions separate from battle results; otherwise the search can exploit predictor errors |
| **Allocate battles like a tournament** | Weak candidates need fewer trials than close finalists | Use adaptive racing with a minimum coverage budget. Use sequentially valid uncertainty methods or separate final evaluation; ordinary fixed-sample intervals do not automatically remain valid under repeated stopping |
| **Make rare opponents expensive and common ones cheap** | Reallocate trials toward matchups that most affect ranking uncertainty | Use stratified allocation while preserving the original opponent weights in scoring. Oversampling hard opponents without reweighting changes the objective |
| **Paired randomness as a compute multiplier** | Better comparisons may need fewer battles without speeding up any battle | Pair opponent schedules and per-battle RNG streams across candidates, then measure variance of score differences. Correlation need not be positive; benefit is empirical |
| **Simulate only the disagreement** | Related candidate teams may share work before they differ | Reuse exact prefixes only until any state, policy input, action distribution, or RNG consumption diverges. EVs/items can affect decisions before their first obvious battle effect, so naive reuse is unsafe |
| **Several machines stealing work from one queue** | Scale independent battles beyond one laptop | Use immutable battle IDs, versioned runtimes, leases, and idempotent writes. Network one job/result per battle chunk, not every turn. This is more total compute, not better single-core efficiency |

Showdown provides a [direct simulator API](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md). A persistent multi-battle wrapper is an implementation project, not simply a command-line flag. [Node worker threads](https://nodejs.org/api/worker_threads.html) can parallelize CPU-bound JavaScript; use a pool rather than creating a thread per battle. Switching to an extracted simulator package alone is not evidence of a speedup.

My favorite ambitious combination is **persistent direct simulator workers + one shared, deadline-based inference service + a dynamic battle queue**. It addresses startup, transport, inference utilization, and uneven work, but each component should earn its complexity in a separate benchmark.

## Keep the speed comparison scientifically useful

Use the same pinned format, legal teams, policy checkpoint, opponent distribution, and fixed workload for every comparison. Include fast, typical, and long battles. Report both cold-start and warm performance.

Record completed and correctly attributed battles/second, turns/second, CPU time per turn, inference/embedding/queue timings, per-worker idle time, p95 completion time, peak memory, errors, timeouts, and unattributed results. A faster error path is not a faster evaluator. Test finalists in repeated alternating baseline/candidate runs rather than one long run of each; report the distribution, not the best run.

The inspected workers reseed Python, NumPy, and Torch at the job level. That alone does not demonstrate independent, reproducible simulator randomness per battle. Concurrency can change which request consumes the next global random draw. Introduce explicit battle IDs and separate simulator/policy random streams before interpreting trace differences as a correctness failure or claiming common-random-number pairing.

Caches must distinguish format/simulator versions, policy weights and preprocessing, relevant team ordering/side assignment, and randomness where storing individual trials. Reusing an existing trial saves work; it does not create another independent observation. The current matchup cache already has policy/cell accounting, so inspect its identities rather than replacing it blindly.

Avoid ending long battles early and calling them losses or draws merely to raise throughput. If a learned cutoff is used for screening, label it as an approximate result and evaluate finalists with completed reference battles.

## A concrete sequence

1. Add stage timing and gather a fixed-workload baseline. Determine whether CPU is spent in Node simulation, Python state processing, tensor inference, or whether workers are mostly waiting.
2. Test batch deadlines and thread counts. These are relatively small changes with a direct code-based motivation.
3. Sweep worker/server counts and concurrency, using the best batcher configuration. Stop increasing concurrency when throughput plateaus or memory/errors worsen.
4. Keep workers warm and test dynamic scheduling. Retain only changes that improve completed, correct throughput under the fixed workload.
5. Test a shared inference queue within one worker; then consider cross-worker sharing.
6. Prototype direct `BattleStream` only if transport/server handling remains material. Preserve the existing simulator as the reference implementation.
7. Separately test adaptive allocation and surrogate screening. Evaluate these by team quality at a fixed wall-clock budget, not by raw battles/second.

There are two useful success criteria: **more exact battles per second**, and **better counter-teams found per hour**. An engine optimization can help the first; a smarter allocation strategy can help the second while running fewer battles. Track both so a wacky idea gets credit for the thing it actually improves.

Further primary-source notes: [Python, Node, and Showdown concurrency](battle-throughput-primary-sources.md).
