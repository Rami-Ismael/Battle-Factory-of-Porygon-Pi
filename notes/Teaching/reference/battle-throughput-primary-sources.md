# Battle simulation concurrency: primary-source notes

Checked 2026-10-13. These are architectural findings and experiment proposals, not measured speedups for this repository.

## Python is not restricted to one core

In conventional CPython, the GIL allows only one thread per interpreter to execute Python bytecode at a time. Multiple processes can execute on different cores; native extensions can also release the GIL. A Python controller waiting for an independent Node simulator does not impose its GIL on the Node process. [Python threading documentation](https://docs.python.org/3.14/library/threading.html)

`multiprocessing` uses separate processes and bypasses the GIL. Use long-lived workers and coarse jobs to amortize process startup and serialization. On macOS, `spawn` is the default, and Python warns against `fork` because system libraries may start threads. Worker entry points must be importable, inputs serializable, and process creation guarded appropriately. [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)

One core can interleave many battles, but adding tasks does not create additional CPU capacity. It can improve throughput when battles spend time waiting on pipes, network operations, or another processor. When that core is already saturated with useful simulation work, reducing work per battle is the relevant experiment.

## Async is coordination, not extra CPU

An asyncio task that performs a long CPU calculation blocks other tasks on its event-loop thread. Async helps overlap independent waits; CPU-heavy work must be moved elsewhere to avoid blocking that loop. Therefore distinguish concurrent battles from simultaneously executing simulation engines. [Python asyncio development documentation](https://docs.python.org/3/library/asyncio-dev.html)

Node worker threads execute JavaScript in parallel. Node explicitly recommends a reusable worker pool because creating one worker per job can cost more than the work itself. Messages introduce copying/serialization unless suitable transferable or shared buffers are used. Node's built-in asynchronous I/O does not automatically distribute arbitrary JavaScript battle calculations across cores. [Node worker threads](https://nodejs.org/api/worker_threads.html), [Node event-loop explanation](https://nodejs.org/en/learn/asynchronous-work/dont-block-the-event-loop)

## Direct simulator integration is supported

Showdown documents `BattleStream`: write choices and consume protocol messages without hosting its multiplayer server. Its documented standard-I/O CLI path requires a separate subprocess per battle. A persistent custom Node worker that constructs a new in-process stream for each job is therefore an implementation proposal, not a claim that the CLI already multiplexes jobs. Compare this path with the current server/client route while retaining legal choices, format settings, teams, seeds, and result semantics. [Showdown simulator API](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md)

`@pkmn/sim` is an automatically extracted, typed/versioned version of Showdown's simulator with a documented `BattleStreams` example. It is not evidence of a faster engine: its README emphasizes staying close to Showdown and says vendoring Showdown may be preferable when its packaging features are unnecessary. Treat switching packages as an integration choice requiring version and format parity checks. [Package README](https://raw.githubusercontent.com/pkmn/ps/main/sim/README.md)

## Free-threaded Python is a conditional experiment

Free-threaded builds have existed since Python 3.13. A compatible build is not enough: unsupported extension imports may re-enable the GIL. Check actual runtime state using `sys._is_gil_enabled()` after imports where available. Shared mutable application state still needs synchronization. Free-threaded builds also have single-thread overhead and higher memory use; they do not accelerate an already saturated Node simulator or create extra hardware cores. [Python free-threading guide](https://docs.python.org/3.14/howto/free-threading-python.html)

## Practical experiment order

1. Measure controller CPU, simulator CPU, completed battles/second, turns/second, startup time, and memory separately.
2. Test several concurrent battles in one worker to establish whether waiting is material.
3. Test a persistent pool with 1, 2, 4, and then additional workers while preserving the same seeded workload. Track errors and unfinished battles alongside throughput.
4. If Python policy computation dominates, trial Python process workers. If Node simulation dominates, trial independent Node workers. Avoid multiplying both pools blindly.
5. If transport and initialization dominate, prototype direct `BattleStream` workers with batched job submission and summary results. Preserve a sampled detailed trace for correctness comparisons.

No throughput multiplier follows from these sources alone. Worker count, policy cost, battle duration, memory pressure, transport, and hardware determine the result.
