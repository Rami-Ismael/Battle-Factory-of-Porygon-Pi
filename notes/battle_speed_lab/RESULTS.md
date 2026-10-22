# Battle throughput experiments — measured results

13 October 2026. Practical settings and adventurous prototypes have been implemented and compared; scope limitations are explicit below. Start with the [decision report](DECISIONS.md).

**Independent confirmation:** the combined implementation is **2.741× faster** than the untouched upstream player across eight candidate teams and 3,072 additional measured battles (95% block-bootstrap interval: **2.663–2.809×**). Aggregate rates were **10.36 → 28.37 battles/second**, with gains in every block. A separate direct-engine comparison completed 960 measured BC battles with matching seeded event traces and a **2.341×** gain (2.287–2.393×). The broader experiments and their limitations follow below; the [decision report](DECISIONS.md) collects recommendations.

**Timing audit:** the first direct random-engine runs contain large wall-clock gaps that did not appear in the performance timer. Affected paired blocks remain invalidated. Fresh five-block reruns have now completed without flagged clock gaps; their results appear below. The BC screens, eight-team confirmation, direct BC comparison, and cold-Python-worker comparison have no detected gaps.

**The largest component improvement in the initial screen was observation encoding: 2.31× the instrumented baseline's throughput. Combining that encoder with shared batching, actor-only inference, inference mode, and one numerical thread reached 2.78×.** These initial screens cover one fixed candidate against 48 legal meta opponents; later confirmation and scaling experiments broaden coverage without establishing performance for every candidate or hardware condition.

| Implemented change | Throughput relative to baseline | Exploratory 95% block-bootstrap interval | Completed paired blocks |
|---|---:|---:|---:|
| Sparse encoder and cached move features | 2.307× | 2.255–2.345× | 5 |
| Combined fast stack | 2.784× | 2.693–2.893× | 5 |
| Actor-only inference plus inference mode | 1.050× | 1.021–1.078× | 5 |
| Shared inference across both players | 1.084× | 1.034–1.171× | 5 |
| One numerical CPU thread | 1.033× | 0.979–1.079× | 5 |
| 1 ms batch deadline | 0.919× | 0.880–0.996× | 5 |
| Shared queue + one thread + 1 ms deadline | 1.029× | 1.017–1.042× | 5 |

Above 1 is faster; below 1 is slower. Intervals are exploratory and unadjusted for screening multiple ideas. All valid measured trials are retained, including slow ones. The two screens comprise **13,440 measured battles**, plus warm-ups and separate correctness runs.

## Why the biggest change worked

In the first screen, building policy observations occupied a median of about 73% of baseline elapsed time. The reference encoder repeatedly searches feature vocabularies and loops through every possible effect for every Pokemon. The alternative uses precomputed vocabulary lookups, sparse effect assembly, a bounded move-feature cache keyed by all fields used by the reference, and a preallocated observation array. Dynamic HP, stats, effects, PP, and other changing fields are still updated.

The 1 ms batch deadline reduced waiting but fragmented inference into many more small batches. Lower latency did not translate into higher battle throughput. Likewise, reducing Torch threads alone did not establish a consistent improvement in this workload.

## What has been verified

- Optimized observation arrays exactly matched the reference on **3,627 live policy observations**, plus 100 varied constructed Pokemon states. This is tested coverage, not a proof over every game mechanic.
- Actor-only inference and cached-logit sampling passed **1,782 seeded action comparisons** over 297 captured real inputs, at batch sizes 1, 8, and 24.
- Cross-batch cached logits have floating-point differences. Their validation uses explicit numerical tolerances and sampled-action checks; they are not called bitwise identical. Logit caching has not yet earned a performance recommendation.
- Every accepted live-server trial completed its full workload and matched the expected preview base-species multiset. Mega forms require normalization. The candidate was fixed within each trial, so varying challenge arrival order does not change the requested candidate/opponent pairs.
- A direct Showdown random-policy prototype passed a small seeded trace-equivalence test across serial and concurrent execution. Only wall-clock timestamp lines are removed from trace hashes. This is not evidence of BC-policy throughput or competitive strength.

## How background activity was handled

Each treatment was bracketed by baseline trials in randomized ABBA or BAAB blocks. Treatment order was shuffled between blocks. Bootstrap resampling uses whole blocks, not individual battles as fake timing replicates. Startup and warm-ups are recorded separately. System load, estimated external CPU demand, and experiment memory/CPU snapshots accompany every trial.

The first screen's unchanged baseline varied from approximately 5 to 12 battles/second. That variation is why neither a single best run nor a simple before/after timing is used. The second screen's encoder gains appeared in all five blocks, and normalization by simulated turns gave essentially the same conclusion: encoder 2.313×; combined stack 2.784×.

These precautions reduce confounding; they do not make uncontrolled background load disappear. The result is conditional on this workload and these runs. The independent eight-candidate confirmation is complete, and the reusable evaluator packages the validated improvements. The original production pipeline has not been replaced; broader formats and hardware remain untested.

## Important limits of the baseline

The screen baseline reproduces the original batch-gathering algorithm with stage instrumentation in a persistent worker. It explicitly puts the policy into evaluation mode. It is not the full unchanged production pipeline, which also includes its own startup, sharding, and player setup. The separate upstream-player/training-mode comparison is complete and reported above as the 2.741× confirmation.

The live server uses fresh simulator randomness, and job-level policy seeds do not make scheduling changes trace-equivalent. The direct-engine BC adapter has per-battle RNG streams and passed the 960-battle comparison with matching event traces. Its initial failed smoke tests are preserved and are not counted as successful performance evidence. The matched whole-job direct/server comparison below establishes no gain for the direct route.

No production runtime, policy checkpoint, shared database, or existing background server has been modified. Implementations and private runtime copies live in this lab.

## Reproduction and evidence

- [Initial screen summary](runs/bc-screen-002/summary.json) and [raw trials](runs/bc-screen-002/trials.jsonl).
- [Encoder/actor screen summary](runs/bc-optimization-001/summary.json), [raw trials](runs/bc-optimization-001/trials.jsonl), and [source snapshot](runs/bc-optimization-001/source/run.py).
- [Live encoder checks](runs/encoder-verify-001/trials.jsonl) and [inference validation](runs/encoder-verify-001/inference-validation.json).
- [Complete experiment scope](scope.json) and [working status](STATUS.md).

The scope records implementation and measured evidence for each idea. Some implementations are deliberately limited prototypes: the exported actor still uses Python preprocessing/sampling, the specialized engine supports only restricted mechanics, and the durable queue was tested on one physical machine. Their measurements do not establish full native colocation, general VGC vectorization, or multi-host speedups.

## Confirmation and later experiment updates

The eight-team confirmation uses the original upstream `BatchPolicyPlayer`, its loaded training mode, and eight numerical threads as the baseline. The optimized condition uses evaluation mode, sparse encoding, actor-only inference, a shared queue, and one numerical thread. Startup metadata confirms the baseline loaded in training mode; the reference transformer has dropout set to zero. The comparison includes all of these changes, not just the encoder.

- [Eight-team confirmation](runs/bc-confirm-001/summary.json), [raw trials](runs/bc-confirm-001/trials.jsonl), [configuration and fingerprints](runs/bc-confirm-001/metadata.json).
- [Seeded direct BC comparison](runs/campaign-001-direct-bc/trials.jsonl): five complete blocks, 48 battles per trial, reference and optimized event traces equal after excluding only wall-clock timestamps. The initial generic summary's hardcoded status text incorrectly says these are live-server/non-equivalent trials; the raw `trace_equivalent` checks are the authoritative evidence. The analyzer has been corrected for subsequent summaries.
- The first persistent direct-engine trials with a random policy contained clock-invalid blocks. Accepted reruns appear below. These engine-only results must not be compared directly with the BC rates above.
- The original exploratory Node cold/warm throughput ratio was **0.068×** (0.061–0.075×). This estimate includes a block rejected by the timing audit and is not an accepted final result. The experiment deliberately launches a process per battle; its startup cost does not describe an existing long-lived server.
- Recreating the Python worker for each 48-battle batch yielded **0.794×** the warm-worker throughput (0.730–0.887×). Larger submitted batches will amortize startup differently.

The first serial campaign has ended. Its [manifest](runs/campaign-001/manifest.json) records successful settings/concurrency stages and the failed multicore warm-up. Its [events](runs/campaign-001/events.jsonl) preserve execution history.

## Additional settings sweep completed

The settings stage exited successfully with **9,600 measured battles**, five complete paired blocks per condition, and no detected legacy clock discontinuities. This is a separate screen under heavy, changing background activity; its ratios should not be multiplied by earlier gains.

| Setting versus instrumented baseline | Throughput ratio | Exploratory 95% block interval |
|---|---:|---:|
| Immediate queue draining | 0.828× | 0.709–0.905× |
| 2 ms batch deadline | 0.730× | 0.662–0.799× |
| Two numerical threads | 1.011× | 0.775–1.318× |
| Inference mode alone | 0.989× | 0.854–1.123× |
| Actor-only + inference mode + logit cache | 1.211× | 1.108–1.317× |

**The last row is not evidence that caching helped:** the cache recorded **0 hits and 23,806 misses** across measured treatment trials. Every request still needed inference, and this variant also skips the unused critic. The isolated fast-stack comparison reported below also established no cache gain. Exact full-input repetition appears unpromising for this workload.

Evidence: [settings analysis with accepted block IDs](runs/campaign-001-more-settings/analysis-v2.json), [raw trials](runs/campaign-001-more-settings/trials.jsonl). Later validation and campaign status are recorded below. Restored repeated seeds will not be counted as new independent battle observations.

## Concurrency completed; further experiments started

The concurrency sweep completed five paired blocks per condition, with no flagged clock interruptions. Relative to 24 in-flight battles, **50 achieved 1.143× throughput** (exploratory interval 1.044–1.283×); **8 achieved 0.934×** (0.905–0.972×). This changes how many battles overlap within a worker; it does not add CPU cores. Evidence: [concurrency summary](runs/campaign-001-concurrency/summary.json).

The first multicore attempt failed during warm-up because its seed calculation produced a negative number. It contributed no accepted performance estimate. The corrected smoke test and full `campaign-002` comparison subsequently completed.

Saved simulator states, pending requests, and both random-policy RNG streams passed **108 full-trace comparisons** against the original stream driver across three candidates and prefix turns 1, 3, and 8. This demonstrates tested restoration correctness. The later timing and fresh-seed miss audit below address usefulness separately. Evidence: [snapshot validation](runs/snapshot-validation-001/validation.json).

The shared actor service passed **891 numerical-logit cases and 1,782 seeded-action cases** over four concurrent clients, then completed a small live two-worker run. Cross-batch floating-point tolerance remains explicit. Workers still retain the policy for reference action sampling, so the prototype does not claim reduced model memory. Completed repeated timing below found it slower. Evidence: [remote validation](runs/remote-validation-001/validation.json).

`campaign-002` runs the rejected timing reruns, multicore comparisons, broader snapshot validation and timing, shared-inference timing, isolated cache comparison, and fresh outcome/training data collection serially. Its [manifest](runs/campaign-002/manifest.json) records completed stages. No new speed claims are made from smoke tests.

## Direct-engine timing reruns

The reruns completed **7,680 measured random-policy battles** for engine settings and **240** for per-battle process startup. All comparisons have five complete paired blocks, exact normalized trace agreement, and no flagged clock gaps.

| Change from persistent serial direct engine | Throughput ratio | Exploratory 95% block interval |
|---|---:|---:|
| Cache parsed/validated teams | 1.023× | 1.002–1.042× |
| 24 concurrent battles inside one engine process | 0.910× | 0.882–0.945× |
| Omit returned log payloads | 0.964× | 0.955–0.974× |
| Combine concurrency, team cache, and omitted logs | 0.967× | 0.958–0.978× |
| Launch a new Node process for every battle | 0.0713× | 0.0660–0.0785× |

The persistent-versus-per-battle comparison corresponds to roughly **14× throughput**, with aggregate rates 54.54 versus 3.91 battles/second. This is an engine-only random-policy result for deliberately frequent process launches; it is not a claim of a 14× improvement to the existing BC server pipeline. Async concurrency cannot create more CPU capacity within one busy process. The tiny team-cache gain remains a screening result and is much less consequential than observation encoding in the BC path.

Evidence: [engine rerun analysis](runs/campaign-002-direct-engine/analysis-v2.json), [cold/warm rerun analysis](runs/campaign-002-cold-engine/analysis-v2.json). The failed original timing blocks remain preserved.

## Multicore, shared inference, and saved openings

The corrected multicore comparison completed **15,360 measured BC battles**, with five paired blocks per condition and no flagged clock gaps. Each group has its own persistent Python worker and private simulator server. All groups use one numerical thread; these scaling comparisons use the original observation/inference path, not the optimized fast stack.

| Comparison | Throughput ratio | Exploratory 95% block interval |
|---|---:|---:|
| Two groups versus one | **1.977×** | 1.916–2.037× |
| Four groups versus one | **3.472×** | 3.354–3.637× |
| Dynamic versus static chunks, two groups | 0.995× | 0.897–1.075× |
| Dynamic versus static chunks, four groups | 0.914× | 0.761–1.079× |

The four-group comparison averaged **10.14 → 35.21 battles/second**. Separate processes are a demonstrated way to use more CPU cores here. Dynamic scheduling did not establish a gain for these equal-sized chunks; a workload with more uneven job durations could differ. Evidence: [scaling analysis](runs/campaign-002-scaling/analysis-v2.json).

The shared-actor test used the **fast encoder/actor stack on both sides**, completing another **7,680 measured BC battles**. One central actor service achieved **0.875×** local-inference throughput with two workers (0.859–0.892×), and **0.762×** with four (0.614–0.881×). Local fast workers averaged 49.33 and 84.23 battles/second in these comparisons; central service rates were 43.16 and 60.74. Keep local inference for this tested CPU configuration. These rates belong to a separate workload/configuration comparison and should not be multiplied by earlier speedup ratios. Evidence: [shared-actor analysis](runs/campaign-002-remote-bench/analysis-v2.json).

Saved turn-three states gave **1.212×** throughput for repeated-seed continuations (1.178–1.245×), with matching normalized traces over 960 measured random-policy continuation/full-run queries. Building 240 checkpoints took 3.82 seconds and stored 8.75 MB. All **48 unseen-seed queries missed**. This supports faster repeat queries, not more independent battle samples; cross-team prefix sharing is not established. Evidence: [snapshot timing](runs/campaign-002-snapshot-bench/analysis-v2.json), [reuse audit](runs/campaign-002-snapshot-bench/reuse-audit.json).

## Cache isolation and exported actor validation

Adding the logit cache to the otherwise identical fast stack gave **1.006×** throughput (0.975–1.043×) over 1,920 measured battles, with **zero hits and 23,930 misses**. No performance benefit is established. This resolves the earlier confounded cache screen: skipping the critic and caching were separate changes. Evidence: [isolated cache comparison](runs/campaign-002-cache-isolation/analysis-v2.json).

The frozen actor exported to ONNX passed **1,188 numerical-logit comparisons and 2,376 seeded-action comparisons**. The Node-hosted actor then matched normalized full traces in a 12-battle comparison against the Torch actor. Numerical tolerance is explicit; this is tested coverage, not a proof for every state. Python still builds observations and samples actions, so this prototype **does not eliminate Python–Node round trips**. Its completed speed comparison below establishes no gain. Evidence: [export validation](runs/export-validation-001/validation.json), [Node battle check](runs/node-actor-validation-001/validation.json).

Fresh data collection completed 8,192 stochastic BC battles over eight candidates, eight common opponents and 128 paired replicates per matchup, plus 20,545 observations from a separate 768-battle collection. Allocation strategies use an exploration/held-out replicate split. Apprentice training uses disjoint battle files; teams can still appear as opponents across splits, so this does not establish generalization to wholly unseen teams. The [third campaign manifest](runs/campaign-003/manifest.json) records the continuing serial queue, restricted-engine, exported-actor, apprentice and allocation experiments.

## Durable queue and restricted array simulator

The leased HTTP queue completed five paired blocks with exact normalized random-policy traces and no flagged clock gaps. Its throughput was **0.957×** static local chunk assignment (0.925–0.991×): a small measured overhead for durability and transport. Recovery tests cover expired-worker fencing, unique concurrent claims, persistent jobs, and idempotent completion. All worker engines were on this laptop; multi-machine network scaling is untested. Evidence: [queue comparison](runs/campaign-003-queue/summary.json).

The restricted simulator supports only two active Pokemon per side with fixed-damage Seismic Toss, unique speeds, no items, no switches, and no changing effects. Both specialized scalar and vectorized implementations matched **3,453 damage events across 128 Showdown battles**, including terminal HP, winner and turn. **Zero of the 49 real teams qualify**, so unsupported VGC battles still need Showdown.

Within those restricted rules, arrays achieved **2.357×** scalar throughput (2.273–2.459×). The specialized scalar/full-Showdown ratio was 405.6× and array/full-Showdown 843.7×, but those comparisons also remove team/stat preparation, generic rules, protocol logging and response payloads; they do not isolate vectorization and cannot be transferred to ordinary VGC. The useful isolated result is 2.36× for arrays over the same simplified scalar rules. Trials use calibrated repetitions rather than single millisecond timings, with all five blocks free of clock flags. Evidence: [restricted validation](runs/campaign-003-array/validation.json), [timings](runs/campaign-003-array/summary.json).

## Exported actor timing and first apprentice

The Node-hosted exported actor completed five paired blocks with exact normalized BC traces and no clock flags. Its throughput was **0.940×** the fast Torch actor (0.803–1.093×), so this partial colocation showed no clear gain. Python preprocessing and action-sampling round trips are still present. Evidence: [exported actor timing](runs/campaign-003-node-actor/summary.json).

The first distilled policy trained on 13,532 observations from 480 battles and was checked on 7,013 observations from 288 separate battles. It matched the teacher's **joint greedy action in only 22.6%** of validation states. Final validation KL was 0.914 under the documented marginal-mask objective. This is poor evidence for replacing the original evaluator. Evidence: [apprentice validation](runs/campaign-003/student.json).

The live apprentice comparison completed five paired blocks without clock flags. Completed-battle throughput was **0.912×** the fast teacher (0.826–0.983×), despite **1.131× turns/second**. Mean battle length increased from **8.30 to 10.33 turns**. Both sides used the changed policy; the candidate win fractions, 59.1% versus 58.3%, are not a head-to-head policy-strength test. This apprentice offers neither established fidelity nor a completed-battle speed gain. Evidence: [student timing](runs/campaign-003-student/summary.json), [outcome audit](runs/campaign-002-quality-corpus/quality-audit.json).

## Allocation and paired randomness

The completed outcome grid contains 4,096 exploration battles and 4,096 held-out battles. Fifty shuffles reuse those same observations; they measure stability within this corpus, not fifty independent experiments. At a **512-battle exploration budget**, the mean held-out score of the selected team was:

| Strategy | Selected team's mean held-out score |
|---|---:|
| Uniform allocation | 59.45% |
| Successive-halving racing | 59.49% |
| Pilot-variance opponent allocation | 58.51% |
| Species-based ridge surrogate with random audits | 58.49% |

These small differences do not justify promoting an adaptive method. At the maximum 4,096-battle budget, racing consumed about 2,773 battles and selected the same team as full uniform sampling, but this also reflects exhausted data for surviving candidates in the finite replay. It is not a demonstrated wall-clock search speedup.

The exploration-best team scored 63.09% on holdout; the empirical holdout-best scored 64.84%. Fixed-data simultaneous Hoeffding bounds give a ±7.51-percentage-point radius per candidate under the stated independent-replicate assumption, so a unique best team is **not certified**. Do not interpret failure to select the empirical holdout winner as proof that a strategy selected the truly weaker team.

Common RNG seeds reduced estimated comparison variance for **12 of 28 candidate pairs** and increased it for the others. Estimated independent/paired variance ratios ranged **0.881–1.060×**, so there is no universal compute multiplier here. Seeds remain useful for reproducibility. Evidence: [allocation replay](runs/campaign-002-quality-corpus/allocation-analysis.json), [held-out uncertainty audit](runs/campaign-002-quality-corpus/quality-audit.json). The completed live quality-at-wall-clock comparison is reported below.

## Optimized pool and usable runner

The combined fast encoder/inference stack completed five paired blocks for four and six worker/server groups versus one. **Four groups reached 3.045×** (2.873–3.182×); **six reached 3.607×** (3.286–3.960×). No clock interruptions were flagged. Aggregate rates were 20.52 → 61.39 and 21.27 → 75.17 battles/second, respectively, under the background activity present in this run. These are direct comparisons of the same optimized stack, not products of unrelated speed ratios. Evidence: [optimized pool analysis](runs/optimized-pool-001/analysis-v2.json).

The reusable [fast evaluator](evaluate_pool.py) keeps the optimized workers and private servers alive across chunks. Its two-worker live check completed 16 battles across four chunks, with exact requested counts and attribution checks. It writes separate workload, provenance and result files and leaves the original database untouched. [Usage instructions](README.md) include candidate listing and a four-worker command; six workers are available as an explicit option. The runner's smoke test is correctness evidence, not another throughput estimate.

## Cross-team prefix safety

Across 64 legal single-nature mutations and turn-three checkpoints, **zero full simulator/policy/RNG states matched**. Nineteen mutations nevertheless had the same active-board fingerprint (species, HP, status and boosts); twelve of those produced different full battle traces. The checkpoint guard correctly refuses reuse in all 64 cases. This tested mutation family provides no cross-team prefix speed gain; matching only the visible board would bypass relevant state. Evidence: [prefix audit](runs/prefix-audit-001/audit.json).

## Remaining deadline screen

The previously untested **0.5 ms deadline** completed five paired blocks and 1,920 measured battles without clock flags. Throughput was **0.865×** the original batching behavior (0.835–0.892×). This completes the 0/0.5/1/2 ms screens; none improved throughput in their tested conditions. Keep the original batching behavior in the packaged fast runner. Evidence: [0.5 ms analysis](runs/deadline05-001/analysis-v2.json).

## Matched direct/server comparison

Over five candidate workloads and 1,920 measured battles, the fast direct adapter achieved **0.950×** the fast live-server adapter's whole-job throughput (0.783–1.077×), with no flagged clock gaps. Aggregate rates were 20.22 versus 22.08 battles/second. This does not establish a gain from the direct route in its current implementation.

The measurement includes the direct adapter's per-job Node launch and both adapters' job setup; it does not isolate WebSocket cost. Policy loading is excluded for both, and server simulator randomness is not exactly paired with direct seeds. Evidence: [matched adapter analysis](runs/campaign-004-transport/analysis-v2.json).

The [evidence audit](EVIDENCE-AUDIT.json) checks raw complete-block accounting and preserved source hashes. The early `bc-screen-002` run retained worker/controller hashes but did not archive the source files; its historical component screen has weaker reproduction provenance. Later confirmation and implementation experiments preserve their source snapshots. No missing archive is treated as an exact source verification.

## Live selection quality at an equal time budget

The live experiment completed **60 trials of 15 seconds**, five randomized four-trial blocks for each adaptive method against its own uniform baseline. All initial sampling rounds completed and no clock gaps were flagged. It ran **17,904 physical battles**, accepting **16,960** before the deadlines; the remaining 944 were drained and logged but could not influence selection. Per-candidate/opponent counts reconcile with every recorded exploration result.

| Method | Uniform baseline held-out score | Method held-out score | Paired difference, percentage points (95% block-bootstrap interval) |
|---|---:|---:|---:|
| Successive-halving racing | 55.66% | 59.28% | +3.61 (−1.84 to +8.96) |
| Pilot-variance opponent allocation | 60.78% | 59.65% | −1.13 (−5.20 to +2.79) |
| Species ridge surrogate with audits | 56.99% | 59.55% | +2.56 (−2.30 to +8.46) |

Each row has its own interleaved baseline; do not compare unrelated baseline columns as treatment effects. All quality intervals include zero. These first adaptive methods therefore have **no established equal-time quality advantage**. Eligible-battle ratios were 0.987×, 1.001× and 0.959×, respectively, also with intervals spanning 1.

The intervals condition on the same finite held-out labels and do not include uncertainty in those labels. Five blocks are a small exploratory screen, with no multiple-comparison correction; no unique best team was certified in the held-out corpus. Evidence: [live accounting and paired analysis](runs/campaign-004-live-allocation/analysis-v2.json), [raw live trials](runs/campaign-004-live-allocation/trials.jsonl), [campaign completion](runs/campaign-004/manifest.json).

## Final optimized-pool concurrency comparison

The final sweep completed **48,000 measured battles across 80 trials**, using twelve fixed 50-battle jobs per trial. Each comparison has five accepted paired blocks and no detected clock discontinuities. Worker counts were held fixed within comparisons. The tested 8/24/50 battle limits produced observed shared policy batches up to 16/48/100 requests, confirming that the setting affected execution.

| Workers | In-flight battles per worker versus 24 | Throughput ratio | Exploratory 95% block-bootstrap interval |
|---|---:|---:|---:|
| 4 | 8 | 0.863× | 0.809–0.956× |
| 4 | 50 | 0.942× | 0.778–1.074× |
| 6 | 8 | 0.837× | 0.750–0.934× |
| 6 | 50 | 0.995× | 0.889–1.098× |

Keep **24 in-flight battles per worker** in the optimized pool. Eight was slower; fifty showed no clear gain. The earlier original-stack single-worker result does not transfer automatically to the optimized multi-worker configuration.

Estimated external CPU usage varied substantially; one sample reached about 896% (roughly nine fully occupied cores). Slow trials remain included. Interleaved normalization and block uncertainty account for some variation but cannot eliminate interference, especially with five blocks. Absolute aggregate baseline rates ranged about 78–98 battles/second across these four comparisons; these are different 50-battle-job workloads and should not be substituted for the earlier pool rates or multiplied by their ratios.

Evidence: [complete concurrency analysis](runs/optimized-concurrency-001/analysis-v2.json), [raw trials](runs/optimized-concurrency-001/trials.jsonl), [archived settings and source fingerprints](runs/optimized-concurrency-001/metadata.json). The final evidence audit checks 21 accepted timing runs plus live allocation, with no failures and one explicitly documented legacy source-archive warning.
