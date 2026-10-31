# Experiment names

Reader names for the arms of every experiment documented in this folder, so blog posts, figures and replies can quote an old experiment without its code labels. Code, results files, JSON keys, checkpoints and the source docs keep their code IDs; the reader name replaces the code ID only in reader-facing text. On first mention, put the code ID in code formatting beside the reader name when a reader may need to look it up.

The names follow the `name-experiments` skill: the reader name comes first, the first two words of each reader name set the arm apart from its siblings, and settings every arm shares sit in the line above the table. "none" in the code ID column means the source doc gives that arm no identifier; quote the source doc and section for traceability instead of inventing one.

The two diffusion models most later runs mix are named by their training data: the loop-teams model (`F1`, trained on loop teams only) and the matrix-rows model (`G2`, conditioned on matchup-matrix rows, no pretraining). See [Structured output, training data and unmasking order](#structured-output-training-data-and-unmasking-order-against-the-old-model).

## Remasking during decoding

Source: `remask_ask_gap.json`, `remask_why.json` (2026-10-30)

Shared by every arm: the masked diffusion model at its usual settings; the arms differ only in remasking. The selection rule leads each name because it decided the result; the 10% least-likely arms at 48 and 96 passes share their first two words and differ in the pass count.

| Reader name | Code ID |
| --- | --- |
| No remasking | `plain` |
| Least-likely 5% remasked, 48 passes | `remask05` |
| Least-likely 10% remasked, 48 passes | `remask10` |
| Least-likely 20% remasked, 48 passes | `remask20` |
| Least-likely 10% remasked, 96 passes | `remask10x2` |
| Random 10% remasked, 48 passes | `random10` |

## Single-edit landscape comparison: Pokémon teams against two benchmarks

Source: [`ruggedness.md`](ruggedness.md)

Shared by every arm: 15 single edits per anchor, 3 replicates per anchor. The Pokémon arm uses 24 real Regulation M-B teams as anchors and scores each edit with 384 battles against the top-50 meta teams, behaviour-cloning policy on both sides; the benchmark arms score edits with their own simulated objective.

| Reader name | Code ID |
| --- | --- |
| Battle win rate under single legal edits | none |
| COMBO Pest Control benchmark landscape | none |
| Hartmann-6 benchmark landscape | none |

## Single-edit landscape comparison, species edits added

Source: [`ruggedness2.md`](ruggedness2.md)

Shared by every arm: 3 replicates per anchor. The Pokémon arm uses 40 anchors (24 meta teams, 16 search-found teams), 3 edits per edit type, and 384 battles per team against the top-50 meta teams, behaviour-cloning policy on both sides; the benchmark arms score edits with their own simulated objective.

| Reader name | Code ID |
| --- | --- |
| Battle win rate under six edit types, species edit included | none |
| COMBO Pest Control benchmark landscape | none |
| Gridded sphere benchmark landscape | none |

## Landscape features of each benchmark, against battle win rate

Source: [`bench_features.md`](bench_features.md)

Shared by every arm: 3 replicates per anchor, minimisation throughout. Benchmark arms use 24 random start anchors plus 16 search-found anchors with 15 single edits each; the reused Pokémon arm uses six edit types with three edits each and battles against the top-50 meta teams.

| Reader name | Code ID |
| --- | --- |
| Battle win rate landscape | none |
| Contamination control benchmark landscape | none |
| Ising sparsification benchmark landscape | none |
| Pest Control benchmark landscape (two implementations) | none |
| Weighted maximum satisfiability benchmark landscape | none |
| Low-autocorrelation binary sequence benchmark landscape | none |
| Ackley-53 benchmark landscape | none |
| Penalised octanol-water partition benchmark landscape | none |
| GuacaMol Perindopril and Zaleplon benchmark landscape | none |
| Dopamine receptor D2 oracle benchmark landscape | none |

## Filler for masked team parts: language model or diffusion model

Source: [`dense-mask-grid-results.md`](dense-mask-grid-results.md)

Shared by every arm: 306 masked tasks across seven mask families, 54 tasks scored in both arms, the original 50-battle meta-opponent schedule, behaviour-cloning policy on both sides, policy seeds match and simulator randomness is not fixed.

| Reader name | Code ID |
| --- | --- |
| Language model fills masks (Ling) | none |
| Diffusion model fills masks, two-model mixture, version 2 | none |

## Search method: posterior-density search or cross-entropy method

Source: [`dibo-v4.md`](dibo-v4.md)

Shared by every arm: seed 5050, the same frozen 5,000 starting labels, 10 rounds of 128 selected teams each battled against 49 top-50 columns (62,720 search battles per arm), then 16 finalists × (192 confirmation + 192 held-out) battles, 68,864 per arm; one training seed; equal battle budgets.

| Reader name | Code ID |
| --- | --- |
| Cross-entropy method with elite cut, historical nature repair | none |
| Posterior-density search (the DiBO method), categorical adaptation | none |

## Filler for masked team parts: language model or random legal fill

Source: [`llm-baseline-pilot-results.md`](llm-baseline-pilot-results.md)

Shared by every arm: three random Champions Regulation M-B starting teams, seven masked features at three mask sizes (63 masked tasks), behaviour-cloning policy on both sides including learned team preview, 50 battles per panel against the 49 distinct stored meta teams, policy seeds shared, Showdown's internal random seed not controlled.

| Reader name | Code ID |
| --- | --- |
| Language model fills masks | `inclusionai/ling-3.0-flash-vl` |
| No language model, random legal fills | none |

## Filler for masked team parts on 49 meta starters: language model, diffusion model, or random legal fill

Source: [`meta-starter-grid.md`](meta-starter-grid.md)

Shared by every arm: 49 distinct meta-pool teams as starters, 102 masks per starter (4,998 tasks, one draw each), each starter's own paste removed from the opponents shown and from its battle panel, the same 50-battle schedule over the other 48 teams, behaviour-cloning policy on both sides, policy seeds match and simulator randomness is not fixed.

| Reader name | Code ID |
| --- | --- |
| Original meta team, unmasked | none |
| No model, random legal fills | none |
| Language model fills masks | `llm-baseline-meta-starters` |
| Diffusion model fills masks | `diffusion-baseline-meta-starters` |

## Species repulsion added to focused-set sampling

Source: [`species-repulsion-replication.md`](species-repulsion-replication.md)

Shared by every arm: both arms sample species at temperature 1 and every other field at temperature 0.5; asked win rates 0.5 and 0.7; sampling seeds 1 to 3; 64 teams per cell; guidance strength 2 on the two-model mixture (loop-teams model and matrix-rows model); one battle per top-50 column (49 per team); behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No species repulsion, focused sets | `settemp05` |
| Species repulsion, focused sets | `repel_settemp05` |

## Redundancy penalty weight in elite selection

Source: [`redundancy-penalty.md`](redundancy-penalty.md)

Shared by every arm: six generations, three seeds, 64 battled teams per generation, 49 battles per team against the top-50 meta teams, behaviour-cloning policy on both sides, elite count max(100, 0.25 × labelled pool), common random numbers per seed. The mmr10 arm was dropped by the 2026-10-04 amendment.

| Reader name | Code ID |
| --- | --- |
| No redundancy penalty, hard top-k cut | `cut` |
| 0.30 redundancy penalty | `mmr30` |
| 0.60 redundancy penalty | `mmr60` |

## Guidance strength for the asked win rate

Source: [`asked-vs-got.md`](asked-vs-got.md)

Shared by every arm: asked win rates 0.0 to 0.9 in steps of 0.1, 64 Showdown-valid teams per cell at temperature 1.0 with corpus Stat Point spreads, 49 battles per team against the top-50 meta teams, behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| Strength-1 guidance | none |
| Strength-2 guidance | none |
| Strength-4 guidance | none |
| No guidance, unconditional | none |

## Team slot order at team preview

Source: [`slot-order.md`](slot-order.md)

Shared by every arm: behaviour-cloning policy on both sides, battles against the top-50 meta teams, same opponent draws in both arms (common random numbers), 12 teams in six cyclic rotations, 192 battles per listing.

| Reader name | Code ID |
| --- | --- |
| Written order, no shuffling | `listed` |
| Shuffled order, every battle | `shuffled` |

## Language model prompt wording: original or ruleset-grounded

Source: [`ling-mb-prompt-comparison.md`](ling-mb-prompt-comparison.md)

Shared by every arm: the same 147 existing masked tasks from seven starters across seven mask families; same masks, opponent exclusions, language model, temperature and output budget; one response per task per prompt. Battle scores are pending.

| Reader name | Code ID |
| --- | --- |
| Original prompt, no ruleset hints | none |
| Ruleset and legal-choice hints, revised prompt | `mb-grounded-v2` |
| Diffusion model fills masks, v3 | none |

## Search method at a matched battle budget

Source: [`dibo-comparison-protocol.md`](dibo-comparison-protocol.md)

Shared by every arm: one training seed (4040), the same starting labels and foundation model, 10 rounds of 512 proposals with 128 new distinct teams per round, each battled against 49 top-50 columns (62,720 search battles per arm), then 16 finalists × (192 top-50 + 192 held-out) battles, 68,864 per arm.

| Reader name | Code ID |
| --- | --- |
| Cross-entropy method with fixed 0.531 elite cut | none |
| Posterior-density search (the DiBO method), full default settings | none |

## Synthetic training data: v2 or v3 move-count coverage

Source: [`empty-moves-and-mega-stone-diagnosis.md`](empty-moves-and-mega-stone-diagnosis.md)

Shared by every arm: 36,000 synthetic Pokémon per dataset, each with a simulator-validated moveset; battle win rates not measured for either version in this doc.

| Reader name | Code ID |
| --- | --- |
| Previous dataset, v2 | none |
| Prepared dataset, v3, with one-, two- and three-move sets | none |

## Score guidance direction and size

Source: [`reverse-score-diversity-results.md`](reverse-score-diversity-results.md)

Shared by every arm: three matched generation seeds, 512 raw attempts per direction and seed (4,608 total), one frozen checkpoint and surrogate, decoding temperature 1.0, no retraining, invalid and repeated outputs kept. Battle strength was not tested.

| Reader name | Code ID |
| --- | --- |
| No score guidance | `0` |
| Normal score guidance, +108 | `+108` |
| Reversed score guidance, −108 | `−108` |

## Tree search over partial teams versus direct sampling of complete teams

Source: [`mdts-experiment.md`](mdts-experiment.md)

Shared by every arm: the same frozen generator (temperature_p0.pt, 600 epochs on 692 corpus teams, no retraining), decoding temperature 1.0, the same number of scored complete teams per replicate (about 400), 16 selected teams each scored with 8 battles against each of 49 distinct top-50 teams, six replicates (seeds 1101 to 1606), and a bootstrap over replicates for the decision rule.

| Reader name | Code ID |
| --- | --- |
| No tree search: complete teams sampled directly and ranked | `w/o MDTS` |
| Masked diffusion tree search: partial teams expanded level by level | `MDTS` |

## Score sign in the cross-entropy method, elites ranked by score or by negated score

Source: [`cem-score-direction-results.md`](cem-score-direction-results.md)

Shared by every arm: the same starting checkpoint (trained on 692 six-member corpus teams), sampling temperature, training budget (five cumulative updates), battle policy and opponent pool; three seeds.

| Reader name | Code ID |
| --- | --- |
| No sign flip: elites ranked by measured score (control) | `score` |
| Sign flip: elites ranked by negated score | `-score` |
| Random elites: elites chosen uniformly at random | none |

## Score sign in the cross-entropy method, locked protocol

Source: [`cem-score-direction-protocol.md`](cem-score-direction-protocol.md)

Shared by every arm: the same starting checkpoint (temperature_p0.pt, 692-team corpus), decoding temperature 1.0 with no score guidance, identical battle schedules and policy seeds within each seed and iteration, five cumulative updates, seeds 2101, 2202 and 2303.

| Reader name | Code ID |
| --- | --- |
| No sign flip: elites ranked by score | `score` |
| Sign flip: elites ranked by negated score | `-score` |
| Random elites: elites chosen at random | none |

## Direction of predicted-score guidance during generation

Source: [`reverse-score-diversity.md`](reverse-score-diversity.md)

Shared by every arm: the same 600-epoch checkpoint (temperature_p0.pt), one ridge surrogate fitted once on 200 anchor teams and 128 generation-0 labels, decoder temperature 1.0, 512 raw attempts per direction and seed, seeds 1101, 1202 and 1303, no battles.

| Reader name | Code ID |
| --- | --- |
| Normal direction: guidance tilts toward higher predicted score | `+108` |
| No guidance: unguided sampling (control) | `0` |
| Reversed direction: guidance tilts toward lower predicted score | `−108` |

## Changes to the generation loop meant to keep species-set diversity

Source: [`diversity-methods.md`](diversity-methods.md)

Shared by every arm: every arm and seed starts from the version-3 cross-entropy loop's generation 0 (cem_v3: 512 proposals, 128 battled teams, labels, seed 4040), branches at generation 1, runs to generation 6 with seeds 1, 2 and 3, and battles against the top-50 meta teams under the behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No diversity rule: the version-3 loop with its stopping rules off | `control` |
| Repeat-set rejection: reject proposals whose six species already appeared | `taboo` |
| Frequent-species ban: mask the most common species out of the sampler, two generations per epoch | `peel` |
| Four islands: separate elite pools with migration between them | `island` |
| Larger pool: 4,096 accepted proposals, top 128 by ridge score battled | `big` |
| Diverse-subset selection: a determinantal point process picks 128 of the same 4,096 | `dpp` |

## Higher-order interaction in win rate, VGC edits versus benchmark landscapes

Source: [`interaction-order.md`](interaction-order.md)

Shared by every arm: a 2^4 factorial design with 16 points per cube, Walsh coefficients, two replicates with independent seeds 101 and 202 under common random numbers, bootstrap confidence intervals over anchors (2,000 resamples).

| Reader name | Code ID |
| --- | --- |
| Random species edits: four random team slots replaced by donor species | none |
| Weather core: weather setters swapped for weather-neutral donors | none |
| Gridded sphere: each term depends on one factor | none |
| Spin glass: random pairwise couplings and fields | none |
| Weighted maximum satisfiability: 106 random three-literal clauses | none |
| NK landscape: 25 parts, interactions up to order five | none |
| Pest Control: 25 stages, five choices each, stochastic | none |

## Diffusion model versus Ling language model on frozen masked-completion tasks

Source: [`diffusion-baseline-comparison.md`](diffusion-baseline-comparison.md)

Shared by every arm: frozen masked-completion tasks, diffusion target win rate 0.5, classifier-free guidance 2, temperature 1, fixed-order inpainting, the same 50-battle schedule, 49 opponent teams.

| Reader name | Code ID |
| --- | --- |
| No change: the original team as given | none |
| Random fill: masked fields filled at random | none |
| Ling language model: prompted completion | none |
| Diffusion model: masked completion conditioned on target win rate | none |

## Corpus vocabulary versus simulator vocabulary in the diffusion model checkpoints

Source: [`diffusion-baseline-protocol.md`](diffusion-baseline-protocol.md)

Shared by every arm: the same two diffusion checkpoints (the loop-teams model `F1_final_medium_regmb_v2` and the matrix-rows model `G2_family_matrix_noprtrain_regmb_v2` in the simulator run), target win rate 0.5, classifier-free guidance 2, temperature 1, the same Ling pilot tasks and opponent panels, and fixed-order inpainting.

| Reader name | Code ID |
| --- | --- |
| Corpus vocabulary: checkpoints built on the corpus vocabulary, seven supported tasks | `diffusion-baseline-f1g2-strict` |
| Simulator vocabulary: retrained checkpoints, all 63 contexts supported | `diffusion-baseline-f1g2-regmb-v2` |

## Deep-surrogate MAP-Elites archive search versus the existing ridge loop

Source: [`dsame-experiment.md`](dsame-experiment.md)

Shared by every arm: both arms start from the same labels (200 anchor teams and 128 generation-0 proposals, 24 battles each against the top-50 pool); each generation each arm battles 128 new teams × 24 battles against all 50 top-50 files; six generations; seeds 11, 22 and 33; primary measure is verified win rate pooled over the three seeds, bootstrap 95% interval over teams within seed (10,000 resamples).

| Reader name | Code ID |
| --- | --- |
| No deep surrogate: the existing ridge loop with decode-time guidance (λ = 108) | `baseline` |
| Deep-surrogate elite archive search, scored by a trained ensemble of networks | `dsame` |

## Hand-written operators vs refill (deep-surrogate MAP-Elites)

Source: [`diffusion-mutation-arm.md`](diffusion-mutation-arm.md)

Shared by every arm: 313 starting labels, surrogate of 3 multilayer perceptrons retrained each generation, 8,000 Showdown-valid children per generation, 6 generations × 128 teams × 24 battles against the top-50 pool, seeds 11, 22, 33.

| Reader name | Code ID |
| --- | --- |
| Hand-written operators, no refill | `A` |
| Uniform random refill | `B` |
| Diffusion model refill | `C` |

## Regulation reference in the prompt

Source: [`prompt-changelog.md`](prompt-changelog.md)

Shared by every arm: `inclusionai/ling-3.0-flash-vl`, temperature 0.7, 4,096-token cap, reasoning off, Novita route; v1 and v2 scored on the 147-task pilot, v1 and v3 on the 294-response regulation test.

| Reader name | Code ID |
| --- | --- |
| No regulation reference | `mb-original-v1` |
| Regulation M-B reference added | `mb-regulation-v3` |
| Legal moves and ability-item pairs added | `mb-grounded-v2` |

## Original checkpoint vs fixed 0.10 selection temperature

Source: [`diversity-metrics.md`](diversity-metrics.md)

Shared by every arm: three seeds, 512-proposal samples, pilot representation of team composition.

| Reader name | Code ID |
| --- | --- |
| Original checkpoint, no refitting | none |
| Fixed-0.10 selection temperature | none |

## Original loop vs clean loop (cross-entropy method)

Source: [`cem-v3-loop.md`](cem-v3-loop.md)

Shared by every arm: version-3 checkpoints, seed 4040, 512 proposals per generation, 128 battled per generation, elites at or above 0.531, 16 finalists re-battled with 192 fresh battles against the top-50 pool.

| Reader name | Code ID |
| --- | --- |
| Original loop, held-out teams in corpus | `cem_v3` |
| Clean loop, held-out teams removed | `cem_v3_noheldout` |

## Decoding methods on the version-3 model

Source: [`foundation-swap.md`](foundation-swap.md)

Shared by every arm: version-3 model with the full Regulation M-B vocabulary, 64 Showdown-valid teams per arm, 49 battles per team against the top-50 pool, one ridge surrogate, same seeds per arm.

| Reader name | Code ID |
| --- | --- |
| No decoding trick | `base` |
| Classifier-free guidance, ask 0.5 | `cfg` |
| Temperature 0.7 with top-p 0.9 | `lowtemp` |
| Keep teams with at least three Protect-family Pokémon | `protect3` |
| Ridge surrogate picks top 64 of 512 | `surrogate` |
| Ridge-gradient tilt on logits at decode time | `guide` |
| Nature matched to move split after decoding | `repair` |
| All six methods stacked | `stack` |

## Species repulsion within one sampling batch

Source: [`species-repulsion.md`](species-repulsion.md)

Shared by every arm: asked values 0.1 to 0.9, guidance 2, first 64 Showdown-valid teams per cell, Stat Point spreads copied from the corpus, 49 battles per team, seed 6161, two-model mixture (loop-teams model and matrix-rows model).

| Reader name | Code ID |
| --- | --- |
| No repulsion, temperature 1 | `today` |
| Free species, focused sets | `settemp05` |
| Repulsion alone, strength 256 | `repel` |
| Repulsion plus free species, focused sets | `repel_settemp05` |

## Annealed vs fixed and adaptive selection temperature

Source: [`temperature-results.md`](temperature-results.md)

Shared by every arm: one starting checkpoint and cached initial labels, behaviour-cloning policy, 50 opponents, 40-epoch refit per generation, 11 generations, seeds 101, 202, 303, 512 proposals with 128 battled per generation, 192 fresh battles per evaluated team.

| Reader name | Code ID |
| --- | --- |
| Annealed 0.30 to 0.10 selection temperature | none |
| Fixed-0.10 selection temperature | none |
| Fixed-0.20 selection temperature | none |
| Fixed-0.30 selection temperature | none |
| Adaptive effective-sample selection temperature, target 0.50 | none |

## Bracket count, draw balance and pool growth

Source: [`plan-vs-results-2026-08-28.md`](plan-vs-results-2026-08-28.md)

Shared by every arm: each arm fine-tunes the shipped checkpoint for 4,560 steps at learning rate 1e-4, decodes 150 Showdown-valid teams with dependency order, temperature 0.7, top-p 0.9 and guidance 2, and battles each team 24 times.

| Reader name | Code ID |
| --- | --- |
| Six-bracket without growth | `base` |
| Six-bracket balanced draws | `balanced6` |
| Eight-bracket without growth | `base_fine` |
| Six-bracket grown pool | `grown` |
| Eight-bracket grown pool | `grown_fine` |
| Eight-bracket balanced grown pool | `grown_fine_bal` |

## Diversity methods on the diffusion black-box optimiser, against no diversity change

Source: [`dibo-diversity.md`](dibo-diversity.md)

Shared by every arm: Diffusion black-box optimisation (DiBO) version 4 control; default 10 rounds of 128 proposal batches × 49 battles = 62,720 fresh search battles and 6,144 final battles per arm; same frozen initial data for every arm.

| Reader name | Code ID |
| --- | --- |
| No diversity change: standard buffer and selection | `control` |
| Annealed selection temperature: prior weights from 0.30 down to 0.10 | `boltz` |
| Fixed selection temperature 0.10: prior weights | `fixed_010` |
| Fixed selection temperature 0.20: prior weights | `fixed_020` |
| Fixed selection temperature 0.30: prior weights | `fixed_030` |
| Adaptive temperature: effective sample size 0.5, bounded 0.10 to 0.30 | `adaptive` |
| One build per roster in prior training, other builds zero weight | `niche` |
| Entropy bonus in prior loss: annealed 0.30 to 0.10 | `refit` |
| 0.30 redundancy penalty in buffer selection (maximal marginal relevance) | `mmr30` |
| 0.60 redundancy penalty in buffer selection (maximal marginal relevance) | `mmr60` |
| Roster taboo: earlier rosters rejected, two builds per roster at most | `taboo` |
| Winning-core peeling: three frequent species banned on alternate rounds | `peel` |
| Four isolated loops: migration of teams between loops | `island` |
| Larger proposal pool: eight times standard | `big` |
| Determinantal point process acquisition over the larger pool, species overlap penalised | `dpp` |
| Reversed prior weighting on low-labelled rows | `negative` |
| Random buffer, uniform prior, random acquisition | `random` |
| No posterior reward tilt, legality penalties kept | `no_guidance` |

## Language model refill against random refill and no refill, on masked team fields

Source: [`llm-baseline.md`](llm-baseline.md)

Shared by every arm: same three starting teams and masked fields per arm; mask sizes k = 1, 3, 6 (stats, items, abilities, alignments, whole Pokémon), 1, 12, 24 (moves), 1, 20, 40 (mixed parts); temperature 0.7; 50 battles per candidate against the top-50 meta teams, behaviour-cloning policy both sides.

| Reader name | Code ID |
| --- | --- |
| No refill: original team kept | none |
| Random legal refill of masked fields, sampled by rejection | none |
| Language model refill, Ling 3.0 Flash | none |

## Guidance timing and annealed condition noise, against the standard sampler

Source: [`guidance-timing-cads.md`](guidance-timing-cads.md)

Shared by every arm: generation-0 members without held-out leakage; ask 0.5; temperature 1; ridge tilt; first 512 validator-accepted proposals per seed; the 64 highest-ridge teams battled 49 times each against the top-50 meta teams, behaviour-cloning policy both sides; seeds 1 to 3.

| Reader name | Code ID |
| --- | --- |
| No sampler change, guidance 2 on every field | `control` |
| Species guidance 1, other fields 2 | `interval` |
| Condition noise annealed, guidance 2 on every field | `cads` |

## Species guidance 1.5, against a redrawn standard-sampler control

Source: [`guidance-timing-cads.md`](guidance-timing-cads.md)

Shared by every arm: same design as the table above, with the control redrawn from seed base 7100 (torch and numpy seeds 7101 to 7103); other fields guidance 2.

| Reader name | Code ID |
| --- | --- |
| No sampler change, guidance 2 on every field (redrawn) | `control` |
| Species guidance 1.5, other fields 2 | `species15` |

## Family-balanced training draws, against training rows drawn as stored

Source: [`flat-prior-retrain.md`](flat-prior-retrain.md)

Shared by every arm: generation-0 sampler with no ridge and no loop; classifier-free guidance 2; temperature 1; asked 0.5 and 0.7; sampling seeds 1 to 3; 64 validator-accepted teams per cell; 49 battles per team against the top-50 meta teams, behaviour-cloning policy both sides.

| Reader name | Code ID |
| --- | --- |
| No balancing: training rows drawn as stored | `fambal=0` |
| Family-balanced draws: each source family an equal share | `fambal=1` |

## Species reweighting inside the cross-entropy method loop

Source: [`species-guidance-loop.md`](species-guidance-loop.md)

Shared by every arm: clean cem_v3 loop, 512 proposals per generation at ask 0.5 and guidance 2, both members fine-tuned for 40 epochs at learning rate 1e-4, 128 teams battled 49 times each, 16 finalists battled 192 times against the top-50 meta teams; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No species reweighting, guidance 2 | `plain` |
| Species reweighting to 1.5, other fields 2 | `species15` |

## Diffusion black-box learning rule inside the cross-entropy method loop

Source: [`cem-dbbo.md`](cem-dbbo.md)

Shared by every arm: shared generation 0, guidance 2, 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No diffusion black-box rule, old sampling | `cem_v3_noheldout` |
| Diffusion black-box rule, all-data shrunk labels | `cem_dbbo` |

## Ask rule and training data inside the loop (ablation)

Source: [`cem-dbbo.md`](cem-dbbo.md)

Shared by every arm: shared generation 0, elite re-steer, 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides. Every arm except the control also uses a taboo list and a sets-only temperature of 0.5.

| Reader name | Code ID |
| --- | --- |
| No taboo list, old sampling | `cem_v3_noheldout` |
| Fixed ask 0.5, elite training | `cem_fixed_elite` |
| Moving ask, elite training | `cem_moving_elite` |
| All-data training, moving ask | `cem_dbbo` |

## Top-20% elites each generation

Source: [`cem-dbbo.md`](cem-dbbo.md)

Shared by every arm: ask 0.5, guidance 2, generation 0 shared, members fine-tuned from the original checkpoints for 10 epochs, 128 teams battled once per top-50 column, 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No top-20% elite cut, cumulative elites | `cem_v3_noheldout` |
| Top-20% elites, each generation | `cem_v3_topfrac` |

## Fixed ask level inside the elite loop, 0.5 to 0.8

Source: [`cem-dbbo.md`](cem-dbbo.md)

Shared by every arm: elite re-steer, taboo list, sets-only temperature 0.5, contrast at ask minus 0.2, shared generation 0, 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| Ask 0.5 fixed, elite training | `cem_fixed_elite` |
| Ask 0.6 fixed, elite training | none |
| Ask 0.7 fixed, elite training | none |
| Ask 0.8 fixed, elite training | none |

## Training from scratch each generation (paper-faithful diffusion black-box rule)

Source: [`cem-dbbo.md`](cem-dbbo.md)

Shared by every arm: 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides. The scratch arm also uses a taboo list and a sets-only temperature of 0.5; the control keeps the old sampling.

| Reader name | Code ID |
| --- | --- |
| No from-scratch training, old sampling | `cem_v3_noheldout` |
| Moving ask, from scratch | `cem_scratch` |

## Best-label ask weight (w) inside the loop

Source: [`cem-dbbo.md`](cem-dbbo.md)

Shared by every arm: the ask equals the weight times the best label so far, so it stays inside the labelled data; the fine-tuned arms and from-scratch arms use the same settings as the diffusion black-box rule tables above. Only the weight 1.0 arms ran, and the doc gives no figures for them.

| Reader name | Code ID |
| --- | --- |
| Fine-tuned, ask 1.0 × best label | `cem_dbbo_bestw10` |
| Scratch-trained, ask 1.0 × best label | `cem_scratch_bestw10` |

## Shrunk labels on all data, no loop (ask sweep)

Source: [`ask-dial.md`](ask-dial.md)

Shared by every arm: source file ask_sweep_dbbo.json; asks 0.1 to 0.9, 64 valid teams per ask, 49 battles each, guidance 2, species sampled at temperature 1 and other fields at 0.5, no ridge guidance; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No shrunk labels, original members | none |
| Shrunk labels, all data, generation 3 members | none |

## Precise labels at guidance 2

Source: [`ask-dial.md`](ask-dial.md)

Shared by every arm: 1,105 matrix teams re-battled to about 196 battles each; asks 0.1 to 0.9, 64 teams per ask, 49 battles each, no ridge guidance; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No precise labels, guidance 2 | none |
| Precise labels, guidance 2 | `ask_sweep_precise.json` |

## Precise labels at guidance 4

Source: [`ask-dial.md`](ask-dial.md)

Shared by every arm: source file ask_sweep_g4.json; all three models at guidance 4; asks 0.1 to 0.9, 64 teams per ask, 49 battles each, no ridge guidance; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No precise labels, guidance 4 | none |
| Precise labels, guidance 4 | none |
| Generation-3 diffusion black-box members, guidance 4 | none |

## Ridge push off inside the loop

Source: [`ask-dial.md`](ask-dial.md)

Shared by every arm: elite re-steer loop, fixed ask 0.5 with contrast 0.3, sets-only temperature 0.5, taboo list, shared generation 0; the ridge still picks the 128 battled teams; 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides. The ridge push is the tilt the ridge applies inside the decode.

| Reader name | Code ID |
| --- | --- |
| No push removal, ridge picks | `cem_fixed_elite` |
| Push removed, ridge picks | `cem_fixed_elite_notilt` |
| Random picks, push removed | `cem_fixed_elite_random` |

## Precise labels inside the loop, ridge push off

Source: [`ask-dial.md`](ask-dial.md)

Shared by every arm: ridge push off and the ridge still picks; fixed ask 0.5 with contrast 0.3; sets-only temperature 0.5; taboo list; 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| Noisy labels, original members | `cem_fixed_elite_notilt` |
| Precise labels, fine-tuned members | `cem_precise_notilt` |

## Full-range training with fixed or moving ask

Source: [`ask-dial.md`](ask-dial.md)

Shared by every arm: both full-range arms retrain from random weights every generation on the full range of labels; 16 finalists battled 192 times against the top-50 meta teams and once against ranks 51–100; behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No full-range training, old sampling | `cem_v3_noheldout` |
| Fixed ask 0.5, full-range training | `cem_scratch_fixed` |
| Moving ask, full-range training | `cem_scratch` |

## Mercury at four effort levels under the Mask Lab rules prompt, against Jev's top pick

Source: [`jev-measurements.md`](jev-measurements.md)

Shared by every arm: held-out grid of 270 masks built from five unmasked teams (unmasked win rate 47.6%); shuffled re-battle, six Pokemon shuffled each battle, 8 battles per opponent against the 49 top-50 meta teams; behaviour-cloning policy on both sides; usable win rate counts a mask with no legal team as zero; intervals are 95%.

| Reader name | Code ID |
| --- | --- |
| Jev top pick, no prompt | none |
| Rules-prompt Mercury, off effort | none |
| Rules-prompt Mercury, low effort | none |
| Rules-prompt Mercury, medium effort | none |
| Rules-prompt Mercury, high effort | none |
| Random legal fill, no prompt | none |

## Mercury at four effort levels under the tuned prompt, against Jev's top pick

Source: [`jev-measurements.md`](jev-measurements.md)

Shared by every arm: same held-out grid, shuffled re-battle and opponents as the table above; the tuned prompt is the instruction text rewritten by GEPA, DSPy's reflective prompt optimizer, with GPT-6 Luna; usable win rate counts a mask with no legal team as zero; intervals are 95%.

| Reader name | Code ID |
| --- | --- |
| Jev top pick, no prompt | none |
| Tuned-prompt Mercury, off effort | none |
| Tuned-prompt Mercury, low effort | none |
| Tuned-prompt Mercury, medium effort | none |
| Tuned-prompt Mercury, high effort | none |

## Diversity of species written into the held-out grid

Source: [`jev-measurements.md`](jev-measurements.md)

Shared by every arm: legal fills only on the 270-mask held-out grid; effective values equal exp(Shannon entropy) of the species frequencies; DSPy is the prompt-optimization framework that wrote the tuned prompt. Verdicts are against random legal fill, and the arms wrote different counts (105, 105, 36 and 76), so the same-masks figures in the doc are the fair comparison.

| Reader name | Code ID |
| --- | --- |
| Random legal fill, no prompt | none |
| Jev top pick, no prompt | none |
| Rules-prompt Mercury, high effort | none |
| Tuned-prompt Mercury, high effort | none |

## Selection temperature schedule for finite-pool resampling

Source: [`temperature-experiment.md`](temperature-experiment.md)

Shared by every arm: shared checkpoint, anchor and generation-0 labels, corpus, behaviour-cloning policy, 50-opponent pool, guidance strength 108, decoding temperature 1.0, entropy bonus 0, 40 epochs, 11 generations of 512 proposals with 128 labels battled 24 times each, seeds 101, 202 and 303.

| Reader name | Code ID |
| --- | --- |
| No annealing, fixed 0.10 | none |
| Annealing 0.30 to 0.10 | none |
| Fixed 0.20 | none |
| Fixed 0.30 | none |
| Adaptive, effective sample size 0.50 | none |

## Reasoning on versus off, Ling

Source: [`reasoning-experiment.md`](reasoning-experiment.md)

Shared by every arm: frozen prompts, starting teams, masks and opponent pool; output cap of 16,384 tokens for every arm; three independent generation requests per task per arm; behaviour-cloning policy on both sides. The doc reports no figures.

| Reader name | Code ID |
| --- | --- |
| No reasoning, Ling | `off` |
| Reasoning on, Ling | `on` |

## Graded reasoning effort, Mercury

Source: [`reasoning-experiment.md`](reasoning-experiment.md)

Shared by every arm: frozen prompts, starting teams, masks and opponent pool; Mercury as model `inception/mercury-2.5` through the Inception provider; output cap of 16,384 tokens for every arm; three independent generation requests per task per arm; behaviour-cloning policy on both sides. The doc reports no figures.

| Reader name | Code ID |
| --- | --- |
| Effort off (no reasoning), Mercury | `off` |
| Effort low, Mercury | `low` |
| Effort medium, Mercury | `medium` |
| Effort high, Mercury | `high` |

## Diffusion black-box optimisation port: label precision and ask multiplier

Source: [`dbbo-faithful.md`](dbbo-faithful.md)

Shared by every arm: each round trains a diffusion model from random weights on the labelled set; learning rate 1e-3, batch 256, label dropout 0.15 (classifier-free guidance, scale 2); 4 rounds × 128 teams × 49 battles; finalists 16 × 192 battles vs top-50, then vs ranks 51–100; ask = w × best label so far.

| Reader name | Code ID |
| --- | --- |
| Raw labels, w = 1 | `dbbo_faithful` |
| Precise labels, w = 1 | `dbbo_faithful_precise` |
| Uncertainty-picked w, precise labels | `dbbo_paper` |
| All five asks battled each round | `dbbo_allw` |
| Full-matrix start, shrunk labels | `dbbo_fix_data` |
| 90th-percentile ask, raw labels | `dbbo_fix_ask` |
| Surrogate-ranked battles, raw labels | `dbbo_fix_pick` |
| All three fixes, raw labels | `dbbo_fix_all` |

## Placement teams vs random teams: sensitivity to single-move edits

Source: [`smoothness-experiment.md`](smoothness-experiment.md)

Shared by every arm: 50 original teams per group, eight single-move edits each, an unchanged-team control per original; four panels of 100 battles per team, each cycling the same 50 top-50 opponents twice; fixed behaviour-cloning policy on both sides; sensitivity in squared percentage points.

| Reader name | Code ID |
| --- | --- |
| Edited placement teams | none |
| Edited random-legal teams | none |
| Unedited placement teams | none |
| Unedited random-legal teams | none |

## Archive refinement (the SPARROW method): parent selection and corruption, against unconditional sampling

Source: [`sparrow-port.md`](sparrow-port.md)

Shared by every arm: frozen v3 diffusion generator with whole-Pokémon re-masking and no difference step; 1,045 steps per run, each step charged 24 battles vs top-50; three seeds per arm; finalists 16 × 192 battles vs top-50, then vs ranks 51–100.

| Reader name | Code ID |
| --- | --- |
| Gentler-corruption, near-uniform parents | `S(5, 0.5)` |
| Full-corruption, near-uniform parents | `S(5, 1)` |
| Full-corruption, elite parents (primary arm) | `S(50, 1)` |
| Gentler-corruption, elite parents | `S(50, 0.5)` |
| No archive, every slot resampled | none |
| Cross-entropy loop, reference only | `cem_v3_noheldout` |

## Structured output, training data and unmasking order, against the old model

Source: [`loss-down.md`](loss-down.md)

Shared by every arm: the same 500 held-out test teams for every row; losses in nats per field, paired against the old model; loss is the evidence lower bound unless marked exact; exact = loss under the sampler's reveal order; own = bound under each run's own unmasking order.

| Reader name | Code ID |
| --- | --- |
| No structured output, old model | `asked_vs_got.pt` |
| Structured output at evaluation only | none |
| All 145k teams, structured output | `A1` |
| Source tags added, all teams | `B1` |
| Family fine-tune after pretraining | `D1` |
| Forced masks, family teams only | `B2` |
| No forced masks, family teams | `B3` |
| No time input, no forced masks | `N1` |
| Wider model, forced masks | `C2` |
| Partial masking, two sub-tokens per field | `P1` |
| Species-first unmasking order | `S1` |
| Medium block unmasking order | `S5` |
| Sharp block unmasking order | `S7` |
| Popular moves first, medium block order | `MO1` |
| Loop teams only, sharp block order (the loop-teams model) | `F1` |
| Seed 1 repeat of the loop-teams model | `F2` |
| Seed 2 repeat of the loop-teams model | `F3` |
| Four-model mixture: three loop-teams seeds and the sharp block model | `F1+F2+F3+S7` |
| All teams and matrix rows, six epochs | `G0` |
| Loop fine-tune after pretraining | `G1` |
| Matrix rows conditioned, no pretraining (the matrix-rows model) | `G2` |
| Two-model mixture: loop-teams and matrix-rows models | `FINAL` |
| Three-model mixture: adds the loop fine-tune model | `F1+G2+G1` |

## Matchup-matrix row as a condition

Source: [`loss-down.md`](loss-down.md) (`M1`)

Shared by every arm: one model (all data, structured output, uniform schedule, 5 epochs), scored on 1,000 matrix teams with at least 40 tested columns that it never trained on, on the same masks; the arms differ only in the condition given at scoring.

| Reader name | Code ID |
| --- | --- |
| No condition | none |
| Win-rate condition only | none |
| Win-rate and matchup-row condition | none |

## Diffusion model sampled from scratch against a real team

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: win rate against the top-50 meta (the 50 top-placing Regulation M-B teams, 2026-08-21 snapshot), behaviour-cloning policy on both sides; 2,048 battles per arm (16 teams × 128 battles), errors clustered by team.

| Reader name | Code ID |
| --- | --- |
| Diffusion model sampled from scratch | none |
| No diffusion model, real corpus team | none |

## Win-rate-bin conditioning on the labelled pool

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: 4,850-team labelled pool, 24 battles per team, win rate against the top-50 meta, behaviour-cloning policy on both sides; the control is unconditioned sampling from the same model.

| Reader name | Code ID |
| --- | --- |
| No win-rate-bin conditioning | none |
| Weight-4 top-bin conditioning | none |
| Bottom-bin conditioning | none |

## Win-rate-bin conditioning on the uniform-legal pool

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: uniform-legal pool, 32 teams × 24 battles per cell (768 battles), win rate against the top-50 meta, behaviour-cloning policy on both sides; real-team reference 0.491 over 768 battles.

| Reader name | Code ID |
| --- | --- |
| No win-rate-bin conditioning | none |
| Weight-1 top-bin conditioning | none |
| Weight-2 top-bin conditioning | none |
| Weight-4 top-bin conditioning | none |

## Reweighted training toward high scores

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: balanced win-rate brackets, 150 teams × 24 battles per arm (3,600 battles), win rate against the top-50 meta, behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| No reweighting, original training | none |
| Reweighted training, balanced brackets | none |

## Uncertainty labelling against random labelling

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: the metric is the Spearman rank correlation between the surrogate and measured win rate on 300 held-out teams; labels are win rates against the top-50 meta, 24 battles each; same battle budget per arm.

| Reader name | Code ID |
| --- | --- |
| No selection, random labelling | none |
| Uncertainty labelling, predictive variance plus max-min diversity | none |

## Trust-region noise search against random noise search and best-of-N

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: matched budgets of about 48,000 battles per arm; top-8 mean of re-evaluated win rates against the top-50 meta, behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| Trust region, re-centered noise search | none |
| No trust region, random noise search | none |
| Best-of-N prior sampling, no noise search | none |

## Annealed temperature for Boltzmann selection, 2026-09-07 runs

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: fresh-generator win rate against the top-50 meta, behaviour-cloning policy on both sides; 128 samples × 192 battles per run; 73,728 battles per arm over 3 seeds; no arm adds an elite cut.

| Reader name | Code ID |
| --- | --- |
| No annealing, 0.10 fixed temperature | none |
| Annealed temperature, 0.30 down to 0.10 | none |
| 0.20 fixed temperature | none |
| 0.30 fixed temperature | none |
| Adaptive temperature, set per run | none |

## Boltzmann weights and other changes against the hard elite cut

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: difference against the hard elite cut (retrain on the top teams only), win rate against the top-50 meta, behaviour-cloning policy on both sides; 11 generations × 128 teams × 24 battles per arm unless noted.

| Reader name | Code ID |
| --- | --- |
| Boltzmann weights in place of the hard cut | none |
| One representative per species set | none |
| Entropy bonus inside fine-tuning | none |

## Language-model-written filter rules against the hand-mined rule

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: rules keep or prune teams without battles; meta retention is the share of 649 real Regulation M-B teams a rule keeps, space retention the share of 20,000 uniform-legal teams it keeps, lift their ratio.

| Reader name | Code ID |
| --- | --- |
| Hand-mined three-Protect rule | none |
| Two-Protect language-model rule, same-type attack | `two-protect-and-stab` |
| Disruption language-model rule | none |
| Common-items language-model rule | none |

## Partial-noise edits of real teams

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: win rate against the top-50 meta, behaviour-cloning policy on both sides; 2,048 battles per arm (16 teams × 128), errors clustered by team; 12,288 battles in all.

| Reader name | Code ID |
| --- | --- |
| No edit, untouched real team | none |
| Slot copy from a real team | none |
| Regenerated candidate, diffusion model | none |
| Steered edit, toward playstyle | none |
| Shuffled Stat Points, within species | none |

## Self-consuming retraining loop against the uniform-legal pool

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: the loop adds every new legal team back into the training data; the metric is reach beyond the uniform-legal pool at a calibrated Hamming radius; no battles run.

| Reader name | Code ID |
| --- | --- |
| Retraining loop, generation 2 | none |

## Cross-entropy search: proposer and elite fraction

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: cross-entropy method (Rubinstein 1999), 200 proposals × 24 battles per round over 3 rounds, win rate against the top-50 meta, behaviour-cloning policy on both sides.

| Reader name | Code ID |
| --- | --- |
| Diffusion proposer, standard elite cut | none |
| Copy-paste proposer, standard elite cut | none |
| Copy-paste proposer, 1% elite cut | none |

## Gradient guidance at decode time

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: 128 teams × 24 battles per single-round arm (3,072 battles), win rate against the top-50 meta, behaviour-cloning policy on both sides; the combined-loop row comes from the loop run (73,728 battles).

| Reader name | Code ID |
| --- | --- |
| No guidance, unguided sampling | none |
| Naive gradient guidance | none |
| Look-ahead gradient guidance | none |
| Full gradient guidance with Jacobian term | none |
| Gradient guidance in the combined loop | none |

## Active search against random labelling

Source: [`failed-methods.md`](failed-methods.md)

Shared by every arm: the difference is over random labelling at the same battle budget (44,736 battles, 24 per team), as the doc reports it.

| Reader name | Code ID |
| --- | --- |
| No selection, random labelling | none |
| Predicted-value labelling, active search | none |
| Margin scoring added to active search | none |

## Reply format with typed fields, before any run

Source: [`dspy-structure-experiment.md`](dspy-structure-experiment.md)

Shared by every arm: Ling 3.0 Flash with reasoning off and on, and Mercury 2.5 at reasoning off, low, medium and high; 20 random legal starters × 7 masks × 3 repeats = 420 requests per setting; typed fills; behaviour-cloned battle policy on both sides; 50-battle panel. The resampled original prompt is an optional control, not a reply format.

| Reader name | Code ID |
| --- | --- |
| Field-marker reply, typed fields | `chat` |
| JSON-object reply, typed fields | `json` |
| No structured prompt, original prompt resampled | `--resample-baseline` |

## Reply format and answer types on Mercury 2.5, legal choices

Source: [`dspy-structure-experiment.md`](dspy-structure-experiment.md)

Shared by every arm: Mercury 2.5 at four reasoning settings; 20 starters × 7 masks × 3 repeats; legal-choice answer types in every arm; each difference is structured minus original; legality is the share of answers that pass the rules.

| Reader name | Code ID |
| --- | --- |
| Field-marker reply format | `chat` |
| JSON-object reply format | `json` |
| No structured prompt, original prompt resampled | `--resample-baseline` |

## Optimized instructions against unoptimized legal-choice types

Source: [`dspy-structure-experiment.md`](dspy-structure-experiment.md)

Shared by every arm: Mercury 2.5 writes the instructions with GEPA, DSPy's reflective prompt optimizer, at its light budget at reasoning off; control is the unoptimized legal-choice types (`dspy-legal-choices-20261015`, reference row above); held-out legality over 20 starters, 1,680 calls.

| Reader name | Code ID |
| --- | --- |
| Optimized instructions, Mercury writes | `dspy-optimize-20261015` |

## Reflection model that writes the optimized instructions

Source: [`dspy-structure-experiment.md`](dspy-structure-experiment.md)

Shared by every arm: GEPA light budget, reasoning off for the optimizer; same 140 tasks; held-out legality over 20 starters (1,680 calls each); Mercury 2.5 answers in every row; pooled legality over all four settings.

| Reader name | Code ID |
| --- | --- |
| No second model, Mercury writes | `dspy-optimize-20261015` |
| Hidden answer shape, GPT-6 Luna writes | none |
| Shown answer shape, GPT-6 Luna writes | `dspy-optimize-luna4-20261016` |

## Frozen DSPy prompt against plain prompt on the held-out Mask Lab grid

Source: [`dspy-structure-experiment.md`](dspy-structure-experiment.md)

Shared by every arm: reasoning high; 270 masks on the held-out grid; win rate against the top-50 meta in the shuffled re-battle, where an illegal fill scores 0; both prompts use the same 48 meta teams; the same-masks lead is over masks both sides filled legally.

| Reader name | Code ID |
| --- | --- |
| Typed fields, optimized prompt | `dspy-optimize-luna4-20261016` |
| No typed fields, plain prompt | none |
| Jev top pick, decision model | none |

## Trust-region noise search against random noise search and best-of-N

Source: [`trs-port.md`](trs-port.md)

Shared by every arm: all new arms share the same opponent schedules, battles per call and promotion rule; each arm's top-8 mean of 192 fresh battles against the top-50 meta, behaviour-cloning policy on both sides; standing rows are quoted from earlier runs, not re-measured.

| Reader name | Code ID |
| --- | --- |
| Trust region, re-centered noise search | `TRS` |
| No trust region, random noise search | none |
| Best-of-N prior sampling, no noise search | none |
| Slot copy from a real team | `graft` |
| Diffusion model sampled from scratch | `invent` |
| Gradient guidance in the combined loop | none |
| Elite fraction 0.01, top teams kept | none |

## Region-side sweep of the trust region, decode only

Source: [`trs-port.md`](trs-port.md)

Shared by every arm: zero battles; 8 centers drawn from the prior, 64 masked perturbations per size; fields counted over the 48 decoded columns; pre-registered rule: the smallest size at which at least half the proposals differ from the center.

| Reader name | Code ID |
| --- | --- |
| 0.05 region side, smallest tested | none |
| 0.10 region side | none |
| 0.20 region side | none |
| 0.40 region side | none |
| 0.80 region side | none |
| 1.60 region side | none |
| 2.40 region side, the whole box | none |

## Repo decoding, elite filter, and guidance runs

Source: [`discrete-diffusion-plan-review.md`](discrete-diffusion-plan-review.md)

Shared by every arm: the repo's own discrete diffusion model on the team task; each row changes one setting against the same starting setup; win rate = share of battles won against the top-50 meta teams.

| Reader name | Code ID |
| --- | --- |
| Changed unmasking order | none |
| Temperature 0.7 and top-p 0.9 decoding | none |
| Elite filter rate lowered to 0.01 from 0.20 | none |
| Classifier-free guidance strength 4 | none |

## Model and ask win rate, one-shot team writing

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: 64 valid teams per arm, each battled once against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, seed 2026; "ask" = win rate written into the prompt; guidance = classifier-free guidance strength; scratch = trained only on this project's teams; the control is the scratch model with no ask.

| Reader name | Code ID |
| --- | --- |
| Unconditioned scratch diffusion model | `scratch:base` |
| Ask-0.50 guided scratch diffusion model | `scratch:cfg` |
| Untuned pretrained text diffusion model | `text:untouched` |
| Unconditioned fine-tuned text diffusion model | `text:base` |
| Ask-0.50 fine-tuned text diffusion model | `text:ask50` |
| Ask-0.60 fine-tuned text diffusion model | `text:ask60` |
| Guided ask-0.50 fine-tuned text diffusion model | `text:ask50_g2` |

## Proposer model inside the selection stack

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: 512-team pool, 64 teams picked by a surrogate (ridge regression) and battled against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, seed 2026; protect filter = at least three Pokémon with a Protect-family move.

| Reader name | Code ID |
| --- | --- |
| Protect-filtered pool, surrogate pick | `text:stack` |
| Unfiltered pool, surrogate pick | `text:stack_nofilter` |
| Scratch full search stack | `v3:stack` |

## Fill method on five random starting teams

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: 270 masks on five random legal starting teams, one fill per mask, no repair, each legal fill battled once against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides; "net" keeps the original team when a fill is rejected; net differences in percentage points of win rate.

| Reader name | Code ID |
| --- | --- |
| No fill, original starting team | none |
| Random legal fill | none |
| Scratch diffusion model fill, constrained | `scratch_decoder` |
| Text diffusion model fill, fine-tuned, unconstrained | none |

## Training speed settings on a laptop

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: batch 4, same batches and loss, bfloat16, Apple M4 Pro laptop; rows are cumulative (each adds one change to the row above); torch.compile = PyTorch's compiler, with dynamic shapes on the Metal backend.

| Reader name | Code ID |
| --- | --- |
| As-run released layout, full checkpointing | none |
| No gradient checkpointing | none |
| Shared clean-token layout | none |
| Attention without key-value copy | none |
| Eight layers left unchecked | none |
| Compiled, twenty checkpointed layers | none |
| Compiled, full checkpointing | none |

## Training speed and model size at equal steps and equal time

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: the same first 2,000 training examples in one order; 500 steps of batch 4 under the full 2,000-step learning-rate schedule (equal-time rows: 60 minutes); validation loss per answer token on the same 256 held-out family rows (recorded run 0.9487); one seed.

| Reader name | Code ID |
| --- | --- |
| Current trainer, no changes | `control` |
| No system message, short prompt | none |
| One token per domain name | none |
| Short prompt plus domain tokens | none |
| Eight middle layers removed | none |
| Eight-layer cut with short prompt | none |
| Domain tokens, eight-layer cut, short prompt | none |
| Fourteen-layer adapters, top only | none |
| Seven-layer adapters, top only | none |
| One mask per example | none |
| Sixteen middle layers removed | none |
| Smaller pretrained model, 500 steps | none |
| Smaller 0.5B model, sixty minutes | none |
| Twelve-layer model, sixty minutes | none |
| Twenty-layer model, stopped at 45 minutes | none |

## Fill method against the Ling language model on the same 270 masks

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: the same 270 masks and five random legal starting teams as the fill-method table, one fill per mask, no repair, each legal fill battled once against the 49 distinct top-50 meta teams; Ling = a large language model (Ling) called through OpenRouter at temperature 0.7; "regulation reference" = the Regulation M-B rules in the prompt; "net" keeps the original team when a fill is rejected.

| Reader name | Code ID |
| --- | --- |
| No fill, original starting team | none |
| Text diffusion model fill, fine-tuned | `lora_r32/final` |
| 16,384-token Ling, regulation reference | `regmb@16384` |
| 16,384-token Ling, no regulation reference | `baseline@16384` |
| 4,096-token Ling, no regulation reference | `baseline@4096` |

## Opponent set: held-out 50 against top 50, same fills

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: the fills from the previous Ling table, each battled once against ranks 51–100 (the held-out teams) and once against the top 50; result = held-out win rate minus top-50 win rate, in percentage points, paired on the same fill.

| Reader name | Code ID |
| --- | --- |
| No fill, original starting team | none |
| Text diffusion model fill, fine-tuned | `lora_r32/final` |
| Scratch diffusion model fill, constrained | `scratch_decoder` |
| Random legal fill | none |
| 16,384-token Ling, regulation reference | `regmb@16384` |
| 16,384-token Ling, no regulation reference | `baseline@16384` |
| 4,096-token Ling, no regulation reference | `baseline@4096` |

## Fill method on held-out starting teams (ranks 51–100)

Source: [`textdiff.md`](textdiff.md)

Shared by every arm: five real teams from ranks 51–100 as starting teams, 270 masks, one fill per mask, each legal fill battled once against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides; the untouched originals average a win rate of 0.480; "net" keeps the original when a fill is rejected.

| Reader name | Code ID |
| --- | --- |
| No fill, original starting team | none |
| Text diffusion model fill, fine-tuned | `lora_r32/final` |
| Scratch diffusion model fill, constrained | `scratch_decoder` |
| Random legal fill | none |
