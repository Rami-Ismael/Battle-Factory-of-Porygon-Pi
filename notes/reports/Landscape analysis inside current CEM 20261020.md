# Your one-edit evidence inside neural CEM

Checked October 20, 2026 against the saved analyses and the current code repository. This explanation proposes ways to use landscape information. It does not implement a new search rule, train a model, run battles, or choose a new evaluation budget.

The useful connection is to test which legal changes are worth proposing near the teams your search currently visits. Your existing studies support treating edit type and base-team region separately. They do not establish that every strong team has an improving one-edit neighbor or that a local-search branch will beat another batch of ordinary proposals.

## What your existing analysis actually measured

There are two studies to keep separate.

| Study | Neighborhood and region | Recorded result | What it supports |
| --- | --- | --- | --- |
| September 28 ruggedness study | Replace one complete Pokémon set, including species, item, ability, moves and stat spread. Separate meta-team and search-team anchors. | Whole-set noise-adjusted RMS change was about 5.535 percentage points around meta anchors and 7.364 around search anchors. Its normalized increment statistic was about 0.8606 and 0.7534 respectively. The saved run used 36 actual opponents. | A whole-set edit can produce a material change while retaining some relationship to its starting team. The increment statistic is not an observed Pearson correlation or a probability that a mutation improves. |
| October 1 full smoothness study | One legal move replacement, eight edits each around 50 HPS-corpus-sampled originals and 50 frozen placement entries, using four evaluation panels. | Mean noise-corrected squared change was 0.292 percentage-points squared for the random group and 18.870 for placements. Placement minus random was 18.579, with the saved 95% interval [12.698, 25.262]. | Move-edit sensitivity differs between these sampled regions. A random-team average is not a description of every region reached by CEM. |

These are different units, operators and protocols. Do not compare the magnitude 18.870 directly with 5.535. The older whole-set analysis also used clipping in its correction; the later move analysis retains negative individual noise-corrected estimates. Its newer statistic must not be silently replaced by clipped values or square roots. Exact provenance, coverage and verification limits are in the [one-edit evidence audit](../research_notes/Landscape%20analysis%20inside%20CEM%2020261020/one-edit-evidence.md).

The random originals were sampled from the frozen HPS corpus, not uniformly from the entire legal team space. Their mean original win rate was 1.845%, versus 49.950% for the placement cohort. That outcome floor partly confounds a general interpretation of quality versus sensitivity. The placement snapshot includes six ranked-season and four ladder entries retained by the owner, so it is not strictly tournament-only. Its 50 opponent entries represent 49 distinct identities with the original duplicate weight preserved.

The saved cluster intervals for the two corrected means are [0.000823, 0.690438] and [12.924148, 25.510712] percentage-points squared. Both unchanged-team-control intervals include zero. These intervals condition on the sampled teams, sampled edits, frozen policy and opponent pool; they do not cover transfer to a new policy or CEM proposal distribution.

The later study also corrects the broad earlier statement that move edits barely matter. For example, the first frozen placement parent had an observed mean of 53.00%. Replacing Charizard's Heat Wave with Rest produced 40.75%, a difference of -12.25 percentage points with a nominal pointwise interval [-19.00, -5.34]. These are saved observations from that evaluator, not a prediction for a current CEM team. They demonstrate a possible large move effect; selecting the first examples does not estimate the prevalence of beneficial or harmful mutations.

### Why repeated panels matter

For each move edge, the study has four panel differences d1 through d4 between edited and base teams. The saved estimator is:

```text
corrected squared change = mean(d)^2 - sample_variance(d) / 4
                        = mean of the six cross-panel products d_j * d_k
```

With independent panels estimating the same difference, cross-panel products separate a repeatable signal from within-panel randomness. An individual estimate can be negative because it is an estimate of a nonnegative quantity. Clipping every negative estimate to zero would introduce a different estimator.

This addresses effect magnitude. Squaring removes direction. An edit that loses ten percentage points contributes the same squared change as one that gains ten. Before choosing a local proposal rule, you also need signed changes, reliable comparisons, and where the parents came from.

The audit inspected saved analysis, manifests and completion checks, and spot-checked raw panel observations. It did not freshly recompute the whole 400,000-battle study. The old teaching prototype with invented values is not a source for these measurements.

## Separate the objective, the ridge and the generator

For fixed rules, opponents, policies and ordering conventions, let f(T) be a legal team's expected win rate. A neighborhood specifies which changes make a team adjacent to T. Together, the candidates, objective and neighborhood define the landscape.

Your search also contains two learned objects with different jobs:

- The ridge predicts scores and helps choose candidates to evaluate.
- The neural generator q_t determines which candidates are proposed at generation t.

Changing q_t changes where you look. It does not make a fixed f smoother. A generation whose scores have become similar might indicate a concentrated generator, a locally insensitive region, noisy labels, or acquisition that discards other regions. One plot of generation scores cannot distinguish those explanations.

Likewise, nearby teams having similar true win rates does not establish that the ridge ranks their small differences correctly. The generator can represent dependencies without learning the useful dependencies from its selected training pool. These mechanisms need separate evidence.

## The specific loop where the information can enter

The code audit found several saved configurations. The following describes the clean CEM-v3 baseline with full proposal texts, not every later DBBO ablation.

1. Form the training pool from anchors and previously battled candidates. The default takes raw labels at least 0.531, with a top-64 fallback if too few qualify.
2. For generations after zero, fine-tune copies of the original neural checkpoints on that pool. The baseline uses 40 epochs and learning rate 0.0001. It resets to the original checkpoints each generation rather than continuing the preceding generation's weights.
3. Refit the ridge using the base data and prior loop labels with their declared weights.
4. Generate 512 legal candidates under the recorded decoding settings.
5. Acquire the 128 highest ridge predictions for battle evaluation.
6. Top up missing evaluator-cache cells, record outcomes, apply the existing stopping checks, and repeat.
7. Select finalists before fresh evaluation. The saved baseline evaluates 16 finalists with 192 fresh battles each and reports a separate held-out-opponent evaluation.

The 128 acquired candidates are not a top-128 elite-training rule. Your cumulative 0.531 training-pool rule is a different selection stage. The previous-generation top-fraction variant and later full-512 battle variants must be named separately.

There is also no tabular CEM frequency vector or implemented alpha blending parameter to turn down. In your code, training-pool membership, neural fitting strength, decoder settings and proposal source are the relevant controls. A mixture of samples from two models would be a new proposal mechanism, not neural-weight averaging. [Implementation audit](../research_notes/Landscape%20analysis%20inside%20CEM%2020261020/current-cem.md), [proposal and acquisition code](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/cem_v3.py:223), [neural refit code](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/cem_v3.py:113).

Classical CEM does have smoothed probability updates, but its deterministic convergence analysis does not prove convergence for this noisy neural loop. [Costa, Jones and Kroese, primary paper](https://people.smp.uq.edu.au/DirkKroese/ps/CEconv.pdf).

### A diagnostic we can already calculate from your saved run

The following values were recomputed read-only from `results/cem_v3_noheldout.json`. Each generation proposed 512 teams and acquired 128. Acquired composition counts were checked directly against the six saved team headers. Mean scores and above-threshold counts were recalculated from the stored labels.

| Generation | Actual acquired compositions, out of 128 | Expected compositions in a uniform 128-row subset of the same proposals | Mean observed search score | Labels at least 0.531 | Recorded training-pool size |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 60 | 51.778 | 45.7813% | 35 | Original model, no refit |
| 1 | 22 | 48.205 | 50.5625% | 52 | 64 |
| 2 | 21 | 37.675 | 50.6406% | 52 | 105 |
| 3 | 14 | 26.271 | 52.7969% | 64 | 157 |

There were zero literal full-paste repeats within these proposal batches or against their earlier proposal batches. Yet the acquired species compositions narrowed from 60 to 14. This shows why exact duplicates and composition concentration are different diagnostics. It does not show that the concentration was harmful; the win-rate comparison still decides that.

The uniform-subset column controls sample size. If composition c appears f_c times in the 512 saved proposals, its chance of appearing at least once in a uniform subset of 128 rows is 1 - C(512-f_c,128)/C(512,128). Summing those probabilities gives the exact expected count, without new battles or Monte Carlo sampling. Acquisition is more composition-concentrated than this reference in generations 1 through 3, while generation zero has the opposite result. This describes the selection mechanism; it does not predict the win rate of a hypothetical random-acquisition arm or establish that concentration is harmful.

The score column contains cached search labels, not independent finalist results or a confidence interval. Training-pool sizes include cumulative history. Historical proposal-composition counts contain a reproducible item-less-header parsing issue, so the counts here were read from all six saved block headers. No parser or experiment result was changed. [Calculation details and schema keys](../research_notes/Landscape%20analysis%20inside%20CEM%2020261020/current-cem.md).

## First use landscape analysis without changing the optimizer

Analyse stages separately. Preserve missing outcomes for candidates that were never battled.

| Stage | Useful observation | Interpretation limit |
| --- | --- | --- |
| All proposals | Exact repeats, species-composition counts, and distances to earlier eligible teams. | Literal duplicates and repeated species compositions are different. A species composition omits moves, items, abilities and stat spreads. |
| Ridge acquisition | Compare the proposed regions with the regions reaching battle evaluation. | A broad proposal batch can become narrow after acquisition. Unbattled ridge predictions are not measured win rates. |
| Battled candidates | Local predicted differences versus measured differences, separated by operator and parent region when recorded. | High overall ridge correlation can conceal poor ranking among similar high-quality teams. Acquired observations are selection-biased. |
| Training pool | Which regions pass the actual threshold, label precision, anchor contribution and cumulative retention. | Many labels just above a threshold can still reflect selection optimism. |
| After neural refitting | Changes in generated support compared with the pre-refit model under controlled sampling. | Lower diversity alone is not failure. Its value depends on final independently measured win rate. |

Your saved CEM-v3 files contain full proposal texts and selected battle labels, so some of this can be checked without new simulations. Later DBBO files retain species compositions for all proposals but full texts only for battled teams. Exact full-set distances for unbattled candidates cannot be reconstructed from composition-only logs.

A CEM batch also has no recorded mutation parent for each sampled team. Drawing an edge to the nearest prior candidate produces a retrospective proximity graph. It is not an observed parent-child search trajectory. The STN paper becomes directly applicable to actual local proposals once their parent and operator are logged. [STNs paper](https://www.storre.stir.ac.uk/bitstream/1893/32613/1/stns_asoc_2021.pdf).

## A concrete local-proposal comparison

Your studies motivate a question: can legal changes around eligible teams produce more useful candidates than the ordinary generator produces at the same cost? They do not answer it for current CEM-generated parents.

Use a named baseline configuration and leave generation zero unchanged. From generation one onward, let m be a declared number of local proposals in the existing batch of 512. Generate m local candidates and 512 - m ordinary candidates. Keep the same pooled top-128 ridge acquisition, training-pool rule and original-checkpoint refit. This changes locality while preserving the other main mechanisms.

```text
for each generation in the selected baseline:
    if generation == 0:
        models = original_checkpoints_without_refitting()
        local_count = 0
    else:
        eligible = build_training_pool_using_the_existing_rule()
        models = reset_and_refit_original_checkpoints(eligible)
        local_count = m
    ridge = refit_using_the_existing_label_rule()

    ordinary = propose_from_models(512 - local_count)
    local = []
    if local_count > 0:
        local = propose_legal_edits(parent_pool=eligible, count=local_count,
                                   declared_parent_rule, declared_edit_operator)
    proposals = ordinary + local

    record proposal source, parent where real, requested and actual edit size
    acquired = existing_ridge_acquisition(proposals, cap=128)
    outcomes = evaluate_within_the_applicable_total_battle_allowance(acquired)
    record actual new battles, cache provenance and observed scores
    apply existing training-pool and stopping rules

select finalists before the same independent final evaluation
```

This is proposed pseudocode, not an implementation or an allocation recommendation. The existing generator's legality, held-out exclusions and decoding conditions must still apply. A fixed mask size does not guarantee the same actual edit size: unchanged regenerated fields and legality repair can alter the result.

Share the selected baseline's proposal-attempt cap across both sources. Local rejection, no-op edits and repairs cannot receive unlimited extra retries. Record attempts and any shortfall as well as retained candidates. Generation zero remains unchanged, so an existing stop rule relative to generation-zero diversity keeps the same reference.

A local operator might replace one complete set or change a single move while preserving the rest. Those are separate operator comparisons. Do not combine a new operator, stronger guidance, a new elite cut and an adaptive restart rule in the first test.

Retaining pooled ridge acquisition means some or all local proposals might be discarded before battle evaluation. Log that survival rate. Adding a source-specific quota would be another change with a different question.

For each real parent and operator, retain validity, acquisition and battle counts, signed child-minus-parent differences, comparison uncertainty, and whether the child meets the actual training-pool threshold. Include failed or unacquired proposals in the appropriate denominators. A large squared effect alone cannot tell you which operator supplies useful candidates.

Equal batch sizes do not alone guarantee equal evaluation cost. Local repeats can hit cached scores; fresh comparison panels consume new battles. Count actual new battles, retained historical observations and elapsed time separately. Any additional parent checks or stagnation checks must come out of the applicable allocation. No new value of m, budget or adaptation threshold is chosen here.

The CE-memetic paper provides a discrete precedent for bounded local refinement with an explicit probability and evaluation cap. Its deterministic tool-switching results do not establish the right allocation or a VGC benefit. [Amaya, Cotta and Fernández-Leiva, section 3](https://link.springer.com/content/pdf/10.1080/18756891.2013.792542.pdf).

## Let different observations lead to different changes

Your separate four-edit interaction study already supplies a limited clue. On its sampled cubes, the noise-corrected variance shares were about 87.13% single effects, 11.60% pair effects and 1.263% order-three-or-higher effects. The interval for the last share included zero. Those cubes changed four whole sets and used the historical 36-opponent setting. They support investigating pair-aware edits on that sample; they do not prove global independence or eliminate every higher-order interaction. [Interaction source keys and scope](../research_notes/Landscape%20analysis%20inside%20CEM%2020261020/one-edit-evidence.md).

| Supported observation near current candidates | Candidate change to compare separately |
| --- | --- |
| Useful, repeatable one-edit gains | A bounded local proposal branch around eligible parents. |
| Stronger joint edits than their separate effects | Conditional or grouped edits that preserve the relevant context. One interaction does not prove a globally decomposable objective. |
| Proposal support narrows sharply after refitting | A proposal-level restart or original-generator mixture, or a fitting-strength comparison. Diversity remains a diagnostic. |
| Local ridge improvements disappear in battles | Audit local ranking and feature coverage before increasing guidance. The current ridge does not model the externally drawn EV spread directly. |
| Training membership changes under repeated evaluation | Study selection stability under the existing threshold, then test a separately specified uncertainty-aware rule if warranted. |

The existing species-guidance intervention is an instructive example: a wider initial proposal population did not reliably preserve its advantage after elite refitting. Its fresh finalist difference was inconclusive. It should not be presented as an untested fix for collapse.

There is also an existing higher-cut ablation, so changing the threshold is not a new landscape proposal. The saved comparison reports a positive fresh top-50 difference for cut 0.56 under its own full-512, three-seed protocol. That does not prove owner adoption or make it the baseline for this explanation. [Recorded interventions and source keys](../research_notes/Landscape%20analysis%20inside%20CEM%2020261020/current-cem.md).

## Check the evaluator before calling a region flat

The current repository defaults to shuffled slot order, while historical studies used earlier ordering and policy configurations. The runtime staging copy also differs functionally from the repository's later DBBO driver. Pin the evaluator, ordering, opponent identities and weights, simulator, cache tags and selected run configuration before combining measurements.

The cache's per-cell plug-in variance can return zero with one Bernoulli observation per opponent because each observed proportion is zero or one. That zero is not certainty. A generation's standard error across different teams is also not the uncertainty of a single team's score or a parent-child difference.

Repeated cache reads need not launch any fresh battles. Equal seed numbers do not by themselves prove useful common randomness across different simulation paths. Adequate repeated or paired observations are needed before asserting neutrality, a reliable improvement, or a local optimum. [Evaluator audit and exact source locations](../research_notes/Landscape%20analysis%20inside%20CEM%2020261020/current-cem.md).

The first practical deliverable is therefore a stage-by-stage diagnostic of a named saved CEM run. The first algorithmic question is whether a declared local proposal share helps the existing pipeline under matched costs. Adaptive mixtures and restart thresholds come after a repeatable diagnosis, rather than being inferred from one global smoothness number. No expected win-rate gain is asserted.
