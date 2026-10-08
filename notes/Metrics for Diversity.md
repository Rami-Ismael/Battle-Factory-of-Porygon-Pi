## Prerequisite for the diversity experiments

Complete the measurement definitions in this note before tracking diversity across CEM generations or comparing classifier-free guidance strengths. Both tasks are in [[New Todo Section with No Long Verbose Opus text geneation]].

- [x] I am surprise that I need to define what is the same pokemon team i think i need to have use the domain skills also, Decide what counts as the same team: ignore Pokémon list order and move order, and specify whether stat spreads are included in the exact-team comparison.
- [ ] 
- [ ] Finish implementing these diversity metrics and log them alongside win rate before fine-tuning and across CEM generations. This is the next step for the concern about mode collapse: measure whether diversity falls before deciding whether to change training to preserve it.

The information-theory and team-distance ideas below can be developed later; these experiments can start once the initial measurements and sampling protocol are defined.

 - [ ] Measure the Exact full-team uniqueness
 - [ ] Create diagram in exliadraw to 

## Reading — information theory and compression

- [Stanford EE274: Data Compression — online course notes](https://stanforddatacompressionclass.github.io/notes/). Free online course notes presented as an electronic book.
- [ ] Start with [Entropy and Neg-log likelihood thumb rule](https://stanforddatacompressionclass.github.io/notes/lossless_iid/entropy.html). Explains why a choice with probability p has an idealized description length of approximately -log₂(p) bits, and why entropy is the average information content. Connection to this project: a probability model learned from the team dataset can assign a surprisal score to a candidate team. More surprise means less expected under that model; it does not establish strategic difference or team strength.

## Assignment problem — comparing teams regardless of list order

An assignment problem finds the pairing between two groups with the smallest total cost, using every member exactly once. For comparing two six-Pokémon teams, pair each Pokémon from the first team with exactly one Pokémon from the second team. Here, cost means a chosen difference score between their sets; it does not mean damage or win probability.

For a small two-Pokémon example, suppose Team A is listed as Incineroar, Rillaboom, and Team B as Rillaboom, Incineroar. Their Incineroar sets differ slightly; their Rillaboom sets are identical. These scores are invented only to illustrate the matching:

| Difference score | Team B: Rillaboom | Team B: Incineroar |
| --- | ---: | ---: |
| Team A: Incineroar | 9 | 1 |
| Team A: Rillaboom | 0 | 8 |

Comparing by list position costs 9 + 8 = 17. Pairing Incineroar with Incineroar and Rillaboom with Rillaboom costs 1 + 0 = 1. The assignment problem chooses the latter pairing. For six Pokémon, it chooses six cells from a six-by-six table, exactly one per row and column, minimizing their sum. Dividing that sum by six gives an average difference per matched Pokémon.

This is a combinatorial optimization problem. It handles the pairing; we still have to define the difference scores using species/form, moves, items, abilities, stats, or another representation. A sum of individual differences does not automatically capture team synergy or battle behavior.

Reference: [SciPy's explanation and solver for the linear sum assignment problem](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).


## Proposed measurement contract — 2026-09-08

This makes the prerequisite above concrete. These are proposed defaults for implementation; the existing checkboxes remain for owner review.

### What counts as the same team

Use three keys for each parsed six-Pokémon team:

| Key | Fields compared | What it tells us |
| --- | --- | --- |
| 48-field | Species/form, ability, item, four moves, nature for each Pokémon | Whether the generator repeats its categorical output |
| With spreads | The same fields plus six Stat Points per Pokémon, ordered HP, Atk, Def, SpA, SpD, Spe | Whether a change is only a stat-spread change |
| Composition | Unordered six species/form identifiers, retaining multiplicity | Whether many distinct sets share the same roster |

Normalize names to the project's identifiers. Sort moves within each Pokémon, then sort complete Pokémon records; never sort individual fields across Pokémon. Ignore nicknames, formatting, Pokémon list order and move order. Preserve form identity. The metric consumes parsed, normalized records: resolve omitted values using the format's actual defaults before comparison, and reject missing required information rather than guessing it.

“With spreads” is exact only within the current pilot representation. If the format varies additional battle-relevant fields such as Tera type, IVs, level or gender, include them in a versioned full-team key before calling that metric exact full-team uniqueness. The existing parser does not preserve every such field.

### Initial measurements

Let N be the number of accepted draws, including repeated teams, and U the number of distinct keys. Report the following separately for the 48-field and with-spreads keys:

- **Unique fraction:** U / N.
- **Repeat fraction:** (N − U) / N. This counts occurrences beyond the first occurrence of each team, not all draws belonging to a repeated group.
- **Novel-draw fraction:** number of draws whose key is absent from the fixed training reference, divided by N. Repeated novel teams still count on every draw.
- **Novel-unique fraction:** number of distinct generated keys absent from that reference, divided by U. Report the numerator as well so one repeated novel team cannot look like broad exploration.

For compositions, save each composition and its count, the number of distinct compositions, and the largest composition's share of draws. Composition novelty is the fraction of draws whose composition is absent from the same fixed reference. A new composition establishes roster novelty; a familiar composition can still contain a different strategy.

**Example:** draws A, A, B, C have unique fraction 3/4 and repeat fraction 1/4. If the training reference contains A and B, novel-draw fraction is 1/4 and novel-unique fraction is 1/3. If all three teams share one composition, composition diversity is one, despite three distinct teams. Zero accepted draws gives undefined fractions, recorded as null with a failure status.

### Sampling and reference protocol

Use the first **512 validator-accepted proposals** per model checkpoint, before scoring, ranking or deduplication. Keep every repeat. Stop after at most 3,072 attempted proposals; if fewer than 512 are accepted, record the shortfall and do not present that sample as an equal-size comparison. This matches the existing search proposal budget. Smaller smoke tests must carry a different sample-size label.

Record attempted, accepted and rejected counts and acceptance fraction. This measures diversity conditional on acceptance; it does not describe the unfiltered generator. Keep the validator version and format fixed. Use rejection without post-validation repair; record any conversion or normalization applied before validation. An experiment with repair needs a separately named stream and records linking raw proposals to repaired outputs, because many proposals may become one repaired team.

Freeze the actual initial training-reference team records, their content hash, format and key version before comparing checkpoints. Do not silently replace the reference with the latest CEM pool. If initial-training membership cannot be recovered, label novelty relative to the available corpus; do not call it training-data novelty. Copying from later fine-tuning data can be measured separately against that generation's recorded training pool.

Use the same protocol before fine-tuning, after each CEM generation, and at each guidance strength. Record model checkpoint, run seed, generation, guidance and decoding settings with the measurements. Use multiple independent run seeds and report each run; successive generations are not independent repeats of an experiment.

Log win rate beside diversity, identifying the evaluated population and battle budget. The score-selected candidates' win rate describes selected candidates. To measure generator win rate, evaluate a random, unranked sample with a fixed opponent pool, battle policy and equal battles per team. Preserve draw multiplicity when averaging, even if repeated teams share evaluation work.

### Implementation gap found in the current pilot

Inspected `vgc-team-generator-pilot/src/encode.py`, `activesearch.py`, `temperature_experiment.py` and `corpus.py` on 2026-09-08:

- The encoder already ignores team order and move order for valid teams with distinct species identifiers. Its 48 fields omit spreads. Unknown vocabulary values map to the same mask identifier, so use normalized records rather than vocabulary codes for exact equality on arbitrary inputs.
- The temperature experiment records distinct textual teams, composition counts summarized as a total, composition entropy/effective count, and species Jaccard distance. Textual equality retains formatting differences that semantic team equality should ignore.
- Its corpus copy metric uses the 48-field encoding. Full with-spreads novelty, explicit unique/repeat fractions, a persisted composition frequency table, and verified initial-training-reference provenance still need implementation and checks.

The later distance, compression and structural-generalization tasks remain open. High exact uniqueness or novelty alone cannot establish strategic variety. For the hard Trick Room example, define the held-out structure independently of these keys and test that structure's battle behavior; a roster absent from training is insufficient evidence.


## Implementation and measured results — 2026-09-08

The initial metrics are now implemented in the pilot and connected to the temperature-comparison and entropy-loop proposal logs. The 48-field and with-spreads keys each report unique/repeat fractions and novelty against a fixed reference; composition output includes every composition's frequency. Existing checkboxes remain for owner review.

**Sampling update:** the guided sampler uses a hard cap of **4,096 attempts for 512 accepted proposals** (eight times the requested count). This is the implemented cap for guided comparisons; the earlier 3,072 proposal described the unguided sampler. Exact attempt/rejection counts are now logged. Empty or incomplete samples are identified explicitly. Omitted nature resolves to Serious, verified against the current simulator; other missing required parsed fields are rejected.

A retrospective report covers **86,400 draws across 180 streams**: 165 generations and 15 final samples. The initial training reference was verified against the checkpoint, parser/data and all 692 training-file hashes. All saved streams have 100% categorical and with-spreads uniqueness and novelty, yet composition diversity falls substantially. With fixed selection temperature 0.10, mean distinct compositions fall from **477.7 to 172.0 per 512 proposals**. This is direct evidence that exact uniqueness alone misses the observed concentration.

- [Readable results](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_diversity_metrics.md)
- [Every stream, composition frequency table and frozen reference](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_diversity_metrics.json)
- [Metric definitions, implementation and reproduction commands](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/diversity-metrics.md)

Still outstanding: a comparable sample from before fine-tuning. The saved campaign starts after its first fine-tuning step. Structural-generalization experiments, team distance, information-theory exploration and the unfinished diagram request remain separate open items.

## Worked example and information-theory starting point

[Editable Excalidraw diagram](Metrics%20for%20Diversity%20%E2%80%94%20worked%20example.excalidraw) · [PNG preview](Metrics%20for%20Diversity%20%E2%80%94%20worked%20example.png)

For the compression idea, distinguish **a team's surprise under a model** from **the distance between two teams**. A probability model assigns idealized code length −log₂ p(team); entropy is its average surprise. These quantities depend on the chosen probability model. [Stanford EE274 explanation](https://stanforddatacompressionclass.github.io/notes/lossless_iid/entropy.html).

A simple counterexample: two completely different teams could each have probability 1/16 and therefore the same four-bit idealized length. Their code-length difference is zero. That difference cannot serve as a team distance. Likewise, changing arbitrary species-to-binary IDs changes bitwise distances without changing Pokémon behavior. Canonical records remove irrelevant order before compression, but compact storage alone does not establish strategic similarity.

One information-theory calculation can already use the saved composition frequency table: set pᵢ = countᵢ/N and calculate H = −Σ pᵢ log₂ pᵢ. Then 2ᴴ is the number of equally common compositions giving the same entropy: one repeated roster gives 1; four equally common rosters give 4. The existing temperature experiment reports the equivalent quantity as exp(H) with natural-log entropy. This describes the empirical distribution of compositions at a fixed sample size; it is not a count of effective strategies or an unbiased estimate of all compositions the generator could produce.

## Baseline complete — 2026-09-08

The missing baseline is now captured from the original checkpoint with no generator fine-tuning: **three seeds, 1,920 accepted draws and 73,728 completed battles**. Its settings, training reference, opponents and battle policy were verified against the completed campaign. The final audit recomputed all **186 streams and 88,320 draws**, including the historical generations. All 26 metric, logging, sampler, baseline and campaign tests passed.

The baseline averages **498.3 distinct compositions per 512 draws**. Final CEM arms average **172.0–276.3**, while exact team uniqueness stays at **100%**. At fixed temperature 0.10, the largest roster's mean share rises to **40.5%**, compared with **1.4%** in the baseline. These results support retaining composition frequencies and concentration measures alongside exact uniqueness and novelty.

- [Complete before/after report and chart](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/diversity_before_after.md)
- [Verified baseline samples and battle results](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/diversity_baseline.json)
- [Full measurement and source audit](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/diversity_verification.json)

The initial measurement prerequisite is implemented and evaluated before and across fine-tuning. The report labels generator win rate separately from score-selected candidate win rate. Structural-generalization experiments, a strategic team-distance function and a compression-based team model remain follow-up work; the measurements above do not establish new playstyles. Earlier “baseline outstanding” statements describe the work before this capture. Checkboxes are retained for owner review.

# Hand written notes section 


![[Rami Hand written notes with no claude interacting at all about diversity]]