# DIMES citing papers — reading-list assessment

Assessed 2026-09-08 against this project's existing generator experiments and reading decisions. Three subagents divided citation discovery, project-fit analysis, and individual paper screening; the parent reviewed the recommendations and assembled the complete ledger.

**Recommend Neural Genetic Search in Discrete Spaces as the sole selective MED addition.** Owner review identified it as the only compelling new reading; on reassessment, the other two leads overlap too much with the existing CEM and partial-regeneration plan to justify queue space. Citation count is an eligibility filter, not a reason to read a paper. The active Reading List has not been edited by this assessment.

## What was covered

The seed is Qiu, Sun and Yang's [DIMES, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/a3a7387e49f4de290c23beea2dfcdc75-Abstract-Conference.html), [arXiv 2210.04123](https://arxiv.org/abs/2210.04123).

- Semantic Scholar returned all **175** direct-citing records reported by its seed, without another page. **111** had at least two citations themselves; **64** did not qualify.
- An OpenAlex cross-check of both exact-title seed records supplied **two additional qualifying records**: GLOP and DOGE-Train.
- All **113 qualifying records** received an individual relevance decision. Four confirmed redundant version rows reduce this to **109 entries**. A strongly supported but still probable GOAL title-version merge reduces the reading assignments to **108**.
- Final ledger: **1 ADD, 0 MAYBE, 43 DEFER, 62 SKIP, 2 ALREADY, 5 DUPLICATE**. These are record-level categories; the five duplicate rows receive no separate reading assignment.

This covers the retrieved indexes, not every citing paper in existence. Citation relationships come from the indexes; not every bibliography was manually checked. Screening used author/publisher abstracts throughout, with targeted full-text method and experiment checks for plausible additions and difficult cases. It is not a claim to have read 108 papers cover to cover. Each row records its evidence level.

[Complete searchable ledger](../Teaching/reference/dimes-citation-audit/screening.md) · [CSV](../Teaching/reference/dimes-citation-audit/screening.csv) · [JSON](../Teaching/reference/dimes-citation-audit/screening.json) · [Retrieval and duplicate methodology](../Teaching/reference/dimes-citation-audit/methodology.md) · [Canonical record mapping](../Teaching/reference/dimes-citation-audit/canonical-inventory.json)

## The project-fit test

The current task is to generate complete legal six-Pokémon teams, estimate performance through noisy battles, and improve the generator using elite selection and retraining while monitoring diversity. A paper earns an addition only if it changes a concrete experiment or component beyond the existing plan.

That plan already includes training-data mixtures, label-quality distributions, model scaling, guidance and temperature, regenerating 1/3/all 6 sets, elite fraction and replay. **Slot shuffling already exists.** Evaluation distinguishes the quality of an unselected generator sample from the quality of selected finalists. Existing reading decisions defer reward/preference integration and park several graph-energy, annealing and molecular-only directions. Surveys are excluded by the owner's preference. These constraints prevent generic “both are combinatorial optimization” recommendations.

[Detailed rubric and local evidence](../Teaching/reference/dimes-citation-audit/project-fit-rubric.md)

## Recommended selective addition

### S074 — Neural Genetic Search in Discrete Spaces

**7 Semantic Scholar citations. MED; read §3, particularly crossover/mutation and selection/replacement.** [Primary paper](https://arxiv.org/abs/2502.10433)

The distinct contribution is using a pretrained generator to recombine two parents. It restricts generation to parental tokens and occasionally lifts that restriction for mutation. This suggests a concrete addition beyond the current one-parent regeneration plan: compare two-parent, legality-aware generation against one-parent completion with the same frozen model and battle budget.

For teams, the adaptation must respect field types and species-dependent legality; taking an unrestricted union of both parents' tokens is insufficient. The paper assumes sequential generation, so integration with masked diffusion needs design work. Its search does not require expert optimal-child labels, and its tests extend beyond molecular design. It does not establish improved CEM training or better Pokémon teams.

## Leads not recommended for the queue

- **S029 — Self-Improved Learning: DEFER.** Reconstruction followed by improved pseudo-label training overlaps strongly with the existing evaluate/select/retrain loop. Retaining improvements per starting team is a possible experiment, but that adaptation alone does not justify another routing paper. The original recommendation overstated its incremental value.
- **S104 — Large Neighborhood Search Meets Iterative Neural Constraint Heuristics: DEFER.** Removal/refill variants overlap the planned partial-regeneration experiments. Explicit constraint penalties and model confidence do not provide battle-based credit assignment to team members. No separate reading assignment is warranted now.

The initial assessment is preserved in [the pre-review snapshot](../Teaching/reference/dimes-citation-audit/assessment-before-owner-review.md). This revision changes project reading priorities, not the citation inventory or claims about paper quality.

## Why other tempting papers did not earn additions

- **Symmetric Replay Training (S064): defer.** It has relevant replay ideas and black-box hardware evidence, but permutation augmentation alone duplicates existing slot-shuffled training. A distinct replay objective or schedule must justify the extra reading.
- **Ant Colony Sampling with GFlowNets (S014): defer.** Its diverse-prior experiments are useful, but proposal-versus-downstream-search attribution already has a place in the project. No need to adopt ACO or add another paper merely for “diversity.”
- **Position: Rethinking Post-Hoc Search-Based Neural Approaches… (S032): defer.** It reinforces holding search fixed while changing the learned component, but overlaps the existing Unify ML4TSP reading and evaluation controls.
- **Winner Takes It All / Poppy (S013): defer.** Training complementary policies for best-of-population performance is a different objective from improving unselected average team quality. Reconsider if that portfolio objective becomes explicit.
- **Preference Optimization (S042): defer.** It supplies a different training loss, but preference/reward integration is already deferred and noisy battle comparisons remain unresolved.
- **Revisiting Sampling (S027): defer.** A useful warning to test simple baselines; its local energy/gradient machinery is not directly available from battles.
- **DIFUSCO (S002) and Tackling Prevalent Conditions (S060): already listed.** No duplicate additions.
- **CADO: excluded before relevance ranking.** Its canonical record and two malformed variants each have zero citations in this snapshot, so it fails the requested threshold.

These are queue decisions for the current project, not claims that the papers are poor research. The ledger gives the source and specific reason for every remaining paper, including all routing applications, graph-energy methods, surveys and unrelated citations.

## Verification and reproducibility

Raw Semantic Scholar and OpenAlex responses, excluded records, individual agent screens and the final merged screen are retained in `Teaching/reference/dimes-citation-audit/`. The final `screening.json` and `screening.md` supersede preliminary agent shortlist decisions. Duplicate rows retain their original evidence and canonical mapping; citation counts are never added across versions or providers.

Assembly checks verified all 113 expected audit IDs appear exactly once, all satisfy their provider's ≥2 threshold, and every row includes a decision, reason, source and evidence level. GOAL remains explicitly marked probable in the canonical mapping. No active reading-list entries were added, removed or reprioritized.
