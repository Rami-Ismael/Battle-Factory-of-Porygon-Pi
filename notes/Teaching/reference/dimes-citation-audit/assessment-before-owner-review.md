# DIMES citing papers — reading-list assessment

Assessed 2026-09-08 against this project's existing generator experiments and reading decisions. Three subagents divided citation discovery, project-fit analysis, and individual paper screening; the parent reviewed the recommendations and assembled the complete ledger.

**Recommend one selective MED addition now: Self-Improved Learning for Scalable Neural Combinatorial Optimization. Keep Neural Genetic Search and LNS Meets Iterative Neural Constraint Heuristics as conditional leads.** Citation count is an eligibility filter, not a reason to read a paper. The active Reading List has not been edited by this assessment.

## What was covered

The seed is Qiu, Sun and Yang's [DIMES, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/a3a7387e49f4de290c23beea2dfcdc75-Abstract-Conference.html), [arXiv 2210.04123](https://arxiv.org/abs/2210.04123).

- Semantic Scholar returned all **175** direct-citing records reported by its seed, without another page. **111** had at least two citations themselves; **64** did not qualify.
- An OpenAlex cross-check of both exact-title seed records supplied **two additional qualifying records**: GLOP and DOGE-Train.
- All **113 qualifying records** received an individual relevance decision. Four confirmed redundant version rows reduce this to **109 entries**. A strongly supported but still probable GOAL title-version merge reduces the reading assignments to **108**.
- Final ledger: **1 ADD, 2 MAYBE, 41 DEFER, 62 SKIP, 2 ALREADY, 5 DUPLICATE**. These are record-level categories; the five duplicate rows receive no separate reading assignment.

This covers the retrieved indexes, not every citing paper in existence. Citation relationships come from the indexes; not every bibliography was manually checked. Screening used author/publisher abstracts throughout, with targeted full-text method and experiment checks for plausible additions and difficult cases. It is not a claim to have read 108 papers cover to cover. Each row records its evidence level.

[Complete searchable ledger](../Teaching/reference/dimes-citation-audit/screening.md) · [CSV](../Teaching/reference/dimes-citation-audit/screening.csv) · [JSON](../Teaching/reference/dimes-citation-audit/screening.json) · [Retrieval and duplicate methodology](../Teaching/reference/dimes-citation-audit/methodology.md) · [Canonical record mapping](../Teaching/reference/dimes-citation-audit/canonical-inventory.json)

## The project-fit test

The current task is to generate complete legal six-Pokémon teams, estimate performance through noisy battles, and improve the generator using elite selection and retraining while monitoring diversity. A paper earns an addition only if it changes a concrete experiment or component beyond the existing plan.

That plan already includes training-data mixtures, label-quality distributions, model scaling, guidance and temperature, regenerating 1/3/all 6 sets, elite fraction and replay. **Slot shuffling already exists.** Evaluation distinguishes the quality of an unselected generator sample from the quality of selected finalists. Existing reading decisions defer reward/preference integration and park several graph-energy, annealing and molecular-only directions. Surveys are excluded by the owner's preference. These constraints prevent generic “both are combinatorial optimization” recommendations.

[Detailed rubric and local evidence](../Teaching/reference/dimes-citation-audit/project-fit-rubric.md)

## Recommended selective addition

### S029 — Self-Improved Learning for Scalable Neural Combinatorial Optimization

**34 Semantic Scholar citations. MED; read §4.1–4.2 and Figure 3.** The method reconstructs solutions, retains improvements and uses them as supervised pseudo-labels in subsequent learning rounds. [Primary paper](https://arxiv.org/abs/2403.19561)

**New experiment for this project:** compare global top-fraction CEM training with training examples retained as improvements relative to each starting team. Keep generated/evaluated candidates and training-example counts comparable; include frozen reconstruction as a control. Assess the next generator's unselected draws, legality and diversity, then freshly evaluate finalists.

This is an adaptation, not a result established for Pokémon. Whole teams must be evaluated: TSP subpath additivity does not justify judging sets independently. Battle noise requires fresh validation of apparent improvements within the same overall battle budget. Per-parent retention might retain more different starting families, but does not guarantee diversity and can also train on mediocre teams. That uncertainty is the reason to run the comparison.

Suggested queue placement: beside the CEM / self-improvement / training-data selection readings. Do not prioritize the paper's large-instance attention engineering.

## Two conditional leads

| Paper | Citations | When it becomes useful | Concrete question and limitation |
|---|---:|---|---|
| **S074 — [Neural Genetic Search in Discrete Spaces](https://arxiv.org/abs/2502.10433)** | 7, Semantic Scholar | Selective MED if testing crossover with a frozen generator | §3 restricts generation using two parents' tokens, occasionally lifting restrictions for mutation. Compare a typed, legality-aware two-parent variant with one-parent completion under the same generator and battle budget. Its sequential decoder assumptions need adaptation to masked diffusion. This tests search operators, not CEM retraining; no expert optimal-child labels are required for the crossover mechanism. |
| **S104 — [Large Neighborhood Search Meets Iterative Neural Constraint Heuristics](https://arxiv.org/abs/2603.20801)** | 2, Semantic Scholar | Selective MED once partial regeneration is an active experiment | Compare random versus confidence-weighted removal, crossed with greedy versus stochastic refill. Model confidence is not evidence that a Pokémon causes poor battle results. The paper uses explicit constraint penalties and separately trained operator configurations; it does not establish improvements from changing inference alone or from noisy win-rate feedback. |

Neither conditional lead needs to expand the immediate queue. S104 is narrower than a general NLNS reading: its value is the removal/refill comparison only. S074 adds a two-parent construction question beyond the existing one-parent regeneration plan.

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
