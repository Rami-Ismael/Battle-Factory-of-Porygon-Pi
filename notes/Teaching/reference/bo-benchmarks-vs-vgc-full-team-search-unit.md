---
type: reference
tags:
  - bayesian-optimisation
  - benchmarks
  - problem-framing
  - search-space
  - vgc
answers: "[[Bayesian optimisation benchmarks versus VGC team building]]"
scope: full-team search unit only (species, ability, item, moves, Stat Points, alignment); team preview, the battle, and the objective function are out of scope
supersedes: "[[bo-benchmarks-vs-vgc-team-building]]"
created_at: 2026-09-26
updated_at: 2026-09-28
common_rule: introduced by one paper and reused by at least two papers that share no author with it or with each other
figure: Blog/visuals/benchmark-to-team/ (planned, not built)
---
Every row of the difference list is read off a benchmark's own variable definition. Nothing is inherited from "Search Perspective" in [[What makes Pokémon a unique problem set]]. A benchmark's input is called a *point*, because *candidate* means a species in this vault.

## Rulings on the instruction (2026-09-28)

1. **Molecule benchmarks count.** Penalised logP, GuacaMol, the Therapeutics Data Commons tasks and docking are in the table. Citing them as benchmarks illustrates the comparison. It does not reopen the molecular-optimisation approach closed on 2026-09-02.
2. **"Common" means introduced + two independent reusers** (the `common_rule` property). A benchmark checked but below that bar is listed under *Checked, not common*, not in the table.
3. **The objective function is out of scope.** It is defined through battles, and the battle is excluded. The 2026-09-24 list was retired for exactly this.
4. **The conditional / tree-structured family is now checked.** It was the one gap the 2026-09-26 version left open. NAS-Bench-101 and NAS-Bench-201 enter the table. NAS-Bench-101 qualifies differences 2, 3, 7 and 9 (marked in place below). Difference 4 survives: the only tree-structured benchmark found is below the bar.
5. **Sizes come from the roster file, and the two pulls disagree.** See the table below. This is difference 16 happening to the list itself. Any figure built from this note reads `domain.json` at build time rather than typing numbers in.

**The VGC search unit (Champions, Reg Set M-B).** A *team* is six Pokémon. Each has a form (231 buildable, 208 Dex numbers), one ability from that form's 1–3, one held item or none (149 per slot), up to four distinct moves from that form's movepool (1–102), Stat Points (integers, ≤32 per stat, ≤66 total) and one of 21 stat alignments. Species Clause (by Dex number) and Item Clause (none exempt) run across the six. No EVs, IVs or Tera. Sizes from [[Clause or the feature of the search space]].

| Quantity | RotomPicks pull, 2026-08-17 ([[Clause or the feature of the search space]]) | Showdown validator pull, 2026-09-18 (`Blog/visuals/team-search-space/domain.json`) |
|---|---|---|
| Forms | 231 buildable | 232 roster entries |
| Distinct abilities | 186 | 186 |
| Held items | 148 (+ none) | 148 (+ none) |
| Legal moves | 486 | 496 |
| Form–move pairs | 14,065 | 14,191 |
| Movepool range | 1–102 (Ditto–Gallade) | 1–105 (Ditto–Gallade) |

## The benchmarks

| Benchmark | Family | What a point is | Built-in constraints | Source (introduced → independent reuse) |
|---|---|---|---|---|
| Contamination control | combinatorial | 25 binary | none | BOCS §4.3 → COMBO §4.1, CASMOPOLITAN §4.1 |
| Pest control | combinatorial | 25 categorical × 5, a one-way chain of stations | none | COMBO §4.2 → CASMOPOLITAN §4.1, BODi §7.1, Bounce §4.3; poli §4.2.1 re-casts it as an Ehrlich function ("PestControlEquiv") |
| Weighted MaxSAT-60 | combinatorial | 60 binary | clauses exist but are hidden | COMBO §4.3 → CASMOPOLITAN §4.1, BODi §7.1, Bounce App. B.1.3 |
| LABS-50 | combinatorial | 50 in {−1, +1}, order matters | none | MerCBO §4 → Bounce §4.1, Heat kernels §4 |
| Cluster Expansion (MaxSAT-125) | combinatorial | 125 binary | hidden clauses | Bounce §4.2 → Heat kernels §4, MOCA-HESP §5.2 |
| Ackley-20 categorical | combinatorial | 20 categorical × 11 grid values | none | MCBO §5 → Heat kernels App. E.5, MOCA-HESP §5.2 |
| Antibody design (Absolut!) | combinatorial | CDRH3 string, length 11 over 20 amino acids | net charge in [−2, 2]; no residue over five times; no glycosylation motif | AntBO §2.1 → Heat kernels App. E.4, MOCA-HESP §5.2 |
| Ehrlich functions (holo-bench) | combinatorial / sequence | length L over v symbols (v 4–20; L 5–256 across papers) | a banned-transition Markov chain: x is feasible only if every adjacent pair has nonzero transition probability; infeasible points score −∞ | Stanton et al. §3.2 → poli §4.2.1 (L 5/15/64, v 20), A-GPS §6.2 (L 15/32/64, v 20) |
| Ackley-53 mixed | mixed-variable | 50 binary + 3 continuous in [−1, 1]³ | box | Bliek et al. §5.4 → CASMOPOLITAN §4.2, BODi §7.3, Bounce App. B.1.1 |
| SVM-53 | mixed-variable | 50 binary feature switches + 3 continuous hyperparameters | box | BODi §7.3 → Bounce §4.4, MCBO App. C |
| Arithmetic expressions | latent-space | a string from a ≤15-rule grammar, decoded from a grammar VAE; 100k-expression pretraining corpus | grammar makes nearly every decode valid | Grammar VAE §4.1 → Grosnit et al. §4, LADDER §4.1, CoBO §3 |
| Penalised logP | latent-space | a molecule (SMILES, junction tree or SELFIES) decoded from a VAE pretrained on ZINC250k | junction-tree and SELFIES decoders are always valid | Gómez-Bombarelli et al. v1 Eq. 1 → Grosnit et al. §4, LOL-BO §5.1 |
| GuacaMol MPO tasks | latent-space | a molecule from a VAE pretrained on 1.27M molecules | decoder validity | GuacaMol §3.2 → LOL-BO §5.2, CoBO §3 |
| PMO (as run in poli) | latent-space | SELFIES padded to length 70 over 64 tokens, searched in a 2- or 128-dimensional latent space | decoder validity | poli §4.2.2 |
| DRD3 docking (Therapeutics Data Commons) | latent-space | a molecule decoded from a VAE | decoder validity | Therapeutics Data Commons §10.3 → LOL-BO §5.3, CoBO §3 (restored 2026-09-28 from [[bo-benchmarks-vs-vgc-team-building]]) |
| NAS-Bench-101 | combinatorial / graph | a cell: a DAG on ≤7 vertices (IN and OUT fixed), each intermediate vertex one of 3 operations; 7×7 upper-triangular adjacency (21 possible edges) + 5 labels; ≈423k unique cells after isomorphism de-duplication | ≤9 edges in total; a path from IN to OUT must exist; isomorphic graphs are the same cell, identified by the benchmark's own iterative graph hash (Supp. S1) | Ying et al. §2.1 → BANANAS §5–6, NAS-BOWL §5 |
| NAS-Bench-201 | combinatorial / graph | a complete DAG on 4 nodes, each of its 6 edges one of 5 operations; 15,625 points | none | Dong and Yang (abstract) → BANANAS §6, NAS-BOWL §5 |

## Differences in the search unit

**Shape of a point**

1. **Flat vector versus sets inside a set.** Every benchmark point is one flat vector or one string. A team is a set of six records, and each record holds a set of up to four moves. The benchmarks have one level; VGC has two.
   *Qualified 2026-09-28:* the NAS-Bench points are a graph (adjacency matrix + label list), not a flat string, but still one level. Nothing in the table nests records inside a record.
2. **Fixed dimension versus a dimension that depends on the point.** Every combinatorial and mixed benchmark fixes its length (25, 50, 53, 60, 125, L = 11, Ehrlich's L). In VGC a Pokémon may carry fewer than four moves (Ditto has one legal move), and an item slot may be empty. The number of active decision variables changes with the species chosen.
   *Qualified 2026-09-28:* NAS-Bench-101 shares this. A cell may use fewer than 7 vertices and fewer than 9 edges, so its active variables also change with the point. The difference now holds for every other benchmark in the table, not all of them.
3. **Order means something in the benchmarks. In a team it means nothing.** Pest control is a chain, LABS scores autocorrelation, and antibody, Ehrlich (motif spacing), expressions and SMILES are strings. Reorder the six Pokémon, or one Pokémon's four moves, and the team is unchanged: 6! × (4!)⁶ slot encodings of one team.
   *Qualified 2026-09-28:* NAS-Bench-101 shares this too. Relabelling intermediate vertices gives an isomorphic, identical cell. The remaining difference is who removes the duplicates: NAS-Bench-101 ships a graph hash that maps every encoding to one cell (Supp. S1), while no published VGC table does this, so the searcher has to build the canonical form itself.

**Domains**

4. **Each benchmark variable has one domain, shared by every point. In VGC the species picks the domain.** Pest control's stations all take 5 values, antibody positions all take 20, Ehrlich positions all take v, and SVM-53's box ignores its binary part. In VGC the form fixes the ability list (1–3) and the movepool (1–102). This is a hierarchical space, not a Cartesian product.
   *Checked 2026-09-28:* still holds for every common benchmark. NAS-Bench vertices and edges all share one operation list. The only benchmark found where one choice decides which variables exist is the Jenatton et al. tree function, below the common bar (see *Checked, not common*). Movepool ceiling is 105 in the 2026-09-18 pull. *Sharpened after the BoTorch/Ax check:* the hierarchical benchmarks that do exist (Jenatton, Ax's surrogate suites) switch whole parameters on or off, each with a fixed domain. None lets one choice rewrite another parameter's list of values, which is what a form does to the move and ability slots.
5. **2–20 values per variable versus hundreds.** The benchmarks top out at 20 (amino acids, Ehrlich's v). VGC slots have 231 forms, 149 items, 186 abilities in the pool, and movepools up to 102 drawn from 486 moves.
   *Numbers 2026-09-28:* the 2026-09-18 pull gives 496 moves and movepools up to 105. The NAS-Bench operation lists (3 and 5) sit inside the 2–20 range.
6. **Homogeneous variables versus mixed-type records.** Each benchmark uses one type throughout (all binary, all 5-way, all amino acids) or binary plus a box. One VGC record mixes six kinds of categorical choice (form, ability, item, moves, alignment) with an integer budget.
7. **The numeric part is a coupled integer budget, not an independent box.** Ackley-53 and SVM-53 have three independent continuous coordinates. Stat Points are integers under a shared cap (≤66 over six stats, ≤32 each). Every allocation trades against the others, and none of the benchmarks has a sum constraint.
   *Qualified 2026-09-28:* NAS-Bench-101 has a budget too: at most 9 of 21 binary edges. The difference narrows to its form. That cap is one count over binary switches; Stat Points are integers with a cap per stat (32) and a cap on the total (66), repeated separately inside each of the six records.
8. **Benchmark symbols are bare labels. VGC symbols come with a free feature table.** MaxSAT, LABS, Ehrlich, pest control and Ackley-categorical give an index and nothing else. Hamming or one-hot kernels are all you can use, and the index order is arbitrary. The molecule and antibody tasks have chemistry, but only through the decoder or the string. Every VGC form, move and item carries known attributes (typing, base stats, power, effect) before any evaluation.

**Constraints**

9. **Benchmark feasibility is local. VGC feasibility is team-wide.** Ehrlich bans adjacent pairs of symbols. Antibody checks count and charge within one string. Grammars check one token at a time. Species Clause and Item Clause are all-different constraints across six separate records.
   *Qualified 2026-09-28:* NAS-Bench-101 feasibility is global over the cell (the edge cap, and a path from IN to OUT). The difference narrows to the constraint's kind: no benchmark has an all-different constraint across separate records.
10. **Species Clause is enforced on a derived attribute.** Each benchmark constraint reads the chosen symbols directly. In VGC a slot holds a *form* but Species Clause compares *Dex numbers*: Rotom-Wash plus Rotom-Heat is two distinct values, and an illegal team.
11. **Benchmark feasible sets follow compact rules. VGC legality is a lookup table.** Ehrlich's feasible set is a transition matrix, antibody's is three rules, and expressions' is a ≤15-rule grammar. VGC move legality is 14,065 form–move pairs with no generating rule a model could learn in a few parameters.
   *Numbers 2026-09-28:* 14,191 form–move pairs in the 2026-09-18 pull. NAS-Bench-101's feasible set is also a compact rule (edge cap + connectivity).
12. **Every benchmark value works everywhere. Some VGC values only work in one place.** In pest control every pesticide acts at every station. A Mega Stone is legal on any Pokémon but only does anything on its own species. It is legal but useless everywhere else.
13. **A grammar or SELFIES decoder cannot express Item Clause or Species Clause without memory.** The latent benchmarks get validity from a context-free grammar or a SELFIES decoder. Both are local rules, applied token by token. Enforcing all-different across six records means carrying the set of used Dex numbers and items through decoding. Neither the grammar VAE nor the molecule decoders do that.

**Pretraining corpus (latent family only)**

14. **Latent benchmarks learn from 10⁵–10⁶ real objects. VGC has hundreds of real teams.** The corpora are 100k expressions, 250k ZINC molecules and 1.27M GuacaMol molecules. VGC's human-built teams (the collected meta list, VGCPastes) are orders of magnitude fewer. Randomly sampled legal teams can be made in any quantity, but they are not real teams.

**What is not a difference**

15. **Size alone does not separate them.** Ehrlich at v = 20, L = 256 has 20²⁵⁶ ≈ 10³³³ strings, far beyond the VGC team count. A raw cardinality comparison cannot support any claim that VGC is harder.
16. **Benchmark domains are frozen. The VGC domain is patched.** Every benchmark's variable set is fixed at publication. Champions has changed its roster partway through a regulation (see the 224/228/231 form counts in [[Clause or the feature of the search space]]), so the search space itself moves, not just who you are playing.

## Not collected

- *Retired 2026-09-28 — checked; see* Checked, not common *and ruling 4.* Hyperparameter-optimisation suites with conditional (tree-structured) parameters, used in mixed-variable BO. They are the one family that might share difference 4, and they were not checked here.

## Checked, not common

- **Jenatton et al. tree-structured function.** Binary choices x1–x3 pick one of four leaves, and each leaf activates its own continuous variable plus one shared one (d = 9, effective dimension 2 at any leaf, known minimum 0.1). This is the one benchmark found where a choice decides which variables exist, the structure behind difference 4. Introduced in Jenatton et al. §4 (Fig. 1); one independent reuse found (Ma and Blaschko §5.1, who rebuilt it because the original code was unavailable). BARK, which cites Jenatton et al., uses BART-sampled tree functions instead. One reuse is below the bar. Ax ships it in its default benchmark registry (`jenatton`, `jenatton_observed_noise`), so it is a standard library problem even though papers rarely reuse it. Ax's copy bounds x4–x7 in [0, 1]; Ma and Blaschko state [−1, 1].

## Smoothness — added 2026-09-28 (draft; outside ruling 3)

*Owner request 2026-09-28. Smoothness is a property of the objective function, which ruling 3 put out of scope, and the VGC side can only be measured through battles. It sits here as a separate axis until the owner decides whether ruling 3 changes.*

- **Method: neighbour correlation.** How strongly a point's score predicts the score of a point one edit away (Weinberger 1990). NAS-Bench-101 §3.3 measures exactly this for its own space (random-walk autocorrelation, where one edit flips one operation or one edge), so the method comes from a benchmark paper, not from this project.
- **Measured on VGC (2026-09-28, `docs/ruggedness2.md`, `results/ruggedness2.json` in the code repo).** Noise-corrected neighbour correlation ρ, 40 starting teams, 6 edit types, 177,408 battles against the top-50 meta, behaviour-cloning policy on both sides. Pest Control ρ 0.99 and a gridded sphere ρ 0.99 are the benchmark reference. Only species edits are rougher than Pest Control beyond battle noise (ρ 0.86 from meta starts, 0.75 from search-found starts). Move and item edits are rougher than Pest Control but inside battle noise. Ability, alignment and Stat Point edits are as smooth as the benchmarks (ρ ≈ 0.99–1.00).
- **Not yet done:** read the numbers in NAS-Bench-101 §3.3, and find smoothness or ruggedness measurements for LABS, MaxSAT, Ehrlich and the latent-space tasks from their own papers.

## Library suites — BoTorch and Ax (checked 2026-09-28)

Read from source: BoTorch `botorch/test_functions/` at commit a1b1ad8 (2026-09-23), Ax `ax/benchmark/problems/` at commit 57efa05 (2026-09-21).

- **BoTorch adds no new row.** Of its 31 single-objective synthetic functions, only two are not continuous: `AckleyMixed` (default 53 dimensions: 50 binary + 3 continuous in [0, 1], optimum optionally moved to a random point) and `Labs` (binary, known optima for 10–60 dimensions). Both are already rows (Ackley-53 mixed, LABS-50). No categorical, latent-space or hierarchical function. This agrees with the table: the library carries the same mixed and binary problems the papers keep reusing.
- **Ax adds three things, none of them a new common benchmark.**
  - The Jenatton tree function, in the registry (see *Checked, not common*).
  - "Discrete" Hartmann, Ackley and Rosenbrock, from Daulton et al. 2022: continuous functions with some coordinates rounded to integers (Discrete Ackley: 13 dimensions, 10 of them integer). Rounded integers are ordered, unlike VGC's unordered categories, which adds a benchmark to difference 8's side.
  - MNIST, Fashion-MNIST and CIFAR-10 surrogates with a hierarchical space: switches that turn dropout and weight decay on or off. These are not in the registry.
- **What the tooling can and cannot state.** Ax's `dependents` field expresses "this value turns those parameters on". Its linear constraints on numeric parameters could state the Stat Point caps (≤32 per stat, ≤66 total). Neither library has a way to state an all-different constraint across choice parameters, so Species Clause and Item Clause stay outside what the tools can say (difference 9).

## Figure plan — one benchmark point becomes a team (not built)

A stepper figure for the blog, `Blog/visuals/benchmark-to-team/`. It opens on one Pest Control point (25 cells in a chain, 5 colours each), the most reused benchmark in the table, and turns it into a real team one difference at a time. It follows the one figure that worked in "One Edit Away": real Pokémon, real mechanics, the reader in control.

- **Shell:** dark theme, Showdown sprites bundled with the page, ← Play → buttons, a slider with ticks for the four chapters (shape, domains, constraints, corpus) and a fraction counter. HTML + JavaScript + canvas, one self-contained file; sizes read from `domain.json` when the page is built.
- **Timing:** layout moves 240 ms, `cubic-bezier(0.77, 0, 0.175, 1)`, transform only; entering elements 200 ms ease-out, 30 ms stagger; dragging the slider or clicking cancels motion in progress; with reduced motion on, elements snap and only crossfades remain.

| # | What changes on screen | Motion |
|---|---|---|
| 1 | The 25 cells regroup into 6 Pokémon cards with 4 move chips each | layout morph |
| 2 | Ditto's card shrinks to 1 move chip; an item slot empties | chips fade and shrink out |
| 3 | Cards and move chips shuffle; badge "same team · 1 of 6!×(4!)⁶ ≈ 1.4×10¹¹ encodings". A NAS-Bench-101 cell shuffles its vertices beside it and its hash stays fixed | shuffle; the number and the hash stay put |
| 4 | **Reader picks the species:** Ditto → Gallade rebuilds the move list (1 → 105) and the ability list (1–3) | crossfade, staggered list |
| 5 | Width of each choice list: 5 (Pest Control) versus 232 forms or 148 items | bars grow |
| 6 | Each field of a card coloured by its kind of choice | colour fade |
| 7 | **Reader drags Stat Points:** raising one drains the others under the 66 cap; NAS-Bench-101's "≤9 of 21 edges" shown as the one-count contrast | follows the drag, no easing |
| 8 | Hover a chip: typing, stats, power; a benchmark cell shows only its index | tooltip |
| 9 | Two cards with the same item turn red, joined by a line (Item Clause) | red outline, line draws in |
| 10 | Rotom-Wash + Rotom-Heat: two sprites, one Dex #479, flagged illegal | same red treatment |
| 11 | VGC's form × move legality table beside Ehrlich's small transition matrix | still image fades in |
| 12 | **Reader moves Charizardite Y:** on Charizard it becomes the Mega sprite; on Garchomp it greys out, "legal, does nothing" | sprite swap versus greying out |
| 13 | A decoder writes a team token by token; without a record of used items it repeats one (red); with the record it does not | tokens appear one at a time |
| 14 | Pretraining corpora: 250k and 1.27M molecules versus hundreds of real teams | still log-scale bars |
| 15 | Ehrlich's 10³³³ strings versus the VGC team count, labelled "not a difference" | still — animating a growing count would suggest VGC is harder, which this difference denies |
| 16 | Form count ticks 224 → 228 → 231, then the pull table's two columns | counter ticks up |

Stops 4, 7 and 12 are the ones the reader drives; the rest connect them.

## Citations

🟢 = a free full text was opened. Rows other than Ehrlich, poli, A-GPS and the two NAS-Bench rows (opened 2026-09-28) were verified 2026-09-24 in [[bo-benchmarks-vs-vgc-team-building]], and their full citations are there.

- [Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms](https://www.semanticscholar.org/arxiv/2407.00236) by Stanton, Alberstein, Frey, Watkins, Cho 🟢 ICML 2024 workshop
- [A survey and benchmark of high-dimensional Bayesian optimization of discrete sequences](https://www.semanticscholar.org/paper/1f532488d98a995ff4a428a3112a903b24dde001) (poli) 🟢 NeurIPS 2024 Datasets and Benchmarks track
- [Amortized Active Generation of Pareto Sets](https://www.semanticscholar.org/arxiv/2510.21052) (A-GPS) 🟢 NeurIPS 2025
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://www.semanticscholar.org/arxiv/1902.09635) by Ying, Klein, Real, Christiansen, Murphy, Hutter 🟢 ICML 2019
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://www.semanticscholar.org/arxiv/2001.00326) by Dong, Yang 🟢 ICLR 2020 (abstract read, not the full text)
- [BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search](https://www.semanticscholar.org/arxiv/1910.11858) by White, Neiswanger, Savani 🟢 AAAI 2021
- [Interpretable Neural Architecture Search via Bayesian Optimisation with Weisfeiler-Lehman Kernels](https://www.semanticscholar.org/arxiv/2006.07556) (NAS-BOWL) by Ru, Wan, Dong, Osborne 🟢 ICLR 2021
- [Bayesian Optimization with Tree-structured Dependencies](http://proceedings.mlr.press/v70/jenatton17a.html) by Jenatton, Archambeau, González, Seeger 🟢 ICML 2017
- [Additive Tree-Structured Covariance Function for Conditional Parameter Spaces in Bayesian Optimization](https://www.semanticscholar.org/arxiv/2006.11771) by Ma, Blaschko 🟢 AISTATS 2020
- [BARK: A Fully Bayesian Tree Kernel for Black-box Optimization](https://www.semanticscholar.org/arxiv/2503.05574) 🟢 (checked for the Jenatton function only; venue not looked up)
- [BoTorch test functions — pytorch/botorch on GitHub](https://github.com/pytorch/botorch/tree/main/botorch/test_functions) 🟢 source read at commit a1b1ad8
- [Ax benchmark problems — facebook/Ax on GitHub](https://github.com/facebook/Ax/tree/main/ax/benchmark/problems) 🟢 source read at commit 57efa05
