---
type: research-note
tags:
  - bayesian-optimisation
  - benchmarks
  - problem-framing
  - vgc
created_at: 2026-09-29
research_date: 2026-09-29
scope: full-team construction only
updated_at: 2026-10-01
scope_version_2: "full-team search unit; f = win rate vs the top-50 meta, behaviour-cloning policy both sides, as a fixed black box"
measured_data:
  - vgc-team-generator-pilot/results/bench_features.json
  - vgc-team-generator-pilot/results/vgc_domain_features.json
  - vgc-team-generator-pilot/results/ruggedness2.json
  - vgc-team-generator-pilot/results/interaction_order.json
  - vgc-team-generator-pilot/results/bench_order_conditioning.json
  - vgc-team-generator-pilot/results/meta_drift.json
  - vgc-team-generator-pilot/results/relevance_stationarity.json
  - vgc-team-generator-pilot/results/multimodality.json
preregistration: vgc-team-generator-pilot/docs/bench_features.md
---

# Version 3 — five more features: higher-order interaction, ill-conditioning, non-stationary relevance, meta drift, multimodality (2026-10-01)

Same scope and evidence labels as Version 2 (**D** declared, **M** measured with the same metric on both sides, **—** neither). Benchmark measurements: `scripts/bench_order_conditioning.py` (no battles). VGC: the 2026-09-30 interaction-order run, the 2026-09-28 ruggedness run, and the matchup database. The VGC runs scored against 36 of the top-50 files (a script bug fixed since); comparisons hold, absolute win rates are against 36 teams. Molecule tasks have no fixed coordinates, so neither design applies to them.

## Higher-order interaction and ill-conditioning

**Interaction order** (BBOB: separability). Four single edits around a starting point, all 16 on/off combinations; share of the variation explained by single edits (order 1), pairs (order 2) and three or four edits together (order 3+), corrected for noise. **Ill-conditioning** (BBOB: variable scaling, condition number). Noise-corrected effect of changing one variable, then largest ÷ median across variables. VGC is per edit type (six types), the benchmarks per coordinate.

| Benchmark | Order 1 · 2 · 3+ (random starts) | Order 3+ after a hill climb | Declared order | Largest ÷ median effect | Variables with no effect beyond noise |
| --- | --- | --- | --- | --- | --- |
| Contamination control | 0.967 · 0.032 · 0.001 **M** | 0.002 **M** | — | 1.4 **M** | 0 of 25 **M** |
| Ising sparsification | 0.997 · 0.003 · 0.000 **M** | 0.000 **M** | — | **7.3 M** | 0 of 24 **M** |
| Pest control, COMBO code | 0.978 · 0.023 · −0.001 **M** | −0.000 **M** | "more complex, higher-order interactions" **D** ([COMBO][combo]) | 1.9 **M** | 0 of 25 **M** |
| Pest control, Bounce code | 0.916 · 0.068 · 0.015 **M** | 0.004 **M** | as above | 1.7 **M** | 0 of 25 **M** |
| Weighted MaxSAT-60 | 1.000 · 0.000 · 0.000 **M** | 0.000 **M** | up to clause length, by its formula **D** | 1.2 **M** | 0 of 60 **M** |
| LABS-50 | 0.906 · 0.088 · 0.006 **M** | 0.014 **M** | up to 4, by its quartic energy **D** | 1.4 **M** | 0 of 50 **M** |
| Ackley-53 | 1.000 · 0.000 · 0.000 **M** | 0.000 **M** | — | 1.2 **M** | 0 of 53 **M** |
| **VGC full team** (four whole-Pokémon swaps; meta starts) | **0.871 · 0.116 · 0.013** [−0.001, 0.026] **M** | — | — | **2.9** (meta starts) · 1.9 (search-found) **M** | **5 of 6 edit types M**: only whole-Pokémon replacement moves win rate beyond battle noise |

**9. VGC carries the largest pair share in the set, and higher-order effects are not measurable.** Pairs explain 11.6% of VGC's variation under four swaps, more than any collected benchmark (LABS 8.8%, Bounce's pest control 6.8%). Three-way and higher effects are 1.3%, with an interval that includes zero; LABS and Bounce's pest control, though small, are measurably above zero. *Strength:* a pairwise surrogate (the project's ridge with species pairs) matches the measured structure; this is a realistic testbed for pairwise models. *Weakness:* claims that VGC needs higher-order models are not supported at this battle budget. One asymmetry: VGC's four edits touch four of six team slots, the benchmarks' four of 24–60 coordinates.

**10. VGC is ill-conditioned in a different way from the benchmarks: most edit types are lost in noise.** Ising sparsification has the widest exact spread (7.3×), but every benchmark variable has a measurable effect. In VGC only whole-Pokémon replacement exceeds battle noise; move, ability, item, alignment and Stat Point edits do not. *Strength:* a clear hierarchy of decisions (which Pokémon first) that search methods can exploit. *Weakness:* fine edits cannot be ranked without many battles, so local search below the species level is mostly blind. The median-based ratio understates this, because five of VGC's six effects are near the noise floor.

## Meta drift (stationarity), measured

| Benchmark | Stationarity |
| --- | --- |
| All collected benchmarks | Fixed instance per run or seed (Version 2 table) **D** |
| **VGC full team** | Over June–August 2026, **no measurable drift M**: 176 teams ranked by their score against the June meta teams vs the August meta teams agree at Spearman 0.904; random halves of the same columns agree at 0.908 (gap −0.004, 95% CI −0.045 to +0.039); mean shift −0.010. |

**11. The meta changes which teams are on the list, but over three months it did not change how teams rank.** The benchmarks are stationary by construction; VGC's objective is defined by a list that players rewrite. Within this window, scoring against June or August teams gave the same ranking within noise. *Strength:* results from one regulation snapshot transfer within that regulation's lifetime, as far as measured. *Weakness:* the window is one regulation; a new regulation changes the legal domain itself (Version 1, difference 7), which this does not test. The 176 rows are mostly search-found teams, weaker than the meta.

## Non-stationary relevance, measured

Does *which* variable matters depend on where you start? At each starting point, rank the variables by the mean size of a single edit's effect (`scripts/relevance_stationarity.py`, no battles). Two independent edit sets per starting point give two rankings. **Within**: agreement of the two rankings at the same starting point (what edit-to-edit spread and battle noise allow). **Between**: agreement across different starting points. A gap below zero means relevance changes with the starting point. VGC: the six edit types of the ruggedness runs (three edits each); benchmarks: every coordinate, 24 random starts.

| Benchmark | Within | Between | Gap (95% CI) |
| --- | --- | --- | --- |
| Contamination control | 1.00 **M** | 0.17 **M** | −0.83 [−0.87, −0.78] **M** |
| Ising sparsification | 1.00 **M** | 0.47 **M** | −0.53 [−0.63, −0.39] **M** |
| Pest control, COMBO code | 0.22 **M** | 0.00 **M** | −0.22 [−0.30, −0.14] **M** |
| Pest control, Bounce code | 0.44 **M** | 0.04 **M** | −0.40 [−0.48, −0.30] **M** |
| Weighted MaxSAT-60 | 1.00 **M** | 0.50 **M** | −0.50 [−0.55, −0.45] **M** |
| LABS-50 | 1.00 **M** | 0.02 **M** | −0.98 [−1.00, −0.95] **M** |
| Ackley-53 | 0.64 **M** | 0.02 **M** | −0.62 [−0.66, −0.59] **M** |
| **VGC, meta starting teams** | −0.04 **M** | 0.02 **M** | +0.06 [−0.11, +0.21] **M** |
| **VGC, search-found starting teams** | 0.28 **M** | 0.09 **M** | **−0.20 [−0.43, −0.04] M** |

No collected paper declares this property, so every cell is measured.

**12. Relevance shifts with the starting point everywhere; in VGC, battle noise hides it except among search-found teams.** Every benchmark's important coordinates change from one starting point to another (LABS almost completely, Ising and MaxSAT by half). This is the norm, not a VGC peculiarity. In VGC, at real meta teams, the ranking of edit types cannot be reproduced even at the same team (within −0.04): battle noise is larger than the differences between edit types, so relevance cannot be measured there at 384 battles. Among search-found teams the ranking is reproducible (0.28) and shifts between teams (gap −0.20, interval below zero). *Strength:* context-dependent relevance, the property behind "Speed matters on some teams and not others", is present and measurable where effects are large enough. *Weakness:* at the meta, detecting it needs far more battles per edit; a method that relies on learning which fields matter at a given team will be noise-limited. VGC ranks six edit types, the benchmarks 24–60 coordinates; the edit-type level cannot test single-field claims such as Speed under Trick Room.

## Multimodality, measured

Version 2 said multimodality could not be characterised. It can be measured as the number of **distinct endpoints of hill climbs** from many starting points (BBOB groups its functions by the same property). One protocol on both sides (`scripts/multimodality.py`): 24 random starts; first improvement with random single edits; a move is accepted only if it improves, beyond noise on noisy tasks; stop after 40 rejected proposals in a row or 120 proposals. Benchmarks also get an exhaustive single-edit check of each endpoint: a capped climb can stop before it reaches a true local optimum, and then its endpoint partly measures the cap.

| Benchmark                                                         | Distinct endpoints from 24 climbs                                                                             | Endpoints that are true local optima                                   |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Contamination control                                             | 24 **M**                                                                                                      | 75% **M**                                                              |
| Ising sparsification                                              | **14 M**                                                                                                      | 88% **M**                                                              |
| Pest control, COMBO code                                          | 24 **M**                                                                                                      | 0% **M**                                                               |
| Pest control, Bounce code                                         | 24 **M**                                                                                                      | 0% **M**                                                               |
| Weighted MaxSAT-60                                                | 24 **M**                                                                                                      | 0% **M**                                                               |
| LABS-50                                                           | 24 **M**                                                                                                      | 50% **M**                                                              |
| Ackley-53                                                         | 24 **M**                                                                                                      | 4% **M**                                                               |
| **VGC full team** (6 meta starts, 196 battles per evaluated team) | 6 of 6 **M**, but each climb accepted at most one edit (5 accepted one, 1 none) before 40 rejections in a row | no exhaustive check possible: thousands of legal single edits per team |

**13. Where climbs converge, the benchmarks are clearly multimodal; VGC's climbs cannot move far enough to tell.** Contamination control's converged climbs all end on different optima, and Ising's 24 climbs end on 14. Pest control, MaxSAT and Ackley never converge within 120 proposals, so their 24 endpoints say "not finished", not "24 modes". On VGC each climb took at most one step: at 196 battles per team, a move must beat its parent by about 0.10 win rate (1.96 standard errors of the difference) to count, and only one such edit turned up per start in 40–71 proposals. The six endpoints are therefore six starting points, not six modes. *Strength:* the same instrument runs on both sides. *Weakness:* local search on VGC is noise-limited before it is mode-limited; resolving multimodality would need several times more battles per comparison.


# Version 2 — objective in scope: feature table and eight differences (2026-09-30)

The search unit is one full team (species, ability, item, moves, Stat Points, alignment). The objective is now fixed: $f$ = **win rate** of the team against the top-50 meta teams, with the behaviour-cloning battle policy playing both sides, treated as a black box. Team preview and the battle produce the score; they are not searched. The comparators are only the benchmarks collected in Version 1 (plus the welded-beam task, the one collected example with black-box constraints). The feature definitions are borrowed from COCO/BBOB (Black-Box Optimization Benchmarking); none of its functions are used.

**Evidence labels.** **D** = declared by the benchmark's paper or its reference code. **M** = measured on this machine with the *same metric on both sides* (scripts `bench_features.py`, `vgc_domain_features.py`; VGC landscape data reused from the 2026-09-28 ruggedness run, no new battles). **—** = neither stated nor measured. Measured benchmarks were the ones whose code runs in milliseconds; XGBoost-MNIST, SVM-Slice, NAS-Bench-101, arithmetic expressions and DRD3 docking are declared only.

## Feature definitions borrowed from COCO/BBOB

| Feature                          | What COCO/BBOB means by it                                                                                                                                                          | Same-metric measurement used here                                                                                                                                                                                                                                                                                                |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Smoothness                       | BBOB "regularity": simple formulas are highly regular; the suite adds small smooth irregularities and a few highly irregular functions ([noiseless definitions, App. A.3][bbob-f]). | Noise-corrected single-edit correlation ρ = 1 − (E[d²] − E[r²]) / (2·var_f): d = edit − anchor, r = replicate − anchor. ρ = 1 is perfectly smooth; ρ ≤ 0 means one edit moves $f$ as much as picking an unrelated start point. 24 start anchors + 16 hill-climbed anchors, 15 edits, 3 replicates, identical on every landscape. |
| Noise                            | Three noise models: multiplicative Gaussian and uniform, additive "seldom Cauchy" outliers, at moderate and severe strength ([noisy definitions, §0.3][bbob-n]).                    | Noise share σ² / (σ² + var_f), σ² from replicate evaluations of the same point.                                                                                                                                                                                                                                                  |
| Multimodality / global structure | Functions grouped as unimodal, multimodal with adequate global structure, and multimodal with weak global structure ([suite page][bbob-suite]).                                     | Not measured on either side (see difference 8).                                                                                                                                                                                                                                                                                  |
| Evaluation cost                  | COCO counts runtime in **number of function evaluations** and deliberately does not use CPU or wall-clock time ([performance assessment, §1][coco-perf]).                           | Wall-clock seconds per evaluation, one process, this machine.                                                                                                                                                                                                                                                                    |
| Constraint type                  | BBOB has none: a box $[-5,5]^D$, with penalty boundary handling in the noisy suite. Known vs black-box is taken from [PR][pr].                                                      | Fraction of uniform draws from the naive encoding that are valid.                                                                                                                                                                                                                                                                |
| Dimensionality / cardinality     | Functions scalable in dimension (2–40 in the standard runs).                                                                                                                        | Encoding variables, log₁₀ domain size, log₁₀ single-edit neighbourhood size.                                                                                                                                                                                                                                                     |
| Known optimum                    | Every function has a known $f_\text{opt}$; the target is $f_\text{opt} + 10^{-8}$ ([App. A.6][bbob-f]).                                                                             | Best known value where none is declared.                                                                                                                                                                                                                                                                                         |
| Stationarity                     | One fixed function per run; **instances** (new shift, rotation and $f_\text{opt}$) give repetition without allowing exploitation of artificial properties ([§1.1][coco-perf]).      | Declared only.                                                                                                                                                                                                                                                                                                                   |

## Feature table

Smoothness is reported as ρ at start anchors · ρ at hill-climbed anchors (bootstrap 95% intervals in the results file). Molecule edits replace one SELFIES token (a STONED-style point mutation); binary and categorical edits change one variable.

| Benchmark | Smoothness ρ | Noise | Evaluation cost |
| --- | --- | --- | --- |
| Contamination control (25 binary) | 0.96 · 0.95 **M** | Randomness drawn once per instance, so deterministic **D** (COMBO, Bounce code); share 0 **M** | 0.08 ms **M**; called expensive, 100 simulations per call **D** ([BOCS][bocs]) |
| Ising sparsification (24 binary) | 0.93 · 0.92 **M** | Exact Kullback–Leibler divergence by enumeration, deterministic **D**; 0 **M** | 0.76 ms **M**; called expensive **D** ([BOCS][bocs]) |
| Pest control, COMBO code (25 × 5) | 0.99 · 0.99 **M** | Fresh random draws on every call **D**; share 0.003 **M** | 0.24 ms **M** |
| Pest control, Bounce code | 0.99 · 0.99 **M** | Seeded on every call, so deterministic **D**; 0 **M** | 0.24 ms **M** |
| Weighted MaxSAT-60 | 0.995 · 0.999 **M** | Deterministic; 0 **M** | 0.79 ms **M** |
| LABS-50 (low-autocorrelation binary sequences) | 0.98 · **0.67** **M** | Deterministic; 0 **M** | 0.03 ms **M** |
| Ackley-53 (50 binary + 3 continuous) | 0.998 · 0.98 **M** | Uniform noise of size 10⁻⁶ **D** (Bounce code); 0.000 **M** | 0.01 ms **M** |
| XGBoost on MNIST | — | — | One model training per call **D** |
| SVM feature selection on Slice | — | — | One model training per call **D** |
| NAS-Bench-101 | — | Every architecture trained 3 times "to obtain a measure of variance" **D** ([NAS-Bench-101][nasbench]) | Table lookup; the table cost over 100 TPU-years **D** |
| Penalised logP | **−4.2 · −5.8** **M** (median edit 0.7 SD, largest 17 SD) | Deterministic; 0 **M** | 0.60 ms **M** |
| Arithmetic expressions | — | Deterministic loss log(1 + MSE) on fixed data **D** ([GrammarVAE][gvae]) | — |
| GuacaMol Perindopril / Zaleplon multi-property objective | 0.43 · 0.03 / 0.86 · 0.53 **M** | Deterministic; 0 **M** | 0.24 / 0.56 ms **M** |
| Therapeutics Data Commons DRD2 | 0.99 · **0.08** **M** (10% of edits at start anchors leave the score unchanged) | Deterministic; 0 **M** | 1.7 ms **M**; Practical Molecular Optimization caps these cheap scores at 10,000 calls **D** ([PMO][pmo]) |
| DRD3 docking | — | Deterministic in DOCKSTRING, which fixes seeds; the Therapeutics Data Commons wrapper does not, giving "considerable variance between runs" **D** ([DOCKSTRING][dockstring]) | About 15 s per molecule on eight CPUs **D** |
| **VGC full team** | **0.96 · 0.92** over all six edit types **M**; species replacement 0.86 · 0.75; ability, alignment and Stat Points 0.99–1.00 **M** | Binomial, variance p(1 − p)/n. Share **0.061 at 384 battles M**; 0.51 at 24 battles (derived from the 384-battle measurement); 0.73 among uniform legal teams at 24 battles **M** (2026-08-29 run) | **7.7 s at 384 battles** on the 50-battle/s pool (12 cores) **M**; 0.5 s at 24 battles |

| Benchmark | Constraint type | Dimensionality / cardinality | Known optimum | Stationarity |
| --- | --- | --- | --- | --- |
| Contamination control | A chance constraint relaxed into a penalty **D** ([BOCS][bocs]); every input valid | 25 binary; 10^7.5 points; 25 neighbours | None declared; papers report best found **D** | Fixed per run; COMBO draws a new instance per seed, Bounce fixes seed 42 **D** |
| Ising sparsification | None; every input valid | 24 binary; 10^7.2; 24 | None; BOCS reports average best, not regret, because of search-space size and cost **D** | New random couplings per seed **D** (COMBO) |
| Pest control | None | 25 × 5 categorical; 10^17.5 (COMBO's 21 stages: 4.77 × 10^14 **D**); 100 | None declared | Fixed |
| Weighted MaxSAT-60 | None on inputs; the clauses are the objective | 60 binary; 10^18.1; 60 | **Known**: the instance file states optimum 38,928 **D** | Fixed instance |
| LABS-50 | None | 50 binary; 10^15.1; 50 | **Known**: energy 153, merit factor 8.170, by exhaustive search **D** ([Packebusch & Mertens][labs]) | Fixed |
| Ackley-53 | Box bounds | 50 binary + 3 continuous | **Known** by construction (Ackley minimum 0) | Fixed |
| Welded beam ([PR][pr]) | **Five black-box outcome constraints D** | Categorical + binary + ordinal | — | Fixed |
| NAS-Bench-101 | Known structural limits: at most 7 vertices and 9 edges per cell **D** | 423k unique architectures, 10^5.6 **D** | **Known**: every architecture is in the table | Fixed table |
| Molecule tasks (penalised logP, GuacaMol, DRD2, docking) | Validity known and cheap (RDKit). Uniform SMILES characters: **0 / 2,000 valid M**; uniform SELFIES tokens: **2,000 / 2,000 M** | Variable-length strings or graphs; SELFIES neighbourhood = tokens × 69 **M** | None declared; GuacaMol scores bounded to [0, 1] **D** ([GuacaMol][guacamol]) | Fixed function; GSK3B and JNK3 score files **fail to load** under scikit-learn 1.9.1 **M** |
| **VGC full team** | Known, cheap, deterministic validator, with per-member conditional rules **and** team-level clauses. Uniform unconditioned fields: **0 / 2,000 legal M**; per-member legal fields: 1,354 / 2,000 (failures: Item Clause 435, Species Clause 203); plus clauses while drawing: 471 / 500 (the 29 failures are errors in our own ability table) **M** | 84 encoding variables; ≤ 10^112 teams (Species Clause applied, Item Clause ignored) **M**; per member 10^16.7 configurations, 10^8.1 of it Stat Point allocations; single-edit neighbourhood 10^8.9 without species replacement, 10^19.9 with **M**; each team has 6!·(4!)^6 ≈ 1.4 × 10^11 encodings | **None**; upper bound 1.0. Best known **0.691 ± 0.018** (a real team, 2,400 battles), best search-found 0.611 **M** (2026-09-28 counter grid) | Frozen by construction: behaviour-cloning checkpoint, top-50 list, one Showdown build, opponent schedule **D** (protocol). The real meta changes with each regulation **D** ([handbook §2.2][vgc-rules]) |

**Multimodality** gets no column: none of the collected Bayesian optimisation papers declares it per task (CASMOPOLITAN says only that high-dimensional mixed spaces make objectives highly multimodal, as a blanket remark), and it was not measured for VGC.

## Eight differences, and what each says about VGC as a research problem

**1. Noise is part of the problem, and its level is a dial.** Ten of the eleven measured benchmark implementations show no measurable noise (Ackley-53's 10⁻⁶ term rounds to zero); the eleventh (COMBO's pest control) has a noise share of 0.003, and Bounce's code for the same task removes it by reseeding. Where other benchmarks are noisy, it is an implementation choice (pest control, the docking wrappers) or a stored repeat (NAS-Bench-101). VGC's noise is binomial, with known variance p(1 − p)/n, and the experimenter sets n. BBOB's Gaussian noise is multiplicative, so it vanishes at the optimum; binomial variance is largest at p = 0.5, which is where the best teams (0.46–0.69) sit. *Strength:* a real, analysable noise model for studying replication versus exploration, rather than injected noise. *Weakness:* at 24 battles about half the observed variance between real teams is noise, so single evaluations misrank teams. Any comparison must fix battles per evaluation.

**2. Evaluation cost is real, sits between the synthetic tasks and docking, and is tied to noise.** The measured benchmarks cost 0.01–1.7 ms per call, including the two that BOCS calls expensive. Practical Molecular Optimization imposes expense as a 10,000-call cap on cheap scores. VGC costs 7.7 s at 384 battles: about 4,500 times the slowest measured benchmark, and cheaper than docking's roughly 15 s on eight CPUs. *Strength:* sample efficiency matters for a genuine reason, yet thousands of evaluations per hour still allow full comparisons with replicates. *Weakness:* cost and noise move together through the battle count, so COCO's cost unit (one evaluation) is not fixed. VGC budgets should be counted in battles.

**3. Smoothness falls inside the benchmark range, and differs by field.** With the same statistic, VGC (0.96 at meta anchors, 0.92 at search-found anchors) is rougher than pest control, MaxSAT and Ackley-53, level with contamination control at start points (0.96) and rougher near good points (0.92 vs 0.95), between Ising and LABS, and much smoother than the molecule tasks under a string edit (penalised logP −4.2, Perindopril 0.43). Every landscape with a clear gap is rougher near good points (LABS 0.98 → 0.67, DRD2 0.99 → 0.08, VGC 0.96 → 0.92). What is unusual is that one VGC team mixes fields with very different roughness: species replacement 0.75–0.86, while ability, alignment and Stat Points stay at 0.99–1.00. *Strength:* a natural test of whether a surrogate learns a different length scale for each field type. *Weakness:* overall non-smoothness is not a distinguishing property; the molecule tasks are rougher by this measure. ρ also depends on which start points are used.

**4. The naive encoding is almost entirely illegal (as with SMILES), but legality is known and cheap.** Every binary and categorical benchmark accepts all inputs. VGC with independent uniform fields gave 0 legal teams in 2,000, as uniform SMILES strings gave 0 valid molecules in 2,000. SELFIES fixes molecules completely (2,000 / 2,000). The per-member analogue fixes VGC only partly: drawing each member from its own legal sets gives 68% legal teams, and the remainder fail the team-level clauses. Those clauses concern pairs of members, which a per-token grammar does not cover. *Strength:* feasibility can be studied separately from the objective, with a deterministic validator instead of the black-box constraint models PR needs. *Weakness:* off-the-shelf categorical Bayesian optimisation on the raw encoding would propose only illegal teams. Results depend on the chosen encoding and repair rule, so both must be reported.

**5. The nominal size exceeds every collected benchmark, but most of it lies in fields that barely change win rate.** The combinatorial benchmarks have 24–60 variables and at most 10^18.1 points; VGC has 84 variables and up to 10^112 teams. Its single-edit neighbourhood (10^8.9–10^19.9) cannot be enumerated, whereas Hamming-1 neighbourhoods of 24–100 can. Stat Point allocations supply 10^8.1 of each member's 10^16.7 configurations, about half the orders of magnitude, and Stat Point edits have ρ = 0.99–1.00. *Strength:* it tests acquisition optimisation at a scale where exhaustive local moves are impossible. *Weakness:* the headline size overstates difficulty. The effective dimensionality is much smaller, and the count repeats every team about 1.4 × 10^11 times unless the encoding is canonicalised.

**6. No optimum and no regret, but a human reference that currently beats search.** MaxSAT-60, LABS-50, Ackley-53 and NAS-Bench-101 come with known optima, as every BBOB function does. VGC has none, and no bound tighter than 1.0. It does have a reference that most benchmarks lack: a distribution of teams built by human players, whose best member (0.691) still beats the best search-found team (0.611). *Strength:* "beats the best human-built team" is a meaningful bar. *Weakness:* regret cannot be computed, and the bar moves whenever more human teams are evaluated.

**7. Frozen by construction, with real instances instead of random shifts.** The collected benchmarks are fixed functions, and COCO and COMBO obtain instances by random redraws. VGC's $f$ is fixed only because the checkpoint, the top-50 list, the Showdown build and the opponent schedule are held fixed; the Therapeutics Data Commons files that no longer load show that other benchmarks drift in the same way. Each regulation and each meta snapshot is a real instance. *Strength:* instance-transfer questions (does a method re-tuned on a new meta still work?) arise naturally. *Weakness:* a team optimised against one frozen snapshot and one battle policy is not validated. Ladder validation remains the only ground truth, and every result has to name its snapshot.

**8. Multimodality cannot be characterised, on either side.** BBOB groups its functions by multimodality. None of the collected Bayesian optimisation tasks declares it. For VGC, certifying even one local optimum would mean evaluating a neighbourhood of 10^8.9 or more teams at 384 battles each. *Weakness (shared):* a statement such as "VGC has many local optima" has no evidence behind it. *Strength:* none can be claimed.

## What this does not show

- ρ is measured under one edit definition per landscape; a different neighbourhood (for example, a graph edit for molecules) could reorder the molecule rows.
- The VGC numbers are conditional on the behaviour-cloning policy and the top-50 list, and are not ladder results.
- Benchmarks marked — were not run; a blank there is missing evidence, not evidence of absence.

[bbob-f]: https://inria.hal.science/inria-00362633
[bbob-n]: https://inria.hal.science/inria-00369466
[bbob-suite]: https://numbbo.github.io/coco/testsuites/bbob
[coco-perf]: https://arxiv.org/abs/1605.03560
[nasbench]: https://arxiv.org/abs/1902.09635
[labs]: https://arxiv.org/abs/1512.02475

Sources added in Version 2 (all 🟢): [Hansen et al., Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions][bbob-f] (INRIA research report); [Hansen et al., … Noisy Functions Definitions][bbob-n] (INRIA research report); [COCO BBOB test suite][bbob-suite]; [Hansen et al., COCO: Performance Assessment][coco-perf] (arXiv); [Ying et al., NAS-Bench-101: Towards Reproducible Neural Architecture Search][nasbench] (ICML 2019); [Packebusch & Mertens, Low Autocorrelation Binary Sequences][labs] (Journal of Physics A 2016). Reference code read for the declared noise and optimum cells: COMBO `experiments/test_functions`, Bounce `bounce/benchmarks.py`, the COMBO MaxSAT instance `frb-frb10-6-4.wcnf`, LOL-BO `mol_utils.py`. Other sources are listed in Version 1.

# Version 1 — construction only, no objective (2026-09-29)

# Bayesian optimisation benchmarks versus VGC full-team construction

This collection uses published benchmark definitions to assess **VGC team building's strengths and weaknesses as an optimisation research problem**. The strongest case is its combination of configured members, conditional choices, discrete resource allocation, and constraints across a complete roster. The literature does not establish that these ingredients are individually unique, or that VGC is harder than existing benchmarks.

The benchmark facts below come from primary papers, author implementations, and official game documentation. The numbered comparisons are **inferences from those facts**, not empirical findings about an implemented VGC benchmark.

## Scope and meaning of “common”

The search unit is one complete team: species, ability, item, moves, Stat Points, and alignment. Team preview and battle decisions are outside this note. No win-rate objective, opponent distribution, battle policy, evaluation noise, or evaluation cost is assumed.

“Recurring” means that the task or clearly identified task family appears in at least two distinct BO papers inspected here. This is a targeted literature collection, not a citation-frequency ranking or an exhaustive survey through 2026. Recurrence of a family does not mean that every paper uses the same instance. Tasks found only in a benchmark suite or a single BO study are labelled separately.

Three levels must stay separate:

- A **benchmark task** supplies a candidate domain and a scoring problem: for example, 25-stage pest control or a particular GuacaMol objective.
- A **suite or provider** groups or exposes tasks: for example, MCBO, GuacaMol, PMO, or Therapeutics Data Commons (TDC).
- A **method or representation** determines how to search: COMBO, CASMOPOLITAN, a molecular VAE, and LOL-BO are not themselves benchmark tasks.

## 1. Recurring combinatorial and mixed-variable benchmarks

| Benchmark                                       | What one candidate contains and what is optimised                                                                                                                                                                                                                                | Evidence of recurring BO use; qualifications                                                                                                                                                                                                                            |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ising-model sparsification**                  | Binary decisions retain or remove edges from a given Ising model. The objective trades distributional fidelity, measured by KL divergence, against sparsity. In the reported BOCS case, 24 edges belong to a 4×4 grid. These variables are **edge selections, not Ising spins**. | [BOCS][bocs], then [COMBO][combo]. A recurring graph-subset task with a fixed starting graph.                                                                                                                                                                           |
| **Contamination control**                       | A binary intervention vector across 25 stages of a food-supply process. The objective combines intervention cost with contamination penalties; BOCS also uses a sparsity term.                                                                                                   | [BOCS][bocs], [COMBO][combo], [CASMOPOLITAN][casmo], [Bounce][bounce]. Penalty-based constraints must not be described as a domain in which every infeasible input is forbidden.                                                                                        |
| **Pest control**                                | One of five intervention choices at each stage; minimise pest-related loss and control expenditure. Stage decisions interact through the process.                                                                                                                                | [COMBO][combo], [CASMOPOLITAN][casmo], [BODi][bodi], [Bounce][bounce], [MCBO][mcbo]. The final COMBO paper reports **21 stages**; later standard versions use **25**. Larger variants also exist.                                                                       |
| **Weighted MaxSAT**                             | Assign Boolean values to a fixed set of variables to maximise satisfied-clause weight, or minimise an equivalent unsatisfied-weight loss.                                                                                                                                        | [COMBO][combo], [CASMOPOLITAN][casmo], [BODi][bodi], [Bounce][bounce]. Preserve the clause instance: 60-variable and 125-variable tasks are not interchangeable.                                                                                                        |
| **Low-autocorrelation binary sequences (LABS)** | A fixed-length binary sequence; maximise its autocorrelation merit factor. Positions have a defined role.                                                                                                                                                                        | The 50-variable task appears in [BODi][bodi] and [Bounce][bounce]. This already supplies interactions within a structured candidate.                                                                                                                                    |
| **Discretised or mixed analytic functions**     | Known numerical landscapes, with selected coordinates made binary, ordinal, or categorical. Examples include grid Branin, mixed Ackley, Rosenbrock, and Func-2C/Func-3C.                                                                                                         | [COMBO][combo], [CoCaBO][cocabo], [CASMOPOLITAN][casmo], [PR][pr], [BODi][bodi], [Bounce][bounce]. **Ackley-53** uses 50 binary and three continuous coordinates; PR's **Ackley-13** uses ten binary and three continuous coordinates. Name the encoding and dimension. |
| **XGBoost hyperparameter tuning on MNIST**      | A model configuration combining categorical and numerical choices; optimise predictive performance on the specified data split.                                                                                                                                                  | [CoCaBO][cocabo], [CASMOPOLITAN][casmo], [MCBO][mcbo]. Dataset, split, ranges, and training protocol are part of the benchmark.                                                                                                                                         |
| **SVM feature selection and tuning on Slice**   | Fifty binary feature-inclusion choices plus three continuous SVM hyperparameters; minimise held-out prediction error.                                                                                                                                                            | [PR][pr], [BODi][bodi], [Bounce][bounce], [MCBO][mcbo]. This combines subset selection with configuration. It is **different from CoCaBO's SVM-Boston task**.                                                                                                           |
| **Neural architecture search (NAS)**            | A neural-network architecture, involving operations and connectivity; score its predictive performance under a training or tabular evaluation protocol.                                                                                                                          | [COMBO][combo] and [CoCaBO][cocabo] establish recurrence of the **application family**. CoCaBO uses NAS-Bench-101; the two papers must not be treated as using an identical search space.                                                                               |

This inventory gives stronger evidence for recurring use than merely finding a familiar optimisation function in one paper. It also shows why a single category called “standard combinatorial benchmarks” would hide important differences between subsets, ordered sequences, logical assignments, configurations, and graphs.

A useful encoding caution: CoCaBO represents NAS-Bench-101 with categorical node operations, continuous edge-ranking proxies, and an integer edge count. Decoding selects the highest-ranked edges; the architecture itself remains discrete. A numerical representation is therefore not evidence of continuous underlying design freedom. [CoCaBO supplement, §G/Table 3][cocabo-supp]

### Relevant examples with narrower recurrence evidence

These help check the comparison for overclaims; they are not promoted to equally common individual tasks.

| Source             | Additional tasks                                                                                                                                                                                                                                | Why they matter to the comparison                                                                                                                                                                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCBO suite**     | Logic-synthesis operator sequences; antibody sequence design; RNA inverse folding. Circuit scores include size/depth objectives; antibody design uses predicted binding energy; RNA design matches a predicted secondary structure to a target. | Existing BO tasks already include biological structure and explicit input-validity checks. See the [suite][mcbo] and its [antibody][antibody-code], [RNA][rna-code], and [circuit][circuit-code] implementations.                                                         |
| **PR experiments** | Welded-beam design, cellular-network configuration, direct arylation, and electrospun oil-sorbent design.                                                                                                                                       | These include categorical/ordinal/continuous mixtures, black-box outcome constraints, and multiple objectives. Welded-beam design has five structural constraints. Their presence in [PR][pr] establishes relevant examples, not widespread recurrence of every instance. |
| **LaMBO**          | Latent multi-objective design of molecules and fluorescent proteins.                                                                                                                                                                            | [LaMBO][lambo] provides a non-small-molecule generative BO comparator. Recurrence of each protein task was not established in this collection.                                                                                                                            |

## 2. Recurring latent-space and generative benchmarks

The **original object** and the **latent vector used by a method** are different spaces. A molecular graph does not become an intrinsically continuous design merely because an optimiser operates through a continuous decoder input.

| Benchmark or collection | Candidate and scoring task | Evidence of BO use and precise inclusion |
| --- | --- | --- |
| **Penalised logP** | A molecule; maximise calculated lipophilicity with synthetic-accessibility and long-cycle penalties. Fix the exact normalisation and implementation. | Recurs in [GrammarVAE][gvae], [JT-VAE][jtvae], and [LOL-BO][lolbo-code]. It is a particular molecular proxy objective, not evidence of therapeutic usefulness. |
| **Arithmetic-expression fitting / symbolic regression** | A grammar-generated expression; minimise discrepancy from a target function's observations. One recurring target is `1/3 + x + sin(x*x)`. | [GrammarVAE][gvae], [LOL-BO][lolbo-code], and [InvBO][invbo]. Fix the grammar, target, input sample, expression limits, and loss. |
| **GuacaMol goal-directed tasks** | Generate molecules for predefined similarity, rediscovery, isomer, substructure, or multi-property scores. Concrete recurring examples are **Perindopril MPO, Ranolazine MPO, and Zaleplon MPO**. | These appear in [LOL-BO][lolbo-code] and [InvBO][invbo-code]; [NF-BO][nfbo] supplies further use. GuacaMol is a suite, and its separate **distribution-learning tests** must not be conflated with goal-directed optimisation. See its [paper][guacamol] and [objective code][guacamol-code]. |
| **TDC oracle tasks, especially through Practical Molecular Optimization (PMO)** | Molecules scored by objectives including **DRD2, GSK3B, and JNK3** predicted activity, QED, and GuacaMol-derived functions. Predicted activity is a proxy, not an experimental assay result. | [PMO][pmo] evaluates 25 algorithms on 23 tasks, including GP BO, with a 10,000-query budget. [NF-BO][nfbo], a latent BO method, evaluates all 23 PMO tasks; Appendix A explicitly reports DRD2/GSK3B/JNK3. [TDC's oracle documentation][tdc-oracles] defines the functions. This does **not** make every TDC dataset an optimisation benchmark. |
| **DRD3 molecular docking** | Generate a ligand for a specified dopamine D3 receptor docking task. A favourable computational docking score is the target. | [LOL-BO][lolbo] uses a TDC docking task; [InvBO][invbo] uses DRD3 with Dockstring. The named target recurs, but receptor preparation, docking engine/settings, and filters must be fixed before treating scores as comparable. See [TDC's docking protocol][tdc-docking]. |

**These rows overlap.** GuacaMol defines task functions, TDC exposes molecular oracles and docking tasks, and PMO assembles an evaluation collection. Docking is a kind of scoring procedure with many possible instances. Counting the same objective once under each label would inflate coverage.

[DOCKSTRING][dockstring] is a useful additional docking reference: it defines design tasks such as druglike F2 and selective JAK2. A [subsequent BO study][dockstring-bo] searches a fixed compound library for these tasks. That is evidence of molecular **library search**, not evidence that this particular experiment performs latent generation. Docking scores remain computational proxies for aspects of binding, not measured affinity or clinical success.

## 3. The VGC construction object used for comparison

For this comparison, a full team is instantiated as **six configured members**. This is the research unit; the official handbook permits four to six depending on tournament format, so six should not be presented as a universal tournament minimum. Each member has the six kinds of choices specified in the request, with species including relevant forms.

The official construction rules supply concrete dependencies: abilities and moves must be available to the selected Pokémon, two members cannot share a National Pokédex number, and nonempty held items cannot be duplicated. Alignment is recorded as part of a team list. The applicable regulation determines eligibility. These are construction facts, independent of any assumption about how good a team is. [Official tournament handbook, §§2.1–2.5][vgc-rules]

The request's Stat Points/alignment terminology matches the Champions construction schema. As an explicit implementation reference, Pokémon Showdown's Champions rules at commit `a5df8274e85b0889bf2a9b3422a08b39732374fc` use a per-member total cap of 66 Stat Points and a per-stat cap of 32. These are **integer allocations**, not free real-valued coordinates. This cites a simulator implementation, not an official game specification; an eventual benchmark should validate and pin its own rules. [Total limit][sp-total], [per-stat validation][sp-validator], [representation documentation][team-format]

One useful formalisation is:

\[
t_i=(s_i,a_i,h_i,M_i,p_i,\ell_i),\qquad
T=(t_1,\ldots,t_6)\in\mathcal F_R,
\]

where \(\mathcal F_R\) contains teams legal under a fixed ruleset \(R\). Species-conditioned choices and roster-level restrictions make this a constrained domain, rather than an unrestricted Cartesian product of the displayed fields. The tuple is an encoding: whether different tuple orders represent the same team must be specified separately.

**This defines a domain, not an objective function.** The following strengths and weaknesses concern construction structure and benchmark readiness. They do not infer landscape properties from battles.

## 4. Differences derived from the benchmark collection

Here, a **strength** is a useful research property that VGC could expose; a **weakness** is a limitation in clarity, generality, or reproducibility. Neither is a measured ranking of difficulty.

### 1. A team combines membership selection with configuration of every selected member

**Benchmark starting point:** Ising sparsification selects from an existing edge set; MaxSAT assigns values to existing logical variables. SVM feature selection goes further by combining subset choice with model hyperparameters. [BOCS][bocs], [BODi][bodi]

**VGC difference:** Selecting the roster does not finish the candidate: every selected member still has its own ability, item, moves, allocation, and alignment. A list of high-scoring individual Pokémon would not by itself specify a legal complete team.

**Strength:** It supports studying joint selection and configuration within one interpretable object. **Weakness:** The contribution becomes hard to identify if improvements could come from changing either roster selection or configuration. SVM and NAS also combine structural and configuration choices, so this is a difference from simpler subset tasks, not a unique property of VGC.

### 2. Legality combines species-conditioned choices with constraints across the roster

**Benchmark starting point:** Binary MaxSAT assignments have a straightforward input domain; contamination control penalises undesirable outcomes. PR's welded-beam task instead uses black-box structural constraints. These are different meanings of “constrained.” [COMBO][combo], [PR][pr]

**VGC difference:** The schema above has both conditional legal choices within members and restrictions involving multiple members. A species replacement can invalidate its retained moves; independently valid members can clash over a held item.

**Strength:** Construction legality can be audited separately from team quality, allowing precise analysis of invalid proposals and repair. **Weakness:** A repair operator can alter the effective search distribution, making a gain look like better optimisation when it is better constraint handling. This is shared territory: MCBO supports cheap validity predicates, and JT-VAE enforces chemical validity during generation. [MCBO][mcbo], [JT-VAE][jtvae]

### 3. Numerical choices are bounded allocations, not the continuous inputs of many mixed benchmarks

**Benchmark starting point:** Mixed Ackley, XGBoost tuning, and SVM tuning include genuine continuous coordinates in their stated search spaces. [CoCaBO][cocabo], [CASMOPOLITAN][casmo], [PR][pr]

**VGC difference:** Stat Points sit on an integer allocation grid with a shared per-member budget; alignment is a categorical choice. Numerical-looking fields do not make this domain continuously adjustable.

**Strength:** VGC exposes the interaction between categorical configuration and constrained integer allocation. **Weakness:** A continuous BO relaxation needs an explicit rounding/repair rule and duplicate accounting; its results cannot be interpreted as direct optimisation of a continuous domain. Ordinal engineering tasks already provide related challenges, so “mixed-variable” alone is not a distinguishing claim.

### 4. The roster has fixed cardinality, whereas molecule and expression generation can change structural size

**Benchmark starting point:** Molecular generation builds graphs, and expression fitting builds syntax trees, whose sizes and structures can vary within the implementation's limits. [GrammarVAE][gvae], [JT-VAE][jtvae]

**VGC difference:** Under the six-member research unit, the roster length is fixed and its member schema is known. The problem changes identities and configurations rather than growing an arbitrary molecular graph or expression tree.

**Strength:** It can test substantial heterogeneity without also requiring variable-size generation. **Weakness:** Success would provide limited evidence about graph-growth, topology-generation, or length-extrapolation problems. Fixed size is already shared with LABS, pest control, and fixed-length biological sequence tasks; it is not a difference from all BO benchmarks.

### 5. Construction identity may permit permutations that ordered-sequence benchmarks do not

**Benchmark starting point:** Pest-control stages and LABS positions carry meaning; reordering their entries generally changes the candidate. [BODi][bodi], [Bounce][bounce]

**VGC difference:** If the construction benchmark defines team identity by its configured members, changing their storage order represents the same roster. This is a **modelling convention to declare and test**, not an assertion about equivalence in every downstream system. The same care is needed before treating move-slot permutations as equivalent.

**Strength:** The benchmark could test whether methods avoid spending evaluations on multiple encodings of one construction. **Weakness:** Unstated ordering conventions can inflate diversity or distort distances. Molecular strings and graph representations already raise representation-equivalence issues, so symmetry handling is shared with the molecular comparators rather than absent from them. [JT-VAE][jtvae]

### 6. The meaning of a “small edit” depends strongly on the field being changed

**Benchmark starting point:** Analytic mixed functions provide declared coordinate geometry, while binary benchmarks often use bit changes. Bounce demonstrates sensitivity to flipped binary encodings and permuted category labels. [Bounce][bounce]

**VGC difference:** Reallocating one Stat Point, changing an item, and replacing a species are all edits, but they have different construction consequences. The last can change which other fields are legal. A flat coordinate count does not capture that difference.

**Strength:** VGC can test representations and neighbourhoods that respect typed, conditional edits. **Weakness:** There is no justified universal team distance in the supplied definition. An embedding advantage could depend on a favourable encoding. This does not prove a rugged or nonsmooth team-quality landscape; that would require an objective and measurements. Related semantic-distance problems already occur in NAS and molecular design. [CoCaBO][cocabo], [LOL-BO][lolbo]

### 7. The legal domain must be tied to a regulation snapshot

**Benchmark starting point:** A MaxSAT instance pins clauses; an expression task pins a grammar and target; a docking task pins a receptor and protocol. [COMBO][combo], [InvBO][invbo]

**VGC difference:** “VGC team building” identifies a family whose permitted species and options depend on the selected regulation and game version. The official rules explicitly accommodate regulation changes. [Tournament handbook, §2.2][vgc-rules]

**Strength:** Named, frozen regulations could provide related construction instances for studying transfer. **Weakness:** An unversioned “VGC” result is difficult to reproduce, and success in one permitted domain does not establish success in another. This is versioning and instance variation, not evidence that a frozen VGC optimisation task must be nonstationary. Molecular and software-backed benchmarks also require version control.

### 8. The compared benchmarks supply scores; the present VGC instruction supplies only the candidate domain

**Benchmark starting point:** LABS supplies a merit factor, GuacaMol supplies scoring functions, and PMO specifies an evaluation budget and aggregation protocol. [BODi][bodi], [GuacaMol][guacamol], [PMO][pmo]

**VGC difference:** The supplied scope does not define a scalar or multi-objective score for a complete team. This is a difference in **benchmark readiness**, not a claim that team building cannot have an objective.

**Strength:** Construction legality and representation can already be specified and tested transparently. **Weakness:** Legality, novelty, and diversity alone cannot establish that an optimiser finds better teams. No conclusion about BO sample efficiency, objective noise, evaluation expense, optimum quality, or superiority to another benchmark follows yet. Respecting the construction-only scope means leaving this gap explicit rather than importing a battle-based evaluator.

## 5. Shared challenges that should not be presented as VGC differences

- **Interactions, constraints, and large combinatorial spaces already occur in the comparison set.** Pest control, LABS, NAS, biological design, and molecules rule out a blanket claim that ordinary BO benchmarks lack them. The defensible contribution is the particular construction schema and the questions it allows researchers to isolate.
- **A learned generator's reachable region is not automatically the legal domain.** InvBO studies encoder/decoder reconstruction mismatch; NF-BO also addresses that problem. If a VGC method uses a decoder, its coverage and reconstruction behaviour need separate reporting. Being able to write down legal team rules does not remove representation restrictions. [InvBO][invbo], [NF-BO][nfbo]
- **A top-k output collection is different from one composite candidate.** GuacaMol's collection of generated molecules contains separate solutions. The six members of a VGC team constitute one solution. Diversity across complete teams must therefore be distinguished from variation within one roster. [GuacaMol][guacamol]
- **Proxy optimisation is not validated usefulness.** Penalised logP, predicted activity, and docking scores establish their declared properties. Any construction-only team score would likewise need evidence for the interpretation assigned to it. The present scope gives no basis for ranking VGC's external validity above those tasks. [TDC oracles][tdc-oracles], [DOCKSTRING][dockstring]

## 6. Research framing supported by this comparison

> VGC full-team construction is a regulation-dependent, constrained combinatorial design problem over a fixed-size collection of configured entities. Its research value lies in combining member selection, conditional categorical choices, integer allocations, and roster-level legality within an interpretable construction. Relative to simple binary and mixed analytic benchmarks, it exposes a richer configuration schema. Relative to molecular and program-generation benchmarks, it offers a fixed roster schema while sharing validity and representation challenges. Establishing it as a BO benchmark additionally requires a reproducible scoring protocol and evidence that the resulting task tests useful optimisation capabilities.

The current evidence supports **a candidate research testbed**, not a demonstrated claim of uniquely difficult search or a recommendation for a particular optimiser.

Before making comparative performance claims, a construction benchmark would need a pinned regulation and validator; an exact complete-team representation and identity rule; a declared score; and a protocol for initial data, invalid proposals, repairs, duplicates, and budgets. For generative methods, it should also identify training data and the decoder's restrictions. These requirements follow from the variant and protocol differences in the collected benchmarks; they do not add preview or battle variables to the search unit.

## Primary-source reading map

| Source | Publication | Most useful part for this question |
| --- | --- | --- |
| [BOCS][bocs] | Baptista & Poloczek, **Bayesian Optimization of Combinatorial Structures**, ICML 2018 | §§4.1–4.4: binary tasks, sparsification, contamination. |
| [COMBO][combo] | Oh et al., **Combinatorial Bayesian Optimization using the Graph Cartesian Product**, NeurIPS 2019 | §4: categorical tasks and reuse of earlier benchmarks. |
| [CoCaBO][cocabo] | Ru et al., **Bayesian Optimisation over Multiple Continuous and Categorical Inputs**, ICML 2020 | §5 and input ranges: synthetic and model-configuration tasks. |
| [CASMOPOLITAN][casmo] | Wan et al., **Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces**, ICML 2021 | §4: categorical and mixed task variants. |
| [PR][pr] | Daulton et al., **Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization**, NeurIPS 2022 | §6: mixed, constrained, and multi-objective examples. |
| [BODi][bodi] | Deshwal et al., **Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings**, AISTATS 2023 | §7: LABS, MaxSAT, pest control, mixed tasks. |
| [Bounce][bounce] | Papenmeier et al., **Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces**, NeurIPS 2023 | Task reuse and categorical-encoding experiments. |
| [MCBO][mcbo] | Dreczkowski et al., **Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization**, NeurIPS 2023 | Table 2 and Appendix D: broader task coverage. |
| [GrammarVAE][gvae] | Kusner et al., **Grammar Variational Autoencoder**, ICML 2017 | BO on expressions and molecules; invalid-output treatment. |
| [JT-VAE][jtvae] | Jin et al., **Junction Tree Variational Autoencoder for Molecular Graph Generation**, ICML 2018 | Molecular graph construction and property optimisation. |
| [GuacaMol][guacamol] | Brown et al., **GuacaMol: Benchmarking Models for de Novo Molecular Design**, 2019 | Distinction between distribution learning and goal-directed tasks. |
| [LOL-BO][lolbo] | Maus et al., **Local Latent Space Bayesian Optimization over Structured Inputs**, NeurIPS 2022 | Expression, molecular-property, GuacaMol, and docking tasks. |
| [PMO][pmo] | Gao et al., **Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization**, NeurIPS 2022 | Named molecular objectives and budgeted evaluation. |
| [InvBO][invbo] | Chu et al., **Inversion-based Latent Bayesian Optimization**, NeurIPS 2024 | §5 and Appendix K: task recurrence and docking implementation. |
| [NF-BO][nfbo] | Lee et al., **Latent Bayesian Optimization via Autoregressive Normalizing Flows**, ICLR 2025 | §§5.1/5.5 and Appendix A: GuacaMol and all 23 PMO tasks. |
| [DOCKSTRING][dockstring] | García-Ortegón et al., **DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design**, 2022 | Docking tasks and the distinction between scoring protocols. |
| [LaMBO][lambo] | Stanton et al., **Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders**, ICML 2022 | Representative latent multi-objective biological design. |

[bocs]: https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf
[combo]: https://papers.neurips.cc/paper/8557-combinatorial-bayesian-optimization-using-the-graph-cartesian-product.pdf
[cocabo]: https://proceedings.mlr.press/v119/ru20a/ru20a.pdf
[cocabo-supp]: https://proceedings.mlr.press/v119/ru20a/ru20a-supp.pdf
[casmo]: https://proceedings.mlr.press/v139/wan21b/wan21b.pdf
[pr]: https://proceedings.neurips.cc/paper_files/paper/2022/file/531230cfac80c65017ad0f85d3031edc-Paper-Conference.pdf
[bodi]: https://proceedings.mlr.press/v206/deshwal23a/deshwal23a.pdf
[bounce]: https://proceedings.nips.cc/paper_files/paper/2023/file/05d2175de7ee637588d1b5ced8b15b32-Paper-Conference.pdf
[mcbo]: https://proceedings.neurips.cc/paper_files/paper/2023/file/dbc4b67c6430c22460623186c3d3fdc2-Paper-Datasets_and_Benchmarks.pdf
[antibody-code]: https://github.com/huawei-noah/HEBO/blob/master/MCBO/mcbo/tasks/antibody_design/cdrh3_design.py
[rna-code]: https://github.com/huawei-noah/HEBO/blob/master/MCBO/mcbo/tasks/rna_inverse_fold/rna_inverse_fold_task.py
[circuit-code]: https://github.com/huawei-noah/HEBO/blob/master/MCBO/mcbo/tasks/mig_seq_opt/mig_seq_opt_task.py
[gvae]: https://proceedings.mlr.press/v70/kusner17a/kusner17a.pdf
[jtvae]: https://proceedings.mlr.press/v80/jin18a/jin18a.pdf
[guacamol]: https://arxiv.org/abs/1811.09621
[guacamol-code]: https://github.com/BenevolentAI/guacamol/blob/master/guacamol/standard_benchmarks.py
[lolbo]: https://papers.nips.cc/paper_files/paper/2022/file/ded98d28f82342a39f371c013dfb3058-Paper-Conference.pdf
[lolbo-code]: https://github.com/nataliemaus/lolbo
[pmo]: https://wenhao-gao.github.io/assets/publications/2022_PMO_Neurips_datasets/paper.pdf
[tdc-oracles]: https://tdcommons.ai/functions/oracles/
[tdc-docking]: https://tdcommons.ai/benchmark/docking_group/overview/
[invbo]: https://arxiv.org/html/2411.05330v1
[invbo-code]: https://github.com/mlvlab/InvBO
[nfbo]: https://proceedings.iclr.cc/paper_files/paper/2025/file/df4f371f1f89ec8ba5014b3310578048-Paper-Conference.pdf
[dockstring]: https://arxiv.org/abs/2110.15486
[dockstring-bo]: https://link.springer.com/article/10.1186/s13321-024-00904-2
[lambo]: https://proceedings.mlr.press/v162/stanton22a.html
[vgc-rules]: https://mcdn.pokemon.com/pokemon-prod/raw/upload/v1/live/static-assets/content-assets/cms2/pdf/play-pokemon/rules/play-pokemon-vgc-tournament-handbook-en.pdf
[sp-total]: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/dex-formats.ts#L348
[sp-validator]: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/team-validator.ts#L1314
[team-format]: https://github.com/smogon/pokemon-showdown/blob/a5df8274e85b0889bf2a9b3422a08b39732374fc/sim/TEAMS.md#L241
