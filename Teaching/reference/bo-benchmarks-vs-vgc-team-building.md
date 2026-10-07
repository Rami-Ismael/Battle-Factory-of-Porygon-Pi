---
type: reference
tags:
  - bayesian-optimisation
  - benchmarks
  - problem-framing
  - vgc
created_at: 2026-09-24
updated_at: 2026-09-26
---
*Retired 2026-09-26 — differences 2–4 and 16 depend on the battle and team preview, outside the full-team scope; superseded by [[bo-benchmarks-vs-vgc-full-team-search-unit]]. The benchmark table and citations remain valid.*

Benchmarks that combinatorial, mixed-variable and latent-space (generative) Bayesian optimisation papers keep reusing, and the ways each one differs from building a team for Champions VGC 2026 Reg Set M-B. Every difference is read off the benchmark's own definition. No result from this project is used. Papers are named by their short names (BOCS, COMBO, CASMOPOLITAN, BODi, MerCBO, Bounce, MCBO, MOCA-HESP, AntBO, LOL-BO, LADDER, CoBO, InvBO); each expands to the full title under Citations. A benchmark's input is called a *point* here, because *candidate* means a species in this vault.

**The VGC side, fixed for every item.** Doubles. Each player registers six Pokémon and brings four at team preview (the *selection*). The search unit is a full *team*: six Pokémon, each with species, ability, held item, up to four moves, Stat Points and a stat alignment. There are no EVs, no IVs and no Tera. The *objective function* is win rate against the *meta*, meaning the collected list of meta teams weighted by how often each appears. Battles are simulated in Pokémon Showdown through poke-env, with the same behaviour-cloning *battle policy* piloting both sides. Per-slot sizes come from the vault's [[Clause or the feature of the search space]]: 208 species (231 buildable forms), movepools of 1–102 moves, 1–3 abilities per form, 149 item choices (Item Clause: all different; Mega Stones only work on their own species), 21 stat alignments, and Stat Points capped at 32 per stat and 66 in total.

## Benchmark table

| Benchmark | Family | Variables and constraints | Objective and noise | Introducing paper | ≥2 independent reusing papers (section) |
|---|---|---|---|---|---|
| Contamination control | combinatorial | 25 binary: apply a prevention effort or not at each stage of a food supply chain. No constraints. | Minimise prevention cost, plus the fraction of T = 100 simulated runs that break a contamination limit, plus an optional ℓ1 penalty λ‖x‖₁. Each fixed set of simulation runs defines one deterministic instance (BOCS App. D.2). | BOCS §4.3, from a model by Hu et al. (2010) | COMBO §4.1 and Supp. 4.1.2 (21 stages in the main text); CASMOPOLITAN §4.1, plus a noisy variant in App. B.2; Heat kernels App. E.1/E.3 |
| Pest control | combinatorial | 25 categorical × 5 (no pesticide or one of four) along a one-way chain of stations. No constraints. | Minimise pesticide cost plus the fraction of simulated runs where pest exceeds a limit (COMBO Supp. 4.2.2, Eq. 15). Prices and effectiveness shift with earlier choices, so interactions are high-order. | COMBO §4.2 and Supp. 4.2.2 (main text says 21 stations, supplement says 25) | CASMOPOLITAN §4.1; BODi §7.1; Bounce §4.3; MCBO §4.5 and App. C; Heat kernels §4 |
| Weighted MaxSAT-60 (weighted maximum satisfiability) | combinatorial | 60 binary. Clauses exist but are hidden from the optimiser. | Maximise the total weight of satisfied clauses. Deterministic. After weights are normalised to zero mean and unit variance, the optimum is all zeros by construction (Bounce App. B.1.3). | COMBO §4.3 (wMaxSAT28/43/60, MaxSAT Evaluation 2018 instances) | CASMOPOLITAN §4.1; BODi §7.1; Bounce App. B.1.3; Heat kernels App. E.1/E.3 |
| LABS-50 (low-autocorrelation binary sequences) | combinatorial | 50 variables in {−1, +1}. Order matters. | Maximise merit factor n²/2E(x). Closed form, deterministic. Known optimum 8.170, found by branch-and-bound (BODi §7.1). | MerCBO §4 (40 and 50 dimensions; earliest Bayesian optimisation use found) | Bounce §4.1; Heat kernels §4 and App. E.1; MOCA-HESP §5.2 (BODi §7.1 does not count: same group as MerCBO) |
| Cluster Expansion (MaxSAT-125) | combinatorial | 125 binary, from a real MaxSAT Evaluation 2018 instance in materials science. | Maximise the weight of satisfied clauses. Deterministic. Optimum unknown, so regret cannot be plotted (Bounce §4.2). Heat kernels normalise the weights; Bounce does not. | Bounce §4.2 (Heat kernels App. E.1 also names Bounce as the introducer) | Heat kernels §4 and App. E.1; MOCA-HESP §5.2 ("MaxSAT125"; the paper does not name the instance, but its released code loads the same cluster-expansion instance file as Bounce) |
| Ackley-20 categorical | combinatorial | 20 categorical × 11 evenly spaced grid values of Ackley's domain [−32.768, 32.768]. | Discretised Ackley. Synthetic, cheap, known optimum. | MCBO §5 and App. C, Table 2 (earliest source found) | Heat kernels App. E.5; MOCA-HESP §5.2 ("Ackley20c", using MCBO's implementation, App. A.8) |
| Antibody design (Absolut!) | combinatorial | CDRH3 (complementarity-determining region 3 of the heavy chain) of length 11 over 20 amino acids. Cheap developability constraints: net charge in [−2, 2], no residue more than five times (MCBO App. C words it as five identical residues in a row), no glycosylation motif. | Minimise binding energy to one fixed antigen. Absolut! is deterministic (AntBO §1). 159 antigens, each a separate task. | AntBO §2.1 | Heat kernels App. E.4; MOCA-HESP §5.2 (MCBO App. C does not count: shared authors) |
| Ackley-53 mixed | mixed-variable | 50 binary + 3 continuous in [−1, 1]³. | Ackley function with 50 coordinates binarised. Optimum 0 at the origin (Bounce App. B.1.1). The introducing paper adds uniform noise in [0, 10⁻⁶] to each evaluation. | Bliek et al. §5.4 | CASMOPOLITAN §4.2; BODi §7.3; Bounce App. B.1.1; MCBO App. C |
| SVM-53 (support vector machine feature selection and tuning) | mixed-variable | 50 binary include/exclude features + 3 continuous hyperparameters (C, ε, γ). | Minimise held-out root mean squared error of a support vector machine on the University of California, Irvine (UCI) slice dataset. Optimum unknown. | BODi §7.3 | Bounce §4.4; MCBO App. C (ν-support vector regression, feature selection "as in" BODi) |
| Arithmetic expression fitting | latent-space | Univariate expression strings from a fixed context-free grammar (≤15 production rules), searched through a grammar variational autoencoder (VAE) latent space. The grammar makes nearly every decoded string syntactically valid (Grammar VAE Table 2). Pretraining set of 100k expressions; later papers keep 50k (Tripp et al. App. C.5) or 40k (Grosnit et al. App. B.1.2) after removing the best ones. | log(1 + mean squared error) against a fixed target expression on 1,000 points in [−10, 10]. Deterministic. Optimum at the target expression (Tripp et al. App. C.5). The target changed from 1/3 + x + sin(x·x) in Grammar VAE to 1/3 · x · sin(x·x) in later papers. | Grammar VAE §4.1/§4.3 | Grosnit et al. §4 and App. B.1.2; LADDER §4.1; CoBO §3 (Tripp et al. §6 and LOL-BO §5.1 also use it, but each shares an author with Grammar VAE: Hernández-Lobato and Kusner) |
| Penalised logP | latent-space | Molecules as SMILES (simplified molecular-input line-entry system) strings, junction trees or SELFIES (self-referencing embedded strings), decoded from a VAE pretrained on ZINC250k. Junction-tree and SELFIES decoders always give valid molecules; a SMILES decoder does not. | logP minus synthetic accessibility minus a large-ring penalty (the introducing preprint normalises each term). Deterministic in the molecule, but decoding a fixed latent point is random (Gómez-Bombarelli et al. v1). No useful ceiling. | Gómez-Bombarelli et al., arXiv v1 (2016), Eq. 1; credited in Grammar VAE §4.1 and Tripp et al. §6 | Grosnit et al. §4 and App. B.1.3; LOL-BO §5.1 |
| GuacaMol multi-property objective tasks (Perindopril, Ranolazine, Zaleplon, Osimertinib MPO etc.) | latent-space | Molecules through a VAE pretrained on the GuacaMol training corpus (ChEMBL-derived, 1.27M molecules as counted in LOL-BO §5). | Similarity to a named target drug, combined with required property changes. Score in [0, 1] (GuacaMol App. 8.3.1). Deterministic in the molecule. CoBO runs budgets up to 70k queries. | GuacaMol §3.2 and App. 8.3 | LOL-BO §5.2; CoBO §3 (InvBO §5.1 as well, same group as CoBO) |
| DRD3 docking (Therapeutics Data Commons) | latent-space | Molecules through a VAE. | Docking score against one protein target, dopamine receptor D3. Minutes per molecule (LOL-BO §5.3). | Therapeutics Data Commons §10.3 | LOL-BO §5.3; CoBO §3 (InvBO §5.1 as well) |

## Differences

1. **The benchmark scores a point against one fixed target. VGC scores a team against a weighted mixture of opponents.**
   - *Benchmarks:* Antibody design (binding to one antigen), DRD3 docking (one protein), GuacaMol MPO (similarity to one named drug), arithmetic expressions (one target expression).
   - *They assume:* the objective function is f(x). A target, if there is one, is baked into the task. MCBO's 159 antigens are 159 separate tasks, not one mixture (MCBO App. C).
   - *VGC instead:* f(T) = Σ_O w_O · P(T beats O). The sum runs over the meta, with w_O the weight of meta team O in the collected list. One team has to do well across many opponents at once.

2. **In VGC the target plays back. In the benchmarks it is passive.**
   - *Benchmarks:* Contamination control and pest control (the dynamics are random but pursue no goal); docking and Absolut! (a physics or lattice model scores the binding); MaxSAT and LABS (closed form).
   - *They assume:* nothing inside an evaluation picks actions to make the point score worse.
   - *VGC instead:* every battle is against an opponent team piloted by the same battle policy, choosing moves turn by turn to win. How good a team is depends on how the opponent's choices react to it.

3. **The VGC objective depends on a battle policy, a decision-maker inside the evaluation.**
   - *Benchmarks:* All of them. SVM-53 comes closest: its score goes through a fixed training procedure (support vector machine fitting), which is a deterministic subroutine, not an agent.
   - *They assume:* once the point is fixed, its score is fixed by the task definition.
   - *VGC instead:* the value of a team is f(T; π), where π picks the selection at preview and every in-battle action. Swap the battle policy and you have a different objective function, with possibly a different best team. A team the policy cannot pilot well scores low whatever its merit under stronger play.

4. **Only four of the six count in any given battle.**
   - *Benchmarks:* Pest control, contamination, LABS, MaxSAT and antibody design: every variable enters every evaluation. In SVM-53 a feature can drop out, but only because its own variable is set to exclude.
   - *They assume:* each coordinate of x always affects the score.
   - *VGC instead:* the battle policy picks four of the six at preview, per opponent. A Pokémon that is never selected against the meta contributes nothing to f, and which Pokémon matter changes from one opponent to the next.

5. **VGC returns a noisy Bernoulli outcome. Most benchmarks return a deterministic value.**
   - *Benchmarks:* Contamination (BOCS App. D.2: each fixed set of simulation runs defines one objective instance), Absolut! (deterministic, AntBO §1), MaxSAT, LABS, Ackley-20 and arithmetic expressions (closed form). Where noise appears it is small and additive: Ackley-53 as introduced adds uniform noise in [0, 10⁻⁶] (Bliek et al. §5.4), and CASMOPOLITAN's noisy contamination variant adds noise of variance 10⁻² (App. B.2).
   - *They assume:* querying the same x twice gives the same value, or noise is a small additive term.
   - *VGC instead:* one battle returns win or loss. The estimate of the objective is a mean of 0/1 outcomes, with variance p(1 − p)/n per matchup. Near p = 0.5 that noise is at its largest, not a small add-on.

6. **VGC lets you buy precision per team. The benchmarks charge a fixed price per query.**
   - *Benchmarks:* COMBO budgets 100–320 queries (§4), Bounce 200–500 (§4), CoBO up to 70k (GuacaMol), 3k (DRD3) and 500 (expressions) (§3). Contamination spends 100 simulations inside every query. Docking takes minutes per molecule (LOL-BO §5.3).
   - *They assume:* one query is one atomic unit of cost with one fixed precision, and the budget is a count of queries.
   - *VGC instead:* one evaluation is however many battles you choose to spend across however many meta teams. The same budget can go to a few teams measured precisely or many measured roughly. Cost and precision are set per team, by the searcher.

7. **The benchmarks either publish the optimum or say plainly that it is unknown. VGC has no reference optimum, only reference teams.**
   - *Benchmarks:* Ackley-53 (optimum 0 at the origin, Bounce App. B.1.1). MaxSAT-60 (optimum at all zeros by construction, Bounce App. B.1.3). LABS-50 (merit factor 8.170, BODi §7.1). Arithmetic expressions (f = 0 at the target, Tripp et al. App. C.5). GuacaMol (bounded at 1, GuacaMol App. 8.3.1). Cluster Expansion is the stated exception: regret cannot be plotted (Bounce §4.2), and the same holds for MCBO's real-world tasks (MCBO §5.3, footnote 3).
   - *They assume:* progress can be reported as regret, or at least against a published best-known value.
   - *VGC instead:* the best team is unknown, and win rate is capped at 1 but is not expected to reach it against a mixed meta. The natural yardstick is how the meta teams themselves score under the same objective. Beating that bar is not the same as knowing how far is left.

8. **Some benchmarks plant their optimum at a special point. The fix is to relocate it.**
   - *Benchmarks:* MaxSAT-60 and Ackley-53 (optimum at all zeros); pest control (a near-best point at all fives, Bounce §4.3). Bounce §4 and §4.6 flip bits or permute category order. Heat kernels §4 do the same and analyse the effect in §4.2. MOCA-HESP §5.2 and App. A.8 add shifted LABS and Ackley.
   - *They assume:* the published version is a fair test. Bounce §4.6 shows it is not: BODi gets worse on five of the seven benchmarks once the optimum is moved.
   - *VGC instead:* nothing plants the optimum at a special point. The lesson that carries over is about encoding: an optimiser that leans on category order or index position will latch onto whatever order the species, move and item IDs happen to be listed in.

9. **Benchmark positions are ordered. A team is not.**
   - *Benchmarks:* Pest control and contamination (stations in a one-way chain), LABS (autocorrelation depends on order), antibody design (positions along a sequence), arithmetic expressions and SMILES (strings). Heat kernels §3.5 and App. D.1 have to add permutation invariance as an extension, and test it on separate permutation-invariant SFU (Simon Fraser University) functions (App. E.5) and neural architecture search (App. F), not on the standard suite.
   - *They assume:* coordinate i means the same thing in every point, so a kernel can compare x_i with x′_i.
   - *VGC instead:* reordering the six Pokémon gives the same team, and reordering a Pokémon's four moves gives the same Pokémon. A slot-by-slot encoding counts the same team up to 6! times and compares Pokémon that happen to share a slot index.

10. **The benchmarks give every position the same, independent domain. In VGC one choice decides which other choices are legal.**
    - *Benchmarks:* Pest control (every station takes the same 5 options), contamination, MaxSAT, LABS (binary), antibody design (20 amino acids per position), Ackley-53 and SVM-53 (the continuous box is shared no matter what the binary part says). Heat kernels §3.4.3 note that every problem in the recent suites has equal-sized domains.
    - *They assume:* the search space is a Cartesian product of per-variable domains.
    - *VGC instead:* the species fixes the legal abilities (1–3) and the movepool (1–102 moves). A Mega Stone only does anything on its own species. A Pokémon may carry fewer than four moves, so the length varies. It is a hierarchical space, not a product space.

11. **Benchmark constraints are local checks on one string. VGC legality couples all six Pokémon.**
    - *Benchmarks:* Antibody design (per-sequence developability checks, AntBO §2.1; MCBO §4.2 handles such cheap input constraints by rejection; Heat kernels App. E.4 had to leave out one method on this task because it could not take input constraints). Arithmetic expressions and molecules (a grammar or decoder keeps each string syntactically valid, Grammar VAE Table 2). MaxSAT, LABS, pest control and contamination (no constraints).
    - *They assume:* validity is a property of one point's own symbols, cheap to check locally.
    - *VGC instead:* validity is still cheap (the Showdown validator runs before any battle), but it is team-wide. Species Clause and Item Clause are all-different constraints across the six Pokémon, on top of each Pokémon's own legality. A per-slot grammar cannot enforce them without tracking the whole team.

12. **Benchmark categorical variables have 2–20 values. VGC's have hundreds.**
    - *Benchmarks:* Binary (contamination, MaxSAT, LABS, Cluster Expansion), 5 values (pest control), 11 (Ackley-20), 20 (antibody design). MOCA-HESP §5.2 says the cardinalities across its suite run from 2 to 20.
    - *They assume:* one-hot or Hamming-distance kernels over a handful of categories per variable.
    - *VGC instead:* 208 species (231 forms), 149 item choices, movepools up to 102, 21 alignments. Two species are equally far apart in Hamming distance, whatever their role or type.

13. **The numeric part of VGC is an integer budget, not a continuous box.**
    - *Benchmarks:* Ackley-53 (x ∈ [−1, 1]³), SVM-53 (continuous C, ε, γ).
    - *They assume:* the continuous coordinates are independent and box-bounded, so gradient steps or continuous trust regions apply.
    - *VGC instead:* Stat Points are integers, capped at 32 per stat and 66 per Pokémon. That is a coupled simplex-like constraint across six stats, and a stat alignment (21 categories) is attached to it. None of the kept benchmarks has a sum budget.

14. **In latent-space benchmarks the noise comes from the decoder. In VGC it comes from the objective too.**
    - *Benchmarks:* GuacaMol, as run in LOL-BO §5.4 and Table 1: decoding one latent point 20 times gives Perindopril MPO scores with a standard deviation of 0.04 (SELFIES VAE) to 0.1 (junction-tree VAE), on a score bounded in [0, 1]. Gómez-Bombarelli et al. v1 already note that decoding is stochastic.
    - *They assume:* the objective function is deterministic in the decoded object, so all the randomness sits on the input side.
    - *VGC instead:* a generative model over teams would add decode randomness on top of battle randomness. The two sources pile up, and a single noisy score cannot separate them.

15. **In latent-space benchmarks the pretraining corpus is large and separate from the target. In VGC the natural corpus is the opponent set.**
    - *Benchmarks:* Arithmetic expressions (100k generated expressions, Grammar VAE §4.1; Tripp et al. App. C.5 and Grosnit et al. App. B.1.2 remove the best-scoring ones on purpose), penalised logP (ZINC250k), GuacaMol (1.27M molecules, LOL-BO §5).
    - *They assume:* plenty of valid, unlabelled objects to learn a latent space from, drawn independently of the objective's target.
    - *VGC instead:* the obvious source of valid, competitive teams is the collected meta list, which also defines the objective. Training the generator on the opponents couples the prior to the evaluation distribution. That list is also far smaller than the 100k–1.27M-object corpora above.

16. **Penalised logP can be gamed without limit. VGC win rate is a bounded proxy that can be gamed differently.**
    - *Benchmarks:* Penalised logP. LOL-BO §5.1 reaches an average of 545, and warns the molecules found "clearly abandon any notion of reality".
    - *They assume:* optimising the score is the goal; LOL-BO notes that it and its baselines treat the task as unconstrained.
    - *VGC instead:* win rate cannot run off to infinity. But a team can exploit blind spots of the one behaviour-cloning battle policy, so its simulated score overstates it. The check sits outside the objective function: ladder validation.

17. **Some benchmarks are white-box problems with the box closed on purpose. VGC is black-box by nature.**
    - *Benchmarks:* MaxSAT-60 and Cluster Expansion (COMBO §4.3: specialised MaxSAT solvers exist; Bounce §4.2: the optimiser gets no access to the clauses). LABS (solved exactly by branch-and-bound, BODi §7.1).
    - *They assume:* hiding the structure is a fair stand-in for expense.
    - *VGC instead:* even with the whole simulator source available, win rate against the meta has no closed form. It runs through battle randomness and a learned battle policy. No exact solver exists to hide.

18. **A benchmark evaluation returns one number. A VGC evaluation returns results per opponent.**
    - *Benchmarks:* All kept benchmarks return one scalar per query. GuacaMol MPO builds its score from several property terms but only reports their combined (arithmetic or geometric) mean (GuacaMol App. 8.3.1).
    - *They assume:* the surrogate learns from (x, y) pairs with y a scalar.
    - *VGC instead:* each evaluation yields win counts against each meta team (a row of a matchup matrix) plus battle logs. A matchup predictor can learn from that structure, and the scalar f is a weighted sum computed from it afterwards.

19. **Benchmark instances are frozen forever. The VGC objective is a snapshot of something that moves.**
    - *Benchmarks:* MaxSAT and Cluster Expansion (fixed competition instances), ZINC250k and GuacaMol (fixed datasets), Absolut! (fixed antigen set), pest control and contamination (fixed simulator settings).
    - *They assume:* the task at test time is the task at benchmark time.
    - *VGC instead:* the meta is a distribution over teams that shifts as players adapt and as regulations change. The collected list is a sample taken on one date, and any change to it or its weights changes the objective function. The objective stays well defined for that snapshot, but a team tuned to it can lose value once the snapshot ages.

## Dropped benchmarks

- **RNA inverse folding (EteRNA100 with ViennaRNA, MCBO App. C).** Only one independent reuse found (Heat kernels App. E.4, which takes it from MCBO).
- **And-Inverter Graph and Majority-Inverter Graph logic-synthesis flow tuning (MCBO App. C).** Only one independent reuse found (Heat kernels App. E.4).
- **And-Inverter Graph flow plus hyperparameter tuning (MCBO App. C).** No reuse found.
- **XGBoost–MNIST (CASMOPOLITAN §4.2, MCBO App. C).** One reuse (MCBO). No second independent reuse found.
- **The other SFU test functions in MCBO's discretised form (Ackley-20 is kept above).** Heat kernels App. E.5 reuses Ackley, Schwefel, Rastrigin, Styblinski–Tang and Sphere at MCBO's 20-variable, 11-value setting. For the non-Ackley functions no second independent reuser was found.
- **2D shape area (Tripp et al. §6).** No reuse found in the latent-space papers read.
- **Topology shape fitting (Grosnit et al. §4 and App. B.1.1).** No independent reuse found in the papers read.
- **Tripp et al. and LOL-BO as reusers of arithmetic expressions.** Not dropped as a benchmark, but not counted: Tripp et al. share Hernández-Lobato and LOL-BO shares Kusner with Grammar VAE. Tripp et al. are not counted for penalised logP either (Hernández-Lobato is also a Gómez-Bombarelli et al. author). LADDER §4.1 runs a ZINC logP task too, but calls it plain logP, so it is not counted for penalised logP.

## Citations

🟢 means a free full text was opened (the arXiv copy for every paper; Gómez-Bombarelli et al. also as arXiv v1). Venues are from proceedings pages, the OpenReview venue field, DBLP or Crossref.

- [Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization](https://www.semanticscholar.org/paper/6f2ab02e77849b0ebdfa8873f8f419820a84132f) (MCBO) 🟢 NeurIPS 2023 Datasets and Benchmarks track
- [Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces](https://www.semanticscholar.org/paper/fe4f76d46e5506eb59a441bc8be482e7b2dd322e) 🟢 NeurIPS 2023
- [Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings](https://www.semanticscholar.org/paper/68500e037f95d21e2d169f81e696e1e0461546bb) (BODi) 🟢 AISTATS 2023, PMLR 206
- [Combinatorial Bayesian Optimization using the Graph Cartesian Product](https://www.semanticscholar.org/paper/65a854702fea5b32d02e13515cae7fc7460e1510) (COMBO) 🟢 NeurIPS 2019
- [Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces](https://www.semanticscholar.org/paper/645e2401c838f690a7a5e14367ac54fc1afe950a) (CASMOPOLITAN) 🟢 ICML 2021
- [Bayesian Optimization of Combinatorial Structures](https://www.semanticscholar.org/paper/a2d1aaedb70777d1fa047af6db30e84c55b81023) (BOCS) 🟢 ICML 2018
- [Mercer Features for Efficient Combinatorial Bayesian Optimization](https://www.semanticscholar.org/paper/0fe589d100806afdaaff574e76616242ad9f13da) (MerCBO) 🟢 AAAI 2021
- [Omnipresent Yet Overlooked: Heat Kernels in Combinatorial Bayesian Optimization](https://www.semanticscholar.org/paper/189d5889f9a6aa3c391a390a78cf5a3ae5eb45b7) (Heat kernels) 🟢 NeurIPS 2025
- [MOCA-HESP: Meta High-dimensional Bayesian Optimization for Combinatorial and Mixed Spaces via Hyper-ellipsoid Partitioning](https://www.semanticscholar.org/paper/b5841efdf6b7b3d29358fe31760fb79121bc3588) 🟢 ECAI 2025, Frontiers in Artificial Intelligence and Applications (open access, CC BY-NC 4.0)
- [Black-box mixed-variable optimisation using a surrogate model that satisfies integer constraints](https://www.semanticscholar.org/paper/524b4e54ece377272a648873d0306e14a8d0ce5b) (Bliek et al.) 🟢 GECCO 2021 Companion, pp. 1851–1859 (open access, CC BY 4.0). Section numbers are from arXiv v2
- [Toward real-world automated antibody design with combinatorial Bayesian optimization](https://www.semanticscholar.org/paper/faedaf9c5d115186408c3ef8e77bbf8f5e3eabff) (AntBO) 🟢 Cell Reports Methods 3(1), 2023. Section numbers are from the arXiv preprint, titled "AntBO: Towards Real-World Automated Antibody Design with Combinatorial Bayesian Optimisation"
- [Grammar Variational Autoencoder](https://www.semanticscholar.org/paper/222928303a72d1389b0add8032a31abccbba41b3) 🟢 ICML 2017
- [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://www.semanticscholar.org/paper/88427d2143d4cff357c3b393ae7580a7b6e19940) (Gómez-Bombarelli et al.) 🟢 ACS Central Science 2018 (open access). The penalised-logP objective is Eq. 1 of arXiv v1 (2016); the published version optimises 5 × quantitative estimate of drug-likeness − synthetic accessibility score instead
- [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://www.semanticscholar.org/paper/c6f9e20f574c37e2ef12fa41a5729df8e0499f6f) (Tripp et al.) 🟢 NeurIPS 2020
- [High-Dimensional Bayesian Optimisation with Variational Autoencoders and Deep Metric Learning](https://www.semanticscholar.org/paper/a9ccb803bf80b47d9be95db4776fd88f4a43bccd) (Grosnit et al.) 🟢 arXiv preprint, no venue found (DBLP lists it under CoRR only)
- [Local Latent Space Bayesian Optimization over Structured Inputs](https://www.semanticscholar.org/paper/290b5017a8e574bec4d810167e711566069d64cc) (LOL-BO) 🟢 NeurIPS 2022
- [Combining Latent Space and Structured Kernels for Bayesian Optimization over Combinatorial Spaces](https://www.semanticscholar.org/paper/c10a28812649d0482672485621b2d3ae59e90eca) (LADDER) 🟢 NeurIPS 2021
- [Advancing Bayesian Optimization via Learning Correlated Latent Space](https://www.semanticscholar.org/paper/acd4d6c8cfc9df063e303cdc6746817f8496c659) (CoBO) 🟢 NeurIPS 2023
- [Inversion-based Latent Bayesian Optimization](https://www.semanticscholar.org/paper/da1a0aaf1a638884fca253490bf8347a4d16a7c6) (InvBO) 🟢 NeurIPS 2024
- [GuacaMol: Benchmarking Models for De Novo Molecular Design](https://www.semanticscholar.org/paper/5bfeb6901db481c08874cfe0ae807d8564513765) 🟢 via arXiv; Journal of Chemical Information and Modeling 59(3), 2019, 🔒 at the publisher
- [Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development](https://www.semanticscholar.org/paper/54ca116f1e9a45768a3a2c47a4608ff34adefa0c) 🟢 NeurIPS 2021 Datasets and Benchmarks track

Vault notes: none of the cited papers has its own note in Papers/. Tripp et al. already appears in Reading List.md under "Reward-weighted retraining".
