---
type: work-list
tags:
  - bayesian-optimisation
  - benchmarks
  - problem-framing
  - vgc
status: open
purpose: collect benchmark research to describe VGC team building as an optimisation research problem and understand its strengths and weaknesses
origin: "[[Why should you care about team building in competitive pokemon as an academic question]]"
output: Teaching/reference/
retires:
  - Teaching/reference/bo-benchmark-reuse-evidence.md
  - Teaching/reference/bo-benchmarks-vs-fixed-opponent-vgc.md
  - Teaching/reference/bo-benchmarks-vs-pokemon-team-building.md
created_at: 2026-09-24
updated_at: 2026-10-05
---
- [ ] Collect the common benchmarks used in **combinatorial / mixed-variable and latent-space (generative) Bayesian optimisation**, and write a fresh list of the differences between those benchmarks and VGC team building. Do not use or refer to the existing 13 points in "Search Perspective" in [[What makes Pokémon a unique problem set]]. The list should be built from the benchmarks themselves.
	- **Purpose:** Collect research to describe VGC team building as an optimisation research problem. Use comparisons with other benchmarks to understand its strengths and weaknesses as a research problem. Explain how the identified differences inform those strengths and weaknesses, rather than merely listing differences.
	- **Research output (2026-09-29):** [[Teaching/reference/bo-benchmark-derived-vgc-research-comparison|Benchmark collection and eight comparisons of VGC's research strengths and weaknesses]].
	- **Research output (2026-09-30):** Version 2 of the same note: feature table (declared / measured) and eight new differences, with $f$ in scope.
	- Scope: "VGC team building" means the full-team search unit only (species, ability, item, moves, Stat Points, alignment). Team preview and the battle are excluded.
		- Objective in scope (2026-09-30): $f$ = win rate vs the top-50 meta, behaviour-cloning policy both sides, as a fixed black box. Preview and battle produce the score; they are not searched.
	- Molecule benchmarks count: penalised logP, GuacaMol, Therapeutics Data Commons tasks, docking. Citing them as benchmarks is illustration, not the closed molecular-optimisation approach.
	- Feature to compare too all bayesian black box academic benchmark
		- Smoothness
		- Noise
		- etc
		- Multimodality
		- Evaluation cost
		- Constraint type (known vs black-box)
		- Dimensionality / cardinality
		- Known optimum
		- Stationarity
	- Comparators: only the benchmarks already collected. Borrow the COCO/BBOB feature definitions, not its functions.
	- Evidence: each cell is labelled *declared* (stated by the paper) or *measured* (same metric on both sides, where the benchmark code is cheap to run).
	- [ ] grill me
- [ ] [[bo-benchmark-derived-vgc-research-comparison]] I liked the benchmark on the details
- [ ] First pick an ui library, then decidee what prootype will be good for my blog that has naimation similar to iclr blog and distill, decide what animation we should create, find some more animation opporuntiues finally improve my animation with the right vocabulary also look at other artifact We need to group the bayesian and crate an visual for bayesian animation and design for the visual for important feature for the bayesian function that will work for fine for each base on **Search-space properties:** what a team is, before any win rate., **Evaluation properties:** how _f_ can be observed. **Landscape properties:** the shape of _f_ over the space, given an edit operator.
- [ ] https://richardcsuwandi.github.io/awesome-bo/
- [ ] There should be writing reason why I choose this benchmark to compare to 
	- [ ] First calculuate all the possible bayesian optimization academic benchmark there are to test out your ideas, then we select a few sample base on some critieria . The first criteria is that the benchmark have to be used by atleast two academic paper to make sure the benchmark is relevant. Second, is structure we have similar structure so theer will be comparison of this benchmark with moleculue datasets as an example. Third, the reason I coding on my only laptop pro I don't have access to gpu hardware. So, I need benchmark that I run and test the comparision with my pokemon.  Fourth, I want benchmark want some noise in there for example in pokemon there is good amount of stochacistitycy in the games and later go in detail. 
	- Grill answers (2026-10-05):
		1. "This benchmark" = the seven measured rows in Version 3 of [[bo-benchmark-derived-vgc-research-comparison]], not the whole field or the full collection.
		2. Reader: the blog first (ML-curious engineers). Start with one plain sentence on what a benchmark is.
		3. Reason one, recurrence: each task appears at the same size in at least two Bayesian optimisation papers (checked 2026-10-05 against BOCS, COMBO, BODi and Bounce).
		4. Reason two, structure: each task shares one part of what a team is (subset choice, fixed positions, categorical choices, mixed variables).
		5. Pitch comparability, not exoticness: the field's own ruler, measured the same way on VGC. Do not claim VGC has more unique features.
		6. Be open: one person on a MacBook Pro measured only the benchmarks cheap enough to run thousands of times; academics are short of GPUs too.
		7. Say Pokémon has dice in it (damage rolls, critical hits, accuracy, speed ties) and the benchmarks have none; that randomness is the main difference.
		8. Name the 25-station pest control instance; COMBO's paper ran 21.
		9. Leave out Hearthstone deckbuilding; interest in it is declining.



# Writin

1.  I will we will comparing pokemon team design space with common bayesian optimization academic benchmark 
2. I will be comparing 
3. I will compare the difference between competiive pokemon team building with other academic benchmark. You will see the variety of collection of unique feature where a couple of them has a couple while this has many morel. 
