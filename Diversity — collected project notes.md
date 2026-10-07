# Diversity — collected project notes

Collected 2026-09-18. This index covers Markdown under this project directory, including its Papers, Teaching, Blog and Workshops folders. It links to existing notes; their content and historical decisions remain in place.

Search coverage: **267 Markdown files searched; 73 matching files collected**. Search terms included diversity/diverse and spelling variants, novelty, mode collapse, composition concentration/entropy, MAP-Elites and QD-score. Broader matches are included below so brief references are not lost. This is a collection of existing material, not a new verification of paper claims or experiment results.

## Suggested reading order

1. [[Metrics for Diversity]] — definitions and latest recorded measurement status.
2. [[Boltzmann selection temperature experiment]] — performance/diversity results.
3. [[Quality Diversity]] and [[Teaching/learning-records/0011-map-elites-two-gates-reteach]] — quality-diversity versus uniqueness.
4. [[Papers/LLM candidate diversity — evidence for the blog]] — evidence for the blog motivation.
5. [[Explain the metric for diversity and quality for the blog]] — turn the collection into the explanation.

## What is already recorded

The final “Baseline complete — 2026-09-08” section of [[Metrics for Diversity]] reports 498.3 distinct compositions per 512 baseline draws versus 172.0–276.3 in final CEM arms, while exact uniqueness remains 100%. That is the clearest local example of why exact uniqueness and roster diversity are different measurements. These are the note’s recorded results, not a rerun performed for this collection. Earlier “baseline outstanding” text is historical.

Keep four questions separate: are complete records different; are rosters different; are outputs absent from a specified reference corpus; and do teams behave differently in battles? The existing notes do not establish that any one of these automatically answers the others.

## Start here: definitions and the blog

- [[Metrics for Diversity]] — Main measurement contract: exact uniqueness, repeats, reference-corpus novelty, composition counts, matching-based distance, entropy, and recorded before/after results. Read the final “Baseline complete” section for the latest status in this note.
- [[Explain the metric for diversity and quality for the blog]] — The writing task this collection supports: distinguish exact-team uniqueness, roster diversity and reference-data novelty.
- [[Teaching/reference/why-measured-team-search-instead-of-prompt-only-generation]] — Explains why candidate diversity must be measured rather than assumed from the generator family.
- [[Why so we initial use a diffusion model instead of Large Language Model]] — Original motivation: novel and diverse proposals, with performance-directed generation.

## Experiments, results and planned comparisons

- [[Boltzmann selection temperature experiment]] — Recorded comparison of selection temperatures; reports roster diversity alongside fresh generator win rate and uncertainty.
- [[Papers/DiffUCO — selection temperature experiment]] — Protocol for measuring diversity on equal-size draws, separating effective sample size from generated-team diversity, and making across-seed claims.
- [[Diffusion as candidate proposer in black-box optimization over structured inputs]] — Experiment axes: data, guidance, decoding, elite fraction and original-data replay; asks how each changes diversity and win rate.
- [[Does naive gradient work as guidance]] — Historical run stopped on a composition-diversity guardrail; useful experiment context.
- [[Cross Entropy Method]] — Short task proposing a diversity filter in cross-entropy selection.
- [[New Todo Section with No Long Verbose Opus text geneation]] — Tasks for diversity across CEM generations and guidance settings.
- [[Todo Section]] — Broader project task history, including diversity measurements and MAP-Elites exploration.
- [[List of Experiments]] — Broader experiment history and team-pool-size comparisons.
- [[trying to retire this convert this into a blog (diffusion model as team generator for model-based search-research finding) into (Using diffusion model for black box optimzation)]] — Historical generator notes: training-distribution variation, descriptors and a quality-diversity archive proposal.

## Quality-diversity, archives and playstyles

- [[Quality Diversity]] — Quality-diversity problem class, QD scores, algorithms and open descriptor choice.
- [[MAP Elite]] — Short MAP-Elites paper list focused on Hearthstone deck construction.
- [[How can we measure from the play history the different play style in pokemon]] — Open question about measuring behavior and clustering playstyles rather than identifying diversity solely through roster features.
- [[Metaheuristics]] — Places quality-diversity methods within the broader search-method taxonomy.
- [[Teaching/learning-records/0011-map-elites-two-gates-reteach]] — Archive insertion, per-niche quality, and the distinction between variation alone and quality-diversity.
- [[Teaching/learning-records/0005-deep-surrogate-assisted-map-elites]] — Surrogate-assisted MAP-Elites lesson, QD-score comparisons and online training.
- [[Papers/VGC AI Competition (IEEE CoG)]] — References for team synthesis and the Architecting Meta-Game Diversity paper.
- [[Workshops/NeurIPS/EvoRobust (NeurIPS 2026)]] — Workshop note connecting diversity-driven search, quality-diversity and robustness.

## Candidate diversity, collapse and generator comparisons

- [[Papers/LLM candidate diversity — evidence for the blog]] — Primary-source discussion of repetitive candidate pools, novelty, verbalized sampling and guidance tradeoffs; separates evidence from Pokémon transfer hypotheses.
- [[Papers/Self-Consuming Generative Models Go MAD - reading guide]] — Model-collapse explanation: quality versus coverage, repeated retraining and limits of the analogy to team optimization.
- [[Papers/Self-Consuming Generative Models Go MAD - visual explanation.excalidraw]] — Markdown-backed visual explanation of the same collapse/coverage distinction.
- [[Papers/LLM versus diffusion for Bayesian search - evidence and verdict]] — Generator comparison caveats and proposed diversity endpoints; best-of-batch performance is not diversity evidence.
- [[Papers/LLMs as black-box optimizers — evidence and mechanisms]] — Diversity-aware candidate histories and evaluation among competitive teams.
- [[Papers/Diffusion candidate proposers - experiments in related papers]] — Related experiment designs, including guidance strength and fitness–diversity tradeoffs.
- [[Papers/Variational Annealing - forward citation evidence]] — Search-distribution mixtures, multiple modes and black-box diversity connections.
- [[Teaching/reference/diffusion-counter-team-count-primary-sources]] — Separates number generated, legality, performance qualification and diversity; discusses roster versus strategy counts.

## Opponent and policy diversity

- [[Papers/VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] — Benchmark reference for battle policies across varied teams; keep this distinct from diversity of generated counter-teams.
- [[Reimplement VGC-Bench — run the release, reproduce BC, self-play, fictitious play, double oracle, then extend on team-pool size]] — Reproduction plan and team-pool-size experiments.
- [[Papers/Advances in Computer Games 2025 — relevant papers for the VGC project]] — Section on static/intrinsic diversity in Go agents and the connection to opponent populations.
- [[Papers/Grandmaster level in StarCraft II using multi-agent reinforcement learning]] — Reference to exploration, conditioning and diversity in the StarCraft policy setting.

## Additional sections, reading decisions and passing references

These files also match the search. Some contain substantial sections; others only link to a diversity-related paper or record a past reading decision. Match locations refer to the source files at collection time.

- [[Behavior cloning a VGC battle policy from human Showdown replays]] — matching lines 15.
- [[Blog]] — matching lines 12.
- [[Blog/Blog for 2026-08-27]] — matching lines 1.
- [[Blog/Blog for 2026-08-28]] — matching lines 1.
- [[Blog/Blog for 2026-09-06]] — matching lines 1.
- [[Blog/Blog for 2026-09-07 — daily summary draft]] — matching lines 7.
- [[Blog/visuals/search-loop/README]] — matching lines 54, 56.
- [[Blog/visuals/search-loop/content-review]] — matching lines 7, 22, 24, 32, 42.
- [[Create a public repository for this project with a runnable README]] — matching lines 55, 70, 76, 84, 93, 94 ….
- [[Determine what is action space]] — matching lines 2.
- [[Does pretraining on singles replays make a better VGC doubles battle policy]] — matching lines 17.
- [[How do you implementing maskout in reinforcement learning and whey do you need to mask out]] — matching lines 1.
- [[Papers/AlphaRank]] — matching lines 13.
- [[Papers/Citation]] — matching lines 13, 20.
- [[Papers/DDEA - reading guide]] — matching lines 33.
- [[Papers/DIMES citing papers - reading-list assessment]] — matching lines 22, 48.
- [[Papers/Epiplexity]] — matching lines 9, 25.
- [[Papers/Generative Bayesian Optimization — relevance to counter-team search]] — matching lines 26.
- [[Papers/Neural combinatorial optimization - relevance grades]] — matching lines 5.
- [[Papers/Reading List — notes and decision history]] — matching lines 43, 47, 55, 59, 77, 89 ….
- [[Papers/Reading List]] — matching lines 22, 41, 46, 48, 49, 50 ….
- [[Papers/Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models]] — matching lines 19.
- [[Teaching/NOTES]] — matching lines 18, 19.
- [[Teaching/RESOURCES]] — matching lines 38, 70.
- [[Teaching/learning-records/0004-analogue-domain-sweep-for-f]] — matching lines 12.
- [[Teaching/reference/bo-benchmarks-vs-pokemon-team-building]] — matching lines 108.
- [[Teaching/reference/dimes-citation-audit/assessment-before-owner-review]] — matching lines 22, 34, 36, 52.
- [[Teaching/reference/dimes-citation-audit/project-fit-rubric]] — matching lines 14, 16, 18, 47.
- [[Teaching/reference/dimes-citation-audit/screen-001-037]] — matching lines 85, 91, 97, 109, 211.
- [[Teaching/reference/dimes-citation-audit/screen-075-111]] — matching lines 87, 183.
- [[Teaching/reference/dimes-citation-audit/screening]] — matching lines 23, 24, 25, 27, 42, 44 ….
- [[Teaching/reference/dimes-citation-audit/transfer-candidates]] — matching lines 7, 31, 51, 53.
- [[Teaching/reference/papers-closest-to-what-i-hoped-reevo-was]] — matching lines 25.
- [[Teaching/reference/team-order-experiment-protocol]] — matching lines 89, 91.
- [[Workshops/ICML/EIML 2026 — paper-by-paper relevance screening]] — matching lines 46, 56, 74, 82, 100, 104 ….
- [[Workshops/ICML/SPIGM 2026 — full-paper reassessment]] — matching lines 23, 30, 51, 58, 60, 62 ….
- [[Workshops/ICML/SPIGM 2026 — paper-by-paper relevance screening]] — matching lines 29, 36, 60, 72, 73, 75 ….
- [[Workshops/IEEE CoG/IEEE CoG (Conference on Games)]] — matching lines 50.
- [[Workshops/RLC/Finding the Frame — papers relevant to the counter-team search]] — matching lines 163.
- [[Workshops/_Workshops index]] — matching lines 21, 52, 93, 103, 104, 120.

## Linked implementation reports outside this vault project

The project notes point to these pilot-repository Markdown files. They are linked here separately and are not included in the project-file count above.

- [docs/diversity-metrics.md](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/diversity-metrics.md>)
- [results/temperature_diversity_metrics.md](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_diversity_metrics.md>)
- [results/diversity_before_after.md](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/diversity_before_after.md>)
- [docs/temperature-results.md](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/temperature-results.md>)
- [docs/temperature-experiment.md](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/temperature-experiment.md>)
- [results/temperature_comparison_parallel.analysis.md](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.analysis.md>)
- [results/temperature_selection_audit.md](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_selection_audit.md>)

## Remaining questions already raised by the notes

- Define a strategic team-distance or playstyle descriptor; exact identity and species overlap do not capture team synergy.
- Test structural generalization and reference-corpus novelty with explicitly versioned representations.
- Compare guidance, elite selection and replay using the same sample and battle budgets.
- Choose whether quality-diversity archives are an experiment to run, rather than treating earlier paper suggestions as current implementation decisions.
