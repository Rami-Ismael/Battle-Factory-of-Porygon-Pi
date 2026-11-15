# Facts and rules for the "Draft Version two" pass, 2026-11-13

Every subagent on this pass reads this file first. Paths are relative to the repository root
`/Users/ramiismael/.t3/worktrees/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/t3-13742f9b`.
`SITE` means `notes/artifacts/one-edit-away-explanation-elements`.

## The post

Rami Ismael is writing his third LinkedIn post in a build-in-public series about his Pokemon team search project. The post explains one face of why Pokemon team building is a hard optimisation problem: smoothness, measured by changing one thing on a team and seeing how far its win rate moves. He writes the post himself. This pass produces material for him: candidate sentences with checked sources, answers to open questions, and versions for other sites.

Reader: engineers and students curious about machine learning, plus hiring managers for data science and machine learning roles. They know what an optimiser is. They do not know competitive Pokemon.

His first draft, word for word, do not correct it and do not quote it as finished prose:

> During, I am planning to make the case why more people should study pokemon as a real scientific problem to understand. Today, I will be discussing about the smoothness in optimization space pokemon. If you look other academic benchmark for optimization they are smooth. Why they are mostly smooth. Smoothness is mathematic term value that are close input space are close in output space aswell. This is useful information in ML context it help that carry information from one point can mean the value move the input to a new space the value will near the value make it easier to hill climb. However, when the space is smooth the value change drastically you need more input space to get meaning near the value of thenext point you travel too. We make the transition function simple to be changine one properties of the pokemon it can be item, moves, entire pokemon then measure the win rate. We measure how much change. If change greater than noise it change it hard. We are using the # One edit away artifact ther should be an index.html with it aswell and the visual edit by edit as wel. It interesting problem it require to solve the problem with different method. One build two optimziation metha MAP and Cross entropy method to apply optimzation .

His checklist for this pass is in the vault note, read only, never edit it:
`/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Third Post on Linkedin about one edit away explain the one face of difficulty of pokemon in smoothness and why smothness matter in pokemon.md`

## The finished measurement: the edit-kind rerun, checked 2026-11-13, GitHub issue #313

Design. 40 starting teams in two groups. 24 tournament-ready teams: real teams players built for tournaments, drawn from outside the top 50. 16 search-found teams: teams an earlier search produced. Each team got 18 edits, three of each of six kinds. Each score is a win rate over 384 battles against the 50 top-placing tournament teams, 49 of them distinct, with the same behaviour-cloning battle policy playing both sides and team slots shuffled. Each starting team was also replayed three times with nothing changed, to measure battle luck. 880 scores in all.

The six edit kinds. Move, ability, item and stat alignment: one field replaced by a random legal value. Stat Points: copied from a real set of the same species. Whole Pokemon: one Pokemon replaced by a real set of a different species.

Size of one edit, in win-rate points, after removing battle luck. The bracket is the 95% interval. Share is the size divided by the spread of true scores between teams, which is 9.04 points.

| Edit kind | Tournament-ready teams | Share | Search-found teams | Share |
|---|---|---|---|---|
| Whole Pokemon | 5.90 [4.78, 7.01] | 0.65 | 5.85 [3.83, 7.47] | 0.65 |
| Move | 4.84 [3.35, 6.23] | 0.54 | 4.92 [2.05, 7.24] | 0.54 |
| Item | 4.35 [2.33, 5.89] | 0.48 | 1.44 [0, 3.33] | 0.16 |
| Ability | 3.66 [2.22, 4.71] | 0.40 | 1.35 [0, 2.96] | 0.15 |
| Stat alignment | 1.95 [0, 2.92] | 0.22 | 0.00 [0, 2.53] | 0.00 |
| Stat Points | 1.60 [0, 2.63] | 0.18 | 1.72 [0, 2.69] | 0.19 |

Reading one, above zero. An edit kind is real when its interval excludes zero. Whole Pokemon and move pass in both groups. Item and ability pass for tournament-ready teams only. Stat alignment and Stat Points pass in neither group. "Not shown above zero" does not mean "does nothing".

Reading two, against a benchmark step. Pest Control is a standard benchmark for Bayesian optimisation with 25 stations and 5 choices each. One step on it changes one station. The step is divided by the spread between benchmark starting points of its own kind: 0.37 from random points and 0.32 from hill-climbed points.

**Decision on the benchmark spread, owner, 2026-11-13.** A first version pooled the benchmark's random and hill-climbed starting points, which gave 0.14, because the gap between their mean scores, 18.5 and 14.2, entered the spread. Rami chose each kind on its own spread. The pooled 0.14 is not used. Under this reading the whole-Pokemon swap from tournament-ready teams is larger than a Pest Control step. From search-found teams it is undecided, missing narrowly. Move, item and ability edits are undecided against the step in both groups. Stat alignment from search-found teams is smaller. This file was written while the choice was open, so the subagents' outputs were corrected afterwards by `merge.py`.

Noise. Two scores of the same unchanged team differ by 3.40 points for tournament-ready teams and 3.73 points for search-found teams, root mean square, at 384 battles each. One score's own noise is about 2.55 points.

Scope limits that every public claim must respect.
- One battle policy, behaviour cloning, plays both sides. Win rate under it stands in for team strength.
- One opponent pool, the 50 top-placing teams.
- Edits are random legal values, not the edits a player or a search would choose. About a quarter of item edits remove a Mega Stone.
- The measurement is the size of a change, not its direction.
- The reading without a size cutoff was chosen after the result was seen. The reading registered before the run was "only the whole-Pokemon swap passes both tests".
- A share of 0.65 means 0.65 of one standard deviation of team scores. It does not mean two thirds of the gap between two typical teams.

Result files. `~/vgc-data/edit_kind_rerun/run/results/ruggedness2_shuffled50.json`, `SITE/rerun-readings.json`, `code/docs/ruggedness2.md` last sections, `notes/reports/edit-kind-rerun-2026-11-13/issue-313/spec.json` for real example edits.

## Result of the top-50 study, GitHub issue #317, checked 2026-11-14

The same six edit kinds applied to the 49 distinct top-50 teams themselves, 18 edits each, 413,952 battles. Plan and result: `code/docs/top50-one-edit.md`. Page copy: `SITE/top50-result.json`.

Size of one edit on a top-50 team, points, with its 95% interval, as registered: whole Pokemon 4.91 [3.74, 5.83], move 3.04 [1.05, 4.10], item 3.12 [1.21, 4.32], ability 2.11 [0, 3.31], stat alignment 0.00 [0, 1.81], Stat Points 0.00 [0, 1.01]. Above zero: whole Pokemon, move, item. Larger than a Pest Control step: the whole-Pokemon swap only, by a margin of 0.01. Stat alignment and Stat Points read smaller than a step. Random edits lower win rate on average for the whole-Pokemon, move, item and ability kinds.

A check made after the result measures each edit against all four unchanged scores of its team: 5.28, 3.87, 3.62, 2.76, 1.06, 0.00. It puts the ability edit above zero. No difference between top-50, tournament-ready and search-found teams is shown, and no trend with a team's rank. This section was added after the subagents wrote; `merge.py` corrected their lines.

## Numbers that no longer count

- Any Pokemon number from September or October 2026, such as neighbour correlations 0.96 and 0.92. Those battles used 36 opponents in listed slot order and are retired.
- The 5-point cutoff. It was dropped on 2026-11-13. Never mention a 5-point line as a current test.

## Research already on record from 2026-11-12

- No paper found says benchmarks are smooth because methods need smoothness. That link is an inference.
- Oh et al. 2019, Combinatorial Bayesian Optimization using the Graph Cartesian Product, https://arxiv.org/abs/1902.00448, Section 1: "GPs rely on the smoothness defined by a kernel to model uncertainty". Pest Control comes from this paper.
- Hansen et al., COCO, https://arxiv.org/abs/1603.08785, Section 2: benchmark functions "designed to be comprehensible". The suite includes rugged functions on purpose.
- Papenmeier, Poloczek, Nardi 2025, https://arxiv.org/abs/2502.09198, Section 4: some popular benchmarks "have a simple structure".
- Pelikan 2008, https://arxiv.org/abs/0801.3111, Section 4.5: on NK landscapes the ranking of search methods reverses as the interaction count k grows.
- González-Duque et al. 2024, https://arxiv.org/abs/2406.04739, Section 4.2.1: "For more complex problems, BO optimizers perform equally or worse than greedy baselines". This is about problem complexity, not smoothness.
- Rami's own benchmark comparison, `notes/Landscape properties comparison with other black box function by smoothness.md`, measured 10 tasks. Five stay smooth. Five turn rough near good points, and the molecule tasks are rougher than Pokemon. Its verdict: "Overall roughness does not set VGC apart".

## Wording rules for anything a reader will see

- Plain sentences. No em dashes. No parentheses. No semicolons. No bullets that start with a bold label.
- Spell out every abbreviation. Say "behaviour cloning", "Bayesian optimisation", "cross-entropy method".
- Say "Stat Points" for a Pokemon's points. Never call them a "spread". "Spread" is only the spread of scores between teams.
- Say "stat alignment". Say "cutoff", never "threshold".
- The first group is "tournament-ready teams". Never "meta teams". The "top 50" are the opponents.
- Never call anything a "baseline" in public text.
- Never say hill climbing fails on Pokemon, or that Pokemon is rougher than every benchmark. Neither was measured.
- "Depends on the other five Pokemon" is a hypothesis until measured. Mark it as one.
- A claim with a number carries its source. A source is a file path with its section, or a paper with its section and link. Open every source you cite.

## Contract for element subagents

Each current element is a JSON file `SITE/NN-name.json` with keys `action`, `element`, `what_it_does`, `when_to_use`, `sentence_pattern`, `dllm_example`, `anchor_in_draft`, `extra`, `candidates`, `gaps`. A candidate is `{"sentence": ..., "facts": [{"claim": ..., "source": ...}]}`. `gaps` are open questions for Rami. A gap that has been answered starts with "Answered 2026-11-13:".

Write the revised element to `SITE/v2/NN-name.json`, the same file name, with the same keys plus two more: `picked`, the zero-based index of the candidate Rami approved or null, and `applied`, one or two sentences on what you changed in answer to his note. Keep three candidates unless your brief says otherwise. Remove or correct any claim this file contradicts. Do not edit any file outside `SITE/v2/`. Do not run `build.py`. Do not start battles or any long job.

## Contract for research subagents

Write a report to `SITE/v2/<slug>.md`: the verdict in the first two lines, then a table of evidence with exact quotes, section numbers and links, then what you could not verify. Quote only text you opened and read. In Bash, a hook blocks commands that contain the words `curl` or `wget`; use the WebFetch and WebSearch tools or the alphaxiv tools.
