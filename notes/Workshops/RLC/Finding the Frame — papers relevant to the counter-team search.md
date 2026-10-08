---
created_at: 2026-08-14
updated_at: 2026-08-21
type: paper-list
source_workshop: Finding the Frame — An RLC Workshop for Examining Conceptual Frameworks
editions_covered:
  - 2024
  - 2025
  - 2026
url: https://sites.google.com/view/findingtheframe
papers_2024: https://sites.google.com/view/findingtheframe/past-years/2024/accepted-papers
papers_2025: https://sites.google.com/view/findingtheframe/past-years/2025/accepted-papers
papers_2026: https://sites.google.com/view/findingtheframe/accepted-papers
schedule_2026: https://sites.google.com/view/findingtheframe/schedule
last_edition: 2026-08-15
last_edition_location: Université de Montréal, room B-4335
screened_2024: 24
screened_2025: 27
screened_2026: 31
screened_total: 82
shortlisted: 16
relevance: medium
tags:
  - reinforcement-learning
  - problem-framing
  - paper-list
  - reading-list
  - multi-agent
  - reward-specification
  - partial-observability
  - relevance/medium
---

> [!summary] Verdict — MEDIUM, for framing only
> None of the 82 papers is about combinatorial search — no coverage, no team selection, no constraint programming, no neural CO. The workshop's value is narrower: it treats *choosing how to state a problem* as research, and the one clarifying move in this project so far was exactly that (weighted set cover → weighted **maximum coverage**). Read these to sharpen how $f$ is written down, not for a method.
> The 2026 list (published 2026-08-18) did not move the grade: it adds one more objective-writing paper and otherwise the same shape.

**How to read the entries.** Grades come from title and abstract, not full reads. Every link is 🟢 (free on OpenReview — 2024/2025 as PDF, 2026 as the forum page). Each entry gives the kind of RL it is · grade · which part of this project it touches ($f$, the battle policy, or evaluation).

## Reading order

> [!tip] Read these four, in this order — stop after #1 if it does not pay off
> 1. [[#1. How to Specify Reinforcement Learning Objectives — HIGH|How to Specify RL Objectives]] — a how-to for writing an objective; the live checkbox *"write the full weighted-maximum-coverage formulation"* is this question.
> 2. [[#11. A Framework for Designing Reward Functions — MED-HIGH|A Framework for Designing Reward Functions]] — the 2026 sequel to #1: objectives → measurable outcome variables → fitted weights, which is the recipe for turning "beat the meta" into a written $f$.
> 3. [[#2. Toward Complex and Structured Goals in Reinforcement Learning — MED-HIGH|Toward Complex and Structured Goals]] — a team that covers a meta is a compositional goal, not a scalar.
> 4. [[#5. Can we hop in general? — MED|Can we hop in general?]] — how to know a proposed counter-team is good: ladder win rate, Monte Carlo evaluation and coverage score are three different benchmarks.
>
> The remaining twelve are context, not reading.

## Shortlist at a glance

| #   | Year | Paper                                                   | Grade    | Touches                                               |
| --- | ---- | ------------------------------------------------------- | -------- | ----------------------------------------------------- |
| 1   | 2024 | How to Specify RL Objectives                            | HIGH     | $f$                                                   |
| 2   | 2024 | Toward Complex and Structured Goals                     | MED-HIGH | $f$                                                   |
| 3   | 2024 | Lambda Discrepancy (partial observability)              | MED      | battle policy                                         |
| 4   | 2024 | Changing and Influenceable Reward Functions             | MED      | $f$ — drift over time                                 |
| 5   | 2024 | Can we hop in general?                                  | MED      | evaluation                                            |
| 6   | 2024 | ODGR: Online Dynamic Goal Recognition                   | MED-LOW  | battle policy                                         |
| 7   | 2025 | Which Rewards Matter?                                   | MED      | $f$ — which threats to weigh                          |
| 8   | 2025 | Analogy making as amortised model construction          | MED      | matchup predictor (concept)                           |
| 9   | 2025 | Remembering the Markov Property in Coop MARL            | MED-LOW  | battle policy                                         |
| 10  | 2025 | Ricci-curvature lens on environment complexity          | LOW-MED  | none clear                                            |
| 11  | 2026 | A Framework for Designing Reward Functions              | MED-HIGH | $f$                                                   |
| 12  | 2026 | Reward Learning via Differentiable Trajectory Alignment | MED      | evaluation — does $f$ rank teams like the ladder does |
| 13  | 2026 | Are Preferences Discovered or Created?                  | MED-LOW  | $f$ — scalar or not                                   |
| 14  | 2026 | Hidden Tuning is Misleading The RL Community            | MED-LOW  | battle-policy experiment                              |
| 15  | 2026 | Recurrent Memory as State Abstraction Over History      | MED-LOW  | battle policy                                         |
| 16  | 2026 | Infra-Bayesian RL for Worst-Case Robustness             | LOW-MED  | worst-case framing                                   |

## 2024 — 6 of 24

### 1. How to Specify Reinforcement Learning Objectives — HIGH

- 🟢 [PDF](https://openreview.net/attachment?id=2MGEQNrmdN&name=pdf) · 🟢 [companion site](https://bradknox.net/rl-objectives-how-to/)
- *objective / reward specification* (reward + discount design)
- The one paper here that changes how I write $f$: which meta threats get weight, and whether $f$ is win rate or coverage.

### 2. Toward Complex and Structured Goals in Reinforcement Learning — MED-HIGH

- 🟢 [PDF](https://openreview.net/attachment?id=YkmTFoeuc1&name=pdf)
- *structured goal representation* (cognitive-science flavoured)
- "Six candidates that cover the meta" is a structured, compositional goal.

### 3. Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy — MED

- 🟢 [PDF](https://openreview.net/attachment?id=5Lv8kD03Mu&name=pdf)
- *partial observability* — a measurable signal that the state is non-Markov
- VGC is heavily partially observed (items, EVs, tera type, the opponent's selection). Bears on the battle policy; nothing to do with the species enumeration.

### 4. AI Alignment with Changing and Influenceable Reward Functions — MED

- 🟢 [PDF](https://openreview.net/attachment?id=36hh5wcLh3&name=pdf)
- *non-stationary and endogenous reward*
- The *changing* half applies: $f$ is win rate against one regulation's meta on one date, and both drift, so re-solve against a fresh meta-team list. The *influenceable* half does not — the meta does not respond to me.

### 5. Can we hop in general? — MED

Full title: *Can we hop in general? A discussion of benchmark selection and design using the Hopper environment*.

- 🟢 [PDF](https://openreview.net/attachment?id=9IgtF63LPA&name=pdf)
- *evaluation / benchmark design*
- The failure modes it describes are the ones ladder validation vs Monte Carlo evaluation vs coverage score can each have.

### 6. ODGR: Online Dynamic Goal Recognition — MED-LOW

- 🟢 [PDF](https://openreview.net/attachment?id=q8zrfiAr24&name=pdf)
- *goal recognition / inverse planning, online*
- Inferring an opponent's plan from partial play. In-battle, not team-building.

> [!info]- Screened out — 18 of 24
> - **Game-theory position paper** — Milnor-Myerson Games and the Principles of Artificial Principal-Agent Problems (dropped 2026-08-18)
> - **Exploration** — Satisficing Exploration · Exploration Unbound · Characteristics of Effective Exploration for Transfer
> - **Continual RL** — Big World Hypothesis · Big World Simulator · Biologically-Inspired Continual RL · Pick up the PACE
> - **Discounting** — two hyperbolic-discounting papers
> - **Method and engineering** — MOSEAC · Realtime RL · hyperparameter sensitivity · classification-vs-regression policy gradients · Reward is (almost) enough · telic state representations · decision-time planning costs · RL in context

## 2025 — 4 of 27

### 7. Which Rewards Matter? Reward Selection for RL from Limited Feedback — MED

- 🟢 [PDF](http://openreview.net/pdf/d6de3803bd5ee147231b25b12604179269b4767a.pdf)
- *reward design from limited feedback*
- Same shape as choosing which meta threats to weight when only so many matchups can be measured.

### 8. Analogy making as amortised model construction — MED

- 🟢 [PDF](http://openreview.net/pdf/3ed5acd1f45fb53f4fa65058a7f1fe11681bac87.pdf)
- *model-based RL, amortised inference*
- For the word *amortised*: pay training cost once, answer any new meta instantly. That is what the matchup predictor is. Concept only.

### 9. Remembering the Markov Property in Cooperative MARL — MED-LOW

- 🟢 [PDF](http://openreview.net/pdf/dd1edc6f45e483ee46c5e11a6cc0f8d53cd18fa6.pdf)
- *cooperative multi-agent RL*
- Doubles is a two-agent cooperative problem (my two active candidates). Battle policy, not team-building.

### 10. A Geometric Lens on RL Environment Complexity Based on Ricci Curvature — LOW-MED

- 🟢 [PDF](http://openreview.net/pdf/f16dbedfbba525c7791426f8348c8c8c68ecdb09.pdf)
- *state-space geometry*
- Measures how hard a state space is; no evidence it says anything about a discrete combinatorial space. Last resort.

> [!info]- Screened out — 23 of 27
> - **Dropped 2026-08-18 after regrading to LOW** — Learning Context-Sensitive State and Action Abstractions for RL with Parameterized Actions · Reframing MARL with Variational Inequalities · Budget-Aware Feature Selection for RL
> - **Offline RL** — Clean Slate · Off by a Beat · Set-Valued Policy / Dead-End
> - **Value-based methods** — Iterated Q-Learning · Data Reuse · Distribution Parameter Actor-Critic
> - **On-policy sampling and parallelism** — On-Policy Without On-Policy Sampling · Staggered Resets
> - **Partial-observability variants** — General Value Discrepancies · Learning What to Remember
> - **Other** — Thinking is Another Form of Control · Agent-centric learning · Is Exploration or Optimization the Problem · Zero-Incentive Dynamics · A Unified MDP Framework · fluid-mechanical environments · Initial State Distribution · Distributional Uncertainty · Temporal Credit Assignment · the evolutionary three-dogmas paper

## 2026 — 6 of 31

Held 2026-08-15; the [accepted-papers page](https://sites.google.com/view/findingtheframe/accepted-papers) went live 2026-08-18 with 4 orals + 27 posters. Links are the OpenReview forum pages (the PDF sits one click behind a bot check).

### 11. A Framework for Designing Reward Functions — MED-HIGH

Full title: *A Framework for Designing Reward Functions: From Objectives to Features to Human-Aligned Reward Functions*.

- 🟢 [OpenReview](https://openreview.net/forum?id=MxwaByqdsD)
- *reward specification* — a three-step process: distil the task into fundamental objectives and measurable outcome variables, pick a causally representative subset as reward terms, fit the weights by preference elicitation
- The follow-up to #1, and the same question as the live checkbox: which measurable quantities (coverage of which threats, at what weight) make up $f$. Read directly after #1.
- Read 2026-08-18: 7-page process paper, no experiments. Only §3.1 + Table 3 (objectives → proxies; diversify gameable proxies) transfer to $f$; the min-cut term selection and ACCPM weight fitting need a causal DAG and a human preference oracle the project doesn't have. Grade under review → MED.

### 12. Reward Learning via Differentiable Trajectory Alignment — MED

- 🟢 [OpenReview](https://openreview.net/forum?id=jCaNuVdXxK)
- *reward quality metric* — the Trajectory Alignment Coefficient: ordinal correlation between a reward's induced preferences and an expert's
- Concept only: rank correlation between "$f$'s ordering of teams" and "the ladder's ordering of teams" is a cleaner validation statistic than a raw win rate. Nothing else transfers.

### 13. Are Preferences Discovered or Created? — MED-LOW

- 🟢 [OpenReview](https://openreview.net/forum?id=UaKyVoAqYA)
- *position paper* — reward learning assumes a pre-existing scalar utility; multi-criteria decision aiding says the preference is constructed
- Names the assumption behind writing $f$ as one scalar. Vocabulary, not a method.

### 14. Hidden Tuning is Misleading The RL Community — MED-LOW

- 🟢 [OpenReview](https://openreview.net/forum?id=3WWVEce7G6)
- *empirical methodology* — unreported hyperparameter tuning makes algorithm comparisons unfair; expose the tuning phase
- Applies to comparing battle-policy arms in a cross-play matrix, where an arm tuned longer wins for the wrong reason. Battle policy, not team-building.

### 15. Recurrent Memory as State Abstraction Over History — MED-LOW

- 🟢 [OpenReview](https://openreview.net/forum?id=WaV17GnVpt)
- *partial observability* — POMDP classes defined by finite-state memory functions
- Same thread as #3 (Lambda Discrepancy). Battle policy only.

### 16. Infra-Bayesian RL Agents Outperform Classical RL for Worst-Case Robustness — LOW-MED

- 🟢 [OpenReview](https://openreview.net/forum?id=9a07cvcYHW)
- *robust / worst-case RL* — agents facing opponents who anticipate them
- This is what "worst-case" would mean. Kept only as the marker of the road not taken.

> [!info]- Screened out — 25 of 31
> - **Orals** — Frameless Agentic System Theory · Toward Enactive Artificial Intelligence · Generalised Bellman recurrence and three dualities of sequential decision-making
> - **Evaluation and methodology** — What are we measuring? A critical examination of MO-MuJoCo · Corroborate: testing mechanism claims · Off Policy Evaluation Under Temporal Mismatch · Confidence Intervals for the Return Process in MDPs
> - **Agency, cognition and framing essays** — On Plasticity and Empowerment · Genuine agency and the cognition of living beings · Toward Aligned Open-Endedness · The Goal-Directed Frame for General Agents · Learning & Remembering: Differential Expressivity · Transitions Decompose as Passive and Agent Dynamics
> - **Exploration and observation** — Learning What to Observe: Endogenous Sensing · Beyond directedness: meta-exploration · Efficient Exploration Is Enough
> - **Method and engineering** — RENEW (world-model repair from preferences) · Do Not Imitate, Reinforce (RIC) · Reinformed Dreamer · Beyond One-Size-Fits-All (offline priors) · Generalization in Offline RL: structure vs pessimism · Stiefel Manifold Optimization · Deep RL and The Tale of Two TD Errors · Beyond the Monolith (higher-order credit assignment) · The Challenges of Using RL for Industrial Energy Systems

**Invited talks** (no papers) — *Intelligence in Action* · *Surveys in RL (and RLHF)* · *Sequences of Frames: The Analogy with Memory Retrieval* · *Does RL have a blind spot for Knightian uncertainty?*. None marked to watch: the subgoal-discovery thread behind the first was an abstraction lead, and abstraction-as-reduction is retired.

- [x] re-check <https://sites.google.com/view/findingtheframe/accepted-papers> and add the 2026 shortlist here — done 2026-08-18: 31 papers screened, 6 shortlisted (#11–#16 above)

## Caveats

> [!warning] Provenance
> - Titles and links are copied from the workshop's accepted-papers pages.
> - Author names and affiliations are omitted throughout, at the owner's request (2026-08-18) — the papers stand on their content.
> - 2026-08-18: four papers dropped from the original 14 at the owner's call (Milnor-Myerson; Context-Sensitive Abstractions; MARL via Variational Inequalities; Budget-Aware Feature Selection) — all LOW once the equilibrium framing and search-space reduction were retired. They are named in the screened-out lists. Nothing else was re-derived.
> - 2026 entries were graded from the abstracts as posted on the workshop page (first ~600 characters read for each), not from the OpenReview PDFs.

## Backlink

Hub: [[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]] ·
workshop note: [[Finding the Frame (RLC 2024)]] ·
related: [[Diffusion for conditional counter-team generation]], [[DiffCoALG (NeurIPS 2025)]]
