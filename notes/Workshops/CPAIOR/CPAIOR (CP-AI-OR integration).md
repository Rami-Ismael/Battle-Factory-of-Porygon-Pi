---
created_at: 2026-08-14
updated_at: 2026-08-18
url: https://sites.google.com/view/cpaior2026/home
series_url: https://cpaior.org/
proceedings: https://link.springer.com/book/9783032272416
series_proceedings: https://link.springer.com/conference/cpaior
submissions: https://openreview.net/group?id=cpaior.org/CPAIOR/2026/Conference
venue: CPAIOR 2026 — 23rd International Conference
dates: 2026-05-26/2026-05-29
location: ESSEC Business School campus, Rabat, Morocco
acceptance_2026: 37 full papers from 92 submissions
type: conference
status: retired 2026-08-18
relevance: low
tags:
  - constraint-programming
  - operations-research
  - machine-learning-for-optimization
  - predict-and-optimise
  - relevance/low
---

*Retired as a venue 2026-08-18.* Scanned the full accepted lists for [2025](https://sites.google.com/view/cpaior2025/accepted-papers) (32 papers) and [2026](https://sites.google.com/view/cpaior2026/accepted-papers) (37): zero highly relevant — MILP heuristics, scheduling, symmetry breaking (retired topic), generic set-cover learning. The one HIGH paper in the series (Demirović 2019, below) belongs to the decision-focused-learning thread, which lives at AAAI / JAIR / ICML, not here. Nothing to watch at this conference; the reading list below is kept because it is the only home for that thread.

## What it is
The conference for CP + AI + OR combined. 23rd edition, Rabat, May 2026. Selective: 37 of 92.

## Relevance — LOW
*Retired 2026-08-18 (second time; first pass same day took it Med-High → Med) — was Med-High as "best conceptual fit": clauses = CP, ILP objective = OR, learned win-rate oracle = AI. [[Decision — enumerate species subsets, don't optimise]] *(note deleted 2026-08-18, no replacement)* removed the solver and the clauses-as-constraints; "oracle" is banned vocabulary (it is the matchup predictor).*

What survives is one term: **predict-and-optimise** — fit the matchup predictor, then choose the team by $f$. CPAIOR's 2026 topics list it by name (with neurally-guided solving and LLM-based approaches). Its decision-focused-learning literature — train the predictor for the quality of the downstream choice, not raw accuracy — is the one thread still worth reading. Everything else (propagation, cutting planes, column generation) is for people who still run a solver.

## Papers
Verified 2026-08-18 (title page fetched, venue looked up). One is CPAIOR; the other three are the predict-and-optimise thread's anchors at other venues, kept here because this note owns the thread.

| # | Paper | Authors | Venue | Access | Grade | Why | Read order | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization | Wilder, Dilkina, Tambe | AAAI 2019 | 🟢 [arXiv 1809.05504](https://arxiv.org/abs/1809.05504) | **HIGH** | Trains the predictor for downstream decision quality; instantiated for **submodular maximisation** — weighted max coverage is submodular, so this is the closest published shape to $f$ | 2 | ☐ unread |
| 2 | An Investigation into Prediction + Optimisation for the Knapsack Problem | Demirović, Stuckey, Bailey, Chan, Leckie, Ramamohanarao, Guns | **CPAIOR 2019** | 🟢 [author PDF](https://people.eng.unimelb.edu.au/pstuckey/papers/cpaior19b.pdf) · 🔒 [Springer](https://doi.org/10.1007/978-3-030-19212-9_16) | **HIGH** | Knapsack = pick a subset under a budget, i.e. my shape. Compares plain regression vs. ranking losses for the predictor — the exact question for a ridge matchup predictor. Defines *regret*; with 593,775 subsets I can compute it exactly | 3 | ☐ unread |
| 3 | Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities | Mandi, Kotary, Berden, Mulamba, Bucarey, Guns, Fioretto | JAIR 81, 2024 | 🟢 [arXiv 2307.13565](https://arxiv.org/abs/2307.13565) | **HIGH** (orientation) | The survey; 11 methods × 7 problems. Vocabulary first | 1 | ☐ unread |
| 4 | Smart "Predict, then Optimize" | Elmachtoub, Grigas | Management Science 68(1), 2022 | 🟢 [arXiv 1710.08005](https://arxiv.org/abs/1710.08005) | **MED** | Origin of the term; SPO+ loss. Linear objectives only — foundation, not a fit | 4 | ☐ unread |

Caveat for all four: they differentiate *through a solver* because their optimise step is expensive. Mine is exhaustive and instant, so only their **losses** (regret, ranking) transfer — not the relaxation machinery.

More:
- 🟢 [CPAIOR 2026 on OpenReview](https://openreview.net/group?id=cpaior.org/CPAIOR/2026/Conference) — papers and reviews, free. Start here.
- 🟢 [Series site](https://cpaior.org/) — past editions; search `predict-and-optimize`, `decision-focused`.
- 🔒 [2026 proceedings (Springer)](https://link.springer.com/book/9783032272416) — table of contents only; read via OpenReview.

- [x] Pick 3 predict-and-optimise / decision-focused papers — done: the four above, 2026-08-18
- [x] Skim the CPAIOR 2026 list for anything newer on submodular / subset-selection DFL — done: 2025 + 2026 scanned, nothing; venue retired 2026-08-18

## Backlink
Hub: [[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]] ·
index: [[_Workshops index]] · siblings: [[CP (Constraint Programming conference)]], [[MLxOR 2026 (NeurIPS 2026)]]
