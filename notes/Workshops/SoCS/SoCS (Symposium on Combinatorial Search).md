---
created_at: 2026-08-14
updated_at: 2026-08-14
url: https://socs26.search-conference.org/
series_url: https://search-conference.org/
accepted_papers: https://socs26.search-conference.org/accepted-papers
proceedings: https://ojs.aaai.org/index.php/SOCS/index
venue: SoCS 2026 — 19th International Symposium on Combinatorial Search
dates: 2026-08-14/2026-08-16
location: ATLANTIC Hotel Sail City, Bremerhaven, Germany
deadline_2026: 2026-03-16
type: conference
status: running now
scope: heuristic search; discrete optimization; constraint programming; planning; OR
relevance: medium-high
tags:
  - combinatorial-search
  - heuristic-search
  - discrete-optimization
  - integer-programming
  - qubo
  - relevance/medium-high
---

## What it is
The field's own venue — "heuristic search and other forms of **combinatorial search
optimization**." Its stated scope is literally the list of things my project touches: AI,
planning, discrete optimization, constraint programming, operations research. The 19th edition
is **running 2026-08-14 to 2026-08-16** in Bremerhaven, Germany, i.e. as I write this.

## Relevance to my search problem — **MEDIUM-HIGH, but narrowly**
Honest read of the full 2026 accepted list: **SoCS is dominated by pathfinding and multi-agent
path finding (MAPF).** Roughly half the accepted papers are MAPF, A*/best-first variants, or
planning heuristics. None of that is subset selection or coverage, and I should not expect to
find my problem here.

But two papers are directly on my line — and both are about **learning-guided solving of the
exact mathematical programs I'd be writing**. That's a real payoff, so MEDIUM-HIGH rather than
MEDIUM. Worth noting Bistra Dilkina is an author on both; that group is the thread to pull.

Also relevant to the venue's *culture*: SoCS explicitly encourages **real-world applications** of
heuristic search. A VGC counter-team search is exactly the kind of applied-search paper it asks
for, which makes it a plausible eventual submission target (2027 edition).

## Papers that could interest me
All from the [SoCS 2026 accepted papers](https://socs26.search-conference.org/accepted-papers):
1. **ML-Guided Primal Heuristics for Mixed Binary Quadratic Programs** — Weimin Huang, Natalie Isenberg, Ján Drgoňa, Draguna Vrabie, Bistra Dilkina. **Mixed binary quadratic programming *is* the QUBO family.** This is machine learning used to find good solutions fast to precisely the objective shape I'm building (binary variables + quadratic synergy terms). The most on-point paper of the whole scan for my QUBO thread.
2. **ID-PaS+: Identity-Aware Predict-and-Search for Solving General Mixed-Integer Linear Programs** — Junyang Cai, El Mehdi Er Raqabi, Pascal Van Hentenryck, Bistra Dilkina. My weighted-maximum-coverage formulation is a MILP; "predict-and-search" is the ML-guided way to solve one. The classical counterpart to the generative approach in [[Diffusion for conditional counter-team generation]].
3. **From Projections to Cartesian Abstractions and Merge-and-Shrink: Comparing the Compactness of Abstraction Representations** — Malte Helmert, Gabriele Röger. About how *compactly* different abstractions represent a state space — the principled version of my open hub question on how to store/compact all possible teams.

## Backlink
Hub: [[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]] ·
index: [[_Workshops index]] · siblings: [[CP (Constraint Programming conference)]], [[CPAIOR (CP-AI-OR integration)]]
