---
created_at: 2026-08-20
type: learning-record
lesson: 0006-common-random-numbers-for-paired-battles.html
---

# 0007 — Common random numbers for simulation-based comparisons (Yadav et al., RLJ 2026)

Taught 2026-08-20 at the owner's request (`/teach` with the paper title), serving the HIGH item in `Todo Section.md` § "the rest of RLC 2026" (RLJ 2026 Paper 52). Retrieval quiz not yet confirmed taken. Note: the topic touches the Monte-Carlo-adjacent area under the owner's moratorium (until 2026-08-23), but the owner raised it himself — the lesson keeps to variance mechanics and quotes his own TODO framing, asserting nothing about the open MC-only-vs-predictor question.

Key insights the lesson carries:

- **One identity**: var(Δ̂) = var(T̂₁) + var(T̂₂) − 2 cov(T̂₁, T̂₂). CRN is the bet that coupling makes cov positive; it never biases, it only moves variance. Negative cov (Proposition 1's constructed MDP) makes full coupling *worse* than independence.
- **Three estimators**: X_I (independent), X_D (fully coupled — usually pays, no guarantee), X_DD (independent through depth d where the compared decisions differ, coupled beyond, where a *shared rollout policy* runs — provably var ≤ X_I, Theorem 2). The proof hinge: once both trajectories run the same policy on the same transitions, the covariance is a variance, hence ≥ 0.
- **Implementation is a seeding discipline**, not infrastructure: seed = state + action + step + simulationIndex, appending the policy id only for t ≤ d.
- **Project mapping is two distinct slots**: team-vs-team comparison against the meta is X_D territory (teams diverge from turn 1, no theorem; the dependable coupling channel is the shared *opponent draw*, and a shared Showdown battle seed couples less than it seems once histories diverge); rollout-based battle-policy planning (UCT shape) is X_DD's literal home.
- **Empirics**: gains largest at small simulation budgets; Ludo (5000 games) — coupled seeding clearly dominates, and plain X_D beat X_DD there (theory doesn't model UCT's adaptive tree growth; flagged open by the authors).

New reusable component: `assets/crn-sim.js` — three-column paired-comparison variance demo (independent / paired opponent / fully paired) with a "×k fewer battles" readout; the middle column exists to show opponent-draw pairing alone already pays.

Quotes in the lesson were verified verbatim against the arXiv HTML (2026-08-20) before use.
