---
created_at: 2026-08-20
type: learning-record
lesson: 0007-coupling-marginals-fixed-joint-is-yours.html
---

# 0008 — Coupling: marginals fixed, the joint is a design choice (prerequisite for CRN)

Taught 2026-08-20, same day as lesson 0006, after the owner said he was missing a prerequisite for 0006 §1 (the "luck cancels" paragraph). Diagnosed in chat as the fifth of five candidate prerequisites — not covariance mechanics (he's a CS grad) but the idea that **the joint distribution of two simulated estimates is engineerable while their marginals stay fixed**, i.e. a coupling. This gap matters pedagogically: probability courses always *give* the joint; simulation *chooses* it. He confirmed by asking for the micro-lesson.

Key insights the lesson carries:

- Marginals pin the row/column totals of the 2×2 outcome table; many joints share those totals. Covariance lives in the joint (E[XY] is a joint cell).
- The shared-U construction: X = 1{U < p₁}, Y = 1{U < p₂} keeps both marginals (U alone is still uniform to each side) while zeroing the (X=0, Y=1) cell; per-pair difference variance drops ≈ ×10 at p₁=0.55, p₂=0.50 (0.047 vs 0.497).
- Unbiasedness is one line: E[X−Y] = E[X]−E[Y] holds under any dependence, because each expectation reads only its own untouched marginal. Coupling is a variance knob with the bias bolted at zero.
- Named the two directions of the knob: CRN (shared draws, cov > 0, for differences) vs antithetic variates (U and 1−U, cov < 0, for sums) — the latter mentioned only as an aside.
- Hook back: independent seeding = product coupling; CRN = shared-draw coupling; X_DD (lesson 0006) = knowing *where in the trajectory* to switch coupling.

New reusable component: `assets/coupling-sim.js` — Part A drag-one-shared-U number line (green both-win zone, orange disagreement sliver), Part B side-by-side 2×2 joint tables under both couplings with marginals dashed, disagreement cell outlined, empirical cov and var(X−Y).

Zone-of-proximal-development note for future lessons: prerequisites #1–#4 from the chat diagnosis (estimator-as-random-variable, covariance definition, bilinearity/the identity, unconditional linearity of expectation) were folded in briefly rather than taught — if a later quiz shows the identity itself is shaky, that is its own micro-lesson.

Primary source filed: Blitzstein & Hwang, free at probabilitybook.net (301 → official Google Drive PDF, verified 2026-08-20), Ch. 7 / §7.3.
