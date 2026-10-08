---
created_at: 2026-08-19
status: active
---

# Metamon — offline RL from human replays, BC placed outside the self-play loop (Lesson 0003)

Context: owner asked to be taught the Metamon paper (RLC 2025, arXiv 2504.04395). Chat digest from the v2 PDF, then Lesson 0003 with a filter-weight simulator (`assets/filter-sim.js`) and a 6-question quiz; cheat sheet in `reference/metamon-cheat-sheet.html`.

Key insights to retain:
1. BC / offline RL from replays is not a self-play method — no Π, no σ on the first pass; it is the seed policy a loop can start from. Fills the gap flagged in record 0002.
2. One actor loss, four rows (IL / Exp / Binary / Binary+MaxQ); the critic is a data filter, not an action picker. Finding: any RL row ≫ IL, RL rows ≈ each other.
3. Realistic self-play (SynRL-V1+SP) improved cross-play and regressed on the ladder — the model learned to model itself. Their fix: deliberately unrealistic "Variety" teams as an OOD watermark. Direct consequence for the owner's cross-play matrix: keep an external (non-arm) column.
4. Their dataset D plays the role the owner's meta plays — an opponent distribution under a stated "history ≈ now" assumption.

Mismatches recorded in the lesson: singles vs doubles, no preview vs preview+selection, 9 vs ~107 actions.

Not yet known: quiz results; whether MISSION.md should be widened to the battle-policy strand (still unasked — two lessons now serve it). Likely next in ZPD: App. D replay reconstruction (how unrevealed sets are inferred — touches the usage-% exclusion), AMAGO multi-γ heads, two-hot value classification, or the singles→doubles pretraining design in the owner's own experiment note.

Update 2026-08-19 (later): owner caught that the lesson's "one equation" covers only the actor — asked where the value function is trained. Researched AMAGO App. A and AMAGO-2 Sec. 3; added §3b to the lesson and a critic block to the cheat sheet (TD regression, λ = 10, target heads, REDQ-4, multi-γ, PopArt; two-hot CE with 96 bins on [−110,110] for V2). Good sign: he is reading for what is *missing*. Retest: "write y_t and L_Critic from memory; where does A enter Eq. 2?"

Update 2026-08-19 (later still): owner asked what "in-context" means and whether other battles are in the context. Answered (single battle only; fn. 1 Bo3 extension unbuilt) and drew `reference/metamon-in-context-window.html` (SVG: window vs weights vs latent vs previous battles). Also clarified Bayesian-RL framing vs RL²-style black-box method. Retest: "what is in one turn token; where does the dataset live — window or weights?"

Grilling round 2026-08-19 (evening) — first retention evidence on Lesson 0003:
- Q1 reconstruction: STRONG — unprompted, flagged the dataset trick as an error source (matches App. D.1 / "sim-to-real gap"). Critical-reading instinct is ahead of recall.
- Q3 findings: HALF — retained "RL rows indistinguishable" but dropped the main clause "any RL ≫ IL". Re-drill as a paired sentence.
- Q4 context window: WEAK — said the model "sees the opponent's point of view" (it sees the active only) and placed the 950k corpus "in the input" (it is in the weights). The context-vs-weights distinction did NOT stick despite the figure; make it the first quiz item next lesson. Keep: "window = this battle; weights = every battle it learned from; only the window updates during play."
- Q2 equations: not yet produced (owner declined LaTeX; plain-text/words offered). Open.
- Q5 self-play failure: not read — Sec. 5.3 / lesson §4 assigned. Q6 transfer verdict waits on it.
- Q7 (where reconstruction errors enter training) open.
Format note: owner will not write LaTeX in chat; accept plain text or words for equation retrieval.
