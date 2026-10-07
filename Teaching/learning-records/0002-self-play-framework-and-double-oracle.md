---
created_at: 2026-08-19
status: active
---

# Self-play framework and Double Oracle — first exposure via Lesson 0002

Context: the owner pasted a secondary summary of the survey's ORACLE concept (with "Oracle Function", missing the regret-minimisation ABR) and asked what BR / ABR are and where Double Oracle is classified. Chat answers plus Lesson 0002 (interactive DO stepper on RPS and a 4-strategy game; 6-question quiz) cover: the five-symbol loop, BR vs ABR vs specially crafted, DO's Table 1 row, DO's Theorem 1 needing exact BR, and the finding that VGC-Bench's "double oracle" arm is PSRO-Nash.

Key insight to retain: the convergence guarantee lives in the oracle, not the MSS; with PPO the arm's claims are empirical only.

Not yet known: quiz results (first evidence of retention); whether the owner wants the mission widened to the battle-policy strand. Likely next in ZPD: why fictitious play's Σ is constant; what Algo. 2's buffer is; how BC sits outside self-play; the survey's "ongoing-training" E>1 regime (which VGC-Bench actually resembles).

Update 2026-08-19: owner conflated "pure strategy" (one fixed row/checkpoint) with "pure self-play" (no frozen model). Clarified in chat; distinction added to the glossary sheet. Retest this in the next quiz.
