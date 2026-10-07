---
created_at: 2026-08-26
updated_at: 2026-08-26
type: learning-record
lesson: 0009-map-elites-two-gates-and-vgc-measures.html
---

# 0011 — MAP-Elites two-gates re-teach + VGC measure menu

Taught 2026-08-26, after the owner returned to QD presenting himself as a first-time learner via the Hearthstone paper — the second time the archive insert rule has failed to land (first: the stopped grill of 2026-08-20). Lesson 0009 delivers what that day's note promised: an interactive archive widget (`assets/archive-sim.js`, the two gates) instead of prose questions.

Corrections that landed in chat and the lesson:

- **Attack/defense/balance is an output, not an input.** MAP-Elites never takes styles; it takes numeric measures computed from simulated battles, and the archetype spectrum re-emerges as grid regions (Mouret & Clune 2015 abstract: "a map of high-performing solutions at each point in a space defined by dimensions of variation that a user gets to choose").
- **His N×N-matrix claim was structurally right** — and battle length is literally one of DSA-ME's two measures (5–15 turns); the other was hand size (1–7), which has no VGC analogue.
- **The insert rule stated precisely:** a child competes only inside the single cell its own measures place it in. Gate 1 = WHERE (measures); Gate 2 = WHETHER (objective vs at most one incumbent). Empty niche admits unconditionally — so a child worse than its parent can enter. This remains the load-bearing sentence.
- **Recommended VGC measure pair:** mean battle length × KO spread (# of the four battlers with ≥1 KO) — decorrelated, log-readable. His attack–defense idea is operationalizable as damage dealt ÷ damage taken but correlates with battle length (fast sweeps are short), so pairing them collapses the grid onto a diagonal.
- **Sharpened his diversity claim:** diversity alone is worthless (random teams are maximally diverse); QD is diversity under a per-niche quality gate. Bonus cited from Mouret & Clune: MAP-Elites also tends to find a better single solution than SOTA search.

Verification notes: Mouret & Clune arXiv 1504.04909 fetched live before citing (title/abstract confirmed, 🟢). DSA-ME numbers reused from the verified 2026-08-19 deep-read (lesson 0004), not re-derived. Quiz options equal-length per skill rule.

Errata 2026-08-26 (later session): recommended pair revised to battle length × static offense/bulk ratio; KO spread demoted (leaks the objective, doubles attribution bias). Aggression ratio × length is a funnel, not a diagonal — still rejected. Lesson 0009 §5 not yet updated.
