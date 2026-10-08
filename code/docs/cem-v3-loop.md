# Cross-entropy loop on the version-3 foundation (pre-registration, 2026-10-08)

**Question.** The full cross-entropy loop (guidance + surrogate acquisition + re-steer) climbed
0.262 → 0.411 over eleven generations on the old model (`gradloop.py`, 2026-09-01). The version-3
model (Codex regmb v3: full Regulation M-B vocabulary) one-shot stacked reaches 0.556. Does the loop
climb further from that foundation, and does it learn to use the newly reachable species?

**Win rate** = wins / battles against the top-50 meta teams, one battle per (team, top-50 column),
BC policy both sides (policy 1), all in the matchup matrix (origin `cem_v3:g<k>`). A top-50 team
scores 0.531 against the other 49.

## The loop (`src/cem_v3.py`)
Per generation g = 0 … 9:
1. **Model.** g = 0: the v3 checkpoints (F1 + G2 mixture). g ≥ 1: both members fine-tuned **from the
   original v3 checkpoints** (never from g−1) on the elites, 40 epochs, learning rate 1e-4, batch 64,
   their own schedule and legality tables, labels as the win-rate condition.
2. **Surrogate.** Ridge on field indicators + species pairs: the 44,887-label fit of the foundation
   swap, refit each generation with every loop label added at weight 10. Guidance λ = 35.4 (fixed).
3. **Propose.** 512 Showdown-valid teams: ask 0.5 at guidance 2, temperature 1, ridge-gradient
   guidance (naive), nature repair. (Low temperature, the Protect filter and the surrogate pick are
   left out of the decode: the full one-shot stack collapsed to 12 line-ups in 64.)
4. **Acquire.** Battle the 128 the surrogate ranks highest.
5. **Elites** for the next generation: every labelled team (loop teams + the 49 top-50 anchors, with
   their matrix win rates) at or above **0.531**; if fewer than 64 qualify, the top 64.

## Stopping (fixed before battles)
- Saturation: the running-best generation mean fails to improve by ≥ 0.01 for 3 generations.
- Guardrails: more than 10% of proposals copy a known team, or the distinct species sets among the 512
  proposals fall below half of generation 0's count.
  *Amended before any real battle (2026-10-08): the first draft said "fewer than 100". The smoke test
  showed the ask-0.5 decode is concentrated from the start (8 sets in 16 proposals), so an absolute
  floor could stop the loop at generation 0; a collapse guardrail has to be relative to where it starts.*
- At most 10 generations.

## Measures
Per generation: battled mean ± SE (team-clustered), p90, max, distinct species sets among proposals
and battled teams, copies, teams holding a species outside the old 189-species vocabulary and their
mean. Final: the 16 best distinct battled teams by loop label, re-battled **fresh** with 192 battles
each (`pool.score` against the top-50 schedule) — reported against their loop labels to show the
winner's curse, and against 0.531.

## Results (2026-10-08)

| gen | battled mean | p90 | max | ≥ 0.531 | species sets / 512 proposals | elites used | new-species teams |
|---|---|---|---|---|---|---|---|
| 0 | 0.427 ± 0.013 | 0.600 | 0.760 | 36/128 | 176 | — | 3 |
| 1 | **0.564 ± 0.008** | 0.680 | 0.780 | 85/128 | 161 | 63 (27 anchors, cut 0.505) | 0 |
| 2 | 0.546 ± 0.008 | 0.660 | 0.740 | 80/128 | **44** | 144 (25 anchors, cut 0.540) | 0 |

Stopped after generation 2 on the diversity guardrail (44 < 176/2). Valid 100% every generation, 0 copies.

**Fresh re-battle, 16 best distinct teams × 192 battles** (`pool.score`, top-50 schedule): mean **0.581**
against their loop labels of 0.721 (winner's curse −0.14); **14 of 16 ≥ 0.531**; best **0.667**
(Whimsicott, Chandelure, Kingambit, Floette-Eternal, Basculegion, Garchomp; label 0.740, gen 1).
For scale: the old loop's best re-battled team was 0.589 ± 0.036 (2026-09-01); the best real counter in
the counter grid is MB607 at 0.691.

Reading: one re-steer on v3 lifts the battled mean from 0.427 to 0.564 (above the one-shot stack's
0.556), and the finalists hold up fresh far better than the old loop's. But the loop converges on the
same core as the old gradient loop with Chandelure in Charizard's place: 14 of 16 finalists are
Basculegion / Chandelure / Floette-Eternal / Garchomp / Kingambit / Whimsicott. It never used a newly
reachable species after generation 0. Run note: one generated team holding an item-less Pokémon parsed
to 5 slots (`corpus.parse_team_text` merges item-less slots); it was skipped from the elites.
