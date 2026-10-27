# 09: Full run on the three grids, and the four free charts

**What to build:** After the owner approves the dry-run cost, Jev runs live through the SDK on the random, held-out and meta grids (top pick plus three sampled runs per mask); results are collected, battled and reported. The four experiments that need no extra fills then get real charts on the page: 9 Sample instead of argmax, 5 Confidence against win rate, 12 Cost per win, and 6 Jev as the judge (one Score and one 100-label Choice call per legal team; the adapter gains the Score question type). The sketch on the Jev Picks a Team page is replaced by the real chart, drawn from the run's records, with a short description of what the run found, the number of battles behind each win rate, and the cost.

**Blocked by:** 08

**Status:** ready-for-agent

- [ ] The owner approved the cost (human step).
- [ ] All three grid reports show the Jev columns, and the report notes that species samples are drawn per tournament round.
- [ ] Experiments 9, 5, 12 and 6 show real charts and descriptions on the page; no sketch remains for them.

Source: `code/docs/jev-baseline-spec.md`. Page: Jev Picks a Team (https://claude.ai/artifact/KMaYSMi5xQkxyGjNN8DTri).

**Meta grid done 2026-10-27** (owner approved the meta grid first, ~$13.5): 1,080 pairs (270 masks × 4 runs) all legal; 12,660 calls in 9 minutes, $12.54 (estimate $13.42); candidates 1,471 → 2,551; 1,792 distinct legal teams battled in 5 minutes; report `~/vgc-data/jev/meta-report.txt` and `meta-report.json`. Top pick vs the original team: −0.038 [−0.096, +0.003] over all 270 masks (bootstrap over the 5 starting teams); vs the random fill +0.177 [+0.164, +0.194]; vs Mercury −0.002 [−0.018, +0.015] on the 202 masks both filled legally. Whole Pokémon: −0.148 vs original, +0.233 vs random. Recall caveat: the state's meta list holds the starting teams, and the top pick equals the hidden original for 98% of abilities, 91% of Stat Alignments, 86% of moves, 69% of items, 39% of spreads and 11% of species. The held-out grid is the test without that shortcut.

**Held-out grid done 2026-10-27** (owner approved ~$13.4): 1,080 pairs all legal; 12,720 calls in 9 minutes, $12.51 (estimate $13.38); candidates 1,392 → 2,472; 1,845 distinct legal teams battled in 6 minutes (3 empty random pastes skipped, as before). Report `~/vgc-data/jev/heldout-report.txt` and `.json`. Top pick vs original +0.001 [−0.041, +0.047] over 270 masks; vs random +0.163 [+0.152, +0.176] (267); vs Mercury +0.003 [−0.015, +0.018] (183); vs text writer +0.030 [+0.023, +0.038] (173); vs sampled runs +0.033 [+0.020, +0.046]. Whole Pokémon: −0.043 [−0.105, +0.014] vs original, +0.241 vs random, −0.051 [−0.082, −0.011] vs Mercury (11 masks). Recall falls without the starters in the state: abilities 84%, Stat Alignment 63%, moves 78%, items 55%, spreads 23%, species questions 12% (same method as the meta figures).

**Random grid done 2026-10-27** (owner approved after topping up; switched to top pick only mid-run): 877 fills all legal (203 masks × 4 runs, then the last 67 masks top pick only); 10,547 calls, $10.46. 1,901 distinct legal teams battled in 5 minutes; report `~/vgc-data/jev/random-report.txt` and `.json`. Starting teams are random legal teams, so every arm scores low (Jev top-pick mean 0.062). All 270: top − original +0.042 [+0.019, +0.064]; − random fill +0.044 [+0.034, +0.054]; − Mercury +0.002 (141); − Ling −0.000 (186); − text writer +0.000 (149); − sampled +0.012 (203 masks). Whole Pokémon: +0.221 vs original, +0.215 vs random fill, +0.040 vs Mercury (11), +0.000 vs Ling (22). Sampled runs are no longer sent (owner, 2026-10-27).
