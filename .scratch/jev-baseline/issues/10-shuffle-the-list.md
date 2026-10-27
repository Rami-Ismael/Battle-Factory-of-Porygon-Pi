# 10: Experiment 2: Shuffle the list

**What to build:** Ask every move question in a fixed alphabetical order and in a shuffled order; record the picked position and how often the pick changes; battle only the masks whose picks differ. The sketch on the Jev Picks a Team page is replaced by the real chart, drawn from the run's records, with a short description of what the run found, the number of battles behind each win rate, and the cost.

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] Verify before the paid run: a fake chooser that picks by position changes its pick under shuffling almost every time, and one that picks by name never does.
- [ ] The dry run's cost is shown to the owner before any call (human step).
- [ ] The page shows the real chart and description for this experiment.
- [ ] Every new team is checked by the Showdown validator before battle.

Source: `code/docs/jev-baseline-spec.md`. Page: Jev Picks a Team (https://claude.ai/artifact/KMaYSMi5xQkxyGjNN8DTri).

**Built 2026-10-27:** `code/scripts/jev_shuffle.py` (tests in `test_jev_shuffle.py`). It replays every move question of the main run's top-pick fills exactly (a replay chooser feeds back the recorded answers; every rebuilt team must equal the recorded one), then asks each once more in one request with two questions: alphabetical (the main run's order, so main-run pick against it is Jev's noise floor) and shuffled (seeded per question). Differing shuffled picks are swapped into the top-pick team and battled.

**Verify, no calls (2026-10-27):** on 300 replayed questions per grid, every top-pick team rebuilt exactly; shuffled lists keep the same moves (300/300), are never alphabetical, and differ per question (300 distinct orders). A first-option fake changes its pick 98% / 98% / 99% (chance of keeping it ≈ 2%); a by-name fake 0%. Dry run: 1,920 move questions per grid, 5,760 requests, ≈ $7.29 (≈ $2.43 per grid).

**Run 2026-10-27** (owner approved ~$7.29, all three grids): 5,542 requests, $6.52. Held-out and meta complete (1,920 questions each); random 1,702 of 1,920, because the TypeSafe account ran out of credits (402) for the last 218 (≈ $0.27 to finish). Pick changes when shuffled vs the same order asked again (Jev's noise), 95% intervals over the five unmasked teams: held-out 12.1% [8.6, 16.0] vs 4.6% [2.5, 6.7]; meta 11.4% vs 5.1%; random 27.4% vs 12.0%. Just over half the held-out changes are a move Jev also picked for another slot (same four moves, new order); a different move 5.6% vs 2.8% noise. No pull toward the top: first option 2.9% shuffled vs 1.9% chance. Swapped teams minus top-pick teams: held-out −0.004 [−0.021, +0.019] (146 battled), meta +0.017 [+0.003, +0.027] (143), random +0.002 (359); swaps that repeated a move on one Pokémon are illegal and were not battled. Chart on the page (version 36). Fix: the adapter now raises `OutOfCredits` on a 402 instead of retrying it as a transport error.
