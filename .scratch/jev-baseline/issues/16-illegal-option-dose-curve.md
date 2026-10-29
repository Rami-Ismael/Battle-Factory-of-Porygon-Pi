# 16: Experiment 1: Illegal-option dose curve

**What to build:** Add illegal options dosed per category (owner, 2026-10-27). Moves and abilities: pad to 0, 25, 50 and 100% of the legal list, capped at 255. Items (teammates' items under the Item Clause for most species; every other item for a Mega form) and species (families on the team or unfinishable slots): two points, none and all. Pools are every value of the field minus the legal ones (dose-pools.json). Stat Alignment has no illegal option and Stat Points spreads would be invented, so both are left out. Fill, battle the legal teams, and plot legality per category against the number of illegal options offered. The sketch on the Jev Picks a Team page is replaced by the real chart, drawn from the run's records, with a short description of what the run found, the number of battles behind each win rate, and the cost.

**Blocked by:** 09

**Pre-registration (owner, 2026-10-27):**
- Length control: keep the list at its 0% length and swap the same share of legal options for illegal ones (legal padding is impossible: every legal option is already offered).
- Position control: added options go to random positions, reshuffled per question.
- Sample: at least 150 questions per category per dose, from every hidden field on the three grids; each question asked at every dose (paired).
- Chance line: the illegal share of the list.
- Pass: the upper end of the 95% interval on the illegal-pick share is below half the chance line at every dose. Fail: the lower end is at or above the chance line. Otherwise inconclusive.
- Dose effect: a paired drop in legality of at least 10 points, McNemar p < 0.05.
- Win rate: only on questions legal at both doses (the same team at each dose).

**Status:** ready-for-agent

- [ ] Verify before the paid run: a uniform fake chooser's illegal-pick share lands on the chance line, which calibrates the measurement.
- [ ] The dry run's cost is shown to the owner before any call (human step).
- [ ] The page shows the real chart and description for this experiment.
- [ ] Every new team is checked by the Showdown validator before battle.

Source: `code/docs/jev-baseline-spec.md`. Page: Jev Picks a Team (https://claude.ai/artifact/KMaYSMi5xQkxyGjNN8DTri).
