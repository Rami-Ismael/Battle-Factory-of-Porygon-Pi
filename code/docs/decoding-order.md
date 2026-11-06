# Does the decoding order change legality, diversity, novelty and win rate?

Pre-registered 2026-10-30, before any experiment sampling. Script: `src/decoding_order.py`. Sampler: `DiscreteDiffusionModel.sample(order=, steps=)` in `src/discrete_diffusion_model.py`. Results: `~/vgc-data/results/decoding_order.json`.

## Question

The decoding order is the order in which the from-scratch discrete diffusion model trained only on valid Pokémon teams fills a team's masked fields (`GLOSSARY.md`). The model always uses species-first order: all six species, then abilities and items, then moves, then Stat Alignments. Under that order the species are drawn while nothing else is visible, and once a Pokémon's moves and ability appear the legality tables lock its species. On an earlier model, filling one Pokémon at a time raised win rate by 0.12 at the same diversity (2026-10-26). Which order gives the best mix of legality, diversity, novelty and win rate, and does any order make the model follow the ask more closely?

## Arms

The same checkpoint, the same asks, the same seeds:

- **Species-first order**: the current sampler, one field per pass, 48 passes.
- **Pokémon-by-Pokémon order**: each Pokémon's eight fields before the next, 48 passes.
- **Random order, one field per pass**: each team its own random order, 48 passes.
- **Random order, two fields per pass**: each team its own random order, 24 passes (half the network compute).
- **Confidence order**: at each pass, each team fills the masked field whose drawing distribution puts the most probability on one value, 48 passes.

The model was trained with random masks, so every order shows it partial teams of a kind it has seen.

## Design

- **Model and sampling:** `DiscreteDiffusionModel.load()`, classifier-free guidance weight 2, temperature 1, rule masks on, `to_paste(repair=True)`. Within a pass that fills two fields, the fields are drawn in species-first order so the legality tables see the first.
- **Asks:** 7/24, 0.3, 8/24, 0.5, 0.531 and 13/24. Three sit on the 1/24 grid that the training labels mostly sit on; the other three sit between steps.
- **Seeds:** 1, 2 and 3 at base 7300. Species-first cells redraw the remasking grid's plain-decoding teams exactly, so their battles come back from the matchup database.
- **Per cell:** draw until 64 teams pass Showdown's validator and are not held-out teams, with a cap of 1,280 tries.
- **Battles:** each team once against each of the 49 distinct top-50 meta teams; the team listed twice counts twice. Behaviour-cloning policy on both sides, Pokémon order shuffled.

## Measures, per order

- **Legality:** Showdown-accepted tries over all tries. Orders other than species-first can reach fields with no legal value, because the legality tables only check a field against the fields already visible. That shows up here.
- **Win rate:** mean over the six asks, and per ask. **Ask response:** the slope of achieved against asked win rate over the four on-grid asks (12/24 counts once), per seed; a slope of 1 would mean the model follows the ask exactly. **Gap:** mean |ask − achieved|.
- **Diversity:** distinct line-ups, species entropy (Shannon entropy, in nats, of how often each species appears) and mean slot-matched distance (the number of the 48 fields that differ between two teams after pairing their Pokémon to differ least; `src/team_diversity.py`), per cell, averaged over asks.
- **Cells with too few legal teams** (added 2026-10-30, after the smoke test, which found 1 legal team in 64 random-order tries): a cell with no legal team has no win rate, and one with a single team has no slot-matched distance. Those cells count toward legality and are left out of the other averages; the report states how many teams each order kept.
- **Size-matched diversity** (added 2026-10-30, after sampling finished; `src/decoding_order_diversity.py`): random-order cells kept 5 to 17 legal teams against 64 for the other orders, and distinct line-ups, distinct species, effective species (the exponential of species entropy), top-species share and the Vendi score (the effective number of distinct teams under a similarity measure) all depend on how many teams are counted. Each cell's values are recomputed on 200 random 5-team subsets (5 = the smallest cell) and averaged. Mean slot-matched distance and novelty are averages over pairs or teams and use every kept team. Reported without a rule.
- **Novelty**, against both networks' training teams, on the drawn tokens before the Stat Alignment repair: **copies** (share of teams identical to a training team in all 48 fields, Pokémon order ignored), **new line-ups** (share whose six species never appear together in training) and **line-up distance** (how many species separate a team from the nearest training line-up, 0 to 6).

## Comparison rule

Each order is compared with species-first order, paired by seed. A difference counts when its mean exceeds twice the larger of the two orders' seed standard deviations. An order **improves on species-first** if its mean win rate rises by more than that, its legality falls by at most 2 percentage points, and neither distinct line-ups nor species entropy falls by more than that. An order **follows the ask better** if its ask-response slope rises by more than that. Novelty is reported without a rule.

## Expectation, written before the run

Pokémon-by-Pokémon order should raise win rate, as on the earlier model. Random orders should cost legality, since moves drawn before their species are not yet constrained by a movepool. Confidence order should behave like the remasking rule: drawing the surest fields first should pull teams toward common choices and cut diversity. I expect no order to change the ask-response slope much, because the response is already squashed in the first species pick.
