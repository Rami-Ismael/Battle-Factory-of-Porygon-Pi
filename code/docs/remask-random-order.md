# Can remasking rescue random decoding order?

Pre-registered 2026-10-31, before any experiment sampling. Script: `src/remask_random_order.py`. Results: `~/vgc-data/results/remask_random_order.json`.

## Question

In the decoding-order experiment (`docs/decoding-order.md`), random order made legal teams on under 1% of tries. The cause is dead ends: a Pokémon's ability, item or moves get drawn before its species, and then no species fits them, so the sampler allows every species and Showdown rejects the team. Remasking masks some filled fields again before each pass and redraws them. A field redrawn after its species is filled is checked against that species, so remasking could repair a dead end. Does it raise random order's legality, and what does it do to win rate, diversity, novelty and the ask response?

## Arms

The same model, asks and seeds as the decoding-order experiment; every arm uses random order (each team its own random order).

- **Random order, no remasking** (control): the decoding-order experiment's random-order cells, reused as they are.
- **Remasking 10%, least likely fields**: before each pass after the first, each team masks again 10% of its filled fields, the ones drawn with the lowest probability (the rule of `docs/remask-ask-gap.md`). 48 passes.
- **Remasking 10%, random fields**: the same count, chosen at random among filled fields. 48 passes. The least-likely rule ranks fields by the probability they had when drawn, before their species existed, so it may never pick the fields that clash with the species; random choice does not depend on that.
- **Remasking 20%, random fields**: twice as many redraws. 48 passes.
- **Remasking 10%, random fields, 96 passes**: new fields spread over 96 passes, so each team gets twice as many remasking rounds.

## Design

- **Model and sampling:** `DiscreteDiffusionModel.load()`, classifier-free guidance weight 2, temperature 1, legal Pokémon construction filter on, `to_paste(repair=True)`.
- **Asks:** 7/24, 0.3, 8/24, 0.5, 0.531 and 13/24. **Seeds:** 1, 2 and 3 at base 7300, as in the decoding-order experiment.
- **Per cell:** draw until 64 teams pass Showdown's validator and are not held-out teams, at most 1,280 tries.
- **Battles:** each team once against each of the 49 distinct top-50 meta teams; the team listed twice counts twice. Behaviour-cloning policy on both sides, Pokémon order shuffled.

**Fixed before the run (2026-10-31, from a GPT-6 code review):** when a remasking pass reaches a dead end, the team gets its pre-pass fields back and redraws only its new field. That redraw used the network's predictions from the pass's input, where the remasked fields were still masked; it now uses fresh predictions on the restored team (one extra network pass for those teams). The smoke test before the fix is kept in `~/vgc-data/remask_random_order/smoke_before_fix_2026-10-31.json`.

## Measures

The decoding-order experiment's measures, computed the same way: legality (Showdown-accepted tries over all tries), win rate, ask-response slope over the four on-grid asks, mean |ask − achieved|, size-matched diversity (effective species, distinct line-ups and top-species share on random 5-team subsets, 200 per cell; mean slot-matched distance on all kept teams) and novelty (copies, new line-ups, line-up distance). Also per arm: teams whose remasking pass was undone after a dead end, and redraws per field.

## Comparison rule

Each arm is compared with random order without remasking, paired by seed; a difference counts when its mean exceeds twice the larger of the two arms' seed standard deviations. Remasking **rescues random order** if legality rises by more than that and reaches at least 50% of tries. Win rate, diversity and novelty are reported against both random order without remasking and species-first order (from the decoding-order results) without a rule.

## Expectation, written before the run

The least-likely rule should barely help, because it ranks fields by stale probabilities. Random-field remasking should raise legality, more at 20% and at 96 passes, but I expect none to reach 50%: the species itself was drawn when every species was allowed, and a redraw of the species still meets the clashing fields. Win rate should stay below species-first order.
