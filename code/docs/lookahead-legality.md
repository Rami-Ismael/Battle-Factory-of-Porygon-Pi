# Does a lookahead sampler fix the decoding orders that fill sets before species?

Pre-registered 2026-10-31, before any experiment sampling. Script: `src/lookahead_legality.py`. Sampler: `DiscreteDiffusionModel.sample(lookahead=True)` in `src/discrete_diffusion_model.py`. Results: `~/vgc-data/results/lookahead_legality.json`.

## Question

In the decoding-order experiment (`docs/decoding-order.md`), random order made legal teams on under 1% of tries and confidence order on 63%. The cause is dead ends: the sampler checks each new field only against fields already filled, so a Pokémon's ability, item and moves can be drawn before its species in a combination no species can have. The network is already a transformer (6 layers, 8 heads); this experiment asks whether the problem is the sampler rather than the network.

The **lookahead** keeps, for every Pokémon whose species is still masked, its candidate species: the species legal with its visible fields and Species Clause. A value for that Pokémon is allowed only if at least one candidate can have it (with the visible item or ability where Showdown ties the two), and a species draw may not take the base species that another masked Pokémon has as its only one left. Under species-first and Pokémon-by-Pokémon order every species is drawn before its set, so the lookahead removes nothing: the tests show species-first sampling with it is identical bit for bit. On 64 teams at ask 0.5 it raised random order from 2 to 55 legal teams and confidence order from 40 to 62.

## Arms

The same model, asks and seeds as the decoding-order experiment, with the lookahead on:

- **Random order with lookahead**, one field per pass, 48 passes.
- **Random order with lookahead**, two fields per pass, 24 passes.
- **Confidence order with lookahead**, 48 passes.

Controls, reused from the decoding-order results: the same three orders without the lookahead, and species-first order (identical with or without it).

## Design

- **Model and sampling:** `DiscreteDiffusionModel.load()`, classifier-free guidance weight 2, temperature 1, rule masks on, `to_paste(repair=True)`.
- **Asks:** 7/24, 0.3, 8/24, 0.5, 0.531 and 13/24. **Seeds:** 1, 2 and 3 at base 7300.
- **Per cell:** draw until 64 teams pass Showdown's validator and are not held-out teams, at most 1,280 tries.
- **Battles:** each team once against each of the 49 distinct top-50 meta teams; the team listed twice counts twice. Behaviour-cloning policy on both sides, Pokémon order shuffled.

## Measures

As in the decoding-order experiment: legality (Showdown-accepted tries over all tries), win rate, ask-response slope over the four on-grid asks, mean |ask − achieved|, size-matched diversity (effective species, distinct line-ups and top-species share on random 5-team subsets, 200 per cell; mean slot-matched distance on all kept teams) and novelty (copies, new line-ups, line-up distance). Also Showdown's most common rejection reasons per arm.

## Comparison rule

Each arm is compared with the same order without the lookahead and with species-first order, paired by seed; a difference counts when its mean exceeds twice the larger of the two arms' seed standard deviations.

- The lookahead **fixes an order's legality** if its legality rises by more than that and reaches at least 90% of tries.
- An order with the lookahead **matches species-first order** if its win rate is within that bound of species-first's.

Diversity and novelty are reported without a rule.

## Expectation, written before the run

Legality should reach 85% to 97% for all three arms; the remaining failures should come mostly from the rule that a Pokémon's four moves are stored in sorted order, which the lookahead does not cover. Win rate should stay below species-first order: Pokémon-by-Pokémon order made legal teams every time and still lost 0.059, so filling sets before all six species are chosen costs strength on its own. Variety should stay above species-first order.
