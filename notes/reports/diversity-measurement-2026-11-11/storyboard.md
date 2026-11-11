# Storyboard: diversity of the default model against its training teams, 2026-11-11

Figures for the result comment on map #263 (issue #264). The reader arrives with the map's question: where does the default model, the 80-epoch labelled 84-field model at guidance 2, lose diversity against its training teams, and is each gap larger than the noise between two training draws?

Rows keep one color and one name in every figure: default model blue; win-rate-matched training, real training teams drawn so their win-rate labels match the asks, orange; any-win-rate training, real training teams drawn regardless of win rate, purple; the two anchors, one training team repeated and random legal teams, grey. Names renamed 2026-11-11 for readers new to the project.

## Teaser: two gaps on the copy-to-random scale

- Sentence: the model writes about 5 fewer builds per species than training, yet its teams sit about 12 fields further from the closest training team.
- Expected: fewer builds per species, and teams closer to training, since fewer builds suggests copying.
- Surprise: the model is narrower and further away at once. Each panel places training between one team repeated and random legal teams, so the reader sees the size of each gap against that scale.
- The nearest training distance figure was folded into this panel. Its own figure would repeat the same means.

## Frame 1: builds per species, per seed, with the noise floor

- Sentence: all three sampling seeds sit 5 builds below training matched to the asks, far outside the band two training draws differ by.
- Expected: the model below training, by more than the floor.
- Surprise: training drawn over all win rates sits between the two, so part of the matched row's variety comes from its label mix.

## Frame 2: Novelty as a stacked share of species swaps

- Sentence: the model copies a training line-up less often, 59% against 71%, and the share moves to one and two swaps, not to new line-ups.
- Expected: the model copies training line-ups more often, given it has half the distinct line-ups.
- Surprise: it copies less. Three or more swaps stays under 4% and does not clear the floor.

## Frame 3: archetype differences per seed, against the noise floor

- Sentence: Trick Room and sun teams are more common than in matched training on every seed beyond the floor; Tailwind and rain are not.
- Expected: no archetype differs.
- Surprise: two do, both upward. Rain clears the floor on two seeds only, Tailwind on one.

## Cut

- Validation teams: they match training on every measure, so they stay in the tables only.
- Nearest training distance by bins: folded into the teaser; the shares stay in the tables.
- Sand, snow and the four terrains: within the floor or near 0%, so they stay in the tables.
