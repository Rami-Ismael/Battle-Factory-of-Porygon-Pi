
1. We are planning to use two surrogate model to determine how often does this pokemon team beat this pokemon in n games hoping to not run n pokemon games
2. We are planning to create two surrogate model 
	1. The first will be about determining how often does this pokemon team beat this pokemon in n games hoping to not run a pokemon n games base on the battle policy we havev
	2. The second will be about determining the win rate of pokemon team against the meta teams

# Todo

- [ ] Test whether the win-rate surrogate correctly predicts the effect of changing one Pokémon's set. Create held-out original/edited team pairs that remain legal; compare the surrogate's predicted win-rate change with fresh battles using fixed policies, the same opponents, and equal battle budgets. Plot predicted change on the x-axis and measured change on the y-axis, with an agreement diagonal and battle uncertainty intervals; illustrate one pair with the exact edit. Check whether it predicts improvements and regressions reliably enough to guide team search.
- [ ] Trained a surrogate model to determine pokemo team wining rate