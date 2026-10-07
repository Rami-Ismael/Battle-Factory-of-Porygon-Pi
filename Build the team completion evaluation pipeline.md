# Team completion evaluation pipeline

The goal is to compare how well an LLM and a diffusion model fill missing parts of a Pokémon team to counter a fixed set of meta teams. Both methods receive the same starting teams, missing parts, regulation, and meta information.

## 1. Create the starting-team list

Choose either randomly generated legal Pokémon teams or a sample of 50 high-placing teams. Treat these as separate experiment groups if we use both.

For the high-placing sample, first remove teams with incomplete set information. Group the remaining teams by tournament and order the tournaments from newest to oldest within the chosen regulation. Select the highest-placing eligible team from each tournament, then the second-highest-placing eligible team from each tournament, and continue until we have 50 teams. If fewer than 50 eligible teams exist, record the available number.

Save each team's full set information, source, tournament date, and placement where available. Select and save the meta teams that the completed teams will face. Keep the evaluation meta separate from the teams used to train or tune the methods.

## 2. Create the masked tasks

Make a copy of each starting team and mark the selected parts as `null`. Save the task type, value of k, random seed, selected slots, and masked team. Use exactly the same masked examples for both methods. Repeat with multiple random masks to reduce dependence on one selection.

| Task | What we leave empty | Values of k |
| --- | --- | --- |
| Stat spreads | The full stat spread of k randomly selected Pokémon | 1–6 |
| Items | The items of k randomly selected Pokémon | 1–6 |
| Moves | k randomly selected move slots across the team | 1–24 |
| Whole Pokémon | k randomly selected Pokémon slots, including their complete sets | 1–6 |
| Abilities, optional | The abilities of k randomly selected Pokémon | 1–6 |
| Natures, optional | The natures of k randomly selected Pokémon | 1–6 |

Tera Types are excluded because the chosen regulation does not support Terastallization.

Define “stat spread” consistently before running the experiment. If it means EVs, mask the full EV spread for each selected Pokémon and keep its nature and IVs fixed. For whole-Pokémon tasks, mask the identity and every set field together.

Everything outside the mask stays fixed. Keep the original complete team for comparison, but do not provide the hidden values to either method.

## 3. Ask each method to complete the team

For the LLM, provide:

1. Instructions to fill only the missing parts, follow the regulation, and counter the supplied meta teams.
2. The masked team and the meta teams with their available full sets.
3. The required JSON output structure for the completed team.

Give the diffusion model the equivalent information through its supported conditioning interface. Record any differences in what the models can receive so the comparison is clear.

Save the output, model version, generation settings, elapsed time, and task identifier. Set the number of candidates and allowed retries in advance, using the same limits for both methods. If we generate several candidates, select between them using a separate development evaluation; reserve the final evaluation battles for measuring performance.

## 4. Validate the completed team

Check that the output follows the required structure, fills every missing part, leaves all unmasked parts unchanged, and passes the chosen regulation's team-legality validator.

Record invalid outputs and their reasons. If retries are allowed, return the validation error to the method and retry within the predefined limit. Record tasks that remain invalid as failures rather than silently removing them from the comparison.

## 5. Check the matchup database

For each valid completed team and each meta opponent, check whether matching battle results already exist.

A reusable result must match the complete sets of both teams, the regulation, simulator version, battle policies and their versions, battle settings, and the required seed schedule. Team names alone are not enough to identify a matchup. Preserve team order unless its irrelevance has been established.

Reuse matching results. If only some required battles exist, run the missing battles rather than repeating the entire matchup.

## 6. Run and store missing battle experiments

Use the same battle policies, number of battles, and seed schedule for both methods. Keep these settings fixed across comparisons. The team-building method chooses the team; the fixed battle policy controls it during evaluation.

Store each battle's team identifiers, opponent identifier, settings, seed, outcome, and replay or log reference. Record simulation errors separately from wins, losses, and draws.

Evaluate the original unmasked starting team under the same conditions as a reference point. Reuse its cached results when the conditions match.

## 7. Compare the methods

Report performance separately for each task and each value of k:

- Win rate against each meta team and the overall win rate across the fixed meta pool.
- Improvement over the original starting team's win rate.
- Valid completion rate and number of retries.
- Generation time and model cost, where measurable.
- Number of new battles required and number of cached results reused.

Use equal opponent weights unless meta-frequency weights were selected before the experiment. Report draws separately and define their treatment before calculating an aggregate score. Compare the methods on paired tasks with the same masks, and include uncertainty across starting teams and masks. Report final generation failures alongside battle performance so a method is not rewarded for producing fewer valid teams.

## Pipeline flow

Starting teams and fixed meta teams → shared random masks → LLM or diffusion completion → output and legality validation → matchup database lookup → missing battle experiments → stored results → comparison by task and k.

Invalid completions → bounded retry → validation again, or recorded failure.
