# 19: Experiment 14: The duplicate trap

**What to build:** On the item masks and whole-Pokémon masks, add to each question's options exactly the items teammates already hold (item questions) or the species already on the team (species questions), and nothing else illegal. Ask each question twice: once with the rules line "Species Clause, Item Clause" in the state and once without it. Report how often Jev picks a duplicate, per field and per prompt version. No battles: a duplicate makes the team illegal. The sketch on the Jev Picks a Team page is replaced by the real chart, with a short description, the call count and the cost.

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] Verify before the paid run: on 20 questions, the added options are exactly the duplicates and nothing else illegal.
- [ ] The dry run's cost is shown to the owner before any call (human step).
- [ ] The duplicate options are added only in this experiment; the main run's option builder is unchanged.
- [ ] The page shows the real duplicate rate for items and species, with and without the rules line.

Source: `code/docs/jev-baseline-spec.md`. Page: Jev Picks a Team (https://claude.ai/artifact/KMaYSMi5xQkxyGjNN8DTri).
