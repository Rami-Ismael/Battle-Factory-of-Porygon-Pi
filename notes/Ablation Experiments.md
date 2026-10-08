---
created_at: 2026-09-24
updated_at: 2026-10-07
---
- [x] Ask the llm model what ablation experiment did you choose for example you can choose negative score function, data
	- [x] guidaince for example score function going in the negative ways
	- [x] In the blog, show that the diffusion model's performance depends on the dataset it was trained on: the same model and recipe, trained on different sets of teams, including a list of random legal teams, give different win rates. Then show how changing the guidance and the sampling parameters changed the win rate.
- [x] @Ablation Experiments.md Given the context what are ablation experiement I should be added to test my one for cross entropy method and one for Map Elite method when you added a checkbox please determine the reason why we are doing in this ablation experimenet 
- [ ]  What is the total number of placement team that is possible to trained our model from previous regulation and all previous regulation
- [ ] Count the unique real Regulation M-B tournament teams available in our project's stored data and traceable sources, using teams with recorded placements. Report both unique six-Pokémon species/form compositions and unique complete builds, ignoring team and move order; state the source coverage, deduplication rules, missing data, and exclusions rather than claiming to count every team ever used. Use the resulting usable real-team count to plan three training conditions: **real tournament teams only**, **teams from our existing [[hierarchical product sampling]] procedure only**, and **a mixture of both (proposed: 50/50)**. Use our existing hierarchical product sampling implementation and legality validation; do not confuse this training-data procedure with Cross-Entropy Method search. Test whether including real tournament teams improves performance, and whether removing them hurts performance, against **the current meta pool already stored in the project**, using our existing performance metric and evaluation protocol. Keep the number of distinct legal training builds, model configuration, training updates, and evaluation budget equal across conditions; exclude evaluation builds from training, report composition overlap, and compare matched seeds. Report the counts and propose a feasible common training-set size and experiment setup; do not launch training yet.
- [ ] For the cross entropy method we can replace the battling with a surrogate model instead let see the difference in performances
	- Why: the loop already mixes the two (the surrogate screens 512 teams, 128 get battled). Removing battles from search entirely shows whether those battles are worth their cost.
- [ ] [[Replace the scratch model with a pretrained text diffusion model finetune it to do your task]]
- [ ] [[Replace the scratch model with a pretrained text llm model finetune to do your task]]
- [ ] [[Cross Entropy Method]] ablation part of the calculuation
	- [ ] **Cross-entropy method's calculation:** two parts are untested:
		- [ ] retraining from the original model each generation (what the loop does now) versus continuing from the last generation;
			- Why: restarting means only the elite set accumulates, never the model. Continuing tests whether letting the model compound climbs faster or collapses faster.
		- [ ] how many training steps to run per generation.
			- Why: this is the method's update rate. Too many steps fit the noisy 24-battle elites exactly; too few barely move the model. It has never been swept.
- [ ] Run these cross-entropy method ablations against the full loop, with the same seeds and battle budget, and re-battle every finalist at 192 battles.
	- [ ] 1. Guide with a flipped sign, a random direction at the same strength, and a surrogate fitted on shuffled labels, at 128 teams × 24 battles per arm.
		- Why: guidance lifted win rate from 0.140 to 0.230, but the full-gradient guidance runs hinted the lift comes from loosening the model, not from the surrogate's direction. If a random direction matches, the surrogate is not steering.
	- [ ] 2. Remove one piece at a time (no guidance, random choice of which teams to battle, no retraining, no fixed elite cut), 11 generations, at least 3 seeds per arm.
		- Why: each piece's worth (+0.090 guidance, +0.055 battle choice, +0.052 retraining, +0.012 cut) comes from a different run, mostly with one seed, so the numbers can't be compared or added up.
	- [ ] 3. Drop the surrogate's pairwise terms, for both guidance and choosing which teams to battle.
		- Why: pairwise effects carry 11.6% of win-rate variance under species swaps (1.0% on Pest Control). If the surrogate without them ties, the method is not using teammate interactions.
	- [ ] 4. Mix 0%, 25% and 50% of the 692 real teams into retraining; the loop keeps about 120 now.
		- Why: by generation 11, 73% of the retraining data was the loop's own output. This tests whether real teams stop the loop from feeding on itself and collapsing.
	- [ ] 5. Label each team with 96 battles instead of 24 at the same total battles, so a quarter as many teams.
		- Why: 73% of label variance is battle noise, and a team labelled 0.79 measured 0.57 when re-battled. This tests whether noise is what stops the climb.
	- [ ] 6. After the masked retrain finishes, redo the legality-rule ablation: the old model decoded with and without the rules, and the masked-trained model with the corrected rules. Report Showdown validity and win rate.
		- Why: the old 0 valid in 80,000 came from a model trained without the mask, and old models wasted probability on impossible values (masking at evaluation alone cut the 2026-10-04 checkpoint's held-out loss from 4.96 to 3.79). The old stone rule also wrongly blocked Mega formes holding their own stones.
		- Caution: a model trained with the mask gets no training signal on illegal values, so decoding it without the rules is not a fair arm.
	- [ ] 7. Decide which of these enter the blog's ablation table.
- [ ] Run these MAP-Elites ablations against the full deep-surrogate MAP-Elites at its own protocol (6 generations × 128 teams × 24 battles, 3 seeds), verifying finalists with fresh battles.
	- [ ] 1. Remove the surrogate: battles decide which children enter the archive, at the same battle budget.
		- Why: MAP-Elites beat the cross-entropy loop by +0.084 [0.044, 0.123], but the two also differ in where they start. This isolates what the surrogate adds.
	- [ ] 2. Swap the starting points: start MAP-Elites from diffusion samples only, and the cross-entropy loop from the 200 real teams.
		- Why: MAP-Elites mutates real and already-battled teams while the loop decodes new teams from scratch, so its win may come from starting next to real teams.
	- [ ] 3. Remove the whole-Pokémon copy operator (F), keeping the move, ability, item, alignment and spread edits.
		- Why: copying Pokémon between strong teams alone reached 0.540. If this operator does the work, MAP-Elites is the copy-paste proposer with a grid on top.
	- [ ] 4. Collapse the 12 × 12 grid into one cell, so only the best predicted team survives.
		- Why: MAP-Elites covered more of the grid than the loop (0.84 vs 0.63) yet kept fewer compositions (130 vs 530 per 768 teams), so whether the offense/bulk × battle-length grid helps win rate or diversity is untested.
	- [ ] 5. Replace the three-network ensemble with the ridge surrogate.
		- Why: on held-out teams the two rank almost the same (Spearman 0.266 vs 0.250), so the deep surrogate may not earn its training cost.
	- [ ] 6. Decide which of these enter the blog's ablation table.
- [ ] We have updates the llm model training recicipe so we need to restart the traditional experiment for abaltion we need to the old model then apply to the current model 
	- [ ] grill me
	- [ ] 
# Method we have done


1. [[Cross Entropy]] Method
2. [[MAP Elite]]
3. Boltzman
4. Tree Search




# Scale Experiment

1. I want to feel like a real researcher


- [ ] Size of the model base on pretrained and non
- [ ] Size of the datasets
- [ ] Size of the metapool
- [ ] Does generating more options improve the best team, or mostly produce duplicates? Measure best-team performance, diversity, and generation cost.

# Metrics

- [ ] What are some obvious metric you can do 

- **Win rate against the meta:** re-battle every reported team at 192 battles. The 24-battle labels are winner's-cursed: a team labelled 0.79 measured 0.57 on re-battle.
- **Best team and top-16 mean**, plus **battles spent to reach the real-team mean** (0.47), which measures sample efficiency.
- [[Metrics for Diversity]]
- **Validity:** the share of generated teams that Showdown accepts.
- **Baselines on every row:** real-team mean, random search, and the LLM-prompted teams from `What is the baseline.md`.