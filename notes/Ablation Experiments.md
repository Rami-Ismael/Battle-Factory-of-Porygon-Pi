---
created_at: 2026-09-24
updated_at: 2026-10-17
---
- [x] Ask the LLM which ablation experiments to run, for example a negative score function or a different training dataset.
	- [x] Test guidance with the score function pointing the negative way.
	- [x] In the blog, show that the diffusion model's performance depends on its training data. Train the same model with the same recipe on different sets of teams, including a set of random legal teams, and compare their win rates. Then show how changing the guidance and the sampling parameters changes the win rate.
- [x] List the ablation experiments that test my methods: one set for the cross-entropy method and one for MAP-Elites. Give every checkbox the reason for its ablation.
- [ ] Count the placement teams that could train our model from the previous regulation, and from all earlier regulations together.
- [ ] Count the unique real Regulation M-B tournament teams with recorded placements in our stored data and traceable sources. Then plan a training-data experiment from that count.
	- [ ] 1. Report the unique six-Pokémon species and form compositions, and the unique complete builds. Ignore team order and move order.
	- [ ] 2. State the source coverage, the deduplication rules, the missing data, and the exclusions. Do not claim that the count covers every team ever used.
	- [ ] 3. Plan three training conditions: real tournament teams only, teams from our [[hierarchical product sampling]] procedure only, and a 50/50 mixture of both. Use our existing hierarchical product sampling code and legality validation. This is a training-data procedure, not the cross-entropy method search.
	- [ ] 4. Test whether adding real tournament teams improves performance, and whether removing them hurts it. Evaluate against the meta pool already stored in the project, with our existing metric and evaluation protocol.
	- [ ] 5. Keep four things equal across the conditions: the number of distinct legal training builds, the model configuration, the training updates, and the evaluation budget. Exclude evaluation builds from training, report the composition overlap, and compare matched seeds.
	- [ ] 6. Report the counts and propose a common training-set size and setup. Do not launch training yet.
- [ ] In the cross-entropy method, replace battles with surrogate predictions during search, and compare its performance with the battle-scored version.
	- Why: the loop already mixes the two. The surrogate screens 512 teams, and 128 of them get battled. Removing battles from the search entirely shows whether those battles are worth their cost.
- [ ] [[Replace the scratch model with a pretrained text diffusion model finetune it to do your task]]
- [ ] [[Replace the scratch model with a pretrained text llm model finetune to do your task]]
- [ ] Ablate the parts of the update step in the [[Cross Entropy Method]].
	- [ ] Test the two parts of the update that no run has tested yet.
		- [ ] Compare retraining from the original model each generation, as the loop does now, with continuing from the last generation.
			- Why: restarting means only the elite set accumulates, never the model. Continuing tests whether a model that compounds climbs faster or collapses faster.
		- [ ] Sweep the number of training steps per generation.
			- Why: the step count is the method's update rate. Too many steps fit the noisy 24-battle elites exactly. Too few steps barely move the model. No run has swept it.
- [ ] Run these cross-entropy method ablations against the full loop with the same seeds and battle budget. Re-battle every finalist at 192 battles.
	- [ ] 1. Guide with a flipped sign, with a random direction at the same strength, and with a surrogate fitted on shuffled labels. Use 128 teams × 24 battles per arm.
		- Why: guidance lifted win rate from 0.140 to 0.230. The full-gradient guidance runs hinted that the lift comes from loosening the model, not from the surrogate's direction. If a random direction matches, the surrogate is not steering.
	- [ ] 2. Remove one piece at a time: guidance, the surrogate's choice of which teams to battle, retraining, and the fixed elite cut. Run 11 generations and at least 3 seeds per arm.
		- Why: each piece's worth (+0.090 guidance, +0.055 battle choice, +0.052 retraining, +0.012 cut) comes from a different run, mostly with one seed. The numbers cannot be compared or added up.
	- [ ] 3. Drop the surrogate's pairwise terms, both for guidance and for choosing which teams to battle.
		- Why: pairwise effects carry 11.6% of win-rate variance under species swaps, against 1.0% on Pest Control. If the surrogate without them ties, the method does not use teammate interactions.
	- [ ] 4. Mix 0%, 25%, and 50% of the 692 real teams into retraining. The loop keeps about 120 real teams now.
		- Why: by generation 11, 73% of the retraining data was the loop's own output. This tests whether real teams stop the loop from feeding on itself and collapsing.
	- [ ] 5. Label each team with 96 battles instead of 24, at the same total battle count, so the run labels a quarter as many teams.
		- Why: 73% of label variance is battle noise, and a team labelled 0.79 measured 0.57 when re-battled. This tests whether noise is what stops the climb.
	- [ ] 6. After the masked retrain finishes, redo the legality-rule ablation. Decode the old model with and without the rules, and decode the masked-trained model with the corrected rules. Report Showdown validity and win rate.
		- Why: the old result of 0 valid teams in 80,000 came from a model trained without the mask. Old models wasted probability on impossible values. Masking at evaluation alone cut the 2026-10-04 checkpoint's held-out loss from 4.96 to 3.79. The old stone rule also wrongly blocked Mega formes that hold their own stones.
		- Caution: a model trained with the mask gets no training signal on illegal values, so decoding it without the rules is not a fair arm.
	- [ ] 7. Decide which of these ablations go into the blog's ablation table.
- [ ] Run these MAP-Elites ablations against the full deep-surrogate MAP-Elites at its own protocol: 6 generations × 128 teams × 24 battles, 3 seeds. Verify the finalists with fresh battles.
	- [ ] 1. Remove the surrogate. Battles decide which children enter the archive, at the same battle budget.
		- Why: MAP-Elites beat the cross-entropy loop by +0.084 [0.044, 0.123], but the two methods also start from different teams. This arm isolates what the surrogate adds.
	- [ ] 2. Swap the starting points. Start MAP-Elites from diffusion samples only, and start the cross-entropy loop from the 200 real teams.
		- Why: MAP-Elites mutates real and already-battled teams, while the loop decodes new teams from scratch. Its win may come from starting next to real teams.
	- [ ] 3. Remove the whole-Pokémon copy operator (F). Keep the move, ability, item, alignment, and spread edits.
		- Why: copying Pokémon between strong teams alone reached 0.540. If this operator does the work, MAP-Elites is the copy-paste proposer with a grid on top.
	- [ ] 4. Collapse the 12 × 12 grid into one cell, so only the best predicted team survives.
		- Why: MAP-Elites covered more of the grid than the loop (0.84 against 0.63) but kept fewer compositions (130 against 530 per 768 teams). No run has tested whether the grid of offense-to-bulk ratio × battle length helps win rate or diversity.
	- [ ] 5. Replace the three-network ensemble with the ridge surrogate.
		- Why: on held-out teams the two rank almost the same (Spearman 0.266 against 0.250), so the deep surrogate may not earn its training cost.
	- [ ] 6. Decide which of these ablations go into the blog's ablation table.
- [ ] The LLM's training recipe changed, so rerun the earlier ablation experiments: first on the old model, then on the current model.
	- [ ] Grill me on this plan.
- [ ] Retrain the diffusion model with condition dropout (`p_uncond`) at 0.05, 0.15, 0.3, and 0.5. Compare the asked-vs-got curves and the win rates.
	- Why: every model so far uses 0.15, and no run has swept it. It sets how strongly the model learns the ask, and the ask barely steers now.
- [ ] Run the [[Diffusion-BBO Diffusion-Based Inverse Modeling for Online Black-Box Optimization|Diffusion-BBO]] port as a 2 × 2 ablation of the cross-entropy loop. The first factor is the training data: elites only, as now, or all labelled teams conditioned on their win rate. The second factor is the ask: fixed at 0.5, as now, or w × the re-measured best, with w ∈ {0.6, 0.7, 0.8, 0.9, 1.0} picked by the surrogate. Use the same seeds and battle budget as the clean re-run of the cross-entropy loop (0.504), 3 seeds per arm, and re-battle finalists at 192 battles.
	- Why: inside the elites-only loop the ask does nothing (got-vs-ask correlation −0.03, 2026-10-12). The paper also trains on low scores, so the ask carries information there. The test is the interaction: the ask should matter only in the all-data arms. On the old model, all-data conditioning (0.195) tied elite refitting (0.187) on 2026-08-27, and no run has re-tested it since.
	- Source: [Diffusion-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization](https://www.alphaxiv.org/abs/2407.00610) 🟢. Algorithm 1 retrains on the whole dataset each round. Eq. 4 is classifier-free guidance. §6 sets the ask to w × best with W = {0.6, …, 1.0}. The −0.03, 0.195, and 0.187 come from this project's runs, not from the paper.
	- [ ] 1. In the all-data arm, condition on the raw 24-battle win rate, and then on a win rate shrunk toward the mean.
		- Why: the paper's scores are noise-free. Ours are not. A team labelled 0.79 measured 0.57 when re-battled, so the best-so-far value that sets the ask is inflated.
		- Source: the same paper, Appendix D.1. The DNA tasks take their scores from a complete lookup table, and the authors dropped the ChEMBL task because its evaluator gave different scores for the same design. The 0.79 → 0.57 comes from this project's gradient-guidance re-battle (2026-09-01).
	- [ ] 2. Replace the surrogate-chosen w with a fixed w = 1.0, and then with w = 1.1.
		- Why: the paper found that asking above the best does not help, and our asked-vs-got curve fell from 0.32 to 0.21 above the training maximum. This tests whether choosing w beats always asking for the best.
		- Source: the same paper. §6 leaves out w > 1 as outside the training data, and in §6.4 a fixed w, including w > 1, loses to the chosen w. The 0.32 → 0.21 comes from this project's asked-vs-got run (2026-10-04).
	- [ ] 3. Decide which of these ablations go into the blog's ablation table.
- [ ] Run these ablations of the fixed-ask loop against that loop. The loop uses ask 0.5, elite re-steer, the taboo list, and sets-only temperature 0.5. Battle all 512 proposals each generation, use 3 seeds (6060, 7070, 8080) and generations 1–3, and re-battle finalists at 192 battles. Pre-registration: `code/docs/cem-ablations.md`.
	- Why: every earlier loop arm ran one search seed and battled only the ridge's top 128. None of them shows which piece of the loop does the work.
	- [ ] 1. Remove the ask. Train the elites with the label hidden, and decode every proposal unconditionally.
		- Why: got-vs-ask spanned zero in every generation of every loop arm. If this arm ties, the ask is not part of the method.
	- [ ] 2. Freeze the generator. The original model proposes every generation, and the ridge still refits and tilts.
		- Why: the 2026-10-16 diagnosis found that the ridge does the steering. This arm measures what retraining adds on its own.
	- [ ] 3. Set the elite cut at 0.50 and at 0.56 instead of 0.531.
		- Why: the elite fraction was the biggest lever in August (copy-paste 0.479 → 0.540), and 0.531 was set as the meta mean, never tuned.
	- [ ] 4. Fine-tune for 10 epochs instead of 40, and drop the top-50 anchors from the elites. Change one thing per arm.
		- Why: the top-20% arm changed the elite rule, the anchors, and the epochs at once, so its null result cannot be pinned on any one of them.
	- [ ] 5. Remove the ridge tilt from the decode.
		- Why: only one seed has tested it, and the 2026-10-16 diagnosis credits the ridge with the steering.
	- [ ] 6. Remove the taboo list from generation 1 on.
		- Why: the taboo list is the project's diversity lever, and its win-rate cost was measured on the old sampler with one seed.
	- [ ] 7. Decode every non-species field at temperature 1 instead of 0.5.
		- Why: sets-only temperature 0.5 was the only near-signal (+0.035 [−0.002, +0.072]), and it rests on one seed.
	- [ ] 8. Fine-tune each generation from the previous generation's model, not the original one.
		- Why: the loop never lets the model compound. Only the elite set accumulates.
	- [ ] 9. Decide which of these ablations go into the blog's ablation table.
- [ ] Fine-tune the model on three label sets and compare their asked-vs-got curves: precise labels only (teams with 192 or more battles), the same teams with 49-battle labels, and every labelled team. Pre-registration: `code/docs/label-precision.md`.
	- Why: the model fine-tuned on 1,105 re-battled teams reached 0.593 at ask 0.5, against 0.496 for the untouched model. Either precise labels or the fine-tune itself could explain the gap. Only the same-teams comparison separates them.
	- [ ] 1. Fine-tune on the 1,105 re-battled teams with their original labels, and with a random 49-battle redraw. Compare both with the re-battled-label model.
		- Why: the original labels chose these teams, so they carry selection bias as well as noise. 13% of them sit at 0.7 or above, against 1.4% after re-battling. The first test redrew labels at random, so it measured noise only.
	- [ ] 2. Fine-tune on three sets of 1,110 teams from the same pool: flat win rates, the pool's own win rates, and elites only (0.531 or above).
		- Why: re-battling changed nothing, but models trained on the evenly spread 1,105 teams reached about 0.58, against 0.55–0.57 for the 2,144-team set. The spread of win rates may be what matters, and the loop trains on elites only.
	- [ ] 3. Decide whether precise labels are worth the cost of re-battling teams.
# Method we have done


1. [[Cross Entropy]] Method
2. [[MAP Elite]]
3. Boltzman
4. Tree Search




# Scale Experiment

1. I want to feel like a real researcher


- [ ] Compare model sizes, with and without pretraining.
- [ ] Compare training-set sizes.
- [ ] Compare meta pool sizes.
- [ ] Test whether generating more teams improves the best team or mostly adds duplicates. Measure best-team performance, diversity, and generation cost.

# Metrics

- [ ] List the obvious metrics to report for every ablation.

- **Win rate against the meta:** re-battle every reported team at 192 battles. The 24-battle labels are winner's-cursed: a team labelled 0.79 measured 0.57 on re-battle.
- **Best team and top-16 mean**, plus **battles spent to reach the real-team mean** (0.47), which measures sample efficiency.
- [[Metrics for Diversity]]
- **Validity:** the share of generated teams that Showdown accepts.
- **Baselines on every row:** real-team mean, random search, and the LLM-prompted teams from `What is the baseline.md`.