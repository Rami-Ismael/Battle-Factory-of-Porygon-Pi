---
created_at: 2026-10-20
updated_at: 2026-11-11
tags:
  - vocabulary
---
# Glossary

One meaning per word in this project.

## Language

**Elite cut**:
The win rate a battled team must reach to be kept as training data for the next round: 0.531, the mean win rate of a top-50 meta team against the other 49.
_Avoid_: threshold, cutoff

**Elite**:
A battled team kept to train the next round. Which teams qualify depends on the run's elite rule.
_Avoid_: winner, survivor

**Cumulative elite rule**:
Elites are the battled teams from every earlier generation, plus the anchor teams, that clear the elite cut, topped up to the elite floor.

**Top-fraction elite rule**:
Elites are the best elite fraction of the previous generation's battled teams, ranked by win rate. No anchor teams, nothing carried over.

**Elite floor**:
The smallest number of elites kept under the cumulative elite rule, 64. When fewer teams clear the elite cut, the best 64 are kept anyway, so some elites can sit below 0.531.
_Avoid_: fallback cut

**Elite fraction**:
The share of one generation's teams, ranked by win rate, that become elites, such as the top 20%. It is a rank rule, unlike the fixed elite cut.
_Avoid_: top x, elite cut (for this meaning)

**Anchor team**:
A top-50 meta team, with its measured win rate, that the loop can keep as an elite, so search-found teams always compete with real ones.
_Avoid_: anchor (alone), reference team

**Keeper team**:
A search-found team that won at least 0.65 of at least 784 fresh battles against the top-50 meta teams.
_Avoid_: good team, winner

**Candidate**:
A whole team the candidate proposer outputs.
_Avoid_: proposal, candidate solution, candidate team

**Candidate proposer**:
The part of the search that outputs candidates. Its job is that every candidate is legal.
_Avoid_: generator (alone)

**Blank candidate**:
A candidate with every field masked. Diffusion sampling starts here.

**Partial candidate**:
A candidate with some fields filled and the rest masked. It occurs during sampling, or when an edit masks some fields of a real team.

**Complete candidate**:
A candidate with every field filled, not yet checked against the rules.

**Legal candidate**:
A complete candidate that Showdown's validator accepts for Reg M-B.
_Avoid_: valid

**Validator**:
Showdown's team validator for Regulation M-B, the regulation this project plays. It judges only complete teams, legal or not, with its reasons; choices during sampling are limited by the model's legal Pokémon construction tables instead.
_Avoid_: legality check (for anything before a team is complete)

**Validator redraw**:
Showdown's validator checks every generated team and any it rejects is drawn again, so every battled team is legal. The model's target is first-try legality above 50%, averaged over a grid's masks, not 100% (set 2026-11-07).
_Avoid_: 100% legal model

**Legal Pokémon construction tables**:
How a Pokémon may be built in Regulation M-B, as the model's lookup tables: each species' abilities, items and moves, plus Species Clause and Item Clause. For a masked field they list the values the filled fields still allow. Stricter than the validator on Mega Stones (#258).
_Avoid_: rule tables, legality tables

**Legal Pokémon construction filter**:
Before each draw, the sampler gives zero odds to every value the legal Pokémon construction tables forbid next to the fields already on the team: abilities, items and moves the species can't have, Species Clause, Item Clause, the Stat Points budget. The loss applies it too, as a logit of −10,000.
_Avoid_: rule mask, rule masks on, rule-table filter

**Battled candidate**:
A legal candidate with a measured win rate.

**Unlabelled team**:
A legal training team without a win rate, because it has not battled enough of the meta; it must be battled against the meta to get one. It arises when a run adds the teams it drew or searched to the training table before battling each one against the meta.
_Avoid_: unscored team, an illegal team (the validator rejects it, so it can never be battled)

**Unmasked team**:
A complete team before a mask hides any field; each Mask Lab grid has five, and every mask is cut from one. Battled unchanged, it is what every fill of its masks is compared with. Its grid says what kind it is: random (random legal HPS teams nobody built), held-out (tournament teams placed 51 to 100) or meta (top-50 meta teams).
_Avoid_: starting team, original team, starter, source team

**Team**:
Six Pokémon. Species, item, ability, moves, Stat Points and Stat Alignment are all decision variables.
_Avoid_: roster

**Species replacement distance**:
The minimum number of species replacements needed to turn one six-species line-up into another, ignoring Pokémon order and sets. It equals six minus the number of shared species. Species forms remain distinct.

**Connected one-swap family**:
A connected group within the collection being split. Teams belong to the same family when linked, directly or through other teams, by a species replacement distance of at most one. Two teams in the same family can differ by more than one replacement when connected through a chain.
_Distinguish_: an n-swap family measured from one reference line-up.

**Pokémon**:
One member of a team: a species with its set.
_Avoid_: candidate, slot, member

**Field**:
One decision variable of one Pokémon: its species, item, ability, one move, its Stat Points or its Stat Alignment.

**Set**:
Everything about one Pokémon except its species: item, ability, four moves, Stat Points and Stat Alignment. Collides with the mathematical sense of the word, so name the mathematical one explicitly.

**Stat Points**:
The points a set gives its six stats: 0 to 32 per stat, 66 at most in total.
_Avoid_: EVs

**Stat Alignment**:
A set's choice of nature, such as Adamant.
_Avoid_: nature (as the field's name)

**Selection**:
The four Pokémon brought at team preview.

**Duplicate team**:
A team with the same six Pokémon as another, in any slot order. Teams that differ only in Stat Points count as duplicates while the model does not choose Stat Points.
_Avoid_: same line-up (two teams can share six species and still differ)

**Species swap**:
Replacing one species of a line-up with another. Counted from the closest line-up in a model's training data: zero species swaps means the line-up was copied.
_Avoid_: species change, mutation

**n-swap family**:
The line-ups that differ from a reference line-up by exactly n species, in any slot order. A one-swap family shares five species with the reference and replaces the sixth.
_Avoid_: neighbours

**Novelty**:
How many species swaps a generated team's line-up is from the closest line-up in the model's training data: zero (copied), one, two, or three or more (a new line-up).
_Avoid_: originality, uniqueness

**Meta**:
The top-50 meta teams, each counted once. A team's win rate against the meta averages one battle against each of the 50.
_Avoid_: usage-weighted meta, species usage %

**Validation opponent**:
One of 10 meta teams, whole one-swap families, kept out of training labels and prompts. Settings are chosen against them.
_Avoid_: held-out meta team

**Training opponent**:
One of the other 39 meta teams.

**Held-out team**:
One of the 50 meta teams placed 51st to 100th on the VGCPastes Reg M-B sheet. Nothing trains on it or searches against it; it scores the final selected counter once, at the end.
_Avoid_: test team, next-50

**Contamination**:
A held-out team, or a team with the same 84 fields, reaching training, a search or stored battle results before the final score. Only held-out teams count: testing a model on a team it trained on that is not held out is not contamination.
_Avoid_: leak, leakage

**Training contamination**:
Contamination of a model's training data. Example: before 2026-10-12, 56 of the scratch diffusion model's 692 training teams were held-out teams or had the same 48 fields.

**Search contamination**:
Contamination of a search: it battles a held-out team, or uses results against one to choose a candidate.

**Cache contamination**:
Contamination of stored battle results: results against a held-out team kept in the matchup cache. Example: the 2,597 results deleted on 2026-10-12.

**Objective function f**:
What the search maximises: the expected win rate of a team against the meta.
_Avoid_: fitness, oracle

**Win rate**:
The share of battles a team wins against the meta, with the behaviour-cloning policy on both sides and the Pokémon order shuffled every battle, stated with its battle count. Only shuffled-slot battles count; listed-order battles are biased (set 2026-11-07). A real meta team scores about 0.47–0.49.
_Avoid_: score (alone)

**Shuffled-slot battle**:
A battle in which each team's six Pokémon are put in a fresh random order before the behaviour-cloning policy sees them. Every battle must be one: the policy picks by slot, bringing slots 2 to 4 and 6 more often than chance and slots 1 and 5 less, so a fixed order biases the score.
_Avoid_: listed-order battle (never a measurement)

**Good enough**:
A new model, sampler or recipe whose curve moves outside the previous Pareto frontier on two named axes. Report where and by how much. Set 2026-11-05 on #169.
_Avoid_: pass/fail, beats the baseline

**Matchup predictor**:
A model that predicts a team's win rate without battling it.
_Avoid_: oracle

**Battle policy**:
The program that plays each battle.
_Avoid_: bot

**Monte Carlo evaluation**:
Estimating f by running battles. The only ground truth is ladder validation.
_Avoid_: ground truth

**Team writer**:
*Retired 2026-10-28 — neither model is called a writer any more. Use **From-scratch discrete diffusion model trained only on valid Pokémon teams** or **Pretrained text diffusion model fine-tuned on pastes**.*
A model that writes whole teams. The scratch team writer learned only from this project's teams; the text team writer is a pretrained text diffusion model fine-tuned on pastes.
_Avoid_: generator (alone)

**Diffusion model**:
One of the two kinds this project experiments with: the from-scratch discrete diffusion model trained only on valid Pokémon teams, or a pretrained text diffusion model. Always say which.
_Avoid_: the diffusion model (alone), mask diffusion model, domain (for a kind of model)

**Discrete diffusion model**:
A generative model for sequences of tokens from a finite vocabulary, such as a team's 84 fields. A forward process corrupts each token on its own, more as a time t runs from 0 to 1, until the sequence is pure noise. Run backwards, the process jumps from a sequence x to a neighbour y at a rate set by the ratio p_t(y)/p_t(x), so the network learns what is needed to estimate those ratios: for a masked model, the probability of each clean token given the visible ones. Sampling starts from pure noise and runs the learned reverse process. What the corruption turns a token into names the kind: a mask token (masked diffusion model) or a random vocabulary value (uniform discrete diffusion model).
_Origin_: D3PM, [Austin et al. 2021](https://arxiv.org/abs/2107.03006), which defines both kinds as transition matrices. The continuous-time view and the time reversal: [Lou, Meng and Ermon's blog post](https://aaronlou.com/blog/2024/discrete-diffusion/) and [Thiéry's note on masked discrete diffusion](https://alexxthiery.github.io/notes/DiscreteDiff/DiscreteDiff.html).
_Avoid_: Gaussian diffusion (the continuous kind, for images)

**Masked diffusion model**:
A discrete diffusion model whose forward process replaces each token with a mask token at a random time and keeps it masked from then on: the mask is absorbing. The reverse process only unmasks, so a filled token is never changed again (carry-over), and the network sees both directions. Training is a cross-entropy on the masked tokens, weighted by the masking rate and averaged over time. Both kinds of diffusion model here are masked, so the term names the kind, never one model: say which.
_Origin_: the absorbing transition in D3PM (Austin et al. 2021); the weighted cross-entropy in MDLM, [Sahoo et al. 2024](https://arxiv.org/abs/2406.07524), and MD4, [Shi et al. 2024](https://arxiv.org/abs/2406.04329).
_Avoid_: mask diffusion model, MDM (alone), absorbing diffusion

**Uniform discrete diffusion model**:
A discrete diffusion model whose forward process replaces each token with a value drawn uniformly from its vocabulary, with no mask token. Pure noise is a random team. Every reverse pass may change any field, including one it already changed, and sampling runs a fixed number of passes because no field is ever finished. The four arms of the uniform discrete diffusion map (#270) are this kind; the default model is not.
_Origin_: the uniform transition in D3PM (Austin et al. 2021); UDLM, [Schiff et al. 2024](https://arxiv.org/abs/2412.10193), which adds guidance; Duo, [Sahoo et al. 2025](https://s-sahoo.com/duo/), which shows it arises from an underlying Gaussian diffusion.
_Avoid_: uniform diffusion model (the name until 2026-11-11), uniform-noise model, uniform-state diffusion (the papers' name)

**Masking rate**:
The probability that the forward process has corrupted a field by time t: masked, for a masked diffusion model, or replaced by a random value, for a uniform one. The per-field masking rate gives each field type its own: t to the power 16 for species, 4 for ability and item, 1.5 for moves, 1 for Stat Alignment and Stat Points. The linear masking rate corrupts every field with probability t.
_Origin_: the linear masking rate is MD4's "linear schedule" (Shi et al. 2024).
_Avoid_: noise schedule, scheduler, shared masking rate

**Partial masking**:
Writing each token's vocabulary index as ℓ digits in base b and masking the digits independently, so a token can be masked, unmasked, or partly revealed in between. On OpenWebText the authors report idle passes falling from 37% to 0.25% at ℓ = 6, and the first masked diffusion model to beat an autoregressive model on perplexity without autoregression (15.36 against 17.54).
_Origin_: MDM-Prime, [Chao et al., NeurIPS 2025](https://arxiv.org/abs/2505.18495), with a [project page](https://chen-hao-chao.github.io/mdm-prime/).
_Avoid_: Prime (alone), sub-tokenization (the tokenizer sense)

**Concrete score**:
For a sequence x and a neighbour y that differs in one token, the ratio p(y)/p(x): the discrete counterpart of the score in continuous diffusion. A model that learns these ratios can run the reverse process without a normalizing constant. Score entropy is the loss that learns them.
_Origin_: SEDD, [Lou, Meng and Ermon 2024](https://arxiv.org/abs/2310.16834).

**From-scratch discrete diffusion model trained only on valid Pokémon teams**:
A discrete diffusion model that started from random weights, not from a pretrained model, and learned only from this project's valid Regulation M-B teams. It generates all 84 fields of a team, Stat Points included (set 2026-11-07). Later mentions in the same passage may shorten it to "the discrete diffusion model".
_Avoid_: writer, team writer, scratch writer, scratch team writer, discrete diffusion model trained from scratch on valid Pokémon teams (the name before 2026-10-30)

**48-field model**:
*Retired 2026-11-07. The project uses only the 84-field discrete diffusion model.*
The earlier from-scratch discrete diffusion model, which generated every field except Stat Points and copied each Pokémon's Stat Points from a random corpus team of the same species. It is never a default, an arm or a baseline.
_Avoid_: copied mixture, deployed mixture

**Default model**:
The discrete diffusion model checkpoint that every run uses unless the run names another. It is the battled checkpoint closest to got = ask. A challenger replaces it only when its ask-versus-got error, read from 3 sampling seeds, is lower by at least 0.001 with the interval above zero. Validation loss still sets how long a model trains; the error only chooses among saved checkpoints. Since 2026-11-07 it has been the 80-epoch labelled model.
_Avoid_: deployed model, best model, the 80 epoch (alone)

**Pretrained text diffusion model fine-tuned on pastes**:
A text diffusion model pretrained on general text by others, then fine-tuned on this project's Showdown team pastes. Later mentions in the same passage may shorten it to "the pretrained text diffusion model".
_Avoid_: writer, text writer, text team writer

**Masked text diffusion model**:
A pretrained text diffusion model that fills masked tokens anywhere in its text at once, seeing both directions. The only kind this project uses (set 2026-11-06).
_Avoid_: block diffusion, BD3LM, Fast-dLLM v2 (as baselines)

**Generation memory ceiling**:
The most memory a generation run may use on the laptop: 17.76 GiB. A model over it is quantized; one under it stays in bf16. Still over: lower the batch size, then rule the model out.

**Domain adaptation**:
Training with similar teams in the prompt and a target team that beats them, tested on this project's teams. The paper's academic benchmark is not reproduced (set 2026-11-06).
_Avoid_: DiBO reproduction

**Jev**:
TypeSafe AI's decision model. It picks one option from a list of at most 255 and gives a probability for each. In the Mask Lab grids it gets only legal options.
_Avoid_: the baseline

**Mercury**:
Mercury 2.5, Inception's diffusion language model, called through OpenRouter as `inception/mercury-2.5`. It writes a team as text, at reasoning off, low, medium or high.

**Training-mix retrain**:
The discrete diffusion model fine-tuned again with one change to which training teams it sees each epoch, in as many steps as its same-labels retrain. Named by its quota.
_Avoid_: balancing, rebalanced dataset

**Same-labels retrain**:
The control for a training-mix retrain: the same model fine-tuned for the same number of steps on its unchanged training teams.
_Avoid_: control (alone)

**Quota**:
The most training teams one group may give a training-mix retrain in one epoch. A group under its quota gives all its teams.
_Avoid_: cap, bucket size

**Win-rate quota**:
A training-mix retrain whose groups are 0.1-wide win-rate bands. Each band's quota is the 0.5–0.6 band's size.
_Avoid_: balanced buckets

**Species quota K**:
A training-mix retrain whose groups are species. A team is kept only while none of its six species is already in K kept teams that epoch. Example: species quota 800.
_Avoid_: Pokémon bucket, species cap

**Epoch**:
One shuffled trip through every training team during training.
_Avoid_: pass (a pass is a sampling step)

**Training run**:
One training or fine-tune of a model, from its first epoch to its last saved checkpoint. Sampling grids, battles and search loops are runs, but not training runs.
_Avoid_: job, training (alone)

**Learning curve**:
Training loss and validation loss at each epoch of one training run.
_Avoid_: curves, loss log

**Experiment tracking**:
Recording each training run's hyperparameters, learning curve and checkpoints in one place, so runs can be compared side by side.
_Avoid_: logging (alone), monitoring

**Early-stopping threshold**:
The smallest drop in validation loss that resets patience. A new lowest validation loss resets the patience count only when it beats the previous best by more than the threshold: 0.001 in the 84-field training loop since PR #246. A smaller drop still becomes the kept best epoch, so the threshold changes when training stops, not which checkpoint is kept. _Origin_: the stopping criteria of [Prechelt 1998, "Early Stopping — But When?"](https://doi.org/10.1007/3-540-49430-8_3); Hugging Face's `EarlyStoppingCallback` calls it `early_stopping_threshold`.
_Avoid_: min_delta (Keras's name), tolerance

**Planted bug**:
A deliberately wrong version of one line of code, run against the test suite to check that at least one test fails. A planted bug that every test passes marks a hole in the tests. _Origin_: mutation testing, [DeMillo, Lipton and Sayward 1978](https://doi.org/10.1109/C-M.1978.218136).
_Avoid_: mutant (mutation testing's name), injected fault

## Sampling

**Ask**:
The win rate the model is conditioned on when it generates a team, set before any battle.
_Avoid_: threshold, target

**Got**:
The mean win rate of the teams generated at one ask, from battles against the top-50 meta teams with the behaviour-cloning policy on both sides.
_Avoid_: achieved, delivered

**Ask-versus-got slope**:
The slope of the straight line fitted to got against ask across a model's asks. At 1, got rises by as much as the ask does; got can still sit above or below the ask. Dropped 2026-11-05: never reported, not even second; the ask-versus-got error judges a model.
_Avoid_: calibration

**Ask-versus-got error**:
Mean squared error between each ask and the got at that ask, on asks 1/24 to 17/24, with 18/24 to 23/24 reported apart. Near the line means under 0.001. It replaced the slope as the judge on 2026-11-05.
_Avoid_: calibration, closeness

**Set-only sharpening**:
Sampling that draws species from the model's full odds and sharpens only the set fields (item, ability, moves, Stat Points, Stat Alignment) toward their likeliest options. Temperature or nucleus share sets how hard it sharpens; that value tells the arms apart and leads their names: "Temperature 0.3 on sets only", "Nucleus 0.7 on sets only". The sibling that sharpens species too is "Nucleus 0.9 on all fields".
_Avoid_: split-variance sampling, set-field temperature, set-field nucleus, cold sets

**Infill template**:
What a masked text diffusion model receives for infilling: a short header and the team written once as compact JSON, each hidden value replaced by mask tokens it fills where they stand. _Origin_: `MDLMSampler.infill` in dLLM.
_Avoid_: in-place text, free-text prompt

**Sampler**:
How the discrete diffusion model fills a blank candidate until it is complete: its decoding order, how many fields each pass fills, and whether it remasks. The ask, guidance weight and temperature shape each draw; they are not part of the sampler.
_Avoid_: decoder, generator (alone)

**Pass**:
One run of the network during sampling. A pass fills one or more masked fields; one field per pass fills a team in 84 passes.
_Avoid_: decoding step, step, iteration

**Carry-over**:
In a masked diffusion model, a field already unmasked keeps its value: the prediction for it is fixed to that value, and only masked fields are drawn. Plain masking is carry-over; remasking breaks it on purpose.
_Origin_: MDLM (Sahoo et al. 2024) and MD4 (Shi et al. 2024).
_Avoid_: copy-over

**Idle pass**:
A pass that changes no field. A masked sampler that unmasks by a fixed rate over many passes has many: 37% of 1,024 passes on OpenWebText. A sampler that fills exactly one field per pass, as ours does in 84 passes, has none.
_Origin_: MDM-Prime (Chao et al. 2025), which calls it an idle step.
_Avoid_: idle step

**Decoding order**:
The order in which a sampler fills a candidate's masked fields. The four used so far are species-first order, Pokémon-by-Pokémon order, random order and confidence order.
_Avoid_: unmasking order, generation order, slot order (slot order is the order of Pokémon in a battle)

**Species-first order**:
The default decoding order: all six species, then abilities and items, then moves, then Stat Alignments.

**Pokémon-by-Pokémon order**:
The decoding order that fills all of one Pokémon's fields before starting the next Pokémon.
_Avoid_: slot-by-slot order

**Random order**:
A decoding order drawn at random for each candidate, so species, sets and Pokémon interleave.

**Confidence order**:
The decoding order in which each pass fills, for each candidate, the masked field the model is surest about.
_Avoid_: greedy order

**Plain masking**:
Sampling in which a filled field is never masked again, so every choice is final once drawn. The default sampler is plain masking in species-first order.
_Avoid_: vanilla sampling, standard sampling

**Remasking**:
Masking again some filled fields of a partial candidate so a later pass draws them anew and an early choice can be undone. The remasking rule says which fields and how many.
_Avoid_: resampling, ReMDM (the paper's name for it)

**Least-likely remasking**:
Our rule (2026-10-30): before each pass after the first, mask again a fixed share of filled fields, those whose values were least likely when drawn, and redraw them in that pass. Run at 5%, 10% and 20%.
_Origin_: no paper; closest is Informed Correctors at temperature zero, [Zhao et al. 2024](https://arxiv.org/abs/2407.21243).
_Avoid_: low-confidence remasking (LLaDA's rule touches only fields filled in the current pass)

**Random-field remasking**:
Least-likely remasking's count of fields, picked at random among the filled ones instead. Run at 10% and 20% under random order.
_Origin_: no paper; ReMDM-cap also ignores confidence, [Wang et al. 2025](https://arxiv.org/abs/2503.00307).

**Confidence-weighted remasking**:
Each filled field is masked again with odds that rise as its draw probability falls, 0.3 fields per pass on average, inside the remasking loop.
_Origin_: ReMDM-conf with the rescaled schedule, [Wang et al. 2025](https://arxiv.org/abs/2503.00307), §4.1.

**Even-odds remasking**:
Every filled field is masked again with the same odds, 0.008 per pass (about 0.5 fields once 63 are filled), inside the remasking loop.
_Origin_: ReMDM-cap, [Wang et al. 2025](https://arxiv.org/abs/2503.00307), §4.1. The paper's main results use 0.02.

**Remasking loop**:
Fill 63 fields without remasking, then 126 passes that only remask and redraw at a fixed mask rate, then fill the last 21: 210 passes in all.
_Origin_: ReMDM's loop strategy, [Wang et al. 2025](https://arxiv.org/abs/2503.00307), §4.2.

**Dead end**:
A partial candidate whose filled fields already rule out every value of some masked field, so no completion of it is legal.
_Avoid_: conflict, deadlock

**Candidate species**:
The species a Pokémon of a partial candidate could still be, given its filled fields and Species Clause.

**Lookahead**:
A sampler rule that allows a value only if every Pokémon whose species is still masked keeps at least one candidate species. It stops a set filled before its species from ruling out every species.
_Avoid_: constraint propagation (the general technique), look-ahead

## Competitions

**Kickstarting**:
"A method for using previously-trained 'teacher' agents to kickstart the training of a new 'student' agent", from [Kickstarting Deep Reinforcement Learning](https://arxiv.org/pdf/1803.03835). My first Pokémon competition's [59th-place writeup](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/writeups/59th-place-solution-for-the-ptcg-ai-battle) led me to it. My words: "I believe that pokemon is spending most time on reflection what I did wrong with the first pokemon competition why did my research did not find this paper".

## Daily blog

Words the daily blog post and its prompt use about my own work with agents.

**Workday**:
One America/Chicago calendar day, from local midnight to the next, that a daily blog post covers.
_Avoid_: yesterday (as a date), the day before

**Session**:
One Claude Code or Codex conversation, named by its title and the first 8 characters of its ID.
_Avoid_: chat, thread, conversation

**Fork**:
A session copied from another at one message, sharing every message before it.
_Avoid_: branch, side chat

**First prompt**:
The first message I typed into a session that started on the workday.
_Avoid_: initial prompt, first instruction

**Task prompt**:
A message I typed that asks for work. A button press, pasted text and a message Claude wrote are not task prompts.
_Avoid_: request (alone), query

**Done condition**:
The part of a task prompt that says how to tell the work is finished.
_Avoid_: success criteria, acceptance criteria

**Frustration**:
An exchange where an agent missed what I asked, or where I repeated an instruction, re-asked a question, corrected an output or rejected it.
_Avoid_: complaint, issue, pain point

**Correction**:
A frustration where I fixed a wrong output myself: a wrong number, a false "done" or a wrong label.
_Avoid_: fix (alone)

**Re-asked question**:
A question I typed that repeats an earlier question in the same session.
_Avoid_: follow-up

**Session packet**:
The extract of one session, at most 20 KB, holding my typed messages, its recaps, tool errors and hook blocks, that a subagent reads for a retro.
_Avoid_: dump, transcript (alone)

**Retro**:
One session's review by the `retro` skill: improvement candidates, each with a severity, quoted evidence and a fix.
_Avoid_: postmortem, review (alone)

**Question test**:
The check that a post carries enough detail: questions written from the transcripts, answered from the draft alone.
_Avoid_: quiz

**Writing note**:
The one-line coaching on a task prompt's prose, graded against four rules: `Writing: rule n` or `Writing: clean.`
_Avoid_: grammar check, writing feedback

**Steering move**:
A kind of feedback I give agents again and again, such as asking for a page I can see. A steering move is encoded once a memory, rule, hook or skill does it for me.
_Avoid_: habit, pattern (alone)

**Hand copy**:
A file copied from a worktree into the vault by a command, because the app blocks vault edits from a worktree.
_Avoid_: sync

**Decision closed**:
A question or experiment that reached a stated verdict on the workday.
_Avoid_: resolved item
