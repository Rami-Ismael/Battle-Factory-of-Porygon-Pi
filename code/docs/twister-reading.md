# Reading *Twist, Don't Tilt* (TWISTER) against the masking-schedule grids

Source: Thimmaiah, Marinov, Srinivasa, Vikalo, Li and Gligoric, "Twist, Don't Tilt: Trajectory-Exact Constrained Decoding for Masked Diffusion Models", [arXiv 2609.35609](https://arxiv.org/abs/2609.35609) (v1, 2026-09-28, ICLR 2027 submission). Read on 2026-11-15 from the arXiv LaTeX source, main text and appendices. Six Haiku subagents read line-numbered chunks; every citation below is `file:line` in that source. A seventh agent was given the paper and 13 statements about it, but not the masking-schedule hypothesis, and tried to refute each one. Its corrections are folded in. The prompt named Fang et al. (arXiv 2604.00375) and ended with this paper's title. Fang was read on 2026-11-14 (`docs/fang-2026-quality-exploration-reading.md`, branch `t3/remasking-paper-masking-schedule-blog`), so this reading uses the same template for TWISTER.

Visual version: [the report page](figures/twister_reading/index.html). Scripts: `code/scripts/twister_reading/`.

Rows: the #297, #322 and #335 ask grids, both checkpoints read at epoch 80 of training on the shuffled-only table. Each name gives the decoding order first, then the masking schedule.

| Reader name | Code ID | Masking schedule in training | Decoding |
| --- | --- | --- | --- |
| Species-first, per-field schedule (the control) | `sched_species_first` | species hidden with chance t^16, ability and item t^4, moves t^1.5, Stat Alignment and Stat Points t | six species first, then sets in a fixed order, one field per network pass |
| Species-first, linear schedule | `sched_linear_species_first` | every field hidden with chance t | same |
| Species-then-random, linear schedule | `sched_linear_species_then_random` | linear | six species first, then the 78 set fields in random order (#322) |
| Random-order, linear schedule | `sched_linear_random` | linear | every field in random order (LLaDA Algorithm 4), with the lookahead |
| Random-order, per-field schedule | `sched_per_field_random` | per-field | every field in random order, with the lookahead (#335) |

Win rate is one battle against each of the 49 distinct top-50 meta teams, one of them counted twice as in the grids, behaviour-cloning policy on both sides, slots shuffled. Each ask has 192 teams (three sampling seeds of 64).

## Answer

Yes, about decoding order, not about the schedule. TWISTER says that a sampler which removes illegal values at each step and renormalises draws from a different distribution than the model's own distribution restricted to legal outputs (the "tilt"). In our grids:

- Species-first rows have no tilt. The legal Pokémon construction tables sit inside the trained model (training loss and every network pass), and one field per pass leaves nothing to re-score. The linear against per-field comparison under species-first order is therefore between the two models' own distributions.
- Random-order rows are tilted by a median 13.6 nats per team (linear schedule) and 15.2 (per-field), guidance 2. Almost all of it is the lookahead.
- Within an ask, the more a random-order team was bent, the more it loses (Spearman +0.39 linear, +0.32 per-field). Reweighting to TWISTER's target estimated that correcting the tilt closes 84% to 89% of random order's mean win-rate gap to species-first at guidance 2. Drawing exactly from that target, by rejection without the lookahead (#358, run 2026-11-17), closes 49% [41%, 57%]: mean win rate 0.410 against 0.334 as drawn and 0.490 species-first. The estimate rested on 1–2% effective teams. Best of 64 does not rise.
- The legal Pokémon construction tables remove far more than the lookahead (a median 108 nats per team on the linear checkpoint, 151 on the per-field), but the training loss masks the same values, so that mass is untrained. A team's tables cut tracks its win rate at only +0.10 and +0.04 (section 2b).
- For the blog: about half of "random order loses" is "the lookahead bends random-order teams toward weak ones"; the other half is the order itself. The species-first schedule comparison is untouched. A random-order schedule comparison mixes the schedule with a tilt that is larger on the per-field checkpoint.

## 1. Scope

| Claim | Where | Training or decoding | Carries over? | Assumption it needs |
| --- | --- | --- | --- | --- |
| Composing exact per-step constrained draws tilts the path law away from the model's relative probabilities over valid trajectories | `abstract.tex:17-22`; Proposition Re-Freezing Tilt, `problem-trajectory-bias.tex:134-152` | Decoding | Yes, to our random-order rows: the lookahead and same-pass legality checks remove probability at each draw | The model's per-position predictions are taken as fixed; "valid" is decided at the end |
| No tilt for a vacuous constraint, or for one-step reveal-all | Corollaries, `problem-trajectory-bias.tex:189-215`; proofs `appendix-trajectory-bias.tex:27-160` | Decoding | Yes, to species-first rows in effect: the tables are inside the model, and nothing is re-scored | Refuter S2: there is also a kernel-level iff condition (`appendix-trajectory-bias.tex:7-19`) |
| The tilt factorises over steps, so a Feynman-Kac (twisted SMC) correction fixes it; for regular languages the potentials are exact | `method-automaton-twist.tex:10-64`; `abstract.tex:36-41` | Decoding | Measured here by reweighting finished teams, not run as SMC | Regular-language constraint compiled to an automaton over tokens |
| TWISTER keeps constraint satisfaction | `appendix-eval.tex:40-43`; JSON table | Decoding | No: we already redraw until Showdown's validator accepts | None |
| The bias grows with the number of denoising steps | Figure 1; `method-eval.tex:10` | Decoding | As a direction: our random order has 84 passes | Toy token-class constraint ('a.*') |

The paper trains nothing, varies no masking schedule, and uses six released models (refuter S1: holds). Section names: §2.3 Twisted SMC, §3.1 Doob-transformed decoder, §3.2 automaton-constrained step-exact decoder, §3.3 frozen-denoiser lookahead, §3.4 trajectory bias, §4.1 Feynman-Kac corrector, §4.2 algorithm, §4.3 trajectory-bias experiment.

How it maps onto our code (`787fac31`):

| Where a rule acts | Code | Effect in TWISTER's terms |
| --- | --- | --- |
| Training loss | `diffusion84.py:296` masks values the tables forbid before the cross-entropy | Part of the model |
| Every network pass | `diffusion84.py:524` masks before `log_softmax` | Part of the model |
| Several fields drawn from one pass (random order) | `remask84.py:69-79`: one `_logprobs` call, then each revealed field in order, legality seeing earlier draws of the same pass | Tilt (TWISTER's mean-field step, enforced greedily) |
| Lookahead (random order) | `diffusion84.py:548`, `lookahead.py` | Tilt (a 0-or-1 version of the step-exact decoder's feasibility) |
| [MASK] value, empty first move | `diffusion84.py:556-562` | Tilt, about 0.000 nats |
| Showdown validator redraw | `ask_grid84.py` `draw` | Exact conditioning, no tilt |

## 2. Predictions, then the data

Predictions were written at 18:07 on 2026-11-15, before any correction weight existed and before the redraws (`predictions.md`, SHA-256 `413666d42f02da4d…`; copied to `~/vgc-data/results/twister_reading/`). Not blind: the Fang and Two Temperatures readings had reported best-of-k, line-up, non-top-share and order-gap numbers, and one cell's try count had been seen.

Measurement. `code/scripts/twister_reading/tilt_weights.py` redrew every cell of asks 13/24 to 17/24, seeds 1 to 3, through `ask_grid84.draw` with its stored seed base. The script swapped in a recorder for `diffusion84.Refusals`, which changes no draw. All 135 cells (9 row-guidance pairs × 15) gave the stored pastes bit for bit, so each team's weight sits beside its 49 stored battle outcomes. For each draw the recorder takes the guided, temperature-scaled distribution the network gave and the set of values the draw used, and adds log(kept share). The team's sum is the log of its correction weight: its probability under the trained model's own per-pass distributions over its probability under the sampler, along the same draws. Up to a constant, this is the importance weight against TWISTER's target (the native path law conditioned on a valid team; refuter S12: holds for a proposal that removes values and renormalises). No kept team had a dead-end draw: the validator had rejected those. `tilt_analysis.py` then reweights. Intervals: 2,000 paired bootstrap replicates, opponents redrawn once and shared, teams redrawn with their weights within each ask; percentile intervals. Best-of-k here draws with replacement under the weights (0.582 at weight 0 against 0.587 for the unbiased without-replacement estimate). Pass@k under weights is the plug-in 1 − (1 − p)^k. The unweighted unbiased estimates are the Two Temperatures reading's.

| # | Paper's claim | Prediction | Confidence | Measured | Verdict |
| --- | --- | --- | --- | --- | --- |
| T1 | No tilt when nothing is re-scored against the constraint | Species-first rows: kept share 1 for every team | 99% | Median 0.000 nats, max 0.08, both checkpoints, guidance 2, 4 and 8 (3,456 fresh teams, other session's `legality_skew.py`); the grid teams agree (effective share 1.000) | Holds |
| T2 | Step-exact composition tilts the path law | Random order: median refused share 20% to 80%; effective teams 20% to 80% | 55% | Median 13.6 nats removed per team (linear) and 15.2 (per-field), guidance 2: refused share 1.000. Effective teams under full weights 2.3% and 1.4% | Fails: the tilt is far larger than predicted |
| T3 | Correcting the tilt changes outputs | Mean win rate moves < 0.01, best of 64 < 0.02 | 75% | Mean: +0.131 [+0.061, +0.183] linear, +0.142 [+0.035, +0.191] per-field (guidance 2); +0.07 to +0.11 at 4 and 8. Best of 64: +0.003 [−0.063, +0.029] and −0.006 [−0.088, +0.009] at guidance 2; −0.012 to −0.065 at 4 and 8 | Partly: mean moves a lot, best of 64 does not rise |
| T4 | (implied) tilted teams differ in quality | Spearman of log weight with win rate below 0.1 in size | 65% | Within-ask Spearman +0.39 and +0.32 (guidance 2), +0.27 to +0.39 across guidance | Fails: less-bent teams win more |
| T5 | Bias grows with how much re-scoring changes the predictions | Per-field random order refuses more than linear | 60% | Median 15.2 against 13.6 nats (guidance 2), 15.8 against 14.5 (4), 21.1 against 18.3 (8); fresh teams 13.4 against 11.8 | Holds |
| T6 | — | Refused mass rises with guidance 2 to 8 | 50% | Linear 13.6, 14.5, 18.3; per-field 15.2, 15.8, 21.1 nats | Holds |
| T7 | Step-exact decoding keeps validity | Species-first first-try legality at or above random order at every ask 13-17 | 70% | Species-first 100% at every ask and guidance; random order 80% to 93%; species-then-random 92% to 99.5% | Holds |
| T8 | Masking schedule | TWISTER says nothing between linear and per-field under species-first | n/a | Both rows have zero tilt; the battle difference is already known to be inside ±0.04 at best of 64 | Cannot decide (nothing to decide) |

Win rate by tilt decile (random order, linear schedule, guidance 2, deciles within each ask, about 96 teams each): 0.265, 0.256, 0.323, 0.321, 0.318, 0.327, 0.368, 0.358, 0.394, 0.412 from most to least bent. Species-first on the same checkpoint: 0.490.

Mean win rate as drawn, corrected, and species-first (same checkpoint, asks 13-17):

| Row | Guidance | As drawn | Corrected | Species-first | Gap closed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random-order, linear schedule | 2 | 0.334 | 0.465 | 0.490 | 84% |
| Random-order, linear schedule | 4 | 0.359 | 0.464 | 0.516 | 67% |
| Random-order, linear schedule | 8 | 0.366 | 0.457 | 0.530 | 55% |
| Random-order, per-field schedule | 2 | 0.330 | 0.473 | 0.490 | 89% |
| Random-order, per-field schedule | 4 | 0.356 | 0.426 | 0.519 | 43% |
| Random-order, per-field schedule | 8 | 0.365 | 0.457 | 0.540 | 53% |

Rejection without the lookahead, run since (#358, 2026-11-17, linear schedule, guidance 2): 0.410, −0.079 [−0.094, −0.064] against species-first, so it closes 49% [41%, 57%] of the gap, not the 84% in the table. Read the corrected column as direction, not level: the full weights leave 1% to 2.5% effective teams, and the five heaviest teams carry 45% to 57% of the weight. The dose curve (weights raised to a strength 0 to 1) rises smoothly from the first steps, where most teams still count (strength 0.1: 54% to 66% effective, mean +0.03).

Where the tilt sits (mean removed log mass per team, per-field schedule, guidance 2): moves 8.8 nats, abilities 5.4, items 1.8, Stat Points 0.3, species 0.25. These are the set fields drawn before their species, which the lookahead cuts. The lookahead alone accounts for 12.5 of the 13.6 median nats (linear) and 14.2 of 15.2 (per-field); same-pass legality checks account for the rest. Species-then-random: median 0, 90th percentile 1.17 nats (Stat Point budget within a pass), effective share 0.89.

A weight does not say why a team is weak. Teams whose sets were drawn long before their species are both the most bent and, from the order results, the weakest; the weight and the order are confounded. What the weights do say is that TWISTER's own target, the model's random-order conditionals restricted to legal teams, is a much stronger set of teams than the one the grids battled.

### 2b. Two follow-up measurements (2026-11-16)

**The tilt on one Pokémon, exactly.** `tilt_exact_toy.py` fixes one Pokémon's first move to Fake Out, masks every other field of the team, and draws ability, then second move, then species on the random-order linear model (ask 15/24, guidance 2). It enumerates all 2,703 legal paths with the model's own probabilities. Against draw-then-reject (the model's own odds among legal outcomes), the lookahead sampler moves 17.4% of the probability (total variation distance). Incineroar falls from 51.8% to 34.6%, and Meowstic-F rises. The lookahead keeps an ability if any Fake Out user has it, but does not weight it by how much of the model's second-move probability stays legal afterwards: 40% for Intimidate, 20% for Competitive. Draw-then-reject loses most Competitive paths at the second move, while the lookahead saves every one. 200,000 real draws each way match the exact numbers within 0.6 points.

**The tables against the lookahead.** `tables_tilt.py` redrew the 1,920 species-first grid teams (both checkpoints, asks 13/24 to 17/24, guidance 2), all bit for bit. For each team it recorded the probability the legal Pokémon construction tables remove from the bare network (`base_law.py`, `Model84._logprobs` without the tables mask):

| Rule | Rows | Median removed per team | Spearman with win rate, within ask |
| --- | --- | ---: | ---: |
| Tables | Species-first, per-field schedule | 150.9 nats | +0.04 |
| Tables | Species-first, linear schedule | 108.1 nats | +0.10 |
| Lookahead | Random-order, per-field schedule | 15.2 nats | +0.32 |
| Lookahead | Random-order, linear schedule | 13.6 nats | +0.39 |

Most of the tables' cut falls on moves (about 45 nats) and abilities (about 28). `elbo_terms` applies the same mask in the training loss, so the forbidden values get no gradient, and the bare network's mass on them is left over from initialisation. Reweighting by the tables' share leaves 1% effective teams and moves the mean in opposite directions on the two rows (+0.08 linear, −0.07 per-field): noise. The tables define the model's distribution and do not tilt it. They could tilt on a model trained without the tables in the loss (#361).

## 3. The paper's assumption: one consistent model

TWISTER takes the denoiser's per-position predictions as conditionally independent given the state (`background-mdlm.tex:25`, `problem-automaton-constrained-decoder.tex:6`) and never checks that they come from one joint distribution (refuter S11, corrected: it assumes the factorisation, it does not test consistency). Ours are not consistent. Reused from the Two Temperatures reading (`order_consistency.py`, 1,024 held-out teams): species-first minus moves-first log-likelihood is 18.2 nats per team [17.7, 18.8] on the per-field checkpoint and 12.3 [11.9, 12.7] on the linear one. By field type, per-field checkpoint: species −16.9 nats (species score better when drawn after their moves), moves +35.4 (moves score far worse drawn before their species); linear −17.5 and +30.3; ability, item, Stat Alignment and Stat Points under 0.5. So TWISTER's target is order-specific here: corrected random order is the model's own random-order distribution, not its species-first one. The tilt and the order gap point the same way: both concentrate in moves and abilities filled before species.

## 4. Experiment-design lessons the paper's own evidence supports

| Weakness | Quote | What a fair version costs |
| --- | --- | --- |
| Compute matching | "Although we report average generation times, our implementation is not optimized for performance." (`appendix-eval.tex:41`). TWISTER with 1 or 4 particles takes 1.4 to 2.1 times DINGO's mean time (table macros, `table-json-mode-macros.tex:11-103`); no method gets TWISTER's budget (refuter S8: holds) | Give the uncorrected decoder and rejection sampling the same denoiser calls: 8 × 10,000 constrained samples per point for 8 particles |
| Seeds | "% We run all experiments once." is commented out (`appendix-eval.tex:32`); the compiled text states no run count or seed. The only uncertainty is a split-half noise floor, "averaged with equal weight across partition configurations" (`method-eval.tex:9`) (refuter S6, S9) | Three seeds per point and bootstrap intervals on the distance: 3 × the sampling |
| The judge | "We use pairwise total variation distance to measure the similarity between the two distributions." (`method-eval.tex:7`), over token classes on a toy constraint ("For 'a.*'", `macros.tex:321`), 3 of the 6 evaluated models shown (refuter S5). JSON is scored only for validity with an unnamed checker, every method at 100% parse and 98-99% schema validity, no unconstrained row and no quality measure (refuter S7: holds) | Show all six models (the points exist); add one task where the tilt could change answer quality |
| Text against algorithm | The prose and appendix algorithm resample "when resampling is triggered by the effective sample size falling below ESS_min" (`appendix-algorithm.tex:14`); the main-text algorithm has the resampling block commented out (`alg-twister.tex:68-84`), and no ESS_min value is reported | One sentence and one number |
| Baselines | "The original code is not publicly available." for both DINGO and Dang & Ermon (`appendix-eval.tex:23, 25`); TWISTER's code "will be made publicly available upon acceptance" (`mandatory.tex:11`) (refuter S10) | Release the code |

The paper's Figure 1, read off the rendered PDFs (±0.005): uncorrected step-exact distance rises with denoising steps 1, 2, 4, 8, 16 from 0.069 to 0.260 (OWT-130M), 0.111 to 0.409 (Dream-Coder-Instruct 7B) and 0.161 to 0.223 (Dream base 7B). TWISTER with 8 particles ends at 0.095, 0.138 and 0.175, against noise floors of about 0.104, 0.081 and 0.166. It reaches the floor on two of the three models, not on Dream-Coder-Instruct. Sample counts: 40,000 rejection and 10,000 constrained samples per point, random remasking, temperature 1 (`method-eval.tex:10`).

What this reading itself got wrong at first: one of my statements said the figure spans "steps 1 to 16" (it is five points) and that TWISTER uses "K particles" (the table uses 1 and 4, the figure 4 and 8). The refuter caught both.

## 5. The decoding rule, quoted from code

Ours, `code/src/diffusion84.py` and `code/src/remask84.py` at commit `787fac31`:

```
# species-first rows (Model84.sample)
for step, c in enumerate(ORDER):                              # diffusion84.py:588  one field per pass
    lc = log_softmax(logits.masked_fill(~tables_ok, -1e4))    # :524  the tables, inside the model
    lc = g * lc + (1 - g) * lc_unguided                       # :592  classifier-free guidance
    legal, plain, kept, fallback = _legal_parts(x, c, lookahead)   # :593  [MASK], empty first move
    lc = lc.masked_fill(~legal, -inf) / temperature           # :595
    v = multinomial(softmax(lc))                              # :596

# random-order rows (remask84.sample_random, fill_pass)
reveal = masked & (rand < odds)                               # remask84.py:223  each field independently
lp = _logprobs(x, ask); lpn = _logprobs(x, none)              # :69   one pass for every revealed field
for c in targets:                                             # in ORDER within the pass
    lc = g * lp[c] + (1 - g) * lpn[c]                         # :74
    legal, plain, kept, fallback = _legal_parts(x, c, lookahead=True)   # :75  sees this pass's draws
    v = multinomial(softmax(lc.masked_fill(~legal, -inf)))    # :79
```

LLaDA's released sampler, `ML-GSAI/LLaDA` `generate.py` at `9182493720`: Gumbel noise at a temperature (`:174`), argmax (`:175`), confidence = probability of the argmax (`:180-182`) or a random number (`:184-185`), top-k of confidence chooses what to keep (`:192-196`). It has no constraint or legality mask anywhere.

TWISTER, `algorithms/alg-twister.tex` (no code): per particle, a step-exact FFBS draw (`:21-26`), the clamped partition sum under the old denoiser (`:27-31`), a new denoiser call and the partition sum under the new one (`:34-41`), weight × new/old (`:53-60`), normalise (`:62-66`); resampling commented out (`:68-84`).

Share of filled fields whose value was not the top value of the distribution it was drawn from (reused, Two Temperatures reading, 480 teams per row and guidance): about two in five overall in every row, from 37% to 41% at guidance 2. LLaDA's default (temperature 0, low-confidence remasking) is 0% by construction. The samplers differ in what they remove, not in how greedy they are.

## 6. Who blames what for the distorted distribution

Each paper's own text was read from its arXiv LaTeX source by a Haiku reader; every code link was opened by hand on GitHub on 2026-11-15.

| Paper | What it blames | Its evidence | Its own code | Text against code |
| --- | --- | --- | --- | --- |
| TWISTER, 2609.35609 | Composing exact per-step constrained draws: valid mass scored by one denoiser and re-scored by the next (§3.4) | TVD on a toy constraint, 3 models shown, no seeds | Promised on acceptance | Main algorithm drops the resampling the prose describes |
| Dang & Ermon, 2607.07026 | Does not discuss distortion; exact per-step sampling replaces DINGO's MAP | Validity and accuracy, LLaDA-8B, Dream-7B; 3 seeds for runtime only | None linked | — |
| DINGO, 2505.23061 | Calls itself "provably distribution-preserving" (`main.tex:91`) while taking the highest-probability string (`dp_block.tex:28`, argmax) | Greedy runs, no seeds | Checklist says "Code is provided" (`checkList.tex:99`); no public repository found | Text claims preservation; the algorithm is MAP |
| Dang et al., 2606.01926 | Locally constrained decoding's "greedy and myopic enforcement" (`problemsetup.tex:33`) | Accuracy against particles; no divergence; 3 seeds for latency only | `MhDang/gelatwo` @ `d974a70b9c` | README matches: LCD, GCD, P-GCD; `hmm/`, `nfa/`, `decoding/` |
| Grammar-Aligned Decoding, 2405.21047 | Grammar-constrained decoding treats every future as equally completable (`problem.tex:253`) | KL to the target, Mistral-7B, 36 problems, no seeds | `ebmoon/transformers-GAD` @ `bc062c68e0` | `eval_gad.py` and `eval_gcd.py` both present |
| Loula et al., 2504.13139 | "greedy approximations that badly distort the distribution" (`iclr2025_conference.tex:296`) | log Z − KL estimate, 4 domains, bootstrap 95% intervals | `probcomp/genlm-control` @ `e8e39c2040` | — |
| Mündler et al., 2508.10111 | Does not discuss distortion; its completion mode "cannot rely on the model's probability distribution" (`discussion.tex:12`) | Validity and pass@1, 95% intervals; text says four seeded runs, appendix says "seeds 0 to 4" | `eth-sri/constrained-diffusion` @ `0908291ae5` | The runner uses seeds `[0, 1, 2, 3]` (`eval/dllm/run_dllm.py:33`): code matches the main text, not the appendix |
| LAVE, 2602.00612 | Does not discuss distortion; optimises validity and time | syntactic@k, functional@k; run count not stated | `THU-Agent/LAVE` @ `b4800d7ab3` | Built on Mündler et al.'s package layout (`constrained_diffusion/`), with `generate_our.py` added; not a GitHub fork |
| DConstrain, 2610.30920 | Argues bias with a toy example (98.78% against 90%), measures none | BFCL, JSONSchemaBench; seed 0, one run | None linked | Cites TWISTER only in introduction and related work (`intro.tex:6, 8`, `related.tex:10`) |

The experiment that would settle it for any of them: same model and constraint; the uncorrected decoder, the corrected one and rejection sampling at matched denoiser calls; seeds; scored by distance to the rejection samples and by task quality. None of the nine runs all four parts. Ours is a small version of the first and last parts: the correction weights against battle outcomes.

## 7. Adoption

TWISTER is seven weeks old (v1 2026-09-28). Semantic Scholar lists one citer on 2026-11-15, DConstrain, which cites it and runs it nowhere. No code is released. No released sampler changed its default: LLaDA's `generate.py` has no constraint, and the most-starred constrained diffusion code (Mündler et al., 59 stars) enforces validity and redraws without correcting. Nothing has replaced TWISTER because nothing has used it. The methods around it (LAVE, DConstrain, Dang & Ermon) are shown faster or more often valid; none is shown closer to the model's distribution at matched compute.

## 8. Experiments to run (your pick; the first has run since)

All are sampling-only decoding rules on the same checkpoints, with common random numbers: the grid seeds 7301 to 7303, the same 49 opponents, paired bootstrap against the existing rows at asks 13/24 to 17/24. Battles: 47,040 per row (960 teams × 49), about 10 minutes on the laptop.

| Arm | Code ID | What changes | Prediction | Changes the blog if | Sampling cost |
| --- | --- | --- | --- | --- | --- |
| Rejection, exact random order | `rand_plain_reject` | Random order without the lookahead; redraw until Showdown accepts. Draws exactly TWISTER's target | Mean win rate near 0.45; best of 64 near 0.59 | Interval on mean win rate against species-first contains 0: random order's loss is the sampler's tilt. Stays 0.05 or more below: the order costs strength on its own | Acceptance 0.8% to 1.1% (estimated from the weights): about 100 tries per team, about 15 h per row; about 3 h at ask 15 alone. **Ran 2026-11-17 (#358):** 0.410, 49% [41%, 57%] of the gap; acceptance 1.52% of 63,295 tries |
| Twisted SMC, 8 particles | `rand_twist_k8` | 8 partial teams per final team; weight by kept share after each pass, resample below 4 effective | Between as-drawn and rejection | Same as above | 8 ×: about 80 min per row (32 particles: about 5 h) |
| Two-top-values gap order | `order_margin` | Fill next the field whose top two values are furthest apart | Tilt near 0 if species come first, as in random order if moves do | A second tilt-free order matching species-first would show the tilt, not the order, decides | 1 ×: about 10 min per row |
| Position temperature 1 on confidence order | `order_conf_tpos1` | Choose the next field in proportion to top-value confidence (Two Temperatures' TLC) | Tilt between species-first and random order | Tilt and mean win rate should move together across the three orders; if not, the tilt is not the cause | 1 ×: about 10 min per row |

The first two test this reading's claim directly. Twisted SMC is the cheaper check, and rejection is the exact one.

## 9. Relevance check

| Paper | Fit to the 84-field model | Needs an LLM? | Small-model experiments | Own code (opened by hand) |
| --- | --- | --- | --- | --- |
| Dang et al. 2606.01926 | Closest. Its three samplers are ours: mask what is illegal now (plain), a mask that knows whether the rule can still be met (our lookahead), and a proposal that knows how likely it is (P-GCD) | No in principle; P-GCD's HMM is distilled from model samples, which our model supplies | No (Llama-3.1-8B, GPT-2 Large) | `MhDang/gelatwo` @ `d974a70b9c` |
| Loula et al. 2504.13139 | Machinery: SMC with potentials over any left-to-right generator, for the twisted-SMC arm | No in principle; all experiments use Llama 1B to 70B | No | `probcomp/genlm-control` @ `e8e39c2040` |
| Grammar-Aligned Decoding 2405.21047 | Idea: learn from samples how likely each prefix is to end legal, instead of a 0-or-1 lookahead | Autoregressive only | No (Mistral-7B) | `ebmoon/transformers-GAD` @ `bc062c68e0` |
| TWISTER 2609.35609 | Theory carries over exactly; the automaton machinery does not (our rules are a finite check over 84 fields, not a token automaton) | No | Smallest is OWT-130M | None |
| Mündler et al., LAVE, DConstrain, DINGO, Dang & Ermon | Low: text grammars, token automata and JSON schemas | Yes as built (7-8B diffusion LLMs) | No | Mündler and LAVE yes; the rest none |

One Haiku reader reported that Dang et al. link no code. The paper links `gelatwo` in a footnote of its introduction (`sections/1.introduction.tex:17`), corrected by hand.

## 10. Vocabulary and proposed glossary entries

| Term | Plain definition | Figure |
| --- | --- | --- |
| Constrained decoding | Drawing outputs while forcing them to obey a rule; here, a legal Showdown team | 01 |
| Tilt (trajectory bias) | How far the sampler's chance of a team drifts from the model's own chance among legal teams | 02 |
| Correction weight | The model's probability of the team over the sampler's, along the same draws | 02, 05 |
| Removed mass | Probability the sampler took away at one draw before renormalising | 03 |
| Step-exact decoder | Draws all fields revealed at a step from the model, restricted exactly to completable values | 01 |
| Twisted SMC | Many partial teams drawn side by side, each reweighted every step by a correction factor (the twist) | 05 |
| Effective sample size | How many equally weighted teams a weighted set is worth | 05 |
| Doob target | The model's own sampler conditioned on ending legal: what rejection sampling draws | 04 |
| Total variation distance | The largest gap in probability two distributions give any set of outcomes | 01 |
| Paired bootstrap interval | Range from redrawing opponents and teams together for both rows, so shared luck cancels | 05 |

Proposed `notes/GLOSSARY.md` entries (you decide which go in; the existing **Lookahead** and **Dead end** entries already cover those two):

```
**Tilt**:
How far a sampler's chance of drawing a team drifts from the trained model's own chance among legal teams, because legality rules removed probability while drawing. Random order with the lookahead: median 13.6 nats per team (linear schedule) and 15.2 (per-field), guidance 2; species-first order: 0.
_Origin_: Thimmaiah et al. 2026, "Twist, Don't Tilt", §3.4 (Proposition Re-Freezing Tilt), where it is called trajectory bias.
_Avoid_: trajectory bias (the paper's name; "bias" already means other things here)

**Correction weight**:
A team's probability under the trained model's own per-pass distributions over its probability under the sampler, along the same draws: the product over draws of the kept share. Under the full weights about 2% of random-order teams are effective.
_Origin_: Thimmaiah et al. 2026, §4.1 (Feynman-Kac potentials); this project's weights: `code/scripts/twister_reading/tilt_weights.py`, 2026-11-15.

**Removed mass**:
The probability a legality rule takes away at one draw before the sampler renormalises. Random order loses most at moves (8.8 nats per team) and abilities (5.4), per-field schedule, guidance 2.
_Origin_: this project, 2026-11-15; the paper's clamped partition sum (§3.2) is the kept share's counterpart.

**Step-exact decoder**:
A sampler that draws all the fields revealed at a step from the model's distributions, restricted exactly to values that can still end legal. The **Lookahead** is a partial version.
_Origin_: Dang & Ermon 2026 (arXiv 2607.07026); named in Thimmaiah et al. 2026, §3.2.

**Twisted sequential Monte Carlo**:
Draw many partial teams side by side; after each step reweight each by a correction factor (the twist) and resample, so the finished set follows the target. Not run here.
_Origin_: Thimmaiah et al. 2026, §2.3 and §4.2.
```

## Files

- Report page: `code/docs/figures/twister_reading/index.html` (built by `code/scripts/twister_reading/build_page.py` from `figures.py` and `page_template.html`), one chart page per figure beside it.
- Scripts: `tilt_weights.py` (redraw and weights, sampling only), `tilt_analysis.py` (reweighting and bootstrap), `tilt_exact_toy.py` (exact one-Pokémon test), `base_law.py` and `tables_tilt.py` (the tables' cut).
- Storyboard: `code/docs/figures/twister_reading/storyboard.md` (one frame per figure, with its sentence, expected result and surprise).
- Data: `~/vgc-data/results/twister_reading/` (one JSON per row and guidance, the run log, `tilt_analysis.json`, `predictions.md`, `tilt_exact_toy.json`, `tables_base/`). The page builds from `page_data/` there: `build_page.py ~/vgc-data/results/twister_reading/page_data code/docs/figures/twister_reading`. Fresh-team removed mass: the other session's `~/vgc-data/results/legality_skew.json`.
- Browser check: `verify-evidence/figure-pages/2026-11-15-2002-twister/load4/load.json`, light and dark at 1440 and 390 px, 7/7 charts, no errors, no sideways scroll.
