# Does remasking close the ask gap?

Pre-registered 2026-10-30, before any experiment sampling. Script: `src/remask_ask_gap.py`. Sampler: `DiscreteDiffusionModel.sample(remask=, steps=)` in `src/discrete_diffusion_model.py`. Results: `~/vgc-data/results/remask_ask_gap.json`.

## Question

The from-scratch discrete diffusion model trained only on valid Pokémon teams falls short of the win rate it is asked for. On the guidance-legality grid (`~/vgc-data/results/guidance_legality.json`, 2026-10-29), teams asked for a win rate of 0.5, at classifier-free guidance weight 2, won 0.452, 0.450 and 0.438 of their battles against the top-50 meta teams (three seeds). Decoding order also changes what the model picks. Both point to the same cause: once the sampler unmasks a field, nothing can change it. Remasking (ReMDM; Wang, Schiff et al., 2025) masks a filled field again and redraws it under the ask. Does that bring the achieved win rate closer to the ask?

The ask is the win rate the model is conditioned on. The gap is |ask − achieved win rate|. Win rate is the fraction of battles a team wins against the top-50 meta teams, with the behaviour-cloning policy playing both sides. The top-50 list holds 49 distinct teams: one team is listed twice and counts double in every score.

## The sampler

The discrete diffusion model fills the 48 fields of a team (six Pokémon × species, ability, item, four moves, Stat Alignment) in a fixed order: all six species, then abilities and items, then moves, then Stat Alignments. The current sampler makes one network pass per field, 48 passes in all.

The remasking option keeps that order for new fields and adds one step before every pass after the first:

1. For each team, count its filled fields and multiply by the remasking fraction σ. Round stochastically, so 0.1 × 23 fields masks 2 fields with probability 0.7 and 3 with probability 0.3.
2. Mask that many filled fields, choosing the ones drawn with the lowest probability. A field's confidence is the probability its value had in the distribution it was drawn from: guided, tempered and restricted to legal values.
3. The pass then redraws the masked fields under the ask, together with the pass's new field. It draws them one at a time in the fixed order, so the legal Pokémon construction tables see every earlier draw of the pass. The network sees none of them: its log-probabilities come from the team as it stood when the pass began. A redrawn field keeps the probability of its new value.

The number of network passes stays at 48, so the remasking arms get no extra network compute; only the cheap per-field draws multiply. The tests in `scripts/test_discrete_diffusion_model.py` check that σ = 0 reproduces the current sampler bit for bit, that remasking keeps the pass count, and that every remasked team satisfies every legal Pokémon construction table and passes Showdown's validator.

### Where this differs from ReMDM

- ReMDM remasks each decoded token with probability σ_t (its Eq. 4). Its confidence schedule spreads that probability across tokens by a softmax over negative confidence (Alg. 2). Here, as specified, each team masks a fixed fraction of its filled fields, the least likely ones first.
- ReMDM returns a remasked token to the masked pool, and a later step unmasks it on the schedule. Here the pass that masks a field also redraws it, because the fixed order already decides which new field each pass fills.
- ReMDM's inference-time scaling holds the mask rate fixed for the extra steps (its loop phase, Alg. 3). The 96-pass arm here spreads the 48 new fields over 96 passes, one new field every second pass, and remasks before every pass after the first. A pass with nothing to draw (no new field and nothing remasked in any team) is skipped; at 96 teams per batch that is rare, and the results record passes per batch.
- The paper reports lower sample entropy with remasking than without at every step count it tried (its Table 1), so a diversity drop here would match the paper.

## Design

- **Model:** `DiscreteDiffusionModel.load()`, an ensemble of the loop-teams model (`F1_final_medium_noheldout_regmb_v3`) and the battle-records model (`G2_family_matrix_noprtrain_noheldout_regmb_v3`). Neither saw the held-out teams (meta ranks 51–100) in training. Every arm uses the same checkpoint.
- **Sampling, as the loop samples:** classifier-free guidance 2 (the loop's weight, `cem_dbbo.py`), temperature 1, legal Pokémon construction filter on, no surrogate tilt, `to_paste(repair=True)`.
- **Arms:**
  - plain masking: the current sampler, σ = 0, 48 passes;
  - remasking 5%, 10% and 20%: σ = 0.05, 0.1 and 0.2 at 48 passes;
  - remasking 10% at 96 passes: the inference-scaling check.
- **Asks:** 0.3, 0.5 and 0.531, the elite cut (the mean win rate of a top-50 meta team against the other 49).
- **Seeds:** 1, 2 and 3 at base 7300, shared by every arm. Plain masking at ask 0.5 therefore redraws the guidance-legality grid's guidance-2 teams paste for paste, and reuses their battles from the matchup database; the report checks that both the pastes and their scores match.
- **Per cell (one arm at one ask and one seed):** draw until 64 teams pass Showdown's validator and are not held-out teams (cap: 1,280 tries).
- **Battles:** each team once against each of the 49 distinct top-50 meta teams (49 battles), behaviour-cloning policy on both sides, the six Pokémon shuffled every battle. This is the guidance-legality grid's battle count.
- **Best fresh team:** for each arm and ask, the four best distinct teams by their 49-battle score, three seeds pooled, re-battled in a private matchup database: 4 battles against each of the 49 distinct top-50 meta teams, 196 per team. Distinct means no two finalists share all 48 fields. None of the battles that selected them is reused.

## Measures, per arm and ask

- Legality: Showdown-accepted tries / all tries.
- Mean win rate per seed, and the gap |ask − win rate|.
- Best fresh team: the highest fresh win rate among the four finalists, and its six species.
- Diversity of each cell's 64 teams: distinct line-ups (sorted six-species sets) and species entropy (Shannon entropy in nats of species frequency over all 384 Pokémon). Also reported, outside the rule: effective species (exponential of the entropy), distinct species, the Vendi score (the effective number of distinct teams, with similarity K = shared species / 6), and mean slot-matched distance (fields out of 48 that differ after pairing the Pokémon to differ least) (`src/team_diversity.py`).
- Redraws: fields remasked per team, and fields whose redraw changed the value, by field type.

## Pass rule

Seed noise at an ask is the larger of two estimates of how much plain masking's mean win rate moves between seeds: the standard deviation across its three seeds, and the mean standard error of a seed's 64-team mean. The second guards against three seeds landing close together by luck.

Narrowing for one seed is plain masking's gap minus the arm's gap, paired by seed. At each ask:

- If plain masking's mean gap exceeds twice the seed noise, the arm must narrow the gap in all three seeds, by a mean of more than twice the seed noise.
- If it doesn't, there is no gap to close, and the arm must not widen the gap by more than twice the seed noise.
- Legality, distinct line-ups and species entropy must not fall below plain masking's mean by more than twice their own seed standard deviation. The smallest drop that counts is 1 percentage point of legality, 1 line-up or 0.01 nats, because plain masking is 100% legal with no spread.

An arm passes if, at every ask, it meets the gap condition that applies there (narrow it, or don't open one) and lowers none of legality, distinct line-ups and species entropy. Remasking passes if remasking 5%, 10% or 20% passes. The 96-pass arm spends twice the compute, so it answers whether more passes help and cannot make remasking pass on its own. If no fixed-compute arm passes, remasking is recorded as a null result and plain masking stays.

## Expectation, written before the run

A probe of 96 teams per setting, before the experiment, counted which fields remasking redraws. At 10%, a team has about 113 fields remasked over the decode, three quarters of them items, and about 2 species change per team. Item columns carry the flattest distributions, so they hold the lowest-confidence fields. Species are fixed first, and once a Pokémon's moves and ability are visible, the legal Pokémon construction tables usually let its species come back only as itself. On the 2026-09-28 ruggedness test (`docs/ruggedness2.md`), a single species edit moved win rate beyond battle noise, while the other edit types, items and moves included, stayed within it. So remasking mostly redraws fields whose single edits barely move win rate.

I expect a small win-rate change at best, and no pass. Mean slot-matched distance may fall, because redrawing the least likely items under guidance 2 pulls them toward the most common item.

## Commands

```
/tmp/vgc-pilot/.venv/bin/python src/remask_ask_gap.py gen
/tmp/vgc-pilot/.venv/bin/python src/battle_supervisor.py src/remask_ask_gap.py battle --then fresh --then report
RA_SMOKE=1 /tmp/vgc-pilot/.venv/bin/python src/remask_ask_gap.py gen    # plumbing check, own files
```

The run itself uses a frozen copy of `remask_ask_gap.py`, `team_writer.py` (now `src/discrete_diffusion_model.py`) and `team_diversity.py` in `~/vgc-data/remask_ask_gap/run/`. The script refuses to start if it loads any other `team_writer.py`.

## Result (2026-10-30)

**Null under the pre-registered rule: no arm passes, so plain masking stays.** Every remasking arm lowers distinct line-ups and species entropy by more than the allowed margin at every ask. Legality stayed at 100% in all 45 cells. Plain masking at ask 0.5 reproduced the guidance-legality grid's teams and scores exactly.

Win rate is against the top-50 meta teams, 49 battles per team, as mean ± standard deviation over the three seeds. Line-ups are the distinct six-species sets among each seed's 64 teams, averaged over seeds.

| Arm | Ask 0.3: win rate | Ask 0.5: win rate | Ask 0.531: win rate | Line-ups at 0.3 / 0.5 / 0.531 |
|---|---|---|---|---|
| Plain masking | 0.337 ± 0.024 | 0.447 ± 0.008 | 0.404 ± 0.005 | 48.0 / 22.7 / 27.0 |
| Remasking 5% | 0.456 ± 0.006 | 0.506 ± 0.009 | 0.516 ± 0.008 | 30.7 / 10.0 / 12.7 |
| Remasking 10% | 0.470 ± 0.015 | 0.502 ± 0.006 | 0.523 ± 0.016 | 21.7 / 7.3 / 9.7 |
| Remasking 20% | 0.487 ± 0.010 | 0.523 ± 0.011 | 0.514 ± 0.020 | 15.3 / 7.0 / 10.7 |
| Remasking 10%, 96 passes | 0.495 ± 0.004 | 0.511 ± 0.010 | 0.516 ± 0.012 | 17.3 / 6.0 / 8.0 |

- At asks 0.5 and 0.531, every remasking arm narrows the gap in every seed. The mean narrowing is 0.030 to 0.048 at ask 0.5 and 0.110 to 0.114 at ask 0.531, against seed noise of 0.013 and 0.015.
- Remasking does not follow the ask better. Remasked teams land near 0.46 to 0.52 whatever the ask. At ask 0.3, where plain masking was already close (gap 0.037), remasking overshoots and widens the gap to 0.156 to 0.195.
- The 96-pass arm is no better than remasking 10% at 48 passes, and has the fewest line-ups at asks 0.5 and 0.531.
- Best fresh team (196 fresh battles, standard error about 0.03): 0.500, 0.590 and 0.605 for plain masking at asks 0.3, 0.5 and 0.531, against 0.520 to 0.655 for the remasking arms. Eleven of the fifteen best fresh teams share one line-up (Basculegion, Charizard, Floette-Eternal, Garchomp, Kingambit, Whimsicott) and differ only in their sets.

Outside the rule (exploratory): at ask 0.5, remasking 5% at guidance 2 won 0.506 with 10.0 line-ups and 8.2 effective species. Plain masking at guidance 4 on the guidance-legality grid won 0.460 with 11.0 line-ups and 7.5 effective species. At about the same species diversity, remasking bought about 0.05 more win rate than doubling the guidance weight.
