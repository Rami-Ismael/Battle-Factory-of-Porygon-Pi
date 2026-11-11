# Why every subset-size model stops near a 0.5 win rate: guidance sweep and strong training teams

Pre-registration, 2026-11-11. Follows [Calibrated asks](calibrated-asks-subset-labels.md) (#282, refuted). The storyboard on #118 showed every subset-size model, the all-39 control included, stopping near a 0.5 win rate against the 39 training opponents whatever the ask. Owner, 2026-11-11, replacing the second-seed and ±1/48-noise tests: "Instead focus on Turn the guidance up and Ask the model directly for strong teams". Mode: **exploratory**, one training seed per model.

## Guidance sweep (#284)

| Row | Value |
| --- | --- |
| Claim | Stronger guidance raises the best got of the control and the 1 to 39 model over asks 12/24, 16/24, 20/24 and 23/24. **Raises the stop** if the best got at guidance 8 is 0.05 or more above guidance 2's (control 0.500, 1 to 39 0.462 on these asks). **No change** if within 0.02. Otherwise undecided. If it raises the stop, guidance becomes a per-model setting to tune; if not, the stop is in what the model learned. Scope: win rate under the behaviour-cloning policy against the 39 training opponents. |
| Arms | Control `gen65_subset_39` and 1 to 39 opponents per label `gen65_subset_1_39`, each at guidance 2 (stored grid), 4 and 8. |
| Fields | All 84 generated; Stat Alignment repaired after sampling, as in every grid. |
| Vocabulary, training teams | Unchanged checkpoints; no training. |
| Scorer | `ask_grid84.draw` at the arm's guidance, temperature 1, 64 legal teams, at most 1,280 tries, sampling seed 7300 + seed; one battle against each of the 49 meta teams, slots shuffled; got split to the 39 training opponents with `train65.got_by_side`. |
| Settings | **Scientific:** guidance. **Fixed:** checkpoint, asks, seeds, sampler, scorer. |
| Plan | **Primary:** best got over the four asks per guidance. **Beside:** got per ask with standard error, legality, distinct line-ups, share of teams shared with guidance 2 at the same ask and seed. **Stopping rule:** one grid per arm. |
| Budget | 2 models × 2 guidances × 4 asks × 3 seeds × 64 = 3,072 teams, 150,528 battles on the laptop. Per ask, got has a standard error of about 0.008 (192 teams), so 0.05 is about six standard errors. |

## Strong training teams (#285)

| Row | Value |
| --- | --- |
| Claim | At asks 18/24 to 23/24, generated teams share more slots with the 868 training teams whose all-39 label is 0.7 or more than with 868 uniformly drawn training teams. **Resembles strong teams** if the 95% interval of the per-team difference in most-shared slots sits above +0.25 slots. **No nearer** if it sits inside ±0.25. Otherwise undecided. If they resemble strong teams yet win about 0.45, the strong teams' labels or battles are the next question; if not, the fix is in training. |
| Measure | A slot is species, ability, item and the set of moves. For each generated team, the most slots it shares with any team in each reference set, and whether its six-species line-up appears there. Bootstrap over generated teams, 2,000 draws, seed 8500. |
| Teams | The stored grids of the control, 10 to 39 and 1 to 39 models: 1,152 teams per model. No new battles. |

## Probes

| Probe | Result | Pass |
| --- | --- | --- |
| Guidance smoke | 4 teams at guidance 8, ask 20/24, control: 100% legal, battled in a private copy of the preflight database | yes |
| Guidance split report | With no new cells, the split reproduces each model's stored got at asks 12/24, 16/24, 20/24 and 23/24 exactly | yes |
| Strong-teams control | At asks 1/24 to 6/24 the difference reverses (control −0.24, 1 to 39 −0.27 slots): low asks give teams nearer ordinary teams, so the measure moves with the ask | yes |
| Training smoke, round trip, coverage, separation | Same checkpoints as the subset-size arms, whose preflights passed | carried |

## Deviations

- 2026-11-11: the strong-teams analysis ran on the control before this doc was written. Its verdict rule and thresholds were already fixed in `strong_teams65.py` before that first run and did not change; the low-ask probe was added after.

## Commands

```
AG84_MODEL=<model> python src/strong_teams65.py ; ST65_ASKS=1-6 ... (probe)
AG84_MODEL=<model> python src/guidance_sweep65.py gen ; battle_supervisor.py src/guidance_sweep65.py battle --then split
```

Queued as `~/vgc-data/subset_ceiling/gs65.zsh` from a frozen copy in `~/vgc-data/subset_ceiling/run/`.
