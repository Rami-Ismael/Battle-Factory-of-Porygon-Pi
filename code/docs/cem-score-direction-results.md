# Reversing CEM ranking: experiment results

Completed on 2026-09-09 UTC: three seeds, five cumulative CEM updates per arm, 19,776 raw generation attempts, 103,680 battles, and 2,880 optimizer steps.

Reversed CEM ranked candidates using **−score**, while retaining the original measured scores and ordinary diffusion denoising loss. The starting checkpoint, sampling temperature, training budget, battle policy, and opponent pool were matched. A third arm selected elites uniformly at random.

The reversed arm retained more Pokémon-ID diversity than normal CEM in each of the three observed seeds. Its mean final diversity was **12.71 percentage points higher**. The random-elite control had an almost identical sample mean to the reversed arm. Both lost diversity relative to the initial generator, and reversed CEM produced a lower fresh battle win rate.

| Final generator | Pokémon-ID diversity | Distinct legal ID rosters per 256 raw attempts* | Fresh win rate |
|---|---:|---:|---:|
| Normal CEM | 66.71% | 238.0 | 9.53% |
| Random elites | 79.47% | 246.7 | 1.00% |
| Reversed CEM | 79.42% | 248.3 | 0.41% |

*Counts and rates are means across the three seeds. Final legality was 97.92%, 96.35%, and 97.01%, respectively. Distinct-roster counts therefore describe the legal portion of the fixed raw-attempt budget.

The diversity percentage is the average fraction of six Pokémon IDs that must be replaced between two legal generated teams: identical rosters score 0%, one replacement scores 16.67%, and disjoint rosters score 100%. It ignores slot order, EVs, IVs, nature, ability, moves, and items. Different form IDs remain distinct. It is not the percentage of unique teams.

Initial mean diversity was **85.12%**. After five updates, normal CEM had lost 18.41 points, random elites 5.65 points, and reversed CEM 5.70 points. Sign reversal therefore did not preserve the initial level of diversity in this experiment.

The adjusted 95% paired intervals were **[−18.37, 43.78] points** for reversed minus normal and **[−8.59, 8.48] points** for reversed minus random. These three-seed, exploratory intervals assume normally distributed seed effects and adjust for two comparisons. They are too wide to establish a reliable advantage over normal CEM or equivalence with random selection. The observed means support investigating diversity retention further; they do not establish sign reversal as a dependable diversity-preservation method.

Fresh quality evaluation used 32 uniformly selected legal holdout outputs per arm/seed, with 48 new battles each. Those scores were excluded from training. The very low absolute win rates apply to this particular fixed battle policy and opponent pool.

The training corpus contained 692 six-member teams, 570 distinct ID rosters, and 189 distinct Pokémon IDs. Its ID diversity was 84.67%. The generator remains limited by this vocabulary, but the starting model already had substantially broader ID coverage and diversity than the final normal-CEM generators. See the [corpus context](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/cem_score_direction.corpus-context.json).

The independent audit revalidated all 19,776 proposals, explicitly checked 4,273,140 team pairs, and verified all 48 battle phases, score signs, checkpoint chains, budgets, and summary calculations. No battle phase required a retry. The eight experiment-owned servers were stopped afterward.

**Source-freeze deviation:** Three shared source files changed during the uninterrupted run. Review found that the changes affect two unimported experiment modules and an unused proposal helper; the executed sampler, selection, training, battle, and metric code are unchanged. The original controller completed every experimental phase and then failed its strict final hash check. Its original raw status and error remain preserved. The independent audit passed with this explicitly reviewed deviation, not as a clean strict source-freeze pass. [Review details](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/cem-source-deviation-review.md).

[Full report and trajectories](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/cem_score_direction.md) · [Verification receipt](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/cem_score_direction.verification.json) · [Protocol](#protocol)

## Protocol

Locked before the run; merged here from `cem-score-direction-protocol.md` on 2026-11-16.

Question: does ranking on negative measured score preserve more Pokemon-ID
diversity through cumulative CEM updates than ordinary CEM?

### Fixed first experiment

- Arms: normal CEM ranks by `score`; reversed CEM ranks by `-score`; the random
  control selects a uniform subset. Stored scores remain wins divided by battles.
- Shared starting checkpoint: `results/temperature_p0.pt`, trained on the same
  692-team corpus. Three new trajectory seeds: 2101, 2202, 2303.
- Five cumulative updates. Each update starts from that arm's preceding
  checkpoint, with a fresh optimizer. No return to p0, replay pool, or anchors.
- Each selection batch contains 192 raw attempts. Select 96 legal attempts
  uniformly without replacement for evaluation, keeping duplicate teams as
  distinct draws. If fewer than 96 are legal, stop that trajectory as infeasible;
  do not refill or discard the failure.
- Each evaluated candidate receives 24 fresh battles using the same fixed BC
  policy on both sides and a shared 24-opponent schedule from the frozen top-50
  pool. Schedules and policy seeds match across arms within each seed/iteration.
  Showdown's internal battle RNG is not seeded by the policy RNG.
- Retain 20 candidates: the existing 20% rule with a minimum of 20 at pool size
  96. Shuffle before stable ranking to randomize ties. Random control retains
  20 using the same shuffle. Only the ranking score sign changes for reversal.
- Every update uses 64 optimizer steps, batch size 64 sampled with replacement,
  AdamW with initial learning rate 1e-4, cosine schedule, weight decay .01,
  gradient clipping 1.0, and random slot permutations. Use the same masked
  denoising loss, null conditioning, and zero additional entropy bonus in all arms.
- Generation always has decoding temperature 1.0 and no score guidance. Species,
  legality constraints, and stat-spread sampling remain fixed.
- After every update, draw 256 fresh raw evaluation attempts with an independent
  phase seed. These never enter selection or training. Preserve repeats and
  invalid outputs. Each seed also has a shared 256-attempt p0 baseline.
- The first candidate batch and its fresh battle results are shared across arms
  within a seed, since the starting model is identical. Later batches come from
  the respective updated generators. Rotate arm execution order over iterations.
- At iteration five, uniformly select 32 legal evaluation outputs per arm/seed
  for 48 fresh battles each. These secondary quality results never affect training.
  If fewer than 32 are legal, record quality evaluation as unavailable.

There are at most 19,776 physical raw generation attempts, 103,680 fresh battles,
and 2,880 optimizer steps. Shared first-round battles are performed once and
exposed equally to all arms. Infrastructure retries are recorded separately;
completed candidate labels are reused on resume.

### Measurements and interpretation

The primary endpoint is Pokemon-ID diversity among legal evaluation outputs at
iteration five: `100 * mean_pair(6 - shared_ID_count) / 6`. Slot order and all
non-ID attributes are ignored. Distinct form IDs remain distinct. Repeated
proposals remain in the calculation. Report the same metric on all raw outputs,
legality, distinct rosters, effective IDs, and ID coverage alongside it.

Show every iteration and the initial baseline. Report paired final differences
against normal CEM and the random control, in percentage points. Exploratory
three-seed t intervals use Bonferroni adjustment for those two comparisons and
assume normally distributed seed effects. Trajectory measurements and secondary
quality results are descriptive. Do not interpret individual generated teams as
independent training replicates.

An infeasible trajectory retains all completed results and prevents a full
three-seed final comparison for the affected arm; do not silently drop its seed.
Conclusions apply to this checkpoint, representation, policy, budgets, and five
updates. A ranking-pool result alone does not establish generated diversity.

### Reproduction

The runner is `scripts/cem_score_direction.py`; `--smoke` uses a separate output,
one seed, two iterations, and reduced budgets to test cumulative updates and
fresh battles. Run the full experiment with the stable runtime Python and eight
isolated local Showdown servers on ports 8134 through 8141. Input fingerprints
and per-phase checkpoints protect resume. The worker does not own those servers;
their launcher records ownership for cleanup.
