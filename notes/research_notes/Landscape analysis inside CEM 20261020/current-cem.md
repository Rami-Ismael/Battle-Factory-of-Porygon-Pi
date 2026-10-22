# Current CEM implementation audit

Checked October 20, 2026. This is a read-only audit for explaining landscape analysis inside the current search. No battle, training job, code change, database write or new budget was started.

## Source and version boundaries

The code repository is `/Users/ramiismael/Documents/code/vgc-team-generator-pilot`, at Git revision `8a7268da2cb290045c9257b504650f9cabc29def` when checked. The runtime staging directory `/tmp/vgc-pilot/src` is a separate copy.

The copies of `cem_v3.py` and `foundation.py` differ only in the checkpoint-loading refactor. The copies of `matchup_db.py` are identical. The runtime `cem_dbbo.py` also has functional batch-two ablation controls absent from the repository copy: `CEM_NO_TABOO`, `CEM_TEMP_SETS` and `CEM_CONTINUE`. Do not call the two copies interchangeable. No matching live CEM or battle process was found in the process inventory at inspection time.

There are several completed arms, not one universal configuration. Use an explicitly named saved run when illustrating numbers. The clean `cem_v3_noheldout` run is a useful baseline with complete proposal texts. Later full-512 ablations and precise-label arms are different configurations. A saved result's configuration is stronger evidence of its settings than today's module defaults.

## What the clean CEM loop actually does

Primary code: [cem_v3.py](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/cem_v3.py:33). Protocol and recorded results: [cem-v3-loop.md](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/cem-v3-loop.md:12).

| Stage | Verified behavior | Source |
| --- | --- | --- |
| Representation and proposal distribution | A neural ensemble generates six slots with eight categorical model fields per slot. These include species, ability, item, four moves and nature/alignment. Stat-point/EV spreads are drawn separately from species-specific corpus spreads, or randomly when absent. This is not an independent categorical-frequency CEM table. | `src/foundation.py:132–193`, `src/encode.py`, `docs/species-guidance-loop.md:7–8` |
| Model update | For generation greater than zero, fine-tune copies of the original two v3 member checkpoints on the selected elites. Default is 40 epochs, learning rate 1e-4, batch 64. The default does not continue training the previous generation's weights. | `src/cem_v3.py:113–139` |
| Ridge update | Refit on base labels plus all previous non-anchor loop labels weighted 10. Feature groups include species, ability, item, nature and moves, plus species-pair features. The generator's EV spread draw is not represented in this ridge feature map. | `src/cem_v3.py:63–108`, `:217–221`; `src/foundation.py:88–128` |
| Propose | Generate 512 validator-accepted teams. Default ask 0.5, classifier-free guidance 2, temperature 1, ridge tilt during decoding, nature repair, rejection of held-out teams. Attempt cap is 8 times the proposal target. | `src/cem_v3.py:223–244` |
| Acquire | Select the 128 largest ridge predictions to battle. This top-128 acquisition step is not the elite rule for fine-tuning. | `src/cem_v3.py:245–248` |
| Score | Add/canonicalize teams, validate, and top up missing matrix cells against the top-50 set to at least one battle per cell. Score is the weighted mean of per-opponent win rates, equal weights by default. The set has 49 distinct teams in these saved runs. | `src/cem_v3.py:249–255`; `src/lossdown_battle.py:40`; `src/matchup_db.py:375–443` |
| Select elites | Default cumulative pool contains anchors and every previously battled generation. Select labels at least 0.531; if fewer than 64 qualify, take the top 64. The absolute bar is the mean meta-team benchmark, not a top fraction of the current generation. | `src/cem_v3.py:159–185`, `:206–215` |
| Stop | Copy rate over 10%, distinct species sets below half that run's generation-zero count, saturation below +0.01 over three generations, or at most ten generations. | `src/cem_v3.py:44`, `:269–280` |
| Final evaluation | Select the 16 best distinct battled teams by loop label, then score each with 192 fresh battles against the top-50 schedule and separately against ranks 51–100. Held-out results do not feed back into the loop. | `src/cem_v3.py:283–318` |

The base seed is 4040. Proposal seeds are `SEED + 100*g`, fine-tuning uses `SEED + g`, and matrix requests pass `SEED + g`. The seed is overrideable. Equal seed labels alone do not prove identical random battle realizations for different candidate jobs.

There is no explicit CEM alpha, probability floor or convex blend of old and elite categorical frequency tables in this implementation. Softmax temperature, guidance, fine-tuning length, the selected training pool and possible sampling mixtures are the relevant mechanisms. A proposal to blend original and fine-tuned generators would be a new sampling mechanism, not a description of existing code.

The `topfrac` variant exists. The saved `results/cem_v3_topfrac.json` records four generations, the previous generation's top 20%, no anchors in that training set, ten epochs, seed 4040, with generation zero shared from the clean baseline. It should not silently replace the cumulative baseline in an explanation.

## The latest family of ablation configurations

[cem_dbbo.py](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/cem_dbbo.py:55) contains fixed and moving asks, elite-only versus all-label training, precise-label members, precise elite re-evaluation, random versus ridge acquisition, and full-512 battle options.

The default decoder proposes three quarters at a main ask and one quarter at an ask lower by 0.2, with temperature 1 on species and 0.5 on other fields. A composition taboo rejects earlier or repeated species compositions according to its existing cap. Default acquisition keeps the corresponding 96 and 32 candidates with highest ridge prediction; `CEM_BATTLE=512` battles every retained proposal. `CEM_TRAIN=elite` uses the cumulative raw-label rule described above. All-label training instead uses shrinkage and replay. See `src/cem_dbbo.py:63–85`, `:333–366`, `:437–538`.

The original three-seed ablation reference is fixed ask 0.5, elite re-steering, all 512 proposals battled, with seeds 6060, 7070 and 8080. Its generation-zero files are shared by seed, after which arms diverge. It runs generation zero plus three later generations. Those settings belong to that comparison, not every future landscape experiment. `src/cem_dbbo.py:403–421`, `:667–714`; result files `results/abl_base_s6060.json`, `abl_base_s7070.json` and `abl_base_s8080.json`.

The current summary `results/cem_ablations.json` records the higher cut 0.56 arm as the one passing its stated test: fresh top-50 finalist difference +0.029406, interval [0.005635, 0.053275], per-seed differences +0.044269, +0.014969 and +0.028981, `matters: true`. Its composition difference is about -2.33. The other entries, no ask, frozen generator, cut 0.50, ten epochs and no anchors, have `matters: false`. This is recorded evidence for that arm and comparison; it is not proof of owner adoption or a reason to redefine the current baseline. It also shows why more distinct compositions cannot substitute for win rate.

## Existing diversity intervention already tested

[Species guidance inside CEM](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/species-guidance-loop.md:19) changes only species guidance from 2 to 1.5 in the clean loop, paired across seeds 4041, 4042 and 4043. The saved summary is `results/cem_v3_species15_loop.json`, and its per-run files retain full proposals.

The intervention failed its preregistered diversity requirement in seed 4041. The generation-zero width gain largely disappeared after elite fine-tuning. The fresh finalist difference was +0.002, interval [-0.019, +0.024]. All six runs stopped on the species-set guardrail. This is evidence that the sampler can be broad before refitting and narrow after refitting, not evidence that species guidance 1.5 solves collapse throughout CEM. Sources: `docs/species-guidance-loop.md:110–146`; JSON `verdict` and `runs`.

## Provenance and uncertainty traps

1. **Cache reads are not independent re-evaluations.** `matchup_db.ensure` adds only the deficit needed to reach `per_cell`. Repeating the same target can run zero battles. Scores may also use more than the nominal minimum if another process previously topped up the same cells. Account for actual new battles and historical cache provenance separately.
2. **Slot-order mode changed.** `src/slot_order.py:1–13,23–26` defaults to shuffled Pokémon order. Listed order is available for reproduction. `matchup_db.get_policy:176–194` tags shuffled policy configurations separately; `ensure:375–389` checks compatibility. Historical policy-1 runs and a new shuffled-order run must not be combined as one objective without verification. Parent's smoothness audit should recover the old evaluator provenance.
3. **Per-cell plug-in SE can degenerate.** `matchup_db.score:439–443` sums `p*(1-p)/n` per cell. At one Bernoulli outcome per opponent, every cell's estimated variance is zero. A returned SE of zero at that target is not evidence that the team's expected win rate is known exactly. Use appropriate repeated/paired evidence before asserting local improvements or neutrality.
4. **Different summaries mean different uncertainty.** CEM's printed generation SE is across evaluated team scores, not the uncertainty of a single team's score or a paired edit effect. The final independent evaluations address selection optimism separately.

## What is already logged

The clean baseline `results/cem_v3_noheldout.json`, and the `sp15loop_*` runs, retain each generation's full `proposals` pastes, `mu`, `battled`, `battled_mu`, `team_id`, `y`, elite summaries, distinct species sets, copies and timing. Later species-guidance runs also have Vendi and elite set counts. They do not record a mutation parent or a local-edit operator for each proposal.

The `cem_dbbo` family retains all proposal species compositions in `stream_comps`, while full pastes are retained for battled candidates. It also records asks, selected ridge predictions, battle counts, shrinkage, generation timings, diversity and delivered fresh evaluations. An unbattled composition is not a complete team. One cannot reconstruct exact full-set distances for unbattled candidates from composition-only logs.

A CEM generation is a batch sampled from an updated distribution. There is no natural single parent for every team. Connecting each proposal to its nearest prior team creates a retrospective proximity graph, not an observed parent-child search trajectory. Label that distinction. An actual STN of local mutations needs logged transitions from a local operator.

## Concrete landscape insertion points

| Point in loop | Useful diagnostic or proposed action | What remains unproved |
| --- | --- | --- |
| After proposals, before ridge acquisition | Measure exact repeats and distances to prior proposals/elites. Separate species-composition overlap from full-set edits. Compare the proposed batch with the acquired batch. | A fall in species-set counts does not itself prove poor fitness-landscape exploration. |
| After battle labels | Examine local observed score differences and ridge residuals for nearby fully specified teams, matched by evaluator. Use comparisons with uncertainty. | Nearby teams in existing data were not randomized local mutations. Correlation does not establish a beneficial edit rule. |
| At elite formation | Track whether the absolute threshold and cumulative anchors leave only a few compositions or local regions. Record label precision and selection optimism. | High nearby-score correlation does not show that a narrow elite pool is optimal. |
| Before and after neural refit | Compare the same controlled sample diagnostics across original and fine-tuned members. The species-guidance intervention provides an existing example of a width gain erased by refitting. | No tabular alpha exists to tune by analogy. Reduced training, replay or model mixtures each need their own comparison. |
| Optional local proposal branch | Reserve a declared part of the same proposal/scoring allowance for one-Pokémon-set or field-level edits around selected teams, log parent/operator/actual edit size, pass ordinary legality and held-out gates, then feed evaluated candidates through the declared selection rule. | The existing smoothness result does not prove that local refinement beats using those evaluations for ordinary proposals. Allocation and acceptance remain choices, not defaults. |
| Optional adaptive response | Repeated sampled revisits could motivate a restart or broader proposal; reliably interacting edits could motivate joint edits. | Diagnostic thresholds, new mutation mixture and adaptation rules need predeclared validation. No new threshold is justified by this audit. |

For a proposed branch, the clean-code seams are proposal creation `src/cem_v3.py:223–244`, acquisition `:245–248`, battle logging `:249–264`, elite formation `:176–185,206–215`, and neural update `:113–139`. In the later driver use `src/cem_dbbo.py:345–366`, `:513–530`, and the selected training branch `:491–512`. Preserve the chosen arm's exact scorer, opponents, policy, legality, selection and total cost controls.

The strongest immediate explanatory comparison is to distinguish the **objective's local smoothness**, the **ridge's ranking accuracy near an incumbent**, and the **generator's support after refitting**. These are three different mechanisms. A one-edit smoothness result can support testing a local branch, but it cannot diagnose all three by itself.

## Direct calculation from the clean saved run

Source: `/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/cem_v3_noheldout.json`. Calculated read-only from this JSON on October 20, 2026. Each generation has 512 saved proposal pastes and 128 acquired/battled pastes. Means and threshold counts were independently recomputed from `gens[g].y`; acquired species-composition counts were independently recomputed from the six saved Showdown block headers and match `gens[g].stats.battled_sets`.

| Generation | Proposals | Acquired | Acquired species compositions | Mean observed score | Labels at least 0.531 | Recorded training elites | Exact proposal repeats versus earlier proposals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 512 | 128 | 60 | 0.4578125 | 35 | None, original model | 0 |
| 1 | 512 | 128 | 22 | 0.5056250 | 52 | 64 | 0 |
| 2 | 512 | 128 | 21 | 0.50640625 | 52 | 105 | 0 |
| 3 | 512 | 128 | 14 | 0.52796875 | 64 | 157 | 0 |

There are also zero exact repeated proposal pastes within each generation, and zero acquired full-paste repeats versus earlier acquired pastes. Identity here means the entire saved paste after outer whitespace trimming only. Pokémon order and move order are preserved, as are item, ability, nature and spread text. This is deliberately stricter than species composition or canonical database identity. The absence of literal repeats does not establish broad coverage of useful team neighborhoods.

Useful JSON keys are `config.cut`, `config.elite_min`, `anchors`, `gens[g].proposals`, `mu`, `battled`, `battled_mu`, `team_id`, `y`, `stats.battled_sets`, `stats.mean`, `stats.above_cut`, and `elites.n`. No individual parent or mutation operator exists in these rows.

### Proposal-composition count caveat

The recorded proposal composition counts `gens[g].sets` are 156, 149, 109 and 76. Reading all six saved block headers gives 154, 147, 109 and 75. The difference is reproducible from the known parser behavior in `src/corpus.py:116–147`: blank lines are ignored, and an item-less header after the first Pokémon does not start a new slot. Replicating that legacy header rule exactly reproduces all four recorded counts. There are 2, 2, 1 and 1 affected proposal pastes respectively. This audit does not modify the parser or historical results. Use the validated acquired counts in the main explanation, or label recorded proposal counts as recorded rather than corrected ground truth.

### Ordered whole-block proximity can be calculated, with a narrow interpretation

Split each paste into its six blank-line-separated Pokémon blocks, preserve block order and move order, and compare corresponding block strings. The distance is the count of positions whose entire block strings differ. This is a literal ordered block-Hamming distance, not a field-normalized, permutation-invariant or battle-equivalent distance. Formatting, order and spread choices can inflate it. Database canonicalization may also mean the battled serialization differs from the saved proposal serialization, so this metric must not be treated as a certified fitness-landscape neighborhood.

For the reference pool, reconstruct the prior labelled rows exactly as the loop does before parsing and vocabulary filtering: anchors plus prior generations' non-null labels; keep labels at least 0.531, or the top 64 when fewer qualify. These label-pool sizes are 64, 105 and 157 at generations 1, 2 and 3, matching the corresponding recorded elite counts. This count match does not independently prove every model-encoded row identity. Generation zero has no training elites; its 49 anchors can only be an explicitly labelled reference.

| Generation | Reference | Minimum ordered block distance 5 | Minimum distance 6 | Minimum distance 0 through 4 |
| --- | --- | --- | --- | --- |
| 0 | 49 anchors, reference only | 7 | 505 | 0 |
| 1 | Prior eligible label pool, 64 rows | 55 | 457 | 0 |
| 2 | Prior eligible label pool, 105 rows | 102 | 410 | 0 |
| 3 | Prior eligible label pool, 157 rows | 110 | 402 | 0 |

This does not establish that proposals are biologically distant or that local search will help. It does show why whole-team neural proposals should not be presented as observed one-edit steps from an elite.

A minimal read-only calculation recipe, run over an already loaded JSON object `d`, is:

```javascript
const blocks = paste => paste.trim().split(/\n\s*\n/).map(x => x.trim());
const identity = paste => paste.trim(); // retain slot and move order
const distance = (a, b) => a.reduce((n, slot, i) => n + (slot !== b[i]), 0);
const seen = new Set();
for (const [g, c] of Object.entries(d.gens).sort((a, b) => +a[0] - +b[0])) {
  const observed = c.y.filter(y => y !== null);
  const repeats = c.proposals.filter(p => seen.has(identity(p))).length;
  console.log(g, c.proposals.length, c.battled.length,
    c.stats.battled_sets, observed.reduce((a, b) => a + b, 0) / observed.length,
    observed.filter(y => y >= d.config.cut).length, c.elites?.n ?? null, repeats);
  c.proposals.forEach(p => seen.add(identity(p)));
}
```

Before using `distance`, assert that both block arrays have length six and specify the reference pool and serialization. It supplies proximity only. Do not label the nearest reference team as an actual parent or draw an observed transition edge to it.

## Control the proposal-versus-acquisition sample size

Comparing all 512 proposals' distinct compositions directly with 128 acquisitions' distinct compositions confounds the selector with the smaller sample. A size-matched reference is the exact expected number of distinct species compositions in a uniform 128-row subset of that same saved 512-row proposal batch, sampled without replacement.

Let composition c occur f_c times in a batch of N = 512 proposals, and k = 128. Its probability of appearing at least once is one minus the probability that the subset avoids all f_c copies. Linearity of expectation gives:

`E[distinct compositions] = sum_c {1 - C(N - f_c, k) / C(N, k)}`.

Set the absent probability to zero when `N - f_c < k`. Otherwise compute the ratio stably as `product_{j=0}^{k-1} (N - f_c - j) / (N - j)`. This is an exact finite-batch expectation evaluated numerically, with no Monte Carlo draws and no new battles. Composition keys use all six saved block headers; the frequency counts sum to 512 in every generation.

| Generation | Full-batch compositions, six headers | Expected compositions in uniform 128 | Actual acquired compositions | Actual minus uniform expectation |
| --- | --- | --- | --- | --- |
| 0 | 154 | 51.7780013 | 60 | +8.2219987 |
| 1 | 147 | 48.2053181 | 22 | -26.2053181 |
| 2 | 109 | 37.6747006 | 21 | -16.6747006 |
| 3 | 75 | 26.2712816 | 14 | -12.2712816 |

The actual acquisition is more composition-concentrated than uniform 128-row selection in generations 1 through 3. Generation zero is less concentrated by this count. This separates an observable acquisition effect from the sample-size difference. It does not establish that concentration harms win rate, that uniformly selected teams would win more battles, that any difference is statistically significant, or that 128 acquired rows require 128 entirely fresh cache-independent evaluations. Those claims require different evidence.

A small calculation over the validated composition frequencies is:

```javascript
function absentProbability(N, frequency, k) {
  if (N - frequency < k) return 0;
  let p = 1;
  for (let j = 0; j < k; j++) p *= (N - frequency - j) / (N - j);
  return p;
}
const expected = [...frequencies.values()].reduce(
  (sum, frequency) => sum + 1 - absentProbability(512, frequency, 128), 0);
```
