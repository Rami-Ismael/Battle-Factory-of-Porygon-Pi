---
created_at: 2026-11-01
updated_at: 2026-11-01
---
Question: what framework should a solo machine learning researcher use to design an experiment, so that a finished run yields a lesson they can act on? Research checked on 1 November 2026. This note informs a future skill and is not the skill.

Terms used throughout. **Win rate** is one battle against each of 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and shuffled slots. **Effective species** and **distinct line-ups** measure diversity. **Swaps** measure novelty: species changes from the closest training line-up. **Elite cut** is 0.531. Numbers marked "re-analysis" are mine, computed read-only from `~/vgc-data/results/line_up_repeats_es.json` and `species_quota_scratch_es.json`. They were not pre-registered.

## 1. Verdict

**Use a named synthesis, outcome-table design.** Before spending compute, write one table: each result the run could produce, which explanation that result rules out, and what you will do next. If two different results lead to the same action, or a rival explanation appears in no row, change the design before training. The controls around the table come from the controlled-experiment tradition (a primary measure, guardrail measures, trust checks, a test where both arms are identical). No existing framework fits, and I found no source that evaluates this combination, so its merit is argued here and untested.

| Candidate | What it gives | Why it loses as the whole framework here |
| --- | --- | --- |
| Strong inference ([Platt 1964, p. 1][platt]) | Rival hypotheses, and outcomes that each exclude one. This is the core of the table | No treatment of noise. It assumes a "clean result" is available by care alone, which a 49-battle simulated score doesn't give |
| Controlled experiments ([Kohavi et al. 2007, pp. 3-4][kohavi07]; [Dmitriev et al. 2017, p. 3][dirty]; [Fabijan et al. 2019][fabijan]) | Primary measure, guardrails, A/A tests, trust checks | Built for thousands of randomised users, where replication is nearly free. Here the costly unit is a trained model |
| Classical design of experiments ([Fisher 1926, pp. 511-512][fisher]; [NIST handbook, 5.3.3][nist-select]) | Factorial arms and confounding, which suit few runs best | Says how to estimate effects and nothing about what to do with them across three outcomes |
| [Deep Learning Tuning Playbook][playbook] | Scientific, nuisance and fixed settings; retrain variance | Written for one validation error. It assumes tuning across many trials |
| Reinforcement learning practice ([Patterson et al. 2024][patterson]; [Agarwal et al. 2021][agarwal]; [Colas et al. 2018][colas]) | Honest treatment of few runs | The remedy is mostly more runs, or many tasks to pool across. This project has neither |
| [Bartz-Beielstein et al. 2020][bartz] | Benchmark checklist across eight topics | A topic list by the authors' own description ([p. 35][bartz]). No order and no decision rule, and it leaves multi-objective design open ([p. 32][bartz]) |
| Pre-registration alone ([Nosek et al. 2018][nosek]) | Separates prediction from postdiction | Locks a plan without making it informative. The line-up repeats run was pre-registered and still confounded (section 4) |

### What the literature does to the first draft

The draft's six points survive, but its order and two of its claims don't.

| Draft point | Verdict | Change |
| --- | --- | --- |
| 1. Decision rule before results | Keep, but it isn't first | The action comes first, since a rule can't be written without knowing what it decides. Goals shape the whole setup ([Bartz-Beielstein, p. 4][bartz]) |
| 2. A contrast isolating one cause | **Contradicted as worded** | [Fisher 1926, p. 511][fisher] calls "ask Nature few questions, or, ideally, one question, at a time" "wholly mistaken". Avoid aliased causes. Don't avoid several factors |
| 3. Check the arm changed only its claim | Keep, move earlier | Most of this check costs no training. It belongs before the run as well as after |
| 4. Know what the uncertainty covers | Strengthen | It is a design choice, not a caption. Treating samples from one trained model as replicates has a name, pseudoreplication ([Hurlbert 1984, p. 187][hurlbert]) |
| 5. Best known alternative | Keep, split the question | "Which wins" and "why" need different experiments ([Hooker 1995, p. 1][hooker]) |
| 6. Scope and action | Split | Action goes first, scope goes last |
| Missing | Add | Smallest effect worth acting on; a rule across several outcomes; the point of comparison (fixed ask or matched diversity); a retrain of the control arm with a new seed; a record of the lesson |

## 2. The framework, in order

"Free" means no new training. An arm costs about 2.2 hours of training (6.5 hours for three arms).

### Before the run

| # | Step | What to write down | Source | One model per arm? |
| --- | --- | --- | --- | --- |
| 1 | Name the decision | The default you'd change, and whether this is an adoption question (which do I use) or an explanation question (why) | [Bartz-Beielstein, pp. 4, 11][bartz]; [Hooker, p. 1][hooker] | Free |
| 2 | Write the outcome table | Rival explanations, then one row per possible result: what it rules out, what you do. Include a "within noise" row | [Platt, p. 1][platt] | Free |
| 3 | Sort the settings | Scientific, nuisance or fixed. Every fixed setting becomes a caveat on the conclusion | [Playbook, "Identifying scientific, nuisance, and fixed hyperparameters"][playbook] | Free |
| 4 | Choose arms so causes aren't aliased | For each arm, what differs from the control, as measured on the data and not only as named. When two causes are suspected, cross them | [Fisher, pp. 511-512][fisher]; [NIST, confounding][nist-conf] | Yes, and it is the best use of few runs. See below |
| 5 | Write the rule across outcomes | One primary measure, guardrails with limits, the smallest effect worth acting on, and the comparison point | [Kohavi 2007, p. 3][kohavi07]; [Dmitriev, p. 3][dirty]; [Benavoli et al. 2017, p. 12][benavoli]; [Zitzler et al. 2003, p. 3][zitzler] | Free |
| 6 | Name the unit of replication | What one independent repeat is, and which noise sources each planned interval covers and omits | [Hurlbert, p. 187][hurlbert]; [Playbook, "Determining whether to adopt"][playbook] | Free to state. Paying for it is section 3 |
| 7 | Run the balance check | Between arms, compare what the arm should not change: label distribution, elite count, label grid, row source mix, vocabulary | [Kohavi 2007, p. 3][kohavi07]; [Fabijan, pp. 1, 3][fabijan] (analogy mine) | Free, and it is the cheapest catch in this list |
| 8 | Freeze and label | Mark confirmatory or exploratory. Fix the asks, cells and pooled read-out now | [Nosek, pp. 2, 7][nosek]; [Gelman and Loken 2013, p. 2][gelman] | Free |

Step 4 in numbers. In a 2 × 2 design each main effect compares two models against two, so four models give each factor the precision of two per arm (arithmetic). Fisher's oat trial made the same point: three sets of comparisons from 96 plots that single-question methods would need 224 plots to match, and over a wider range of conditions ([Fisher, p. 512][fisher]). His trial was also replicated in eight blocks, so this doesn't remove the need for an error estimate. The interaction stays unreadable without a noise figure.

Step 5 options, ranked for this project:

1. **Primary plus guardrails.** Win rate at matched diversity is primary. Line-ups and swaps must not fall by more than stated limits. Guardrail metrics are ones "we do not want to significantly harm" even though they don't indicate success ([Dmitriev, p. 3][dirty]). I recommend this, because the project's exchange rate between win rate and diversity isn't settled.
2. **A single weighted score.** [Kohavi 2007, p. 3][kohavi07] calls this "highly desired and recommended". It forces the trade-off to be set once, but the weights have no agreed source here.
3. **Dominance only.** Honest but rarely decisive: all four arms in section 4 are non-dominated. A single summary number can't substitute for it, since no unary quality measure, nor a finite combination of them, can show one set is better than another ([Zitzler et al. 2003, p. 3][zitzler]).

One-number generative scores hide which failure occurred ([Sajjadi et al. 2018, p. 1][sajjadi]). That supports reporting quality and diversity separately, but it gives no rule for choosing between them.

The smallest effect worth acting on also makes a null result useful. A region of practical equivalence lets the run conclude "no change that matters" instead of "not significant" ([Benavoli, pp. 7, 12][benavoli]). Without it, a null teaches nothing.

### During the run

| # | Step | What to write down | Source | One model per arm? |
| --- | --- | --- | --- | --- |
| 9 | Compare planned and actual counts | Rows kept, teams, battles and seeds per arm. Treat a mismatch as a stop. About 6% of Microsoft experiments showed a sample ratio mismatch, where assignment counts differ from the plan ([Fabijan, p. 3][fabijan]) | [Fabijan, p. 1][fabijan] | Free |
| 10 | Stop on the rule and log deviations | Each change, and whether it came before or after seeing outcomes. Documented deviations don't void the test ([Nosek, p. 7][nosek]) | [Nosek, p. 7][nosek] | Free |
| 11 | Record where training stopped | Best epoch against the cap. If best equals the cap, the arm compares models at a budget, not trained-out models | Project `early-stopping` skill; I found no primary source for the interpretation | Free |

### After the run

| # | Step | What to write down | Source | One model per arm? |
| --- | --- | --- | --- | --- |
| 12 | Validity first | Repeat the balance check on what actually ran. Surprising results get a root-cause hunt before any write-up | [Kohavi et al. 2012, p. 1][puzzling] | Free |
| 13 | Read the table as written | Which row fired. Report every arm, including losers | [Bartz-Beielstein, p. 32][bartz]; [Lucic et al. 2018, p. 2][lucic] | Free |
| 14 | State what the interval omits | The sources it covers, then a sensitivity check at a coarser unit | [Hurlbert, pp. 187, 199-200][hurlbert]; [Dietterich 1998, p. 22][dietterich] | Free |
| 15 | Attack rivals with data you already have | Other asks, members scored alone, same-grid comparisons. Mark these exploratory | [Nosek, p. 2][nosek]; [Platt, p. 1][platt] | Free or battles only |
| 16 | Compare with the best known alternative | Same budget and protocol. Flag controls reused from another run, and protocol changes | [Bartz-Beielstein, p. 18][bartz]; [Agarwal, p. 5][agarwal]; [Patterson, p. 42][patterson] | Free if one exists |
| 17 | Write the lesson | The sentence below | [Playbook, fixed settings caveat][playbook]; [Bartz-Beielstein, p. 6][bartz]; [Platt, p. 1][platt] | Free |

**The lesson sentence.** "With *what actually differed*, *measure* moved by *size*, in *these settings*; this run can't separate *named rivals*; so next we *action*." A run has produced a lesson when all five slots are filled and the action differs from what you'd have done with the opposite result. This test is mine, built from Platt's recycling step and the Playbook's fixed-setting caveat.

## 3. When replication is unaffordable

### What can be claimed from one trained model

Replication isn't always required. A single replicate per treatment is "neither surprising nor bad", and when gross effects are expected, a rough estimate is enough, or replication is very costly, it may be the best option ([Hurlbert, p. 199][hurlbert]). What is objectionable is giving those tentative conclusions "an unmerited veneer of rigor" through inferential statistics ([Hurlbert, p. 200][hurlbert]).

Three sources agree on the boundary, each in its own terms:

- [Bouthillier et al. 2021, section 6, p. 10][bouthillier]: when the models are given, the only variation left is the test data, and their framework doesn't apply. That compares models, not procedures.
- [Dietterich 1998, p. 22][dietterich]: for algorithms run only once he recommends McNemar's test, which does not "measure the variation resulting from the choice of training sets or internal randomness in the algorithm".
- [Playbook, "Determining whether to adopt"][playbook]: "often the trial variance alone can produce statistically significant differences between two different trained models that use the same hyperparameter settings."

So the honest claims are:

| Allowed | Not allowed |
| --- | --- |
| These two trained models produce teams that differ by X, beyond team sampling and battle noise | The filter causes X |
| The direction held in every ask and seed cell | The effect would reproduce on retraining |
| A rough size, if the rival explanations are ruled out by other means | A 95% interval for the effect of the method |

Pairing doesn't rescue this. Reusing a training seed across arms with different rows likely doesn't pair the models: one changed bit in one weight produced about as much run-to-run variability as all sources combined ([Summers and Dinneen 2021, pp. 1, 5][summers]). That was measured on image and language models, so applying it to this diffusion model is my extrapolation.

### The smallest seed count a source supports

**No source I read supports a universal minimum, and I won't invent one.**

| Source | What it actually says |
| --- | --- |
| [Henderson et al. 2018, p. 5][henderson] | Two groups of 5 seeds of one configuration differed significantly. "there can be no specific number of trials specified as a recommendation" |
| [Colas et al. 2018, pp. 16-19][colas] | Derive the count from a power analysis. Use at least 20 samples in the pilot that estimates the standard deviation. Don't use the bootstrap test below 20. Use Welch's t-test, and choose more than the analysis prescribes |
| [Agarwal et al. 2021, pp. 5-6, 20][agarwal] | Stratified bootstrap works from about 5-10 runs per task when pooling across many tasks. Per-task intervals need 20-30 runs. At 3 runs, intervals undercover. Atari 100k needed 50-100 runs for its medians |
| [Bouthillier et al. 2021, pp. 9-10, 16][bouthillier] | A minimum sample size of 29 reliably detects a probability of outperforming above 0.75 (Figure C.1). The 0.75 was chosen on five deep-learning tasks, so it may not transfer here (`run-experiment` already warns against borrowing this number) |
| [Patterson et al. 2024, p. 42][patterson] | Averaging over 3 or 5 runs is "a risky practice" in most cases, because the variance may be underestimated through luck |
| [Playbook, "Determining whether to adopt"][playbook] | Run the best trial N times with N unspecified; recharacterise usually only after major pipeline changes; sometimes it is too costly. Then adopt if better "taking into account the retrain variance... as best we can" |

Plain arithmetic gives the size of the problem. To exclude a difference δ with n models per arm, the half-width is about t × s × √(2/n), where s is the retrain standard deviation, which this project hasn't measured for this model.

| Models per arm | Degrees of freedom | Half-width | Largest s that still excludes zero at the observed −0.061 |
| --- | --- | --- | --- |
| 1 | 0 | none | no interval exists |
| 2 | 2 | 4.30 s | 0.014 |
| 3 | 4 | 2.27 s | 0.027 |
| 5 | 8 | 1.46 s | 0.042 |

Colas warns that s estimated from few runs is usually too small ([p. 16][colas]), so these figures are optimistic.

### What to buy first when training is scarce

1. **Free balance checks (step 7).** They would have caught the line-up repeats confound with no compute.
2. **Cross the factors (step 4).** This gets more from each trained model.
3. **One retrain of the control arm with a new seed, an A/A test.** About 2.2 hours. An A/A test both gives the variability for power calculations and tests the system ([Kohavi 2007, p. 4][kohavi07]). One pair gives a single degree of freedom, so it's a rough scale, not an interval. The Playbook's advice that trial variance needs refreshing only after major pipeline changes means this can accumulate across experiments. A running bank of control retrains could approach Colas's pilot of 20 (about 44 hours in total) without any single experiment paying for it. This is my inference, and it stays valid only while the pipeline doesn't change.
4. **Score members alone.** Each arm trains two members from different parent configurations. Scoring each member alone costs battles and no training, and gives a second look at direction. These aren't independent replicates of one recipe, and both share one row pick (`PICK_SEED`), so report sign agreement only.
5. **Don't buy more battles for this.** At arm level, battle noise is already small (section 4), and more battles per team do nothing for retraining uncertainty.

Evaluation randomness can be shared across arms, as [Patterson, pp. 33, 42][patterson] advise. Training-side randomness should vary if you're measuring a method, as [Bouthillier, p. 2][bouthillier] advises. Section 6 covers that conflict.

## 4. Applied to the line-up repeats run

### Scorecard

| # | Step | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Decision | **Missed** | `code/docs/line-up-repeats.md` asks "Does training on one team per line-up change win rate, diversity or novelty?" It names no default that would change |
| 2 | Outcome table | **Missed** | No result maps to an action. Almost any result could be read as interesting |
| 3 | Settings sorted | Partial | Seeds, caps and scoring fixed. The doc doesn't state the resulting caveats |
| 4 | Causes not aliased | **Partly met, then missed** | Random rows, same count cleanly removes row count as a rival. One team per line-up still aliases repeats with elite loss |
| 5 | Rule across outcomes | **Missed** | Reports three outcomes with no rule. No smallest effect. Comparison point fixed by ask |
| 6 | Unit of replication | Partial | "One training seed per arm, so provisional" is stated. But the interval still treats teams as independent |
| 7 | Balance check | **Missed** | Elite counts fell from 1,223 to 169 and from 920 to 167. Countable before training |
| 8 | Freeze | Met, on the doc's word | Dated doc with asks and read-out, and early stopping added with a dated note. The doc is untracked in git, so no commit proves the timing. Commit before results next time |
| 9-10 | Counts and deviations | Met | The report asserts matching stopping splits and full row sets. Early stopping recorded |
| 11 | Stopping point | Met, not interpreted | Stage 1 best equals the cap in every line-up arm and Every row. The species quota 800 arm stopped earlier (68 of 80, 54 of 60), so that comparison isn't at the same training state |
| 12 | Validity first | Done after the fact | The elite drop was found only after results |
| 13-14 | Report and interval | Partial | All arms reported, and a provisional note given. The interval itself reads as firm |
| 15 | Rivals on existing data | Not yet | See below. Most of the numbers needed were already on disk |
| 16 | Best alternative | Met | Species quota 800 is comparable, with byte-identical frozen code. Every row is a reused control, which is a weaker design than a concurrent one |
| 17 | Lesson | Open | Below |

### Rival explanations, tested on existing data (re-analysis)

**The interval depends heavily on the unit.** The 576 pooled teams aren't 576 independent draws. One line-up accounts for 271 of Every row's teams and 300 of Random rows', against at most 8 for One team per line-up.

| Against Every row | Teams as units (the report) | Nine ask and seed cells as units, paired | Bootstrap over line-ups |
| --- | --- | --- | --- |
| One team per line-up | −0.061 [−0.074, −0.047] | −0.061 [−0.076, −0.045], 9 of 9 cells negative | −0.048 [−0.095, +0.007] |
| Random rows, same count | +0.017 [+0.003, +0.031] | +0.017 [+0.006, +0.027], 7 of 9 positive | +0.021 [−0.049, +0.090] |
| Species quota 800 | not computed | −0.062 [−0.084, −0.039], 9 of 9 negative | −0.050 [−0.109, +0.017] |

The line-up bootstrap is a rough sensitivity check, since one dominant line-up makes it unstable. Still, **"random filtering raises win rate" doesn't survive it**. None of the three columns covers retraining.

**Battle noise is not the main source.** One team's standard deviation is about 0.126 in Every row, and binomial noise at 49 battles is about 0.070, so battles account for about a third of the variance between teams. Over 576 teams, battle noise on an arm's mean is only about 0.003. More battles wouldn't settle this comparison.

**The comparison point changes the result.** The read-out fixes the ask. But each arm has eight asks, so each arm traces a curve:

| Arm and ask | Win | Line-ups per 64 | Effective species |
| --- | --- | --- | --- |
| Every row, ask 0.3 | 0.344 | 47.0 | 21.0 |
| One team per line-up, ask 0.531 | 0.335 | 63.3 | 24.1 |
| One team per line-up, ask 0.5 | 0.357 | 63.0 | 22.7 |
| Species quota 800, ask 0.5 | 0.363 | 48.7 | 22.8 |

At roughly matched species, the −0.061 shrinks to between −0.009 and +0.013, while One team per line-up keeps about 16 more line-ups. Each cell holds 192 teams, so one win mean has a standard error of at least 0.009. That's no detectable loss, not a gain. Only the −0.009 compares two asks off the 1/24 grid; the +0.013 sets an on-grid ask against an off-grid one, so the label-grid rival below applies to it. The match also rests on one Every row cell: its neighbouring ask 0.2917 gives win 0.361 with only 16.9 effective species. The project already compared decoders at matched diversity on 2026-10-26, so this matches its own past practice.

**The arms may lie on one trade-off line.** Win rate change per extra effective species, against Every row: Random rows −0.0069, One team per line-up −0.0075, Species quota 800 −0.0063. Three different filters giving nearly one slope suggests they all move along one curve instead of shifting it. This is post hoc, with one model per point. It gives a target: an arm is interesting if it lands above this line.

**Nothing dominates.** At the read-out, all four arms are non-dominated on win rate, line-ups and species. A filter comparison with no rule can't end in a decision, and that is what happened.

| Arm | Win | Line-ups | Species | Swaps 0 / 1 / 2 / 3+ |
| --- | --- | --- | --- | --- |
| Random rows, same count | 0.426 | 27.3 | 12.3 | 85.8 / 12.3 / 1.6 / 0.3% (against Every row's line-ups) |
| Every row | 0.409 | 31.8 | 14.7 | 75.2 / 21.9 / 2.8 / 0.2% |
| One team per line-up | 0.348 | 63.2 | 22.8 | 26.7 / 65.5 / 7.6 / 0.2% |
| Species quota 800 | 0.347 | 49.3 | 24.6 | 3+ swaps 1.6% (from the brief, not rechecked) |

**A rival nobody listed: the label grid.** The read-out pools two asks on the 1/24 grid (12/24, 13/24) with one off it (0.531). A project finding from 2026-11-01, measured on a different model, is that asks on this grid also select a team population. Here, mean win on the four on-grid asks beats the four off-grid asks in every arm: Random rows by 0.014, Every row by 0.038, One team per line-up by 0.028, Species quota 800 by 0.038. If the filter changes how many rows sit on grid labels, that's a third thing it changed. Counting this is free. I didn't count it.

**Novelty moved less than diversity.** The 3+ swaps share is 0.2% in both arms, which is one team of 576. The filter moved teams from zero to one swap. It didn't produce far-from-training line-ups.

### What the framework would have changed

A free count before training would have shown the elite drop. In the filtered arm, elites fell to about a third of what a same-size random subset held: 169 against 562 in the battle-records model's stage 1 rows, and 167 against 442 in the loop-teams model's. The design would then cross the two suspected causes, at equal row count:

| | Elite count as in a random subset | Elite count as in one-per-line-up |
| --- | --- | --- |
| **Repeats kept** | Random rows, same count (`random_same_count`, exists) | Random rows, elites cut to match (new) |
| **One team per line-up** | One team per line-up, elite copy preferred (new) | One team per line-up (`one_per_lineup`, exists) |

Plus one Every row retrain with a new seed. Total: three trainings, about 6.6 hours, plus battles. Two checks come first. Count the line-ups holding at least one elite. If there are too few, the "elite copy preferred" arm can't reach a random subset's elite count, and the elite factor must be set at a level both rows of the table can reach. And pick the elite copy at random among those at or above the elite cut, not the top scorer, since the top label is the luckiest (a team labelled 0.79 re-battled at 0.57 in this project).

The outcome table would then be written first. The limits below are for the owner to set.

| Result | Rules out | Action |
| --- | --- | --- |
| Removing repeats costs win rate, losing elites doesn't | "Elite loss explains the drop" | Repeats carry value: keep them in training, get diversity elsewhere |
| Losing elites costs win rate, removing repeats doesn't | "Repeats cost win rate" | Filter with elites protected, then re-test for adoption |
| Both within the control retrain's difference | Neither cause supported at this budget | Close this line, record the size that was excluded |
| Both arms on the existing trade-off line at matched diversity | "The filter shifts the curve" | Don't adopt: asks and guidance already give that trade at no retraining cost |
| An arm above the line at matched diversity, guardrails held | "Just a slide along the curve" | Spend seeds to confirm |

The read-out would add: win at matched effective species and on same-grid asks as the primary measure, with line-ups and 3+ swaps as guardrails.

### The lesson it permits now

> With one team per line-up, on one trained model per arm, samples held about twice as many line-ups and 8 more effective species, but win rate fell 0.061 at fixed asks and fell by at most about 0.01 at matched species. Elite training rows fell five- to sevenfold at the same time. This run can't separate fewer repeats from fewer elites, can't separate either from retraining variation, and pooled asks that may select different team populations. So: don't adopt this filter as a default, don't record "repeats cost win rate", and count elites and label-grid rows before designing the crossed follow-up.

Three things the run did teach, within that scope. Row count alone isn't the driver, since Random rows, same count did not fall. Species quota 800 and One team per line-up reach the same win rate with different diversity profiles: more species and more 3+ swaps for the quota, more line-ups for the filter. And the "+0.017 from random filtering" shouldn't be reported as a gain.

## 5. Comparison with the existing skills

| Framework step | `run-experiment` | Other project material |
| --- | --- | --- |
| 1 Decision and action | Claim row states what is claimed, not what changes | `notes/Ablation Experiments.md` "Why" lines come close, and give a reason without an action |
| 2 Outcome table | **Missing** | Missing |
| 3 Settings | Covered in the Settings row | None |
| 4 Causes not aliased | Partial: Settings and Invariance probes. **Factorial design is missing** | `name-experiments` lists settings that differ, from definitions, not measured rows. `Ablation Experiments.md:58` has one 2 × 2 proposal, and `:76-77` shows the same problem (an arm that changed three things). The practice exists with no rule behind it |
| 5 Rule across outcomes | Partial: primary measure and smallest effect are there, "report diversity beside win rate" is there. **No rule across outcomes, no guardrails, no matched-diversity comparison** | Memory holds a matched-diversity precedent. No skill has it |
| 6 Unit of replication | Partial: "one seed per arm cannot separate the two". **Doesn't warn that pooled teams from one model and repeated line-ups aren't independent** | None |
| 7 Balance check | **Missing.** The probes test whether the model is in the right space, not whether arms differ only as named | None |
| 8 Freeze | Covered: confirmatory or exploratory, Deviations | `name-experiments` for names |
| 9-10 Counts and deviations | Covered | None |
| 11 Stopping point | Refers to `early-stopping` | `early-stopping` records best against cap. **Doesn't say what a best-at-cap arm means for a comparison** |
| 12 Validity first | Covered: check the contract, audit the surprise | None |
| 13 Every arm | Covered | None |
| 14 Interval's coverage | Partial | None |
| 15 Rivals on existing data | **Missing** as a step | None |
| 16 Alternative | Partial: baselines, but **doesn't flag reused controls** | None |
| 17 Lesson | **Missing.** "Report what the result can't show" is the closest | `pskill:reflect` mines chat transcripts to edit skills, so it covers agent behaviour and not experiment lessons. It doesn't overlap. `code/docs/experiment-names.md` is naming only |
| A/A retrain | **Missing.** The Repeat probe replays one team through the scorer, which tests battles, not training | None |

So the new skill should stay narrow. `run-experiment` already keeps the model in the right space and judges numbers. The gap is the decision layer: the action, the outcome table, crossed arms, the balance check, the multi-outcome rule, the unit, the rival pass, the lesson sentence.

One defect I found: `run-experiment` cites `code/docs/ml-experiment-practice-research.md` as its evidence, and that file isn't in this worktree or in the vault search I ran. Git history shows it only on checkpoint refs.

## 6. Disagreements and unknowns

### Where sources conflict

- **Randomise or fix seeds.** [Bouthillier, pp. 2, 9][bouthillier] says to randomise as many sources as possible. [Patterson, pp. 33, 42][patterson] says to control seeds and pair conditions. [Summers and Dinneen, p. 5][summers] suggests that pairing can't last through training anyway. My reading: share evaluation randomness across arms (opponents, sampling seeds), and don't count a shared training seed as pairing.
- **One question or many.** [Fisher, p. 511][fisher] rejects one question at a time. [Lipton and Steinhardt 2018, p. 3][lipton] and this project's "change one thing per arm" (`Ablation Experiments.md:76`) favour ablation. These fit together: a bundle of changes with no factorial structure is the actual fault, and a crossed design is many questions without the aliasing.
- **One score or many.** [Kohavi 2007, p. 3][kohavi07] recommends a single combined metric. [Sajjadi, p. 1][sajjadi] and [Zitzler, p. 3][zitzler] show what a single number loses. These answer different needs: deciding and diagnosing.
- **Competitive comparison.** [Hooker, p. 1][hooker] says competitive testing is suited only to development. [Bartz-Beielstein, p. 18][bartz] says to compare against the best algorithms. This project needs both, since it's building a proposer and also explaining it. Label each run.
- **Is one replicate acceptable.** [Hurlbert, p. 199][hurlbert] accepts it, with tentative conclusions. [Henderson, p. 5][henderson] and [Patterson, p. 42][patterson] warn against small n. The [Playbook][playbook] accepts adopting on best-effort variance. My reading: they apply to different claims. Adopting a working default needs less than asserting a cause.
- **Bootstrap at low run counts.** [Agarwal, pp. 5-6][agarwal] finds it usable from 5-10 runs per task across many tasks. [Colas, pp. 16, 19][colas] says not below 20. Agarwal's result needs many tasks to pool. Whether asks count as tasks here is unverified, and I doubt it, since they share one model.

### What I couldn't verify

- **Retrain variance for this model is unknown**, and every statement about the −0.061 depends on it. No source substitutes for measuring it.
- **Transfer.** Bouthillier, Summers and Colas studied classifiers, language models and reinforcement learning. I found no primary source measuring run-to-run variability for a discrete diffusion team generator scored by simulation.
- **Outcome-table design has not been tested.** It's a synthesis, and I haven't shown that it improves lessons learned. A cheap test: apply it to three past runs and see whether it changes their recorded conclusions.
- **Re-analysis limits.** My numbers are post hoc, and I compared many ask pairs before reporting the matched ones. Treat them as hypotheses. My species parser reads paste header lines rather than using `team_diversity.slots_from_paste`, which wasn't in this worktree's `code/src/`; its one-per-line-up maximum of 8 and line-up counts looked sensible, but I didn't cross-check it against the project function.
- **The elite-copy arm may be unbuildable** at the wanted count. Countable, not counted.
- **Elite counts and label-grid effect in this run.** The elite figures come from the brief. The label-grid mechanism comes from a project note about a different model.

## Sources

Read, at the pages cited: [Strong Inference, Platt 1964][platt] (a transcribed copy of the Science article, noted there as made from a scratched microfilm, so punctuation may differ); [The Arrangement of Field Experiments, Fisher 1926][fisher] (scanned, read as page images, pp. 509-512 only); [Pseudoreplication and the Design of Ecological Field Experiments, Hurlbert 1984][hurlbert] (course-hosted copy. Abstract, pp. 199-200); [Testing Heuristics: We Have It All Wrong, Hooker 1995][hooker] (author version, pp. 1 and 8); [Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, Dietterich 1998][dietterich] (section 8 only, read as page images); [Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis, Benavoli et al. 2017][benavoli] (pp. 7, 12); [The Garden of Forking Paths, Gelman and Loken 2013][gelman] (pp. 1-2); [The Preregistration Revolution, Nosek et al. 2018][nosek] (preprint on the Open Science Framework; the journal page refused the fetch. Pages cited are the preprint's); [Practical Guide to Controlled Experiments on the Web, Kohavi et al. 2007][kohavi07] (author version, pp. 3-5); [Trustworthy Online Controlled Experiments: Five Puzzling Outcomes Explained, Kohavi et al. 2012][puzzling] (p. 1 and passages only); [A Dirty Dozen: Twelve Common Metric Interpretation Pitfalls in Online Controlled Experiments, Dmitriev et al. 2017][dirty] (p. 3); [Diagnosing Sample Ratio Mismatch in Online Controlled Experiments, Fabijan et al. 2019][fabijan] (pp. 1, 3); [Deep Learning Tuning Playbook, Godbole et al.][playbook] (the sections named above); [Accounting for Variance in Machine Learning Benchmarks, Bouthillier et al. 2021][bouthillier] (pp. 1-5, 9-10, 16); [Deep Reinforcement Learning that Matters, Henderson et al. 2018][henderson] (pp. 5-7); [Deep Reinforcement Learning at the Edge of the Statistical Precipice, Agarwal et al. 2021][agarwal] (pp. 2-8, 20-21); [How Many Random Seeds?, Colas et al. 2018][colas] (whole paper); [Empirical Design in Reinforcement Learning, Patterson et al. 2024][patterson] (pp. 2, 5, 23, 31, 33, 35, 42-43); [Nondeterminism and Instability in Neural Network Optimization, Summers and Dinneen 2021][summers] (pp. 1-5, 7-8); [Benchmarking in Optimization: Best Practice and Open Issues, Bartz-Beielstein et al. 2020][bartz] (pp. 4, 6-7, 10-11, 18, 31-32, 35); [Performance Assessment of Multiobjective Optimizers: An Analysis and Review, Zitzler et al. 2003][zitzler] (pre-publication version, p. 3); [Assessing Generative Models via Precision and Recall, Sajjadi et al. 2018][sajjadi] (abstract); [Are GANs Created Equal?, Lucic et al. 2018][lucic] (pp. 1-2); [Troubling Trends in Machine Learning Scholarship, Lipton and Steinhardt 2018][lipton] (pp. 1, 3); [NIST Engineering Statistics Handbook 5.3.3, How do you select an experimental design?][nist-select]; [NIST Engineering Statistics Handbook 5.3.3.4.3, Confounding][nist-conf].

Opened and dropped, because they didn't earn a place: [Statistical Comparisons of Classifiers over Multiple Data Sets, Demšar 2006](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf) (its sample is the data set, p. 5, and this project has one scoring setup; read abstract and p. 5); [Leakage and the Reproducibility Crisis in Machine-Learning-based Science, Kapoor and Narayanan](https://www.alphaxiv.org/abs/2207.07048) (`run-experiment`'s Separation probe already covers leakage; read pp. 1, 3 only); [Improving Reproducibility in Machine Learning Research, Pineau et al.](https://www.alphaxiv.org/abs/2003.12206) (a reporting checklist that adds no design step here; read passages on pp. 6, 11-12 only); [A Metric Learning Reality Check, Musgrave et al.](https://www.alphaxiv.org/abs/2003.08505) (read abstract only, so no claim rests on it).

Not read, from memory or from bibliographic listings only. No claim above rests on these:

- Box, Hunter and Hunter, *Statistics for Experimenters*: not read, from memory. The term "hidden replication" comes from there; step 4 relies on Fisher and arithmetic instead.
- Fisher, *The Design of Experiments* (1935): not read, from memory.
- Kohavi, Tang and Xu, *Trustworthy Online Controlled Experiments* (2020): not read, no open copy. I used three papers by overlapping authors instead. Whether the book's chapters on guardrails and A/A tests add to them is unknown.
- Sculley et al., *Winner's Curse?* (2018): not read. Both OpenReview links returned a browser check.
- Knowles, Thiele and Zitzler, *A Tutorial on the Performance Assessment of Stochastic Multiobjective Optimizers* (2006): not read. The ETH link returned an archive stub.
- Mayo on severe testing, Shadish, Cook and Campbell on validity types, Lakens on equivalence testing, Daniel 1959 and Lenth 1989 on unreplicated factorials: not read, from memory. They look relevant for a later pass, Daniel and Lenth especially, since they address reading a factorial with no replicates.
- Published criticism of Platt's claim that strong inference speeds up science: not read, from memory. It is why this note uses his method and doesn't repeat that claim.

[platt]: https://pages.cs.wisc.edu/~markhill/science64_strong_inference.pdf
[fisher]: https://repository.rothamsted.ac.uk/id/eprint/20740/1/Fisher%20Arrangment%20of%20field%20experiments%20J%20Ministry%20%20of%20Agriculture%201926%20p%20503%20to%20513.pdf
[hurlbert]: https://www.zoology.ubc.ca/~bio501/R/readings/hurlbert%201984%20ecol%20monogr%20-%20pseudoreplication.pdf
[hooker]: https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers08/testing-heuristics.pdf
[dietterich]: https://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf
[benavoli]: https://www.jmlr.org/papers/volume18/16-305/16-305.pdf
[gelman]: https://sites.stat.columbia.edu/gelman/research/unpublished/p_hacking.pdf
[nosek]: https://osf.io/2dxu5/download
[kohavi07]: https://exp-platform.com/Documents/GuideControlledExperiments.pdf
[puzzling]: https://exp-platform.com/Documents/puzzlingOutcomesInControlledExperiments.pdf
[dirty]: https://exp-platform.com/Documents/2017-08%20KDDMetricInterpretationPitfalls.pdf
[fabijan]: https://exp-platform.com/Documents/2019_KDDFabijanGupchupFuptaOmhoverVermeerDmitriev.pdf
[playbook]: https://github.com/google-research/tuning_playbook
[bouthillier]: https://www.alphaxiv.org/abs/2103.03098
[henderson]: https://www.alphaxiv.org/abs/1709.06560
[agarwal]: https://www.alphaxiv.org/abs/2108.13264
[colas]: https://www.alphaxiv.org/abs/1806.08295
[patterson]: https://www.alphaxiv.org/abs/2304.01315
[summers]: https://www.alphaxiv.org/abs/2103.04514
[bartz]: https://www.alphaxiv.org/abs/2007.03488
[zitzler]: https://sci2s.ugr.es/keel/pdf/specific/articulo/zitzler02performance.pdf
[sajjadi]: https://www.alphaxiv.org/abs/1806.00035
[lucic]: https://www.alphaxiv.org/abs/1711.10337
[lipton]: https://www.alphaxiv.org/abs/1807.03341
[nist-select]: https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm
[nist-conf]: https://www.itl.nist.gov/div898/handbook/pri/section3/pri3343.htm
