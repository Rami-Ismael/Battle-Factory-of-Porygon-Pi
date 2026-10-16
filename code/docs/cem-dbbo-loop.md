# Cross-entropy loop with Diffusion-BBO's rule (pre-registration, 2026-10-16)

**Status: queued, not run.** The run is item 1–3 of [the run queue](run-queue.md). Nothing below is a result.

**Question.** The version-3 loop (`cem_v3.py`, [results](cem-v3-loop.md)) re-steers the model on the
elites only and asks for a fixed win rate of 0.5. It climbed to 0.564 in one generation, then collapsed
onto one six-species core and stopped on the diversity guardrail. Diffusion-BBO (Wu et al., 2024) does
two things differently: it trains the score-conditioned model on **every** scored design, and it asks
for **best so far × w**. Does that rule climb further, and does it hold its diversity?

## What changes, what stays

| | Diffusion-BBO's own loop | Our loop with Diffusion-BBO's rule (`cem_dbbo`, `cem_scratch`) |
|---|---|---|
| Train on | all scored designs | all scored teams: the 49 top-50 anchors and every battled team, each with its matrix win rate as the condition |
| Target | best so far × w | best so far × w, capped at 1.0 (a win rate cannot exceed it) |
| Picking w | adaptive, among fixed multipliers | adaptive among 1.0, 1.1, 1.2, 1.3 by the ridge's lower confidence bound on a 64-team probe at each ask (see below); `CEM_W=1.2` fixes it, as the paper's fixed-multiplier ablation |
| Proposals | generate N, evaluate all | generate 512 Showdown-valid teams, the ridge picks the 128 to battle |
| Extras | none | ridge guidance in the decode (λ fixed, as `cem_v3`); a taboo list refusing proposals whose species set was already battled |

Changed from the first brief: the target was "delivered + 0.1, capped"; it is now Diffusion-BBO's own
rule, best so far × w. The lowered temperature is left out (the one-shot stack collapsed to 12 line-ups
with it, [foundation swap](cem-v3-loop.md)), and there is no separate low-ask group: the fixed-ask,
elites-only loop (`cem_v3`) is the comparison, and `CEM_W=1.0` asks for exactly the best so far.

**Best so far** is the best matrix win rate among every scored team of this run: the anchors and every
battled team. At generation 0 it is the best anchor.

**Picking w.** Diffusion-BBO chooses the requested score with an uncertainty-aware acquisition; its exact
criterion could not be re-read for this pre-registration (arXiv was unreachable from the session), so the
stand-in is: for each candidate w, decode 64 teams at the ask, score them with the ridge, and take the
mean prediction minus the mean ridge posterior standard deviation (noise 0.07, a 50-battle binomial at
0.5). The w with the highest bound is used. The probes are recorded per generation. **Check this against
§4 of the paper before the run**; if the paper's rule differs, replace `Target.pick` and amend here.

## Two groups

- **Re-steered foundation on every scored team (`cem_dbbo`).** Generation 0: the version-3 members as
  shipped, the win-rate-conditioned member (F1) and the matchup-row member (G2). Generation g ≥ 1: both
  fine-tuned from the **original** checkpoints on every scored team, 40 epochs, learning rate 1e-4,
  batch 64, as `cem_v3`. Only the training set and the ask differ from `cem_v3`.
- **From-scratch network on every scored team (`cem_scratch`).** One network with the F1 architecture,
  random weights, trained each generation on the regulation-vocabulary foundation rows (labelled and
  unlabelled) plus the validated synthetic teams, plus every scored team repeated ten times (the ridge's
  loop weight). Ten epochs, learning rate 5e-4 with warm-up and cosine decay, batch 256 (lossdown's
  default schedule). Its generation-0 model is also the first regulation model trained without warm
  start, a control for the version-3 members themselves.

Shared: guidance 2 at the ask, temperature 1, ridge guidance, nature repair; ridge refit each generation
with the loop labels at weight 10; stopping as `cem_v3` (saturation 0.01 over 3 generations, copy rate
above 10%, species sets below half of generation 0's, at most 10 generations); final: the 16 best
distinct battled teams re-battled fresh with 192 battles each.

## Measures

Per generation: best so far, w, ask; battled mean ± SE, p90, max; teams at or above 0.531 and **at or
above the ask** (asked versus got); species sets among 512 proposals; proposals refused by the taboo list;
copies; new-species teams; the training-set size. Final: fresh mean against loop labels (winner's curse),
teams at or above 0.531 fresh, best fresh.

## What the result settles

- Battled mean per generation against `cem_v3` (0.427 → 0.564 → 0.546, stopped at generation 2).
- Whether the ask rises with the best so far without the achieved win rate falling behind it, the
  failure Diffusion-BBO reports for over-ambitious requested scores.
- Species sets per generation against `cem_v3`'s 176 → 161 → 44: does training on every scored team
  hold diversity where elites-only training did not?
- From scratch against re-steered: same rule, same data, different starting weights.

## Cost

The re-steered group costs what `cem_v3` cost per generation. The from-scratch group retrains on tens of
thousands of rows every generation; its first generation measures the price before the loop commits to
nine more (`CEM_SCRATCH_EPOCHS` lowers it). Both run under `battle_supervisor`, resumable at every step.

```
CEM_SMOKE=1 CEM_GROUP=dbbo    python cem_dbbo.py run
CEM_SMOKE=1 CEM_GROUP=scratch python cem_dbbo.py run
CEM_GROUP=dbbo    python battle_supervisor.py cem_dbbo.py run --then final
CEM_GROUP=scratch python battle_supervisor.py cem_dbbo.py run --then final
```
