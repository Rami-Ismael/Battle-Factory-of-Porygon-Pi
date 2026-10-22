---
created_at: 2026-10-20
updated_at: 2026-10-21
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
A battled team kept to train the next round. Elites come from every generation so far and from the top-50 meta teams, each only if it clears the elite cut.
_Avoid_: winner, survivor

**Elite floor**:
The smallest number of elites kept, 64. When fewer teams clear the elite cut, the best 64 are kept anyway, so some elites can sit below 0.531.
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
A species.

**Team**:
Six candidates, each with its set. Species, item, ability, moves, nature/EVs and tera are all decision variables.
_Avoid_: roster

**Set**:
A candidate's competitive configuration. Collides with the mathematical sense of the word, so name the mathematical one explicitly.

**Selection**:
The four Pokémon brought at team preview.

**Meta**:
A distribution defined by the collected list of meta teams, never a list and never species usage %.
_Avoid_: usage-weighted meta

**Objective function f**:
What the search maximises: the expected win rate of a team against the meta.
_Avoid_: fitness, oracle

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
A model that writes whole teams. The scratch team writer learned only from this project's teams; the text team writer is a pretrained text diffusion model fine-tuned on pastes.
_Avoid_: generator (alone)
