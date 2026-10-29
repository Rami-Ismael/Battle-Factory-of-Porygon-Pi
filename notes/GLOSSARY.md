---
created_at: 2026-10-20
updated_at: 2026-10-27
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

**Battled candidate**:
A legal candidate with a measured win rate.

**Unmasked team**:
A complete team before a mask hides any field; each Mask Lab grid has five, and every mask is cut from one. Battled unchanged, it is what every fill of its masks is compared with. Its grid says what kind it is: random (random legal HPS teams nobody built), held-out (tournament teams placed 51 to 100) or meta (top-50 meta teams).
_Avoid_: starting team, original team, starter, source team

**Team**:
Six Pokémon. Species, item, ability, moves, Stat Points and Stat Alignment are all decision variables.
_Avoid_: roster

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

**Meta**:
The top-50 meta teams, each counted once. A team's win rate against the meta averages one battle against each of the 50.
_Avoid_: usage-weighted meta, species usage %

**Held-out team**:
One of the 50 meta teams placed 51st to 100th on the VGCPastes Reg M-B sheet. Nothing trains on it or searches against it; it scores the final selected counter once, at the end.
_Avoid_: test team, next-50

**Contamination**:
A held-out team, or a team with the same 48 fields, reaching training, a search or stored battle results before the final score. Only held-out teams count: testing a model on a team it trained on that is not held out is not contamination.
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
