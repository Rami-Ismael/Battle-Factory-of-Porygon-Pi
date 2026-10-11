# VGC diffusion team generation

This glossary names the diffusion models and matchup evidence used to propose competitive Pokémon teams.

## Language

**Overall-win-rate diffusion model**:
The diffusion model conditioned on a single overall win rate and trained on teams collected from earlier search experiments plus the real-team corpus. Its original experiment identifier is `F1_final_medium` (F1); these data sources describe where its training examples came from, not a recurrent architecture or specifically cross-entropy-method training; see the [training report](code/docs/loss-down.md#final-model).
_Avoid_: unexplained F1, first-generation model, loop-trained model, CEM-trained model.

**Matchup-conditioned diffusion model**:
The diffusion model trained on the earlier-search and real-team data plus eligible matchup-matrix teams, with scalar win-rate and per-opponent matchup-row conditioning, excluding a reserved matrix holdout. Its original experiment identifier is `G2_family_matrix_noprtrain` (G2), trained without the all-data pretraining used for G1; see the [training report](code/docs/loss-down.md#final-model).
_Avoid_: unexplained G2, second-generation model, model trained on the full matchup matrix.

**Diffusion ensemble**:
The combination of the overall-win-rate diffusion model and the matchup-conditioned diffusion model that averages their predicted probabilities for each team field. The original experiment labels are `F1 + G2` and `ENS_F1_G2`; see the [ensemble definition](code/src/lossdown.py).
_Avoid_: unexplained F1/G2, a single newly trained model.

**Matchup matrix**:
The collection of measured battle results for candidate teams against the reference opponent panel; it can contain untested pairings. The matrix is evidence from sampled battles, not a complete evaluation of every possible team pairing.
_Avoid_: full matchup matrix when implying exhaustive coverage.

**Matchup row**:
One team's measured win rates against individual reference opponents, with untested matchups marked as missing. It carries opponent-specific information beyond the team's scalar win rate; see [matchup conditioning and holdout selection](code/src/lossdown.py).
_Avoid_: overall win rate when referring to the whole row.
