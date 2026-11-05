# Opponent validation split decision

Recorded in [Which meta teams stay out of training?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/117#issuecomment-6665860310).

Agreed with the owner on 2026-11-05: reserve exactly **10 validation opponents** and use the other **39 opponents for training**. Keep connected one-swap families intact.

### Family rule

- Treat each of the 49 distinct opponent teams as a vertex. The duplicated paste in the original top-50 panel does not create another distinct opponent.
- Species replacement distance is six minus the number of shared species. Ignore Pokémon order, moves, items, abilities, Stat Alignment and Stat Points. Preserve the existing distinction between species forms.
- Connect teams at distance zero or one, meaning they share at least five species. A family is a connected component, including chains. Assign each whole family to one side of the split.

### Selection and reuse

Use a fixed random seed to choose a union of whole families containing exactly 10 teams. Choose without looking at model results. Freeze that membership for every model and setting compared in this experiment.

The implementation step must record the seed, canonical input order, normalization, family membership and selected opponent identities in a shared split manifest. The actual draw has not been made in this decision session.

### Scope of the holdout

The split concerns opponent columns, not a limit of 39 candidate training examples. Training prompts use only the 39 training opponents. Recompute training win-rate labels from the selected training-opponent subset, excluding validation outcomes. An existing all-49 average cannot serve as a training label for this split.

Validation supplies the reserved opponent descriptions at evaluation time and measures generated teams against them. Its results choose settings. Ranks 51 to 100 stay untouched until the single final prompted test with settings locked.

Whether reserved opponents or their near copies must also be excluded when they occur as candidate reconstruction rows remains a training-data decision, carried into [Which battles and subset sizes make the training labels?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/118).

### Supporting check

A read-only calculation on the current 49 opponents found one-swap connected-family sizes of 4, 4, 2 and 2, plus 37 singletons. A whole-family selection of exactly 10 is therefore feasible. The two-swap alternative creates a 16-team connected family, including endpoints sharing no species. These checks used `code/src/top50.py` and the existing `paste_species` normalization in `code/src/temperature_experiment.py`; no ranks 51 to 100 contents were read and no battles or training were run.
