# Storyboard: Train the 84-field model with the linear masking rate

Figure sentence: Two trainings that differ only in the masking rate, read before any battle.

## 1. What changes between the two trainings

- Sentence: Only the masking rate differs, so a gap between the two models can be pinned on it.
- Measured: Both use the same 214,598 teams, one seed each, and at most 80 epochs.

## 2. Validation loss over training

- Sentence: Each line tracks one model's validation loss as its training runs.
- Expected: Both curves fall and flatten, and neither turns upward.
- Surprise: One curve turns upward, a sign of overfitting, and its run stops early.

## 3. Each model under each masking rate

- Sentence: Each model should score best under the masking rate it trained on.
- Expected: The linear model is lower under the linear rate and slightly higher under the per-field rate.
- Surprise: The linear model is lower under both rates.

## 4. The t check on the new per-field model

- Sentence: The t check measures how far the model's choices move when told the corrected t.
- Expected: Every field type stays under 0.01, as on today's default model, and the row is not battled.
- Surprise: One field type reaches 0.01, and Per-field, corrected t is battled.
## Cut

- Losses 1.404 and 1.209: older table, so no verdict reads against them.
- Training-versus-validation gaps: results table.
