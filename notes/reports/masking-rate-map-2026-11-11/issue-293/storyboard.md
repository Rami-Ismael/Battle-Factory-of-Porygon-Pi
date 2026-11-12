# Storyboard: Corrected t: how should a per-field-masking-rate model be given t while decoding in a fixed order, to match MD4?

Figure sentence: The corrected t fixes a real mismatch, but the model barely reads t.

## 1. Halfway through sampling: 42 of 84 fields filled

- Sentence: The sampler fills species first, then abilities and items, then moves. Here half the team is still hidden.
- Measured: The sampler tells the model t = 0.50.

## 2. What t = 0.50 meant in training

- Sentence: Training almost never hid a species at t = 0.50, so it hid less than half the team.
- Measured: 36% of the team hidden, 30 of 84 fields. Half was hidden at t = 0.65.

## 3. Today's t against the corrected t

- Sentence: The corrected t sits above today's t at every pass.
- Measured: 0.65 against 0.50 at the halfway pass, and 0.97 against 0.93 after six fields.

## 4. Does the model care which t it is told?

- Sentence: The model barely reads t, so the fix changes almost nothing.
- Measured: Largest shift 0.005 on today's default model, and 0.008 on a model trained on the shuffled-only table.
## Cut

- Nothing recorded.
