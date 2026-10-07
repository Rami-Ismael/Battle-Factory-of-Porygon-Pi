---
created_at: 2026-09-07
status: proposed-not-run
tags:
  - diffusion
  - experiment-design
  - permutation
---

# Does rearranging a team change the model's item predictions?

## The question in plain language

Take a complete team and hide Garchomp's item. Ask the diffusion model how likely each item is. Move Garchomp, together with all its fields and the hidden item, to another place in the same team list. Ask again. Did its guesses change even though the Pokémon and their sets stayed the same?

The output here is a distribution over missing items, not predicted win rate. The model is filling in a blank. This is a diagnostic of the denoiser, not a battle experiment or a test of which item is best.

## Why it is worth a small experiment

The pilot already sorts teams and shuffles complete Pokémon blocks during training. `TeamDiffusion` nevertheless adds a different learned positional vector to each of its 48 input columns. Training may have taught it to mostly ignore arbitrary team order; we have not measured whether it did. No architecture defect or performance improvement is established yet.

Sources: [encoding](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/encode.py), [model and training loop](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/diffusion.py), [[tegmark-interpretability-lessons-for-vgc]].

## A small reproducible protocol

1. Use one existing trained `TeamDiffusion` checkpoint. Record its hash, architecture, source revision, vocabulary/corpus hashes and training split. The default path `/tmp/vgc-pilot/diffusion.pt` was absent when inspected on 2026-09-07. Do not retrain merely to illustrate this proposal. Recover the actual checkpoint and matching vocabulary first.
2. Select 32 distinct teams from its documented held-out set with a fixed sampling seed, e.g. 20260907. Canonicalize/deduplicate before selection. If split provenance is unavailable, label the sample as corpus teams with unknown training membership, not held-out teams. The current training function defaults to seed 0 and a 15% split, but that does not establish the provenance of a missing checkpoint.
3. For each team, make six cases: hide only member 1's item, then only member 2's item, and so on. Every other field stays fixed. Each case has exactly one hidden field.
4. For each case, evaluate the six cyclic rotations of the complete six-member list. Each member appears in every slot exactly once; the original order is rotation zero. This gives 32 × 6 × 6 = 1,152 predictions, plus 192 repeat-input control predictions. Six rotations are a bounded screening test, not exhaustive coverage of all 720 permutations.
5. Run in `model.eval()` and `torch.inference_mode()` so dropout is disabled. Fix timestep to `1/48` for this single-mask diagnostic and conditioning to `NULL` for every case. Use the same checkpoint, hardware, dtype and evaluation settings throughout. Do not sample an item, generate a full team, retrain, or run battles.
6. Extract the logits for the hidden item's current column, remove the `[MASK]` category, and normalize the remaining logits using softmax at temperature 1. Save the entire item probability vector. Use the same vocabulary and raw output transformation in every order. This deliberately measures the denoiser before legality masks or constrained decoding; it does not claim these are its final deployed generation probabilities.
7. Repeat the unchanged original input as a control. Its distribution should agree within numerical tolerance. Investigate a nontrivial control difference before interpreting the order experiment.

## Implementation detail that prevents an invalid experiment

`Vocab.encode()` calls `team_fields()`, which sorts the team. Therefore, **encode once, then rotate the encoded tensor**. Rotating the source team and calling `encode()` again would silently sort it back and test identical inputs.

Reshape the 48 fields into six blocks of eight. Move the whole block, including its mask; never shuffle individual fields across Pokémon. Track which new block contains the target member before reading its item logits. Preserve move order within each member for this test.

Sketch, not an executed implementation:

```python
# model, vocab and row must come from matching checkpoint/corpus artifacts.
grid = encoded_row.reshape(6, 8).copy()
grid[target_member, 2] = 0  # item column; 0 is [MASK]

for shift in range(6):
    order = np.roll(np.arange(6), shift)
    shuffled = grid[order].reshape(1, 48)
    new_target = int(np.flatnonzero(order == target_member)[0])
    item_column = new_target * 8 + 2
    # h = model(shuffled_tensor, timestep_tensor, null_condition_tensor)
    # logits = model.logits(h, item_column)[0]
    # probabilities = torch.softmax(logits[1:], dim=-1)
    # Save probabilities aligned by item ID, original member and current slot.
```

Assert that all six blocks, masks and target identities are preserved under each rotation, and that inverse rotation restores the original encoded input. Team preview leads, active battle positions and action/target indices are outside this test: their ordering can have game semantics.

## What to measure

For each team/member/order, compare its full item probability vector with that member's prediction in the original order:

`distribution_shift = 100 × 0.5 × sum(abs(p_original - p_reordered))`

This is total variation distance on a 0–100 scale: zero means identical guesses. For intuition, changing probabilities from `[70%, 20%, 10%]` to `[40%, 45%, 15%]` moves 30 percentage points of probability between alternatives. These example numbers are invented.

Also record whether the most probable item changes, with the top-two probability gap to distinguish a near tie from a large change. A single low-probability rank flip should not be the headline.

Summarize by taking the largest distribution shift across the five changed orders for each target member, then average across the six target members to get one value per team. Report the median, 90th percentile and observed range of those 32 team values. Treat teams as the sampling unit; the 1,152 predictions are not independent replicates. A pilot on one checkpoint measures that checkpoint, not all training runs.

Suggested output rows: checkpoint hash, team ID, target species, original member index, current slot, rotation, timestep, condition, item ID, probability, distribution shift and top-item change.

## Visuals for actual results

**First, explain one example.** Show original and reordered team lists with the hidden item attached to the same Pokémon. Below them, compare the two item distributions on a shared 0–100% axis. Label the y quantity as model probability, not win rate. For a crowded vocabulary, choose the union of the top three items from both orders and aggregate the rest as “other.”

**Then show the whole pilot.** Use a horizontal dot plot: x = per-team distribution shift (0–100), y = the 32 team IDs, one dot per team. Add each team's repeat-input control using a distinct marker on the same axis. Keep exact values in tooltips or an accompanying data file. This answers whether order sensitivity is widespread or comes from a few examples.

For an optional detailed view, select a team/member and plot its leading item probabilities against the target's six slot positions. Label it “six rotations”: the whole team's positions move together, so this does not isolate the target slot as the only cause.

Preselect the explanatory team before inspecting results, or explicitly label a result selected afterward as the largest observed case. Show the all-team plot either way to avoid presenting an extreme example as typical.

## How to interpret the outcome

- Predictions stay near the repeat-input noise floor: the current model passes this narrow screening test. No redesign follows from this result alone; passing six rotations does not prove invariance for every mask, field, timestep or permutation.
- Predictions differ visibly beyond repeat-input noise: the denoiser is sensitive to arbitrary ordering. Check the pipeline's input orders and whether constraints suppress the effect before treating it as a practical failure.
- A later architecture comparison is warranted only if the sensitivity matters. An order-consistent model must preserve Pokémon grouping and within-member field identity. Simply removing all position information can destroy that grouping. Retrain any alternative under matched conditions; do not zero learned positions at inference and call it a fair comparison.
- Whether an alternative actually generates stronger teams requires a separate matched-budget generation and battle comparison. Passing this diagnostic alone does not demonstrate strength, novelty, legality or improved search.

This is a useful inexpensive check of an assumption, with modest research novelty by itself. Its value is deciding whether arbitrary list order deserves further work in this particular pilot.
