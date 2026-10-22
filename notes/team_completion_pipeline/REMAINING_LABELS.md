# Missing training labels

`label_remaining.py` fills only missing labels in the versioned dataset from the VGCPastes run. It preserves existing labels and the original-row mapping. The protected evaluation records were already removed from that input.

The job uses six workers and concurrency 24 through `FastBattleRunner`. Requests can now hold up to 96 scheduled battles, independently of concurrency, reducing the number of times each worker must drain a small batch. Each playable team is evaluated against the 49 distinct top-50 opponents, excluding itself. Labels use membership-weighted opponent win rates, with draws counted as non-wins. A label is published only after full opponent coverage. One battle per opponent gives an initial estimate, not a precise ranking.

Original pastes are matched by their encoded fields and checked for legality. Stat Point spreads come from the matched saved paste; they are not reconstructed or resampled. When several saved pastes match, the first legal match is used. Missing neutral natures use the evaluator's neutral default. Illegal, unmatched, or protected records are reported rather than assigned invented labels.

## Run or resume

From the project root, with the existing pilot environment:

```sh
/tmp/vgc-pilot/.venv/bin/python team_completion_pipeline/label_remaining.py --background
```

A process lock prevents overlapping runs. The detached job keeps the Mac awake while running. Completed battle batches and labels are committed to SQLite. Restarting this command reuses those results. Worker errors trigger up to three attempts; persistent errors are shown in the live status. Turning the computer off stops the process; resume afterward using the same command.

`--workers` and `--chunk-size` configure scheduling. The default chunk size is now 96. New execution configurations receive separate cache identities. Completed earlier labels are retained only after checking that the policy, simulator, encoder, inference code, data, and numerical configuration still match. Each row's `label_value.policy_id` identifies its own results; cells from different execution identities are never pooled into one new label.

Outputs are in `team_completion_pipeline/runs/all-labels-20261013/`:

- `status.json` and `job.log`: progress and errors.
- `battles.sqlite`: exact source mappings, battle cells, and completed labels.
- `lossdata-all-fast.npz`: updated training corpus, checkpointed every ten batches and at completion.
- `dataset-manifest.json`: input/output hashes and the checkpoint's completion status.
- `exclusions.json`: any records that could not be labeled, written at completion.
- `evaluator-*`: policy, implementation and simulator fingerprints, plus individual battle evidence.

`artifacts/win-rate-frequency-live.json` updates after every completed batch. The chart polls it every 30 seconds, counts only available labels, and retains the original 0.01 rounding rule. No model is retrained automatically.
