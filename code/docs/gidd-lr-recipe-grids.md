# Ask grids for the shared GIDD-style learning-rate recipe (#431, plan step 4)

Masked, full filter trained with the shared recipe (peak `5e-4`, cosine to a `5e-5` floor, checkpoint
`stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80`, row `shuffled_linear_filtered_gidd_lr`) against the
same model on the old recipe (`4e-4` cosine to 0, row `linear_masked_filter`, grids from #385). Guidance 2, 4 and 8;
23 asks × 3 seeds × 64 teams per grid; one battle per top-50 meta team (49 distinct), slots shuffled, behaviour-cloning
policy. Both rows draw on sampling seed base 7300. Uniform's grids belong to #438.

## Storyboard

Written 2026-11-25 before any shared-recipe grid had landed.

| Frame | Reader takes away | Expected | Would surprise |
| --- | --- | --- | --- |
| 1. Got against ask, one per guidance (per-grid post) | The two recipes' teams hit the asked win rate equally closely. | The curves overlap; paired error difference inside the 0.002 luck bar. | The shared recipe misses asks by more than the bar at any guidance. |
| 2. Paired error difference by guidance (final post) | Lower validation loss does not change how closely teams hit their ask. | All three intervals inside the ±0.002 band. | An interval wholly outside the band, either way. |
| 3. Error against distinct line-ups (final post) | Whether the earlier validation-loss cliff buys line-up variety at the same targeting. | Points near the old recipe's; at most a small shift in line-ups. | Many more line-ups at equal error, a move outside the old recipe's curve. |
| 4. Shared minus old, every measure (final post, matrix) | Which measures move beyond training luck, at which guidance. | Mostly grey cells. | Coloured cells clustering at one guidance or one measure. |

Teaser, after all three grids: one view pairing the training result (validation loss 1.3143 → 1.2138) with frame 2's verdict.

## Deviations

- 2026-11-25 11:30: battles moved from the laptop to Modal CPU containers (owner). 20 of 69 guidance 2 cells had battled on the laptop. `src/modal_battle_check.py` re-battled 768 stored teams on Modal: Modal minus laptop −0.0035 [−0.0096, +0.0027], ask-versus-got error +0.0002 over 12 asks.
- 2026-11-25 11:32–11:38: the battle step loaded the pilot's `matchup_db.py` (no Modal backend) and restarted 10 times without battling; fixed in `asked_vs_got.py` (review finding).
- 2026-11-25 12:01: battles back on the laptop for guidance 4 and 8 (owner). The Modal battle app had used about $7.00 of credit in 40 minutes (equivalence checks and 49 guidance 2 cells), about $7–8 per grid against the ~$1 estimated. Guidance 2 battled on the laptop (20 cells) and Modal (49 cells).
