# 08: Jev columns in one grid's report

**What to build:** On the meta grid, legal Jev teams are appended to the grid's candidates after a timestamped backup, labelled by run and mask, validated, battled by the grid's battle step, and shown in the grid report beside the original team, the random fill and the other arms.

**Blocked by:** 07

**Status:** ready-for-agent

- [ ] Every existing candidate is byte-identical after collect, and the backup exists.
- [ ] The report shows Jev top-pick and sampled columns with battle counts.

Source: `code/docs/jev-baseline-spec.md`.

**Result 2026-10-27 (smoke sample):** `collect --grid meta` appended the 20 legal smoke-run fills (1,471 → 1,491 candidates; the existing ones byte-identical; backup `candidates.before-jev-20261027-053455.json`; a rerun appends 0). The grid queue's battle script scored all 1,491 (11 new distinct teams, 35 s). `report --grid meta` on 5 Stat Points masks of MB158 (original 0.633): Jev top pick −0.149 vs the original, +0.012 vs the random fill, −0.041 vs Mercury (4 masks); sampled −0.139 and +0.021. Too small to read as a result. Report: `~/vgc-data/jev/meta-report.json`. Incident: this branch's `mask_lab_heldout_starters.py` lacks the meta setting, so the first battle run rebuilt the held-out grid's panels.json from cached cells (no battles, same scores, new evidence path).
