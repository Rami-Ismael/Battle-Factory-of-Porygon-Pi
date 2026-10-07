# Revised recommendation: three figures

## 1. Introduce smoothness

Keep the two neighborhood plots: small score changes on the left, larger changes on the right, with a shared scale. Define nearby teams as one legal edit under a specified protocol. Introduce smoothness before naming sensitivity. Label illustrative data until measurements exist.

## 2. Inspect local sensitivity with search and scrolling

Use a **searchable, scrollable edit list beside a persistent performance comparison**.

- Search by Pokémon, original move, or replacement move. For example, searching “Incineroar” narrows the list to edits affecting that Pokémon.
- Each row shows the exact edit, edited score, and signed difference from the original team in percentage points. Include uncertainty when results are measured.
- Scrolling browses more edits while the original team and its score remain visible. Selecting a row updates the detailed comparison; scrolling alone does not select a new row.
- Keep the selected row marked with both a visible label and color. Show the number of matching results and a clear-search action.
- If nothing matches, show “No evaluated edits match this search” and a clear-search control. An unmeasured edit must appear as “Not evaluated,” never as zero performance.

The detail panel shows the original six-Pokémon team, highlights the edited slot, names the before/after move, retains the baseline score, and shows the edited score and difference. Use a fixed 0–100% win-rate scale. On mobile, place a compact persistent baseline above the scrollable list and the selected comparison below it.

**Interaction purpose:** “Search or select an edit to find out how much performance changes relative to the same original team.”

**Animation:** A brief transition of the edited bar preserves the comparison; the baseline stays still. Keep numbers at the measured endpoint rather than counting through invented intermediate values. Search results update immediately. Keyboard selection and reduced-motion mode use immediate updates. Avoid animating the list while someone is reading it.

**Evidence:** The wireframe uses invented scores. Real rows must come from evaluated edits under the same fixed format, opponent pool, battle policy, and budget. Report paired uncertainty. The list represents evaluated neighbors, not all legal teams or all possible edits.

## 3. Compare benchmarks

Keep the benchmark comparison as the third figure. Show three aligned distributions of absolute score changes: Pokémon, the selected combinatorial benchmark, and the selected continuous benchmark. Label the edit or step protocol for each. Declare objective normalization before sharing an axis; retain original units in the detail view.

Select an observation to inspect its original input, perturbation, score difference, and uncertainty where relevant. Leave distributions empty until measurements are available; do not imply a benchmark ordering.

Explain the implications for local search, exploration, and exploitation in the connecting prose. The three-figure sequence is **smoothness → inspect edits → benchmark comparison**.
