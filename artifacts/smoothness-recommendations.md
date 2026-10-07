# Recommended visual sequence

Use four small figures in this order. Introduce smoothness before naming sensitivity. The sketches in [smoothness-recommendations.svg](smoothness-recommendations.svg) are layout proposals; their shapes and example values are illustrative, not experiment results.

## 1. Smoothness: compare two neighborhoods

**Recommendation:** Start with two side-by-side discrete dot plots. Both show an original team and three single-edit neighbors, with the same score scale. One illustrative neighborhood has small score differences; the other has larger differences. Do not join the teams with a smooth curve: intermediate teams have not been evaluated.

**Reader's question:** Do nearby teams have nearby scores?

**Interaction:** Select an edit to highlight its neighbor in both plots. Keep every dot and the original team visible. Update a short sentence with the corresponding score differences.

**Animation:** Use a brief opacity emphasis on the selected dot. Keep the measured dots stationary. Keyboard selection should be immediate.

**Copy:** “Informally, local smoothness describes whether nearby teams tend to receive similar scores. Here, nearby means one legal edit under a specified protocol.”

**Limit:** The figure introduces an informal notion of smoothness. A mathematically smooth continuous function can still be steep and sensitive. One neighborhood does not establish a property of all team space.

## 2. Local sensitivity: isolate one edit

**Recommendation:** Show a six-Pokémon team strip with one selected slot. Beneath it, show the original and edited move side by side and two score bars on a fixed 0–100% scale. Retain the baseline and label the difference in percentage points.

**Reader's question:** How much did the score change after this specific edit?

**Interaction:** Choose one move substitution, reveal its recorded comparison, and reset. Show the other five Pokémon and all unchanged settings consistently.

**Animation:** Briefly resize the edited bar using a transform (240 ms), retaining the original bar. Display the measured endpoint immediately; the animation is a transition, not additional observations. Respect reduced motion.

**Copy:** “Local sensitivity is the size of the score change caused by an edit. Measuring it across several edits helps characterize the neighborhood.”

**Evidence:** With measured results, show uncertainty for the paired score difference and identify the fixed format, opponent pool, battle policy, and budget. Until then, label every score illustrative.

## 3. Why it matters: choose where to search next

**Recommendation:** Use a small neighborhood graph, rather than a smooth terrain drawing. A highlighted team connects to legal neighboring teams; a separate cluster represents a farther neighborhood. Draw all candidate nodes before revealing their scores.

**Reader's question:** How much can one evaluation tell us about what to try next?

**Interaction:** Let the reader choose a nearby candidate or a candidate from another neighborhood, then reveal the illustrative score. Retain the previous evaluations. A reset clears the revealed scores.

**Animation:** Fade in the revealed score. Do not make a particle glide between teams or imply a continuous path.

**Copy:** “Similar nearby scores can make local search useful. Sharp local changes can make broader sampling valuable. Neither pattern alone identifies the best search strategy.”

**Limit:** A toy graph demonstrates the decision, not the superiority of an algorithm. Use several examples, including a case where a local edit succeeds despite high sensitivity.

## 4. Benchmark comparison: show the distributions

**Recommendation:** Use three aligned distribution plots of absolute score changes: Pokémon, a selected combinatorial benchmark, and a selected continuous benchmark. Under each plot, state its edit or step protocol. Use a declared objective normalization for the shared axis, with original units available alongside it.

**Reader's question:** Under the stated protocols, where are small changes more or less predictive?

**Interaction:** Selecting a point reveals its original input, edited input, score difference, and uncertainty if applicable. For Pokémon, allow comparison between random and top-placing starting teams; neither group is assumed to be more sensitive.

**Animation:** Highlight the selected observation without moving the distribution.

**Evidence:** Populate positions from measurements only. The sketch uses empty plot regions because benchmark results have not been supplied. Define normalization, sampling, evaluation noise, and the number of edits per starting input before interpreting a comparison.

**Copy:** “These distributions compare score changes under explicitly defined perturbations. The perturbations are not inherently equivalent across search spaces.”

## Design recommendation

Use a quiet white surface and loose ink outlines for the wireframe. Keep the original team in dark ink, edits in violet, and search choices in blue. Use direct labels and shapes as well as color. Keep the writing readable without clicking; use interaction to inspect evidence and preserve the comparison. Start by building figures 1 and 2; figures 3 and 4 support the search argument and measured comparison afterward.
