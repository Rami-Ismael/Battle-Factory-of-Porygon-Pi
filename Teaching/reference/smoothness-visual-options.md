# Visual options for the team-building smoothness subsection

**I would start with two linked views: a comparison of the two groups, and a detailed view of the legal edits around one selected team.** This lets the reader see the overall result and inspect the evidence behind it. These are proposed designs; no measured results are supplied here.

1. **“Are top-placing teams more sensitive to small changes?” — two-group dot plot.**

   Put **50 random legal teams** and **50 top-placing teams** in two columns. Each dot represents one original team. Its height summarizes the magnitude of expected win-rate changes across that team's sampled single-edit neighbors, expressed in percentage points. One possible summary is the median absolute change; define it before collecting results and propagate battle uncertainty into its estimate. Show the individual team dots and a group summary with its uncertainty.

   This answers whether the sampled groups differ in local sensitivity. Their positions should come from the measurements; tournament placement does not determine which group should appear higher. Estimate changes using several edits per team, with the same edit-type sampling protocol in both groups. A raw absolute difference between two noisy win-rate estimates includes measurement noise, so that alone is insufficient.

   **Interaction:** click a dot to reveal its six Pokémon and the edit-level results in visual 2. Keep both groups visible.

2. **“What happens when I change just one thing?” — horizontal difference plot.**

   For the selected team, give each tested legal edit a row: the Pokémon sprite, the field changed, and its old and new values. Plot **edited-team win rate minus original-team win rate** on the horizontal axis, in percentage points. Put zero at the center. Each row has an estimated change and a horizontal 95% confidence interval. Left means weaker against the fixed opponent pool; right means stronger.

   This is the most direct figure for the single-edit claim. A large estimated change with a narrow interval is different evidence from a similarly large estimate with an interval spanning zero. Compute the interval for the **difference itself**, respecting any pairing in the battle design, rather than judging it from overlap between separate win-rate error bars. The distinction between uncertainty in individual estimates and their difference is illustrated by this [NIST paper](https://www.nist.gov/publications/should-t1-t2-have-larger-uncertainty-t1).

   **Interaction:** selecting a row shows the original and edited roster together, highlights the changed field, and reveals battle counts. A control can choose the edit category if multiple categories were actually measured.

3. **“How does this compare with the benchmarks?” — four aligned panels.**

   Repeat the neighborhood-change summary for **random Pokémon teams**, **top-placing Pokémon teams**, **the combinatorial benchmark**, and **the smooth continuous function**. Each sampled starting point contributes a summary of changes around it. Use matching visual structure so the reader can compare spread and typical change.

   Define the neighborhood for every problem: a legal Pokémon edit, the benchmark's discrete edit, and a specified continuous step size in scaled input coordinates. For a shared vertical axis, normalize objective changes using a declared, fixed scale for each problem and retain raw values in details. Otherwise, use separately labeled axes and avoid interpreting numerical heights across domains as directly comparable. Smoothness alone does not imply small changes at every arbitrarily chosen step size.

   **Interaction:** optional point inspection. The four-way comparison should already be readable without operating a control.

4. **“Does improvement repeatedly reverse?” — paths of legal edits, if you collect extra data.**

   Follow several actual sequences in which each step makes one legal edit to the previous team. Plot step number horizontally and estimated win rate vertically, with uncertainty. Show individual paths in small panels for both starting-team groups, including a predeclared selection of paths rather than only the most dramatic example.

   This adds evidence about rises, falls, and possible local peaks along sampled paths. Step number is not necessarily distance from the original team. Do not connect unrelated teams or sort teams by score and call the resulting line a landscape. Single-edit differences establish local sensitivity; a broader ruggedness claim needs an explicit definition and evidence about neighborhood structure.

**Recommended order:** use visual 1 for the main result, visual 2 for its concrete Pokémon explanation, and visual 3 for the benchmark comparison. Add visual 4 only if ruggedness along edit paths becomes part of the measured experiment. Keep the opponent pool, battle policy, evaluation budget, and edit protocol visible in the figure caption.
