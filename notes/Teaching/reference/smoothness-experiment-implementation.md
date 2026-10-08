# Smoothness experiment: runnable code

Implemented on 2026-10-01 in the existing VGC team-generator pilot project.

- [Code and run instructions](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/smoothness-experiment.md)
- [Experiment implementation](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/scripts/smoothness_experiment.py)
- [Tests](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/scripts/test_smoothness_experiment.py)
- [Measured pilot report](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_pilot/report.md)
- [Measured pilot figure](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_pilot/smoothness.png)
- [Full-study manifest](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/manifest.json)

**Verified:** 14 tests passed and a 4,800-battle real simulator pilot completed.
The pilot uses four random originals and the first four top-placement entries,
two single-move edits per original, unchanged-team controls, and three panels of
50 battles per candidate. It is an implementation check, not the full result.

**Prepared, not battled:** 50 random legal originals and 50 frozen top-placement
entries, eight validated move edits per original, unchanged-team controls, and
four panels of 100 battles per candidate: 400,000 battles total. All fifty
opponent entries are retained. Pest Control and Hartmann-6 benchmarks are computed.

The user explicitly chose to retain the existing top-50 snapshot, including its
six ranked-season and four Showdown-ladder entries. Use **top placements** when
writing about this cohort, rather than describing it as strictly tournament-only.
The identical MB493/MB494 entries belong to one bootstrap cluster, so the placement
group has 49 distinct identities. Random teams are sampled from the frozen HPS
corpus, not uniformly from all mathematically legal teams.

The primary statistic is noise-corrected mean squared expected win-rate change
under a single legal move replacement. The code retains negative corrected
estimates, uses original-team cluster intervals, and saves pointwise edit intervals.
This measures local sensitivity under the fixed battle policy and pool; it does
not by itself establish mathematical non-smoothness.

## Full study launched — 2026-10-01

The user requested the full run. It started at 12:39 p.m. America/Chicago with four
dedicated local battle workers, retaining the agreed frozen 50-entry placement
snapshot and opponent pool. The budget is 400,000 completed battles. A supervisor
saves progress, runs the final analysis, and stops its own servers on completion;
a chat heartbeat checks for completion or failure without routine notifications.

[Live full-study status](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/run-status.md)

The earlier "prepared, not battled" description above records the state before this
launch. It is not the current run status.

## Full study completed — 2026-10-01

All **400,000 battles** finished at 3:54 p.m. America/Chicago (20:54:33 UTC).
The completion review verified 4,000 complete panels, 200,000 battles per group,
all frozen input/code/runtime fingerprints, and all 900 edit/control effects
independently recomputed from raw counts. The full PNG figure was visually
inspected; report, CSV, JSON and PDF outputs are saved. The supervisor and its
dedicated battle/server processes stopped successfully.

- [Full measured report](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/report.md)
- [Full measured figure](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/smoothness.png)
- [PDF figure](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/smoothness.pdf)
- [Completion verification](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/verification.json)

Mean noise-corrected squared expected win-rate change was **0.292 pp²**
(95% cluster interval **[0.001, 0.690]**) for the 50 random originals and
**18.870 pp² [12.924, 25.511]** for the 50 frozen placement entries. The group
difference was **18.579 pp² [12.698, 25.262]**. These units are squared percentage
points, not percentage points. Unchanged-team controls were consistent with zero:
random **0.403 pp² [−0.173, 1.113]**, placements **−1.673 pp² [−5.480, 2.694]**.

This supports greater local sensitivity to the sampled single-move replacements
around the retained placement snapshot under the fixed evaluator. It does not
establish mathematical non-smoothness. The random originals won only **1.845%**
of battles versus **49.950%** for placements; proximity to the win-probability
floor may partly explain their low sensitivity. Their small corrected effect
should not be treated as strong evidence on its own given the control uncertainty.
Keep the cohort and sampling qualifications above when writing the subsection.

The figure also compares the one-stage COMBO Pest Control edit and a Hartmann-6
coordinate step of 0.01, with smaller/larger benchmark steps in a separate panel.
Their normalized squared effects were **4.4482e−5 [3.6107e−5, 5.3387e−5]** and
**2.3822e−6 [8.9747e−7, 4.3902e−6]**, respectively. The normalization and different
neighborhoods make this a descriptive comparison, not a universal smoothness
ranking. No frozen protocol, policy, opponent pool or placement selection changed.
This completion update supersedes the earlier preparation and launch statuses.
