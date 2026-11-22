# Novelty search variants in pyribs

## Scope, definitions, and algorithm differences

### Takeaway
As checked on 11 November 2026, stable documentation identifies itself as v0.12.1. Plain novelty search explores behavioral differences; local competition adds pressure to outperform behaviorally similar solutions. [Stable API](https://docs.pyribs.org/en/stable/api/ribs.archives.ProximityArchive.html)

### Cited Findings
- **Novelty formula:** rho(x) = (1/k) sum(i=1..k) dist(x, mu_i), where x is the candidate's behavioral descriptor (pyribs: measures), and mu_i are its nearest neighbors in that space. The original paper calls this the “average distance to the k-nearest neighbors”; admission is above a minimum novelty threshold. Objective performance does not drive selection, though a goal test can stop a run. The printed original equation indexes 0..k despite describing k neighbors; use the unambiguous k-term formula for an explainer. [Original paper, section 4](https://gwern.net/doc/reinforcement-learning/exploration/2011-lehman.pdf)
- Original novelty search includes historical archive behaviors and discusses the current population as additional neighbor references. Pyribs instead computes against the **current archive**, using Euclidean distance and k_eff=min(k_neighbors, archive size); batch candidates are not each other's neighbors. This implementation distinction matters when drawing the neighbor set. [Original paper](https://gwern.net/doc/reinforcement-learning/exploration/2011-lehman.pdf); [stable source, compute_novelty and add](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)
- **Local competition:** LC(x)=sum(i=1..k_eff) 1[f(x)>f(mu_i)]. Equal objective values do not count. The paper defines the competition objective by counting neighbors with lower fitness and combines it with novelty through Pareto multiobjective evolution, rather than a weighted sum. Its experimental system also uses genotypic diversity as an objective, based on an NSGA-II modification. [Local competition paper, sections 3–4](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_gecco11.pdf); [stable source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)
- **Plain novelty search versus local competition:** plain novelty selects diversity without objective pressure; local competition compares performance within the behavioral neighborhood, rather than globally. [Local competition paper](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_gecco11.pdf)
- **Versus MAP-Elites:** pyribs MAP-Elites uses a GridArchive and GaussianEmitter; novelty search uses an unstructured ProximityArchive. Local competition's neighborhoods follow nearby archive points rather than a fixed grid tessellation. [Supported algorithms](https://docs.pyribs.org/en/stable/supported-algorithms.html); [archive source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)

### Inferences
- Designer suggestion: separate three operations visually: measure-space distance, archive admission/replacement, and emitter selection. They are separate computations in the source. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py); [rankers](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/emitters/rankers.py)

### Gaps
- A uniform user-supplied distance-function constructor argument for ProximityArchive: **not found**; its implementation uses KDTree Euclidean queries. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)

## ProximityArchive API, admission, replacement, and threshold decay

### Takeaway
Both stable v0.12.1 and master contain the three requested threshold-decay parameters. The fetched archive files were identical; this is a November 11 snapshot, not a claim about later November changes. [Stable source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py); [master source](https://github.com/icaros-usc/pyribs/blob/master/ribs/archives/_proximity_archive.py)

### Cited Findings
All constructor arguments are keyword-only. The following is the complete signature inventory, including the deprecated argument retained in source. [Stable constructor](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py); [API](https://docs.pyribs.org/en/stable/api/ribs.archives.ProximityArchive.html)

| Parameter | Default | Meaning |
|---|---|---|
| solution_dim | required | Solution shape/dimensionality |
| measure_dim | required | Measure dimensionality |
| k_neighbors | required | Maximum neighbor count |
| novelty_threshold | required | Admission novelty cutoff |
| local_competition | False | Enable objective comparisons and replacement |
| initial_capacity | 128 | Initial storage allocation; grows by doubling |
| qd_score_offset | 0.0 | Subtracted from objective for quality-diversity score |
| threshold_decay_rate | None | Disabled unless a rate is supplied |
| threshold_decay_itrs | 1 | Consecutive add calls without novel insertions before decay |
| threshold_decay_min | 0.0 | Threshold floor |
| seed | None | Random seed |
| solution_dtype | None | Individual dtype override |
| objective_dtype | None | Individual dtype override |
| measures_dtype | None | Individual dtype override |
| dtype | None | Combined dtype shortcut; effective default float64 |
| extra_fields | None | Additional stored field definitions |
| kdtree_kwargs | None | KDTree construction options |
| ckdtree_kwargs | None | Deprecated; non-None raises ValueError |

Table source: [v0.12.1 constructor and argument descriptions](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py).

- **Both modes:** novelty is the mean Euclidean distance to available nearest archived neighbors. Empty archive: novelty is initialized to the current threshold, local competition to zero. The admission code is exactly `novel_enough = novelty >= self.novelty_threshold`—equality passes, despite prose saying “exceeds.” All sufficiently novel batch members append, even when similar to one another. [Stable add/compute_novelty](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)
- **local_competition=False:** objective=None becomes zeros; supplied objectives can be stored and reflected in statistics but do not affect admission. Return dictionary contains `status` (int32 batch array: 0 rejected, 2 novel addition) and `novelty` (measure dtype batch array). No `local_competition` or `value` fields. No replacement. [Stable source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)
- **local_competition=True:** objectives are mandatory; objective=None raises ValueError. Compute novelty and LC for every candidate against the pre-add archive. Novel candidates append regardless of objective; non-novel candidates replace their **single closest** neighbor only if strictly better in objective. The LC count itself does not determine replacement. If several candidates target the same elite, highest objective wins; ties choose first. [Stable source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)
- **Local-competition return fields:** `status` (0 no improvement, 1 improves existing neighbor, 2 novel addition), `novelty`, `local_competition` (int32 count, 0..k_eff), and `value` (objective dtype). For status 2 value=f(candidate); otherwise value=f(candidate)-f(nearest archived neighbor). A rejected tie has value zero, although documentation calls rejection values negative. [Stable source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)
- **Batch status caveat:** status 1 is set before conflicts are resolved, so more than one candidate can receive status 1 while only the winning candidate is stored at that neighbor's index. Returned novelty, LC, and value likewise describe pre-insertion comparisons. [Stable add implementation](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)
- **Decay exists:** after `threshold_decay_itrs` consecutive add calls with zero novel additions, threshold becomes max(threshold_decay_min, threshold*threshold_decay_rate), then the counter resets. Any novel addition resets the counter; replacements alone do not. Default None disables decay. The code rejects rates <=0 or >1: effective allowed interval (0,1], despite an error/doc description saying [0,1]. This is downward decay, not automatic threshold increase. [Stable source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py); [master source](https://github.com/icaros-usc/pyribs/blob/master/ribs/archives/_proximity_archive.py)
- The exact supported-algorithms entry **“Novelty Search with Threshold Decay” is not found**. The page instead lists Novelty Search and NSLC, and gives no distinct recipe under the requested name. [Supported algorithms](https://docs.pyribs.org/en/stable/supported-algorithms.html)

### Inferences
- A threshold-decay explainer should animate a changing admission cutoff, while leaving the novelty-distance formula unchanged. This follows the separation of `_maybe_update_threshold` and `compute_novelty`. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)

### Gaps
- Original-paper adaptive-threshold numerical schedule: **not found in the inspected passages**; no rise/fall constants claimed here. The library parameters themselves are confirmed, so an older paper mechanism is not needed as a substitute. [Original paper](https://gwern.net/doc/reinforcement-learning/exploration/2011-lehman.pdf)

## Emitters, rankers, maze tutorial, and release history

### Takeaway
Use the novelty ranker for plain novelty search, and the local-competition Pareto ranker for novelty search with local competition. Threshold decay changes archive configuration, not the ranker. [Ranker source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/emitters/rankers.py); [archive source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/archives/_proximity_archive.py)

### Cited Findings
- **Plain novelty:** EvolutionStrategyEmitter with `ranker="nov"` or `"NoveltyRanker"` ranks descending by add_info["novelty"]. GaussianEmitter and IsoLineEmitter are also supported alternatives; do not imply every emitter requires a ranker. [Ranker aliases/source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/emitters/rankers.py); [supported components](https://docs.pyribs.org/en/stable/supported-algorithms.html)
- **Novelty plus local competition:** EvolutionStrategyEmitter with `ranker="nslc"` or `"NSLCRanker"`, ProximityArchive(local_competition=True). Fast non-dominated sorting maximizes novelty and LC; higher crowding distance breaks ties within fronts. `diversity_field=None` is the default; a named field adds a third maximized objective (looked up in add_info first, data second). This is Pareto ordering, not novelty multiplied by LC or a sum. [Stable rankers](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/emitters/rankers.py)
- **Three-objective variant:** `NSLCClassicRanker` / `"nslc_classic"` sets diversity_field="genotypic_diversity"; missing field raises KeyError. Its three objectives are novelty, LC, and provided genotypic diversity; source still applies crowding-distance ordering within fronts. Avoid claiming pyribs automatically computes the original creature-domain diversity signal. Both aliases also occur in master. [Stable rankers](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/emitters/rankers.py); [master rankers](https://github.com/icaros-usc/pyribs/blob/master/ribs/emitters/rankers.py)
- **Maze setup:** Kheperax policies have 138 flattened parameters; measures are final robot x,y in [0,1]^2. ProximityArchive: measure_dim=2, k_neighbors=15, novelty_threshold=0.01. Five EvolutionStrategyEmitters: sigma0=0.5, ranker="nov", selection_rule="mu", restart_rule="basic", batch_size=30 each. Scheduler joins them and a passive 100x100 GridArchive; total_itrs=2500. [Tutorial](https://docs.pyribs.org/en/stable/tutorials/ns_maze.html)
- **Reported tutorial result:** the displayed final log reports 23,633 proximity-archive entries and 88.930% passive-grid coverage at iteration 2500. Treat these as one published run, not a guaranteed benchmark average. The grid is for measurement, not search guidance. [Tutorial output](https://docs.pyribs.org/en/stable/tutorials/ns_maze.html)
- **Version history:** v0.8.0 introduced ProximityArchive for novelty search (PRs 472/479/484/521/527), NoveltyRanker (477), and local-competition support in ProximityArchive (481). v0.11.0 added NSLCRanker (690), and fixed incorrect non-novel insertion when no improving solutions exist (704). v0.12.0 added novelty_threshold decay (709). v0.12.1's listed fix concerns DensityArchive reservoir sampling. Thus archive support for local competition predates the dedicated NSLC ranker. [HISTORY.md](https://github.com/icaros-usc/pyribs/blob/master/HISTORY.md)

### Inferences
- The tutorial evaluates 150 candidates per scheduler iteration (five emitters × 30), implying 375,000 requested evaluations across 2500 iterations. This is arithmetic from configuration, not a separately reported evaluation count. [Tutorial](https://docs.pyribs.org/en/stable/tutorials/ns_maze.html)

### Gaps
- A separately logged maze result for local competition or threshold decay: **not found** in this tutorial. [Tutorial](https://docs.pyribs.org/en/stable/tutorials/ns_maze.html)
- Exact release introducing NSLCClassicRanker separately from NSLCRanker: **not found** in the inspected changelog entries; presence in v0.12.1 is confirmed. [History](https://github.com/icaros-usc/pyribs/blob/master/HISTORY.md); [stable rankers](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/emitters/rankers.py)
