# Using landscape information inside neural cross-entropy search

Checked October 20, 2026. This memo separates published mechanisms, mathematical deductions and proposed VGC changes. It assigns no new battle budget, runs no experiment and establishes no VGC improvement. The companion implementation and one-edit audits supply the project's actual settings and measurements.

## The important distinction is the landscape versus where search looks

Fix the legal candidate representation, opponent distribution, battle policies, simulator, rules and outcome convention. The target f(T) is the resulting expected score of team T. Its neighborhood must say what an edit changes, such as one whole Pokémon set, one move or one item. Changing q_t, the current candidate proposal distribution, changes visitation. It does not make this fixed f smoother or rougher.

A generation whose observed scores become less variable can therefore have several explanations: proposals concentrate in a small region, the selected region has small true differences, the evaluator cannot resolve differences at its current precision, or a surrogate/acquisition filter hides other candidates. These are different diagnoses. Looking only at one generation's score variance cannot identify which occurred.

The existing one-edit measurements should be interpreted for the exact edit operator, base-team distribution and evaluator they measured. Small average changes do not imply global unimodality, additivity, low epistasis, or a guarantee that the best team can be reached by improving one edit at a time. A whole-set replacement is also a different neighborhood from a single item or move replacement.

## Classical CEM is background, not a description of the neural code

For a binary product-model illustration, classical CEM selects a high-scoring fraction, estimates each component's frequency in those elites, and smooths the new frequencies with the old ones. Costa, Jones and Kroese state the update p_t,i = (1-alpha_t) p_(t-1),i + alpha_t w_t,i. Their analysis explicitly assumes exact deterministic objective values and addresses acceptance-rejection constraints; it does not prove convergence of noisy neural VGC search. [Primary paper, section 2 and algorithm 1](https://people.smp.uq.edu.au/DirkKroese/ps/CEconv.pdf), [author publication record](https://people.smp.uq.edu.au/DirkKroese/publications.html).

The integrator's code audit instead identifies a masked neural generator ensemble refitted from original checkpoints using a selected training pool. Some variants use a cumulative raw-score threshold with a minimum-size fallback; another uses a previous-generation fraction. See the companion current-code audit for the exact run. A textbook frequency update is not an existing knob in that implementation.

Two consequences follow from the classical formula, but their scope matters:

- A strictly increasing score transform preserves order-based elites when ties and the selection rule are preserved. Absolute edit changes can change under a nonlinear transform. Therefore absolute smoothness is not, by itself, the same as ease of elite selection. A fixed raw-score threshold, as used by some current variants, must also be transformed to retain the same training-pool membership.
- A factorized proposal family need not imply an additive objective. Conversely, a smooth measured objective does not prove an independent proposal family is adequate. The current neural generator can represent dependencies, but whether its training pool and fitting procedure learn useful dependencies remains empirical.

The categorical version of the frequency example replaces binary means with per-category elite frequencies. This is explanatory background, not a recommendation to replace the current neural sampler. Vaisman's author-hosted tutorial gives an accessible derivation and the smoothed binary algorithm. [Slides 21 through 24](https://people.smp.uq.edu.au/RadislavVaisman/Talks/CrossEntropyTutorial.pdf).

## Start with diagnostics that can reuse existing evaluations

Log each stage separately: proposed, legal, acquired for battle, eligible for training, and retained as a finalist. Keep unavailable battle results missing. A predicted score for an unbattled proposal is not a recorded win rate.

| Diagnostic | What to record or recompute | Decision it could inform |
| --- | --- | --- |
| Proposal concentration | Exact duplicate rate and the distribution of edit distance to previous eligible teams. Keep operator and source-generator labels. | Whether the sampler is revisiting a narrow region. Marginal token entropy is not the full team's joint entropy. |
| One-edit response | Changes between a parent and its recorded legal neighbors, separated by edit type and base-team quality. | Whether a local kernel has a plausible benefit near the teams search actually visits. Global-average smoothness may not describe the upper-score tail. |
| Selection reliability | Repeated observations of the same teams, or matched per-battle outcomes where retained. Check scores near the actual threshold or elite boundary. | Whether retraining is amplifying uncertain membership decisions. Do not invent repeat observations if only one aggregate score was stored. |
| Surrogate ordering | Predicted versus observed differences on the subset actually battled, separated by local and global proposals. | Whether local edits exploit model errors rather than improve battles. Acquired data are selection-biased. |
| Coverage and transitions | Candidate identity, provenance, operator, iteration and actual recorded parent-child transitions. | Whether different proposal rules visit different regions or repeatedly return to the same states. Unlogged parentage cannot be reconstructed as observed fact. |

Search trajectory networks explicitly summarize visited search states and transitions. Their p-median example supplies a discrete subset setting. A trajectory is a record of an algorithm's path, not a complete map of basins, and graph construction does not itself improve search. [Ochoa, Malan and Blum, sections 3 and 5](https://www.storre.stir.ac.uk/bitstream/1893/32613/1/stns_asoc_2021.pdf).

A log-only audit consumes computation but need not launch new battles when the required fields exist. Any missing local probes, repeated battles or additional validation have a real cost and require an applicable experiment allocation.

## Diagnose noise before interpreting a small local change

Write the recorded difference as delta_hat = f_hat(T') - f_hat(T). Its variance is Var[f_hat(T')] + Var[f_hat(T)] - 2 Cov[f_hat(T'), f_hat(T)]. This is a variance identity, not a project noise estimate.

Matched opponents and appropriately coupled randomness can make that covariance useful. Reusing a seed does not automatically provide useful coupling when simulation branches consume randomness differently. Common random numbers have conditions under which they reduce comparison variance; they are not a universal guarantee. [Glasserman and Yao, publisher abstract](https://pubsonline.informs.org/doi/10.1287/mnsc.38.6.884).

If local differences are unresolved, calling the region neutral or declaring a local optimum is too strong. That observation may justify studying selection stability, but it does not justify assuming the objective is flat. Likewise, showing that a fixed neural pool includes several borderline scores does not reveal which are truly above threshold without adequate evidence.

There is a primary literature specifically on selecting an elite subset from noisy simulation outputs under a fixed total allocation. Chen, He, Fu and Lee's workshop paper formulates correct top-m subset selection and explicitly motivates it with CEM. Its derivation assumes independent normally distributed simulation outputs, so its allocation rule is not a drop-in guarantee for correlated, stratified battle outcomes. It is a source for the problem distinction, not a new controller recommendation. [2007 primary paper, sections 1 and 2](https://higherlogicdownload.s3.amazonaws.com/INFORMS/ef27cc87-0593-4a9b-85b3-52b5bbeae306/UploadedImages/10.pdf).

## The most direct first algorithm test is a proposal mixture

If the recovered measurements show useful local moves near current strong candidates, test a mixture of the existing sampler and a controlled local kernel. This is a proposed variant, not a change already implemented.

Let K_r(T|A) complete a legal candidate while fixing the unedited part of anchor A and allowing exactly the specified r blocks to change. Its actual support must be checked: mask probabilities do not guarantee exactly r changed Pokémon, and repair can change additional fields. Define Q_t(T) = (1-lambda) q_t(T) + lambda E_A[K_r(T|A)], with anchors selected by a declared rule from the current eligible set.

For a clean first comparison, keep pool construction, neural fitting, acquisition, legality handling and final evaluation fixed. Replace a declared share of ordinary proposals with local proposals rather than adding free candidates. If a pooled surrogate acquisition step is retained, log how often each source survives it. Changing to source-specific acquisition quotas would be another algorithmic change.

```text
load one explicitly selected baseline run configuration
for each generation:
    reserve the same total proposal attempts as the baseline
    make the declared local share using logged anchors and one edit rule
    make the remaining proposals using the baseline sampler
    apply the same legality and acquisition procedure
    spend from the same applicable battle allocation
    log source, anchor, actual edit distance and outcomes where measured
    construct the training pool using the baseline rule
    refit using the baseline checkpoint-reset and training procedure
select finalists before independent evaluation
```

This asks whether local proposals improve the output of the existing search pipeline at matched cost. It does not isolate every interaction with the surrogate or prove a global local-search advantage. Report elapsed time too, since proposal costs can differ.

Amaya, Cotta and Fernández-Leiva provide a concrete CE-plus-local-search precedent. They refine selected elites with a local-search probability and an evaluation cap, then use the resulting candidates in their CE scheme. Their tool-switching objective is deterministic permutation scheduling. The precedent supports a bounded hybrid design, not a VGC win-rate claim or an automatic choice of refinement settings. [Primary paper, section 3](https://link.springer.com/content/pdf/10.1080/18756891.2013.792542.pdf).

## Other diagnoses should lead to different tests

| Supported observation | Candidate response to test | What it would not establish |
| --- | --- | --- |
| Many exact repeats, narrow empirical support and no reliable independent progress | A proposal-level mixture with the original generator, a restart component, or less aggressive neural refitting. Change one control first. | Zero progress does not certify a local optimum. Weight averaging is not equivalent to mixing model-generated candidates. |
| Stable benefits for joint edits but weak or harmful single edits | A grouped-edit kernel or conditional completion that preserves the relevant context. | One pair interaction does not establish a globally decomposable objective or independence of other slots. |
| Small differences with unstable threshold or elite membership | Study pool-selection stability or a separately specified uncertainty-aware selection rule. | Fewer elites or stronger fitting does not remove label noise. A changed selection rule is a separate ablation from changed proposals. |
| Good predicted local improvements that disappear in battles | Audit local surrogate ranking, representation and out-of-distribution proposals before adding stronger guidance. | High global surrogate correlation need not validate the local or near-threshold ranking. |
| Several separated good groups whose components fail when recombined | Preserve multiple anchors or use conditional proposals, then test their contribution. | Elite correlation alone does not prove causal fitness interaction; selection can create dependence. |

For pair edits A and B, the four-team contrast f(AB) - f(A) - f(B) + f(base) measures an interaction in that context. In battle data it is an estimated contrast with uncertainty. It is not a full decomposition of six-Pokémon performance. A single average interaction or a few successful neighbors cannot validate every future grouping rule.

Use calibration runs or already available exploratory data to formulate any adaptive rule, then freeze it before the held-out comparison. Do not invent a universal trigger such as a particular entropy value or number of stagnant generations. A controller earns its place only if its response beats the same-cost baseline on independently evaluated win rate across the applicable comparison tasks and seeds.

## Recommended sequence

First reproduce the user's existing one-edit measurements and identify what neighborhood and score precision they actually support. Next add the available read-only diagnostics to the verified neural loop. If those diagnostics support local proposals, run the bounded mixture comparison once its actual configuration and allocation are selected. Try training-strength, restarts or linkage changes as separate hypotheses, not as a bundle whose cause cannot be identified.

No new reading-list entries are required to explain this sequence. The already added STNs and CE-memetic papers cover the first two useful mechanisms; the CEM and noisy-subset sources above supply background and explicit assumption limits.
