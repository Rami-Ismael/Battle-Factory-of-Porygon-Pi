# DeepHyper for constrained VGC team search

Reviewed: 2026-09-30. This is a feasibility assessment and proposed experiment, not an integration or benchmark result. The inspected `latest` documentation identifies itself as DeepHyper v0.13.2-; pin and recheck an actual release before implementation.

## Why this is relevant

**DeepHyper is a useful candidate for the project's outer team-search loop and a baseline for judging whether diffusion proposals help.** Its homepage describes black-box optimization with a cheap surrogate that learns from expensive evaluations, and its CBO API provides a manager–worker Bayesian optimization loop with parallel evaluation workers. That matches the proposed cycle of suggesting teams, running battles, learning from outcomes, and selecting new teams. [DeepHyper homepage](https://deephyper.github.io/), [CBO API](https://deephyper.readthedocs.io/en/latest/_autosummary/deephyper.hpo.CBO.html).

The local [[Problem Statement]] describes both team construction and battle-policy work, including ranking teams against a dated meta. [[Surrogate Model]] proposes win-rate prediction, and [[Generate random Legal Pokemon team]] connects hierarchical legal sampling to active learning. These notes describe intended work; they do not establish a completed battle evaluator or trained surrogate. My inference is that DeepHyper can organize the outer experiment, while the project must supply team encoding, regulation-specific legality, opponents, and battle policies.

## What the linked rejection example does

The tutorial maximizes `x² + y²` with `x,y ∈ [-10,10]` and `|x| + |y| ≤ 10`. `HpProblem.set_constraint_fn` receives a DataFrame and returns a feasibility mask; CBO uses `acq_optimizer="ga"`. Rejection sampling proposes from the unconstrained space and rejects infeasible candidates. The documentation explicitly warns that this can become computationally intractable. [Rejection-sampling tutorial](https://deephyper.readthedocs.io/en/latest/examples/examples_bbo/plot_constrained_black_box_optimization.html).

For our proposed representation, rejecting independent species/move/item choices could require many attempts if legality is rare. This is a conditional risk, not a measured property of this repository. With independent attempts and acceptance probability `p`, the expected proposal count per accepted sample is `1/p`; that calculation concerns proposal cost, not battle count. Rejecting illegal teams also says nothing about whether accepted teams are competitive.

## The custom sampler is the closer match

A separate, already available tutorial uses `HpProblem.set_sampling_fn`: its callable takes a requested size and returns a list of parameter dictionaries. It samples increasing integer sequences legally by construction, then uses an Extremely Randomized Trees (`ET`) surrogate and `mixedga` acquisition optimization. Despite the rejection page's “coming soon” wording, this tutorial exists. [Custom-sampler tutorial](https://deephyper.readthedocs.io/en/latest/examples/examples_bbo/plot_constrained_black_box_optimization_chained_sampler.html).

Crucially, that example retains a constraint function and checks the objective's inputs, returning `"F_constraint"` for invalid candidates. Its explanation identifies manually constructed inputs and mutation-based acquisition optimizers as possible sources of infeasibility. Thus a legal sampler alone does not guarantee every subsequent proposal is legal. [Custom-sampler tutorial](https://deephyper.readthedocs.io/en/latest/examples/examples_bbo/plot_constrained_black_box_optimization_chained_sampler.html).

Proposed adaptation: wrap the project's hierarchical sampler to return encoded teams, keep an independent complete validator immediately before any battles, and ensure acquisition-generated candidates also pass it. A diffusion generator could later supply proposals through an adapter, but compatibility, coverage, and computational cost remain untested. The tutorial demonstrates a small integer constraint, not Pokémon teams or diffusion integration.

## Representation and noise

`HpProblem` accepts integer, real, and categorical parameters, plus ConfigSpace conditions and forbidden clauses. These provide possible building blocks for team fields and dependencies; they do not supply Pokémon rules. [HpProblem API](https://deephyper.readthedocs.io/en/latest/_autosummary/deephyper.hpo.HpProblem.html). CBO's `mixedga` supports continuous, discrete, and categorical variables. Its API warns that the GP surrogate does not support constrained search spaces; ET is a documented starting point. [CBO API](https://deephyper.readthedocs.io/en/latest/_autosummary/deephyper.hpo.CBO.html).

Representation still needs design: species-conditioned choices, shared team constraints, and duplicate encodings of the same team can all affect search. [[Diffusion Model Constraint Generation]] already distinguishes canonical ordering from legality. Start with a restricted set of known legal Pokémon sets, then decide whether expanding to individual fields improves results.

DeepHyper's noisy-objective example uses `ArgMaxEstSelection(..., noisy_objective=True)` to select by a surrogate estimate, and reports estimated aleatoric and epistemic uncertainties. This demonstrates machinery for noisy optimization, not calibrated battle-win uncertainty or automatic allocation of game repetitions. [Noisy-optimization tutorial](https://deephyper.readthedocs.io/en/latest/examples/examples_bbo/plot_black_box_optimization_noisy.html).

## Proposed first comparison

Use the same restricted search space, fixed policies, opponent snapshot, and total battle budget for legal random sampling, the existing proposed legal-mutation baseline, and DeepHyper CBO with legal sampling. Add diffusion proposals only after that comparison works. Measure independently reevaluated win rates of the best teams, legality acceptance, duplicate rate, battles consumed, and proposal/surrogate overhead across several search seeds. Reserve fresh battle seeds for finalist verification; keep failed infrastructure jobs distinct from invalid teams.

This experiment would answer whether surrogate-guided search improves the project's current proposed baselines, then whether diffusion supplies additional value. No package was installed, no benchmark was run, and no performance or ladder-success claim follows from these tutorials.
