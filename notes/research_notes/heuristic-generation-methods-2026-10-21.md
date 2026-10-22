# Methods that generate heuristics

Research checked on 21 October 2026. This note explains methods and implementation choices. It does not report a Pokémon experiment.

The terms to search for are **heuristic generation**, **generation hyper-heuristics**, **automatic heuristic design**, and **automated algorithm design**. A heuristic is a rule that guides a decision, such as which candidate to evaluate, which change to try, or when to restart a search. A generation hyper-heuristic searches for that rule. The result can be a formula, a small program, or a learned decision function.

The distinction is what the outer search changes. A selection hyper-heuristic chooses among existing heuristics. A generation hyper-heuristic constructs heuristics from components. Both operate above the underlying problem's solutions. Evolving Pokémon teams directly therefore differs from evolving a rule that searches for teams. This follows the classification in [Burke et al.'s hyper-heuristics survey](https://people.cs.nott.ac.uk/pszrq/files/HHSurveyJORS2013.pdf).

## Main approaches

These are useful starting families, rather than an exhaustive taxonomy. The suitability column is a recommendation based on their representations and implementation requirements.

| Approach | What it generates | What you provide | When it is useful |
|---|---|---|---|
| Genetic programming | Expression trees or small programs | Available inputs, permitted operations, a fitness evaluator | You want inspectable rules with a constrained vocabulary |
| Grammatical evolution | Programs allowed by a formal grammar | A grammar, an evaluator, search settings | You want conditions, formulas, or operator sequences with a chosen structure |
| LLM program evolution | Executable heuristic functions, sometimes with written explanations | A function contract, problem description, examples or seed code, evaluator, LLM access | You want to explore code beyond a small hand-built grammar |
| Neural or reinforcement learning methods | A learned decision policy, or a generator of rules if explicitly designed that way | State representation, available actions, reward, training instances | You can afford repeated training and accept a learned model as the heuristic |
| Algorithm configuration | Parameter settings or choices among predefined components | An existing algorithm, permitted settings, evaluator | You mainly need better weights or settings; this alone does not invent new code |

The mechanisms and available implementations are detailed below.

## Genetic programming and symbolic rules

Represent a heuristic as a tree. Its leaves are measurable features or constants; its internal nodes are operations such as addition, minimum, comparison, and conditional choice. Evolution selects effective trees, mutates subtrees, and combines parts of parent trees. The retained tree becomes the decision rule.

[DEAP's genetic programming tools](https://deap.readthedocs.io/en/stable/tutorials/advanced/gp.html) support typed inputs and operations and can compile trees into callable functions. A practical implementation defines the allowed feature inputs, builds a typed primitive set, and evaluates each compiled rule inside the same fixed search procedure. Bound tree size and depth so evolution does not produce increasingly large rules without useful gains.

Searching formulas this way overlaps with symbolic regression. The training target matters. Fitting a formula to predict observed win rates produces a predictor. Scoring the formula by the quality of the decisions it causes produces a decision heuristic. These are different experiments, even if they use the same expression representation.

## Grammatical evolution

A grammar specifies which programs the search can construct. For example, it can permit an arithmetic priority formula, or a conditional rule that chooses an edit and then a candidate. The evolutionary search changes a representation that maps through that grammar into a program.

[PonyGE2's paper](https://arxiv.org/abs/1703.08535) describes this genotype-to-program mapping and its Python search engine; the [official repository](https://github.com/PonyGE/PonyGE2) provides the implementation. You supply a grammar and a task-specific fitness function. Grammar design limits both errors and discoveries. A grammar containing only linear weighted sums cannot discover an arbitrary branching algorithm. Syntactically valid output still needs numerical and runtime checks.

## LLMs that write and evolve heuristic code

The useful pattern is an execution loop. An LLM proposes code, an evaluator measures its behavior, and a search procedure chooses which programs inform the next proposals. Asking once for a clever heuristic supplies no evidence that it works.

**FunSearch** keeps a program skeleton fixed and evolves a selected function. Its paper demonstrates priority functions inside greedy procedures. The authors identify an efficient evaluator, informative numerical feedback, and an isolated function to evolve as helpful conditions. This makes a small scoring function a sensible first target. See the [Nature paper](https://doi.org/10.1038/s41586-023-06924-6).

The [official FunSearch repository](https://github.com/google-deepmind/funsearch) includes search and code manipulation components and a single-threaded pipeline. It explicitly excludes the language models, the sandbox for generated code, and the original distributed infrastructure. It is useful reference code, with integration work still required.

**Evolution of Heuristics, EoH**, evolves both a natural-language idea and its executable implementation. The idea accompanies the code as candidates are generated and evaluated. See the [ICML 2024 paper](https://proceedings.mlr.press/v235/liu24bs.html). The [official EoH repository](https://github.com/FeiLiu36/EoH) supplies example problems and instructions for adding a problem class and configuring a remote or local LLM. Its framework is a practical starting point for an experiment with an explicit function contract and objective.

**ReEvo** adds reflection to evolutionary program search. It compares candidate programs and their measured outcomes, creates short-term feedback for crossover, and accumulates longer-term feedback for mutation. The explanations guide proposals; measured fitness still decides whether they helped. See the [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf). Its [official repository](https://github.com/ai4co/reevo) documents the custom problem configuration, evaluation pipeline, function description, signature, and seed function required for adaptation.

These papers show successful experiments on their studied tasks. They do not establish that any of these frameworks will improve Pokémon battle win rate, or that ReEvo will beat EoH in this project.

## Learning a neural heuristic

You can train a model to choose the next construction or improvement action using reinforcement learning. For example, [Kool et al.](https://arxiv.org/abs/1803.08475) train an attention model with REINFORCE to construct routing solutions. The [authors' implementation](https://github.com/wouterkool/attention-learn-to-route) is available.

Here, the reusable heuristic is the trained model and its inference procedure. It need not produce human-readable source code. A model that directly outputs solutions is not automatically a generation hyper-heuristic under the classical classification. To generate explicit rules, design its output as a program or expression in a specified language. To select among existing search operators, design its actions as operator choices, which gives a selection hyper-heuristic.

## Configuration is a useful comparison

If a useful rule already exists and only its weights or thresholds are uncertain, tune those first. The [irace documentation](https://iridia.ulb.ac.be/irace/README.html) describes automated configuration over training instances. With a fixed formula, this searches settings within that formula. Exposing categorical algorithm components broadens what configurations can express, but it still cannot invent components absent from the supplied space.

## A minimal generation experiment

The following is an implementation recommendation, rather than a claim from a particular paper.

1. Choose one decision to automate and define its inputs and output precisely.
2. Keep the search procedure and final objective fixed while changing that decision rule.
3. Supply a simple baseline rule and a rule generator, initially typed genetic programming or an EoH-style loop.
4. Reject invalid outputs and run generated programs with isolation, time limits, and resource limits. A timeout alone is not code isolation.
5. Score the resulting search outcomes under equal inner-search budgets. Record generation cost separately.
6. Use repeat evaluations where outcomes are noisy. Keep development, validation, and final test instances separate; repeated selection can overfit an evaluator.
7. Export the selected formula, code, or model with its inputs, seeds, evaluation protocol, and source revision. Recheck its benefit on untouched cases.

The key cost is nested evaluation. If one heuristic needs many full search runs to assess it, searching over many heuristics multiplies that work. Start with a small decision and a trustworthy evaluator before expanding to whole-algorithm generation.

## Application to the Pokémon counter-search project

You already have a working example of the central idea. Your [LLM prompt](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/hyperheuristic.py:265>) requests new Python functions with the contract `keep(team) -> bool`. The available helpers describe typing, coverage, move roles, and stats. The [generation loop](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/hyperheuristic.py:422>) proposes functions, evaluates them, and saves a portfolio. This is a project-specific LLM generation hyper-heuristic. It has not been established as a reproduction of FunSearch, EoH, or ReEvo.

Its [fitness evaluator](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/hyperheuristic.py:201>) measures two retention rates:

- `r` is the fraction of collected competitive teams that pass the rule.
- `q` is the fraction of sampled random legal teams that pass the rule.

High `r` and low `q` describe a filter that preserves the competitive sample while pruning much of the sampled legal space. They do not demonstrate higher battle win rate. The [source's own caveat](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/hyperheuristic.py:19>) identifies the risk: a filter can preserve known teams while excluding the unusual counter-team you want to discover.

My recommendation is to reuse this proposer loop for a narrow search decision, then score its effect on battle outcomes. A concrete first target is a rule that chooses which team slots to regenerate. The separate [mutation experiment](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/scripts/diffusion_mutation_arm.py:109>) already isolates mutation choices and refill methods. The proposed rule would return a valid slot mask; the existing machinery would refill and validate it. This lets you generate a reusable heuristic rather than another individual team.

This is a proposed interface, not an existing function:

```python
choose_mask(team_features, opponent_features, search_state) -> slot_mask
```

Use only features already available at that decision, or account for their computation cost. Keep legal-team validation and battle scoring outside the generated function. A priority formula that ranks proposed teams is another option. However, the [current DiBO ranker](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/dibo_core_v4.py:234>) uses its specified target score, so replacing it would define a different search variant and needs a separate comparison.

The implementation has two nested loops:

```text
Outer loop: propose a heuristic, run searches with it, receive results, revise it.
Inner loop: use that heuristic to guide a search that produces a candidate team.
Fitness: measure the resulting team's battle win rate on fresh development battles.
```

Compare complete search runs using the same starting inputs, battle policies, opponent weights, and inner-search budget. Assess more than one search seed. The [existing mutation pilot protocol](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/diffusion-mutation-arm.md:14>) provides a concrete precedent, including separate finalist verification. Its settings belong to that pilot; they do not settle the budget for a new outer heuristic search.

Develop heuristics on development runs, choose one using validation runs, then freeze its code before final testing. Keep final battle results out of both heuristic revision and team selection. For one fixed counter target, retain that target and use independent search and battle seeds. Use unseen opponents only when testing a claim that the heuristic transfers to other targets. A new experiment still needs a recorded opponent set and comparison budget; this note does not invent either.

Report the final teams' win rates and variation across independent searches. Also record the cost of discovering the rule, including failed candidates, battle evaluations, and LLM calls, separately from the cost of using the frozen rule. A fast final heuristic can still be expensive to discover.

For your first implementation, I would reuse the existing LLM loop rather than migrate frameworks. Give it the small mask-selection contract and a downstream search evaluator. Then compare it with typed genetic programming over the same decision, if you want an alternative that generates inspectable rules without LLM calls. Neither approach has an established win-rate advantage in this project yet.
