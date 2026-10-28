---
created_at: 2026-08-14
updated_at: 2026-10-27
type: curriculum
hub: "[[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]]"
verdict: study methods against the actual team-search problem; coverage and ridge are not established requirements
canvas: "[[Online course pathway.canvas]]"
category_theory: skipped — [[Do I need category theory]]
rewritten: corrected coverage-versus-team-building mismatch and removed generic course requirements
tags:
  - curriculum
  - self-study
  - combinatorial-search
  - ridge-regression
  - submodularity
  - reinforcement-learning
  - problem-sets
---

## Purpose

The goal is a legal six-Pokémon team that performs well against the chosen opponents. This guide assumes UT Dallas intro data science and machine learning, and covers only what those courses skip: stating the search, comparing candidates and reading the methods on the table.

See [[Problem Statement]] for the project's scope. The resources below are free online or already in the vault.

## Find the relevant material

| Subject | Question it helps answer | Study when |
| --- | --- | --- |
| Discrete search and legality | What can I change in a candidate, and which changes remain legal? | Designing candidate generation and search moves |
| Evaluation | How do I tell whether one candidate performs better? | Comparing search results |
| Gaussian processes and Bayesian optimization | Can previous evaluations help choose the next candidate to evaluate? | Investigating a surrogate-based search method |
| Reinforcement learning | How does the battle policy learn to play the team? | Working on the battle-policy experiment |
| Submodularity | Does a particular set objective have diminishing returns? | Understanding the lecture or checking a proposed formulation. Unproven for team performance |
| Ridge regression and quadratic forms | Would a simple fitted model help, and what does it represent? | A specific model or paper calls for them |

Study the sections in any order. Only the Gaussian process chapters build on each other.

## Submodularity

Watch 🟢 [Fabien Mathieu's Introduction to Submodular Functions](https://www.youtube.com/watch?v=EIoUwkwoiu0), then read section 1 of 🟢 [Krause and Golovin's survey](https://las.inf.ethz.ch/files/krause12survey.pdf). Learn the discrete derivative, monotonicity and diminishing returns before continuing to the greedy algorithm.

For a set score $f$, the discrete derivative is the gain from adding one element:

$$
\Delta(e\mid S)=f(S\cup\{e\})-f(S).
$$

**Monotone** means this gain is always nonnegative. **Submodular** means the gain cannot increase when the existing set grows: for $A\subseteq B$ and $e\notin B$, $\Delta(e\mid A)\geq\Delta(e\mid B)$. These are different properties. A submodular function can still have negative gains.

Replacing a Pokémon removes one element and adds another. A worse result after a replacement does not, by itself, disprove monotonicity. Also, if the score is defined only for complete six-Pokémon teams, you must define scores on smaller subsets before the addition-based definition applies.

This project has to keep two problems apart. Selecting a collection of complete teams is a coverage problem. If each team covers a fixed set of opponents, the nonnegatively weighted union of those sets is monotone and submodular.

Building one team is different. Its performance depends on the whole composition and on how the policy plays it. Nobody has shown that fixed coverage sets for individual Pokémon model that performance.

**Submodularity of the team-performance objective remains unproven.** The greedy $1-1/e$ guarantee applies to a normalized, nonnegative, monotone submodular objective under a cardinality limit. To use it here, first prove those assumptions hold for the chosen formulation.

For practice, work exercises 2.10–2.11 in 🟢 [Williamson and Shmoys](https://www.designofapproxalgs.com/). Explain what the selected elements represent in that exercise. For more proofs, see 🟢 [Jan Vondrák's notes](https://theory.stanford.edu/~jvondrak/CS369P/).

## Gaussian processes and Bayesian optimization

A Gaussian process predicts a score and how unsure that prediction is. Bayesian optimization uses the model to pick the next candidate to evaluate. It pays off when each evaluation is expensive.

Watch from [[Gaussian Process|your video collection]] alongside [[Gaussian Processes by Mutual Information|your existing notes]]. The collection is unranked.

Use 🟢 [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/) in this order:

1. Chapter 2 connects familiar regression to Bayesian regression, then to the function-space view. Learn how observations change the predictive mean and uncertainty.
2. Chapter 4 covers what a kernel assumes about how inputs relate. Similarity between legal teams is a modelling choice. A kernel does not discover a useful representation automatically.
3. Read chapter 5 once you need to fit kernel hyperparameters.

## Candidate generation and legality

Start from [[Clause or the feature of the search space]] and [[hierarchical product sampling]]. For each search move, write down the field it changes, such as species, moves or item, and which dependent choices need a fresh legality check.

To practice, test a rule with one valid candidate and one invalid candidate. Before merging two representations into one canonical form, check that they really are equivalent. If slot order changes how the policy plays, the order matters and stays.

## Comparing candidates

Before comparing methods, specify the opponent teams and their weights, the battle policies, the evaluation budget and how you aggregate results. Hold them fixed across every run you compare, or a gain may come from the setup and not the method.

If you fit a predictor, decide what must generalize: another battle involving familiar teams, unseen teams, or a changed opponent distribution. Choose the data split to match that question. Good scores on a random split do not show that the predictor handles unseen teams.

For validation methods, read chapter 5 of 🟢 [An Introduction to Statistical Learning, free Python edition](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html) or [[Inteligent Systems/Richard_Golden_-_Statistical_Machine_Learning__A_Unified_Framework (2).pdf|Golden chapter 16]].

## Learning a battle policy

This branch supports [[Reimplement VGC-Bench — run the release, reproduce BC, self-play, fictitious play, double oracle, then extend on team-pool size]]. A matchup predictor estimates an outcome. The battle policy chooses actions during play.

Use 🟢 [David Silver's lectures](https://www.davidsilver.uk/teaching/), lectures 1–3 for the decision-process framework, then 4–7 for prediction, control, function approximation and policy gradients as needed. Pair them with [[Reinforcement Learning an Introduction SEcond Sedition Richard S. Sutton and andrew G. Barto.pdf|Sutton and Barto]]: chapters 3–4, 5–6 and 13 respectively. For practice, do Silver's Easy21 assignment and the book exercises. [[The Little Book of reinforcement Learning.pdf]] gives a quick overview.

Before training, answer three questions: what can the policy observe, what signal does it learn from, and which opponents does it train and evaluate against?

## Optional references

- **Ridge and interaction features.** Chapters 6 and 3 of 🟢 [An Introduction to Statistical Learning](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html), or [[Inteligent Systems/Richard_Golden_-_Statistical_Machine_Learning__A_Unified_Framework (2).pdf|Golden chapter 13]]. Read these when you evaluate a ridge predictor.
- **Quadratic notation.** 🟢 [Glover, Kochenberger and Du](https://arxiv.org/abs/1811.11538). Read it when a paper writes $x^\top Qx$. Check what the pairwise terms assume about team performance.
- **Matrix operations.** 🟢 [Boyd and Vandenberghe](https://web.stanford.edu/~boyd/vmls/), 🟢 [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), or Golden chapters 4–5. Look up only the operation a derivation is stuck on.
- **Further statistical or optimization theory.** 🟢 [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/download.html), 🟢 [Stanford Convex Optimization I](https://web.stanford.edu/class/ee364a/) and its 🟢 [exercise collection](https://github.com/cvxgrp/cvxbook_additional_exercises). Open these only when a specific derivation or proof needs them.

Two study maps still need revising: [[Online course pathway.canvas]] and [[Online study flow — free resources.excalidraw]].

<details>
<summary>Earlier decisions and task history</summary>

## Retired

- **Quadratic unconstrained binary optimisation and Ising hands-on — D-Wave Ocean and Leap** — *Retired 2026-08-24.* Nothing in this project anneals; quadratic unconstrained binary optimisation survives as a model class only.
- **Game-theory tier** — *removed 2026-08-17*, served a framing the project dropped.
- **Synergy caveat** — *removed 2026-08-24*; measuring synergy is a research problem of its own and is out of this project.

# Todo

- [ ] Create a cavas to test my prompting skills for creating canva for this thing
- [ ] /goal I want to create a tree diagram to show the pathway to understand the courses I have to take that are all online to understand effifective understand what is going on pokemon using obsidian.cavas
- [x] Add two new books in reinforcement learning start with [[The Little Book of reinforcement Learning.pdf]] and [[Reinforcement Learning an Introduction SEcond Sedition Richard S. Sutton and andrew G. Barto]]
- [ ] Add some online course you take 
- [x] Check if there something from brilliant for stuff — done: checked, then dropped 2026-08-24; vault textbooks used instead — the vault textbooks listed above
- [x] I am person need lot of problem to understand what is going on — done: the vault textbooks listed above — Sutton and Barto (145 exercises) and Golden (54 exercise sections), both already in the vault 
- [x] Can find other source of problem to practice the new concept I will be learning. — done: the exercises linked in this guide
- [x] Stop using abbraviation — done: this note swept 2026-08-24 — every shortened word spelled out, including the relevance grade MEDIUM; standing rule from here on

</details>
