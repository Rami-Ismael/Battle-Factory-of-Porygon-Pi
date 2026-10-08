---
created_at: 2026-08-14
updated_at: 2026-09-07
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

Find a **legal six-Pokémon team that performs well against the chosen opponents**. This guide assumes introductory data science and machine learning at UT Dallas. It focuses on the additional concepts needed to formulate the search, compare candidates and understand the methods being considered.

See [[Problem Statement]] for the project’s scope. The resources below are free online or already in the vault.

## Find the relevant material

| Subject | Question it helps answer | Study when |
| --- | --- | --- |
| Discrete search and legality | What can I change in a candidate, and which changes remain legal? | Designing candidate generation and search moves |
| Evaluation | How do I tell whether one candidate performs better? | Comparing search results |
| Gaussian processes and Bayesian optimization | Can previous evaluations help choose the next candidate to evaluate? | Investigating a surrogate-based search method |
| Reinforcement learning | How does the battle policy learn to play the team? | Working on the battle-policy experiment |
| Submodularity | Does a particular set objective have diminishing returns? | Understanding the lecture or checking a proposed formulation; applicability to team performance is unproven |
| Ridge regression and quadratic forms | Would a simple fitted model help, and what does it represent? | A specific model or paper calls for them |

Each section can be studied independently, except for the stated prerequisites within the Gaussian process material.

## Submodularity

Watch 🟢 [Introduction to Submodular Functions — Fabien Mathieu](https://www.youtube.com/watch?v=EIoUwkwoiu0), then read **section 1** of 🟢 [Krause and Golovin’s survey](https://las.inf.ethz.ch/files/krause12survey.pdf). Learn the discrete derivative, monotonicity and diminishing returns before continuing to the greedy algorithm.

For a set score $f$, the discrete derivative is the gain from adding one element:

$$
\Delta(e\mid S)=f(S\cup\{e\})-f(S).
$$

**Monotone** means this gain is always nonnegative. **Submodular** means the gain cannot increase when the existing set grows: for $A\subseteq B$ and $e\notin B$, $\Delta(e\mid A)\geq\Delta(e\mid B)$. These are different properties; submodularity alone does not require positive gains.

Replacing a Pokémon removes one element and adds another. A worse result after a replacement does not, by itself, disprove monotonicity. Also, if the score is defined only for complete six-Pokémon teams, the addition-based definition cannot be applied directly without defining scores on other subsets.

The distinction that matters for this project is:

- **Selecting a collection of complete teams:** if each team has a fixed set of opponents it covers, the nonnegatively weighted union of those sets is monotone and submodular.
- **Building one team:** its performance depends on the complete composition and how it is played. Fixed coverage sets for individual Pokémon have not been justified as a model of that performance.

**Submodularity of the team-performance objective remains unproven.** The greedy $1-1/e$ guarantee applies to a normalized, nonnegative, monotone submodular objective under a cardinality limit. Applying it here would require establishing those assumptions for the chosen formulation.

**Practice:** work exercises 2.10–2.11 in 🟢 [Williamson and Shmoys](https://www.designofapproxalgs.com/) to test your understanding of submodularity and coverage. Explain what the selected elements represent in that exercise. For further proofs, use 🟢 [Jan Vondrák’s notes](https://theory.stanford.edu/~jvondrak/CS369P/).

## Gaussian processes and Bayesian optimization

A Gaussian process supplies predictions and uncertainty. Bayesian optimization uses a model to choose the next evaluation. This is useful to investigate when candidate evaluations are expensive.

Watch from [[Gaussian Process|your video collection]] alongside [[Gaussian Processes by Mutual Information|your existing notes]]. The collection is unranked.

Use 🟢 [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/) in this order:

1. **Chapter 2:** connect familiar regression to Bayesian regression, then the function-space view. Understand how observations change the predictive mean and uncertainty.
2. **Chapter 4:** learn what the kernel assumes about relationships between inputs. Similarity between legal teams is a modelling choice; a kernel does not discover a useful representation automatically.
3. **Chapter 5:** consult it when fitting kernel hyperparameters becomes necessary.



## Candidate generation and legality

Start from [[Clause or the feature of the search space]] and [[hierarchical product sampling]]. Write down what a search move changes—species, moves, item or another field—and which dependent choices must be checked again.



**Practice:** test a rule with one valid candidate and one invalid candidate. For canonicalization, establish that two representations really are equivalent before merging them; an ordering that changes policy behaviour cannot simply be discarded as irrelevant.

## Comparing candidates

Before comparing methods, specify the opponent teams and their weights, the battle policies, the evaluation budget and how results are aggregated. Keep those choices consistent when the question is whether a candidate or search method improved performance.

If fitting a predictor, decide what must generalize: another battle involving familiar teams, unseen teams, or a changed opponent distribution. Choose the data split to match that question. A random split is not automatically evidence of performance on unseen teams.

For validation methods, consult **chapter 5** of 🟢 [An Introduction to Statistical Learning — free Python edition](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html) or [[Inteligent Systems/Richard_Golden_-_Statistical_Machine_Learning__A_Unified_Framework (2).pdf|Golden chapter 16]]. Use the relevant validation section directly.

## Learning a battle policy

This branch supports [[Reimplement VGC-Bench — run the release, reproduce BC, self-play, fictitious play, double oracle, then extend on team-pool size]]. A matchup predictor estimates an outcome; the battle policy chooses actions during play.

Use 🟢 [David Silver’s lectures](https://www.davidsilver.uk/teaching/), lectures 1–3 for the decision-process framework, then 4–7 for prediction, control, function approximation and policy gradients as needed. Pair them with [[Reinforcement Learning an Introduction SEcond Sedition Richard S. Sutton and andrew G. Barto.pdf|Sutton and Barto]]: chapters 3–4, 5–6 and 13 respectively. Silver’s Easy21 assignment and the book exercises provide practice. [[The Little Book of reinforcement Learning.pdf]] is an orientation reference.

**Questions to answer:** what can the policy observe, what signal does it learn from, and which opponents does it train and evaluate against?

## Optional references

- **Ridge and interaction features:** chapters 6 and 3 of 🟢 [An Introduction to Statistical Learning](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html), or [[Inteligent Systems/Richard_Golden_-_Statistical_Machine_Learning__A_Unified_Framework (2).pdf|Golden chapter 13]]. Use these when evaluating a ridge predictor.
- **Quadratic notation:** 🟢 [Glover, Kochenberger and Du](https://arxiv.org/abs/1811.11538). Use it to interpret $x^\top Qx$ when encountered. Check what the pairwise terms assume about team performance.
- **Matrix operations:** 🟢 [Boyd and Vandenberghe](https://web.stanford.edu/~boyd/vmls/), 🟢 [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), or Golden chapters 4–5. Look up the operation blocking the current derivation.
- **Further statistical or optimization theory:** 🟢 [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/download.html), 🟢 [Stanford Convex Optimization I](https://web.stanford.edu/class/ee364a/) and its 🟢 [exercise collection](https://github.com/cvxgrp/cvxbook_additional_exercises). Consult these for a particular derivation or proof.

**Study maps awaiting revision:** [[Online course pathway.canvas]] and [[Online study flow — free resources.excalidraw]].

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
