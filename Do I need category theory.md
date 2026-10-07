**Decision (2026-08-14): You do not need to learn category theory for this project.** Prerequisite for it would be abstract algebra (groups/rings), and that chain buys you nothing here. Verdict below is evidence-based, not a hunch.

## The argument: check every tool the project uses, ask what math it rests on

| Project component (from your notes) | Actual math it rests on | Category theory? |
|---|---|---|
| Weighted set cover / **maximum coverage** framing | Integer/linear programming, greedy approx, submodularity | No |
| **Constraint programming** — Species/Item Clause, all-different | Finite-domain CSP, propagation, backtracking search | No |
| **Symmetry breaking** (Gent–Petrie–Puget) | *Group* theory of permutations — operationally, not abstractly | No |
| **QUBO / Ising** quadratic framing | Linear algebra ($x^\top Q x$), binary quadratic forms | No |
| Search-space reduction / pruning | Algorithms, branch-and-bound, LP relaxation | No |
| RL thread (RLC, PokeAgent, metamon) | MDPs, dynamic programming, probability, gradients | No |

The one word that even smells like category-theory-adjacent is "symmetry breaking" — but that uses the *group* of board symmetries operationally (swap two identical rows). The Discrete Optimization course teaches it without a single categorical diagram. Group theory ≠ category theory, and you don't even need the former abstractly.

## Evidence from your own cited sources

- **Sanokowski et al., arXiv 2406.01661** (the paper you were excited to find): built on **variational inference, KL divergence, diffusion models, deep learning**. Zero category theory.
- **Glover QUBO tutorial, arXiv 1811.11538**: linear algebra and integer programming. Zero category theory.
- The whole combinatorial-optimization / RL literature this project lives in does not use it.

*Removed 2026-08-17 — the steelman here was a game-theoretic one. Dropped with the rest of that framing.*

## Bottom line

Category theory is **optional aesthetics, never a prerequisite** for this project. Every open question in your notes is reachable through: discrete optimization → integer/constraint programming → QUBO → RL. Learn those.

## Reading list (added 2026-08-24)

- **Bengio, Lodi & Prouvost — *Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon*** (arXiv 1811.06128) — <https://arxiv.org/abs/1811.06128>
  Why: canonical survey bridging the CO thread (set cover, branch-and-bound) and the RL thread. Read first.
- **Mirzasoleiman et al. — *Lazier Than Lazy Greedy*** (AAAI 2015, arXiv 1409.7938) — <https://arxiv.org/abs/1409.7938>
  Why: linear-time submodular maximization; relevant when the maximum-coverage team-selection framing outgrows toy size.

