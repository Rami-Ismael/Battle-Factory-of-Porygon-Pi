# Mathematical ideas beyond the matchup matrix

Yes—your matchup matrix can become a serious mathematical research object. My strongest recommendation is to study **spectral geometry, nontransitive games, and experimental design together**. They address three linked questions: What hidden structure exists? What counts as a strategically new team? Which battles would reveal that structure most efficiently?

This guide is tailored to your existing plan: a sparse database of team-versus-team results under fixed battle policies, initially anchored to tournament teams. You already have notes on ordinary optimization, active learning, and MAP-Elites, so the emphasis here is on different mathematical questions.

The mathematical foundations below are established. The proposed Pokémon experiments are hypotheses and adaptations, not proven improvements or claims of first-ever novelty. I checked primary research sources and worked through a small numerical example; I did not run Pokémon battles.

**The five ideas I would put at the front of your queue:**

| Idea | The question it answers | My assessment |
|---|---|---|
| Hodge decomposition | How much of the matchup structure cannot be explained by one strength score? | Best first diagnostic |
| Schur decomposition | Are there a few hidden planes of strategic advantage? | Closest to your spectral-analysis idea |
| Gamescape geometry | Is a new team strategically different from the teams we already have? | Strong bridge to counter search |
| Decision-aware experiments | Which battle could change the team we choose? | Most direct way to save simulation budget |
| Minimal counter-breaking edits | How many legal changes does it take to destroy a claimed counter? | Particularly interesting research experiment |

## First, define what the matrix means

Let P(i,j) be expected score for team i against team j, with a win worth 1, a loss 0, and a draw 1/2. Freeze the rules, battle-policy versions, preview procedure, and evaluation protocol. If draws are impossible, this is simply win probability.

For a common team pool on both axes, under an exchangeable protocol with any side effects balanced, P(i,j) + P(j,i) = 1. Define A = 2P − 1 entrywise; then A is skew-symmetric: Aᵀ = −A. This structure makes the spectral ideas especially useful. Independently noisy estimates need a statistically coherent fit; do not silently assume their measured entries satisfy the identity.

A rectangular matrix of generated candidates versus your 50 fixed opponents is a different object: SVD applies directly, but a skew-symmetric Schur analysis requires a common pool on both axes. An untested cell is unknown, not a 50% matchup.

## 1. Hodge decomposition: separate strength from cyclic advantage

**Imagine an electrical flow of advantage around the metagame.** Some of it points consistently from stronger teams to weaker teams. Some keeps circulating around loops.

Fit a strength score sᵢ for each team by minimizing

Σ over observed pairs (i,j): wᵢⱼ [Aᵢⱼ − (sᵢ − sⱼ)]².

The fitted differences describe the portion explained by an additive ranking. The residual records what that ranking misses. Hodge theory provides a principled decomposition of pairwise comparison flows into ranking and cyclic components. [Jiang et al., Statistical ranking and combinatorial Hodge theory](https://arxiv.org/abs/0811.1067).

**Your experiment:** Compare the residual energy with the residual energy expected from battle noise under a fitted ranking model. Map teams that participate in large, reproducible residual patterns. Those are candidates for strategically important specialists that an average score may conceal.

An illustrative pattern is A beating B, B beating C, and C beating A. But a large additive-ranking residual does not automatically prove a directed win cycle: nonlinear but transitive probabilities can also violate the additive model. Compare a Bradley–Terry/logistic strength model as well, and inspect actual cycle signs.

**Wacky extension:** Give each team two measurements: ordinary strength and participation in strategic circulation. Search for teams that exploit a particular circulation pattern rather than merely climbing a scalar ranking.

On a sparse comparison graph, separate connected components do not have identifiable relative strengths. The finer split between local cycles and harmonic flow also depends on the chosen graph and its filled triangles.

## 2. Schur decomposition: discover hidden strategic planes

**This is the closest mathematical match to your question.** A real skew-symmetric payoff matrix can be decomposed into two-dimensional blocks:

A = Σₖ λₖ(uₖvₖᵀ − vₖuₖᵀ).

Each team gets coordinates in each plane. The expression uᵢvⱼ − vᵢuⱼ describes directional advantage between two teams in that plane. This decomposition appears directly in research on game payoffs and limitations of Elo-style models. [Bertrand et al., On the Limitations of the Elo](https://proceedings.mlr.press/v206/bertrand23a.html).

**Intuition:** Instead of assigning a team one number, give it a position in several strategic diagrams. Two teams can have the same overall performance while occupying very different positions.

**Your experiment:** Fit a few planes and ask whether they predict held-out matchup outcomes better than a strength-only model. Then investigate whether the coordinates associate with measured behavior: reliance on speed control, defensive switching, or concentrated early damage. These interpretations must emerge from evidence; an eigenvector is not automatically a recognizable Pokémon concept.

**Wacky extension:** Search for a legal team whose predicted payoff vector points toward a poorly covered direction. The target is a strategic capability missing from the current population.

Do not rank teams by the largest eigenvalue of A. Its eigenvalues are imaginary or zero; SVD and real Schur blocks are the useful tools here. Also, rank-two structure alone does not prove rock–paper–scissors behavior: an additive strength matrix can be rank two too. Analyze the ranking residual as well as the full matrix.

## 3. A tiny example: all teams look equally strong, but the structure is real

Consider this invented expected-score matrix:

| Row against column | A | B | C |
|---|---:|---:|---:|
| A | 0.50 | 0.70 | 0.30 |
| B | 0.30 | 0.50 | 0.70 |
| C | 0.70 | 0.30 | 0.50 |

Every row averages 50%. A single ranking cannot express the advantages. Yet A beats B, B beats C, and C beats A.

For the centered payoff matrix 2P − 1, the eigenvalues are 0 and ±0.692820i. Its two nonzero singular values are both 0.692820. One two-dimensional block captures the entire pattern. The least-squares strength scores are equal, so all its nonzero payoff structure remains in the residual.

This illustrates why **low-dimensional does not mean rankable**, and why average win rate can discard useful matchup information. These numbers are a calculated toy example, not measured VGC results.

## 4. Gamescape geometry: discover teams with a new reason to exist

Represent each team by its payoff vector against a fixed opponent panel:

r(T) = [A(T,O₁), …, A(T,Oₘ)].

Now take the convex hull of the existing vectors. This is a gamescape: mixtures of existing strategies produce points inside that hull. The concept was developed for open-ended learning in symmetric zero-sum games. [Balduzzi et al., Open-ended Learning in Symmetric Zero-sum Games](https://proceedings.mlr.press/v97/balduzzi19a.html).

**Your experiment:** Reward candidates that improve a strategically relevant frontier, not just candidates with different species. A visually unusual team can have a redundant payoff profile; a one-move variation can create a useful new one.

There is a crucial distinction: lying outside the old convex hull is geometric novelty, but it might simply mean being worse in a new way. A stronger test is whether there exists an opponent distribution q ≥ 0, Σq = 1 for which the candidate outperforms every existing team:

max over q: [r(T)·q − max over existing U r(U)·q] > 0.

Require a meaningful margin and fresh validation. This asks whether the candidate is the best available answer to some plausible field.

Mixtures here describe random selection between complete strategies in the mathematical model. They do not imply that an event lets you change registered teams between rounds.

## 5. Strategic tomography: diagnose a team with a small panel of opponents

**Analogy:** A medical test panel uses a few probes to reveal different properties. Can a handful of opponent teams reveal most of a new team's matchup profile?

If payoff rows approximately lie in a low-dimensional space, some opponent columns should be more informative than others. Select probe opponents with complementary responses, then predict the remaining columns from their results.

**Your experiment:** Compare a small strategically selected panel with an equally sized random panel and a popularity-based panel. Evaluate full-row prediction and, more importantly, whether they select the same strong candidates under an independent large evaluation.

**Wacky extension:** Design a legal opponent specifically to distinguish two competing hypotheses about an unknown team. A diagnostic opponent need not itself have high average win rate.

This is my proposed application of low-rank modeling and experimental design. It is not a promise that a tiny panel is sufficient: isolated hard counters can be exactly the rare features that compression misses. Keep random audit opponents to detect that failure.

## 6. Decision-aware sampling: simulate the battle that could change your mind

The most uncertain cell is not necessarily the most useful cell. A nearly irrelevant matchup can be highly uncertain while a moderately uncertain matchup decides which candidate wins your search.

Research on estimating AlphaRank already selects payoff observations by their information gain about the desired game-theoretic result. [Rashid et al., Estimating α-Rank by Maximizing Information Gain](https://arxiv.org/abs/2101.09178).

**Your extension:** Target uncertainty about the recommendation itself: the best counter, whether a new team expands the useful frontier, or whether a proposed cycle is real. Approximate the expected improvement in that decision per unit of simulation cost.

**Experiment:** Give uniform allocation, uncertainty-only allocation, and decision-aware allocation the same battle budget. Compare their final recommendations using fresh battles. Also track how often they confidently recommend the wrong candidate.

This connects naturally to your existing active-learning notes, but changes the target from learning every cell accurately to resolving a particular strategic decision. Budget for the acquisition calculation too; it can become more expensive than the battles it saves.

## 7. Optimal transport: ask how much the metagame must change to break a team

Let q₀ be today's assumed opponent distribution. Instead of trusting it exactly, allow an adversary to move probability mass to other opponents within a transport budget ε:

robust_score(T) = min over q with W(q,q₀) ≤ ε: Σⱼ qⱼP(T,Oⱼ).

Here W measures the cost of changing the metagame. Wasserstein distributionally robust optimization is an established framework for such uncertainty sets. [Mohajerin Esfahani and Kuhn, Data-driven Distributionally Robust Optimization Using the Wasserstein Metric](https://arxiv.org/abs/1505.05116).

**Your experiment:** Plot each candidate's worst-case score as ε grows. Some teams may win narrowly under today's weights and deteriorate quickly; others may remain useful across many plausible fields.

**Wacky extension:** Define a team's “meta fragility” as the smallest transport budget that pushes its expected score below a chosen threshold. That gives “this counter is fragile” a measurable meaning.

The transport ground cost is a modeling choice. Species overlap and similarity of payoff profiles express different notions of nearby opponents. Compare them. Redistributing mass among the current 50 teams does not cover unseen teams unless those are added to the uncertainty model.

## 8. Epistasis and Boolean Fourier analysis: find changes that only work together

Choose a small set of legal binary edits to a generated team. Measure performance for each combination, then decompose the resulting landscape into single-edit effects and higher-order interactions. Walsh–Hadamard analysis is used for this kind of interaction analysis in fitness landscapes. [Poelwijk et al., Learning the pattern of epistasis linking genotype and phenotype in a protein](https://www.biorxiv.org/content/10.1101/213835v1.full).

**Example:** Edit A loses performance alone; edit B also loses performance alone; together they improve performance. A search that accepts only individually improving edits can miss that route.

**Your experiment:** Use four independently compatible edits, giving 16 variants. Evaluate against several distinct opponents. Ask whether the interaction remains useful across opponents or is specific to one matchup. This is a local experiment; it does not describe the whole legal team space.

**Wacky extension:** Compare interaction order across opponents. Perhaps one opponent can be beaten with independent adjustments, while another requires a coordinated three-edit change. That suggests different mutation operators for different targets.

Legality matters: an incomplete set of allowed combinations is not a full Boolean cube. Do not apply the standard transform as though missing illegal variants had measured scores. Keep the original tournament benchmark intact; use generated variants in a separate experiment.

## 9. Adversarial edit distance: how hard is it to break a counter?

Suppose T beats opponent O. Search for the smallest legal change to O that makes T stop winning:

minimize d(O,O′), subject to P(T,O′) ≤ τ.

Use τ = 0.5 for losing the positive edge, or a stricter margin. Define d explicitly: species changes, move changes, and EV changes need not have equal cost.

**Your experiment:** Compare two generated counters with similar original win rates. One may fail after one opponent move change; the other may resist every nearby variant searched.

**Wacky extension:** Produce a “counter robustness certificate” limited to an explicitly enumerated neighborhood. If that neighborhood is too large to exhaust, report only the smallest breaking edit found, which is an upper bound on the true minimum—not a proof that no smaller edit exists.

This is a proposed constrained-search problem rather than a claimed published Pokémon method. It is especially appealing because the result can be explained to a player in concrete terms. Uncertainty-aware confirmation is essential near τ.

## 10. Preview phase transitions: tiny changes can switch the optimal plan

Think of a team matchup as containing a smaller strategic game between preview plans. For a fixed outer team pair, let B(a,b) be expected payoff when each side commits to a specified preview plan and then follows its battle policy.

Study the zero-sum value maxₓ minᵧ xᵀBy and which plans receive positive probability. As team details change, the supported plans may switch abruptly, even when estimated payoffs change modestly.

**Your experiment:** Vary one legal team parameter across a small range and track both overall value and selected plans. A sudden behavioral change can reveal why a local team edit mattered.

**Wacky extension:** Search specifically near these support changes, where a small edit may unlock a different strategic mode.

Your notes mention a 15×15 preview game. Fifteen counts four-member subsets of six; it does not by itself encode lead choices or every aspect of preview. Define the action abstraction carefully, or let a fixed inner policy handle the omitted decisions. Sensitivity and parametric game analysis are the mathematical directions to study here; this Pokémon experiment is a proposal.

## 11. Persistent directed topology: do the counter loops survive scrutiny?

Make a directed edge i → j when i beats j by a threshold margin. Vary that threshold and examine structural features that persist. Persistent path homology is a developed tool for directed networks; ordinary undirected topology can lose the asymmetry central to matchups. [Chowdhury and Mémoli, Persistent Path Homology of Directed Networks](https://arxiv.org/abs/1701.00565).

**Your experiment:** Start with strongly connected components and explicit three-team cycles. Bootstrap battle outcomes and check which structures recur. Only then test whether directed topological summaries reveal useful structure beyond these simpler statistics.

**Wacky extension:** Seek a new team that connects previously separate strategic regions or changes a persistent feature. Test whether that topological novelty predicts useful frontier expansion.

This is the highest-risk idea in the list. A homology class is not automatically a literal rock–paper–scissors loop, and persistence across thresholds is not the same as statistical confidence. Missing edges also mean unmeasured comparisons, not demonstrated absence of advantage.

## 12. Separate team strength from the ability of your bot to use it

Your matrix is conditional on battle policies. A team may look weak because the evaluator cannot execute its plan. Study an array with axes for own team, opposing team, own policy, and opposing policy, rather than just one matrix.

**Your experiment:** Evaluate a smaller team panel with several meaningfully different policy versions. Decompose performance into broad team effects and team–policy interactions. Ask whether the spectral structure and strongest counters survive the evaluator change.

**Wacky extension:** Search for teams with a large difference between mediocre-policy performance and stronger-policy performance. That is a hypothesis about unrealized capability, not evidence of human potential by itself.

This is a proposed use of crossed experimental designs and latent-factor models. More policy snapshots are not necessarily independent evidence if all share the same blind spots. The full array is expensive; start with a deliberately small crossed design.

## The combined research project I would choose

**Question:** Can strategically structured matchup models discover useful counter teams with fewer simulated battles than mean-score search?

1. Build a small, balanced common-pool matchup panel with battle counts and policy identifiers.
2. Compare a scalar strength model, an unrestricted low-rank model, and a strength-plus-skew-residual model.
3. Use their predictions to propose legal teams that improve the useful payoff frontier against plausible opponent distributions.
4. Allocate new battles to resolve uncertainty about those improvements, with some random audit sampling.
5. Confirm selected teams using fresh battles, new opponent teams, and an additional evaluator policy where available.

The interesting contribution would be evidence that this combination finds useful strategic capabilities at lower cost. “We ran an eigendecomposition” alone would be a much weaker result. Nor can we call the combination novel without a broader related-work review.

For a first descriptive panel, 30 teams produce 435 unordered pairs. At 100 battles per pair, that is 43,500 battles under the complementary-score protocol. A 50% matchup still has an approximate individual 95% interval half-width of 9.8 percentage points with 100 independent no-draw battles. This is enough to see some large effects, not reliably distinguish tiny edges across hundreds of comparisons. Use fresh confirmation and a noise-aware analysis rather than treating every estimated edge as real.

Hold out entire pairs when evaluating reconstruction, and entire teams when evaluating generalization. Ordinary matrix factorization cannot embed a never-observed team without probe battles or a learned mapping from team features. Benchmark both costs. A small retained rank is useful only if it preserves the decisions and rare counters you care about.

## What to study, in an order that serves this project

| Order | Mathematical field | Learn first | Concrete project payoff |
|---|---|---|---|
| 1 | Linear algebra | SVD, skew-symmetric matrices, real Schur form | Understand matchup dimensions |
| 2 | Statistical inference | Binomial uncertainty, hierarchical estimation, bootstrap | Distinguish structure from battle noise |
| 3 | Graph theory and discrete Hodge theory | Incidence matrices, Laplacians, weighted least squares | Separate ranking from residual structure |
| 4 | Convex geometry and game theory | Convex hulls, minimax, linear programming | Define strategic novelty and useful counters |
| 5 | Bayesian experimental design | Value of information and decision uncertainty | Choose informative battles |
| 6 | Robust optimization and optimal transport | Distribution uncertainty and transport costs | Test resistance to metagame changes |
| 7 | Discrete Fourier analysis | Boolean functions and interaction coefficients | Discover coordinated edits |
| 8 | Topological data analysis | Directed filtrations and persistence | Investigate higher-risk structural questions |

Start with the three-team example, then Hodge residuals and Schur planes on a modest measured panel. If those structures survive uncertainty and predict held-out matchups, gamescape-guided counter search is the next experiment I would fund.
