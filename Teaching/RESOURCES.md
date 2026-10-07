---
created_at: 2026-08-18
updated_at: 2026-09-07
type: teaching-resources
---

# Combinatorial-search modeling — Resources

## Knowledge

- 🟢 [Feige — A Threshold of ln n for Approximating Set Cover (JACM 1998, Duke course copy)](https://courses.cs.duke.edu/spring07/cps296.2/papers/p634-feige.pdf)
  Abstract + §1 define set cover and max k-cover side by side; §5 Prop 5.1 is the half-page greedy 1−1/e proof. Use for: definitions, both hardness thresholds. Skip the PCP body.
- 🟢 [Williamson & Shmoys — The Design of Approximation Algorithms (free electronic edition)](https://www.designofapproxalgs.com/download.php)
  §1.2 set cover (weighted, LP), §1.6 greedy H_n, §2.5 float model = soft coverage with 1−1/e, Ex 2.11 max coverage. Use for: any set cover / coverage / greedy question first.
- 🟢 [Wikipedia — Maximum coverage problem](https://en.wikipedia.org/wiki/Maximum_coverage_problem)
  ILP, weighted and budgeted variants, pointers to Feige and Khuller–Moss–Naor. Use for: quick formulation check, never as a citation of record.
- 🔒 Khuller, Moss & Naor — The Budgeted Maximum Coverage Problem, IPL 70 (1999) — [abstract on Semantic Scholar](https://www.semanticscholar.org/paper/465a755ef79ff8d74f7a355012269fa5fcbba477)
  Set costs + element weights + spend budget; modified greedy 1−1/e. Use for: if species ever carry a cost.
- 🔒 Nemhauser, Wolsey & Fisher 1978 (Math. Programming) — the general submodular 1−1/e. Read through W&S §2.5 notes instead.
- 🟢 [Blitzstein & Hwang — Introduction to Probability (free official copy)](https://probabilitybook.net)
  Ch. 7 *Joint distributions*, §7.3 *Covariance and correlation* — the marginal/joint/covariance vocabulary under CRN. Use for: probability prerequisites whenever a lesson leans on variance identities. Taught: lesson 0007 (coupling).

## Knowledge — LLM hyper-heuristics (pruning-rule strand, 2026-08-19)

- 🟢 [Ye et al. — ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution (NeurIPS 2024)](https://arxiv.org/abs/2402.01145) · 🟢 [repo](https://github.com/ai4co/reevo) · 🟢 [project page](https://ai4co.github.io/reevo/) — generator LLM writes heuristic code, reflector LLM writes "verbal gradients", meta-objective = average performance on a cheap instance dataset. Use for: the evolved-pruning-rules idea (Lesson 0005); §3 + prompt appendix first, then the repo README. Vault note: `Papers/ReEvo and TIDE — the LLM writes heuristic code, evolutionary search scores it.md`.
- 🟢 [Chen et al. — TIDE: Tuning-Integrated Dynamic Evolution (arXiv 2601.21239, preprint, no venue)](https://www.semanticscholar.org/paper/c711eea4075d940f24d32050492689a6ba2f6da1) — ReEvo successor: DE inner loop tunes numeric constants inside generated code. *Dropped by the owner 2026-08-19; kept as the citation for the structure-vs-constants decoupling only.*

## Knowledge — analogue-domain papers graded for $f$ (sweep 2026-08-19)

Answers the 2026-08-18 todo (weighted max coverage or a learned pairwise objective over a k-subset, in VGC-shaped domains). Rubric: **HIGH** = learns an outcome model over the members **and** optimizes the k-subset against it or an opponent distribution; **MED** = one half; **LOW** = adjacent flavor. Verified negative result: no published paper does $f$'s exact conjunction — weighted coverage of an opponent-team distribution over a k-subset — and none fits a pairwise objective from battle results and then searches the subset; that fitted-quadratic loop lives in materials design (BOCS, FMQA below).

### HIGH

- 🟢 [Haugh & Singal — How to Play Fantasy Sports Strategically (and Win) (Management Science 2021, author copy)](http://www.columbia.edu/~mh2078/DFS_Revision_1_May2019.pdf) · INFORMS version 🔒 — fits a Dirichlet-multinomial distribution over opponents' lineups from contest data, then maximizes expected payoff against it via binary quadratic programs over the roster. Closest structural match to $f$ found: meta-as-distribution + quadratic subset objective. Read first.
- 🟢 [Chen et al. — The Art of Drafting (RecSys 2018)](https://arxiv.org/abs/1806.10130) — MCTS over the remaining picks, leaf values from a learned lineup win-rate model. Caveat quoted from the 2026-08-18 sweep: MOBA pick/ban is sequential-alternating, unlike VGC's simultaneous blind 6-pick — HIGH for the learn-then-search shape, not the draft structure.
- 🟢 [Chen, Zhu et al. — Which Heroes to Pick? / JueWuDraft (IEEE Transactions on Games 2021)](https://arxiv.org/abs/2012.10171) — policy/value networks + tree search over picks, valued across a best-of-N series. Same caveat as above.
- 🟢 [Chen et al. — Q-DeckRec (IEEE CIG 2018)](https://arxiv.org/abs/1806.09771) — learned deck win-rate predictor + learned search policy over the card subset, built against explicit opponent decks (single opponents, not a distribution). *Deep-read 2026-08-19: the predictor-argmax pipeline is the paper's own losing baseline (win rate degrades as sampling widens — the argmax selects predictor errors), and the learned policy only ties a small GA in quality; HIGH here rests on the learn-then-search shape, not on beating search.*
- 🟢 [Zhang et al. — Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding (GECCO 2022)](https://arxiv.org/abs/2112.03534) — online-trained outcome surrogate + MAP-Elites over decks, periodically re-grounded in simulation: the "estimator of $f$, then search" loop. *Deep-read 2026-08-19: HIGH confirmed — ~2.5× QD-score over plain MAP-Elites at the same 10,000-evaluation budget; note the surrogate predicts a health-difference proxy plus the QD measures, and the opponent suite is 6 decks weighted evenly.*
- 🟢 [Hunter, Vielma & Zaman — Picking Winners (venue not stated on arXiv)](https://arxiv.org/abs/1604.01455) — submodular, coverage-flavored P(at least one lineup wins) maximized by integer programming, with means and pairwise correlations fitted from data; not opponent-aware.

### From the parallel 2026-08-19 sweep (its grades, quoted from the todo pointer; links re-verified here)

- 🟢 [Skowron & Faliszewski — Chamberlin–Courant Rule with Approval Ballots: Approximating the MaxCover Problem with Bounded Frequencies in FPT Time (JAIR 60, 2017)](https://jair.org/index.php/jair/article/view/11095) — winner determination under approval-based Chamberlin–Courant **is** weighted MaxCover: a k-committee "covers" voters the way a team covers meta teams, so multiwinner voting is the mathematical home of formulation (a). The parallel sweep graded it HIGH ("$f$ ≡ weighted MaxCover"); under this section's rubric it is the coverage half only (nothing fitted, no opponent). *Deep-read 2026-08-19: lowered to LOW — its MaxCover is unweighted and binary, and the FPT algorithms are infeasible at VGC scale by the authors' own admission; keep only as the CC ≡ MaxCover citation.*
- 🟢 [Neoh, Teh, Chooi, Goldberg & Tambe — Efficient Ensemble Selection from Binary and Pairwise Feedback (venue not stated on arXiv)](https://arxiv.org/abs/2605.09588) — selects a k-ensemble as multiwinner voting over an unknown domain distribution: greedy with coverage guarantees, objective estimated from binary/pairwise feedback queries. The closest thing found to "estimate the coverage objective from results, then greedily cover"; graded HIGH by the parallel sweep. *Retired 2026-08-19 — the owner rejected this paper as not relevant; the HIGH grade is withdrawn.*

### MED (one rubric half each)

- 🟢 [GAE — Modeling Game Avatar Synergy and Opposition (unpublished; arXiv comments: draft rejected by AIIDE 2017)](https://arxiv.org/abs/1803.10402) — bilinear same-team + cross-team pairwise terms, the closest $Q_{ij}$ analogue; predicts and recommends greedily, never searches the subset.
- 🟢 [DraftRec (WWW 2022)](https://arxiv.org/abs/2204.12750) — two-transformer per-pick recommendation; interactions via attention, no explicit pairwise terms, no subset search.
- 🟢 [OptMatch (KDD 2020, author copy)](https://linxiagong.github.io/OptMatch/KDD2020_OptMatch.pdf) — mined pairwise synergy/suppression embeddings; optimizes matchmaking partitions for balance, not a team against a meta.
- 🟢 [Lappas, Liu & Terzi — Finding a Team of Experts in Social Networks (KDD 2009, author copy)](https://cs-people.bu.edu/evimaria/papers/fp525-lappas.pdf) — skill coverage + pairwise communication cost over a subset; costs are given graph weights, nothing learned, no opponent.
- 🟢 [Vombatkere, Terzi & Lappas — A QUBO Framework for Team Formation (ECML PKDD 2025)](https://arxiv.org/abs/2503.23209) — coverage-vs-cost as a quadratic objective over the member-selection vector; $Q$ is designed, not regressed. Directly relevant to QUBO-as-model-class.
- 🟢 [Maymin, Maymin & Shen — NBA Chemistry (IJCSS 2013, author copy)](https://philipmaymin.com/papers/Maymin%20Maymin%20and%20Shen%20-%20NBA%20Chemistry%20-%20IJCSS.pdf) — pairwise skill-by-skill synergies fitted from play-by-play (a lineup ≠ sum of parts); valuation only, no roster search.
- 🔒 [Reis et al. — An Adversarial Approach for Automated Pokémon Team Building and Metagame Balance (IEEE Transactions on Games 2024)](https://doi.org/10.1109/tg.2023.3273157) — no free copy found (no arXiv, U.Porto repo 403); already in [[VGC AI Competition (IEEE CoG)]]. Team search against a co-evolving builder; simulated win rate, neither coverage nor a learned pairwise model.
- 🟢 [Guimarães et al. — Pokémon GO Team Optimization (Journal on Interactive Systems 17(1), 2026)](https://journals-sol.sbc.org.br/index.php/jis/article/view/6773) — 3-Pokémon subset vs a fixed pool of 1,000 rival teams via metaheuristics (VNS wins); the subset-vs-opponent-pool half, with raw simulated fitness as the objective.
- 🟢 [Zeng et al. — Team Composition in PES2018 Using Submodular Function Optimization (IEEE Access 2019)](https://doi.org/10.1109/ACCESS.2019.2919447) — players cover a fixed skill universe, proven submodular, greedy over an 11-subset; the coverage half without an opponent distribution.
- 🟢 [Streeter & Golovin — An Online Algorithm for Maximizing Submodular Functions (NIPS 2008)](https://papers.nips.cc/paper/3569-an-online-algorithm-for-maximizing-submodular-functions) — solver portfolio covering a SAT-instance distribution: instances play the role the meta plays in $f$.
- 🟢 [Baptista & Poloczek — Bayesian Optimization of Combinatorial Structures (ICML 2018)](http://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf) — sparse Bayesian regression of a pairwise surrogate $\alpha_{ij}$, acquisition solved as a binary quadratic program: the ridge-with-$Q_{ij}$ spirit, no opponents, no hard $k$. *Demoted in the reading list 2026-08-25 (#5 → beside the Frazier tutorial): assumes Gaussian-process surrogate + acquisition vocabulary the owner has not learned yet; read only after Frazier.*
- 🟢 [Kitai et al. — metamaterials design with quantum annealing and factorization machines (Physical Review Research 2020)](https://arxiv.org/abs/1902.06573) — a factorization machine learns a QUBO from simulation results, an annealer optimizes it, repeat; the canonical fitted-pairwise loop.
- 🔒 [García-Sánchez et al. — Automated playtesting in collectible card games (Knowledge-Based Systems 2018)](https://doi.org/10.1016/j.knosys.2018.04.030) — no OA copy exists (Unpaywall checked). GA over the deck, fitness = simulation vs a fixed, unweighted suite of human-made decks.
- 🔒 [Bhatt et al. — Exploring the Hearthstone Deck Space (FDG 2018)](https://dl.acm.org/doi/10.1145/3235765.3235791) — no OA copy found. Evolution Strategy decks vs fixed opponents; also measures deck-space non-transitivity, which bears on whether a single best-response team to a meta exists.
- 🟢 [de Mesentier Silva et al. — Evolving the Hearthstone Meta (IEEE CoG 2019)](https://arxiv.org/abs/1907.01623) — works directly on the deck-vs-deck matchup matrix, but the decision variable is card balance, not a subset.
- 🟢 [Ramos & Soria — From Rules to Nash Equilibria (venue not stated on arXiv, 2026)](https://arxiv.org/abs/2607.08692) — machine-checked Nash + replicator dynamics over a Pokémon TCG 14×14 archetype matchup matrix from tournament data: the meta-as-distribution machinery, but selects among existing archetypes rather than constructing a team.

### LOW

- 🟢 [Summerville et al. — Draft-Analysis of the Ancients (AIIDE 2016)](https://ojs.aaai.org/index.php/AIIDE/article/view/12899) — predicts which hero a human picks next; imitation, no objective over the subset.
- 🟢 [Hanke & Chaimowicz — A Recommender System for Hero Line-Ups (AIIDE 2017)](https://ojs.aaai.org/index.php/AIIDE/article/view/12938) — association rules mined from winning compositions; co-occurrence, not a fitted objective.
- 🟢 [Kowalski & Miernik — Evolutionary Arena Deckbuilding with Active Genes (IEEE CEC 2020)](https://arxiv.org/abs/2001.01326) — sequential pick-1-of-3 arena drafting, structurally unlike a free k-subset choice.
- 🟢 [VGC-Bench (AAMAS 2026)](https://arxiv.org/abs/2506.10326) — battling benchmark with explicit team diversity; no team-selection objective. Already in the vault.

One qualifying result excluded per `Thing I don't want my claude be talking about.md`: the adversarial synergy-graph lineup paper (ALA@AAMAS 2015) — Synergy Graphs are on the do-not-discuss list.

## Wisdom (Communities)

- [Operations Research Stack Exchange](https://or.stackexchange.com/) — formulation questions ("is this max coverage or set cover?", ILP modelling). High signal, patient with modelling beginners.
- [Computer Science Stack Exchange](https://cs.stackexchange.com/) — approximation-algorithm and greedy-guarantee questions.
- VGC domain: Smogon VGC forums / r/VGC — for the Pokémon side only; "coverage" there means type coverage.

## Knowledge — self-play / battle-policy strand

- 🟢 [Zhang, Xu et al. — A Survey on Self-Play Methods in Reinforcement Learning (arXiv 2408.01072 v4, 2025-10-18)](https://arxiv.org/html/2408.01072v4)
  Sec. 3.1 framework (Π, Σ/σ, ORACLE, EVAL/𝒫, MSS, h(k), E); Sec. 3.1 oracle types BR / ABR / specially crafted; Sec. 3.3.2 Double Oracle walk-through on RPS; Table 1. No venue on arXiv. Use for: classifying any self-play arm.
- 🟢 [McMahan, Gordon & Blum — Planning in the Presence of Cost Functions Controlled by an Adversary (ICML 2003, AAAI PDF)](https://www.aaai.org/Papers/ICML/2003/ICML03-071.pdf)
  Sec. 4.2 Double Oracle definition, stopping rule, Theorem 1 (needs exact BR + finite sets). Silent on approximate oracles.
- 🟢 [Lanctot et al. — A Unified Game-Theoretic Approach to Multiagent RL (PSRO, arXiv 1711.00832 v2)](https://arxiv.org/abs/1711.00832)
  Sec. 2–3: "DO is an instance of PSRO with n = 2 and Nash meta-strategy"; RL = approximate best response; convergence stated for DO only.
- 🟢 [VGC-Bench repository](https://github.com/cameronangliss/VGC-Bench) — `train.py` / `src/callback.py`: PPO in every arm; FP = uniform over checkpoints; DO = nashpy over win-rate matrix.

- 🟢 [Grigsby, Xie, Sasek, Zheng & Zhu — Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers (Metamon, RLC 2025, arXiv 2504.04395 v2)](https://arxiv.org/abs/2504.04395)
  Sec. 3 replay reconstruction; Sec. 4 Eq. 1–2 + Table 1 (IL / Exp / Binary / Binary+MaxQ); Sec. 5.3 synthetic self-play and the V1+SP failure; Fig. 12/14 ladder; App. D.1 missing actions; App. E.1 reward; Table 3 sizes. Singles, Gen 1–4. Use for: BC-vs-offline-RL evidence, self-play data-shift caution.
- 🟢 [UT-Austin-RPL/metamon repository](https://github.com/UT-Austin-RPL/metamon) · 🟢 [jakegrigsby/metamon on Hugging Face](https://huggingface.co/jakegrigsby/metamon) — code, checkpoints, 3.5M-trajectory replay corpus (now incl. Gen 9 OU). Use for: the singles→doubles pretraining experiment.
- 🟢 [Grigsby, Fan & Zhu — AMAGO: Scalable In-Context RL for Adaptive Agents (ICLR 2024, arXiv 2310.09971)](https://arxiv.org/abs/2310.09971) — the training engine under Metamon. App. A Eq. 2–3: the critic TD loss and joint loss; REDQ ensemble, target heads, multi-γ, PopArt (Fig. 11). Use for: the critic equation Metamon omits.
- 🟢 [Grigsby et al. — AMAGO-2: Breaking the Multi-Task Barrier in Meta-RL with Transformers (NeurIPS 2024, arXiv 2411.11188)](https://arxiv.org/abs/2411.11188) — Sec. 3 Eq. 1–3: critic/actor losses with h_t, two-hot value classification; App. "Value Classification Details". Use for: SynRL-V2's critic.
- 🟢 [Nair et al. — AWAC (arXiv 2006.09359)](https://arxiv.org/abs/2006.09359) · 🟢 [Wang et al. — Critic Regularized Regression (arXiv 2006.15134)](https://arxiv.org/abs/2006.15134) — the Exp and Binary weight functions. Use for: what "filtered BC" means.

## Wisdom (Communities) — self-play strand

- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/) — PSRO / self-play implementation questions; moderate signal.
- [poke-env Discord / GitHub discussions](https://github.com/hsahovic/poke-env) — Showdown + RL practitioners; the VGC-Bench author is active in that ecosystem.

## Knowledge — variance reduction for simulated-battle comparisons (2026-08-20)

- 🟢 [Yadav, Maliakkal, Khadilkar & Kalyanakrishnan — Using Common Random Numbers for Simulation-based Planning with Rollouts (RLJ 2026, Paper 52)](https://rlj.cs.umass.edu/2026/papers/Paper52.pdf) · 🟢 [arXiv 2605.04732](https://arxiv.org/abs/2605.04732)
  §2 the three estimators X_I / X_D / X_DD and Theorem 2 (coupling beyond the depth where the policies agree is provably ≤ var(X_I)); §1.2 Proposition 1 (full coupling can backfire when covariance turns negative); §4 Ludo/UCT deployment, 5000 games. Use for: seed-pairing in any battle-based comparison of two teams (X_D, no theorem) and in rollout-based battle-policy planning (X_DD, exact fit). Taught: lesson 0006 · ELI5: `eli5/common-random-numbers-share-the-dice-where-you-agree.html`.

## Gaps

- No free, gentle treatment of partial set cover found yet.
- Community preferences not yet stated by the owner.
- $f$'s exact conjunction (weighted coverage of a meta over a k-subset) has no published instance — verified 2026-08-19; nearest halves are Haugh & Singal 2021, PES2018 (IEEE Access 2019), and Pokémon GO (JIS 2026).

## Knowledge — submodular maximization

- 🟢 [Krause & Golovin — Submodular Function Maximization, author-hosted chapter](https://www.cs.cmu.edu/~dgolovin/papers/submodular_survey12.pdf)
  Definitions 1.1–1.3 and Theorem 1.5: marginal gains, diminishing returns, and the cardinality-constrained greedy guarantee. Use for lesson 0010 and its reference sheet.
