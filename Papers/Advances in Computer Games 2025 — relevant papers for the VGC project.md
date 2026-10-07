# Advances in Computer Games 2025 — relevant papers for the VGC project

Research date: 2026-09-06. Scope: the conference linked at [ICGA](https://icga.org/?page_id=4052), screened against this project's legal-team optimization, doubles battle policy, team preview, opponent coverage, and limited local compute. Priorities below are recommendations for this project, not assessments of paper quality.

## Recommendation

This is **Advances in Computer Games 2025 (ACG 2025)**, the nineteenth ACG conference, held online October 21–23, 2025. It is a research conference rather than a Pokémon competition. Its revised proceedings appeared in Springer LNCS 16463 in May 2026. [Conference page](https://icga.org/?page_id=4052); [publisher proceedings](https://link.springer.com/book/10.1007/978-3-032-23657-9).

The strongest matches are **Evolutionary Tree Search for Turn-Based Strategy Games**, **Adapting MCTS Algorithms for Stochastic Games**, and **Static and Intrinsic Diversity in Go Agents**. They address components surrounding the project rather than the full team-building problem. **None warrants replacing the existing VGC-Bench reproduction with a new architecture solely on this evidence.** Treat them as conditional medium-priority reading after the baseline works.

For hidden information, keep Grassauer's information-set computation paper and the Geister inference paper as targeted references. For evaluation, SpinGPT is a useful cautionary case about imitation scores, benchmark mismatch, and the distinction between a base model and a patched evaluated agent.

## What the event actually contains

The official event lists 18 presentations, while the proceedings contain 17 papers. The Civilization VI volunteer-engagement presentation appears in the event list but not the proceedings table of contents. The reviewed program does not list a Pokémon paper. ICGA also links recordings of all sessions and three keynotes; Mark Winands's **New Adventures with Monte-Carlo Tree Search** is the most directly relevant keynote by topic, but its contents were not reviewed here. [Official event and recordings](https://icga.org/?page_id=4052); [proceedings](https://link.springer.com/book/10.1007/978-3-032-23657-9).

The screening table covers every presentation. “Medium” means read when the named component becomes a bottleneck; it does not create a new HIGH item in the existing reading list. Methods and access were checked from publisher abstracts or author material, except the explicitly marked presentation-only entry.

| Paper / presentation | Priority for this project | Evidence and reason |
|---|---|---|
| [Evolutionary Tree Search for Turn-Based Strategy Games](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_12) | Medium: action search | Evolves complete action sequences inside a tree; closest structural connection to coordinated action choices. |
| [Adapting MCTS Algorithms for Stochastic Games — An Example on EinStein Würfelt Nicht!](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_11) | Medium: stochastic search | Stochastic but perfect-information test game; chance handling is relevant, hidden simultaneous actions remain separate. |
| [Static and Intrinsic Diversity in Go Agents](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_10) | Medium later: opponent population | Demonstrates ways to create different playing styles; diversity must still be measured in VGC payoffs. |
| [Leveraging Answer Set Programming for Information Set Computation in Imperfect Information Games](https://github.com/2503-phd/developments-benchmarks) | Medium if belief consistency is difficult | Author appendix implements logical enumeration of histories consistent with observations. |
| [Rule-Based Red Piece Inference and Board Evaluation in the Imperfect Information Game Geister](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_15) | Medium: inexpensive inference baseline | Behavior-based hidden-piece inference combined with risk-aware evaluation and minimax. |
| [SpinGPT: A Large-Language-Model Approach to Playing Poker Correctly](https://arxiv.org/html/2509.22387v1) | Medium for evaluation; low for architecture | Human imitation followed by solver supervision; substantial caveats in actual matchup evidence. |
| [Mastering Othello with Genetic Algorithm and Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_9) | Low now | Evolves n-tuple feature structures, not teams or team members. |
| [Making KataGo HumanSL More Human-Like for Amateur-Level Play](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_13) | Low; possible human-opponent modeling later | Previous-move policy adjustment and softmax improve perceived human-likeness; different objective from finding strong counters. |
| [Measuring Strategic Similarity Between Human and Engine-Generated Chinese Chess Opening Trees](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_14) | Low; coverage analysis later | Sequence, structure, and coverage metrics compare opening distributions; conceptual inspiration for preview/lead coverage. |
| [Relevance-Zone Reduction in Game Solving](https://arxiv.org/abs/2510.00689) | Low | Reuses localized proof patterns in Killall-Go; no established equivalent of independent spatial zones in VGC. |
| [SEGClobber — A Linear Clobber Solver](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_2) | Low | Specialized one-dimensional combinatorial-game solver. |
| [MCGS: A Minimax-Based Combinatorial Game Solver](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_3) | Low | Exploits sums of independent subgames; team synergy makes that assumption unsuitable without proof. |
| [Mechanism Design for Player-Judged Games](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_5) | Low | Uses CFR to study fairness incentives for strategic judges; does not contribute a VGC battle solver. |
| [Enhancing Cost-Effective Large Language Models with Reinforcement Learning for Multi-turn Text-Based Environments](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_6) | Low | Learns a controller selecting prompts for GPT-3.5 in TextWorld/WebShop; does not address the team's simulator-search problem. |
| [Real Time Expert Chess Commentary Using LLM](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_7) | Low | Engine-informed prompting for commentary quality, not action strength. |
| [An Application of Secure Computation to Tagiron](https://www.tains.tohoku.ac.jp/netlab/mizuki/conf/tagiron_acg2025_web.pdf) | Low | Physical-card cryptography verifies guesses without revealing secrets; not hidden-state inference for a battle policy. |
| [Player Preferences For Play of the Game: A Study of Player Engagement on Video Game Highlights](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_4) | Low | Overwatch highlight preferences and engagement. |
| [From Passion to Practice: Volunteer Engagement and Organizational Challenges in Civilization VI Esports Events](https://icga.org/?page_id=4052) | Low, provisional | Official presentation listing only; no method or empirical result inferred from its title. |

## Selected papers: what transfers and what does not

### 1. Evolutionary Tree Search — coordinated action candidates

**Verified:** Neller, Huang, and Cho represent turn-start states as nodes and complete multi-action sequences as edges. Mutation and crossover generate candidate children. The publisher abstract reports improvements over M-UCT, RHEA, and FH-EMCTS in TUBSTAP. [Publisher abstract](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_12).

**Project transfer idea:** Test mutation of legal joint action pairs for the two active Pokémon, including targets, switches, and special mechanics. Compare against enumerating legal pairs or sampling from the existing policy under an equal simulator-call budget. Consider a preview variant only as a separate experiment.

**Limit:** Multi-action turns are not the same as two opponents choosing secretly and simultaneously. Do not let the simulated opponent respond after observing the player's hidden choice. This paper also does not optimize a six-Pokémon roster. Review depth: abstract only; the related [author thesis record](https://vtechworks.lib.vt.edu/items/50f43283-31e2-4dc1-a002-c1c8b7db6e83) is public, but its PDF routes to a request-a-copy form.

### 2. Adapting MCTS — randomness inside battle evaluation

**Verified:** Hsu and colleagues study perfect-information stochastic EinStein Würfelt Nicht!, introduce virtual nodes grouping moves adjacent to captured pieces, and modify the upper confidence bound. The abstract reports a convergence-rate result. [Publisher abstract](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_11).

**Project transfer idea:** Use it to investigate how tree search allocates simulation effort across chance outcomes. A useful VGC experiment measures action quality and team-ranking stability under a fixed simulation budget, not just the number of expanded nodes.

**Limit:** The virtual-node construction is game-specific. The abstract is insufficient to establish that its theorem, update rule, or grouping preserves correctness in VGC. Random outcomes and an unknown opponent's choice need separate treatment. Read the full method before implementing it.

### 3. Static and Intrinsic Diversity — avoid a narrow opponent population

**Verified:** Murakami and Kaneko compare user-specified territory preferences with diversity-driven pseudo-rewards in Gumbel AlphaZero, evaluated in 9×9 Go. Their abstract reports diverse styles with playing strength depending on parameter choices. [Publisher abstract](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_10).

**Project transfer idea:** A team optimizer should face opponents that differ in consequential decisions, not merely random seeds. Later, compare a population of behaviorally different battle policies against an equally sized collection of ordinary checkpoints. Evaluate payoff coverage and held-out opponents.

**Limit:** Go territory style is not Pokémon matchup diversity. Different behavior does not by itself imply resistance to exploitation. Keep this as later reading: the project's reward-related reading has already been deferred until reward integration. No reward change is proposed now.

### 4. Answer Set Programming — possible states versus likely states

**Verified:** Grassauer's author appendix uses Clingo to enumerate game histories compatible with partial observations and compares against depth-first baselines. Supplied games include Battleship and Phantom Connect. It records time to first history, enumeration time, and whether enumeration completed. [Author appendix](https://github.com/2503-phd/developments-benchmarks).

**Project transfer idea:** Build a small consistency checker for candidate hidden sets after revealed moves, damage, or speed ordering. Respect the project's chosen information protocol: only model attributes that are actually hidden. Try a bounded hypothesis pool before translating the full simulator into a logic language.

**Limit:** Consistency is not probability. Enumerating possible histories does not establish a calibrated posterior over opponent sets or their actions. Full VGC histories may be far too large. Review depth: author README/experimental framework; the [published chapter](https://doi.org/10.1007/978-3-032-23657-9_17) could not be fetched here, so no quantitative speedup or theorem is claimed.

### 5. Geister — interpretable opponent inference

**Verified:** Narita, Kubota, and Suzuki infer hidden red pieces from opponent behavior, then use minimax with piece advantage and estimated risk. The abstract reports strong results against baselines including PurpleMax, without numerical details. [Publisher abstract](https://link.springer.com/chapter/10.1007/978-3-032-23657-9_15).

**Project transfer idea:** A compact inference baseline could update hypotheses from revealed evidence and expose uncertainty to the evaluator. Compare uniform, usage-prior, and behavior-updated hypotheses under the same battle policy.

**Limit:** Opponent behavior can be misleading. Separate hard contradictions from soft behavioral assumptions; do not eliminate legal sets merely because an opponent played unexpectedly. The Geister result is not evidence that ordinary minimax solves VGC's simultaneous-action problem. Full paper was unavailable in this review.

### 6. SpinGPT — useful evaluation lessons

**Verified:** Maugin and Cazenave train an 8B model on expert decisions, then refine it using solver-generated examples. The author preprint reports roughly twenty A100 GPU-hours across the two stages. Crucially, the Slumbot result is for **SFT-only with an added deep-stack heuristic**, not the final model. The final model was compared against SFT-only; the paper acknowledges it was not evaluated against Slumbot. [Author preprint, §§3–5](https://arxiv.org/html/2509.22387v1).

**Project transfer idea:** Keep behavior-cloning accuracy, battle win rate, and team-search performance as distinct evaluations. Record which checkpoint, policy patches, team distribution, and information protocol produced each result. Test whether solver-derived improvement generalizes beyond the teacher or training opponent.

**Limit:** This is not a reason to replace VGC-Bench with an LLM. Its action-label agreement cannot establish low exploitability, and its poker benchmark differs from the target tournament format. Review depth: full public preprint, not a verified comparison against the final Springer text.

## Practical reading order and experiments

1. **Complete the current baseline first.** Establish reproducible battle outcomes and legal actions before adding any conference method.
2. **If joint-action branching dominates runtime, read Evolutionary Tree Search.** Compare proposed joint actions against the existing search at equal simulator cost.
3. **If stochastic outcomes destabilize evaluations, read Adapting MCTS.** Track both action decisions and stability of team rankings across independent evaluation runs.
4. **If one opponent policy is being exploited, read the diversity paper.** Keep held-out opponents and separate training from final assessment.
5. **If hidden-set sampling creates impossible states, inspect the ASP appendix.** Compare logical consistency and runtime before adopting a new reasoning system.
6. **Read SpinGPT's evaluation section as a short methodology exercise.** It gives a concrete example of why the exact evaluated agent matters.

These are proposed experiments, not results established by the conference. None of the reviewed sources establishes a scalable solution to the project's outer noisy combinatorial team search, team-preview selection, and simultaneous imperfect-information battle policy together. The most defensible use of this conference is to borrow one component at a time and check whether it improves the existing pipeline.

## Access and confidence

The event page intermittently rejected direct retrieval; its indexed first-party content provided the full accepted-paper list. Publisher abstracts were accessible for the screened proceedings papers except Grassauer's chapter, where the author appendix supplied method evidence. SpinGPT's full author preprint and Tagiron's accepted manuscript were accessible. Most other recommendations remain abstract-level screening, so implementation details, statistical reliability, and VGC transfer require further reading and experiments. No paywall was bypassed, papers purchased, or authors contacted.
