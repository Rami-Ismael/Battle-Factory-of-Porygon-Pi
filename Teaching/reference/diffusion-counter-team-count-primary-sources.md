# Does guided diffusion produce only ten teams that beat the meta?

Research date: 2026-09-16. This note explains a proposed search design; it does not report implemented project outcomes or measured winning teams.

**No fixed number such as ten is built into diffusion.** Ten can be the number of samples requested, the size of an evaluation batch, or the maximum number of finalists returned. None of those numbers establishes how many successful teams exist.

## What the primary sources establish

- **Candidate generation and feasibility are separate.** DIFUSCO represents combinatorial candidates with discrete variables. Section 3.5 explicitly states that its generative model cannot guarantee feasibility, so it uses problem-specific decoding. Different random seeds can produce different outputs; Appendix C describes sampling multiple solutions and reporting the best. This supports using a diffusion model as a proposer with a separate legality mechanism, not as a certificate of Pokémon legality or victory. Its experiments concern TSP and independent sets, not Pokémon. [DIFUSCO, NeurIPS 2023, §3.5 and Appendix C](https://proceedings.neurips.cc/paper_files/paper/2023/file/0ba520d93c3df592c83a611961314c98-Paper-Conference.pdf)
- **Guidance changes sampling preferences.** Classifier-free guidance combines conditional and unconditional model estimates and studies a quality/diversity tradeoff. Applying an analogous approach to team rewards would be a design choice requiring its own experiments. The paper does not show that Pokémon reward guidance guarantees victory, unique teams, or discovery of every good team. [Ho and Salimans, Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- **Team strength depends on piloting and opponents.** VGC-Bench explicitly distinguishes team building from team usage and addresses the latter. Its cross-play equation includes both policies and team configurations. It evaluates training pools of 1, 4, 16, and 64 teams and reports degradation as team diversity increases. Those are benchmark pool sizes, not counts of generated counters. Its evaluation methods include unseen-team generalization and approximate exploiter policies. It also reports finite-game uncertainty. [VGC-Bench, §1, §4.3, §5.1](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/angliss2026vgc.pdf)

## Implications for this project — reasoning, not published Pokémon results

An illustrative search could be:

**1,000 proposal attempts → legality checks and deduplication → screening battles → fresh finalist battles → up to 10 qualified, diverse teams.**

These numbers are examples only. The final count could be zero, three, ten, or more before the output cap. If only three qualify, report three; do not relabel seven weak teams as successful merely to fill ten slots. More proposals may repeat existing teams, fail legality, or perform poorly.

Define “beat the meta” before counting. A useful target is an expected win rate above a chosen threshold against a dated, format-specific distribution of opposing **full teams and pilots**. Beating one top team, beating a weighted mixture on average, and beating every individual matchup are different objectives. An average advantage can hide a bad matchup. Winning one battle does not establish a reliable advantage.

Finalists need fresh battles because selecting the highest noisy screening scores can exaggerate performance. Report win rates with uncertainty, the battle budget, opponent pool, simulator/rules version, and the pilot used. A statistical qualification rule provides evidence under those conditions, not a guarantee of winning every battle. Testing many teams also requires attention to selection and multiple comparisons before making simultaneous confidence claims.

If “guide my Pokémon” means **choose a favorite species or fixed core**, retain those choices as explicit constraints and search the remaining configuration. This narrows the allowed space but does not establish whether ten good completions exist. If it means **you personally pilot the team**, bot evaluation is only a proxy for your outcomes; human playtesting is needed because the pilot changes.

Keep three counts separate:

1. Proposal attempts (including repeats and invalid outputs).
2. Distinct legal candidates evaluated.
3. Distinct candidates meeting the declared performance criterion.

Also define diversity: ten full configurations differing only in EVs are not necessarily ten distinct species lineups or ten strategic archetypes. Canonicalize order where appropriate and report the selected diversity definition.

The total number of winning teams remains unknown. Guided samples preferentially visit some areas of the search space, so finding ten successful candidates does not estimate how many successful teams exist overall. Failing to find another successful team does not prove none remains. Counting the entire successful set would require a separate, well-defined counting/estimation problem with suitable coverage assumptions, not merely additional guided sampling.
