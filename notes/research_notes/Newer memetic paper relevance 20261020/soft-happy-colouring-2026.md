# Relevance of the 2026 soft happy colouring paper

Assessed on 20 October 2026 for the Pokémon counter-team search project.

**Verdict: medium relevance for a selective design read. Keep it supplementary. Do not replace a current search-loop priority with this paper just because it is newer.** It gives a recent example of cross-entropy search combined with local improvement. It does not supply a practical local improver for Pokémon teams.

## What was verified

The [version 3 record](https://arxiv.org/abs/2603.11050v3) identifies *Overcoming Tight Constraints in Soft Happy Colouring*, by Mohammad H. Shekarriz, Asef Nazari, and Dhananjay Thiruvady, revised 3 August 2026. The record lists a January 2027 journal issue. This assessment concerns the available 2026 preprint. The [full manuscript](https://arxiv.org/pdf/2603.11050) identifies version 3 on its first page.

Section 4.1 uses a separate categorical probability vector for each free graph vertex. Section 4.2 and Algorithm 4 apply a graph-specific local improver to every sampled colouring before scoring and selection. The improver uses neighbouring colours and is described as linear in the number of graph edges. Elite frequencies update the sampling probabilities. Figure 1 shows this sequence. These are deterministic graph scores. [Manuscript, sections 4.1 and 4.2](https://arxiv.org/pdf/2603.11050).

## Fit to the actual project

| Question | Fit judgment |
| --- | --- |
| Does it study categorical choices? | Yes. This clears the project's continuous-only exclusion. Graph colouring also includes interactions between choices. |
| Can the local improver be reused? | No direct implementation transfer. Pokémon has no established equivalent of its neighbouring-colour rule. A legal move replacement still needs a reliable strength comparison. |
| Does its sampler match the current generator? | No. The project resets and refits neural generators. Copying a frequency-smoothing setting would change the algorithm rather than configure an existing control. |
| Does it solve expensive, noisy evaluation? | No demonstrated transfer. The project must estimate win rate through battles. Its ridge model chooses which proposals receive battle evaluation, and nearby teams can be misranked. |
| Does it establish improved Pokémon win rate? | No. That remains the project's experiment to conduct under its applicable opponents, policies, and cost controls. |

The comparison uses the existing [implementation audit](../Landscape%20analysis%20inside%20CEM%2020261020/current-cem.md). It does not assume that every saved search arm has the same configuration.

## What is worth taking from it

The useful question is **where local improvement enters the loop**. This paper improves candidates before selection. That differs from refining only selected candidates or generating a limited share of local proposals.

For the current project, a limited local-proposal branch remains a more defensible first comparison. Keep the chosen baseline's neural refit and ridge acquisition fixed. Count the local candidates and their battle costs within the existing allowance. Compare the selected output on independent battles. This is a proposed test, not a result from the paper.

Do not copy its full-population refinement rule. That choice needs a cheap, trustworthy local improver. Designing and validating such an improver is the missing work in Pokémon.

## Limits in the paper's evidence

The main graph experiment uses a 600-second limit. Section 5.5 reports about 6% of hybrid runs still trapped at that limit. Theorem 4.1 concerns a deterministic local-search map and categorical probability matrices. The main setup reports a smoothing value of 0.1, while section 5.4 reports adopting 0.15. [Manuscript, sections 4.3, 5, 5.4, and 5.5](https://arxiv.org/pdf/2603.11050).

Consequently, the abstract's broad language about resolving stagnation should not become a claim about the current neural search. Convergence of sampling probabilities also differs from finding the strongest team within a finite battle allowance. This assessment does not certify the theorem's proof. A theorem audit and reconciliation of the reported settings would be needed before relying on either.

## Selective reading order

1. **Algorithm 4, printed page 13, and Figure 1, printed page 15.** Identify exactly where local improvement enters.
2. **Section 4.1, printed pages 9 to 12.** Check how the categorical sampler differs from the neural generator.
3. **Sections 5.4 and 5.5, printed pages 29 to 32.** Read the cost trade-off and finite-time failures before interpreting the headline claims.
4. **Theorem 4.1, printed pages 16 and 17.** Read only if a convergence claim becomes relevant to the project.

For learning the general combination, this can replace the 2013 paper as the newer illustration. It is not a stronger reason to implement local search, and it does not move ahead of the existing game-based search and evaluation readings.

Verification depth: primary full-text review of the problem definition, sampler, local-search placement, convergence argument, experimental setup, calibration, and time-to-target discussion. No code execution, reproduced result, new battle, or experiment-budget choice was made.
