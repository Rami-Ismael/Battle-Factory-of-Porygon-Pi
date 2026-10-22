# Immediate reading recommendation

## Which candidates have the strongest immediate fit?

### Takeaway
Read **Diffusion-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization** now, completing the part you are already trying to understand. **Generative Bayesian Optimization: Generative Models as Acquisition Functions** is the strongest next new paper, but the current notes provide a more urgent, specific learning question for Diffusion-BBO.

### Cited Findings

| Candidate | Evidence of fit | Why it wins or waits |
| --- | --- | --- |
| **Diffusion-BBO** | Its note explicitly says “What that I am reading right now”; the owner asks how inverse modeling actually optimizes rather than merely matching a score, how its iterations work, and how target scores should be chosen. [Owner notes, lines 1–15](../../Papers/Diffusion-BBO%20Diffusion-Based%20Inverse%20Modeling%20for%20Online%20Black-Box%20Optimization.md#L1) | **First now:** it answers the most concrete unresolved question about the proposed team generator. The purpose is to understand its optimization mechanism, not assume it improves Pokémon win rate. |
| **Generative Bayesian Optimization (GenBO)** | Current list: HIGH and “go through this paper.” Local assessment identifies categorical/text/protein experiments and a concrete CEM comparison: improvement-magnitude weighting versus binary elite membership. [Current list, line 125](../../Papers/Reading%20List.md#L125); [assessment, lines 16–28](../../Papers/Generative%20Bayesian%20Optimization%20%E2%80%94%20relevance%20to%20counter-team%20search.md#L16) | **Best next new paper:** useful once the existing loop is clear. Its weighted generator update is a concrete alternative, but overlap with CEM is not evidence of an advantage. |
| **Posterior Inference with Diffusion Models for High-dimensional Black-box Optimization** | The current list says “Reading this right now”; the owner connects it to CEM, forward prediction, and lack of diverse data. A lesson on its prior/predictor/posterior and follow-ups was requested October 12. [Current list, line 128](../../Papers/Reading%20List.md#L128); [teaching log, line 32](../../Teaching/NOTES.md#L32) | **Close alternative:** continue after distinguishing inverse target conditioning from a forward predictor plus posterior sampling. This is the 2025 paper, not the different 2026 diffusion-language-model DiBO paper. |
| **Design by adaptive sampling** | Its subsection is marked “Reading this now”; the owner's row explicitly values its relationship to CEM. The historical direction describes the project as CEM with a generator as its density. [Current list, lines 108–113](../../Papers/Reading%20List.md#L108); [history, lines 58–64](../../Papers/Reading%20List%20%E2%80%94%20notes%20and%20decision%20history.md#L58) | **Use as a short CEM bridge:** helps clarify the current algorithm, but addresses target-score selection less directly than Diffusion-BBO. |
| **Think Global and Act Local (CASMOPOLITAN)** | Owner-selected September 28 as a reference for high-dimensional categorical/mixed choices and useful team neighborhoods; BOCS is selected beside it for expensive combinatorial search. [Current list, lines 144–145](../../Papers/Reading%20List.md#L144) | **Later baseline choice:** stronger domain fit than continuous-only analogies, but it introduces a different surrogate/acquisition framework before the immediate inverse-model question is settled. |

### Inferences
- Immediate reading value favors resolving the current generator question over starting another method. This is a recommendation about the next reading session, not an algorithm ranking.
- Battle-policy papers and generic game-theoretic rankings do not directly answer the question of how to propose stronger legal teams. The user's objective here remains battle win rate, with diversity serving as a search diagnostic rather than a replacement objective. The list separates search from battle-policy learning. [Current list, lines 85 and 369](../../Papers/Reading%20List.md#L85)

### Gaps
- Two papers have “reading now” annotations, without completion dates. Neither should be described as unread or completed. [Diffusion-BBO note, line 1](../../Papers/Diffusion-BBO%20Diffusion-Based%20Inverse%20Modeling%20for%20Online%20Black-Box%20Optimization.md#L1); [current list, line 128](../../Papers/Reading%20List.md#L128)

## Which previous priorities are stale or conditional?

### Takeaway
The decision history is explicitly an archive. Its old tier order and HIGH-first rules cannot determine today's first paper.

### Cited Findings
- History line 12 says to use the current list for acceptance. October 13 explicitly puts current-stage relevance before the older HIGH → reimplement → MED rule and promotes Diffusion-BBO ahead of reward fine-tuning papers. [History, lines 12–25](../../Papers/Reading%20List%20%E2%80%94%20notes%20and%20decision%20history.md#L12)
- The October 13 premise that fine-tuning had not started conflicts with the owner's dated October 16 discovery that `cem_v3` fine-tunes 40 times and that “elite only” meant a score threshold rather than local generation elites. This warrants describing the current loop accurately; it does not automatically reopen all reward-integration papers. [Diffusion-BBO note, lines 27–29](../../Papers/Diffusion-BBO%20Diffusion-Based%20Inverse%20Modeling%20for%20Online%20Black-Box%20Optimization.md#L27)
- Old SPIGM recommendations are explicitly superseded: only Two Temperatures and Re-evaluating Confidence Remasking remain accepted, both MED. Older COExpander MED and other neural-CO grades also differ from the current LOW entries. [History, lines 67–84 and 449–463](../../Papers/Reading%20List%20%E2%80%94%20notes%20and%20decision%20history.md#L67); [current list, lines 101–103](../../Papers/Reading%20List.md#L101)
- The separate global ranking contains only Active Flow Matching, while that paper's note records unresolved forward-KL and variational-objective questions; it supplies no dated comparative rationale. [Global ranking, lines 1–2](../../Papers/Global%20Ranking%20of%20Reading%20List%20ignore%20group%20and%20categories.md#L1); [AFM note, lines 2–5](../../Papers/Active%20Flow%20Matching.md#L2)

### Inferences
- Do not use HIGH grades, citation counts, historical numbering, or an unexplained single-entry ranking as substitutes for the owner's latest concrete question.

### Gaps
- The notes do not resolve whether the October 16 statement refers to generic CEM refitting, the deferred reward methods, or a changed project stage. Do not silently collapse those distinctions. [Owner note, line 29](../../Papers/Diffusion-BBO%20Diffusion-Based%20Inverse%20Modeling%20for%20Online%20Black-Box%20Optimization.md#L29)

## What should the next reading session accomplish?

### Takeaway
**Confidence: moderate to high** that Diffusion-BBO is the best current reading action; lower confidence that any listed method will improve the project without an experiment. Stop this reading session when one full iteration can be explained in terms of the owner's existing team search.

### Cited Findings
- The unresolved questions concern target selection, iteration, diversity collapse, and the distinction between the inverse proposer and a ridge filter choosing 128 of 512 candidates; the owner also requests comparisons against evaluating all candidates now that battles are faster. [Owner notes, lines 6–15](../../Papers/Diffusion-BBO%20Diffusion-Based%20Inverse%20Modeling%20for%20Online%20Black-Box%20Optimization.md#L6)

### Inferences
1. Trace: evaluated teams and measured scores → model update → requested target score → generated batch → actual battles → expanded dataset. Mark the difference from CEM's elite selection.
2. Explain why asking for a higher score alone is insufficient, where new information enters, and how the next request is selected. Separate paper mechanisms from Pokémon adaptations.
3. Write one concrete comparison question for the existing experiment protocol. Do not invent an opponent distribution or battle budget, and do not promise a gain.
- **The condition that changes this recommendation:** if the owner can already explain this entire Diffusion-BBO loop and target-selection mechanism confidently, switch to **GenBO**, reading Section 3/Algorithm 1, Section 6.2, and Appendix C.3–C.5 as the local guide recommends. [GenBO guide, line 28](../../Papers/Generative%20Bayesian%20Optimization%20%E2%80%94%20relevance%20to%20counter-team%20search.md#L28)

### Gaps
- Original-paper section verification is delegated separately. This comparison relies on local reading evidence; local recorded uncertainty percentages are not treated as independently verified experimental facts.
