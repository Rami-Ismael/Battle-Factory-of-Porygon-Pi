---
created_at: 2026-08-23
type: learning-record
lesson: 0008-fictitious-play-is-a-mixture-not-a-mean.html
status: lesson ready — not yet confirmed learned
---

# 0009 — Fictitious play is a mixture, not a mean

Requested 2026-08-23 as step 1–2 of `What are the step to red the paper` against `Papers/A Survey on Self-play Methods in Reinforcement Learning`. The owner supplied the correction himself: people describe fictitious play as "an average of all past versions" when it is a population of **frozen** snapshots — each saved after n battles and never updated again — sampled one per battle, in contrast to vanilla self-play where the opponent is the live network. His own paper-note body carries the wrong phrasing (`- **Fictitious play** — train against the _average_ of all past versions`); he flagged it rather than asking for it to be edited, and it was left in place.

What the lesson carries:

- The survey's four framework objects (§3.1): policy population Π, interaction matrix Σ, meta-strategy solver MSS: ℝ^{K×C₂} → ℝ^{K×C₁}, and ORACLE. Mapped onto his existing `Battle Policy` pseudocode — step 1 reads a row of Σ, step 2 is the ORACLE, step 3 grows Π, and what he called "the flavor" is the MSS.
- Three disambiguations of "average", only the third being FP: (A) parameter averaging (EMA/SWA/Polyak — a deployment trick, and the thing the Generals.io paper contributes), (B) NFSP's supervised average-policy network over a reservoir — real, but an implementation of the mixture, not the definition, (C) a mixed strategy, σ_{m+1,n} = 1/(m−1) for n ≤ m−1. Brown's "empirical frequency of the opponent's past play" is where the word legitimately comes from — an average over the distribution, never over parameters, and never instantiated.
- The contrast the mistake erases is **frozen vs live**, not one-vs-many.
- Answers his open question in `Why Self-Play is consider Multi-Agent.md` ("does a frozen agent make the environment stationary?"): stationary conditioned on the sampled opponent; stationary marginally while Σ's row is fixed; drifting by O(1/m) at each append — versus vanilla self-play's per-gradient-step drift with no shrinking bound. That is the precise content of "piecewise stationary", which was already a ticked checkbox there.
- All five arms as one row of Σ (§3.2.2–§3.2.5, §3.3.2), plus the note that δ-uniform interpolates: δ=0 is FP, δ=1 is vanilla. A fourth arm is a new δ, not a new algorithm.
- Re-flagged from lesson 0002: his gated-promotion "double oracle" arm is a PSRO-family heuristic, not DO.
- The four branches (§3.2 traditional / §3.3 PSRO / §3.4 ongoing-training / §3.5 regret-minimisation) and the one he is not considering — NeuPL (§3.4.3), one conditional network instead of a checkpoint zoo, which is a throughput argument at poke-env speed.

Section numbering verified against arXiv 2408.01072v4 (fetched 2026-08-23); the frontmatter `key_sections` list on the paper note is correct. No new asset; the lesson uses one inline SVG (live-chain vs frozen-population) and `assets/quiz.js`.

Sources added: 🟢 Heinrich, Lanctot & Silver, *Fictitious Self-Play in Extensive-Form Games*, ICML 2015 — PMLR v37 PDF; 🟢 Liu et al., *NeuPL*, ICLR 2022 — OpenReview `MIX3fJkl_1`.

Open, not taught: whether NeuPL earns a fourth arm; whether §5.2 says anything beyond the three stationarity bullets.

## Addendum, same day — the AlphaGo family (owner: "AlphaZero is a unique form of self play")

He pushed back on the ELI5's throwaway "(this is the AlphaZero family)" attached to vanilla self-play. He is right, and §4.1.1 is richer than the shorthand: the family changes its MSS three times. **AlphaGo** stage 2 refines p_ρ "against a randomly selected historical version p_ρ⁻, similar to the MSS shown in Equ. (6)" — that is *fictitious play*, not vanilla. **AlphaGo Zero** is Eq. (5) *plus a 55% promotion gate* ("must surpass a 55 percent win rate against its predecessor, aligning with the stipulation set in Algo. 2 at Line 1"). **AlphaZero** removes it — "the only difference … is that AlphaZero utilizes the newly updated network without the validation process." **MuZero** keeps AlphaZero's row.

Two consequences worth carrying into the experiment, both now in the lesson as §4a/§4b and in ELI5 §7:

- AlphaGo Zero's gate **is** his DO arm's promotion rule, pointed at the same Algo. 2 line — and the successor paper deleted it. One published data point against the gate, in a transitive perfect-information game; not decisive for VGC, but he should know who dropped it.
- AlphaZero wins on the *simplest* row because of two ingredients that are not opponent selection: MCTS as the real policy-improvement operator (§3.2.2 notes the coupling), and Go's transitivity — "effective in transitive games, but it can lead the agent to cyclic learning patterns in non-transitive games." His arms have neither. So AlphaZero is evidence *for* the FP / δ-uniform / gated arms, not against. Reinforced by §4.3.2: OpenAI Five is 80% vanilla / 20% PFSP, Honor of Kings 1v1 is δ-uniform — nobody in a non-transitive game runs a pure row.

ELI5 gained §7 (four-generation strip SVG + ladder-vs-circle SVG, reusing the rock-paper-scissors circle from §2 as the callback); old §7 renumbered to §8 and its "vanilla self-play in Go" clause corrected. Lesson gained §4a, §4b and three quizzes (now 8).
