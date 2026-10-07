---
created_at: 2026-08-23
type: research-report
question: What does "A Survey on Self-play Methods in RL" (arXiv:2408.01072) give me that this vault doesn't already have?
relevance: battle-policy arm only
---

# Verdict

The vault already owns the **VGC instantiation** of self-play (VGC-Bench's four arms, the FP-is-a-mixture lesson, the DO lesson). The survey owns three things the vault cannot generate for itself: a **map** (where each arm sits in one taxonomy), a **theory column** (which methods carry convergence guarantees and under what assumptions), and an **evidence table from other games** (which recipes survived contact with which game shapes). Details below, then the two gaps that actually change decisions in this project.

## What the vault already covers (survey adds nothing)

| Topic | Where it already lives |
| --- | --- |
| Vanilla / naive-FP / PFSP / DO as opponent-selection variants | `Papers/A Survey….md` table + `Fictitious play arm…` + `Pure self-play arm…` |
| FP = draw one frozen snapshot, not an averaged network | Lesson 0008 + record 0009 |
| ORACLE vs BR vs ABR; DO needs exact BR for its guarantee | Lesson 0002 + `self-play-framework-glossary.html`; quote verified verbatim |
| VGC-Bench's "double oracle" arm is really PSRO-Nash | Record 0002 finding |

## Survey-only content

1. **The unified loop as typed functions (§3.1, Algo. 1)** — ORACLE/EVAL/MSS with explicit input-output types (MSS: 𝒫 → Σ). The vault had the pseudocode but not the type view; the type view is what makes "swap the MSS" a cheap, well-defined experiment. 🟢 [paper §3.1](https://arxiv.org/html/2408.01072v4)
2. **δ-uniform self-play (§3.2.4)** — uniform over the newest slice of history, pruned storage; the recipe behind Honor of Kings. Absent from every vault note except the paper-note frontmatter line.
3. **Convergence theory placement (§3.3–3.5)** — which family carries a guarantee (DO → NE of the full game *if* the oracle is exact BR; regret-minimisation series → CFR/NFSP) and Corollary 2 (Σ stays lower-triangular). The vault has the CFR primer pointer (Todo §5, shallow-red) but no statement of *what* is guaranteed or why.
4. **Non-transitivity diagnosis (§5.2)** — vanilla SP "is effective in transitive games, but it can lead the agent to cyclic learning patterns in non-transitive games." This single sentence is the theoretical backing for the ladder-vs-circle figure and for measuring who-beats-whom tables instead of one Elo number. The vault asserted the circle; the survey cites it.
5. **Cross-game evidence of mixed recipes (§4)** — e.g. the Dota bot sampling newest-self 80% / prioritised 20%. Licence to mix MSS rules inside one run instead of treating arms as mutually exclusive.
6. **Policy-pool storage economics (§5.3)** — what to keep, prune, or compress; directly relevant to the frozen-snapshot population growing forever under FP.

## The two gaps that change project decisions

- **Decision gap:** if the battle-policy experiments stall in cycles (the RPS failure mode), the survey says the fix is upstream of training — change the opponent-selection rule (MSS) or add a promotion gate — not more PPO epochs.
- **Scope guard:** the survey covers the battle-policy arm only. It has nothing on team search, synergy measurement, or the $f$ predictor strand; do not let its authority leak into those notes.

🟢 all claims checked against arXiv:2408.01072v4 HTML ([abs](https://arxiv.org/abs/2408.01072) · [alphaXiv overview](https://www.alphaxiv.org/overview/2408.01072v4)); 🔒 none needed (no paywalled sources used).
