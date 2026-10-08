---
created_at: 2026-08-19
type: learning-record
lesson: 0005-reevo-llm-writes-the-heuristic.html
---

# 0006 — ReEvo: Language Hyper-Heuristics (Ye et al., NeurIPS 2024)

Taught 2026-08-19, serving the ReEvo/TIDE todo's child 5 (evolved pruning rules); retrieval quiz not yet confirmed taken.

Key insights the lesson carries:

- **The search runs over programs, not solutions** — each individual is heuristic code; the best program is the product. This is the level-up ("hyper-") that distinguishes it from every search method previously studied on the $f$ strand.
- **Two LLM roles**: generator writes code; reflector writes *text* ("verbal gradients") explaining pairwise score comparisons, which is prompt-fed into the next generation. Short-term reflection is per-pair; long-term reflection is the accumulated cross-generation memory steering elitist mutation.
- **Everything rides on the meta-objective being cheap** — F = average performance on an instance dataset, run once per candidate. This is the taught justification for the project's application choice: pruning rules score battle-free (rank-weighted retention × prune fraction), whereas evolving the team-builder itself would put an estimator of $f$ inside the loop.
- **White-box beats black-box here**: the LLM's Pokémon knowledge is usable in prompts, which blind mutation cannot exploit — the owner's regulation description belongs in the task specification.
- **Scope honesty**: no games/team-selection domain anywhere in paper, repo, or project page; SOTA claims are vs FunSearch/EoH on cheap-evaluation COP benchmarks.

Open decision left with the owner (lesson's closing decision, also the open grilling question): hard filter vs soft penalty for what an evolved rule does to a condemned team.

Context: TIDE was dropped by the owner 2026-08-19 (note frontmatter edit); this lesson deliberately covers ReEvo only.
