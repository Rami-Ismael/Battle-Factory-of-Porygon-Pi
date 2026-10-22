# Research diffusion-guided mutation for Pokémon team search

Help me develop and critically evaluate research on diffusion models as mutation operators for Pokémon team search. Continue from the decisions below. Use recall to recover relevant history before asking questions, and apply writing-for-agents when briefing subagents.

## Carry forward my goal and proposed strategy

My goal is to generate complete Pokémon teams with higher measured battle win rates than the best teams in the training data. I want search to use its evaluation budget more effectively. Higher battle win rate is the outcome that matters.

My proposed strategy is local search around good parent teams. Preserve part of a parent and use a score-conditioned diffusion model to propose changes. Use classifier-free guidance to steer proposals toward higher requested win rates.

Use diffusion for most mutations and retain some legal random mutations for exploration. Increase the share of random mutation when search gets stuck. The exact proportions and adaptation rule remain undecided.

My hypothesis is that the model can concentrate evaluations on local changes that are more likely to improve the parent. This could help search find stronger teams within a fixed evaluation budget. Treat this as a hypothesis to test. Neither classifier-free guidance nor a high requested score guarantees improvement beyond the training data.

## Recover facts before interviewing me

1. Read the project instructions and apply [recall](.agents/skills/recall/SKILL.md). Recover relevant decisions, experiment protocols, and results. Separate explicit user decisions from assistant recommendations and implementation defaults.
2. Apply [writing-for-agents](.agents/skills/writing-for-agents/SKILL.md) to brief a claim-checking subagent. Give it a bounded question, ownership of its evidence sources, and a checkable completion criterion. Require a compact claim ledger with direct sources, limitations, and unresolved gaps. Keep parallel work read-only or assign distinct output files.
3. Resolve discoverable facts yourself. Use project files, relevant chat history, and primary research sources. Ask me about unresolved choices and my reasoning. Avoid questions that repeat settled goals or ask me to retrieve available facts.

Start with the completed pilot under `/Users/ramiismael/Documents/code/vgc-team-generator-pilot`:

- `docs/diffusion-mutation-arm.md` records the protocol.
- `scripts/diffusion_mutation_arm.py` implements the comparison.
- `results/diffusion_mutation_arm.json` contains the saved results.

That pilot compared handwritten mutation, uniform legal refill, and diffusion refill. It found better verified finalists against the opponents used for search, with inconclusive differences against held-out ranks 51–100. Verify the artifacts before quoting exact results. The pilot does not establish that classifier-free guidance caused the gain or that the proposed adaptive mixture works.

For nearby prior work, inspect [Diffusion Mutation Kernels for Circuit and Physical Design](https://arxiv.org/abs/2608.27649) and [Diffusion Models for Black-Box Optimization](https://proceedings.mlr.press/v202/krishnamoorthy23a.html). Check each claim against the actual method and evaluation.

## Continue grilling the research idea

Challenge the reasoning behind my proposal. Ask focused questions in dependency order and explain why each unresolved decision matters. Give a recommendation with its tradeoff. Carry my answers forward instead of restarting the interview.

The next open decision is what counts as stagnation under noisy battle evaluation. Fresh-battle checks across several batches were suggested, but I have not accepted a specific rule. Also resolve the initial mutation mixture, how it changes, when it returns toward diffusion, and whether the model stays frozen or learns from new evaluations.

Keep requested win rate, CFG strength, model probability, and measured battle win rate separate. Distinguish unconditional sampling, ordinary score conditioning, and amplified guidance. Confirm the implementation's guidance equation before interpreting numeric settings.

## Prepare a concrete study proposal

After resolving the necessary decisions, propose a comparison that tests whether the hybrid finds stronger teams within the same budget. Recover the existing opponent pools, battle policy, and historical budget. Identify any new choices without silently assigning a new budget or opponent distribution.

Recommend controls that distinguish the benefit of learned proposals, score guidance, and adaptive mixing. Consider uniform legal mutation, the existing handwritten operators, diffusion mutation, a fixed mixture, and the adaptive mixture. Justify which comparisons are necessary before adding more arms.

Compare finalists and strong training-data baselines on fresh battles under the same protocol. Account for noisy selection and variation across independent searches. Record which operator produces improvements. Report new battle evaluations separately from training-data collection, model training, tuning, and sampling costs. Charge stagnation checks to the declared evaluation budget.

Finish with a concise research brief containing the testable hypothesis, confirmed decisions, remaining choices, proposed comparisons, evaluation protocol, cost accounting, and limits on public claims. Treat a conditional or negative result as a valid research outcome. Present the concrete plan for confirmation before implementation. This instruction authorizes investigation and planning; experiments and publication require subsequent authorization.
