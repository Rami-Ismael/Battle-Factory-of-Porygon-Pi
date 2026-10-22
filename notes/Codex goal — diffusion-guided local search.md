Create and pursue a goal to research, implement, run, and evaluate diffusion-guided local search for Pokémon team optimization. Complete the experiment campaign and report its results. Proceed autonomously without a grilling or requirements interview.

Work in this project:
`/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team`

My scientific objective is to generate complete Pokémon teams with higher measured battle win rates than the best teams in the training data, while using the evaluation budget more effectively. Treat improvement as a hypothesis to test, not a result required for completion.

Carry forward these decisions:

- Start local search from good parent teams. Preserve part of each parent and use a score-conditioned diffusion model with classifier-free guidance to propose mutations.
- Use diffusion for most mutations and retain some legal random mutations for exploration.
- Increase the share of random mutation when search gets stuck.
- Judge success through actual battle win rates. Requested scores, model probabilities, and CFG strength are separate quantities.

The hypothesis is that guided mutation concentrates evaluations on useful local changes, allowing search to find stronger teams within the same battle budget. The exact mutation proportions, stagnation rule, adaptation schedule, and guidance settings remain open.

Follow this process:

1. Read the project instructions and apply recall. Recover previous decisions, current code, experiment protocols, and results. Resolve discoverable facts yourself. Distinguish my decisions from assistant recommendations and implementation defaults.
2. Apply writing-for-agents when briefing subagents. Assign a claim-checking subagent bounded ownership of evidence checks. Require direct sources, supported conclusions, limitations, and explicit gaps. Separate concurrent output files and preserve other sessions' work.
3. Inspect the existing pilot in `/Users/ramiismael/Documents/code/vgc-team-generator-pilot`, especially `docs/diffusion-mutation-arm.md`, `scripts/diffusion_mutation_arm.py`, and `results/diffusion_mutation_arm.json`. Verify the recorded gain against search opponents and inconclusive held-out results. That pilot does not isolate CFG's contribution or validate the proposed adaptive mixture.
4. Check relevant primary research, including diffusion mutation kernels and diffusion-based black-box optimization. Assess novelty narrowly. Separate learned team structure, ordinary score conditioning, amplified CFG, and adaptive mixing as possible explanations for improvement.
5. Choose and justify practical rules for mutation proportions, stagnation under noisy evaluation, adaptation and recovery, guidance, and whether the model learns during search. Record these implementation choices separately from my settled decisions. Use the existing documented pilot protocol, opponent pools, battle policy, seed count, and per-arm battle budgets as the default. Preregister the comparisons, settings, stopping rules, and total campaign cost before the main runs. Account for tuning, training, verification, and all arms and seeds. Charge stagnation checks to the adaptive arm's budget. Keep held-out evaluation separate from tuning and disclose any previous use of that pool.
6. Implement the experiment using existing project mechanisms. Include legal random mutation, handwritten mutation, diffusion mutation, a fixed mixture, and the adaptive mixture where needed to test the claims. Add focused controls for score conditioning and CFG. Define the guidance equation and share the trained model across guidance controls. Match initialization, selection, legality handling, evaluator settings, mutation size where applicable, and total evaluation budgets. Record which operator produces each offspring and improvement.
7. Inspect running jobs and shared resources before writing or launching runs. Use an owned checkout or worktree, isolated run directories, compatible checkpoints, and separate simulator resources. Preserve other sessions' edits and results. Save configuration, data and checkpoint provenance, code revision, seeds, logs, and resumable state. Run meaningful tests and a small smoke experiment before the full campaign.
8. Run the preregistered campaign to completion within its recorded budget. Monitor failures, investigate invalid runs, and resume compatible checkpoints when appropriate. Evaluate finalists and strong training-data baselines on fresh battles under the same protocol. Use independent search seeds and report uncertainty that reflects the sampling structure. Account for incomplete battles explicitly. Keep historical results separate from new results unless their protocols are demonstrably compatible.
9. Analyze win rates, improvement per evaluation, adaptation behavior, operator contributions, diversity where relevant, and runtime. Report data collection, model training, tuning, inference, and battle-evaluation costs separately. Have the claim-checking subagent audit the conclusions against the saved evidence. Write a reproducible results report with tables, figures, limitations, and the strongest defensible conclusion.

Complete the goal when:

- The implementation and meaningful validation checks are complete.
- All required campaign runs and fresh evaluations have finished, with their actual costs recorded.
- Raw results, provenance, logs, and reproduction instructions are saved.
- A results report and sourced claim ledger distinguish findings, uncertainty, limitations, and remaining hypotheses.
- The final response links the code and artifacts, reports the numerical comparisons, and states whether the evidence supports the proposed method.

Make reasonable implementation choices, document them, and continue through execution without stopping for approval of a research plan. A negative or inconclusive result is a valid outcome. Starting a background job does not complete the goal. If an unavoidable blocker prevents completion, report the blocker and preserve resumable work. Keep the goal incomplete while required work remains. Completion never requires unlimited runs or a positive result.

This goal authorizes research, implementation, necessary model training, local experiment execution, analysis, and reporting. Save a report suitable for later public sharing.
