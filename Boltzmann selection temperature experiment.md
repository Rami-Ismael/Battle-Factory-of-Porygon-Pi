# Boltzmann selection temperature experiment

**Completed 2026-09-08:** 15 runs across three seeds, 165 search generations,
and 921,600 battles. The full protocol audit and all 21 tests passed.

| Selection temperature | Fresh generator win rate | Distinct species sets per 512 proposals |
|---|---:|---:|
| Annealed 0.30 → 0.10 | 33.25% | 255.0 |
| Fixed 0.10 | 36.05% | 172.0 |
| Fixed 0.20 | 34.58% | 204.3 |
| Fixed 0.30 | 32.15% | 276.3 |
| Adaptive ESS/pool = 0.50 | 32.87% | 265.7 |

These are means across seeds 101, 202, and 303. Annealing retained more diversity
than fixed 0.10 and 0.20 but had lower observed win rates than both in every seed.
It was the only arm without a diversity guardrail failure. All four adjusted
win-rate confidence intervals include zero; three seeds do not establish a
win-rate improvement or relative diversity noninferiority.

- [Completed findings, uncertainty, and plot](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/temperature-results.md)
- [Independent full audit](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.protocol-verification.json)
- [All individual results](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.analysis.md)

## Launch record (historical)

Started 2026-09-07. **Full battle comparison is running; no efficacy conclusion yet.**

The five arms are linear 0.30 → 0.10 over 11 generations, fixed 0.10 / 0.20 /
0.30, and temperature adapted to ESS/pool = 0.50 within [0.10, 0.30]. This tests
selection from measured teams, as described in [[Papers/DiffUCO — selection temperature experiment]].

All arms use one shared checkpoint, three retraining/search seeds, identical
training-step and battle budgets, decoding temperature 1.0, and entropy bonus 0.
The original temporary checkpoint was missing, so a new shared checkpoint was
trained for 600 epochs on the current 692 valid corpus teams. The result will
be conditional on that replacement checkpoint and the fixed BC battle policy.

Each of the 15 arm/seed runs has 11 generations of 512 proposals and 128 × 24
search battles. Final evaluation measures both 128 unranked new samples and 16
search-selected finalists, each with 192 fresh battles. Total: 921,600 battles.

Completed checks:

- Eight numerical and campaign-isolation tests passed.
- Zero entropy bonus exactly matched ordinary denoising loss in a real-model check.
- The restored battle runtime completed its four-battle smoke test.
- A complete five-arm plumbing run passed all 140 battle-budget checks.
- Annealed and fixed-0.30 generated identical initial proposals under the same seed.

The historical-pool audit found initial ESS/pool of 0.181 at T=0.10 versus 0.684
at T=0.30; adaptive selected T≈0.204. These are selection diagnostics, not evidence
of improved generator win rate.

Working artifacts:

- [Protocol and restoration details](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/temperature-experiment.md)
- [Resumable full results](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison.json)
- [Historical-pool audit](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_selection_audit.md)
- [Full-run log](/Users/ramiismael/.local/share/vgc-pilot-runtime/temperature-comparison.log)

After all arms finish, verify the persisted budgets with
`scripts/check_temperature_results.py` and generate seed-level analysis/plots
with `scripts/analyze_temperature_comparison.py`. Compare mean fresh generator
win rate jointly with diversity trajectories. Three seeds may leave the result
inconclusive; a positive point estimate or better finalist alone is insufficient.

Execution update: seeds now run concurrently in independent processes and
four-server pools to use the 12-core host. The model, treatments, opponents,
training steps, and battle budgets are unchanged. All additional servers' compiled
simulator files were checked against the original pool. Raw results remain separate;
the collector rejects configuration/input drift before combining them.

- [Combined live results with source provenance](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/temperature_comparison_parallel.json)
- [Concurrent-run log](/Users/ramiismael/.local/share/vgc-pilot-runtime/temperature-parallel.log)
