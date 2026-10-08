# Ling masked-completion baseline

This pilot implements `notes/What is the baseline.md`, with the user's October 8
clarifications: only `inclusionai/ling-3.0-flash-vl`, randomly generated legal
starting teams, and the behaviour-cloning checkpoint playing **both** sides.
It does not use a random-action battle policy.

## Regulation checked before execution

The project README and simulator identify Champions Regulation M-B,
`gen9championsvgc2026regmb`. The official [M-B notice](https://champions-news.pokemon-home.com/en/page/776.html)
dates the regulation to June 17–September 9, 2026. [M-C](https://champions-news.pokemon-home.com/en/page/816.html)
is current on October 8. The historical project baseline deliberately retains M-B.
Git's initial import is October 8 and does not establish the original research start date.

## Frozen pilot

- Three independently generated legal teams; seed 20261008. No starting team is
  a member of the frozen meta pool. Sampling uses the existing hierarchical
  product sampler: 188 species entries, true legal move tables, corpus item
  vocabulary, and random 66-point spreads. This is not uniform sampling over
  every legal team, species form, item, or point allocation.
- Seven tasks, three nested mask sizes per task, one completion per cell: **63
  LLM calls**. Stats/items/abilities/alignments/whole Pokémon use k=1,3,6;
  moves use k=1,12,24; mixed parts use k=1,20,40. There are 48 eligible mixed
  parts, but the requested experiment stops at 40. Stat spreads count as one part.
- An untouched original, one random legal fill, and one Ling completion share
  each starting team and mask. Random controls are sampled by rejection, with
  a finite failure limit; no fallback to the original.
- Nulls denote masked fields. Ling returns explicit JSON path/value patches.
  Unknown, missing, duplicate or unmasked paths are rejected. Shape checks and
  Showdown validation precede any battle. Every unmasked JSON value is preserved.
- OpenRouter, Novita route only, temperature 0.7, reasoning disabled,
  max output 4096 tokens. No model fallback. No repair calls. A separate initial
  smoke directory preserves default-reasoning truncation failures and must not
  be pooled into the instant-mode pilot. Novita's route did not accept JSON
  response-format enforcement, so the prompt requests JSON and the client
  strictly parses and validates the response.
- Raw provider responses, usage/cost, validation failures and request hashes are
  saved. A pending marker blocks automatic reissue after an uncertain request.
  API credentials are read from `OPENROUTER_API_KEY`, never saved in outputs.

## Battle evaluation and cache

The stored `top50` set contains **49 distinct legal opponents**. The frozen panel
uses all 49 plus one deterministic repeated opponent, for 50 battles per candidate.
All candidates receive the same opponent multiset. The existing attribution-aware
worker splits preview-colliding species sets into separate jobs and reorders the
panel round-robin. This is shared across candidates. The objective is the panel's
win fraction; it slightly overweights the repeated opponent, not exactly the
uniform mean of 49 opponents.

`bc_100.zip` is loaded through PPO and passed to both `BatchPolicyPlayer` instances.
The checkpoint's project identity is `57f5edcab415cf6c`; the full SHA-256 is in
`battle-config.json`. An isolated Showdown process uses the same checkout as the
validator. Concurrency is one battle per job. Shared Python/NumPy/Torch seeds do
**not** fix Showdown's simulator RNG, so outcomes are not bitwise reproducible.
This existing limitation remains explicit; the pilot does not claim identical
simulator random draws across methods.

All complete outcomes are stored in `~/vgc-data/matchup_regmb.sqlite` with batch
origin/note `llm-completion:<run>`. The new `llm_completion_panel` cache includes
candidate, opponent schedule, policy checkpoint, simulator/worker/bench file
hashes, seed, concurrency and format. Existing aggregate matchup cells cannot
prove identical conditions and are not reused. Incomplete or unattributed panels
are rejected. Candidate labels resume from exact cached panels.

## Run

From the repository root, using the already installed project runtime:

```sh
/tmp/vgc-pilot/.venv/bin/python code/scripts/llm_baseline.py prepare --output code/results/llm-baseline-instant-pilot
# Configure OPENROUTER_API_KEY in the process environment, then:
/tmp/vgc-pilot/.venv/bin/python code/scripts/llm_baseline.py complete --output code/results/llm-baseline-instant-pilot
/tmp/vgc-pilot/.venv/bin/python code/scripts/llm_baseline.py battle --output code/results/llm-baseline-instant-pilot
/tmp/vgc-pilot/.venv/bin/python code/scripts/llm_baseline.py report --output code/results/llm-baseline-instant-pilot
```

Prepare refuses to overwrite a manifest. Completion and battle have separate
locks, so controls can be battled during generation; rerun battle after completion
to score later arrivals. Local runtime paths are inherited from this project;
this is not yet a portable installation. There is no external evaluation-library
dependency: the harness uses the existing simulator, validator and SQLite library.

## Interpretation

Report legality over **all** attempts, and paired win-rate changes versus both
controls for legal, completely scored outputs. Do not treat missing or illegal
outputs as observed battle losses. Task/k cells have at most three independent
starting teams: this pilot tests the experiment harness, not statistical efficacy.
The starting-team bootstrap interval is exploratory and unstable with three clusters.
The meta pool is visible in the prompt; this is in-pool optimization, not a held-out
generalization test. No individual team is recommended as a result without a
separate fresh 192-battle confirmation.

The full original six-task grid has **54** cells; including mixed k=1..40 makes
**94** cells. A later 50-team × 3-completion full seven-task run would therefore
need 14,100 LLM calls, not the note's six-task count of 8,100. Do not launch that
larger run automatically based on this pilot.
