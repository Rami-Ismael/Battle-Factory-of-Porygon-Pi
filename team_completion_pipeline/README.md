# Runnable team completion pipeline

This implements the written workflow: select starting teams, generate shared random masks, complete them with a model, validate the results, reuse matching battle results from SQLite, and run missing battles. No paid API calls are needed for the smoke test.

## Run now

Requires Python 3.10+ and Node.js. The supplied configuration points to the already built local Showdown runtime at `/Users/ramiismael/.local/share/vgc-pilot-runtime/servers/8134`. The bridge loads its compiled modules directly; it does not start or contact a Showdown server. To move this project, update `showdown_runtime` to a compatible built checkout containing `dist/sim` and its installed dependencies.

From this directory:

```sh
python3 -m unittest -v
python3 pipeline.py run --config smoke.json
```

The smoke test runs eight examples: minimum and maximum masks for items, stat spreads, moves, and whole Pokémon. It uses a random donor baseline and real local Showdown battles under the built-in random policy. Both teams use the same example fixture in smoke mode. These results test the software and are **not evidence of counter-team quality**.

Rerunning the same command reuses saved candidates and matching battle results. Changing the pipeline implementation, generation settings, proposal pool, or task input changes candidate identities. Battle cache identities include full ordered teams, format, runtime fingerprint, policy settings, and individual seed. Changing one seed only requires that seed's missing battles.

## Files produced

Under `runs/smoke/`:

- `manifest.json`: configuration, selected teams, rejected inputs, simulator fingerprint, masks, and original teams for auditing. Hidden originals are never included in model requests.
- `candidates/`: each completion attempt, response, validation error, generation time, and OpenRouter response metadata when available.
- `matchups.sqlite3`: battle requests and outcomes, including complete battle logs.
- `results.json`: results per masked example and method, opponent win rates, draws, original-team comparison, cache usage, and battle errors.
- `summary.json`: results grouped by method, task, and k; validity and evaluation coverage are reported alongside win rates.

The first version uses equal opponent weights. Draws are counted as non-wins. It does not compute confidence intervals or perform candidate selection from multiple valid proposals. `attempts` is a bounded retry count for invalid completions, not a best-of-N search. Transport failures stop that candidate's attempts and remain recorded; they are not automatically retried, because the request may already have incurred cost. Keep or archive the run directory before intentionally restarting failed generations.

## Input datasets

The pipeline accepts the existing `team-example.json` fixture or a dataset in this shape:

```json
{
  "teams": [
    {
      "id": "tournament-player",
      "format": "gen9championsvgc2026regmb",
      "tournament": "Tournament name",
      "date": "2026-09-01",
      "placement": 1,
      "team": []
    }
  ]
}
```

Replace the empty example `team` array with six complete Pokémon objects, following the existing fixture. The core fields are `species`, `item`, `ability`, `nature`, `moves` (four strings), and `statPoints` (a spread object with keys `HP`, `Atk`, `Def`, `SpA`, `SpD`, `Spe`). Optional fields are `slot`, `ivs`, `level`, `gender`, `shiny`, and `happiness`. An empty item string represents no held item; `null` represents a masked item only inside tasks. Nullable IV fields outside the mask must remain unchanged. Tera Types are excluded for this regulation: legacy null `teraType` fields are removed when loading datasets, and generated Tera fields are rejected.

`statPoints` means Champions Stat Points for the supplied configuration. The Showdown bridge maps this field to the simulator's `evs` representation, which this format uses for Stat Points. For a conventional EV format, change both `format` and `spread_system`, and supply conventional EV values in this same spread field. Do not mix the two allocation systems.

Import local Showdown pastes with:

```sh
python3 pipeline.py import-pastes --config smoke.json --directory /absolute/path/to/team-pastes --output data/imported.json
```

This reads `.txt` files, imports supported fields, validates the teams, and records rejected files. It does not retrieve VGCPaste or invent tournament metadata. Add verified `tournament`, ISO `date`, and integer `placement` metadata if using `selection: "top50"`. That selection filters illegal/incomplete starting teams, orders tournaments newest first, and selects one eligible team from each tournament per round until reaching 50 or exhausting the data. `selection: "sample"` samples the supplied legal teams; it does not synthesize random legal starting teams.

## Configure the full experiment

Copy `benchmark.example.json` to your own configuration and prepare three datasets:

1. Starting teams whose parts will be masked.
2. Meta opponents used for the final evaluation.
3. A proposal pool for the random donor baseline, if using it.

Benchmark mode rejects exact team overlap between meta opponents and starting/proposal teams. This check cannot verify whether a model has seen a team during training; document training and tuning splits separately. The models receive the evaluation meta sets because countering those supplied teams is the task. Do not use final evaluation results to tune model settings.

Tasks use `items`, `stats`, `moves`, or `pokemon`. Omit `k` to run the full range (1–6, or 1–24 for moves); provide an array such as `[1, 3, 6]` for a subset. Optional tasks are `abilities` and `natures`. Tera tasks are not supported. Use the same `mask_repetitions` and `attempts` for every method.

### OpenRouter, when funded

Replace `REPLACE_WITH_MODEL_ID` with a chosen model ID supporting JSON object output. Set `OPENROUTER_API_KEY` in your environment, then run:

```sh
python3 pipeline.py run --config benchmark.json
```

The key is read from the environment and is not written to results. Requests use OpenRouter's [documented chat completions endpoint](https://openrouter.ai/docs/api_reference/overview), with the masked team, regulation, missing paths, and meta teams in the prompt. Responses must pass structural checks, preservation checks, and the actual Showdown team validator before evaluation. Model usage/cost metadata is retained when the API supplies it. The number of calls is bounded by selected teams × task/k combinations × mask repetitions × methods × attempts; review this before running the full experiment. The default example includes only the LLM until a diffusion adapter is connected.

### Diffusion or another local method

Add a method like:

```json
{"name": "diffusion-v1", "type": "command", "command": ["python3", "/absolute/path/to/your_diffusion_adapter.py"]}
```

The command reads one JSON request from standard input and prints one JSON response to standard output. It receives `format`, `spread_system`, `task` (including `seed`, `k`, slots, and masked paths), `masked_team`, `meta_teams`, and retry `feedback`. Return `{"completed_team": [...], "reasoning": []}`. Send diagnostic logs to standard error. Command arrays execute without a shell; use absolute paths. This interface is implemented, but no trained diffusion model is bundled.

### Stronger battle policies

The bundled battle adapter uses Showdown's `RandomPlayerAI` with seeded policy and simulator randomness. It runs actual battles, but random play is only an initial baseline. Meaningful conclusions about competitive counter-teams require a suitable fixed battle policy and enough battles.

To connect your battle policy, use:

```json
{
  "type": "command",
  "command": ["python3", "/absolute/path/to/your_battle_adapter.py"],
  "version": "immutable-policy-and-simulator-version",
  "seeds": [101, 202, 303]
}
```

The adapter reads a JSON request containing both complete teams, the seed, format, simulator fingerprint, and battle configuration. It returns `{"outcome": "win"}`, `"loss"`, or `"draw"` from the candidate's perspective, optionally with a log/replay reference. Its version must change whenever policy weights, code, simulator, or relevant settings change. Simulation errors and timeouts must produce a nonzero exit or an error outcome; they are never cached as draws. Keep candidate/opponent side assignment fixed across methods; side-swapped evaluation is not implemented yet.

## Remaining research setup

The runnable infrastructure is complete for the supplied smoke experiment. A full comparison still needs your real team datasets and tournament metadata, a chosen OpenRouter model, a trained diffusion adapter, and the competitive battle policy. Mixed-part masking, automatic team synthesis, uncertainty estimates, weighted meta scoring, and cost aggregation are not implemented in this first version.
