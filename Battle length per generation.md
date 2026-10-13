# Battle length per generation

Put battle-length measurement in the **shared battle evaluator**, then let Cross Entropy, MAP-Elites, and DiBO attach their method, run, and generation identifiers. All three methods should use the same aggregation code.

This is a proposed integration design, not a change to the search algorithms. Inspection of the local Python code found an existing collection point in `battle_speed_lab/worker.py:151`: it already stores `turns: b.turn` for each battle. At line 166 it also sums turns across the workload. The speed lab is an isolated benchmark; use this pattern in the evaluator actually called by the search methods. The three search loops were not located in the inspected local Python files.

## 1. Keep individual results

Keep one result per battle, not just a running average. This lets you recompute summaries, examine unusually long games, and separate wins from losses later.

```text
Search method proposes candidates
    → shared evaluator plays scheduled matchups
    → persist each battle result with its evaluation context
    → shared summarizer groups results by run / method / generation / phase
    → plot generation summaries
```

Suggested record:

```python
{
    "evaluation_id": "cem-run03-gen07-candidate12-opponent04-repeat02",
    "battle_id": "simulator-battle-id",
    "run_id": "run03",
    "method": "cem",           # "cem", "map_elites", or "dibo"
    "generation": 7,
    "phase": "search",         # separate from "checkpoint_eval"
    "candidate_id": "candidate12",
    "opponent_id": "opponent04",
    "replicate": 2,
    "status": "completed",     # "turn_cap", "timeout", "error"
    "outcome": "win",          # "loss", "draw", or None if unfinished
    "turns": 13,               # final simulator turn index
    "cache_hit": False,
}
```

Record simulator version, format, policy/checkpoint, opponent-panel identity, turn cap, and randomness settings in a run manifest. Store the actual battle seed when the evaluator supports it; a search RNG seed alone does not establish paired battle randomness.

Capture the final simulator turn counter when the battle finishes. Do not count move-selection calls: doubles actions, switches, and forced replacements need not correspond to new turns. Record once from the candidate's perspective, rather than once for each player. Persist results as they arrive so a later batch failure does not erase successful battles.

Pass context with each scheduled evaluation, before starting concurrent battles. Do not read a mutable global `current_generation` when a battle finishes, or assign candidate identities by completion order.

## 2. Use one summarizer for all methods

For one run, method, generation, and phase:

**Mean turns = sum of final turn counts / number of completed battles with known turn counts.**

For example, 10, 15, and 20 turns give an average of 15 turns. Combine batch totals and counts; do not take an unweighted average of batch means when batch sizes differ.

The following is illustrative Python for records that have already been validated and deduplicated by `evaluation_id`:

```python
from collections import defaultdict
from statistics import mean, median


def summarize_generations(records):
    groups = defaultdict(list)
    for r in records:
        key = (r["run_id"], r["method"], r["generation"], r["phase"])
        groups[key].append(r)

    summaries = []
    for (run, method, generation, phase), rows in sorted(groups.items()):
        completed = [r for r in rows if r["status"] == "completed"]
        measured = [r for r in completed if r["turns"] is not None]
        turns = [r["turns"] for r in measured]
        wins = [r["turns"] for r in measured if r["outcome"] == "win"]
        losses = [r["turns"] for r in measured if r["outcome"] == "loss"]
        summaries.append({
            "run_id": run, "method": method,
            "generation": generation, "phase": phase,
            "n_evaluations": len(rows),
            "n_completed": len(completed),
            "n_measured": len(measured),
            "n_missing_turns": len(completed) - len(measured),
            "n_turn_cap": sum(r["status"] == "turn_cap" for r in rows),
            "n_timeout": sum(r["status"] == "timeout" for r in rows),
            "n_error": sum(r["status"] == "error" for r in rows),
            "mean_turns": mean(turns) if turns else None,
            "median_turns": median(turns) if turns else None,
            "mean_turns_win": mean(wins) if wins else None,
            "mean_turns_loss": mean(losses) if losses else None,
            "win_rate": (
                sum(r["outcome"] == "win" for r in completed)
                / len(completed) if completed else None
            ),  # draws count as non-wins
        })
    return summaries
```

Use an explicit termination reason to distinguish natural draws from turn-cap truncations. Keep partial turn counts for interrupted battles, but exclude them from this completed-battle mean. Report cap, timeout, error, and missing-count totals beside the mean: excluding many long unfinished battles can make the surviving games look artificially short. An empty generation has a missing mean, not zero turns.

## 3. Define what each generation contains

| Method | Suggested generation boundary | Search-phase battle-length sample |
|---|---|---|
| Cross Entropy | One candidate sampling and distribution-update iteration | Actual evaluated candidates from that iteration, before elite filtering |
| MAP-Elites | One explicitly numbered batch of emitted candidates | Actual evaluated offspring, whether or not they enter the archive |
| DiBO | One outer proposal and true-evaluation round | Candidates actually battled in that round |

These are proposed conventions; map them to the actual loop boundaries when integrating. A diffusion denoising step or surrogate-training epoch is not a battle-evaluation generation. Surrogate predictions do not produce observed turn counts. Do not include untested proposals in the denominator or resummarize the whole accumulated MAP-Elites archive as though it were newly evaluated.

Keep two distinct views if you want to answer both questions:

- **Search-phase mean:** How long are the games played for the candidates each method explores?
- **Checkpoint-evaluation mean:** How long are the games played by each method's selected team after a given amount of search?

For the second view, use the same predeclared team-selection rule, opponent panel, policy, and repeat count for every method. Select the team using search data, then evaluate it separately; do not choose the winner using the held-out evaluation results. Keep these records under `phase="checkpoint_eval"`.

## 4. Compare methods fairly

Plot mean turns against generation for understanding each method's trajectory, and also against **cumulative actual battle evaluations** for comparing methods with different batch sizes or surrogate usage. Count all search battles, including initial-data and reevaluation battles; report checkpoint-evaluation cost separately. Record elapsed search time as another axis if runtime matters: turn count is not wall-clock duration.

Use the same opponent panel, policy, battle format, turn cap, and evaluation schedule across methods. If adaptive search allocates extra games to some teams, the pooled mean describes the battles actually evaluated, not an equal-weight average across teams. Use a balanced checkpoint panel for the latter comparison.

Repeat each method with independent search runs. Summarize within each run first, then show the mean and variation across runs at matching budgets. Do not treat all battles from one search run as independent replicates of the search method.

Show win rate alongside mean turns, with separate turns-in-wins and turns-in-losses. **Shorter battles are not automatically better: the team may simply lose faster.** Initially keep battle length as a reported metric; adding a length penalty to the fitness function would change the experiment.

## 5. Cache and resume rules

Store physical battle results separately from the context in which a generation uses them. An evaluation record can reference a cached battle result; it does not imply a new simulator execution. Count a scheduled evaluation once even if a task retries or resumes. Mark cache hits so they do not increase the actual battle-cost budget or appear to be independent new evidence.

For the simplest comparison, give each candidate/opponent/repeat slot one evaluation record. Include a cached result once when fulfilling that slot, and report cache usage. If an old result has no turn count, keep it missing rather than converting it to zero. Regenerate the summary from persisted records after a resume instead of incrementing a saved average again.

## Suggested code boundaries

```text
battle_evaluator.py   # shared battle execution and final result capture
metrics.py            # shared grouping and summary functions
search/cem.py         # supplies method, run, generation, candidate context
search/map_elites.py  # supplies the same context for each emission batch
search/dibo.py        # supplies the same context for each true-evaluation round
results/<run_id>/
    manifest.json
    evaluations.jsonl # or an existing SQLite table with a unique evaluation ID
    generations.csv   # derived output; can always be rebuilt
```

These are responsibility boundaries, not a requirement to rename existing files. Before connecting all three methods, check a small known fixture: unequal batch sizes produce a weighted pooled mean, failed games are excluded and counted, missing turns stay missing, retries do not duplicate evaluations, and overlapping generations retain their original context.

## Run the aggregation tests

From the repository root, run:

```sh
python3 -m unittest discover -s tests -v
```

`tests/test_battle_length_example.py` loads the actual `summarize_generations` example above and checks known averages, unequal batch sizes, win/loss/draw denominators, incomplete battles, missing and zero turn counts, empty inputs, grouping across all three methods, completion order, and cached evaluation slots.

These unit tests check aggregation from recorded battle results. Validation and evaluation-slot retry deduplication remain preconditions of the example, not implemented behavior.

## Run real-battle integration tests

The existing bridge at `notes/team_completion_pipeline/showdown.cjs` now returns `turns` from Showdown's end-of-battle result. The bridge rejects completed results without a valid nonnegative integer counter. Its existing runtime fingerprint includes the bridge source, so newly fingerprinted runs do not reuse older bridge results lacking this field.

With Node.js on PATH and a built Showdown runtime supporting `gen9championsvgc2026regmb`, run from the repository root:

```sh
VGC_SHOWDOWN_RUNTIME=/absolute/path/to/built/showdown python3 -m unittest discover -s tests -v
```

Set the path to the runtime root containing `dist/sim/index.js` and installed dependencies. The local runtime configured in `notes/team_completion_pipeline/smoke.json` is one suitable option on the development machine; a standard upstream runtime may not contain this Champions format. No server, network connection, or paid model calls are needed.

`tests/test_battle_length_integration.py` plays two real seeded battles using the committed six-Pokémon fixture and Showdown's random policies. It independently parses the battle log's turn events and checks that the recorded count equals the final turn. It then closes and reopens a temporary SQLite cache, forbids another simulator call, and verifies the stored result and row count. Finally, it feeds those recorded results into the documented summarizer and checks the average. A separate real-battle test verifies that exceeding the turn cap fails without caching a completed result.

Without `VGC_SHOWDOWN_RUNTIME`, these two integration tests are explicitly skipped; the 12 unit tests still run. If the variable is supplied but the runtime is missing or incompatible, the integration tests fail rather than silently skip. Each simulator call has a 30-second timeout. All test databases are temporary.

Local validation produced battles of 9 and 11 turns, with a mean of 10; both counts matched their logs and persisted cache records. The tests derive expected counts from the actual logs rather than hardcoding those lengths. This verifies the team-completion bridge and result cache, not the separate battle-speed worker or integration into the three search loops. Evaluation-slot deduplication across search generations still requires coverage when that orchestration is implemented.
