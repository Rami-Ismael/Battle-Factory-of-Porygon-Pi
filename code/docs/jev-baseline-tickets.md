# Jev decision-model baseline: ticket draft (2026-10-26)

Input for `/to-tickets`. Each ticket is one tracer-bullet slice with a check you can see. Blocking edges are in the table.

## Decisions already taken

- One decision model, Jev (`jev-latest`), called through the project's own adapter on `typesafe_sdk` (MIT, tag `v0.7.5`). No LiteLLM.
- Grids: random (`runs/prompt-ab`), held-out (`runs/heldout-starters`), meta (`runs/meta-starters`), all under the vault's `team_completion_pipeline/`. 270 masks each.
- Fill order inside a slot: species, ability, item, moves, Stat Alignment, Stat Points. Each answer joins the state before the next question.
- Species blank: tournament. Split the legal species at random into groups of at most 255, ask each group, then a final between the group winners.
- Per mask: the top pick, plus 3 runs sampled from Jev's own probabilities (temperature 1).
- Options: the real name, with a JSON object of game facts as the description.
- Stat Points: a menu of spreads seen for that species (corpus and meta teams), with a few generic spreads for species that have none.
- The owner reads the prompt before any Jev call.
- Jev's state includes the top-50 meta teams, taken from the same list Ling's prompt uses (`benchmark.ling.json` `meta_teams`, loaded with `pipeline.load_teams`). That is about 12k tokens per question (estimate), leaving about 20k of the 32k limit for options, so ticket 1 sizes species tournament groups by tokens as well as the 255-option limit.

## Facts found while mapping the code

- A mask is a row of `eval2.tasks.json`: `task` (`pokemon`, `moves`, `items`, `abilities`, `natures`, `stats`), `k` (how many fields are hidden, 1 to 6, or 1 to 24 for moves), `masked_team` (hidden fields are `null`; a hidden Pokémon is a `null` slot), `masked_paths` (for example `team[0].moves[2]`), and `original`. A `pokemon` mask with k = 6 has Jev build the whole team.
- The arm to copy is Ling's grid runner, `scripts/ling_grid.py` (snapshot in `~/.local/share/vgc-pilot-runtime/textdiff/grids_ling/code/`). It appends answers to `<grid>/<arm>.jsonl`, resumes, and `collect` appends legal teams to `<grid>/candidates.json` after a timestamped backup. Never call `mask_lab_prepare.main` on a grid: it rebuilds `candidates.json` and drops the other arms.
- Legal options already exist as pydantic enums in `scripts/dspy_structure.py` `legal_completion_type`: learnable moves not already on the slot, ability and item pairs, items no teammate holds (Item Clause), and species outside families already on the team (Species Clause). The data comes from `data/regmb-vocabulary.json` through `scripts/mb_prompt.py` `regulation()`.
- Descriptions: `data/pokedex.json` (types, base stats, abilities) and `data/moves.json` (type, category, power, accuracy, priority, target, shortDesc) are full Showdown tables. `data/items.json` and `data/abilities.json` hold only the mod's 8 and 7 overrides, so item and ability text must come from the pinned simulator's data (`~/.local/share/vgc-pilot-runtime/validator-913da36`).
- Jev limits (docs, 2026-10-26): 255 options per Choice; 32k tokens per question (state plus that question) and 64k per request; no limit on questions per request; Score has at most 10 levels. Over the limit returns 400 `max_tokens_exceeded`.
- Multi-field masks need separate runs for each sample, because a later question's state holds the earlier answers. That's 4 runs per mask (top pick plus 3 samples).

## Tickets

| # | Ticket | Check | Blocked by |
|---|---|---|---|
| 0 | **Prefactor: plain legal-option lists.** Pull the option logic out of `legal_completion_type` into a function that returns plain option-name lists for one path, given the masked team so far. `dspy_structure.py` builds its enums from it. | `scripts/test_dspy_structure.py` passes unchanged, and the new function has its own tests. | none |
| 1 | **Prompt builder.** Turn one mask into Jev's ordered questions: `state` as structured data (format plus the masked team), one `Choice` per hidden field, legal options, and a JSON description per option. No network. | The printed prompt for one mask of each task type, approved by the owner. A token estimate shows the largest question under 32k. | 0 |
| 1b | **Prompt on the page.** Replace the draft "What Jev is sent" section of the Jev Picks a Team page (https://claude.ai/artifact/KMaYSMi5xQkxyGjNN8DTri, source `notes/Blog/visuals/jev-picks-a-team/jev-picks-a-team.html`) with the real printed prompt from ticket 1: one full request per task type, with its token count. | The page shows the exact text Jev would receive, and the owner approves it there. | 1 |
| 2 | **Jev adapter.** A thin wrapper on `TypeSafeClient`: address, key (Keychain item `typesafe` or `TYPESAFE_API_KEY`), model and option limit as settings. Logs every request and response, and maps the SDK's error types to retry or fail. | One live call on one mask, after the prompt is approved. | 1b |
| 3 | **Fill loop and tournament.** Ask the questions in fill order, write each answer into the state, run the species tournament, and produce the top-pick run and 3 sampled runs per mask. | With a fake model that returns uniform probabilities, every filled team on all three grids passes Showdown validation. | 1 |
| 4 | **Grid runner.** `scripts/jev_grid.py run --grid …`: resumable `jev.jsonl`, `--dry-run` that counts calls and estimates tokens and cost, `--limit`, and `--max-calls`. | The dry run prints counts for the three grids. A `--limit 5` run on one grid writes 5 records. | 2, 3 |
| 5 | **Collect.** Append legal teams as `jev:top:<task>` and `jev:sample<n>:<task>` candidates after a timestamped backup, validated as Ling's are. | The candidate count rises by the legal Jev teams, and every other arm's candidates are byte-identical. | 4 |
| 6 | **Battles and report.** Run each grid's battle step on the new candidates and add Jev (top pick, sampled) to the grid report next to the other arms. | A per-task table of legality and win rate for Jev against the original team, the random fill and the other arms. | 5 |

Tickets 2 and 3 can run in parallel once ticket 1 lands. New code keeps Ruff's current rules clean, with C901 at 10 or less.
