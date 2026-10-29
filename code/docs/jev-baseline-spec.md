---
title: Jev decision-model baseline on the Mask Lab grids
labels: [ready-for-agent]
created_at: 2026-10-26
updated_at: 2026-10-26
tickets: jev-baseline-tickets.md
ideas_page: https://claude.ai/artifact/KMaYSMi5xQkxyGjNN8DTri
---

## Problem Statement

The project compares team writers and language models on one fill-in-the-blank test, the Mask Lab grids. Each grid hides parts of five starting teams in 270 ways (masks), a model fills every mask, illegal fills are rejected, and each legal team is battled against the top-50 meta teams. The score is a win rate: the share of battles a filled team wins against the top-50 meta teams, with the behaviour-cloning battle policy playing both sides.

Ling, Mercury 2.5, the scratch team writer, the text team writer, the 3B text team writer, the prompt-version writers and a random legal fill all have columns on these grids. A decision model does not. Jev, TypeSafe AI's decision model, never writes text: it reads a state, picks one option from a list it is given, and returns a probability for every option. That makes it a natural fit for filling blanks from legal-option lists. Nobody has measured how it compares with the language models on the same masks, and the project has no code that can call it.

## Solution

Add Jev as one more arm on all three grids (random, held-out, meta). For every mask, Jev answers one Choice question per hidden field, in a fixed fill order, with only legal options on offer. Each answer is written into the team before the next question. Every mask is filled four ways: once with Jev's top option at every question, and three times with each answer sampled from Jev's own probabilities. The legal teams join each grid's candidates, the grid's battle step scores them, and the grid report gains Jev columns beside the other arms.

The owner reads the exact prompt for each task type before any call is sent. A dry run reports the number of calls and the estimated token cost for each grid before any money is spent.

## User Stories

1. As the project owner, I want Jev to fill the same 270 masks per grid that every other arm filled, so that its win rate is comparable with theirs.
2. As the project owner, I want Jev run on the random, held-out and meta grids, so that I can see whether a fill adds strength, generalises, and keeps a top team strong.
3. As the project owner, I want each hidden field asked as its own Choice question, so that each of Jev's picks is from a list it can actually choose from.
4. As the project owner, I want only legal options offered, so that Jev's teams are legal by construction and legality failures point to bugs, not to Jev.
5. As the project owner, I want learnable moves not already on the slot offered for a move blank, so that a move pick can't duplicate a move or be unlearnable.
6. As the project owner, I want items held by no teammate offered for an item blank, so that the Item Clause holds.
7. As the project owner, I want species outside the families already on the team offered for a species blank, so that the Species Clause holds.
8. As the project owner, I want abilities and items limited to the species' legal ability and item pairs, so that no fill fails the simulator's set check.
9. As the project owner, I want a hidden Pokémon filled in the order species, ability, item, four moves, Stat Alignment, Stat Points, so that each later question can be built for the species already chosen.
10. As the project owner, I want each answer written into the team before the next question, so that Jev sees its own earlier picks.
11. As the project owner, I want the species blank answered through a tournament (random groups that fit the option and token limits, one winner per group, then a final), so that all 347 species can be offered despite the 255-option limit.
12. As the project owner, I want the tournament's random split recorded, so that the species tournament experiment can measure whether the split changes the winner.
13. As the project owner, I want Stat Points chosen from a menu of spreads seen for that species, with a few generic spreads for species that have none, so that a numeric field becomes a list Jev can pick from.
14. As the project owner, I want each option described by a JSON object of game facts (types and base stats for a species, effect for an item or ability, type, category, power, accuracy, priority, target and effect for a move, raised and lowered stat for a Stat Alignment, the six values for a spread), so that Jev chooses from facts, not prose we wrote.
15. As the project owner, I want the real option names shown, so that the main run measures Jev at its intended use.
16. As the project owner, I want the state to carry the format, the rules, the top-50 meta teams from the same list Ling's prompt uses, the team so far, and the field being asked about, so that Jev knows who it is building against.
17. As the project owner, I want one instruction template per field type, with slot number and species filled in, so that wording is identical across runs and can be varied deliberately later.
18. As the project owner, I want every mask filled once with Jev's top option and three times sampled from Jev's probabilities, so that I can compare argmax with sampling on win rate and on how different the teams are.
19. As the project owner, I want each sampled run to ask its own questions, so that a sampled early answer shapes the later questions as it would in use.
20. As the project owner, I want to read the exact prompt for one mask of each task type, and approve it on the ideas page, before any call is sent, so that no money is spent on a prompt I haven't seen.
21. As the project owner, I want the token count of every question checked against Jev's 32k-per-question limit before sending, so that no call fails with `max_tokens_exceeded` mid-run.
22. As the project owner, I want a dry run that prints, per grid, the number of masks, runs, questions and estimated input tokens and dollars, so that I approve the cost before spending.
23. As the project owner, I want a limit on the number of calls a run may send, so that a mistake can't spend more than I agreed.
24. As the project owner, I want a smoke-test option that sends only the first few masks, so that I can check a live response before the full run.
25. As the project owner, I want the run to resume where it stopped, skipping finished masks and retrying transport errors, so that an app restart or a rate limit costs nothing.
26. As the project owner, I want every request and response logged, with Jev's choice, confidence and full probabilities, so that the confidence, sampling and shuffle experiments can reuse the main run without new calls.
27. As the project owner, I want the Jev key read from the environment or the Keychain and never printed or saved, so that it doesn't leak into logs or the repository.
28. As the project owner, I want the model name, address and option limit to be settings, so that a later run could point the same code at another decision model by changing settings alone.
29. As the project owner, I want the code to call Jev through TypeSafe's own SDK and no gateway, so that no third-party proxy sits between the project and the API.
30. As the project owner, I want legal Jev teams appended to each grid's candidates with labels that name the run (top pick or sample n) and the mask, after a timestamped backup, so that the other arms' fills are never rebuilt or lost.
31. As the project owner, I want each appended team checked by the Showdown validator, so that the battle step only sees legal teams.
32. As the project owner, I want the grid's existing battle step to score the Jev candidates, so that Jev's win rates come from the same battles, opponents and policy as every other arm.
33. As the project owner, I want the grid report to show Jev's top-pick and sampled columns beside the original team, the random fill and the other arms, per task type, so that I can read the comparison in one table.
34. As the project owner, I want the report to state the number of battles behind each win rate, so that I can judge the noise.
35. As the project owner, I want the report to say that species samples were drawn per tournament round, not from one 347-option distribution, so that nobody misreads the sampled species odds.
36. As the project owner, I want the real prompt printed onto the ideas page once the prompt builder exists, so that the page shows exactly what Jev receives.
37. As a later experiment, I want the main run's logged probabilities, so that confidence against win rate, the shuffle test and Jev as the judge can start without refilling the grids.
38. As a reviewer, I want new code to pass the repository's Ruff rules with complexity at 10 or less, so that it meets the same bar as the rest of the code.

## Implementation Decisions

- **Legal options, one source.** The legal-option rules that the DSPy experiment builds as typed enums move into one function that returns plain option-name lists for one hidden field, given the team so far. The DSPy experiment builds its enums from that function, so both arms use the same rules. The rules come from the pinned simulator's Reg M-B vocabulary, as now.
- **Prompt builder.** A pure function turns one mask, plus the answers so far, into the next request: the state described in story 16 and one Choice question for the next hidden field. A companion lists a mask's questions in fill order. It counts tokens per question and refuses a question over 32k.
- **Option descriptions.** Species and move facts come from the Showdown tables in the repository's data folder. Item and ability effect text comes from the pinned simulator's data, because the repository's item and ability files hold only the mod's few overrides.
- **Spread menu.** Stat Point spreads for a species come from the corpus and the top-50 meta teams; species with none get a small fixed set of generic spreads.
- **Chooser interface.** One small interface separates the fill loop from the decision model: send a state and named Choice questions, get back each question's choice, confidence and probabilities. The Jev adapter implements it on TypeSafe's SDK. A fake implementation drives the tests.
- **Jev adapter.** It reads the key from the environment or the Keychain, and takes the model name (default `jev-latest`), address and option limit as settings. It logs every request and response, and maps the SDK's error types to retry (timeouts, connection errors, server errors, rate limits) or stop (authentication, bad request).
- **Fill loop.** For one mask and one run (top pick, or sample n with its own seed), it asks the questions in fill order, writes each answer into the team, and returns the filled team plus the records. The species blank uses the tournament, and group sizes respect both the option limit and the token budget. Sampling draws from the returned probabilities at temperature 1.
- **Grid runner.** A command-line script with `run` and `collect` subcommands, modelled on Ling's grid runner. It reads a grid's tasks, writes records to a per-grid `jev` log, resumes, has dry-run, limit and maximum-call options, and runs the four runs per mask.
- **Collect.** It backs up the grid's candidates file, appends legal teams with labels `jev:top:<task id>` and `jev:sample<n>:<task id>` and kind `jev`, and validates each team with the pipeline's Showdown check. It never rebuilds the candidates file.
- **Report.** The grid report adds the two Jev columns; the battle step needs no change beyond recognising the new kind if it filters by kind.
- **Meta list.** The top-50 meta teams in the state come from the same configuration Ling's grid runner loads, so both models see the same opponents.

## Testing Decisions

- **One seam: the chooser.** Tests replace Jev with a fake chooser and drive the fill loop and the grid runner through their public entry points. No test calls the network or spends money.
- **What a good test checks.** Only outside behaviour: the filled teams, the records written, the candidates appended, the dry-run counts. No test asserts on internal helpers or on prompt wording beyond what story 16 requires.
- **Legality test.** With a fake chooser that returns uniform probabilities, every mask on all three grids, in all four runs, produces a team the Showdown validator accepts.
- **Order test.** With a fake chooser that records what it was asked, a hidden Pokémon's questions arrive in the decided fill order, and each question's state contains the earlier answers.
- **Tournament test.** A species blank never sends more options than the limit in one question, and the final question offers exactly the group winners.
- **Token test.** A question built over the 32k budget is refused before sending.
- **Resume test.** A run interrupted partway writes no duplicate records when restarted.
- **Collect test.** Every candidate that existed before collect is byte-identical after it, and the backup file exists.
- **Prefactor test.** The DSPy experiment's existing tests pass unchanged after the legal-option move, and the new function has its own tests on a known partial team.
- **Prior art.** The DSPy structure tests (legal-choice types), the meta-starter tests (grid tasks), the LLM completion and OpenRouter generation tests (fake transport), and the Showdown validator tests.

## Out of Scope

- Any other decision model, ensembles, model-agreement studies or gateways such as LiteLLM. One model, Jev, is run.
- Running the thirteen experiments on the ideas page is not part of the main run. It is planned in the tickets after it (owner, 2026-10-26: all thirteen get real charts and descriptions on the page, the four that need no extra fills first: sample against argmax, confidence against win rate, cost per win, Jev as the judge). The main run's logs are kept so that those four can reuse them.
- DSPy. Jev is called through its SDK directly.
- Changing the masks, the starting teams, the meta list, the battle policy or the number of battles.
- Fine-tuning or few-shot prompting Jev. Jev is used as released.

## Further Notes

- Cost: about 3,000 questions per run per grid, four runs per mask, and about 12k tokens of meta teams in every question. My estimate is a few dollars per grid at $0.042 per million input tokens, roughly three times that with the meta teams included. The dry run gives the real number before any spend.
- Token estimate: the first live call (2026-10-27, `jev-1.14.0`) reported 24,644 input tokens for a 62,548-character request, 2.54 characters per token. The prompt builder now estimates at 2.3, a 10% margin; at that rate the species tournament splits into three groups, not two, so a whole hidden Pokémon asks 12 questions.
- Jev's documented limits (checked 2026-10-26): 255 options per Choice question, 32k tokens per question and 64k per request (262k with TypeSafe's long-context header, which it calls slower and less accurate), and no limit on questions per request.
- The ideas page is "Jev Picks a Team". Its "What Jev is sent" section is a draft until the real prompt replaces it.
- The ticket breakdown with blocking edges is in the companion tickets file.
