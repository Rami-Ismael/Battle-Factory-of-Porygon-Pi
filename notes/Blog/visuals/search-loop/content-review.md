# Search-loop content review

Reviewed 2026-09-13 against local project notes. This is a content handoff, not an audit of the implementation or underlying results.

## Scope and labels

Use **“Proposed workflow informed by pilot experiments”** for the combined figure. The pilot notes report implemented generation, legality handling, battle evaluation, retraining, surrogate ranking, and diversity measurement, but the complete diagram combines those experiments with proposed comparisons and independent held-out evaluation.

The objective is to find teams that counter a target metagame while remaining competitive against other teams. Fresh battles reduce reuse of noisy scores; they do not by themselves establish generalization to unseen opponents. Label held-out-opponent evaluation as proposed unless an additional verified result establishes it.

Use **“Select training teams”** for the post-battle selection node. Hard top-fraction elite selection and Boltzmann score-weighted sampling are alternatives. The completed temperature campaign uses the latter, so the diagram should not imply that every implemented loop retrains only on hard elites. Selecting candidates for battles is a different decision from selecting training examples after battles.

## Control explanations

| Control | Where it acts | Reader-facing explanation | Status in reviewed notes |
|---|---|---|---|
| Initial training-data mixture | Initial training | Change the proportion of known real teams, legal mutations, and hierarchical-product samples used to train the initial generator. | Source-mixture sweep proposed. |
| Guidance strength | Candidate generation | Amplify the model’s requested win-rate condition while generating a team. Battle evaluation checks whether the generated team actually performs well. | Guidance studied; coefficients in the axis table are proposed comparisons. |
| Generation temperature | Candidate generation | Lower temperature favors higher-scoring field values; higher temperature spreads the sampling probabilities more evenly. | Proposed decoding sweep; separate from selection temperature. |
| Elite fraction | Training-team selection | Retain a specified top fraction of evaluated teams according to measured win rate. Smaller fractions apply stronger selection. | Proposed hard-selection sweep; not the rule used in every campaign. |
| Selection temperature, if shown | Training-team selection | Sample evaluated teams with weights proportional to exp(measured win rate / T). Lower temperature concentrates training on higher measured scores. | Completed comparative campaign; uncertainty remains. |
| Original-data replay, if shown | Generator update | Mix original training examples into retraining alongside the newly selected teams. | Proposed response to diversity loss. |

Avoid claiming that any control guarantees legality, strength, or diversity. Changes to controls in the visual should reveal explanations, not fabricate response curves or simulated outcomes.

## Evaluation boundaries

- Archive observations with matchup identity and battle count; measured win rates are noisy and conditional on battle policies and opponent distribution.
- Fix regulation, legality handling, policies, opponent weights, and battle budgets within a controlled comparison.
- Show the optional surrogate as a predictor used before expensive battles, updated from stored observations.
- Keep final held-out results out of the search feedback path.
- Distinguish exact team uniqueness, roster composition diversity, and novelty relative to a frozen reference corpus. None alone establishes strategic novelty.

## Evidence pointers

Paths below are relative to the project root; sections supersede older historical statements in the same notes.

- `Blog.md`: current article outline, motivation, evaluation, failure, reproduction and conclusion requests.
- `Diffusion as candidate proposer in black-box optimization over structured inputs.md`, “Proposed experiment axes and measurements — 2026-09-07”: source mixtures, guidance, generation temperature, hard elite fraction, replay, common metrics and controls. “Measured 2026-09-01” records surrogate ranking as a pilot result.
- `New Todo Section with No Long Verbose Opus text geneation.md`, completed Boltzmann item and adaptation: score-weighted sampling, fixed decoding temperature, fresh finalist battles, pending held-out tests.
- `Boltzmann selection temperature experiment.md`, “Completed 2026-09-08”: completed 15-run campaign and conditional conclusions; historical launch record specifies fixed BC policy and separate unranked-generator versus selected-finalist evaluation.
- `Metrics for Diversity.md`, “Baseline complete — 2026-09-08”: pre-fine-tuning baseline now exists; exact uniqueness remains high while roster compositions concentrate. Earlier “baseline outstanding” text is historical.
- `Matchup Matrix.md`: measured win rate under fixed policies and explicit unknown matchups.

No existing vault notes were edited.
