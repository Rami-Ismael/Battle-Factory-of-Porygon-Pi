# Project language

## Authoring boundary

Rami writes his own blog. Help him clarify his thoughts through explanations, questions, and optional wording suggestions in chat. Keep blog prose, figure captions, and reader-facing labels untouched unless he explicitly asks to apply a specific proposed edit. Requests to help articulate an idea are requests for discussion.

## Vocabulary

Before explaining diffusion models, comparing experiments, or writing model labels in prose, tables, or figures, read [CONTEXT.md](CONTEXT.md) and use its canonical names.

- Lead with the descriptive name. When traceability matters, give the original experiment or checkpoint identifier in parentheses on first mention; use the descriptive name afterward.
- Keep executable identifiers, checkpoint paths, configuration values, commands, and quoted historical results exact. Add a descriptive explanation beside them when needed.
- For retrained descendants, append the explicit version to the canonical name, such as “matchup-conditioned diffusion model, M-B v3.” Check the descendant's training report or configuration before attributing its training data or initialization to the original model.
- Resolve short codes from their local context: model identifiers, search-loop generation numbers, guidance strengths, and metric names can reuse similar text. Apply the glossary only to the verified model identity.
- Describe training-data provenance separately from model architecture and search algorithms. For teams collected from earlier search experiments, name the experiments when relevant; reserve “cross-entropy method” for verified CEM runs and explain model feedback or recurrence only when the implementation uses it.
- For an unfamiliar model, verify its training data and conditioning in code or a training report, then extend the glossary with a descriptive name. Identify unresolved meanings explicitly instead of guessing from a code.
- Before finishing prose or label edits, check every model reference for a descriptive name and preserve version distinctions. Claims about matrix coverage must account for held-out teams and untested pairings.
