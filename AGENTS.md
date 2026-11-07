# Reader-facing wording

Apply these rules to chat responses, documentation, tables, chart labels, and interface text:

1. Name each model, experiment, group, and method by its purpose or distinguishing property. Derive the name from its definition in the project source or documentation; when the meaning is unclear, inspect the definition before naming it.
2. Present a descriptive name first. When an experiment identifier such as F1 or G2 is needed for traceability, append it in parentheses on first mention and use the descriptive name thereafter. Give each component of an ensemble its own meaningful name.
3. Preserve exact identifiers in code, configuration, filenames, and data. Translate them into descriptive names at the reader-facing boundary. Expand specialized abbreviations on first use when the intended reader needs the explanation.
4. When naming an experiment, arm or run, or writing a list of them (results table, figure legend, checkbox list, pre-registration arm list), follow `.claude/skills/name-experiments/SKILL.md`.
5. Before completing the response or edit, review every heading, label, table, and explanation you authored. Wording is complete when every project-specific shorthand has a source-backed meaning introduced before or alongside its use, and each name communicates that meaning without a separate legend.

# Settled decisions

Before proposing a model family, baseline or experiment scope, or asking the owner about one, read `code/docs/decisions/README.md` and the matching decision file. Ask only when the task would change that decision, the evidence conflicts, or the file leaves the point open; then cite the file and the open point.

- No block diffusion models: `code/docs/decisions/no-block-diffusion-models.md`.
