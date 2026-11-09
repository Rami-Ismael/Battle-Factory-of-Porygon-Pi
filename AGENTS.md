# Reader-facing wording

Apply these rules to chat responses, documentation, tables, chart labels, and interface text:

1. Name each model, experiment, group, and method by its purpose or distinguishing property. Derive the name from its definition in the project source or documentation; when the meaning is unclear, inspect the definition before naming it.
2. Present a descriptive name first. When an experiment identifier such as F1 or G2 is needed for traceability, append it in parentheses on first mention and use the descriptive name thereafter. Give each component of an ensemble its own meaningful name.
3. Preserve exact identifiers in code, configuration, filenames, and data. Translate them into descriptive names at the reader-facing boundary. Expand specialized abbreviations on first use when the intended reader needs the explanation.
4. When naming an experiment, arm or run, or writing a list of them (results table, figure legend, checkbox list, pre-registration arm list), follow `.claude/skills/name-experiments/SKILL.md`.
5. Put each choice to the user by its effect, in project terms: "Which model does the plan test: the base masked Qwen, or the fine-tune once it exists?". A ticket or reply that only writes code names the ticket or step that runs it.
6. Before completing the response or edit, review every heading, label, table, and explanation you authored. Wording is complete when every project-specific shorthand has a source-backed meaning introduced before or alongside its use, and each name communicates that meaning without a separate legend.

# GitHub issue bodies

An issue's body states the current plan; its comments are the history of how the plan got there. When the user changes a decision, update every open issue and Wayfinder map whose body still states the old plan: rewrite the affected wording in the body, then post a dated comment naming what changed, the user's words and why. The update is done when no open issue body describes a dropped plan. Before building from an issue, read its comments too, and treat a comment that overrides the body as a stale body to fix first.
