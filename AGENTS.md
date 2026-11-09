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

# Charts

ECharts is the default chart library (`echarts@6.1.0` from cdn.jsdelivr.net), owner decision 2026-11-09. A Python script builds the ECharts option as a dict and writes the page with `write_chart` from `code/src/echarts_chart.py`; the page follows light and dark and redraws on resize. Give each chart the interactions ECharts has built in: `emphasis: {focus: "series"}` on lines, an axis tooltip, and `dataZoom` when the x axis is long. matplotlib and the libraries drawn through it (seaborn, plotnine, mpld3) are out for every figure because their output looks poor; ruff's banned-api rule (TID251) flags new imports. Every chart script in `code/` was ported on 2026-11-09; `scripts/test_chart_ports.py` holds each figure's words and data points as a golden, so a chart edit that loses either fails it. Run it with `/tmp/vgc-pilot/.venv/bin/python` before reporting a chart change; it reads this laptop's results. One exception stays on matplotlib by owner decision (2026-11-09): `notes/Blog/visuals/team-search-space/render.py`, a paper diagram exported as PDF, SVG and PNG.
