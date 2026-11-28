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

# Text diffusion training logs

Every future training run of a text diffusion model on Pokémon teams must log the percentage of rule-breaking samples over training alongside training and validation loss, owner decision 2026-11-27. At each validation interval (or epoch), generate a fixed-seed sample batch from the current checkpoint and record the overall illegal share plus the rule-by-rule violation percentages (following `RULES` and `rule_breaks` in `code/scripts/inplace_prompt_grid.py`, paralleling the 84-field discrete diffusion model's per-epoch `rule_shares` in `code/src/epoch_legality.py`). Write the curves into the run's results JSON as the authoritative record, mirror them to Weights & Biases (`rule_shares/<rule>` via `code/src/wandb_tracking.py`), and show the rule-breaking log over training in every training report.

# Running external models

Before running inference, prompting, or evaluation on any pretrained or fine-tuned model checkpoint, inspect its primary sources and tokenizer configuration first and record what you found in the pre-registration doc:

1. **Paper and model card.** Read the checkpoint's paper, Hugging Face model card, and official inference/evaluation code to verify how its authors prompt and sample it (temperature, top-p/top-k, block length, step count, remasking/fill order, logit shift, BOS/EOS tokens).
2. **Chat template and thinking mode.** Inspect `tokenizer.chat_template` and `tokenizer_config.json`. If the checkpoint is chat- or instruction-tuned, use its chat template (or explicitly justify and compare against it as an arm). If the model emits `<think>...</think>` reasoning blocks, verify how thinking is disabled or budgeted (e.g., `enable_thinking=False` or an explicit thinking budget) before launching paid compute.
3. **Small-sample smoke decode before full launch.** Decode 2–3 representative masks with the exact prompt and sampler settings and read the raw text and emitted end tokens by eye before launching a full Modal or laptop run. If the output shows template mismatch, thinking-budget exhaustion, or zero end-token emissions, stop and fix the setup first.
