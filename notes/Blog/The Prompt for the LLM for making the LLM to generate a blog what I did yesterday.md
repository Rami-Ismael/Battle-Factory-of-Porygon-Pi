---
created_at: 2026-08-23
updated_at: 2026-10-19
tags:
  - blog
  - daily-log
  - prompt
---
## Todo

- [ ] /improve-codebase-architecture survey: finds candidates (loads codebase-design itself)
- [ ] codebase-design ← here: shape the interface, place the seam, run the deletion test, design it twice if unsure
	- [ ] Return a list of modules that are too thin in my current codebase / 
- [ ] I want you add another step in the process in creating a dailty blog for the code section you will determin ewhich module a
- [ ] Use the Can we do some /mattpocock-skills:tdd for the new written code you have write about about for the thin module
- [x] retro
## Active prompt — Codex and ChatGPT

Copy from **Goal** through **End of prompt**. Supply notes or transcripts when the agent cannot access them.

### Goal

Write a daily blog about what I did on **{DATE}** for this Pokémon VGC team-search project, why each task mattered, and what it established or left unresolved. Follow the workflow and constraints below.

### Workflow

1. **Set the scope.** Use `YYYY-MM-DD`; default to yesterday in `America/Chicago` unless I specify another date or timezone. Establish the interval from local midnight up to, but excluding, the next midnight. Match sources to this project, including related checkouts, before inspecting their contents. This step is complete when the date, timezone, project boundary, and available sources are identified.

2. **Gather evidence.** Use task-history tools, project files, supplied notes, and connected sources that are actually accessible. Include conversations that began earlier but continued within the interval. With local access, inspect matching Codex transcripts under `CODEX_HOME/sessions/` and `CODEX_HOME/archived_sessions/`, defaulting to `~/.codex`. Also inspect matching Claude transcripts under `~/.claude/projects/`, including nested subagent and workflow logs and related siblings ending `--claude-worktrees--<id>`. Discover actual formats and paths. Missing sources do not prevent using others.

   Filter timestamps inside events, converting them to the chosen timezone. Check adjacent UTC dates when necessary. Read my typed messages and relevant tool calls and results; distinguish tool output from my messages and remove duplicate records. Use summaries, memory, filenames, and modification times only to locate evidence. Inspect relevant changes and project-related deleted notes in the vault's `.trash`; exclude unrelated vault notes and Obsidian's `lastOpenFiles`.

   When subagents are available, delegate independent conversation, file-change, and citation checks. Give each the scope and these constraints. Require evidence records containing the timestamp, source location, supporting excerpt or result, actor, action, purpose, outcome, artifact link, and uncertainty. Subagents gather evidence without editing the blog. The main agent checks the underlying sources and resolves conflicts. Without subagents, perform these checks directly.

   Also collect every frustration in those conversations with Claude Code or Codex. Miscommunication counts as frustration. So do failing to do exactly what I asked, not reasoning carefully enough, instructions I had to repeat, wrong assumptions or labels I caught, outputs I rejected, questions I re-asked, hook or tool blocks, and stalls. Record each with the exchange that shows it, in the same evidence-record form. Also record which skills each session loaded, from Skill tool calls and typed slash commands.

   This step is complete when each candidate task has an evidence record and coverage gaps are identified. If no usable evidence exists, ask once for dated notes or excerpts before writing. For partial coverage, write from the evidence and briefly acknowledge that limit inside the blog.

3. **Draft and cite.** Write the draft with the `pskill:technical-writing` skill, then give it a final `pskill:unslop` pass. Neither skill may change a fact, number, citation, quotation, or the length limit. Select supported tasks and apply the constraints below. Preserve substantive discussions even when no files changed. For each included task, connect the action to its supported purpose and outcome. This step is complete when every factual claim has evidence and every mentioned generated output has a verified link or an explicit missing-link note.

4. **Verify and save.** When subagents are available, give one the draft and the evidence records and have it re-check every factual claim, number, attribution and link against its source, flagging anything unsupported; the main agent resolves each flag at the source before saving. Check the target date, each claim's attribution and event-time support, each task's purpose, every citation, and the rendered-text length. Save to `Blog/Blog for {DATE}.md` in the actual project folder; the filename describes the workday. Preserve an existing blog unless I explicitly authorize replacement; otherwise save a clearly named separate draft. Before writing, check for intervening edits or another writer and use a separate draft if the target cannot be updated safely. Read back the saved file and verify its content and links. Completion requires a successful save and readback. If writing to the project is unavailable, return the post for me to save and accurately identify that delivery limit. A downloadable file is not a vault save.

### Constraints

**Attribution.** Establish completed changes from successful tool results, patches, dated notes, or my explicit account. A missing file alone does not prove deletion; an absent tool call does not prove my authorship. Check relevant text differences and main-agent and subagent records. Assistant edits remain assistant-authored regardless of size. Use "I asked Codex to..." or "I asked Claude to..." for my direction and "I decided..." for supported decisions. Distinguish discussions, plans, attempts, approvals, and completed work. For vault changes, use file and conversation evidence rather than assuming Git history. Cite relevant external pull requests.

**Purpose and content.** Put confirmed deletions first and name what was removed and what it contained. Cover creations, edits, renames, questions, decisions, and substantive discussions. Explain the problem, question, or risk each included task addressed and why it mattered to the project. Use my stated reasons or supported project context. When my motivation is unrecorded, explain the supported technical purpose without attributing an invented intention to me, or state that the reason is unclear. Describe what a note says, not only its title. Mention unusual retries only when evidenced, without inventing a cause. Never claim I "learned" something. Keep thin days brief.

**Artifact and source citations.** Use standard, unescaped `[descriptive label](destination)` links within the prose. Cite every AI-generated output mentioned as created, changed, or used, including reports, visuals, lessons, briefs, prompts, datasets, experiment results, and code from ChatGPT, Codex, Claude, or subagents. Link the output itself; a paper or PR citation does not replace an available artifact link.

Recover destinations from successful creation or publication records and verify the target's identity and relevant content. Prefer existing artifact URLs or paths relative to the blog for project files. Use verified absolute paths for outputs outside the vault; enclose destinations containing spaces in angle brackets. If an output cannot be located, retain independently supported activity and briefly identify the missing artifact link. Mark private links where useful and preserve sharing settings. Never invent URLs or create public links to private conversations.

Also link mentioned papers, PRs, issues, repositories, and documentation directly to their primary sources. A source's contents support its subject matter; dated evidence must separately establish my activity and the source's state on the target day. Omit claims that depend on unverified sources. Keep citations inline in the post, without a separate bibliography.

**Visuals and existing diagrams.** Include a relevant visual when it helps explain a task, method, or result. Prefer reusing a verified existing diagram, chart, screenshot, or generated artifact. Inspect it before including it and cite its source or original artifact. Identify older visuals as reused context rather than work created on the target day. If creating a new explanatory diagram, label it as an illustration added when writing the blog; base its claims and numbers on verified evidence. Place visuals and concise captions in the post body, next to the text they support. Use Markdown image embeds or supported diagram blocks. When an interactive artifact cannot be embedded reliably, provide a descriptive link. Verify that embeds render in the intended Markdown viewer when available; otherwise verify their targets and state any rendering limitation.

**Frustrations and communication problems.** I collect frustrations to find solutions, so later sessions with Claude or Codex do not repeat them. Include a frustration table, drawn from the frustration records, with one row per frustration and these four columns:

- **Frustration:** what happened, quoting or closely paraphrasing my words, and its kind, such as miscommunication, not doing exactly what I asked, or not reasoning carefully enough. Add the cost (rework, battles, time) when the evidence shows it.
- **Source:** the tool (Claude Code or Codex) and the session where it happened, by session title or ID, with the time.
- **Fix:** one concrete change that would prevent it next time: in my prompts (for example, grilling before a run), the project instructions, a skill, a visual, or the tool's behaviour. Name an existing skill when one fits.
- **Status:** whether the fix already exists, was done that day, or is a proposal.

Separate the assistant's errors from my unclear requests, and never attribute feelings I did not express. If there was no frustration, say so in one sentence.

**Skills I could have used.** List the skills I did not use that day but that would have helped one of the day's key tasks, as checkbox to-do items I can copy. Each item names the task, the skill, and in one sentence how it would have helped. Build the list in four steps, each ending in a check:

1. Here: list every Claude Code and Codex session with events in the interval. For each one, record the skills it loaded (Skill tool calls and typed slash commands) and my typed messages. Check: the session count matches the transcripts found in step 2 of the workflow.
2. Here: list the installed skills with their one-line descriptions, from Claude Code and Codex. Check: every skill named later appears in this list.
3. Subagent: give one Haiku subagent per small packet of sessions (messages, loaded skills, and the installed-skill list). Each subagent names the task each session did and proposes installed skills that the session did not load but that fit the task. Check: every session in the packet has a task, and every proposal names an installed skill absent from that session's loads.
4. Here: verify each proposal against the skill's description and the session's transcript, keep the ones that clearly fit, and write the checkboxes. Check: each kept item names its session and its skill.

Mark a skill that is not installed as a suggestion. If no skill fits, say so in one sentence.

**Images of the day.** Include one to three images of what I produced that day, so the post shows the day at a glance: a screenshot of a key artifact or prototype, a chart from an experiment result, or another visual output. Prefer the day's own outputs to older visuals. Inspect each one, screenshot the rendered page rather than its source, save the PNG under `Blog/images/{DATE}/` with a descriptive name, and embed it with a relative Markdown image link next to the paragraph it supports, with a one-line caption linking the original artifact. Do not capture private conversations or secrets. If no visual exists or capture fails, say so in one sentence.

**Format and length.** Write plain Markdown paragraphs, not a `>` blockquote or callout, with prose in complete first-person sentences; visuals may use short labels and captions. Use at most **16,384 visible characters** across the entire entry, counting rendered prose, link labels, captions, text visible inside visuals, spaces, and punctuation. Visual labels and captions count against this same limit. Exclude image pixels, destination URLs, and non-rendered Markdown or diagram syntax. Count image alt text only when displayed as fallback text. Count rendered text rather than raw file length. Shorten lower-priority prose before dropping required explanations or citations. The limit is a ceiling, not a target. Use occasional emphasis when useful. Include no title, list of note titles, "Yesterday:" opener, preamble, closing commentary, or filler.

**End of prompt**

---

## Previous active prompt — archived 2026-10-13

Reference only; the goal and constraints above supersede this version.

<details>
<summary>Previous Codex and ChatGPT prompt</summary>

Copy this section, ending at **End of prompt**, into Codex or ChatGPT. In ChatGPT, attach or paste the relevant notes or transcripts when they are not already accessible.

Write one blockquote about what I did on **{DATE}** for this Pokémon VGC team-search project. Use `YYYY-MM-DD`; default to yesterday in my timezone (`America/Chicago` unless I specify another). The filename describes the day of the work, not the day you write the blog.

**Evidence — use the access you actually have.**

1. **Codex, or ChatGPT with project-file tools:** read the project files and relevant conversation history through available tools. If task-history tools are available, find this project's conversations and read the turns for the target day; use summaries to locate evidence, not as proof that an edit happened. Include conversations that began earlier and continued that day. When local transcripts are needed and readable, check `sessions/` under `CODEX_HOME` (default `~/.codex`), plus `archived_sessions/` if present. Discover their format and match the recorded project directory or explicit project references before reading their content. These are fallback locations, not a promise that every installation exposes them.
2. **ChatGPT with supplied or connected context:** use messages visible in this chat, available project sources, uploaded notes or transcript exports, and sources retrievable through enabled tools. Do not assume access to my Mac, all past chats, or local Codex logs. General memory about the project is not dated evidence of yesterday's work.
3. **Claude Code / t3code, when available:** also use the project-matching JSONL transcripts under `~/.claude/projects/`. The historical directory for this project is recorded in the archived prompt below. Missing Claude logs do not prevent using other sources.
4. **Filter by event time:** parse timestamps inside transcripts, convert them to my timezone, and keep events from local midnight on `{DATE}` up to, but excluding, the next midnight. Check adjacent UTC dates when necessary. Transcript filenames and modification times are not event dates. Read my actual typed messages, relevant tool calls, and their results; exclude tool output masquerading as user text and duplicated transcript records. Assistant prose alone does not prove an action or my agreement with an idea.
5. **Check file changes:** when filesystem tools are available, find project files whose modification times fall in that same local-day interval, then inspect relevant contents and available edit records. Treat mtimes as leads: sync, bulk rewrites, or assistant edits can change them. Use successful tool results, patches, dated notes, or my explicit account to establish what changed. A missing file alone does not prove a deletion. This vault has no Git history; use files and conversations, and keep Git/worktree terminology out of the blog.
6. **Subagent and sibling logs:** top-level `*.jsonl` misses them. Also read `<session-id>/subagents/**/agent-*.jsonl` and `<session-id>/workflows/` in this directory and every sibling ending `--claude-worktrees--<id>`, filtered by event time like step 4. A subagent write is assistant-authored.
7. **My own Obsidian edits:** look only inside this project folder; ignore other vault notes, including `lastOpenFiles` in `.obsidian/workspace.json`. A note changed that day with no write in any main or subagent log, or whose text differs from the last version an assistant read or wrote, is likely mine; call it mine only when that diff or my typed account supports it, otherwise say the note changed. Check the vault `.trash` for deletions.

If there is no usable evidence for the target day, ask me once for dated notes or transcript excerpts instead of inventing a blog. If evidence covers only part of the day, write from that evidence and acknowledge the limited coverage briefly within the blockquote.

**Attribution and content.** Report my edits, creations, deletions, renames, questions, decisions, and intent. A confirmed assistant write, patch, or shell edit is assistant-authored regardless of size; the absence of a matching tool call does not establish that I wrote it. Describe my direction as “I asked Codex to…” or “I decided…” when supported, and claim a completed change only when there is evidence it succeeded. Keep plans, discussions, and completed work distinct.

Rank confirmed deletions first: name the note and what it contained. Also cover what I created and the substantive topics I discussed, including conversations that changed no files. Say what the note or decision means, not just its title. If one note required an unusual number of retries or corrections, describe the supported pattern briefly without inventing a cause. Never claim I “learned” anything.

**Form and delivery.** Write one Markdown `>` blockquote, at most 2,048 characters including Markdown, in complete first-person sentences. No title, list of note titles, “Yesterday:” opener, preamble, or closing commentary. Name specific decisions, the way “killed bandits from the team search” does. If the day was thin, say the thin thing; do not pad it with “thinking about the problem.”

If tools can write to the actual project folder, save the blockquote as `Blog/Blog for {DATE}.md` and verify the saved content. Preserve an existing blog unless I explicitly request replacement; otherwise save the new version as a separate, clearly named draft. If you cannot write to the project folder, return the blockquote for me to save there; a downloadable file, if available, is not a save to my vault. Only claim a save after it succeeds. Before delivering, check the date, evidence, attribution, and character limit.

**End of prompt**

---

</details>

## Archived versions — reference only

The prompt above supersedes the following instructions. They are retained verbatim for history and are not active instructions.

<details>
<summary>Claude-focused prompt, last updated 2026-08-24, and earlier history</summary>

Write one blockquote about what I did on **{DATE}** (default: yesterday). Save it as `Blog/Blog for {DATE}.md` — the filename is the day being described.

**Evidence.** There is no git in this vault; do not look for one, and do not use the word "worktree."

1. **Read my Claude Code sessions.** Every session I ran on this project — in Claude Code and in t3code, they share one directory — is a JSONL transcript, one JSON object per line, in:

    ```
    /Users/ramiismael/.claude/projects/-Users-ramiismael-Documents-yakumsi-vault-Personal-Project-Using-Advance-method-search-in-the-whole-search-of-possible-vgc-pokemon-format-to-find-the-right-counter-to-a-pokemon-team/*.jsonl
    ```

    Select by the `"timestamp"` field **inside** each line, not by the file's mtime — the mtimes get rewritten in bulk and are worthless. Keep only lines whose `"timestamp"` starts with `{DATE}`. Within those, read my typed prompts (the `"type":"user"` records that are text I wrote, not tool results) and the file-operation record (`file_path` in tool calls). Skip assistant prose.
2. `find` the project folder for files whose mtime falls on `{DATE}`.

**Attribution.** A file that changed with a matching Claude tool call that day is Claude's; a file that changed without one is mine. `Write` is Claude's regardless of size. Report mine. If one note absorbed an unusual pile of operations, say so — that is usually Claude flailing, and it counts as part of the day.

**Content.** My edits, deletions, and intent. Rank deletions first — name the note and what it was. Never claim I "learned" anything.

**Form.** One `>` blockquote, ≤2048 characters. Complete sentences, first person — not a list of note titles; say what the note says. No "Yesterday:" opener, no preamble. Name specific decisions, the way "killed bandits from the team search" does. If the day was thin, say the thin thing — do not pad it with "thinking about the problem."

---

*Retired — superseded 2026-08-23 by the prompt above. Kept for the record; "worktree" has no referent in this vault and the opening sentence pre-answered the question.*

> Yesterday, I spent some time thinking about the problem and making some edit for markdwon . I want to write summary what I did yesterday look at all the worktree so what I did


- [x] Fix the blog the current [[Blog for 2026-08-22]] is poor only talked about what I deleted only not what i created or topic discuss with t3code — done: [[Blog for 2026-08-22]] keeps the deletions and adds the notes written and the action-space / diffusion threads 

</details>
