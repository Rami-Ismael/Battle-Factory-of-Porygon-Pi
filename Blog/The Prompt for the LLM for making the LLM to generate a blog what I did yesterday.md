---
created_at: 2026-08-23
updated_at: 2026-09-07
tags:
  - blog
  - daily-log
  - prompt
---

## Active prompt — Codex and ChatGPT

Copy this section, ending at **End of prompt**, into Codex or ChatGPT. In ChatGPT, attach or paste the relevant notes or transcripts when they are not already accessible.

Write one blockquote about what I did on **{DATE}** for this Pokémon VGC team-search project. Use `YYYY-MM-DD`; default to yesterday in my timezone (`America/Chicago` unless I specify another). The filename describes the day of the work, not the day you write the blog.

**Evidence — use the access you actually have.**

1. **Codex, or ChatGPT with project-file tools:** read the project files and relevant conversation history through available tools. If task-history tools are available, find this project's conversations and read the turns for the target day; use summaries to locate evidence, not as proof that an edit happened. Include conversations that began earlier and continued that day. When local transcripts are needed and readable, check `sessions/` under `CODEX_HOME` (default `~/.codex`), plus `archived_sessions/` if present. Discover their format and match the recorded project directory or explicit project references before reading their content. These are fallback locations, not a promise that every installation exposes them.
2. **ChatGPT with supplied or connected context:** use messages visible in this chat, available project sources, uploaded notes or transcript exports, and sources retrievable through enabled tools. Do not assume access to my Mac, all past chats, or local Codex logs. General memory about the project is not dated evidence of yesterday's work.
3. **Claude Code / t3code, when available:** also use the project-matching JSONL transcripts under `~/.claude/projects/`. The historical directory for this project is recorded in the archived prompt below. Missing Claude logs do not prevent using other sources.
4. **Filter by event time:** parse timestamps inside transcripts, convert them to my timezone, and keep events from local midnight on `{DATE}` up to, but excluding, the next midnight. Check adjacent UTC dates when necessary. Transcript filenames and modification times are not event dates. Read my actual typed messages, relevant tool calls, and their results; exclude tool output masquerading as user text and duplicated transcript records. Assistant prose alone does not prove an action or my agreement with an idea.
5. **Check file changes:** when filesystem tools are available, find project files whose modification times fall in that same local-day interval, then inspect relevant contents and available edit records. Treat mtimes as leads: sync, bulk rewrites, or assistant edits can change them. Use successful tool results, patches, dated notes, or my explicit account to establish what changed. A missing file alone does not prove a deletion. This vault has no Git history; use files and conversations, and keep Git/worktree terminology out of the blog.

If there is no usable evidence for the target day, ask me once for dated notes or transcript excerpts instead of inventing a blog. If evidence covers only part of the day, write from that evidence and acknowledge the limited coverage briefly within the blockquote.

**Attribution and content.** Report my edits, creations, deletions, renames, questions, decisions, and intent. A confirmed assistant write, patch, or shell edit is assistant-authored regardless of size; the absence of a matching tool call does not establish that I wrote it. Describe my direction as “I asked Codex to…” or “I decided…” when supported, and claim a completed change only when there is evidence it succeeded. Keep plans, discussions, and completed work distinct.

Rank confirmed deletions first: name the note and what it contained. Also cover what I created and the substantive topics I discussed, including conversations that changed no files. Say what the note or decision means, not just its title. If one note required an unusual number of retries or corrections, describe the supported pattern briefly without inventing a cause. Never claim I “learned” anything.

**Form and delivery.** Write one Markdown `>` blockquote, at most 2,048 characters including Markdown, in complete first-person sentences. No title, list of note titles, “Yesterday:” opener, preamble, or closing commentary. Name specific decisions, the way “killed bandits from the team search” does. If the day was thin, say the thin thing; do not pad it with “thinking about the problem.”

If tools can write to the actual project folder, save the blockquote as `Blog/Blog for {DATE}.md` and verify the saved content. Preserve an existing blog unless I explicitly request replacement; otherwise save the new version as a separate, clearly named draft. If you cannot write to the project folder, return the blockquote for me to save there; a downloadable file, if available, is not a save to my vault. Only claim a save after it succeeds. Before delivering, check the date, evidence, attribution, and character limit.

**End of prompt**

---

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
