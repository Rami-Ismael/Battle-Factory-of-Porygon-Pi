I want Claude Code to consider subagents on its own, without my typing it every time. The goal is for me to learn when delegation pays off, so make the decision visible instead of automatic.

Constraints:

- Leave the default system prompt and the project CLAUDE.md alone. My personal ~/.claude/CLAUDE.md is fine.
- Keep the criteria in that file under "Subagent check". Use a UserPromptSubmit hook only as the trigger. A hook cannot rewrite my prompt; it can only add a note beside it.

What each reply does:

- On a task, open with one line: "Subagent:" plus the piece a subagent would take, the agent type and the one reason. Or "Subagent: none" plus the reason.
- On a task with three or more parts, replace the line with a short split. Mark each part "subagent" or "here", and end each part in a check, following principle-sequence-verifiable-units.
- Suggest, don't spawn. A subagent starts only when I say so.
- Until 2026-10-31, write the line on every task, "none" included. After that, write it only when a subagent fits.
- Apply this in every project, not just the vault.

Subagent briefs:

- Subagents run on Haiku, which compacts at 100k tokens. Give each one a single job with a done-condition it can check. Scaffold, layout and script get separate briefs.
- Write every brief with the writing-for-agents skill.
- For a design question, explore directions side by side: one Explore agent per direction, with the results compared in the main session, as in the egg-drop demo.

How to tell whether it works:  
Run eight of my real prompts in fresh sessions under three setups: hook only, CLAUDE.md only, and both. Use four tasks, two questions and two short replies. Count how often the line opens a task reply, and how often it appears where it shouldn't.