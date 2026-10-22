# Project instructions for Codex

## Keep the skills ranking current

After user-requested skill installations or configuration changes, follow the personal skills-report synchronization rule in the user's Codex AGENTS.md. Update `pstack-audit/rankings.json` and `pstack-audit/index.html` together. User-selected skills take priority in the default ranking; installation and automatic invocation remain separate facts.

## Recall context before grilling

Before interviewing or grilling the user, automatically read and apply [recall](.agents/skills/recall/SKILL.md), including when using a plugin-provided grilling skill. No slash command is required. Recover relevant previous chats and project decisions before forming questions, and carry supported answers forward as settled. Follow recall's source checks and report unavailable history honestly.

For this project's Pokémon counter search, the user's stated meaning of “better” is a higher battle win rate. Do not ask the user to define that again. Recover the existing opponent set, evaluation protocol, and comparison budget from the project where available. Ask only about unresolved choices that materially affect the comparison; do not silently invent an opponent distribution or evaluation budget. Later user instructions override this default. Briefly state the assumptions being carried forward, then ask only the remaining questions.

## Blog and documentation writing

Unslop also applies independently to ordinary replies and progress updates, following the personal Codex instructions. This does not expand technical-writing or writing-workshop to unrelated tasks.

When the user wants to learn technical writing through an HTML before-and-after comparison or feedback on their own revisions, use [writing-workshop](.agents/skills/writing-workshop/SKILL.md). Prioritize their own attempts, explained rules, and optional examples. Avoid offering a ready-to-copy AI rewrite in this learning mode; ordinary writing requests continue to use technical-writing as described below.

When drafting, editing, or reviewing this project's blog posts or documentation, read and apply [technical-writing](.agents/skills/technical-writing/SKILL.md) automatically; no slash command is required. Its project-specific priorities preserve the user's voice, requested format, citations, and scientific uncertainty. Use its linked prose helper only within the current writing task. Ordinary coding, research analysis, and conversational answers do not require this skill unless the user asks for it.

## Separate shared state before serializing

Apply this rule when agents, processes, experiment runs, or other project chats could write to the same mutable resource. It does not require extra agents, locks, or worktrees for read-only or already isolated work.

1. **Identify shared writes before starting.** Check the relevant running jobs, output paths, Git working state, caches, databases, checkpoints, manifests, and temporary files. Different fields in one JSON file, rows in one CSV, or sections in one document still share a write target. Do not assume another chat or process is idle.
2. **Separate independent work first.** Assign each writer owned files or a distinct run directory. Use collision-resistant run IDs; retain inputs, configuration, seed, code revision, and checkpoint provenance needed to interpret results. Use supported run-directory and database options rather than allowing concurrent experiments to collide with defaults. Share immutable inputs read-only where possible.
3. **Protect Git state too.** Parallel edits to disjoint files may share a checkout only with explicit ownership and one integrator. Use separate worktrees when workers need independent branch, staging, commit, or merge operations. Worktrees do not isolate external databases, output directories, services, or all repository metadata; account for those separately.
4. **Integrate completed outputs deliberately.** A designated integrator checks each producer has finished and validates its output before merging results. Publish complete files atomically where supported so readers do not consume partial results. Atomic replacement alone does not coordinate competing writers.
5. **Coordinate only unavoidable shared writes.** Use a single writer, a process-safe lock honored by every writer, database transactions with the necessary isolation, or atomic compare-and-swap. Protect the whole read–modify–write operation. Chat messages, timing assumptions, and an in-process mutex are not cross-process concurrency control. If existing writers cannot join the protocol, use isolated outputs or defer the conflicting write.

Preserve other sessions' edits and existing experiment results. Resolve collisions by selecting a new owned target or coordinating integration, not by deleting results, resetting a checkout, or killing another session's job. Resume checkpoints only when intentionally selected and compatible with the run. Do not build a new coordination framework when existing project mechanisms suffice.

Source: [Pstack — Separate Before Serializing Shared State](https://github.com/cursor/plugins/blob/195d9359bdc2890f83745df69927528ad4538406/pstack/skills/principle-separate-before-serializing-shared-state/SKILL.md), adapted for this project's Codex sessions and experiment workflows.
