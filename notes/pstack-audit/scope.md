# Audit scope and additional evidence

Date: 2026-10-13. User confirmed MacBook only and GPT-6 Astra for audit subagents. No other machine accessed. No skills installed.

Source revision: https://github.com/cursor/plugins/tree/195d9359bdc2890f83745df69927528ad4538406/pstack/skills

Primary sample: 60 visible chat summaries (50 recent plus 10 pinned), plus three recent turns from six selected chats. Three archived summaries were also inspected. Some visible summaries are duplicate/forked work; counts are not independent task counts. Evidence includes requests and reported agent outcomes; outcomes were not re-tested in this audit.

Supplement: inspected the first qualifying user request from each of the 100 most recently modified top-level Claude session files under ~/.claude/projects/*/*.jsonl. Excluded metadata and selected system/notification prefixes, then deduplicated exact request text: 89 qualifying requests, grouped by folder as 43 VGC, 21 TCG, and 25 other. This is a convenience sample, not an exhaustive or statistically representative audit. File modification recency is not necessarily conversation creation recency. Earlier/later turns and sessions beyond this window may change recommendations.

Selected additional evidence (Claude session identifiers; user requests, not verified outcomes):

- claude:6a78b75d-b75e-43a1-8cce-28c96abf395b — asks for a walkthrough of Learning-to-Beat-ByteRL.
- claude:b88712c4-86d8-474d-8fe9-32a7f3a7f401 — asks for exercises and explanations for the poker 4×2 odds rule.
- claude:5a0a0230-22ab-44d5-9040-6d4ee6652338 — asks to create Mochi flashcards from YouTube material.
- claude:28c2bfb7-7144-446c-83b8-a075ee16496f — complains about an AI doubling down on nonsensical experiment claims.
- claude:ab8e4535-b994-4e38-80e5-0efb67b18787 — asks for a competition retrospective and improvement toward medal performance.
- claude:09d3529c-340f-4dbf-9564-190da9a0bd03 — reports a benchmark deck-override bug that collapsed different configurations to one.
- claude:28cf6bc6-610d-4c5b-92af-028c28b8c969 — asks for a precompetition Kaggle checklist.
- claude:87797425-44ff-4cd8-b283-491032a714b4 — asks for Haiku subagent auto-compaction at 100k tokens.
- claude:ebdbc626-0734-4eef-887a-ef82f8c36b48 — corrects a claim about testing all 100k generated teams.

Interpretation: prioritize evidence-backed experimentation, teaching, reusable verification, independent comparison of alternatives, concise writing, and efficient agent context. Do not interpret a user's request to fix a problem as proof that the fix succeeded.

Scoring: fit measures alignment with observed work (1–10); benefit estimates incremental value beyond existing instructions and skills (1–10). These are editorial judgments, not measured productivity improvements. Existing tools already include Matt Pocock research, grilling, debugging, TDD, domain modeling, writing-for-agents, prototype and numerous visual design skills, plus context-mode. Availability does not prove consistent usage or effectiveness. Revised ordering prioritizes benefit, then fit, then name. All 44 benefit assessments were reconsidered after user feedback; fit scores remain unchanged. The user explicitly rejected the value of create-verification-skill and requested revision of the entire list. Treating that as a preference for direct outcomes over process and maintenance overhead is an assistant inference, not 44 explicit user judgments. Some scores remain unchanged after reconsideration.

Design: a local dark report inspired by Theo's ranked table. Near-black #101310, off-white #edf1e9, muted #acb5a6, green #b4d89b, divider #354032. Georgia display title and system sans-serif body. Left-aligned sortable/searchable rows; detail disclosures hold evidence and caveats. Rank numbers encode ordering. No autoplay, sound, remote assets, or decorative animation. This is a new report, not a reproduction of an unpublished Theo design prompt.
