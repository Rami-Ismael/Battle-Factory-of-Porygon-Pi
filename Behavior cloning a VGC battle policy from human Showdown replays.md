---
created_at: 2026-08-18
updated_at: 2026-08-18
type: experiment
status: open
title: Behavior cloning a VGC battle policy from human Showdown replays
one_line: human VGC Reg M-B replays / ladder logs from Pokémon Showdown → behavior-cloning model → doubles battle policy for the pipeline's first stage
decided: 2026-08-18 — doubles only; BC is the first candidate in the battle-policy comparison (BC / SP / FP / DO)
moved_from: "[[Todo Section]] — checkbox block of 2026-08-18, relocated 2026-08-18"
data_v0: VGC-Bench replays
data_v1: v0 + Reg M-B Showdown scrape
codebase: VGC-Bench BC agent on poke-env — assumed, not yet confirmed
related:
  - "[[Why Self-Play is consider Multi-Agent]]"
  - "[[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]]"
tags:
  - battle-policy
  - behavior-cloning
  - imitation-learning
  - showdown-replays
  - doubles
---
- [ ] 2026-08-18, Collect human battle data from Pokémon Showdown (VGC Reg M-B replays / ladder logs) and train a behavior-cloning model on it, to get a VGC battle policy for the pipeline's first stage — one that ranks matchups the way human play would, since Monte Carlo evaluation is its only consumer
	- decided 2026-08-18: doubles only — BC is the first candidate in the 08-15 comparison below
	- [ ] dataset v0 = VGC-Bench replays; v1 = v0 + Reg M-B Showdown scrape — compare v1 vs v0 to test whether more data helps
	- [ ] two heads: team-preview selection (4-of-6) + per-turn action
	- [ ] rating filter on cloned players; train one model per rating band and compare — a research experiment in itself
	- [ ] yardstick: win rate vs VGC-Bench's own BC baseline as the cheap proxy; rank correlation against ladder validation before it enters Monte Carlo evaluation — Claude's recommendation, not yet confirmed
	- [ ] code: fork VGC-Bench's BC agent on poke-env — assumed, not yet confirmed
