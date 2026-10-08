---
created_at: 2026-08-17
updated_at: 2026-08-23
type: paper
title: "VGC-Bench: Towards Mastering Diverse Team Strategies in Competitive Pokémon"
authors: Cameron Angliss, Jiaxun Cui, Jiaheng Hu, Arrasy Rahman, Peter Stone
lab: Learning Agents Research Group, UT Austin (Peter Stone) — same lab as UT-Austin-RPL/metamon
venue: AAMAS 2026 — 25th International Conference on Autonomous Agents and Multiagent Systems
arxiv: https://arxiv.org/abs/2506.10326
arxiv_versions: v1 2025-06-12 · v2 2025-06-13 · v3 2026-01-13
openreview: https://openreview.net/forum?id=lt9sp9JVIy
repo: https://github.com/cameronangliss/vgc-bench
dataset: https://huggingface.co/datasets/cameronangliss/vgc-battle-logs
paper_license: CC BY 4.0
code_license: MIT
dataset_license: MIT
stack: Pokémon Showdown + poke-env (doubles)
access: 🟢 all three links open
relevance: high
tags:
  - pokemon
  - vgc
  - benchmark
  - evaluation-protocol
  - battle-policy
  - relevance/high
---
# Todo

- [ ] [[Why Self-Play is consider Multi-Agent]]
- [ ] Define how did you self play
	- Team sampling ([`vgc_bench/src/teams.py` on GitHub](https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/src/teams.py), checked 2026-08-23): pool = first N of a run_id-seeded shuffle of the scraped VGCPastes teams (`RandomTeamBuilder._select_paths`); each battle both players independently draw uniformly from the pool (`yield_team` → `random.choice`); `take_from_end` carves a disjoint held-out pool; the policy still makes the selection at preview.
- [x] What were the action space, and detemine how they made it equip to handle team selection — done 2026-08-18: 107 per active slot (Table 6), factored per slot, −∞ masking (§4.2.1); team preview = two joint switch-in actions (§4.1), no separate head — see [[Determine what is action space]]
- [x] Tell claude to stop talking about mirror matching as the paper [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] did not do any experiment of mirror match the less thing we have to do the better — done 2026-08-23: recorded in Claude's memory; Mirror Matching is already item 3 in [[Thing I don't want my claude or opencode be talking about]]
- [ ] What were the observation space
- [ ] Determine what is thier baseline agent it was random 
- [ ] Determine what is their success criteria and deliverable [[Determine what are some deliverable can I do]]
- [ ] Go through the paper
	- [ ] **A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning** (Lanctot et al., 2017)
	- [ ] **α-Rank: Multi-Agent Evaluation by Evolution** (Omidshafiei et al., 2019)
	- [ ] **[Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers](https://www.alphaxiv.org/abs/2504.04395)** (UT Austin, 2025)
- [ ] Can I replae PettingZoo with Pufferlib
- [ ] [[How do you implementing maskout in reinforcement learning and whey do you need to mask out]]
- [ ] What is pettingzoo, how is relevant to mutliagent RL  in double pokemon as we self play is consider multiagent


# Self Implementation

1. my uind


# Quotes


1. VGC-Bench builds upon the poke-env library, extending it to support the intricacies of Gen 9 Double Battles and integrating it with the PettingZoo multi-agent framework. This integration allows for parallelized training, which is crucial for modern reinforcement learning (RL) techniques.[1][[Citation]] 
	1. Most of the changes adding double support in poke-env



