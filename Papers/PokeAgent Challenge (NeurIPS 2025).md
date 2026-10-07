---
created_at: 2026-08-14
updated_at: 2026-08-23
url: https://pokeagent.github.io/
venue: NeurIPS 2025
type: competition
status: past
organizers: Seth Karten et al.
tracks: Battling; Speedrunning (RPG)
resources: 20M+ battle trajectories; heuristic + RL + LLM baselines
relevance: high
tags:
  - workshop
  - pokemon
  - competitive-battling
  - reinforcement-learning
  - llm-agents
  - relevance/high
  - plasticity
---

- Foul Play method trying to learn gain more information for each move such speed, items
- 4thLesson

## A. Foundational Reading & Research

- [ ] [[Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Papers/Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers|Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]]
	- [ ] Where did the Human-Level Competitive Pokémon via Scalable… paper get this idea from: "Key tricks: multi-γ critics, Binary+MaxQ losses, and two-hot value regression in V2."
**owner verdict 2026-08-26: survey with no citations, contribution worthless — skip** — arXiv 2605.09965, 15 authors, 201 references, 0 citations, no venue beyond arXiv
- [ ] Collect all the todo items about [[PokeAgent Challenge @ NeurIPS 2025]] under one todo item
	- [ ] look at [[PokeAgent Challenge @ NeurIPS 2025]] and find similar research topics and generate new todo items from this — I will probably only be interested in track 1, the battle track, not the speed running track
		- [ ] Look at the citations of the paper — that is a good start; select the ones that pique your interest
		- [ ] Collect the NeurIPS video links about track 1 and not track 2
			- [x] [The PokeAgent Challenge: Competitive and Long Context Learning at Scale](https://neurips.cc/virtual/2025/loc/san-diego/129362)
			- [x] [Winner Awards & Lightning Presentations](https://neurips.cc/virtual/2025/loc/san-diego/129365)
			- [x] [Panel Discussion: Research Frontiers in Game AI](https://neurips.cc/virtual/2025/loc/san-diego/129366)
			- [x] [Kaggle Game Arena](https://neurips.cc/virtual/2025/loc/san-diego/129364)
			- [x] [Pokémon as an AI Problem](https://neurips.cc/virtual/2025/loc/san-diego/129363)
			- [x] Watch: [Pokémon as an AI Problem](https://neurips.cc/virtual/2025/loc/san-diego/129363)
- [ ] What is dynamic data weighting in [The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](https://www.semanticscholar.org/paper/The-PokeAgent-Challenge%3A-Competitive-and-Learning-Karten-Grigsby/80c6fcde8b5e8190a2e970db128eaf095887a46b)

## B. Participant Methodologies

- [ ] Read each of these approaches: "Battling Track names worth knowing when reading Appendix E: **FoulPlay** (#1 Gen 9 OU, search), **PA-Agent** (#1 among Gen 1 OU participants, offline RL), **Porygon2AI**, **Q**, **piploop**, **4thLesson**." — NOTE: these are **PokeAgent/VGC** participants (a different competition from the Kaggle TCG sim); useful as reference, not as TCG baselines.
- [ ] Participant Methodologies
	- [ ] P Porygon2AI ([[Grandmaster level in StarCraft II using multi-agent reinforcement learning|AlphaStar]]-style league)
		- [ ] [[Prioritized Fictitious Self-Play]]
		- [ ] Pull [[Grandmaster level in StarCraft II using multi-agent reinforcement learning|AlphaStar]] [43] itself (I have the arxiv ID resolver; it's Nature 575, also on [arXiv 1911.08265](https://www.semanticscholar.org/paper/c39fb7a46335c23f7529dd6f9f980462fd38653a)) and extract the exact PFSP sampling formula + league training loop so youhave the canonical algorithm to adapt for your own Track-1 agent. — paper pulled 2026-08-23: no arXiv version exists (1911.08265 is MuZero); free copy is DeepMind's PDF, sampling formula in Methods *Prioritised Fictitious Self-Play*
	- [ ] piploop
	- [ ] 4thLesson (curriculum (mechanics phase → strategy phase) + coach cycle)
		- [ ] How big was the model to be a concern about this and bigger the model more unstable it then coming from this ideas "Kron (Kronecker-factored Approximate Curvature) optimizer instead of theAdamW optimizer originally used in the Amago framework. Although the Kron optimizer is a second-order optimizer and incurs higher computational cost than AdamW, recent work demonstrates that it stabilizes reinforcement learning by ensuring more consistent gradient flow when scaling up model size, outperforming Adam-based optimizers in this regard."
		- [ ] First time instance I read a person change the activation function to change performannce and plasticy why where they are concern about plasiticty is just noise "they adopt AID (Activation by Interval-wise Dropout) instead of the previously used Leaky ReLU. AID introduces additional linearity into the model, mitigating plasticity loss in continual learning and reinforcement learning settings where model plasticity tends to degrade. Since AID also acts as a form of dropout, they removed the dropout modules used in the original model"
		- [ ] I don't understand what it trying to say here "To collect high-quality self-play data, they employ a multi-stage data generation strategy based on a local ladder setup. They first generated self-play replays using the 19 baseline models provided by a local ladder setup, collecting approximately 30k replays for small-size models, 40k for medium-size models, and 50k–100k for large-size models. All replays were generated using the modern_replays (v1) teamset, resulting in roughly 800k replays in total. In addition, they performed self-play between intermediate checkpoints of their own models, collecting another 300k–400k replays and increasing the total dataset size to approximately 1.1–1.2M replays. After the preliminary round, they repeated a similar process, generating about 10k samples per model, using the modern_replays (v2) teamset and adding approximately 130k more samples, increasing the total dataset size to 1.3–1.4M sample"


### FoulPlay


- [ ] Use claude code education to go through what is set skilled
- [x] get the foul play github
- [pmariglia/foul-play](https://github.com/pmariglia/foul-play) — FoulPlay PokeAgent entry (#1 Gen9 OU, #8 Gen1 OU); search bot over `pmariglia/poke-engine`.
- [pmariglia/poke-engine](https://github.com/pmariglia/poke-engine) — Rust Pokémon battle engine (singles only, gens 4–8) that searches through battle states: Expectiminimax, Iterative Deepening, MCTS, damage calc. Python bindings in `poke-engine-py`. Relevant to Track 1 competitive battling / search under partial observability. (Repo owner is `pmariglia`, not `pmargilia`.)
- Foul Play method trying to learn gain more information for each move such speed, items
### PaAgent (iterative offline RL + dynamic data weighting (anneal human→self-play data 100%→10%) + tournament team selection.)
- [ ] Reimplement PA-Agent's dynamic data weighting: anneal the training-data mix from 100% human replays to 10% human / 90% self-play across 6 rounds, regenerating self-play battles each round on top of the Metamon / qiaodev-pokeagent codebase.
- [ ] What is dynamic data weighting in [The PokeAgent Challenge: Competitive and Long-Context Learning at Scale (→ www.semanticscholar.org)](https://www.semanticscholar.org/paper/The-PokeAgent-Challenge%3A-Competitive-and-Learning-Karten-Grigsby/80c6fcde8b5e8190a2e970db128eaf095887a46b)
### Team Q use curriculm learning approach

- they created a 50 million paramter actor critic model
- Here are there github https://github.com/qiaodev/pokeagent
- Sources is from [Winner Awards & Lightning Presentations](https://neurips.cc/virtual/2025/loc/san-diego/129365)

## C. Prior Work / Baselines

- [ ] Prior work / baselines
	- [ ] Wang 2024 / [Winning at Pokémon Random Battles Using Reinforcement Learning](https://dspace.mit.edu/handle/1721.1/153888) — Jett Wang, MIT M.Eng thesis, Feb 2024 (advised by Joshua Tenenbaum); MCTS guided by an actor-critic net trained with PPO on self-play; peaked rank 8 / 1693 Elo on gen4randombattles. Strongest prior ladder scores for the Showdown battle environment (Track 1), and the baselines the challenge winners compare against.

## D. Advanced / Synthesis

- [ ] For the PokeAgent battle, find all the GitHubs and the ranking of where they competed
	- [ ] [UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon) — "Pokémon Showdown RL Agents and Datasets" (Python, 124★); PA-Agent's offline-RL base (Metamon framework)
		- [ ] Go through the UT-Austin-RPL/metamon GitHub to prepare for the competition: work the 12-section README (Install → Quick Start → Pretrained Models → Battle Datasets → Team Sets → Baselines → Obs/Action/Reward Spaces → Training & Evaluation), run a pretrained policy (e.g. Kadabra3) on the local Showdown ladder, then attempt finetuning so the offline-RL base is understood before layering PA-Agent's dynamic-data-weighting on top
	- [ ] [qiaodev/pokeagent](https://github.com/qiaodev/pokeagent) — official PokeAgent Challenge code: starter kit, baselines & synthetic battle generation (Python)
	- [ ] [sethkarten/pokechamp](https://github.com/sethkarten/pokechamp) — "PokéChamp: an Expert-level Minimax Language Agent" (Python, 176★); LLM+minimax baselines
- [ ] Determine if there is anything relevant going on in [[Automatic Generation of High-Performance RL Environments]]


