---
created_at: 2026-08-23
updated_at: 2026-08-23
type: paper
title: "Grandmaster level in StarCraft II using multi-agent reinforcement learning"
short_name: AlphaStar
authors: Oriol Vinyals, Igor Babuschkin, Wojciech M. Czarnecki, Michaël Mathieu, Andrew Dudzik, et al. (43 authors), Chris Apps, David Silver
lab: DeepMind, London (with Dario "TLO" Wünsch, Team Liquid)
venue: Nature 575, 350–354 (2019)
doi: https://doi.org/10.1038/s41586-019-1724-3
doi_access: 🔒 Nature paywall
pdf: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
access: 🟢 open — DeepMind's unformatted PDF, 29 pages, includes Methods and Extended Data
arxiv: none — this paper was never posted to arXiv
relevance: HIGH battle policy · LOW f
relevance_scope: the human-replay pipeline and league training, not the architecture; no team-search content
key_sections: "Methods — Supervised Learning (971,000 replays, MMR > 3500 = top 22%; fine-tune on winning replays with MMR > 6200) · Exploration and diversity (the z statistic = build order + units/buildings/upgrades built) · Multi-agent Learning · Prioritised Fictitious Self-Play · Populating the League (main agents, main exploiters, league exploiters) · Fig. 3 multi-agent ablations · Extended Data Fig. 8 payoff matrix"
tags:
  - self-play
  - league-training
  - behaviour-cloning
  - starcraft
  - deepmind
---
- [ ] 1. State in one sentence what AlphaStar's core contribution is — the human-replay pipeline plus population training, not the network.
- [ ] 2. Decide the read scope: the full Nature paper, or the main text plus only the human-data Methods subsections.
- [ ] 3. Predict which rating band wins the BC band experiment before running it — AlphaStar's hard MMR filter was only affordable at StarCraft data volume.
- [ ] 4. Decide whether the pure SP arm stays from-scratch as a control, so its gap to a BC-initialised arm measures what human data buys.
- [ ] 5. Decide whether VGC needs an analogue of AlphaStar's z statistic, or the piloted team already plays that role.
- [ ] [[What are the step before reading  the paper]]
