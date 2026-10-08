---
created_at: 2026-08-19
updated_at: 2026-08-23
eli5: Teaching/eli5/self-play-survey-choosing-your-sparring-partner.html
type: paper
title: "A Survey on Self-play Methods in Reinforcement Learning"
authors: Ruize Zhang, Zelai Xu, Chengdong Ma, Chao Yu, Wei-Wei Tu, Wenhao Tang, Shiyu Huang, Deheng Ye, Wenbo Ding, Yaodong Yang, Yu Wang
labs: Tsinghua University, Peking University, Tencent, Zhipu AI, 4Paradigm
venue: arXiv preprint (no journal-ref as of v4)
arxiv: https://arxiv.org/abs/2408.01072
arxiv_versions: v1 2024-08-02 · v4 2025-10-18
access: 🟢 open
relevance: high
relevance_scope: battle-policy arm only (self-play taxonomy) — not the team-search objective
key_sections: §3.2.2 vanilla self-play · §3.2.3 fictitious play · §3.2.4 δ-uniform self-play · §3.2.5 prioritized fictitious self-play · §3.3.2 double oracle · §5.2 non-stationarity · §4.1.1 AlphaGo family (AlphaGo = Eq. 6 FP; AlphaGo Zero = Eq. 5 + 55% gate; AlphaZero drops the gate) · §5.3 policy-pool storage
tags:
  - self-play
  - multi-agent-rl
  - battle-policy
  - survey
  - relevance/high
---

- [ ] [[What are the step before reading  the paper]]
- [ ] Explain to yourself why we don't make a copy of the model after each episode
- [ ] Make sure don't confuse alphago vanilla gating self play with double oracle
- [ ] What does oracle measn
- [ ] In one sentence: what is the single design question every self-play method answers?
- [ ] A friend says "fictitious play trains you against the average of all your past networks." Two things are wrong with that?
- [ ] What does the survey mean by "uniform over all earlier — Eq. (6)"?
- [ ] Which side of that split is Pokémon VGC on, and what does this project therefore measure instead of one score?
- [ ] The famous Go/Dota bots are often summarised as "just vanilla self-play." Give two reasons that summary is wrong.
- [ ] > Is the "neural" in Neural Fictitious Self-Play doing any real work, or is it just Fictitious Self-Play with a network as the function approximator?


# Prompt 

1. I want to learn about this paper named [[A Survey on Self-play Methods in Reinforcement Learning]] in my vault. Can you firsts eli5, teach m skills  , after do the rest of the step in [[What are the step before reading  the paper]]. Please write response in chat not in my vault
	1. This is some common mistake I see when people trying to explain self play , 
		1. First, they called ficititous splay as a form average of all past version, instead it should be where the population of previous self NN after n steps were created after n battle and model are frozen now instead what you see in vanilla self play
		2. They use abbreviation some is a beginner 
		3. Use the word draw which could been gaming or just picking someting use different word with no overlapping in meaning
	2. Example of weird wording
		1. "This seems wrong "FPuniform over all earlier — Eq. (6)" it what does it mean to be uniforom over all "earlier"
2. For the ELI5 Section in the aspect
	1. Don't reference the Alpha paper I have no read I will plan watch this youtube to understand https://www.youtube.com/watch?v=ro3zpqEmGEc
3. 3 · alphaXiv blog
	1. What is MSS Sovler and what is it user for 
4. 4 · Research — what the survey gives you that the vault doesn't
5. # 5 · Questionnaire — retrieval practice, no peeking