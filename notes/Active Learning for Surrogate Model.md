---
created_at: 2026-08-29
updated_at: 2026-08-29
protocol: vgc-team-generator-pilot/docs/self-training-loop-protocol.md
tags:
  - diffusion
  - self-training
  - model-collapse
---
- I was doing pool base active learning 
# Version three

2026-08-29 — wrong title, wrong answer. I wanted diffusion as the gradient-based search, HPS teams as fuel; Fable answered collapse-stopping. The gap I left: HPS teams are non-competitive — pool 0.015 vs meta, real teams 0.491, oracle best-20-of-2000 just 0.110.

# Version two

1/. 2026-08-29 the thing we we cannot use [[hierarchical product sampling]] to create a dataset that because the number of possible pokemon team is incredibly small if compare all possible pokemon team. The old method of creating the datasets I think was the copy paste in was in these claude session Diffusion model as VGC team generator. I am worried the pokemon. I think alternative appraoch for now we start using the current meta pokemon we add n random perturbnation to until it become a valid pokemon team. I feel like i am reinventing evolatonary strategies and algorithms. We can trained a surrogate model to determine how well these two pokemon team match up.

# Version 1
- /goal 2026-08-29, we the current base method to generate random legal pokemon using diffusion model as search tool. After, it create a pokemon that does not exist in the dataset add to the datasets. The retrained the diffusion model.  The question what is stopage. We need evlaution to make sure the diffusion keep working. 

---
Answer 2026-08-29 — full protocol in the repo, `docs/self-training-loop-protocol.md`. Stop on plateau (frozen-holdout loss and new-species-set yield both flat for 2 generations), not on validity — collapse keeps every team legal. Six verified papers in [[Reading List]].

Ran 2026-08-29: stopped at generation 2, S1 + S0. All 19,663 admitted teams sit within HPS's own spacing (Hamming radius 41), so the loop is a slow HPS; validity rose 0.477→0.535 (concentration alarm), coverage intact. Table in the repo doc.