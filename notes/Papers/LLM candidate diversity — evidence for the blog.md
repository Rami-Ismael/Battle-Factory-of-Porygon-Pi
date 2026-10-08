# LLM candidate diversity — evidence for the blog

Research date: 2026-09-06. Scope: evidence for a blog motivation, not a demonstration that diffusion beats LLMs at Pokémon team generation.

## What the evidence supports

Repeated LLM sampling can yield highly overlapping candidates in some settings. That motivates measuring search coverage. It does not establish that LLMs cannot generate novel ideas, or that diffusion is inherently more diverse.

## 1. Novel research ideas and repetitive candidate pools can coexist

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto, *Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers* (2024), [primary paper, version 1](https://arxiv.org/html/2409.04109v1).

Section 7.1 examines 4,000 generated seed ideas per topic and reports roughly 200 non-duplicates, with new unique ideas eventually plateauing. Their duplicate criterion is cosine similarity above 0.8 using all-MiniLM-L6-v2 embeddings; it is not literal text identity. Figure 4 averages across topics. This is relevant to generating many candidates and finding diminishing returns from repeated sampling.

Crucially, the same study finds selected AI ideas were judged more novel than expert-written ideas in blind reviews. A repetitive pool can contain individually novel items. Cite this paper for limited *pool diversity*, not an inability to produce novelty.

Transfer limitation: these are NLP research proposals under a particular generation pipeline, not legal Pokémon teams, strategic coverage, or measured battle performance. The duplicate rate cannot be assumed to transfer to our project or newer models.

## 2. Diversity depends on elicitation, not only the model

Jiayi Zhang and colleagues, *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity* (2025), [primary paper, version 1](https://arxiv.org/html/2510.01171v1).

The authors analyze typicality bias in preference data as a cause of reduced diversity after alignment. Their proposed mitigation asks the model for several responses together with their probabilities, rather than one response. They report 1.6–2.1 times greater diversity than direct prompting on creative-writing tasks, with broader experiments in dialogue simulation, open-ended question answering, and synthetic data generation.

This supports concern about repetitive direct sampling, but also shows that a prompting intervention can improve diversity. A fair LLM baseline should therefore include a diversity-oriented prompting strategy, not only repeatedly asking for one team.

Transfer limitation: these reported gains are not Pokémon team results. Verbalized probabilities should not be treated as validated win-rate estimates. Novel wording and semantic variation do not establish distinct competitive strategies.

## Classifier-free guidance is a hypothesis to test

Ho and Salimans, *Classifier-Free Diffusion Guidance* (2022), [primary paper](https://arxiv.org/abs/2207.12598), frame guidance as a tradeoff between sample quality and diversity. CFG is not an automatic diversity-increasing mechanism.

Project inference: explicitly varying strategy or niche conditions could be a way to seek coverage, if those conditions are represented and learned. Conditioning only on high win rate does not specify multiple niches. Measure both variation within each niche and coverage across niches, alongside actual win rates. This remains a proposed experiment, not an established advantage over an LLM.

## Suggested comparison

Compare direct LLM sampling, a diversity-oriented LLM variant, diffusion at several guidance strengths, and copy-and-mutate. Use the same target opponents and battle budget. Track unique legal teams after ignoring team-slot order, species-combination overlap, strategic niche coverage, and the diversity of teams that meet a measured performance threshold. Evaluate winners on fresh battle seeds.
