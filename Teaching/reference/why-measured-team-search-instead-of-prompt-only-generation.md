# Why measured team search instead of prompt-only generation?

Research and interpretation: 2026-09-16. This note separates the author's inferred motivation, recorded project results, and external evidence. It does not establish that diffusion beats LLMs for VGC.

## What the project notes suggest

The most defensible interpretation is that the project aims to learn and control a distribution of legal, strong candidates, with improvement grounded in battle evaluations. Prompt wording alone is not the experimental object the author wants to study.

The note [[Why so we initial use a diffusion model instead of Large Language Model]] explicitly asks for novel, diverse proposals and steering toward performance against the meta. [[Diffusion as candidate proposer in black-box optimization over structured inputs]] specifies experiments varying training sources, guidance, generation temperature, regenerated team members, elite selection, and replay. These are evidence that controllable exploration and measured adaptation are central to the project. This is an interpretation of the notes, not a claim about the author's unspoken motives.

Three concrete motivations follow:

1. **Coverage across candidates.** One plausible team is insufficient if a large batch repeatedly explores the same roster or strategy. Familiar-team concentration could reduce opportunities to discover useful alternatives. This is a hypothesis to test for a particular LLM and prompting setup, not an established Pokémon-specific result.
2. **Objective-aligned feedback.** Competitive value is expected performance against declared opponents under declared pilots. A convincing explanation of a team is not evidence of that value. A generator should be evaluated and updated through a search loop. An LLM can also participate in such a loop.
3. **Experimental control.** A locally trained generator offers explicit choices of corpus, representations, sampling, and updates. These suit the proposed ablations. They are not exclusive to diffusion: open-weight language models can also be trained and controlled, and prompt-based methods can be reproducible when their configurations are recorded.

Avoid saying the goal is an unbiased generator. The notes deliberately start from real teams and favor stronger candidates. The goal is to use useful prior knowledge while measuring and limiting unwanted concentration.

## Diversity is not a model-family guarantee

[[Metrics for Diversity]], in its 2026-09-08 baseline section, reports 498.3 distinct compositions per 512 draws before fine-tuning, compared with 172.0–276.3 in final CEM arms, while exact team uniqueness remains 100%. These are recorded project results; this research pass did not rerun their audit.

They illustrate why exact uniqueness, novelty relative to a corpus, composition coverage, and strategic diversity must be distinguished. Changing one EV or move can create a new complete-team record without establishing a different strategy. Novelty relative to a fixed corpus also does not prove worldwide originality. For an opaque LLM training corpus, report novelty relative to the accessible reference, not novelty relative to its unknown training data.

The observed concentration belongs to this generator-and-selection pipeline. It does not isolate diffusion architecture as its cause. Nor does concentration alone prove search failure: exploitation can be useful if performance improves. Preserve and report quality alongside coverage.

## External evidence and its limits

- [Jentzsch and Kersting, 2023, WASSA](https://aclanthology.org/2023.wassa-1.29/): over 90% of 1,008 generated jokes repeated the same 25 jokes. This corrects the recollection that ChatGPT could only generate five jokes. It concerns an early ChatGPT system and joke prompts, not contemporary Pokémon optimization.
- [Doshi and Hauser, 2024, Science Advances](https://doi.org/10.1126/sciadv.adn5290): AI-assisted short stories received improved creativity evaluations but became more similar across writers. This supports a concern about collective output diversity, not a claim about VGC team generation.
- [Meincke, Nave and Terwiesch, 2025, Nature Human Behaviour](https://www.nature.com/articles/s41562-025-02173-x): ChatGPT-assisted brainstorming improved average idea creativity while reducing diversity across the idea pool. Individual output quality and batch diversity are different outcomes.
- [Liu et al., 2024, LLAMBO](https://proceedings.iclr.cc/paper_files/paper/2024/hash/84b8d9fcb4e262fcd429544697e1e720-Abstract-Conference.html): LLMs improved components of Bayesian optimization, including candidate sampling, on the authors' HPO experiments. This is relevant counterevidence to a blanket dismissal of LLMs as search components; it does not establish VGC performance.
- [Meincke, Mollick and Terwiesch, 2024 preprint, Prompting Diverse Ideas](https://arxiv.org/abs/2402.01727): GPT-4 product-idea pools under several prompts were less diverse than human-group pools, but prompt engineering substantially improved diversity; their chain-of-thought condition approached human-group diversity. Prompt-only generation is not a single fixed baseline, and prompting is not necessarily ineffective.

No source above establishes that current LLMs cannot generate novel competitive teams, or that a diffusion model is more diverse at a matched performance level. A useful direct comparison would freeze the regulation, validator, opponent population, pilots, and battle budget; record generation attempts, legal yield, duplicates, composition frequencies, corpus novelty, and fresh-game performance. Compare an LLM proposer both with and without evaluation feedback against the learned proposer and a legal mutation baseline. Log compute and account for different amounts of available prior information.

## Suggested blog wording

I chose to investigate a learned team generator because I want to control how it explores and how it changes after battle feedback. My concern with relying on prompts is that a batch of plausible teams could repeatedly return to familiar compositions, even when every output looks different. Research on AI-assisted writing and brainstorming makes that concern worth testing, but does not establish it for Pokémon. Diffusion is my proposed approach, not a guarantee of diversity: I need to measure legality, composition coverage, novelty, and battle performance together.
