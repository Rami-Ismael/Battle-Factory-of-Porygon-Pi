- [x] I create the structure of the paragraph i want to write for this section now , i need to fill with sentence and evidence for my claim. 
- [ ] Work on the current hyptothesis section in the paragraph
	- [ ] Before starting this project, my hypothesis was that a diffusion model could build competitive Pokémon teams for the current meta. Diffusion models had already been steered toward high-scoring designs in black-box optimisation — with classifier-free guidance and reweighted training data (DDOM, Krishnamoorthy et al., ICML 2023 🟢) and by fine-tuning on high-scoring samples (DiBO, Yun et al., ICML 2025 🟢) — and I expected that to carry over to team building.. By "build" I meant generating a novel team whose win rate against the top-placement teams is higher than the win rate those teams have against each other. The system needed a few pieces, but I believed the most important was guidance: steering the diffusion model toward teams that win against the current meta, with classifier-free guidance or another steering method. In this post I show what I tried, and I reveal the results in the conclusion.
		- [ ] grill me 
			- [x] What does novel means in the situation
			- [ ] Which guidance do you mean?
				- [ ] Explain classifer free guidance and how are you using in these example where did you get the ideas of classifer free guidacnse
- [ ] Can you grill me on the quality of my hook
- [ ] Go each each section of my initial hesitation of using llm to these work instead use a diffusion model
- [ ] https://www.youtube.com/watch?v=DOeVsVUuX4M

# Current Writing

1. My hooks is 
	1. I think the most common question that I expect to receive why don't use prompt an LLM and building an entire pokemon team building agent construction in 2026 did you know diffusion model are not relevant anymore.
2. Provide some problem with the current LLM approach
	1. Talked about it struggle with 
		1. Diversity
			1. My biggest concern use a huge LLM such as deepseek, OpenAI and my favorite llm Opus 5.5 model struggle diverse output[citation]. I strongly desire to see variety of new pokemon team that underexplore or not use frequential. I want all pokemon reach it true poentialand find unique team st
			2. My biggest concern with large language model is the model to not generate diversity set of teams instead use the top meta threat. As the number of possible pokemon in meta is huge. I want all pokemon reach it true potential and find unique strategy and create a diverse meta games.As we seen in the current pokemon world champion, we saw Mega Dragonite won world champion.
			3. I am will be listing the reason why I choose diffusion approach instead of llm a
			4. I will be discussing why the problem os using a llm and choose not to use even do llm i use daily. First reason is seem there to be a cap of diversity in llm generated output[citation] and it seem RLHF as effect on that. I want to my tool to enhance the meta games in where we see diverse set of pokemon team that are capable. I feel like there is unexplore pokemon team and strategy. As we seen in this yearpokemon champion, mega dragonite won the world champion even do the usage rate of low. 
			5. Evidences
				1. Kirk, Mediratta, Nalmpantis, Luketina, Hambro, Grefenstette, Raileanu — _Understanding the Effects of RLHF on LLM Generalisation and Diversity_, **ICLR 2024**. 🟢 [arXiv 2310.06452](https://arxiv.org/abs/2310.06452) · [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/5a68d05006d5b05dd9463dd9c0219db0-Paper-Conference.pdf). Reinforcement learning from human feedback "substantially reduces output diversity" both per-input and across-input — the authors name it mode collapse — and they frame it as an inherent generalisation↔diversity tradeoff.
				2. - Padmakumar & He — _Does Writing with Language Models Reduce Content Diversity?_, **ICLR 2024**. 🟢 [arXiv 2309.05196](https://arxiv.org/abs/2309.05196). Controlled experiment: the feedback-tuned model produced a statistically significant diversity reduction; **the base model did not**. That nuance matters — the defect is instruction tuning, not language models as such.
		2. Bias
			1. Second thing is biad, {define what is bias}, then llm struggle with bias[1] on the pretrained dataset. I want novel pokemon team and plus pokemon company is always adding new pokemon. 
			2. - _Pretraining Exposure Explains Popularity Judgments in Large Language Models_, **SIGIR 2026**. 🟢 [arXiv 2605.12382](https://arxiv.org/html/2605.12382). Popularity judgments track pretraining exposure, effect strongest in larger models, persists in the long tail. **I've only seen the abstract — read it before you quote it.**
		3. Exploration
			1. Third, llm model still struggle with exploration require a lot of human intervention[citation]. I don't what make a harness and the search space is huge. 
			2. - Krishnamurthy, Harris, Foster, Zhang, Slivkins — _Can Large Language Models Explore In-Context?_, **NeurIPS 2024**. 🟢 [arXiv 2403.15371](https://arxiv.org/abs/2403.15371) · [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/d951f73c521d069fefbb73396df01424-Paper-Conference.pdf). Models "do not robustly engage in exploration without substantial interventions" — only one configuration out of many worked, and it required an externally summarized history fed in as sufficient statistics. This is your strongest citation. _Caveat:_ it's a multi-armed bandit study, and you ruled that family out on 2026-08-20. Cite it as evidence about language models, and don't let the bandit framing walk back in.
		4. Hitting Context limit
			1. Let say, I build a llm to build harness. It will be huge chunk of text where llm sturggle to long context depensnce with each other [1]. This is entire sub field of harness engineer maybe it can be another future project.
			2. - Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang — _Lost in the Middle: How Language Models Use Long Contexts_, **TACL 2024**. 🟢 [ACL Anthology](https://aclanthology.org/2024.tacl-1.9/). U-shaped curve — retrieval degrades sharply when the relevant item sits mid-context.
			3. Hsieh et al. — _RULER: What's the Real Context Size of Your Long-Context Language Models?_, **COLM 2024**. 🟢 [arXiv 2404.06654](https://arxiv.org/html/2404.06654v1). Effective context length is substantially shorter than advertised once the task is harder than needle-in-a-haystack.
3. Hypothesis



# Guidance means to me
