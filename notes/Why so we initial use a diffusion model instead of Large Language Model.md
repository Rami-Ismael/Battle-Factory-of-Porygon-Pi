- [x] I create the structure of the paragraph i want to write for this section now , i need to fill with sentence and evidence for my claim. 
- [x] Work on the current hyptothesis section in the paragraph
	- [x] Before starting this project, my hypothesis was that a diffusion model could build competitive Pokémon teams for the current meta. Diffusion models had already been steered toward high-scoring designs in black-box optimisation — with classifier-free guidance and reweighted training data (DDOM, Krishnamoorthy et al., ICML 2023 🟢) and by fine-tuning on high-scoring samples (DiBO, Yun et al., ICML 2025 🟢) — and I expected that to carry over to team building.. By "build" I meant generating a novel team whose win rate against the top-placement teams is higher than the win rate those teams have against each other. The system needed a few pieces, but I believed the most important was guidance: steering the diffusion model toward teams that win against the current meta, with classifier-free guidance or another steering method. In this post I show what I tried, and I reveal the results in the conclusion.
		- [x] grill me 
			- [x] What does novel means in the situation
			- [x] Which guidance do you mean?
				- [x] Explain classifer free guidance and how are you using in these example where did you get the ideas of classifer free guidacnse
- [ ] Can you grill me on the quality of my hook
- [ ] Go each each section of my initial hesitation of using llm to these work instead use a diffusion model
- [ ] https://www.youtube.com/watch?v=DOeVsVUuX4M
- [ ] Explain what is guidance that will be one that requise explanation
- [ ] Explain what is negative prompting that will be on the require explanation
- [ ] Explain what are the different sampling method like method planning

# Todo

- [ ] Go to each section of the inititial hesitation to use large language model for these explain the problem with each method
# Current Writing

1. My hooks is 
	1.  Before starting this project, my hypothesis was that a diffusion model could build competitive Pokémon teams for the current meta. Diffusion models had already been steered toward high-scoring designs in black-box optimisation — with classifier-free guidance and reweighted training data (DDOM, Krishnamoorthy et al., ICML 2023 🟢) and by fine-tuning on high-scoring samples (DiBO, Yun et al., ICML 2025 🟢) — and I expected that to carry over to team building.. By "build" I meant generating a novel team whose win rate against the top-placement teams is higher than the win rate those teams have against each other. The system needed a few pieces, but I believed the most important was guidance: steering the diffusion model toward teams that win against the current meta, with classifier-free guidance or another steering method. In this post I show what I tried, and I reveal the results in the conclusion.
2. Provide some problem with the current LLM approach
	1. Talked about it struggle with 
		1. [[One reason for using diffusion over llm model is diversity]]
		2. Bias
			1. Second thing is biad, {define what is bias}, then llm struggle with bias[1] on the pretrained dataset. I want novel pokemon team and plus pokemon company is always adding new pokemon. 
			2. The second thing I worried about using Large Language Model was trained on the entire internet. It will have a bias[citation] toward to most popular pokemon. This could lead a model where I have to put in a lot of work to make sure there is diverse sample. Instead, I build my own model in where I can control the datasets which can be done quite resanlbae.
			3. - _Pretraining Exposure Explains Popularity Judgments in Large Language Models_, **SIGIR 2026**. 🟢 [arXiv 2605.12382](https://arxiv.org/html/2605.12382). Popularity judgments track pretraining exposure, effect strongest in larger models, persists in the long tail. **I've only seen the abstract — read it before you quote it.**
		3. Exploration
			1. Third, llm model still struggle with exploration require a lot of human intervention[citation]. I don't what make a harness and the search space is huge. 
			2. - Krishnamurthy, Harris, Foster, Zhang, Slivkins — _Can Large Language Models Explore In-Context?_, **NeurIPS 2024**. 🟢 [arXiv 2403.15371](https://arxiv.org/abs/2403.15371) · [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/d951f73c521d069fefbb73396df01424-Paper-Conference.pdf). Models "do not robustly engage in exploration without substantial interventions" — only one configuration out of many worked, and it required an externally summarized history fed in as sufficient statistics. This is your strongest citation. _Caveat:_ it's a multi-armed bandit study, and you ruled that family out on 2026-08-20. Cite it as evidence about language models, and don't let the bandit framing walk back in.
		4. Hitting Context limit
			1. Let say, I build a llm to build harness. It will be huge chunk of text where llm sturggle to long context depensnce with each other [1]. This is entire sub field of harness engineer maybe it can be another future project.
			2. - Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang — _Lost in the Middle: How Language Models Use Long Contexts_, **TACL 2024**. 🟢 [ACL Anthology](https://aclanthology.org/2024.tacl-1.9/). U-shaped curve — retrieval degrades sharply when the relevant item sits mid-context.
			3. Hsieh et al. — _RULER: What's the Real Context Size of Your Long-Context Language Models?_, **COLM 2024**. 🟢 [arXiv 2404.06654](https://arxiv.org/html/2404.06654v1). Effective context length is substantially shorter than advertised once the task is harder than needle-in-a-haystack.
3. Hypothesis



# Guidance means to me


# Prompt for claude code
@Why so we initial use a diffusion model instead of Large Language Model
1. The [citation] is suppose placement for the future llm model to place the citation below is the citation for it please and thank you 