
1. [[Membership Query Synthesis]]


# Todo 

- [x] I was wondering the process doing gradient search base where my hyperhuristic was a diffusion model in where i create new pokemon team that was new inputs that was some form of active learning before of not knowing this today 2026-08-29
- [x] Is there research on the topic of using diffusion model to membership synthetic query, I thought I was doing but because I was doing classifier free guidance i was doing the opposite — done: thin. LaMBO-2 is the only discrete closed-loop precedent; nothing in games/decks/teams. Chat 2026-08-29. MQS is defined by provenance, not informativeness — the inversion is wider than the note says.
- [ ] I was wondering on the details what are the open research question that is avaible going to the research direction — done: six questions delivered in chat 2026-08-29
- [x] [[Uncertainty as the acquisition, diffusion as the generator]] — done 2026-08-29: dead as written. Ridge predictive variance is already epistemic-only, so BALD is a monotone transform of the acquisition already run null twice. Live remainder: retarget the information from θ to y*. Report in chat.
- [x] [[Diffusion as candidate proposer in black-box optimization over structured inputs]] — done 2026-08-29: verdict in that note's frontmatter; [[Reading List]] 174-203; mass-lift audit run on the labelled pools, report in chat
- [ ] Rewrite the modified reverse-diffusion sampling rule from Enhanced Low-Density Region Exploration in Classifier-Guided Diffusion Models Through Modified Reverse Diffusion Sampling for the discrete case — the paper shifts a Gaussian reverse step, and the team sampler has categorical logits at masked slots instead — then decide whether it differs from the negated-typicality arm already on the sweep.
- [ ] I would not called "Mine your elites to steer the generator" it was a data collection policy or a model classes I don't know
- [x] I think we are drop Bayesian Active Learin by Disagreement because we learn stop doing active learning because we learn looking for good pokemon team that can beat the meta or finding good example — done: retired in [[Surrogate Model]] item 8 and in [[Todo Section]]. Narrower than "stop active learning": the synthesis loop is still an active-learning scenario, only informativeness as the acquisition dies.
- [ ] Membership query synthesis was abandoned for a reason that doesn't apply to you explain to yourself and read the paper Active Query Synthesis for Preference Learning
- [x] How far off a bad training support can conditional generation be pushed? — answered 2026-08-29: not at all. The regret paper defines X := supp(p₀) and calls it Cromwell's rule; a Gibbs tilt reweights mass, it cannot add support.
	- [ ] Regret Analysis of Guided Diffusion for Black-Box Optimization over Structured Inputs
- [x] Active Learning — done: five answers in chat 2026-08-29
	- [x] What is active Learning — done: a setting, not an algorithm; the goal is label complexity — chat 2026-08-29
	- [x] What is Membership Query Synthesis — done: Angluin 1988, an access model; Baum & Lang 1992 is why it was abandoned — chat 2026-08-29
	- [x] Explain to yourself why you are not doing active learning narrow definition of active learning but still doing membership query synthesis — done: scenario kept, goal changed — chat 2026-08-29
	- [x] What are the axis of active learning — done: scenario / acquisition / goal / batch / labeler noise / theory, each placed on this project — chat 2026-08-29
	- [x] Is membership query synthssis a subfiled in active learning or something else — done: a scenario inside it, inherited from Angluin's query learning — chat 2026-08-29

# What is active learning

1. Is the ideas when you have a list of inputs what to determine which input you label to maximize the initial goal 



# What is the opposite of Membership query synthesis because you were the opposite of that

1. You are method was not active learning but a [[cross-entropy-method elite retraining]]




# Question

## Question 1 
- I was wondering the process doing gradient search base where my hyperhuristic was a diffusion model in where i create new pokemon team that was new inputs that was some form of active learning before of not knowing this today 2026-08-29
- So, you created a synthesis appratus, the main problem that was the wording in you said it was memberhsip query synthesis which is not true because it defined by constructing to be maximally informative. You steered toward predicting win rate the model will give you signal it know well that the area it does not well enough which is opposite objective of maximal information

