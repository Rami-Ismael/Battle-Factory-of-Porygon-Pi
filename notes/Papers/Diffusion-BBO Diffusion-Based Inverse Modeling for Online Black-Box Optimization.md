- [Diffusion-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization](https://www.semanticscholar.org/paper/d3e0e933576e485274e66e6df452dd0a0eb78765) — 🟢 NeurIPS 2024 BDU workshop, MED ( What that I am reading right now)
	- [x] [[Inverse Modeling]]
	- [x] how should your inverse model choose target scores and learn from new battle evaluations to generate stronger counter-teams
- [x] I am interesteded in doing similar work to such as this paper from this websitse Diffusion - BBO  https://nikkilkuang.github.io/publication/diffbbo/ I think the main the problem with my method i thought beucase i have a team and their win codition i thought it was consider condiiton score function instead it was just classifier free guidances method so I am interested in doing conditioning score inverse model as replacement for the proposer model please use sugbagent like a model tier less than we are using right now please sure the make output of the context is less than 100k if is greater than do compaction . Please use your design the those design skills is for UX and UI and please use animation skills to help also also try to /teach me 
	- [ ] grill me 
- [ ] I am confuse how there method can be called "_inverse-model-based online black-box optimization_." i don't seem their method optimizing anything other to make sure the proposer model which is a diffusion model to have the same function space as the block box functio is the goal is to happen accidently because the model have imperfect information som eatribute will overshot the target that will be the baseline for experiment for this 
- [ ] I was wondering what is their iteration method becasue
- [ ] I think a main problem with my cross entropy method that I keep generate the same thing over and over again and you see the drop in diversity problem I think a method similar to a text diffusion model to have a ban list you are not allowed to generate this this will be fine tuning example with a pretrained text encoder what is the smallest text diffusion model that I can trained it a hour
- [ ] I am interetsed what workshop this paper was part of. The paper in the workshop they got in was BDU in Neurips 2024. I am interested in paper in the workshop please tell me the paper in the workshop which paper will be releavnt in my workshop . So I want you use subagent to gather all the paper, use subagent to determine the paper is releavnt between a score, each scaore fan out decide which paper is releavnt to me and later add to my reading list 
- [ ] Send to opus on the ideas using subagent split the task to complete task. I am plannint to read this paper on the details i am planning intrerestged in bceuase it simlar to what ai am doing in where bayesiaion tiotiunzation in where a difusion model is proposer candidates i strugggle to understand how are they optimizating a function actaully work on this details on the aspect the paper is these Diffusion-BBO Diffusion-Based Inverse Modeling for Online Black-Box Optimization please use the /writing-for-agent when doing subtask 
- [ ] I was wondering what is relax logitsd
- [ ] Why do they do these "how should your inverse model choose target scores and learn from new battle evaluations to generate stronger counter-teams" they got these ideas from another which paper did they go
- [ ] How do I address this problem "**Don’t port the paper’s uncertainty term.** Its “epistemic uncertainty” is how much 5 seed-ensemble models disagree about the length of the sample vector. For your teams that measures almost nothing useful. Uncertainty-based acquisition was also dropped in August, because 73% of your label variance is battle noise. A better replacement for `y − uncertainty` is the measured asked-vs-got curve: pick the ask whose _delivered_ win rate is highest. That is the quantity the paper’s formulaapproximates."
- [ ] I forgot on the detail that i have surrogate model  which is a ridge regression model determine which of the 512 sample the diffusion model is will have a high score we only select 128 of them base on the 128 which has elite score which will be for the sittutioan .531 that was a problem with battle simulation cost to much now it does cost so much will can get the actually score let compare the performance with the surrogate model and with surrogate model and drop the elite fraction also
- [ ] Add another arm where we don't use cap the limit 128 instead pokemon team to battle instead use all the pokemon battle because the battle speed is much faster than it was before than it was ever before now


# How does their method hill cim

1. They use the inveverse model to spread between an ask value which is the highest value recorded so far as the model does have perfect information it will generate sample that will over shot the value hopefully this can be used to hill climb later

# Lesson

- [ ] Classifier guidances


# **What one generation of `cem_v3` does**

1. 2026-10-16,I did not know we finetune the model 40 time, it should be elite only is miss number producing elite only mean value greater than .531 i thought elite been local elite in the generation

- [ ] Ridge Refit
- [ ] Sample
- [ ] 



# Instruction
