 - [ ] What is the purpose of the cross entropy method
 - [ ] What did we choose to use the cross entrop method instead something else
 - [ ] Please bring citation also 
 - [ ] We should also have a filter for diversity for the cross entropy method
- [ ] Animation for this items
- [ ] Grills me
- [ ] Talk about how did you use cross entropy method
- [ ] Talk about how the initial use of cross entropy method hurty,
- [ ] The main problem with cross entropy method method is that the model learn p(team), fitted to elites only, i trully wanted to get a p(team | score), fitted to every team on the actually real score there is a threshould figure that out for the cross entropy mtehod, instead the theshould is w × best so far, with w picked by the ensemble rule the my understanding the cross entropy method i implementing is a forward appraoch


# Todo

- [ ] 2026-10-16, **Diffusion-BBO moves the ask every round.** It sets it to w × (best score found so far), with w between 0.6 and 1.0. As the best team improves, the ask rises with it, but never above the best.
- [ ] What was the difference between your fix cross entropy method and why did it take so long to realize there is abug in your code "**Standard cross-entropy method.** In Rubinstein’s original method the bar already rises every round. Each round sets the level γₜ to a quantile of the current round’s scores (for example the top 10%) and refits on the samples above it"  the nearest paper to the ideas will be Conditioning by Adaptive Sampling (Brookes, Park & Listgarten, ICML 2019  the slight difference because the elite cutoff and threshould are the same thing which is the askthis is a form of active generation, what is the differnece active gengeation and standard cross entropy method
- [ ] Admit your mistake that I thought elite just mean local elite within a generation you select the top 128 out 512 generation. Instead, what it was before hand it was surroage model select 128 pokemon out of 512 which will perform good the pokemon pokemon team that have an elite greater than the top 50 placement mean average that is the elite threshould, 2026-10-16, the next step because of the improvement of the battle simulation spead we reasonly so 50* 512 samples quickly enough 
	- [ ] **I never said what 0.531 is.** It’s the mean win rate of a top-50 meta team against the other 49 (`code/docs/cem-v3-loop.md:9-10`). So an elite is a team that beats the average meta team. It’s a real, absolute bar, and leaving that out made it easy to read “elite” as “the best of this generation”.



# What is the purpose of the cross entropy method


1. {Talking about the ideas came organically, First, the ideas came organically, I aware the maksed discrete diffusion model trained on rebuilding pokemon from masked pokemon will have the ability given enough data generate a pokemon team. However, the pokemon team it will generate will match the dataset compsotion. I am interested in searching for a pokemon that will have higher win rate than any pokemon team in the dataset. We a model that can generate let say a inititla iwn rate of .4 the goal is above .562. Let, say the function cannot generate totally .41 but it can give good enough result if it generalize well.Therefoew, we ask the model to generate. Let transformer the diffusion model to candidiate proposer. A candidiate poropser will generate base on the inverse score with is win rate. It will get maybe x amount that is atleat at or above. we test the candidiate it propose if above or alteas the training score we should ad the datasets. we can repeat the step multiple time until it reach. There should be limitation of the method,  }, {you did some research the method named was a cross entropy method}


{
The goal of the section will be about the ideas you came up with the ideas naturally let start step with how you come up with the ideas, 
	I aware the maksed discrete diffusion model trained on rebuilding pokemon from masked pokemon will have the ability given enough data generate a pokemon team. However, the pokemon team it will generate will match the dataset compsotion. I am interested in searching for a pokemon that will have higher win rate than any pokemon team in the dataset. We a model that can generate let say a inititla iwn rate of .4 the goal is above .562. Let, say the function cannot generate totally .41 but it can give good enough result if it generalize well.Therefoew, we ask the model to generate. Let transformer the diffusion model to candidiate proposer. A candidiate poropser will generate base on the inverse score with is win rate. It will get maybe x amount that is atleat at or above. we test the candidiate it propose if above or alteas the training score we should ad the datasets. we can repeat the step multiple time until it reach. There should be limitation of the method we can see from the ask and got chart there is limit how much it grow.
	
}
{
Luckly, i don't have Ai psychosis. I quickly ask llm to find the what type of search method is these called. It said it was the cross entropy method. I read the orginal work of the cross entropy method if was the same. I called cross entropy. {Explain why it was good to have Ai psyhcosies and find share vocabulary that the field use to understand what is going on.}
}


# Why did you choose other method method to cross entropy method

1. I don't know other search method that will work as the complex of pokemon. We did enough baseline to compare with lllm, random search and believe this is good enough work for someone to bounces


# **What one generation of `cem_v3` does**

1. 2026-10-16,I did not know we finetune the model 40 time, it should be elite only is miss number producing elite only mean value greater than .531 i thought elite been local elite in the generation

- [ ] Ridge Refit
- [ ] Sample
- [ ] 