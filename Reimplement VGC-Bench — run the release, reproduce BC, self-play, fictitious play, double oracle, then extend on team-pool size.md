---
created_at: 2026-08-20
updated_at: 2026-08-23
type: work-list
paper: "[[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]]"
repo: https://github.com/cameronangliss/vgc-bench
relevance: HIGH — the battle policy starts from this
tags:
  - vgc
  - battle-policy
  - reproduction
---

# Goal

1. The goal of this task is to Rami me is reimplement this [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] from scratch and hopefully get better performance than the orginal author as this will be useful in building a battle policy and having model can find an optimal number of 
# ToDo
- [ ] I want to create a named for the project is understandable for pokemon van is does word play and reference pokemon lore for my project of currently creating a battle policy that is base on VGC Bench right the top competition is Frontier Brain ,  Porygon-π , Frontier Factory.
- [ ] Compare the difference in performances have the unknown value be for example ev, nature and iv spread to what is the current baseline while compare to  [[Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Papers/Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers|Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]] does try it best to speculate 

# Recrete their dataset

- [ ] Recreate their datasets for VGC Bench the below is my prompt that I am planning to feed into a llm 
- [ ] When writing you own code you should atleast use the improve-codebase-architecture skills
	- [ ] For example improvment I notice , 
		- [ ] Looking at the code that get the data determine is there a better software design practice also to do better using the improve-codebase-architecture skills I see much better improvement for code quality [`scrape_logs.py`](https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/scrape_logs.py) FORMAT constant list should have list of UPPERCASE FORMAT constant 



## Prompt for "Recreate their datasets for VGC Bench the below is my prompt that I am planning to feed into a llm " 

- I think we need to find battle double battle demonstration for the current regulation on the pokemon showdown how did [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] and [[Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Papers/Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers|Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]] did this aspect I don't remember. It should be more clear the second paper it only does single battle. When you reference code from a link the code link from the github repo so it easier to follow. How do get the EV, IV and nature of the oponent pokemon. Maybe we need to check on [pmariglia/foul-play](https://github.com/pmariglia/foul-play) method or [[VGC AI Competition (IEEE CoG)]]. Make sure don't use abbreviation. When creating variable name lj is bad compare log_json
	- My current Understand of implementation details
		1. [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]]
			1. The code responsbile for scraping the the log , the log is our target 
				1. Q:What is a log? 
				2. Q:What does it lok like 
			2. Filter Process
	1. My understanding for the [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] implementation for behavior cloning they get the pokemon showdown replay buffer from Open Team Sheet. 
		1. My understanding of scraping The 
		2. The filter they apply 
			1. It also discards Zoroark cases because the disguised Pokémon make reconstruction unreliable. This is implemented in [`scrape_logs.py`](https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/scrape_logs.py).
		3. What data format do they store the ?
		4. Answered 2026-08-23 — the sheet is all non-stat info, but Champions also reveals Nature ([`sim/battle.ts`](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle.ts#L3183)); Individual Values are fixed at 31, so only the Stat Point spread is hidden — VGC-Bench masks opponent stats to `-1` ([`policy_player.py`](https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/src/policy_player.py#L420)).
