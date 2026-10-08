---
created_at: 2026-08-17
updated_at: 2026-08-24
type: problem-statement
status: draft
tags:
  - problem-framing
  - vgc
  - combinatorial-search
---



- I want you to grill my current problem statement each section of version just means there was a certain update current problem statements 
# Version 7


1. Why
	1. My current ADHD interst is pokemon and programming. I want to use my ability in software development in buildig a pokemon that can reach top 100 in a vgc ladder like pokemon showdown for a regulation to display what can I achieve with claude and I think the best approach is to build the best  team by using modern search method to find a pokemon team that win on average than loss over the list of viaable pokemon in the regulation. Afterward, we create a train a policy that is near human level . 
2. Differente layer of the games
	1. Team Building
	2. Selection
		1. 15x15 Zero sum matrix games
	3. Battle
		1. Use a NN policy
3. Tool we are planning to use 
	1.  https://github.com/hsahovic/poke-env
		1. To run simulation on my local computer
		2. To deployment on pokemon showdown 
	2. https://github.com/pmariglia/poke-engine
		1. A Pokemon battle engine that can search through Pokemon states, it currently support single format.
		2.  I can ask claude code to add functionality for double team, mega evoltion and 
		3. Later experiment , if we trained a model on single format does it transfer for double format 
	3. Python Library
		1. PyTorch
		2. Transformers
		3. Pufferlib
4. [[Citation]]


# What version 6

- [x] Explain what type of problem there is two aspect of the problem is create a policy that does well against other policy and team building policy — done 2026-08-22: bilevel — outer is a noisy black-box combinatorial optimisation over teams, inner a two-player zero-sum partially observable stochastic game; team preview is a 15×15 matrix game between them. Detail in chat.



# Version 5


1. Why
	1. My current ADHD interst is pokemon and programming. I want to use my ability in software development in buildig a pokemon that can reach top 100 in a vgc ladder like pokemon showdown for a regulation to display what can I achieve with claude and I think the best approach is to build the best anti-meta team by using modern search method to find a pokemon team that win on average than loss over the list of viaable pokemon in the regulation. Afterward, we create a train a policy that is near human level . 
2. Tool we are using
	1.  https://github.com/hsahovic/poke-env
		1. To run simulation on my local computer
		2. To deployment on pokemon showdown 
	2. https://github.com/pmariglia/poke-engine
		1. A Pokemon battle engine that can search through Pokemon states, it currently support single format.
		2.  I can ask claude code to add functionality for double team, mega evoltion and 
		3. Later experiment , if we trained a model on single format does it transfer for double format 
	3. Python Library
		1. PyTorch
		2. Transformers
		3. Pufferlib
3. Computer
	1. The computer we are using is just my computer which is my current computer 
4. 2026-08-17, the current regulation is M-B this project might take a long time so there will be different regulation but the aspect
5. Policy needs to beat to beat 80 percent of people if want to be 100 top
6. 

- [ ] Why
- [ ] What is the goal
- [ ] What tools are we using


# Version 4

1. What are the essential properties it need to have in the problems statment
	- Why
	- What is the goal
	- How to 
	- Determine who play on the ladder, we use https://github.com/hsahovic/poke-env. I think it supp
	- Battle Policy
		- First we building a random policy as a baseline, then we use the method from metamon and neurips paper to building an powerful agent alive
	- How much pokemon teams are generated
		- There will be one pokemon generated
	- How to determine the evlation
		- We use the [[PokeAgent Challenge (NeurIPS 2025)]] Foul play MCTS given a pokemon team have it compete against each other this will determine how effective is the pokemon team compare to other pokemon team
- SubGoal

2. Why
	1. My current ADHD interst is pokemon and programming. I want to use my ability in software development in buildig a pokemon that can reach top 100 in a vgc ladder like pokemon showdown for a regulation to display what can I achieve with claude and I
3. What is the goal


- [ ] Use claude code skills to determine if the https://github.com/hsahovic/poke-env can do double battles th


# Version three


1. Why I am doing these
	1. My currently niche intersest is pokemon and programming. I want use my ability in software development in build an optimal pokemon team for each regulation to show my programming skills
2. What is the goal
	1. Given a regulation and if you have the meta (the collected list of meta teams) measured on a specific date,return the top-$k$ teams ranked by expected win rate against that meta 
	2. 
3. How to build a pokemon as these search huge is now consider a combinatorical search problem
	1. You need to a pokemon team by accepting the condition of [[Clause or the feature of the search space]] of the rule of building a pokemon team
		1. Look through for each item in the clause to determine what is how big is the search space for legal pokemon team
4. The method to show off this method works effectively is upload the team to pokemon showdown and compete on a ladder
5. What are some subproblem I have to 
	1. I have to build Human-Level Competitive Pokémon 


# Version Two


1. This number of bullet point was created by the grilling me section I want to take deep analysis on the concept of determining that my problem statement is an accruate describtion of my problem for solving [[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]]

2. 
3. What type of problem statements
	1. Inverse Problem
	2. Combinatoric Search Space meaning the search is big for a full scan on a single computer


# Version one

1. I am intersted in competitive pokemon vgc. One advantage in pokemon can i do instead of battling is to team building. I believe this is an inverse problem. The search space finding the optimal pokemon team for a regulation is a combinatoric search. So, given a regulation of pokemon which entail I want produce the most pokemon the on average win the most battle 
