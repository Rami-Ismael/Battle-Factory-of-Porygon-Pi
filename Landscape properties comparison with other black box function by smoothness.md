
2. Usually in bayesian approach the optimizer spaces is smooth  for exmaple eucilcidean distance is natural for many continious tasks; categorical/graoh kernesl use other similaries. however, changing on epsiecies ID is not a small numerical change. Similarityu should reflect legal edit or relevant features; even a one-move edit chan change the strategic role.





# Todo 
- [ ] [[Visual Plan]]
	- [ ] I am planning to write a subsection on what makes competitive Pokémon team building different from the benchmarks used in Bayesian optimisation, starting with non-smoothness. I will test whether a single legal edit can change a team's expected win rate by more than the battle noise. To examine whether local smoothness differs across regions of team space, I will compare neighborhoods around 50 randomly sampled legal Pokémon teams and 50 teams from top tournament placements. Both groups will use the same format, legal-edit sampling protocol, fixed opponent pool, battle policy, and evaluation budget. I will compare the distributions of win-rate changes between each original team and its edited neighbors, accounting for battle noise, to see whether the two groups differ in local sensitivity. I will present the results in a measured figure with error bars and compare them against one combinatorial Bayesian optimisation benchmark and one smooth continuous function, defining an appropriate notion of a small change for each benchmark.
		- [ ] Can you grill me 
		- [ ] Can you recommend a list of visual that help me show the smoothness of the team building expected win rate please starting use exlaidraw to create a wireframe and later we will decide which wireframe is good we will use artfiact using javascript, html and tailwind css and pokemon 
		- [ ] Define what is smoothness
- [x] Why does it matter is smoothes
- [x] Compare the a sample 50 random pokemon team see what is smoothness of hte difference is these
- [x] Compare the top 50 placement pokemon team see what is the level of smoothness 
- [x] I was wondering can I can say the space is ruggedness as see the sample of 50 random pokemon smoothness is different from the top 50 placment smoothness
- [ ] We should ask how do we measure smoothness of a function spaces for the benchmark anyway
# Vocabulary

1. Smoothness
	1. **Smoothness describes how consistently an objective changes when its input changes slightly.** Nearby inputs should usually produce nearby outputs.
2. Local sensitivity
	1. neighboring teams can be have substantially different expeceted win rate
3. **Ruggedness:** 
	1. nearby edits repeatedly reverse whether performance improves, producing local peaks and valleys?

# Writing

1. One make this search space in pokemon is non smoothness. Smoothness describe the how small change input space closely releated. neary by output
2. We use this example in where we change one move in a pokemon moves define how much is the win rate againt top 50 placment team. As we seen, change dramatically can change in value. We can called this non smoothness. Smoothness for function spaces is specific meaning in mathemtaic where nearby output are close with other in neary input. The model is localy sensivitivite. Why does this matter. For machinelearning method we assume smoothness, go in direction in a function does not suddenly transport somewhere else dramatically. It hard to make 
3. Third version
	1. Talk about the visual
		1. As you can see right above, the edit that changes one Pokémon (its species, item, ability, moves and stat spread together) can change win rate a lot, while changing one move or ability barely does. We call this property of the function non-smoothness, or ruggedness. We can call the function locally sensitive.
	2. Define smoothness
		1. Smoothness describes how close the values of the function are at nearby points. There is a stronger mathematical definition.
	3. Why smoothness matters for search
		1. Smoothness matters because machine learning models and methods usually assume it: if you move one step in a direction in the space, the value won't change dramatically, which makes future outcomes easier to predict. Bayesian optimisation guesses a team's score from the teams it has already tried nearby. If one swap can flip the score, those guesses are wrong.
	4. Then we compare to other benchmarks
		1. First, which benchmarks are we collecting? We measure the smoothness of each benchmark where smoothness makes sense and show it as a diagram. Then we show the Pokémon small edit. Overall, VGC is mid-pack. Pokémon swaps alone are rougher than every combinatorial benchmark, but the molecule tasks are rougher still.
