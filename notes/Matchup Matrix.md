
Goal: Design a matchup matrix of any legal team, stored sparsely for a chosen VGC format. Rows and columns represent teams; each cell stores the row team's estimated win rate against the opposing column team under fixed battle policies. Record battle counts, leave untested matchups unknown, and assess storage and simulation costs for full coverage. We will use a SQlite database to store the record. The first seeding of pokemon teamsis contain existing tournaments teams from vgcpaste

Purpose: (a) a cache of the objective function, so no pair of teams is battled twice, and a team's score is its row averaged over the top-50 meta columns; (b) training labels for the surrogate model.


# Todo


- [x] How do we store this matrix in numpy, a database for matrix, text file
- [ ] Determine what will be the compression algorithm to fit all the data 
- [ ] Determine if we should probably use bayesian approach for this PyMC
	- [ ] grill me 
- [ ] Do we need to compress the matrix how much compressing the loseless
- [ ] redo this work base on different skills
	- [ ] Use the grilling skills to understand the goal
	- [ ] Create an interactive matchup heatmap showing estimated win rate, battle count, and uncertainty, with untested cells visibly distinct.
- [ ] # Spatial Data Structures[](https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/OtherSpatial.html#other-spatial-data-structures "Link to this heading")
- [ ] How do we share the database with the entire website
- [ ] Full-coverage storage and simulation costs
- [ ] What is our battle policy
- [ ] How should we spend the battle budget?
	- [ ] I think we gonna have a surrogate model given a model given to pokemon team decide what is win rate, i think we are gonna use memebrship query synthesis base on uncertainty the most select. I think we could use the ideas we have a **Score = predicted win rate + β × uncertainty** we select has a plausible change of beating our stongest counter. 
		- [ ] Where the synthetisi team comes froms
			- [ ] I think the ideas it should comes fromes diffusion model you worked or tournament. I think we could make an experiments to see the difference in the performances. 
- [ ] Can tournament teams be modified?
	- [ ] CLearly the answer is no 
- [ ] Is there any research on the topic of people use the concept of hypernetwork to generaet a huge matrix
	- [ ] grill me
- [ ] This is part of the matchup matrix idea in the section I was wondering can you determine the strength of a pokemon team againt all possible pokemomn combination which is impossible to generate all number of all possible pokemon team b/c can there be an ideas in where scalable approach in giant matrix cascading or something
- [ ] Determine if this should a seperate repo or not


