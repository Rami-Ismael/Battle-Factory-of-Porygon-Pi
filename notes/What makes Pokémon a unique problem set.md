---
type: atomic
tags:
  - vgc
  - combinatorial-search
  - problem-framing
created_at: 2026-08-18
updated_at: 2026-10-07
---

Team-building is a two-layer problem — a 6-of-N species subset, then each species' set (item, ability, moves, nature/EVs, tera) — whose objective is a stochastic, partially observed black box (battles), whose target is a drifting distribution over meta teams rather than a fixed instance, and whose every team is re-scored by a 4-of-6 selection at team preview.

- 'You and opponent make turns simultaneously
	- Your opponents moves affect the effects of your moves (e.g. if you’re KO’ed)'[2][[Citation]]


# Layers in VGC Pokemon

- [ ] Define what are the different layer in vgc pokemon


## Team Building



## Team Selection 4x4 from your 6 pokemon team 




1. In pokemon competitive building, you don't bring all of you pokemon to battle. Intstead, you have to select 4 out of 6 pokemon at the same time while looking at your opponent pokemon team. In pokemon there is two form team preview. One is close sheet format which is the popular format in pokemon champion. In where you can only see your opponent 6 pokemon speciies. However, in open team sheet format which is tournmanet and pokemon shodown ladder, you can see more than 6 pokemon species. You can see the pokemon items, abilities, move and nature. 
2. Let talked about the math behind of team preview. First , let count the number of lead ( the first two pokemon) that will be combinatorics of 6 of 2 multiple by 4 of 2 for the back lead which produce 90 possible pokemon team for team selection. Then there is opponent selecting so there is 90^2 possible game lead. It will be impossible to memorize you have to think generally on a time crunch. 
3. Let use concept from game theory to define what this problem. The games is simulatnenous move games, with still imperfect information, incomplete information


4. Introduction
5. What is team seclection
	1. Team selection is about the ideas in where you have 6 pokemon, you see the opponent 6 pokemon you decide what 4 pokemon you used for battle
	2. You are not allowed to change moves, items and other stuff. 
6. What are the different format of team selection
	1. Open, tournmanet and pokemon showdown 
	2. Close, pokemon champion
7. Show the math behind
8. Game theory word behind these


- [x] Talk about the difference in competitive pokemon battle in pokemon showdown where is popular platform where people usually spend time playing pokemon before pokemon champion becomes an options and pokemon champion. The pokemon showdown follow the same procedure in regular pokemon tournmanet in where there is an open team sheet. You are given the pokemon team, nature, ability, items. I could also talke about the history of competiive pokemon in where over time open team format people are given more information. However, in pokemon champion is a close sheet format where you are given the list of pokemon the person has that is. No preview of nature, abilities, moves. This warp metagames as now becomes a guessing games more like poker than chess also increase variances. https://www.youtube.com/watch?v=mHG8ocKRCL8&t=172s wolfe talked about the details in where because the open team format the better player eventaully reach on top. 
- [x] Count the ways to bring 4 of my 6 and pick 2 leads, as a ladder: 4 of 6, 2 leads, full plan, left/right slots, Showdown's order, both players at once, every legal lead pair in Reg M-B; then decide whether 90 or 180 is the number I quote.
- [x] Write the meta structure of your pargraph for this
- [ ] I am planning to write a section of why make pokemon vgc is bi level optimization the first one is just team preview. I was wondering what visual I should create. I want some more of WACKY ideas. The ideas can will try as many possible to be need I don't care there is more than three options. I would like to truly full of depth os possible for the visual. Make some prototype. I will give you my meta structure prompt of my wiriting and further instruction this should be place in [[Visual Plan]]
- [ ] Work on the writing
	- [x] Fix the speeling and obvious grammar issues
	- [ ]  Ask the model to spot problems in your writing add those items in the bottm list of todo items
	- [ ] Ask what are some checklist can I do to determine the quality of the writing and work effective with my ideas. Please show me a long list of WACKY ideas. I don't care how long it takes to make a list for me. I want to improve on my writing a lot.

### Writing


1. Version 1
	1. Introduction 
		1. In competitive Pokémon you don't bring your whole team: at team preview, each player sees the other's six Pokémon and secretly picks four, two of which lead. How much you see depends on the team-sheet rule.
	2. Define what is to form team preview is
		1. Close Sheet format 
			1. In pokemon there is two form of team preview. The first form is close sheet team format which is the format for pokemon champion which is most popular place where people compete on ladder against each other. With the close sheet format you only see the 6 pokemon specieies.
		2. Open 
			1. In the open team sheet format which is done in pokemon tournmanet and pokemon shodwon, the old place where people compete againt each on ladder. You can see the pokemon held item, abilities, nature and moves. 
		3. As you can see, two difference format one feel like chess and the other feel more like poker. 
	3. Mathematics
		1. The math behind team preview. We have to do combinatorics specicailly combination where order does not matter as there no benefit in left and right position in pokemon battles in double battles. You have combination (6,2) which is 15 then we have (4,2) which is 6 which equal to 90 possible pokemon team. As the game is done simulatnoeysly with imperfect information with time limit there is 90^2 which is 8100 possible battle state in team preview. Thus making it impossible.
2. Version 2
	1. Introduction
		1. Two main things make competitive VGC (Video Game Championships) different from the RPG. First, the games are double battles: you have two Pokémon on your side, and the opponent has two Pokémon on theirs. We call the first two Pokémon in the front the leads. The other two Pokémon are in the back. Finally, the biggest difference: you don't bring all six of your Pokémon to the battle. Instead, there is a team preview, where the two players simultaneously select four Pokémon, two for the lead and two for the back. How much information you see at team preview depends on the format's team sheet rule.
	2. Open
		1. The open team sheet format is used in Pokémon tournaments and on Pokémon Showdown, the old online platform where people compete on a ladder. You can see the Pokémon's held items, abilities, and moves, but you cannot see the nature or EV spread at all. There are so many possible Pokémon and moves that, without the sheet, almost anything could come at you. The open sheet shows you what your opponent actually has, so there are far fewer surprises, and the game plays more like chess.
	3. Closed
		1. Pokémon has a second team preview format, called closed. You can probably guess from the name that information is missing, and you'd be right. Closed team sheets are the popular format for double battles in Pokémon Champions, the newest way to play against people on a ladder. You can see only the opponent's six Pokémon, and nothing more. You have to guess their items, abilities and moves from what is common in the meta, so the game plays more like poker.
	4. How many ways to bring your team to battle
		1. Whatever the team sheet rule, there are 90 ways to bring four of your six Pokémon to battle against your opponent. Let's count how many selections are possible at team preview. First, with combinatorics there are C(6,2) = 15 ways to pick the two leads. Order does not matter, because the two leads go out together. That leaves 4 Pokémon, and we select 2 of them, which gives C(4,2) = 6 possibilities. In total there are 90 possible selections for one player. Equivalently, choose the four you bring, C(6,4) = 15, then which two of them lead, C(4,2) = 6. The 15 alone undercounts because it ignores who leads. This is a two-player game and the opponent also has 90, so there are 90² = 8,100 possible matchups. Both players choose at the same time, without seeing the other's choice, and against a timer. So no player can work through all 8,100 matchups before the timer runs out. You have to pick a selection that holds up against the ones your opponent is likely to choose.


## The battle itself, in game-theory terms

- **Two-player** — exactly two decision-makers, you and the opponent; no third party's choices affect the outcome.
- **Zero-sum** — one side's win is the other's loss; there is no outcome both prefer, so there is nothing to cooperate on.
- **Partially observable** — each side acts on incomplete information: the opponent's items, abilities, EV spreads, remaining moves and back-line are hidden until revealed in play.
- [[Randomness inside in pokemon]]
- 1. Variable Actions – depending on the game state, an agent’s legal actions (e.g. permissible actions defined by the game mechanics) will change based on what pokemon you have out on the field and in your party, and your previous actions (e.g. a Choice Scarf’ed Nihilego can’t select Power Gem the first turn and Sludge Bomb the second). This is a relatively unstudied subfield of RL; very little existing work covers this type of subproblem.
- stochastic 

- [ ] List the game-theory attributes of battling itself 
	- [ ] finite action 
	- [ ] Two player
	- [ ] Partially obversable
	- [ ] Stochastic , randomness in pokemon
	- [ ] Variable Action , State-dependent action sets
		- [ ] What are some example of variable action space those will be ()
	- [ ] Zero sum game
	- [ ] Combinatorial (joint) action space
- [ ] Work on the writing
	- [x] I am wrinting on the details on the aspect in where game theoretoic foundation of battling in pokemon battling in vgc format can you dix the speeling and obvious grammar issues
	- [ ]  Ask the model to spot problems in your writing add those items in the bottm list as checkbox under the list checkbox under the instruction to it so we can keep track in the markdown notes
	- [ ] Ask what are some checklist can I do to determine the quality of the writing and work effective with my ideas. Please show me a long list of WACKY ideas. I don't care how long it takes to make a list for me. I want to improve on my writing a lot.
- [ ] [[Visual Plan]]


### Writing

1. Pokemon battling is zero-sum two playey games. In where two player simulataneous selecting action base on up to 4 moves for each pokemon and switch out to different pokemon. Furthermore, action can be variable for example if a pokemon use a choice scarf which double the pokemon speed however limit to only use 1 move the entire it stayed on the field. In pokemon, there is stochasticy for example a pokemon move  power is multiplied by 85% to 100%. Therefore, there multiple future where a pokemon will be knockout or knockout.
2. Pokemon is a two player zero game. In where two player simulatneous select a finite set of action returning a pokemon, selecting one up to 4 moves of each pokemon. Furh
3. It's common knowledge that pokemon battle is a two player zero sum game with finite action you only select up to 4 moves. There are other stuff the general people just don't know. First, there is stochastic in games [[Randomness inside in pokemon]] generally the assumption the stronger the moves or pokemon there will be something to balance such less accurate move or a terrible abilities. Paritable obseverabtion matter if open or close sheet format. Finally, The action space is also combinatorical as the fastest pokemon go first speed is dynamic can change using these moves trick room or tailwind. 
4. We will talk about the complexity of battling in Pokémon. First, we already know that a competitive VGC battle is a two-player zero-sum game, where players send out their leads simultaneously, revealing which leads they chose at team preview. The finite actions you can take are switching out your Pokémon or using one of the moves from that Pokémon's move pool. The finite action set is also variable: for example, you cannot switch out a Pokémon once at least two of your Pokémon have fainted. The actions are also combinatorial, as the order of moves is based on speed and move priority. Most moves have priority 0. However, other moves have higher priority, meaning they go first even if the Pokémon is slower. The most common example of a high-priority move is Fake Out, which can only be used on a Pokémon's first turn on the field and stops the target from doing anything. Finally, there are the sources of randomness in Pokémon. I will list each source of [[Randomness inside in pokemon]] with some visuals. 
5. Version 5
	1. Introduction
		1. We will talk about the complexity of battling in VGC Pokémon.
	2. What type of game is Pokémon?
		1. First, we already know that a competitive VGC battle is a two-player zero-sum game, where players send out their leads simultaneously, revealing which leads they chose at team preview.
	3. What are the possible actions you can take?
		1. The finite actions you can take are switching out your Pokémon or using one of the moves from that Pokémon's move pool of up to four moves. Each move has a priority. Most moves have priority 0, which means the order of attacks is based on the speed of the Pokémon.
			1. Switching
				1. The reason why you want to switch out
			2. Move
				1. Priority
					1. Most moves have priority 0. However, other moves have higher priority, meaning they go first even if the Pokémon is slower. The most common example of a high-priority move is Fake Out, which can only be used on a Pokémon's first turn on the field and makes the target flinch, so it cannot act that turn.
	4. Then we talk about how actions can be reduced: variable action spaces
		1. The actions you can take can be limited by the game state. For example, choice items lock you into one move only (Choice Scarf multiplies Speed by 1.5; Choice Band and Choice Specs multiply Attack and Special Attack by 1.5), Taunt stops the Pokémon it hits from using any non-damaging moves, and a move can run out of PP.
	5. Then we talk about the combinatorial nature of choosing for two Pokémon
		1. The action space is combinatorial because you pick actions for _two_ Pokémon at once (each move with its target, or a switch), so the choices multiply into dozens. Three common styles show the idea: aggressive, where both of your Pokémon hit one opposing Pokémon; balanced, where each of your Pokémon attacks a different one; or passive, where none of your Pokémon attack your opponents.
	6. Randomness
		1. Finally, each Pokémon game is a stochastic environment. [[Randomness inside in pokemon]] 
# Search Perspective and objective function  how different from other search spaces for optimization 


1. [[Objective Function comparison with other black box function by Ill-conditioning]]
2. [[Search Space Comparison with other black box function by variable type]]
3. [[Evaluation properties with other black box function by evaluation cost and budget]]
4. [[Evaluation properties with other black box function by a meta drift over time (non stationay target)]]
5. **Search-space properties:** what a team is, before any win rate.
6. **Evaluation properties:** how _f_ can be observed.
	1. [[Objective Function Comparison with other black box function by noise]]
7. **Landscape properties:** the shape of _f_ over the space, given an edit operator.
	1. [[Landscape Properties comparison with other black box function by high order interaction]]
	2. [[Landscape properties comparison with other black box function by smoothness]]
	3. [[Lanscape Properties comparison with other black box function by non-stationaery relevance]]
	4. [[Objective Function Comparison with other black box function by non global optimal]]
	5. Non-separability / epistasisx, Indeed, these interactions, known as epistatic in protein evolution, violate the underlying sequential factorisation of AR models, which commits to each token before observing its downstream context.
		1. Do good components combine into good solutions?A member’s value depends on its teammates. A support move may be useless without its intended partner; combining individually strong Pokémon need not produce a strong team.


Independent list, built from the benchmarks and not from the points above: [[bo-benchmarks-vs-vgc-full-team-search-unit]].








# Todo 

- [x] There is three layer to the vgc pokemon — yes, but nested, not parallel: each layer is scored by the value of the layer below it
- [x] /goal I was wondering is possible to determine you can split competiive vgc as three layer problem 1 is team building, second team selectionand third is battle itself if so that is an accruate if there alternative way please tell me if not then please write some more checkbox in the todo Section in [[What makes Pokémon a unique problem set]] to fill in the headers please and thank you /research — done 2026-08-24: accurate but nested, not three separate problems; alternatives and the 90×90 correction in chat, checkboxes below 
- [ ] Fix the problems found in Version 5 of the battle writing.
	- [ ] 1. Fill the empty Switching subsection with why you switch out.
	- [ ] 2. Replace the intro sentence with the claim the section makes.
	- [ ] 3. Add partial observability; Version 5 skips it.
	- [ ] 4. Explain priority once; sections 3 and its Priority subsection repeat it.
	- [ ] 5. Move Fake Out to variable action spaces, its clearest example.
	- [ ] 6. Fill the burn and paralysis placeholders and add consecutive Protect.
	- [ ] 7. Decide whether Mega Evolution counts as an action in Champions.
