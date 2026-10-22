- [ ] What can be the baseline of the project
	- [ ] [[How did we create the top 50 placement pokemon team we consider top meta threat to atleast counter and show my method is effective]]
	- [ ] Build two baselines, each producing a new list of teams from the top-50 placement teams. LLM edit: give an LLM a random team from the top 50 and prompt it to edit the team so it can compete against the current meta. Random legal mutation: take a random team from the top 50 and apply random mutations that keep it legal. Compare the two lists against each other and against the teams they started from.
		- [ ] grill me

# ToDo

- [ ] Find out whether a language model builds better Reg M-B teams when it explains its choices before filling in a team. **Task:** the Whole Pokémon fill-in from [[What is the baseline]], with 1, 3 or 6 slots blank, starting from random legal teams. **Change:** every prompt version (original, regulation reference, DSPy, future ones) gets an explanation twin that adds a JSON `explanation` field before the team. The field is a checklist drawn from [Wolfey's report](https://nuggetbridge.com/reports/eggscelent-execution/): core, each Pokémon's role, why each move/item/ability/Stat Alignment, Stat Point benchmarks, threats answered, rejected alternatives, and a bring-4 and lead plan (saved, not forced in battle). **Same in both twins:** hidden reasoning off, 16,384-token cap. **Models:** Ling, Mercury, GPT-6 Luna. **Measure:** win rate over 50 battles against the top-50 pool, compared where both twins are legal. Legality and how complete the checklist is are reported, never used as a gate. **Sample:** 40 starting teams × 3 repeats, run as a 20-team pilot and then 20 more, all drawn from one seed.
- [ ] Determine the right claude artifact to store the baseline I want to do research on the concept of using diffusion model as a mutation operator compare to random or heuristic approaches 
- [ ] Text Diffusion Model 
	- [ ] I want a near exact copy of the baseline we applied to llm baseline example /recall this information you we finetune a local text diffusion model on we compare the performance on the heldout none top 51-100 pokemon team 

# Work on the baseline

- [x] Determine what llm model we are planning to use, let start with the cheapest first
- [x] What gateway are using

- [[Replace the scratch model with a pretrained text diffusion model finetune it to do your task]] joined the Mask Lab grid on 2026-10-15: text writer 149/270 legal, level with the scratch decoder where legal, +7 pp on whole-Pokémon fills. Table in `code/docs/textdiff.md`.

## Decision Model Baseline

- [ ] We will be using the teask and use the same pokemon team as seen in llm baseline with dspy , ling, and the reasoning llm model we will ask the decision model to select a pokemon one at a time, then abilities,ite,move, and etc with a good prompt to make sure we tried our best to make i ifst need to see the propmt before I get go ahead after battling the also compare if jev select all pokemon moves and legal pokemon move,
- [ ] We will be testing the ability of decision model to to complete the baseline task fill in the blank.Because, a decision model can select answer from categories we have give categories for each section. one interesting experiment we are testing is that have happen if give all possible even that will break legality and compare to one don't brek legality. Please come up with any more wacky idea you can have.
## Random Search Baseline

- [ ]  Given a an empty slot randomlety selected a valid answer to put then we see the win rate percetange against the top meta team  did it improve or decrease and how much 
## LLM baselines


### Todo LLM baselines

- [ ] Grill me on the details on this aspect 

**Approach:** masked completion. The LLM fills only the empty parts and nothing else may change.
### LLM model we are using for now is



OpenRouter price per million tokens, cheapest first (2026-10-08). Grid input cost = 8,100 calls × 14k tokens, output excluded:

| #   | Model                         | Input $ | Output $ | Grid input $ |
| --- | ----------------------------- | ------- | -------- | ------------ |
| 1   | inclusionai/ling-3.0-flash-vl | 0.021   | 0.0616   | 2.38         |
| 2   | inception/mercury-2.5         | 0.04    | 0.15     | 4.54         |
| 3   | openai/gpt-6-luna             | 0.10    | 0.50     | 11.34        |
| 4   | deepseek/deepseek-v4.1-flash  | 0.30    | 1.20     | 34.02        |
| 5   | anthropic/claude-sonnet-5     | 2.00    | 10.00    | 226.80       |

### LLM gateway we are using


- [x] put in 10 dollars

1. OpenRouter

### Prompt to feed to the model to make their decision

- [ ] Ask the llm to feed into the prompt
- [ ] Determine if there anything else I should add to the prompt to feed into the model 

1. Instruction
2. Meta Team
3. Desired output is a JSON

### Build the Pipeline

- Create a list of randomly generated Pokémon teams or select 50 high-placing Pokémon teams.
- Randomly mask certain features based on the task we are interested in. Keep all other parts of each team fixed.
- Feed the masked team, instructions, and meta teams into the LLM or diffusion model, then collect the completed team in JSON format.
- Validate the output to ensure the completed team is legal and all unmasked features remain unchanged.
- Check whether our database already contains matchup results for the completed team against the meta teams under the same battle conditions.
- If matching results exist, reuse them. Otherwise, run a battle experiment and store the results in our database.
- Compare each method’s performance using the same starting teams, masks, opponents, and battle conditions.

### What is the expected cost

1. One team in Showdown paste format averages 901 characters, which is about 225–300 tokens. All 50 meta teams come to about **11k–15k tokens**, so you were right that it's well under 30k. Add ~1k for instructions and the masked team, and each call is about **14k input tokens**. Writing the meta in JSON instead of paste format makes it roughly 1.5× bigger.
2. Full grid: 6 tasks with these k values (stats, items, abilities and natures 1–6, whole Pokémon 1–6, moves 1–24) is 54 cells. 54 × 50 starting teams × 3 completions = **8,100 calls**.
### Create a visual for me








### Starting pokemon team 


1. Just a random pokemon 
2. One of the top 50 placement pokemon 

### **Scoring**

- The behaviour-cloning policy plays both sides, with 50 battles per completion and the same seeds for every method.
- Battles are stored in `~/vgc-data/matchup_regmb.sqlite` with origin `llm-completion:<run>`.
- 192 battles only for any single team presented as a result.







# Tasks we are interested in 

- [ ] Create a list of team-building tasks that test how each method performs on different parts of a team. In each task, everything in the team stays fixed except one part, which is left empty, and a method fills it in (the diffusion model, or an LLM). The goal is the fill that best counters the meta:

1. **Stat Points:** Randomly leave \(k\) Pokémon’s Stat Points empty, for \(k = 1\) to \(6\). Each method fills in the missing Stat Points to best counter the meta; everything else stays fixed.
2. Randomly leave \(k\) Pokémon’s item slots empty, for \(k = 1\) to \(6\). Each method fills in the missing items to best counter the meta; everything else stays fixed.
3. **Moves:** Randomly leave \(k\) move slots empty across the team, for \(k = 1\) to \(24\). Each method fills in the missing moves; everything else stays fixed.
4. **Whole Pokémon:** Randomly leave \(k\) Pokémon slots empty, for \(k = 1\) to \(6\). Each method fills those slots with Pokémon and their full sets, including Stat Points, items, and moves. The remaining Pokémon and their sets stay fixed.
5. **Abilities:** Randomly leave \(k\) Pokémon’s abilities empty, for \(k = 1\) to \(6\). Each method fills in the missing abilities to best counter the meta; everything else stays fixed.
6. **Stat Alignment:** Randomly leave \(k\) Pokémon’s Stat Alignments empty, for \(k = 1\) to \(6\). Each method fills in the missing Stat Alignments to best counter the meta; everything else stays fixed.
7. **Mixed Parts:** Randomly leave a combination of items, moves, abilities, Stat Alignments, and Stat Points empty. Each method fills in the missing parts to best counter the meta; everything else stays fixed between k=1 k to 40 
