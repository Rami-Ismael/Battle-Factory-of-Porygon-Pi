- [ ] Think about the meta structure of writing a blog the blog will start should have atleast two thing why you should care and conclusion and image
	- [ ] [[Why so we initial use a diffusion model instead of Large Language Model]]
	- [ ] [[What are the metaheuristic appraoch in creating pokemon team ]]
	- [ ] [[What is a diffusion model]]
	- [ ] What experiment can I show
		- [x] Fix the overlap between the two markdown file [[What is the baseline]] and [[Ablation Experiments]]
		- [ ] [[What is the baseline]]
		- [ ] [[Ablation Experiments]]
	- [ ] [[What counts as a good counter pokemon team]]
	- [ ] [[Explain the metric for diversity and quality for the blog]]
	- [ ] [[What failed and what I learned]] 
	- [ ] [[Explain why creating AI for pokemon does not hurt pokemon acutall enhance pokemon]]
	- [ ] [[Go through in the blog about the ML pipelines]]
	- [ ] [[Add a related work section]]
	- [ ] [[Cross Entropy]]
	- [ ] [[The engineering behind the project doesn’t appear in the outline at all: the persistent battle workers, the Showdown validator and the cache of battle results.]]
	- [ ] [[Run it yourself — from clone to a battled team]]
	- [ ] [[Metaheuristics]]
	- [ ] [[Conclusion]]
	- [ ] [[Short VGC primer from your play]]
	- [ ] [[What actually moved win rate, ranked]]
	- [ ] [[The bill behind every number, battles and dollars and hours]]
	- [ ] [[EDA on meta pokemon team]]
	- [ ] [[How I ran this project with AI agents, and where they got it wrong.]]
	- [ ] [[Limitation]]
	- [ ] [[A pretrained text diffusion model won alone, then lost inside the search]]
	- [ ] [[158 Pokemon my first model could never pick. A measured bug story and a limitation.]]A

# Steps for the workflow


## Todo

- [x]  Ask to find relevant visual artifact that are already created to add that determine the right placement of them inside wiki, in the order of the wiki please use subagent to determine the placement, then use subagent to check the work 
- [ ] Determine the order of each section ther


## Meta Structure of the prompt

The wikilinks in [[Blog]], in their current order, are the future sections of my final blog post presenting this work. My starting note on structure: the post needs at least a reason to care, a conclusion, and images. Read every linked note, including its headings, and the notes it links to, one level down (for `Reading List` and `Todo Section`, only the relevant heading). Treat writing-process notes as guidance, not sections.

Readers: ML-curious engineers and students. The post is a worked example by a regular Pokémon Champions player. It must also work for hiring managers: I’m job-hunting, and a hiring manager should finish it convinced I can take an ML project from idea to measured result.

Judge what is missing against that reader: the problem and why it matters, a short VGC primer from my own play, the setup, the method from scratch with visuals, the big result early, what failed, limitations, how to reproduce, and measured results with no section yet. For each section, report which of my real teams it uses as a running example, or “none”; this is a report, not a fix. List at most 8 titles for new markdown files in chat, ranked, each with a one-line reason, and wait for my yes before creating any file. Each file I approve gets the full title as its filename and frontmatter only. I’m open to WACKY ideas.

Then place the visuals I have already built:

1. Collect every existing visual: the folders in `Blog/visuals/`, the files in `Blog/images/` and `artifacts/`, and my published Artifact pages (list them with the Artifact tool; subagents can’t reach them). Use [[Visual Plan]] as guidance on which visuals I prefer. Done when you have one list with a path or URL for each visual.
2. Split the list into batches of about 10, and give each batch to its own Explore subagent (Haiku). Give each subagent the section order from [[Blog]], with one line per section saying what it covers. Each subagent reads only a visual’s title, headings and first screen of text, and returns one row per visual: the visual, what it shows, the best section, where in that section it goes (before which heading or paragraph), and confidence (high, medium or low). Done when every visual has a row or is marked “fits no section”.
3. Merge the rows into one placement table in the order of [[Blog]], and list the sections that have no visual.
4. Give the table to a second, separate Explore subagent to check. For each row, it confirms the path or URL exists, that the visual shows what the row claims, and that the section fits. It returns pass or fail with a one-line reason. Done when every row has a verdict.
5. Show me the checked table in chat, with failed rows fixed or marked. Edit no note until I say which placements to keep.

# Blog Section Orders

- [ ] [[When writing for a technical blog you want to present big result to determine why they should care]]
- [ ] [[Why should you care about team building in competitive pokemon as an academic question]]
- [ ] [[EDA on meta pokemon team]]



# Guidances

- [ ] [[Visual Plan]]
- [ ] [[Title for the blog]]
- [ ] [[Source of good reading tips]]