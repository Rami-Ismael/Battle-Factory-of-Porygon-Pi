- The [Research Article Template](https://huggingface.co/spaces/tfrere/research-article-template) shows this on Hugging Face Spaces with interactive diagrams, math, citations, and PDF export. Inspired by [Distill's framework](https://distill.pub/guide/), it treats technical writing as native to the web.
- Make sure you use highest before ultracode

# Todo
- [x] Build "Random Noise to Pokémon Team Archetypes: An Animated 2D Diffusion Demonstration", a dark-themed animation in the style of Alec Helbling's diffusion animations in which a cloud of Gaussian-noise points drifts into archetype clusters, using teams sampled directly from my masked diffusion model, and decide how a partly masked team gets a 2D position. Please use pokemon sprite for help in visual
	- [x] grill me
		- [x] ❓ **Q4** - **Which tool and which output?** Helbling's animations are made with Manim. Options:
		- [x] ❓ **Q3** - **What counts as an archetype?** trick room, weather, hyperoffensive, and etcgr
- [ ] Give me a numnber list of recommendation good visual that will be helpful for my current technocal blog that use the thing from the Hugging face blog we talked about https://huggingface.co/spaces/HuggingFaceM4/vlm-data-autoresearch#what-levers-does-the-agent-have and use the visual tool similar to see in pokemon we can use pokemon sprite aswell
	- [ ] **Watch a Pokémon team emerge.** Keep your existing sprite-based generation slider as the opening visual. Readers scrub from empty slots to a complete team and inspect its moves, items, and abilities. This makes the generator’s output immediately understandable.
	- [x] **Which teams beat which opponents?** A matchup heatmap: candidate teams down the rows, opponent teams across the columns. Click a cell to see both rosters, battle count, estimated win rate, and uncertainty. Leave untested matchups visibly blank. This shows weaknesses that an average score hides.
		- [x] Build a matchup heatmap that answers "which teams beat which opponents?": candidate teams down the rows, opponent teams across the columns; clicking a cell shows both rosters, the battle count, the estimated win rate and its uncertainty; untested matchups stay visibly blank; the reader can re-sort rows and columns (by mean win rate, or by win rate against the selected opponent), and each re-sort uses a layout animation in which every row and column slides to its new place (240ms, ease-in-out `cubic-bezier(0.77, 0, 0.175, 1)`, transform only, redirected mid-motion by a new click, snapping instead when reduced motion is on), so the reader can follow a team and see the weaknesses an average score hides.
		- [ ] Can you create some help using animation and visual vocabulary and grill my intiail instruction
	- [x] **Who gets selected for the next training round?** A temperature slider redistributes selection probability across teams with different scores. Pokémon sprites make the candidates recognizable. This explains how selection can concentrate on high scorers or retain more variety.
		- [x] Can you create some help using animation and visual vocabulary and grill my intiail instruction
		- [x] Can you create an exlaidraw to design wireframe what it look like 
	- [x] **Are the generated teams actually different?** A clickable grid of team rosters, grouping identical rosters and showing differences in sets on inspection. Readers can distinguish new Pokémon combinations from small changes to the same six species.
		- [ ] Can you create some help using animation and visual vocabulary and grill my intiail instruction
		- [ ] Can you create an exlaidraw to design wireframe what it look like please us best practice for wireframe and ux
		- [ ] Comparison
			- [x] 1. **Collapse** (`Teams differ 1 — composition collapse`): all 11 generations are stacked as strips of 128 teams, grouped by composition. The count falls from 112 compositions to 22, and the largest one grows from 8 teams to 96. You can read that without touching the slider or the Play button.
			- [ ] 1. **Which compositions** (`Teams differ 2 — which compositions`): the cards are now in rows by how many species differ from the largest composition. Of the other 21 compositions, 12 differ by one species, 8 by two, and 1 by four. The row a card sits in tells you whether it is a new combination or a small change.
			- [ ] Sets inside a composition (Teams differ 3 — sets inside a composition): a table counts how many different values each field takes across the 96 teams. Stat Points and moves change the most; ability and Floette's item hardly change. Below it, the best team (0.71) and the worst (0.21) are compared, showing only the fields that differ, with their actual values.
	- [ ] **Does the winning team survive a fresh test?** Show each finalist’s search performance beside fresh target-opponent and held-out-opponent results, with battle counts and 
	- [ ] uncertainty. Clicking a team reveals its full roster. This gives your conclusion a concrete test.
		- [ ] Can you create some help using animation and visual vocabulary and grill my intiail instruction
		- [ ] Can you create an exlaidraw to design wireframe what it look like please us best practice for wireframe and ux
	- [ ] **Best team so far, generation by generation.** This copies the post's "19 experiments, in the order they finished" chart.
	- [ ] Can you create some help using animation and visual vocabulary and grill my intiail instruction
- [ ]  I think a nice featuerwill be creating some visual for live training run to show how thing work out (Working on it right now )
- [ ] Read https://distill.pub/2020/communicating-with-interactive-articles/
	- [ ] What are the lesson can i learn from this and applied to my prompt that is ability that does generation the visual for me create a visual design like a workflow for me please (working on it writh ) please use the teaching skills for me please 
	- [ ] When reading 
- [ ] Read this also Lessons from interactive articles for our explanations and VGC blogchatgpt
- [ ] How do I add vertification and step in visual design like a checkbox that will be help
	- [ ] Checkbox for visual design 
- [ ] Use the claude design skills 
	- [ ] how to encourage the design skills more often in my prompt
		- [ ] look at previous prompt and determine the reason why it does not use design skills at all
- [ ] Read through the chat section "Pokemon VGC bi-level optimization visuals"
- [ ] Determine what are some vertification design
- [ ] please read the design books
- [ ] It will be interesting to have a third dimension for the 2d graph in where we should the model team strength againt the top meta threatz
- [ ] Read through apply interactive article from chatgpt
- [ ] Teach me the basic of visual design in where given a design I find the error in the design
- [ ] https://sophtang.github.io/discrete-beckmann-transport/blog/
- [ ] Update the prompt instruction to reduce from 24 to atmost 10 then 


# Visual Project we are creating

## Team Lineage

- [ ] I was inspired from reading this blog https://huggingface.co/spaces/HuggingFaceM4/vlm-data-autoresearch the first thing you see what is going on I want something similar in where we are given a slider in where you see an empty pokemon you scrool it will generate a current pokemon team please use pokemon sprite and please use dark theme

### Instruction to feed to a llm

```
Build a dark-themed interactive animation, modelled on the hero animation at the top of the Hugging Face article "Autoresearch for Data" (huggingface.co/spaces/HuggingFaceM4/vlm-data-autoresearch). I've attached its source file, banner.html. Adapt it: keep its flow, controls and styling, and replace its content with Pokémon teams.

WHAT THE BLOG'S ANIMATION DOES (reproduce this)
- A flow of nodes, read left to right, that appear one at a time and connect to the node they came from with square elbow lines.
- The top bar has a colour-coded legend on the left and a fraction counter on the right ("NN / N"), updated as nodes appear.
- The bottom bar has ← [Play] → buttons and a left-to-right slider. The button reads Play, then Pause, then Replay at the end. Arrows step one node; dragging the slider scrubs; both stop playback.
- Autoplay starts shortly after load (about 500ms) and reveals one node every ~420ms.
- The lines from the current node back to the first node are drawn brighter and thicker.
- Hovering a node shows a details tooltip; clicking pins it. While a node is inspected, nodes outside its lineage fade.

MY VERSION
- The first node is an empty Pokémon team (six blank slots).
- As the slider moves forward, each new node shows a current Pokémon team: one generation of the combined loop (results/gradloop.json, gens 1–11), showing that generation's top 3 teams by 24-battle win rate. Link each team to the most similar team in the previous generation (the loop records no parent).
- Each team is drawn with Pokémon sprites, six per node. Use Pokémon Showdown sprites: https://play.pokemonshowdown.com/sprites/gen5/<id>.png (<id> = lowercase species name, no spaces or punctuation; Mega forms use a hyphen, e.g. "charizard-megax"). Use image-rendering: pixelated, and show a grey silhouette if a sprite fails to load.
- Colour coding: how many of the final core (garchomp, kingambit, basculegion, whimsicott, charizard, floette-eternal) a team holds: start / 0–3 / 4–5 / all 6. A ★ badge marks teams re-battled 192 times.
- The tooltip shows the team's details: generation and rank, wins/24, the generation's mean ± SE, the ★ 192-battle win rate ± SE, and per Pokémon: species, item, ability, nature, 4 moves.
- Data: real teams only: results/gradloop.json (pastes, y) and results/rebattle_top.json. No placeholders.

OUTPUT
a single self-contained HTML file with the data inlined: Blog/visuals/team-lineage/team-lineage.html.
```
# How distil animation work 

I've identified the article's style: single-column prose with inline numbered figures, margin notes, slider-driven interactions, and a minimal palette/typography. I'm now rebuilding the team-preview figure to match, using three slider- or click-driven figures embedded in short prose.


# What animation skills do I use first


1. Pick an UI library
2. Prototype
3. animate
4. find animation opportunities
5. improve animation 


## Prompt Instrucion


```
/goal Done when variants/DECISIONS.md exists and records, for stages 0–5 below, every version made, each verifier's pass/fail table, and the judge's ranking with reasons, and the final UI passes every stage 3 check.

You direct; subagents do all the work. Per stage: makers (model "haiku", parallel, different briefs, each in its own folder variants/stage-N/x) → verifiers (one "haiku" per version, pass/fail table only) → judge (one "opus", runs /design:design-critique on each built version, ranks blind to briefs, ranks down for shallow, confusing, off-subject or generic). No maker judges or verifies its own work. If nothing is good enough, one more round with the judge's objections, then proceed with the best.

I want different, unique and novel ideas. Each maker's brief names a different form (for example: something the reader plays, something they watch unfold, two linked views, direct manipulation of the real objects, a before/after comparison, a build-it-yourself, a map, a timeline), so no two makers start from the same place. Novel means it is not the default treatment of this subject — not a plain bar chart, a slider over a chart, a card grid or a click-through stepper — and it is not a copy of another idea in the set.

Read my further instructions first: <file path>.

0. Ideas (no skill): 8 makers × 3 ideas. Each: name, form, what the reader sees and does, what makes it different from the usual way this subject is shown, and depth in three layers (at a glance / on interaction / on a second visit). Judge merges ideas that share a form, cuts any that is a default treatment, shortlists at least 6 with no two of the same form; the top 3 go forward.
1. Library: /frontend-design:frontend-design. 4 makers each argue one library; judge picks.
2. Prototype: one maker per idea runs `/prototype <idea> x5`, stops at its Phase 5, keeps the picker, and loads /dataviz if the idea has a chart, matrix or heatmap. Variants differ on layout, density, personality or interaction model, not motion. Verify: /design:accessibility-review and /mobile-native; every variant renders and responds; console clean; 375/768/desktop with no clipping; every state; light and dark. Judge picks one variant per idea. Then one "haiku" runs /design:ux-copy on each winner.
3. Animate: /animate, with /apple-design for anything dragged, swiped or thrown. 4 makers per winner, naming each effect by its /animation-vocabulary term. Verify: reduced motion leaves no sliding or scaling; only transform and opacity; under 300 ms, exits shorter than entrances; double-click retargets; no long frames at 4x CPU throttle; no layout shift. Judge picks per component and cuts any motion whose purpose it cannot state.
4. Opportunities: /find-animation-opportunities. 4 makers, different lenses, /animation-vocabulary term plus exact values, read-only. Judge merges and rejects with reasons.
5. Improve: /improve-animations. 4 makers audit; judge merges into one prioritized list; one "haiku" per fix applies it; rerun stage 3 verifiers and /design:accessibility-review; one "haiku" runs /design:design-system to flag one-off colours, spacing, radii, durations and easings; one "opus" reviews.

Rules: no stage skipped or merged; no animation before stage 3; stage 4 writes no code; respect prefers-reduced-motion; no idea cut in stage 0 before the judge has seen it.
```



# Verification or judge

1. It can be pickup from a beginner to understand is going on 
2. Can I state what the visual teaches in one sentence?
3. Can I follow one example without interacting?
4. Does each control answer a specific question?

# When you don't like a certain design or animation

1. Ast the model can you teach me what important deliverable action that I can take to make better