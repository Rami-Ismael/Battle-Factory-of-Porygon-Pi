---
created_at: 2026-10-01
tags:
  - research
  - technical-writing
  - interactive-explanations
source: https://distill.pub/2020/communicating-with-interactive-articles/
---

# Lessons from interactive articles for our explanations and VGC blog

**The main lesson I take is to design each interaction around something the reader should be able to reason about afterward.** Before choosing a slider, animation, or chart, write the question it will help the reader answer.

Distill organizes the possibilities around connecting people with data, experimenting with systems, reflection, personalization, and managing cognitive load. Its closing discussion matters as much as its examples: evidence about complete interactive articles remains limited, and creating them carries substantial editorial and maintenance costs. I read this as a menu of techniques to choose deliberately. [Hohman et al., *Communicating with Interactive Articles*](https://distill.pub/2020/communicating-with-interactive-articles/).

The recommendations below are my synthesis. Project applications are proposals informed by the local notes; they are not findings established by the cited learning studies.

1. **Make the claim testable through the interface.**

   Bret Victor's 2024 postscript sharpens the original idea: readers should be able to examine and change the assumptions and calculations supporting an argument. That is a stronger ambition than adding interactive illustrations. His original essay also insists that the author provide guidance and an explanation that reads coherently before anyone touches a control. [*Explorable Explanations*, including the 2024 postscript](https://worrydream.com/ExplorableExplanations/).

   For our work, every proposed control should complete this sentence: “Change ___ to find out whether ___.” Changing opponent weights to test a team's ranking has a clear purpose. A decorative control with no question attached needs more design work.

2. **Give the reader a small experiment with a clear starting point.**

   I would use a sequence of **question → prediction → change → visible consequence → explanation**. Start with one meaningful contrast, supply a useful default, and provide reset. Broader exploration can follow once the reader understands what the controls mean. This is my design response to Victor's distinction between an authored explanation and an unguided sandbox. [*Explorable Explanations*](https://worrydream.com/ExplorableExplanations/).

   For selection temperature, hold team scores fixed and ask which candidates will receive more training probability when temperature changes. Explain the redistribution next to the bars. That interaction explains selection; establishing effects on later win rate requires measured training experiments.

3. **Keep important comparisons visible.**

   In Victor's *Ladder of Abstraction*, a moving car becomes a complete trajectory, then a view across parameter settings. The reader can move back down to inspect individual states. This provides a useful design progression: show an example, reveal the pattern across examples, and retain access to the details. [*Up and Down the Ladder of Abstraction*](https://worrydream.com/LadderOfAbstraction/).

   I would keep a baseline visible while a reader changes the current state. For diversity across generations, show the overall trend or several generation summaries together; let selection reveal actual rosters. A slider showing one generation at a time makes the reader remember the comparison. An animation is useful when following a particular team's movement matters.

4. **Preserve identity and control unrelated variation.**

   Amit Patel describes how changing tie-breaking paths and random map layouts can make diagrams flicker even when readers adjust unrelated parameters. He also documents consistent visual references between text and diagrams, and the usefulness of showing alternatives together. These are practitioner observations, not universal experimental findings. [Red Blob Games, *Little Design Things*](https://www.redblobgames.com/making-of/little-things/).

   My implementation rule would be to keep team identities, labels, colors, and the baseline stable. For a stochastic teaching simulation, retain the same random draws during a controlled comparison and provide a separate “new trial” action. Label this choice. Genuine discontinuities in the modeled process should remain visible; smooth animation should not invent intermediate scientific results.

5. **Teach the reader how the visual can mislead them.**

   *How to Use t-SNE Effectively* is valuable because its examples expose misreadings. Cluster area and separation can misrepresent the original data, apparent clusters can arise from random data, and outcomes depend on parameters and runs. The article uses deliberately simple datasets to make those failures inspectable. These specific results concern t-SNE. [Wattenberg, Viégas, and Johnson, *How to Use t-SNE Effectively*](https://distill.pub/2016/misread-tsne/).

   For our project, I would pair an attractive result with a case that challenges its interpretation: similar-looking rosters with different sets, a high search score that weakens on fresh evaluation, or a two-dimensional team map whose distances disagree with a declared team-distance measure. These are proposed teaching cases, not claims that those outcomes have been measured here.

6. **Name exactly what the reader is observing.**

   Our existing documentation already makes useful distinctions. The weather figure illustrates mechanisms without measuring win-rate effects. The generation figure replays recorded categorical decisions from a saved checkpoint and does not run browser inference. Those distinctions should be visible beside the graphics. [Weather figure documentation](/Users/ramiismael/Documents/yakumsi-vault/Personal%20Project/Using%20Advance%20method%20search%20in%20the%20whole%20search%20of%20possible%20vgc%20pokemon%20format%20to%20find%20the%20right%20counter%20to%20a%20pokemon%20team/Blog/visuals/team-interactions/README.md), [inference replay documentation](/Users/ramiismael/Documents/yakumsi-vault/Personal%20Project/Using%20Advance%20method%20search%20in%20the%20whole%20search%20of%20possible%20vgc%20pokemon%20format%20to%20find%20the%20right%20counter%20to%20a%20pokemon%20team/Blog/visuals/real-diffusion/README.md).

   I would distinguish recorded observations, recomputed summaries of those observations, and illustrative simulations. Changing opponent weights can recompute a score from measured matchups. It cannot create evidence for an untested matchup. Scrubbing a recorded generation trace reveals that run; it does not generate a new team.

7. **Design for what survives after the explanation closes.**

   Matuschak and Nielsen's Quantum Country work embeds retrieval questions and spaced review into an essay. Their report explicitly describes its results as preliminary, and recognizes that learning involves more than memory. The useful design idea is to make revisiting essential concepts part of the reading experience. [*How can we develop transformative tools for thought?*, mnemonic-medium sections](https://numinous.productions/ttft/).

   I would end an important explanation with a new case: “Two teams have the same average score but different opponent-specific results. What information would make you prefer one?” For evaluation, check whether readers can predict an unfamiliar case, explain why, and identify a limitation. Click counts and time on the page can diagnose usage; they do not directly answer those questions.

The empirical papers help set the strength of these recommendations:

| Primary study | What it supports | What it does not establish |
|---|---|---|
| [Kim, Reinecke, and Hullman, *Explaining the Gap* (2017)](https://idl.cs.washington.edu/files/2017-ExplainingTheGap-CHI.pdf) | In the main experiment, 373 analyzed participants recalled voting data after a three-minute distractor. All four elicitation techniques improved value recall in visual conditions; prediction combined with explanation or feedback also reduced trend error. | Durable retention, general mastery, or transfer to unfamiliar problems. This supports trying prediction and explicit feedback, with evaluation in our own setting. |
| [Zhi, Ottley, and Metoyer, *Linking and Layout* (2019)](https://visualdata.wustl.edu/files/linking.pdf) | A 180-person study of one immigration story found better comprehension with its slideshow layout. Text–chart linking improved reported engagement and, within that layout, recall. | That slideshows or guided exploration always outperform other formats. Results depend on the tested story and interface. |
| [Conlen, Kale, and Heer, *Capture & Analysis of Active Reading Behaviors* (2019)](https://idl.cs.washington.edu/files/2019-IdyllAnalytics-EuroVis.pdf) | Logs from over 50,000 sessions across three articles showed varied reading behavior and greater use of interactions central to the narrative. | Causal learning gains. I infer that core reasoning should remain visible and hidden controls should serve optional depth. |

The most useful changes to our existing visual plan are small and specific:

| Visual | Reader's question | Change I would prioritize |
|---|---|---|
| Matchup matrix | Does the best team depend on the opponents I care about? | Let opponent weights recompute the aggregate score; preserve the original ranking and expose battle counts and missing cells. |
| Selection temperature | Why does changing temperature concentrate or spread selection? | Keep candidate scores fixed; compare two probability distributions and ask for a prediction before revealing the second. |
| Composition diversity | Are these new compositions or different sets for the same species? | Keep composition counts visible across generations; drill into actual species and set differences. |
| Recorded generation | What did this model choose at this step? | Link the selected field to its recorded probabilities and legality constraints, retaining the recorded-run label. |
| Weather interactions | Which relationship changes when the environment changes? | Keep the roster fixed; highlight the changed mechanism and ask the reader to explain it. |

These build on the questions already in [Visual Plan.md](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Visual Plan.md>). I reviewed that plan and relevant figure documentation, not the implementations or underlying experiment data.

A particularly useful new explainer would address **why a search winner needs a fresh test**. Use an explicitly labeled toy model in which every candidate has the same known win probability. Give each candidate a small number of simulated trials, select the apparent winner, and evaluate it on independent fresh trials. Let the reader vary candidate count and trials per candidate, comparing search and fresh estimates across repeated runs. Show the known equal probabilities throughout. The experiment would isolate selection from noisy estimates; it would not establish the size of that effect in actual Pokémon battles. This would fit the existing [“Why 24 battles lie” topic](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Why 24 battles lie — noise and the winner's curse.md>).

For future explanations, I would write the learning question and an informative default view first. Then add the smallest interaction that answers the question, a way to compare outcomes, and a short prompt that checks whether the reader can use the idea. Keep essential meaning available in text and static views, support keyboard and touch interaction, and let readers pause motion. These are proposed design requirements; their value should be checked with readers rather than assumed from visual polish.
