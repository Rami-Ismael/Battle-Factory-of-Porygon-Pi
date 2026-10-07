---
created_at: 2026-09-07
source: https://www.youtube.com/watch?v=nRrt7AczYV4
status: source-abstract-based-analysis
---

# Lessons from Tegmark's interpretability talk for the VGC project

**Most useful next step: test whether changing arbitrary Pokémon slot order changes the model's behavior.** The broader connection is to representation quality and checks on learned models, rather than a new team-search algorithm.

## What was verified

The video is Max Tegmark's **Neural network interpretability: symmetry, geometry and formal verification**, published by IPAM and described as recorded on September 1, 2026. Its official description introduces symmetry and geometric structure as helpful for generalization, followed by research toward exporting learned algorithms into formally verified code. [Official video and abstract](https://www.youtube.com/watch?v=nRrt7AczYV4).

Evidence limit: the YouTube page and embedded official description were retrieved directly. English automatic captions were listed, but their endpoints returned empty responses. The full spoken content was not reviewed; no timestamps or claims about specific examples in the talk are supplied. The applications below are our project-specific inferences, not advice attributed to Tegmark.

## 1. Turn symmetry into a measurable property

The pilot already sorts species for canonicalization and permutes team slots during training. This is a useful start, but augmentation alone does not guarantee consistent predictions. Its denoiser also adds learned positional embeddings. [Pilot README](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/README.md), [denoiser implementation](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/diffusion.py).

A concrete diagnostic: in evaluation mode, hold timestep and conditioning fixed, permute complete Pokémon blocks together with their masks, run the denoiser, undo that permutation on its logits, and compare with the original logits. For a future team-value surrogate, compare its score across the same equivalent representations. A large change exposes sensitivity to representation order. This could motivate a shared per-Pokémon encoder and attention or pooling that treats arbitrary member order consistently; measure the diagnostic before redesigning the architecture.

Only apply the test where ordering is representational. Battle active positions, selected leads, and action/target indices have semantics and must be preserved or correctly remapped. This is an application of the abstract's symmetry/generalization theme, not a finding demonstrated for VGC by the video. [Video abstract](https://www.youtube.com/watch?v=nRrt7AczYV4).

## 2. Test an explanation through controlled changes

If a surrogate or embedding seems to recognize a useful team interaction, turn that interpretation into a prediction: replacing one member or set should change estimated matchup strength in a particular way. Check that prediction with fresh battles while holding the evaluation policy and opponent pool fixed. This can reveal a model that ranks a familiar-looking team highly without correctly evaluating the interaction.

A colorful embedding plot alone is weak evidence. IPAM's workshop overview explicitly notes that even untrained models can produce compelling but uninformative visual explanations. The controlled-change experiment above is our proposed response to that concern, not a specific experiment verified in the talk. [IPAM workshop overview](https://www.ipam.ucla.edu/programs/workshops/foundations-of-interpretability/?tab=overview).

## 3. Keep exact guarantees scoped to exact claims

The talk's abstract motivates formally checked code extracted from learned systems. For this project, the nearby practical distinction is between enforcing known legality constraints and estimating competitive strength. The existing legality masks and Showdown validation address the former. Passing a legality validator does not establish a team's win rate, and testing a validator is not itself a formal proof.

Formal verification could later target a narrow specified property, such as a constraint-preserving transformation. It is not currently evidence for replacing battle evaluation or adding a proof assistant to the pilot. This priority judgment is our inference from the project's existing separation between generation and empirical evaluation. [Video abstract](https://www.youtube.com/watch?v=nRrt7AczYV4), [[Diffusion as candidate proposer in black-box optimization over structured inputs]].

The incremental lesson is to **check whether the model respects a known equivalence before spending scarce battle evaluations on its proposals**. It complements the existing real-team warm starts, mutation baselines, and candidate-quality experiments.
