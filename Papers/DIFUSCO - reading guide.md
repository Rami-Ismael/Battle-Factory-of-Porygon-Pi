---
tags: [paper, diffusion, combinatorial-optimization]
source: https://arxiv.org/abs/2302.08224
---

# DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization

Zhiqing Sun and Yiming Yang, NeurIPS 2023. [Published paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/0ba520d93c3df592c83a611961314c98-Paper-Conference.pdf) · [Official code](https://github.com/Edward-Sun/DIFUSCO).

Open [[DIFUSCO - visual explanation.excalidraw]] in Obsidian's Excalidraw plugin. The accompanying `.excalidraw` file also opens through **Open** in [Excalidraw](https://excalidraw.com). Read panels 1-7 in order. The toy graphs illustrate the method; they are not experiment outputs.

![[DIFUSCO - visual explanation.png]]

## What the method learns

DIFUSCO learns what good **selections on a graph** look like. For traveling salesman problems (TSP), a selection says which edges belong to the route. For independent set problems, it says which nodes belong to the set. A bit of 1 means selected and 0 means unselected.

The graph describes the problem; the bits describe a proposed answer. During diffusion, the problem graph stays fixed while the selection bits become noisy. This distinction is the starting point for Section 3.1.

## Why would removing noise solve an optimization problem?

Because the clean examples used for training are already good solutions. The model learns to reconstruct those examples from corrupted versions, conditioned on the corresponding graph. At inference, repeated denoising generates selections resembling the good solutions it learned from.

Optimization enters through the quality of training labels, the decoder, and any additional search or best-of-many selection. There is no rule saying that every denoising step must reduce tour length. Intermediate selections can be invalid, and the learned model has no guarantee of finding a global optimum.

Training labels come from Concorde for TSP-50/100, LKH-3 for larger TSP instances, and KaMIS for independent sets. The method therefore needs good labeled training examples, but they need not all be exactly optimal. See Sections 3.1 and 4.1, Section 5, and Appendix A.1.

## Follow the four-city example

The six possible undirected edges are ordered **AB, AC, AD, BC, BD, CD**. The perimeter tour A-B-C-D-A is encoded as **[1, 0, 1, 1, 0, 1]**.

During training, bit flips corrupt that vector. The graph neural network receives the corrupted bits, the city information, and noise level t. It predicts the probability that each clean bit should be 1. The known clean tour supplies supervision.

At test time, the new graph has no supplied solution. Begin with random bits, predict clean-bit probabilities, and use the reverse diffusion transition to obtain a less-noisy state. Repeat with the same trained network at lower noise levels. This is a sampling process, not retraining the network on every new graph.

The network predicts the clean state, while the reverse transition determines the next noisy state. Those are two distinct operations in Equations 5-6.

## Why use a graph neural network?

A choice of edge or node affects other choices. Message passing allows neighboring nodes and edges to exchange information rather than treating each decision in isolation. DIFUSCO uses an anisotropic GNN with learned edge gates to control those messages. The paper uses 12 layers of width 256.

Repeated denoising can generate different coherent choices from different random seeds. This helps represent multiple good solutions for the same graph; it does not imply that every good solution will be discovered. See Sections 3.4-3.5.

## The decoder is part of the solver

The final heatmap retains confidence scores. Simply turning all scores above a threshold into selected edges can violate the tour constraints.

For TSP, greedy decoding ranks edges using confidence and distance, then adds edges while avoiding conflicts such as excess degree or premature subtours. Optional **2-opt** replaces two tour edges when doing so shortens the route. An alternative **Monte Carlo tree search (MCTS)** uses the heatmap to guide tour improvements.

For independent sets, greedy decoding selects nodes in score order and skips conflicting choices. Different generated heatmaps can be decoded separately, then compared using the actual objective. See Section 3.5.

## The design choices to remember

- **Discrete vs. continuous noise:** discrete diffusion flips bits; continuous diffusion maps bits to -1/+1 and adds Gaussian noise. The continuous network predicts noise instead of clean-bit class probabilities.
- **Inference schedule:** the models use 1,000 training noise levels, but inference can skip levels. The cosine inference schedule allocates more steps near the clean end. This differs from the linear forward noise schedule.
- **Compute budget:** more denoising steps refine one sample; more random starts explore several samples. The paper evaluates both choices.
- **Scaling:** large TSP graphs retain nearby edges to avoid a dense quadratic-size graph.

Sections 3.2-3.3 and 4.2 explain these choices. Discrete diffusion wins the main TSP comparisons; the ER independent-set experiments use continuous diffusion and remain a weak point.

## How to interpret the headline results

The reported TSP gaps of **0.46%, 1.17%, and 2.58%** are for **DIFUSCO + MCTS** on 500, 1,000, and 10,000 cities. They are not scores for a raw denoised bit vector. Table 2 uses Concorde references for 500/1,000 cities and LKH-3 for 10,000, so the latter is not a certified gap to the true optimum. Hardware differences also limit direct runtime comparisons.

Terminology: the paper sometimes says **maximal independent set**, but its objective minimizes the number of unselected nodes, which seeks a **maximum** independent set. Maximal means no further node can be added; maximum means largest possible. These are different properties.

## Connection to your Pokémon generator — an adaptation idea

The transferable pattern is: **learn from strong solutions → generate candidate decisions → enforce legality → evaluate and improve**.

For your project, the hard work would be defining the representation and obtaining strong training examples conditioned on the opposing team. Pairwise graph edges alone may not capture all team interactions. You would also need a legality decoder and battle evaluation; DIFUSCO does not supply those components or demonstrate results for Pokémon.

A useful distinction from the MAD paper: DIFUSCO learns from solver-labeled examples. Its repeated denoising steps are not generations of models trained on their own outputs.

For a first read, use **Section 3.1 → Section 3.5 → Section 3.4 → Section 3.2 → Section 4.2**. Understand the representation and decoder before tackling the diffusion equations.
