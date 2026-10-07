# Large Language Models as Optimizers — visual reading guide

**Paper:** Chengrun Yang et al., *Large Language Models as Optimizers* (ICLR 2024), [arXiv full text](https://arxiv.org/html/2309.03409) and [official code](https://github.com/google-deepmind/opro).

**Visuals:** [[OPRO - visual explanation.excalidraw]] · [Animated explainer](../Teaching/reference/opro-visual-explainer.html)

## The idea in one sentence

Optimization by PROmpting (OPRO) asks an LLM to propose new solutions after reading a natural-language objective and a scored history of previous solutions; an external evaluator supplies the scores. No gradient or model-weight update is required for the optimizer (§§1–2).

## Read the diagram from left to right

1. **Make a meta-prompt.** Describe the task, objective, and any constraints. Include past solution–score pairs. In prompt optimization, include examples of the target task (§§2.2, 4.2).
2. **Generate candidates.** The optimizer LLM proposes several new solutions. The paper's default prompt-optimization experiments request eight instructions per iteration and use optimizer temperature 1.0 (§§2.3, 5.1).
3. **Evaluate outside the optimizer.** Apply each proposed instruction to a *scorer LLM* on a training subset. Accuracy becomes the objective value. The optimizer and scorer can be the same or different models (§§4.1, 5.1).
4. **Update the history.** Add the new instruction–score pairs. The default meta-prompt retains the highest-scoring 20 instructions, sorts them from lower to higher score, and shows three sampled task examples (§§4.2, 5.1).
5. **Repeat, then test.** Iteration stops when improvement stalls or the step limit is reached. The authors evaluate the selected instruction on held-out test examples after search (§§2, 4.1).

The score is feedback *about* a candidate, not a reward used to update the optimizer's weights. The loop changes the optimizer's next input by editing its in-context history.

## What the experiments show

| Setting | Paper observation | How to read it |
| --- | --- | --- |
| Small linear regression | LLMs use scored coefficient pairs to move toward a better fit (§3.1). | A continuous toy demonstration, with only two parameters. |
| Traveling salesperson | OPRO can find good small tours, but its optimality gap grows quickly with problem size (§3.2). | Evidence that the generic method does not replace specialized combinatorial search. |
| GSM8K prompt optimization | In one PaLM 2-L scorer setup, the best OPRO instruction reached **80.2%** test accuracy; the human-written “Let's think step by step.” baseline reached **71.8%** (Table 4). | This comparison belongs to a particular scorer, optimizer, prompt position, and dataset. |
| Iteration versus one-shot sampling | On GSM8K, the best of 50 one-shot instructions had **60.8%** test accuracy; iterative OPRO found an instruction with **76.3%** test accuracy by step 5 in the reported comparison (§5.3). | Fresh score feedback, rather than proposal count alone, is central to the method. |

The paper also reports that instruction order, visible numerical scores, and task examples all affect optimization (§5.3). Multiple proposals per step improve stability, but very large batches leave fewer feedback rounds under a fixed evaluation budget.

## Limits the visuals should not hide

- **Generalization is imperfect.** In the reported prompt experiments, training accuracy is often 5–20 percentage points above test accuracy (§5.4). A small training subset is useful for cheap search, but selection can overfit it.
- **Prompt construction matters.** Starting instruction, history order, score presentation, examples, and temperature can change the trajectory (§§5.3, 7).
- **Scaling is not established by the toy math tasks.** The authors note context-length limits, bumpy objectives, and worsening TSP results with larger instances (§3.2).
- **No Pokémon result is claimed.** The paper evaluates math and prompt tasks. Team search would need its own legality checks, battle evaluator, held-out opponents, and repeated runs. This is a project inference, not a finding of Yang et al.

## A fair OPRO-style baseline for this project

Give the proposer a bounded list of legal Pokémon teams and measured battle scores. Ask for several *new* teams per round. Check legality externally, evaluate every accepted team against the same opponent distribution and battle policy, then put the scored teams into the next prompt. Compare with random legal generation, copy-and-mutate, and the diffusion proposer under matched simulator budgets. Keep a separate opponent set for final evaluation. Measure validity, duplicate rate, diversity, and independently re-evaluated win rate.

That adaptation preserves OPRO's central mechanism while treating team representation, noisy battles, and hard legality constraints as new experimental questions.
