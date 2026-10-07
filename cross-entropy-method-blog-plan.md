# Blog plan: Pokémon team building with the cross-entropy method

Status: planning. This document proposes the explanation and experiment; it does not report completed results.

## Central question

Can the cross-entropy method find a Pokémon team with better battle performance than random search under the same battle budget?

For the counter-team project, start with one fixed opposing team. Use one scalar objective: estimated win rate against that opponent, under a fixed battle-playing policy and evaluation protocol. Choose and document the format before defining legal candidates.

## Working title and opening

**Building Pokémon Counter-Teams with the Cross-Entropy Method**

Lead with the strongest verified result once the experiment is complete: what team was found, how it performed on fresh battles, and how much search it required. Until those measurements exist, frame the opening as a question and experiment rather than claiming that CEM beats the meta.

Then explain why the reader should care: the number of possible teams grows rapidly when species, moves, items, and stat choices are combined. CEM offers a simple way to learn where to search from the teams already evaluated.

## The explanation readers need

The central loop is **sample → battle → select elites → update the sampling distribution → repeat**.

1. Start with a probability distribution over candidate choices.
2. Sample a batch of legal teams.
3. Evaluate each team through battles.
4. Select the highest-scoring fraction, called the elite set.
5. Fit the sampling distribution to those elites.
6. Sample the next batch from the updated distribution.

The optimization method uses cross-entropy to fit its search distribution. Win rate remains the objective being maximized.

Introduce the method using the authors’ [A Tutorial on the Cross-Entropy Method](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf). Keep the historical and mathematical treatment brief; devote most of the article to the Pokémon adaptation and measured behavior.

## A worked categorical update

Use a small illustrative decision with three legal alternatives, A, B, and C. This could be an item choice for a fixed Pokémon; it should not imply that independent choices are enough to represent an entire legal team.

Suppose the current probabilities are:

| Choice | Current probability | Frequency among 10 elites |
|---|---:|---:|
| A | 0.40 | 0.70 |
| B | 0.35 | 0.20 |
| C | 0.25 | 0.10 |

For an independent categorical variable, maximum-likelihood fitting to elites gives their observed frequencies. A smoothed update is:

\[
p_{t+1,j}=(1-\alpha)p_{t,j}+\alpha\widehat p_{\mathrm{elite},j}.
\]

With an illustrative update rate of \(\alpha=0.5\), the next probabilities are **0.55, 0.275, and 0.175**. The next batch is more likely to use A while retaining opportunities to sample B and C.

These numbers demonstrate the mechanism; they are not benchmark results or prescribed hyperparameters. The elite fraction and update rate are separate parameters.

For a general parameterized distribution, the elite fitting step is:

\[
\widehat\theta_{t+1}
=\arg\max_\theta\sum_{T\in E_t}\log q_\theta(T),
\]

where \(E_t\) is the elite set and \(q_\theta\) is the team generator. The frequency update above is a special case. Masking and sampling without replacement change the likelihood; their exact fitting rule must match the implemented sampler.

## Pokémon decisions to settle before writing the implementation section

- [ ] **Candidate space:** start with curated legal sets, or search moves, items, abilities, and stats directly? Curated sets provide a simpler first experiment but restrict what CEM can discover.
- [ ] **Team representation:** how are six members represented without treating permutations of the same team as different discoveries?
- [ ] **Legal generation:** how are species duplication, item restrictions, and species-dependent choices handled? Conditional sampling, rejection, and repair produce different sampling behavior.
- [ ] **Dependencies:** does the generator learn only individual frequencies, or also teammate combinations? An independent model can lose useful synergies.
- [ ] **Exploration:** use smoothing and, if needed, a documented probability floor or mixture with an exploratory sampler to avoid premature collapse.
- [ ] **Elite selection:** specify the fraction and the rule for tied scores.
- [ ] **Evaluation:** identify the battle-playing agent, opponent team, lead selection procedure, battle count, and treatment of draws or failed battles.
- [ ] **Stopping:** set a total battle budget and record the best candidate throughout the run.

Present these as implementation decisions. Do not describe a proposed feature as something the project already does.

## Minimum experiment for a credible blog

Compare CEM with random search over the **same legal candidate space**, using the same policy, evaluation protocol, and total number of battles. Count all repeated evaluations in the budget. Run multiple independent search seeds.

Keep the search outcome separate from final validation: evaluate each selected finalist with fresh battle seeds. When targeting one opponent, conclusions apply to that matchup and policy. Broader performance needs a separate opponent test suite.

Record:

- [ ] Fresh-battle win rate and uncertainty for the selected finalists.
- [ ] Results across independent search runs.
- [ ] Performance versus cumulative battles, rather than iterations alone.
- [ ] Unique teams evaluated and duplicate rate.
- [ ] How rapidly sampling probabilities concentrate.
- [ ] Time spent on generation and battles.

Unique-team counts and distribution concentration are diagnostics, not additional optimization objectives.

## Suggested article sequence

1. **The result and why it matters:** use measured evidence when available.
2. **The team-search problem:** define the candidate space and scalar objective.
3. **CEM in one loop:** explain sampling, elites, and refitting.
4. **A numerical example:** show one categorical update.
5. **Making the generator legal:** explain representation, constraints, and dependencies.
6. **The experiment:** state baseline, budgets, policy, and validation.
7. **What happened:** show learning curves, finalists, uncertainty, and failures.
8. **What remains unresolved:** describe limitations supported by the experiment.

Useful figures: a loop diagram, a before/after probability bar chart, and measured win rate versus battle budget. Save the broader metaheuristic catalog for background or a later post.

## Evidence boundary

CEM can concentrate probability around choices appearing in high-scoring samples. Whether that improves Pokémon team search, preserves synergy, or beats random search must be measured here. It does not guarantee the globally best team, and a noisy elite set can steer the generator toward lucky candidates.
