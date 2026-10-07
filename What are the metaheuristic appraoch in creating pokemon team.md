1. I created a learned generator with a diffusion model to generate new pokemon team can I steer possibl

- [x] Explain the roles researchers give a diffusion model when it is used to optimise a function, with academic citations from 2023 to now. Cover all seven roles: candidate proposer, inverse model, guided sampler, reward-tuned policy, sampling-time searcher, learned solver, and score-only sampler. Focus on online, simulator-in-the-loop optimisation, where we can query new inputs whenever we like to learn how the input relates to its value, not offline model-based optimisation from a fixed dataset. Where a role's papers are offline, include them labelled as offline, with one line on what would change with a simulator. First check whether a published survey already names these roles; if one does, use its names instead of mine. If none does, say the seven roles are my own taxonomy.
	- [x] grill me 
- [ ] What are alternative of approach ask an llm to generate this should be all places in [[Metaheuristics]]
- [ ] This is the place where you write down the your [[Cross Entropy Method]]

# instruction

Explain the roles researchers give a diffusion model when it is used to optimise a function, with academic citations from 2023 to now. Put the answer in chat, not in the vault.

Cover all seven roles, and define each by the job the diffusion model does inside the optimisation loop: what goes in and what comes out. Put each paper under its main role and name any secondary role. **(rec.)**

- **Candidate proposer:** generates candidates that another component scores and selects.
- **Inverse model:** generates an input for a requested value.
- **Guided sampler:** a trained model whose sampling is steered by a value signal (classifier-free guidance, or gradients from a predictor). **(rec.)**
- **Reward-tuned policy:** the model's weights are fine-tuned to maximise the objective.
- **Sampling-time searcher:** search over the denoising steps themselves (beam search, sequential Monte Carlo, tree search) to pick the best path. **(rec.)**
- **Learned solver:** trained on solved instances to output solutions to new instances, e.g. DIFUSCO for the travelling salesman problem. Say in one line why this role fits poorly when there is one fixed objective.
- **Score-only sampler:** no training data; the diffusion process samples from a distribution proportional to exp(f), using f alone. **(rec.)**

Focus on online optimisation: we have access to the black box and a query is cheap, so we can query new inputs whenever we like to learn how the input relates to its value. Count any paper that queries the objective during the run as online, and tag each with its query budget: small (≤1k), medium, or large (≥10k). **(rec.)** Tag each paper as noisy or noiseless objective. **(rec.)** This is not offline model-based optimisation from a fixed dataset. Where a role's papers are offline, include them labelled "offline", with one line on what would change with a simulator.

Include discrete and continuous papers, tagged discrete or continuous. Molecule and protein papers may be cited.

Citations: peer-reviewed first. An arXiv preprint is allowed only when it is the only online example for a role, and is labelled "preprint". Two to four papers per role. Look up each paper's venue at the source, and label every link 🟢 (openable) or 🔒 (paywalled). **(rec.)**

First check whether a published survey already names these roles. Count only a peer-reviewed or widely cited survey. Where its names match my roles one-to-one, use its names and say which are the survey's and which are mine. **(rec.)** If no survey names them, say the seven roles are my own taxonomy.

Grade each role HIGH, MED or LOW for my problem: searching for a full VGC team (species, ability, item, moves, Stat Points, alignment) that maximises win rate against the meta, where the objective is a noisy battle simulator. Give one reason per grade, and count noise against a role only when its method relies on exact values.