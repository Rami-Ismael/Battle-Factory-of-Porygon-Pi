# LLM versus diffusion for Bayesian search: evidence and verdict

Reviewed 2026-09-16. Focused primary-source review, not an exhaustive meta-analysis or a reproduction of the experiments. Read alongside [[LLMs as black-box optimizers — evidence and mechanisms]] and [[Diffusion candidate proposers - experiments in related papers]].

## Verdict

There is no established overall winner for Bayesian optimization, and no inspected paper establishes a winner for full VGC team construction. The strongest matched comparison found favors a diffusion language model in mean best-candidate scores on two discrete DNA tasks, while trained autoregressive and diffusion models are approximately tied on two continuous robotics tasks. This is offline optimization. Evidence for sequential Bayesian search is positive for both approaches but is mostly not head-to-head.

For this project, retaining diffusion as the main hypothesis is reasonable; excluding an LLM-based proposer on the assumption that it cannot explore or learn from feedback is not. This recommendation is our analysis, not a result demonstrated on Pokémon.

## Compare methods, not overlapping labels

- **Autoregressive LLM:** generates the serialized design sequentially; may be prompted, fine-tuned, or combined with a separate surrogate and acquisition rule.
- **Diffusion proposer:** generates or edits a design by denoising; may use conditional scores, predictor guidance, or search. It need not be a language model.
- **Diffusion language model:** belongs to both categories in the user's original wording. Recent direct comparisons often study autoregressive versus diffusion language models, not a generic LLM versus the project's small task-specific generator.
- **Online BO:** acquires new expensive labels over rounds, with an acquisition principle based on modeled uncertainty or expected utility. Some generative BO frameworks learn acquisition distributions directly rather than using a separate surrogate.
- **Offline BBO:** uses fixed labeled data to produce candidates, evaluated at the end. Search against a frozen surrogate does not turn it into online oracle-feedback optimization.

An elite-selection/CEM loop is generative black-box optimization. Diffusion or a surrogate alone does not make it Bayesian optimization; name the acquisition principle and uncertainty treatment before using that label.

## Strongest matched comparison

[Sun et al., Training Diffusion Language Models for Black-Box Optimization, v3, Table 2](https://arxiv.org/html/2603.17919v3#S4.T2), revised 2026-05-29, reports final matched-training scores:

| Task | Autoregressive | Diffusion |
|---|---:|---:|
| TFBind8 | 0.915 ± 0.008 | 0.946 ± 0.043 |
| TFBind10 | 0.682 ± 0.053 | 0.741 ± 0.027 |
| Ant | 0.930 ± 0.000 | 0.932 ± 0.022 |
| D'Kitty | 0.912 ± 0.011 | 0.912 ± 0.017 |

These are normalized best-of-128 scores, mean ± SD over eight seeds, using 500 offline observations. Shared adaptation/fine-tuning/RL procedures narrow the much larger comparison against prompted OPRO. Different pretrained backbones remain a confound; these are not win rates or diversity metrics. The v1 figure 0.965 for TFBind8 in earlier vault notes is superseded by v3's 0.946.

## Complementary evidence

| Paper | Role and setting | Findings and boundary |
|---|---|---|
| [LLAMBO — Liu et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/84b8d9fcb4e262fcd429544697e1e720-Abstract-Conference.html) | Prompted GPT-3.5 supports initialization, prediction and candidate generation in sequential HPO. Shared five initial observations followed by 25 trials in the main comparison. | Strong aggregate results versus conventional BO baselines, especially early. Mostly low-dimensional HPO; no diffusion comparison. An LLM can use numerical evaluation history. |
| [When Do LLMs Improve Bayesian Optimization? — Akke et al., 2025 workshop](https://openreview.net/pdf/52aaa8937717bcfe04888b96a89f91b10ad92fd6.pdf) | Reasoning LLMs and agents in molecular/protein optimization. | Results depend on task and budget. GB1 at budget 480: UCB maximum fitness 0.90 ± 0.12 versus GPT-5 0.74 ± 0.18. Tool access is not uniformly beneficial. No diffusion comparator; a workshop study, not a universal ranking. |
| [GOLLuM — Ranković et al., Nature Machine Intelligence 2026](https://www.nature.com/articles/s42256-026-01283-z) | LLM encoder jointly trained with a GP via its marginal likelihood; sequential selection over enumerated chemical candidates. | Across 23 tasks, reported top-5% coverage averages 36.3% versus descriptor GP 29.7%; median 41% fewer iterations to match GP final performance. Evidence for learned language representations plus calibrated BO, not direct open-ended generation. No diffusion comparator. |
| [DiBO — Yun et al., ICML 2025](https://proceedings.mlr.press/v267/yun25a.html) | Task-specific diffusion prior plus uncertainty-estimating ensembles and posterior-guided sampling; sequential continuous optimization. | Reports strong results versus TuRBO and generative/evolutionary baselines. Includes 200/400-dimensional functions and 100–180-dimensional application tasks. No autoregressive LLM head-to-head. This DiBO is different from Sun et al.'s 2026 method. |
| [Diffusion-BBO — Wu et al.](https://arxiv.org/html/2407.00610) | Online score-conditioned generation; uncertainty-aware selection of requested objective values. | Six scientific optimization tasks; reported advantage except TFBind8. Uses 100 new evaluations per round for 16 rounds. Supports online diffusion optimization, not a diffusion-versus-LLM winner. Asking for an arbitrarily higher score does not guarantee better achieved results. |
| [SGPO — Yang et al., 2025](https://arxiv.org/html/2505.15093v2) | Guided protein diffusion compared with autoregressive protein models trained by DPO; includes adaptive optimization with ensemble guidance. | In the 200-label regime, guidance generally outperforms the AR+DPO baseline. Fitness comes from learned oracles fitted to experimental data, not fresh wet-lab assays. Different steering mechanisms and priors prevent attribution solely to architecture; protein language models are not general-purpose chat LLMs. |
| [LaMBO-2 — Gruver et al., NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/29591f355702c3f4436991335784b503-Abstract-Conference.html) | Guided discrete diffusion within multiobjective BO for constrained antibody design. | Includes physical experimental validation; a strong precedent for structured design and limited edits. It does not demonstrate superiority to current prompted LLMs in VGC. |
| [GenBO — Oliveira et al., 2026](https://arxiv.org/html/2510.25240v3) | Acquisition-driven generative training; sequence experiments use categorical models and causal transformers. | Causal-transformer variants match or outperform baselines including guided diffusion LaMBO-2 on reported tasks. This is not a general-purpose LLM win, but demonstrates that diffusion is not required for effective generative BO. Diffusion extensions are not the evaluated generator here. |

The frozen diffusion-language-model/tree-search approach in [Yuan et al., 2026](https://arxiv.org/html/2601.14446v1) is another direct comparison: ten offline observations, 128 final candidates, GP expected-improvement-guided masked tree search. It beats its prompted autoregressive baseline, but changes both the backbone and search machinery. Do not pool its scores with the 500-observation Sun et al. experiment as if the protocols matched.

## What the evidence does not establish

Higher best-of-batch score does not show broader candidate diversity, stronger robustness, lower compute cost, or statistical significance of every mean difference. Identical numbers of expensive evaluations do not imply identical generation/training budgets. Trained versus frozen models test a different question from generation architecture. Shared framework settings do not equalize unknown pretraining information.

The literature supports distinct model roles: a language model may supply an embedding while a GP selects the next candidate; a diffusion generator may supply proposals while an external surrogate ranks them. Comparing these as interchangeable complete algorithms obscures the cause of improvement.

The current team's problem also includes noisy battle estimates, legality dependencies, opponent-conditioned utility and pilot generalization. DNA and robotics benchmark scores do not directly resolve those issues. The existing local diversity results concern the diffusion-plus-selection pipeline and already show why diversity must be measured rather than assumed.

## Proposed VGC decision experiment — our analysis

Hold fixed the format, opponent mixture, pilots, initial labeled teams, legality checker and total battle budget. Compare:

1. Legal mutation of known teams.
2. An autoregressive LLM proposer receiving explicitly selected team–score history, with repeated evaluation-feedback rounds.
3. The current diffusion proposer with the same feedback opportunities.

To isolate candidate generation, use the same surrogate/acquisition rule and equal proposal/evaluated batch sizes. Separately compare full pipelines if the question is practical end-to-end performance. Distinguish frozen versus adapted models; if only diffusion receives weight updates, describe that asymmetry rather than attributing all improvement to architecture. Record retries and generation/training compute separately from battle cost, and acknowledge unequal pretrained knowledge.

Primary endpoint: best team's fresh-game win rate against the fixed population versus cumulative battles, over independent search seeds. Secondary endpoints: legal yield, composition concentration, unique competitive teams above a predeclared performance threshold, reference-corpus novelty, wall time and compute. Use separate opponents or pilots for a transfer test. Define strategic diversity separately from exact uniqueness.

For the blog: "Some diffusion-based optimization methods outperform autoregressive baselines, especially on structured discrete tasks. However, matched training narrows the advantage, and online studies do not establish a universal winner. I am testing whether that advantage transfers to legal Pokémon teams evaluated through noisy battles."
