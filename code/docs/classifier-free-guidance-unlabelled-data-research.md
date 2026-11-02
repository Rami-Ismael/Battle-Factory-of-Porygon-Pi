# Classifier-free guidance and unlabelled data: what the primary sources support

Researched 2026-11-02 from the papers themselves, read through the alphaXiv PDF query tool (it returns the relevant pages, not always the whole paper). "Stated" means a source says it. "Inference" means I derived it; no source says it.

## What the code does today

- `batch_loss` in `code/src/diffusion84.py` sets the win-rate label to NaN for each row with probability `p_uncond = 0.15`. Rows that have no label already hold NaN (`train_new`).
- `Net84.wvec` maps every NaN to the same learned vector `wnull`. Dropped labels and label-less teams therefore take one path.
- `sample` guides in log-probabilities: `guidance * cond + (1 - guidance) * null`. Guidance 2 equals Ho and Salimans' weight w = 1, because their (1 + w) is this code's `guidance`.
- Arithmetic from the table: 304,580 teams; 148,196 conditional rows per epoch; 156,384 null-path rows (51.3%), of which 83.3% are label-less teams and 16.7% are dropped labels. A dropout-only table of the 174,348 labelled teams would give the same 148,196 conditional rows and 15% null.

## 1. What each method trains

**Classifier guidance** (Dhariwal and Nichol 2021, §4.1 Eq. 2, §4.3): a diffusion model p(x_t | x_{t+1}) plus a separate classifier p_φ(y | x_t, t) trained on noised data. Sampling uses p(x_t | x_{t+1}) · p_φ(y | x_t)^s, so the score gets s · ∇ log p_φ(y | x_t). The diffusion model may be unconditional (it then estimates p(x)) or conditional (§4.3, Table 4: a guided unconditional model reaches Fréchet Inception Distance (FID) 12.00; an unguided conditional one 10.94).

**Classifier-free guidance** (Ho and Salimans, §3.2, Eq. 6, Algorithms 1-2): one network, with a null token ∅ in place of the label. Sampling uses ε̃ = (1 + w) ε(z, c) − w ε(z). The unconditional term estimates the score of the marginal p(z_λ), the data distribution with the label ignored.

**How dropout trains p(x)** (stated, Algorithm 1 line 3): with probability `p_uncond` the label is replaced by ∅ and the usual denoising loss is taken. Fu et al. (2024, §2 Eq. 2.4, Appendix D.1 Eq. D.1) write this as a mixed objective whose second half is the unconditional score error against ∇ log p_t(x_t). Inference: the loss on ∅ rows is minimised by the score of whatever distribution the ∅ rows come from. Dropout draws those rows from the labelled rows' marginal, since the coin flip ignores x and y.

**Discrete form** (Schiff et al. 2025, §2.2 Eq. 4, §3.1 Eqs. 6-8): p^γ ∝ p(· | y)^γ · p(·)^(1−γ), the form this code uses. Their training drops the condition at 10% (Appendix D.4.2).

## 2. What the guided direction means

Stated (Ho and Salimans §3.2): the direction is "inspired by" the gradient of an implicit classifier p_i(c | z) ∝ p(z | c) / p(z), with gradient −(1/σ)[ε*(z, c) − ε*(z)] for exact scores. Guiding with weight w targets p(z | c)^(1+w) · p(z)^(−w).

Stated caveats:
- Learned scores come from unconstrained networks, so they need not be gradients of any classifier, and Bayes-inverted generative classifiers can be inconsistent under misspecification (Ho and Salimans §3.2).
- The sampler does not follow the density p_w, because p_w is not a valid heat diffusion of its t = 0 form (Karras et al. 2024, §2).
- The decomposition in Dhariwal and Nichol Appendix H rests on defining q̂(x_{t+1} | x_t, y) = q(x_{t+1} | x_t), and it failed to hold in one-dimensional experiments (Zhao and Schwing 2025, §3.2 Eq. 5, Fig. 1).
- Pavasovic et al. (2025, §4.1) assume exact scores; they say practical benefits may tie to score errors.

Assumption about x: Bayes' rule needs p(x) to be the marginal of the same joint distribution that gives p(x | c). Ho and Salimans §5 write Σ_c p(x | c) p(c) = p(x). Dhariwal and Nichol Appendix H sets q̂(x_0) := q(x_0). Fu et al. take the unconditional target from the same pairs (x_0, y). No source states what happens when the two terms use different x distributions.

Inference for this project: let the null branch learn p_null = 0.167 · p_labelled + 0.833 · p_label-less. Then the score difference is ∇ log p_l(y | x) + ∇ log [p_l(x) / p_null(x)]. The second term is the gradient of the log-probability that x came from the labelled source. It vanishes only if both sources share one x distribution, and the weight 2 scales it like the label term.

## 3. Unconditional branch on unlabelled data

No source I found compares classifier-free guidance with and without unlabelled rows in the null branch, against a dropout-only control with matched labelled data. Nearest evidence:

| Source | What it did | Result |
|---|---|---|
| MakeSinger (Kim et al. 2024, §2.1, §3.3, Tables 1-2) | Unlabelled songs take ∅ for text and pitch; labelled 12 / 36 / 108 songs, unlabelled from the same corpus | Gains at all three sizes. At 12 labelled songs: pitch error 0.2543 → 0.1736, overall listener score 2.54 → 3.48. At 108: 3.76 → 4.08. The gap shrinks as labelled data grows |
| Guided-TTS 2 (Kim et al. 2022, §3.1, §5.1 Table 1) | Adds 6,300 h of untranscribed Libri-Light to LibriTTS; null speaker embedding 50% | Zero-shot listener score 4.16 → 4.23, character error 1.03% → 0.89%. Fine-tuned: 4.22 → 4.21, 1.16% → 1.12%. Authors: LibriTTS is "sufficient" for fine-tuning |
| Label-augmented conditional diffusion (Su et al. 2026 preprint, §2.3 Eq. 5, Figs. 3 and 5) | Unlabelled rows get ∅ in joint score matching; conditional sampling, no guidance weight | CIFAR-10, 1,000 labels: class-averaged FID 28.75 → 14.94 at 10× unlabelled |
| Dual pseudo training (You et al. 2023, §3.2, §6) | Pseudo-labels for every row | Says it is unclear how to use unlabelled data with classifier-free guidance; low-confidence pseudo-labels to ∅ is an untested suggestion |
| Independent condition guidance (Sadat et al. 2025, §6.1, Fig. 3) | Removes label dropping; an unrelated condition stands in for ∅ | Purely conditional model beat p = 0.1 dropout at every checkpoint: about 30% faster or about 20% lower FID |

Nulls and negatives:
- Label-augmented conditional diffusion got worse before better at small ratios: CIFAR-10, 1,000 labels, overall FID 9.33 → 9.75 → 10.00 at 0×, 1×, 2× unlabelled, then 7.76 and lower. Intel Scenes, 1,200 labels: overall FID 9.55 → 12.52 at 1×. The authors note "mild fluctuations".
- Guided-TTS 2 fine-tuned numbers sit inside the stated ±0.09 intervals.
- Confounds: in MakeSinger and Guided-TTS 2 the unlabelled data also trains the shared network and the conditional branch, and the compared systems differ in more than the data (MakeSinger adds dual guidance and pitch-only rows). None isolates the null branch. In Guided-TTS 2 every row has a speaker embedding taken from its audio, so no row lacks a condition.
- Dhariwal and Nichol §7 propose using unlabelled data to pre-train, but run no experiment.
- Mechanism in the label-augmented paper (§2.2): every conditional equals a shared baseline plus a label part, b0 + Φ e_y. This network keeps `wnull` free, so the mechanism may not carry over (inference).

## 4. Unlabelled data from a different distribution

No primary source tests, under classifier-free guidance, an unconditional branch fed from a different distribution than the labelled data. Related results:
- The label-augmented paper assumes labels are missing independently of (X, Y) (§2.1) and lists dependent missingness as future work (§5).
- Autoguidance (Karras et al. 2024, §4, §5.1): guidance helps only when both models share task, conditioning and data distribution, with compatible degradations. A dropout-degraded main model with an input-noise-degraded guide gave no gain; the best weight was 1, meaning guidance off. A guide trained on less data did not beat baseline. Models and data differ in kind from this project; the result is about two score fields that disagree for unrelated reasons.
- Guided-TTS (Kim et al. 2022, §3, §5.2): an unconditional model on the target speaker's untranscribed audio, a classifier trained on LibriSpeech. This is classifier guidance, not classifier-free. It worked after norm-based scaling, because plain gradient scales mispronounced "for all cases"; the authors blame the score norm near t = 0, not the mismatch.
- Repo note (not a source): `code/docs/foundation-swap.md:93` calls some unlabelled teams synthetic, and `empty-moves-and-mega-stone-diagnosis.md:52` calls others auxiliary. Whether those sit in the 130,232 is for you to check.

## 5. Effect of the unconditional share

Ho and Salimans §4.2, Table 1, Fig. 4 (64×64 ImageNet, 400,000 steps, `p_uncond` 0.1 / 0.2 / 0.5): 0.5 is "consistently worse" across the Inception Score-FID frontier; 0.1 and 0.2 are about equal. At w = 0 FID is 1.80 / 1.80 / 2.21; at w = 3, FID 24.83 / 22.36 / 19.75 with Inception Score 250.4 / 237.65 / 208.9, so the verdict rests on the frontier. They conclude only a small share of capacity need go to the unconditional task. One model, one dataset, no seed spread reported.

Other settings: GLIDE 20% (§4.2), Guided-TTS 2 50% (§3.1), Schiff 10%, Fu et al. 50% in theory (§2, "no real difficulty" for other rates). These are choices, not studies. Karras et al. (§3, footnote 1) report the unconditional branch is a worse fit (FID 11.67 vs 2.56) and argue that gap drives part of the quality gain.

Gap: no source varies the unconditional share by adding label-less rows. In the sources, a higher share removes conditional rows; here it adds null rows and keeps 148,196 conditional ones (inference).

## What each position has

- "Dropout alone is enough": supported for working guidance (Ho and Salimans §4.2; Sadat §6.1). Not tested against unlabelled rows, so "adds nothing" has no direct support or refutation.
- "Unlabelled data improves the unconditional branch": gains in conditional quality appear in three papers, none isolating the null branch, plus a worse-then-better dip. Nobody measured the unconditional fit with and without unlabelled rows. Karras et al. suggest a stronger unconditional branch could shrink guidance's quality gain (inference).

## Read in full pages versus search-list only

Read: Dhariwal and Nichol; Ho and Salimans; Schiff et al.; Fu et al.; Karras et al.; Sadat et al.; Zhao and Schwing; Pavasovic et al.; GLIDE; MakeSinger; Guided-TTS and 2; Su et al. (preprint); You et al.; Cheng et al. 2026 (unlabelled conditions, off-target).
Seen only in search results: 2502.12154, 2610.16419, 2510.00815, 2607.16725, 2510.03016. Chidambaram et al. and Bradley and Nakkiran are known only through citations.

## Links

- https://arxiv.org/abs/2105.05233 (Dhariwal and Nichol)
- https://arxiv.org/abs/2207.12598 (Ho and Salimans)
- https://arxiv.org/abs/2412.10193 (Schiff et al.)
- https://arxiv.org/abs/2403.11968 (Fu et al.)
- https://arxiv.org/abs/2406.02507 (Karras et al.)
- https://arxiv.org/abs/2407.02687 (Sadat et al.)
- https://arxiv.org/abs/2503.10638 (Zhao and Schwing)
- https://arxiv.org/abs/2502.07849 (Pavasovic et al.)
- https://arxiv.org/abs/2112.10741 (GLIDE)
- https://arxiv.org/abs/2406.05965 (MakeSinger)
- https://arxiv.org/abs/2205.15370 and https://arxiv.org/abs/2111.11755 (Guided-TTS 2, Guided-TTS)
- https://arxiv.org/abs/2607.16685 (Su et al.)
- https://arxiv.org/abs/2302.10586 (You et al.)
- https://arxiv.org/abs/2605.25210 (Cheng et al.)
