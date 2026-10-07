---
tags: [paper, generative-models, model-collapse]
source: https://arxiv.org/abs/2307.01850v1
---

# Self-Consuming Generative Models Go MAD

[Paper by Alemohammad et al.](https://arxiv.org/abs/2307.01850v1). This explanation uses the July 2023 arXiv v1; figure and section numbers refer to that version. The work was subsequently published at ICLR 2024.

Open [[Self-Consuming Generative Models Go MAD - visual explanation.excalidraw]] in Obsidian's Excalidraw plugin. The `.excalidraw` file with the same name also opens in [Excalidraw](https://excalidraw.com) using **Open**. Read the board in numbered order.

![[Self-Consuming Generative Models Go MAD - visual explanation.png]]

## The question

If the next generator learns from the current generator's outputs, do we get better models just by producing more training examples?

The problem is that the generated dataset reflects an imperfect approximation of reality. Repeated retraining can pass errors forward and narrow the range of examples the model produces. The authors call deterioration across generations **Model Autophagy Disorder (MAD)**. A generation here is a newly trained model, not one diffusion denoising step or one training epoch. Their formal setup trains each generation from scratch.

## Walk through the drawing

1. **Follow the loop.** Train the first model on real examples, generate a synthetic dataset, and train the next model on that dataset. Repeat. More examples need not provide more independent information about the target distribution.
2. **Ask where the real data comes from.** The paper separates synthetic-only training, reuse of a fixed real dataset alongside synthetic data, and independent fresh real samples alongside synthetic data. The fixed-data experiments slow deterioration; sufficient fresh data can stabilize performance.
3. **Separate realism from coverage.** Imagine generating readable handwritten digits but only a few handwriting styles for each digit. Individual images can look excellent while much of the real variation has disappeared. The dot clusters illustrate this distinction; they are not measured data.
4. **Read both metrics.** Generative precision estimates realism; generative recall estimates coverage of real variation. These are distributional measures, not classification accuracy. FID summarizes mismatch: lower is better. The MNIST experiments use LeNet features instead of Inception features but retain the name FID.
5. **Match the claim to the evidence.** The paper combines Gaussian analysis with image-generator experiments. It does not experimentally establish inevitable collapse for every language model, every synthetic dataset, or every training procedure.

## Why cherry-picking can backfire

The authors use a sampling-bias parameter, **lambda**, to represent favoring high-quality, typical outputs. Lambda = 1 means unbiased sampling in their setup; smaller values favor a narrower set of outputs. Its implementation differs by model: covariance scaling for Gaussians, latent truncation for StyleGAN, and guidance for diffusion.

In **Figure 6**, precision stays high or improves while recall falls sharply. Training repeatedly on attractive samples can therefore hide a worsening loss of diversity. **Figure 7** visualizes clusters shrinking around a few variations.

## The smallest mathematical example

Fit a Gaussian to a finite sample from the previous Gaussian. Its estimated mean changes randomly; its estimated covariance also changes. With sampling bias:

$$\mathbb{E}[\Sigma_t\mid\Sigma_{t-1}]=\lambda\Sigma_{t-1}.$$

For lambda below 1, expected spread contracts each round. Section 3.1 also establishes almost-sure covariance collapse for its finite-sample Gaussian process when lambda = 1. Constant expected covariance at lambda = 1 does not mean that typical individual trajectories maintain their variance. This is a result for the stated toy process, not a universal theorem about all neural networks.

## What fresh data changes

Independent samples from the target provide information that does not originate in the last model's approximation. **Figure 10** shows performance approaching a level that depends on the ongoing real/synthetic mixture and sampling bias, with the initial model's advantage fading.

**Figures 11-12** examine the Gaussian fresh-data setting more closely. Modest synthetic additions can improve effective sample size, while too many biased samples can make it worse than training on the fresh real samples alone. These experiments do not establish one universally safe synthetic-data percentage. A stable limiting error also need not be zero or be acceptably small.

One useful experimental detail: the fixed-real StyleGAN experiment accumulates previous synthetic datasets; the fixed-real MNIST diffusion experiment uses synthetic samples from only the immediately preceding model. The two experiments do not have identical data mixtures.

## Connection to your Pokémon project — interpretation, not a paper result

If a team generator repeatedly trains on its own favored teams, it may become concentrated around a few archetypes. A convincing or strong-looking individual team would not tell you whether useful strategies have disappeared from its proposals.

Your optimization target differs from reproducing an image distribution: concentrating on strong teams can be intentional. The question is whether narrowing the proposal distribution loses strategically useful alternatives. A battle simulator provides external outcome feedback, which is an additional mechanism beyond simply copying a generator's outputs; this paper does not test that setting.

A useful experiment would compare generators across retraining rounds on both battle performance and team/strategy coverage, using a consistent external evaluation. This is an application idea, not a conclusion demonstrated by the paper.
