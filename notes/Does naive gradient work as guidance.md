
- [ ] The honestly a great question is the training distribution by adding non legal pokemon team combination for example magikarp learning thunderbolt the latent structure include junk I thought in the future there will be pokemon format in smogon illegal pokemon format are legal there is a chance is hurting only hurting the legal pokemon if 
- [ ] Ask claude code to go through this youtube video [https://www.youtube.com/watch?v=8W9qHXN0weg&t=1587s](https://www.youtube.com/watch?v=8W9qHXN0weg&t=1587s), I was wondering my basic implementation i worked with fable from claude hurt the latent data structure
- [ ] Check my implementation diffusion model in where I search for new pokemon team that can beat the meta I was going through this youtube video [https://www.youtube.com/watch?v=8W9qHXN0weg&t=1587s](https://www.youtube.com/watch?v=8W9qHXN0weg&t=1587s) I think there could be a possible mistake with my implementation with naive gradient guidances if there are difference run an experiment to compare the performances
- [ ] I thought classifier free guidance is a form of guidance or it something else that classifer free guidance is not good for cross entropy method
- [ ] What is gloss

Answered 2026-09-01 (`src/gradguide.py`, `results/gradguide.json` in the repo): yes — naive gradient guidance works here, 0.230 win rate vs 0.140 unguided p0. The talk's look-ahead form ties it (0.231) and only softens the validity cost (0.82 vs 0.74); constrained decoding absorbs the structure damage.

Combined loop, same day (`src/gradloop.py`, `results/gradloop.json`): guidance + acquisition + re-steer compounds — 0.262→0.334 over 4 generations vs 0.218 for the unguided active arm, still rising at budget end. Guidance without the re-steer plateaus at 0.27.

Extended to saturation: 0.339→0.411 by generation 10, stopped at 11 on diversity (176/512 distinct sets), not win rate. Best team re-battled at 192 battles: 0.589 ± 0.036, above the real-team mean 0.4625 — garchomp/kingambit/basculegion/whimsicott/floette-eternal/glimmora (`results/rebattle_top.json`).

Her fix applied faithfully (`gjac`: autograd through the denoiser, gradguide.json): it keeps validity ≥ 0.90 at every strength, as her theorem says, but wins less — 0.209 vs 0.231 at matched strength — and at 3× strength over-fits the ridge (predicted 0.31, real 0.16). The constrained decoder already gives the protection, so the extra term only buys validity we discard for free.

gjac in the loop (gradloop.json): λ=108 (5–10× the gloss loop's effective tilt) peaked 0.283 at g2 then collapsed, stopped g5; strength-matched λ=24 stayed at 0.26 (0.263/0.234/0.252/0.245), stopped g4. The gloss loop reached 0.411. Faithful guidance stays regularized to p0; the climb needs the prior loosened.

On classifier-free guidance: yes, it is guidance — it steers by extrapolating the conditional vs unconditional prediction, no gradient. It is orthogonal to the cross-entropy-method re-steer (finetune on elites), not bad for it. It failed here on its own: the win-rate token gave ~1× mass lift (2026-08-29).

What gloss is: my mode name for her look-ahead-loss guidance G_loss (Guo et al. 2024) — tilt by the ridge gradient at the model's expected completion, scaled by the gap to the target y. `gjac` is the full form with autograd through the denoiser; `naive` is the raw gradient at the partial team.