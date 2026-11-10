# Uniform-noise reverse sampling: token departures, returns, and confidence

Research date: 2026-11-09. Read the supplied earlier note first. Paper references use Schiff et al., *Simple Guidance Mechanisms for Discrete Diffusion Models*, December 2024 version, arXiv:2412.10193v1, Appendix A, Eqs. 20–25. The equations also appear with the same numbers in v3. Upstream code is pinned to `edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661`, the repository HEAD returned by `gh api` during this research. Read `diffusion.py` using the requested `gh api .../contents/diffusion.py?ref=<SHA> -q .content | base64 -d` method. Calculations below are deterministic NumPy evaluations, with no model execution, training, or Monte Carlo sampling.

## How much re-editing comes from the posterior itself versus changed predictions?

### Takeaway

**Inference, from Eq. 25:** a certain prediction still permits random departures. Three different quantities must be kept separate:

| Quantity | Meaning | Example: 128 steps, 189 categories, linear schedule |
|---|---|---:|
| Conditional departure sum, H | Sum of one-step departure probabilities evaluated as if the model always endorsed the current token | 1.444097 |
| Actual edits after first reaching a fixed predicted target, R | Includes departures, returns, and changes between wrong tokens after the first hit; excludes the first hit | 0.095813 |
| Remaining edits conditional on already being at the target at initialization | Same fixed predictor, but conditions on an unusually early first hit | 5.396629 |

**Inference:** these are different experiments, not alternative estimates of one count. The formulas defining H and R and the conditional recurrence appear below. The source posterior is [Appendix A, Eq. 25](https://arxiv.org/html/2412.10193v1#A1). “Final value” below means a fixed clean target x that the predictor endorses throughout; with a positive stopping time the last sampled token can still differ from x. Conditioning instead on the realized endpoint would define a different bridge process.

### Cited findings

- Eqs. 20–21 specify the forward interpolation and transition, Eq. 22 applies Bayes' rule, Eqs. 23–24 specialize to uniform noise, and Eq. 25 substitutes a predicted clean-token probability vector for the clean one-hot token. [Paper, Appendix A](https://arxiv.org/html/2412.10193v1#A1)
- `_compute_posterior` implements Eq. 25; `_ddpm_denoise` passes the model probabilities to it and samples its output. [diffusion.py:356–381](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L356-L381), [diffusion.py:1220–1246](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1220-L1246)
- The continuous-time sampling branch constructs K+1 equally spaced times from 1 to `eps`, takes K reverse transitions, and obtains alpha through `1-move_chance`. [diffusion.py:1126–1155](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1126-L1155)

### Inferences

#### One-step derivation and confidence

**Inference, Eq. 25:** write i for the current token, p for the predicted clean-token distribution, c=p_i, a=alpha_t, b=alpha_s, r=a/b, and d=b-a, with s<t and b>0. Let

```text
D = N*a*c + 1-a
P(z_s=j | z_t=i) = [N*a*1[j=i]*p_j + (r-a)*1[j=i]
                    + d*p_j + d*(1-b)/(N*b)] / D.            (1)
```

This is the component form of [Eq. 25](https://arxiv.org/html/2412.10193v1#A1) and [diffusion.py:369–380](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L369-L380).

**Inference, summing (1) over j≠i:** the exact probability of a value change is

```text
J(c;t,s,N) = d * [(1-c) + (N-1)*(1-b)/(N*b)] / [N*a*c+1-a]
          = (1-r) * [1 - b*c - (1-b)/N] / [N*a*c+1-a].       (2)
```

Confidence must mean probability assigned to the *current* token, not maximum softmax confidence. No other detail of p affects total unguided change probability, though p determines the destination. This is the plug-in posterior used here, not a mixture of separately normalized clean-token posteriors. [Eq. 25](https://arxiv.org/html/2412.10193v1#A1), [implementation](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L356-L381)

**Inference, substitutions in (2):**

| Model prediction | c | Exact change probability |
|---|---:|---|
| Certain of current token, case (a) | 1 | A = (d/b)*(N-1)/N*(1-b)/[1+(N-1)*a] |
| Certain of a different token, case (b) | 0 | B = (d/b)*[1-(1-b)/N]/(1-a) |
| Uniform clean-token prediction, case (c) | 1/N | U = (d/b)*(N-1)/N |

**Inference, differentiating (2):** J decreases with c. If C0=(N-1)*(1-b)/(N*b), then

```text
dJ/dc = -d * [(1-a) + N*a*(1+C0)] / [N*a*c+1-a]^2 <= 0.   (3)
```

For 0<t<1 and a positive-sized reverse step, A≤U≤B. At t=1, a=0, all three can still differ because b>0. There is no guarantee of retaining the token merely because the model is certain. [Source posterior](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L369-L380)

**Inference, splitting (2):** for this plug-in parameterization the off-diagonal mass separates exactly into

```text
model-directed mass = d*(1-c)/D
uniform-noise mass  = d*(N-1)*(1-b)/(N*b*D).                (4)
```

These are nonnegative probability contributions, not observed event labels. When c=1, only the noise term remains and each alternative gets A/(N-1). When c=0, the noise term still exists; the predicted alternative additionally receives d/D. A changed prediction is therefore sufficient to raise change probability when it reduces c, but is not necessary for repeated edits. With a frozen predictor pointing at x, a noise departure makes x different from the *current* token on the next step; its return then uses case (b) without any change in the model's prediction. [Eq. 25](https://arxiv.org/html/2412.10193v1#A1), [code](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L369-L380)

#### Full-run sums and the first-hit distinction

**Inference, substituting the requested schedule:** use

```text
t_i = 1 - i*(1-10^-5)/K,  i=0,...,K; s_i=t_(i+1)
alpha(t) = 1-t^e
A_i = (t_i^e-s_i^e)/(1-s_i^e) * (N-1)/N
      * s_i^e/[N-(N-1)*t_i^e].                           (5)
H_j = sum_(i=j)^(K-1) A_i.                               (6)
P(at least one departure after a hit at t_j)
    = 1 - product_(i=j)^(K-1)(1-A_i).                    (7)
```

H_j is the accumulated conditional departure probability along the path that remains correct. It is also the exact expected change count in a different process where the predictor switches to endorse the current token after *every* draw. It is not the expected count under a fixed final prediction. Formula (7) is exact under a fixed predictor because, before the first departure, every step is in case (a). The grid is from [diffusion.py:1126–1145](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1126-L1145); (5) is derived from [Eq. 25](https://arxiv.org/html/2412.10193v1#A1).

**Inference, exact fixed-target recurrence:** let x be the model's unchanged one-hot prediction. In addition to A_i and B_i, the probability of entering x from any non-x token is

```text
E_i = (d_i/b_i) * [b_i+(1-b_i)/N]/(1-a_i).              (8)
```

Given the first hit at t_j, initialize v_j=1 and compute

```text
v_(i+1) = v_i*(1-A_i) + (1-v_i)*E_i
C_j = sum_(i=j)^(K-1) [v_i*A_i + (1-v_i)*B_i].          (9)
```

Here v_i is the probability of being at x at step i conditional on that first hit. C_j counts every subsequent edit, including returns. This answers “after first reaching its final value” for a specified arrival step; without that step or its distribution there is no unique answer. Formula (8) selects destination x in (1); (9) propagates that two-state Markov chain. [Source posterior](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L369-L380)

**Inference, full-run average with a fixed predictor from initialization:** the ideal alpha(1)=0 makes the uniform prior exactly consistent with the forward marginal. Let v_0=1/N and u_0=1-1/N, where u_i is the probability of never having visited x. Then

```text
u_(i+1) = u_i*(1-E_i)
v_(i+1) = v_i*(1-A_i)+(1-v_i)*E_i
R = sum_i [v_i*A_i + (1-v_i-u_i)*B_i].                  (10)
```

The first term counts departures after a visit; the second counts changes while away after a visit. Transitions making the first visit are excluded. Initialization at x counts as having reached x. An initialization that must first be changed would require a different event definition. Equivalently R is the first-hit weighted average of C_j, with weights h_0=1/N and h_(j+1)=u_j*E_j. [Source posterior](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L369-L380), [uniform prior](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L855-L862)

**Inference, numerical results from (5), (6), and (10):** all entries are expected counts per position. Vocabulary sizes and exponents are crossed independently; this does not assert a field-to-vocabulary mapping. The e=1 rows give the requested ideal log-linear schedule alpha=1-t.

| Exponent e | Vocabulary N | H, K=84 | H, K=128 | H, K=256 | R after first hit, K=84 | R, K=128 | R, K=256 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 189 | 0.834964 | 0.902707 | 0.995081 | 0.020802 | 0.027387 | 0.040487 |
| 16 | 300 | 0.831586 | 0.895271 | 0.975301 | 0.013146 | 0.017312 | 0.025606 |
| 16 | 500 | 0.829233 | 0.890073 | 0.961266 | 0.007904 | 0.010411 | 0.015404 |
| 16 | 33 | 0.870919 | 0.980220 | 1.184015 | 0.114513 | 0.150195 | 0.220631 |
| 4 | 189 | 1.030561 | 1.093657 | 1.233819 | 0.046064 | 0.056138 | 0.074916 |
| 4 | 300 | 1.003405 | 1.050832 | 1.152633 | 0.029139 | 0.035522 | 0.047428 |
| 4 | 500 | 0.983962 | 1.019586 | 1.090594 | 0.017531 | 0.021377 | 0.028552 |
| 4 | 33 | 1.277594 | 1.449747 | 1.800952 | 0.250438 | 0.303993 | 0.403054 |
| 1.5 | 189 | 1.201119 | 1.310661 | 1.558707 | 0.070605 | 0.083124 | 0.105796 |
| 1.5 | 300 | 1.128911 | 1.208995 | 1.398040 | 0.044694 | 0.052636 | 0.067026 |
| 1.5 | 500 | 1.074313 | 1.129336 | 1.262805 | 0.026904 | 0.031691 | 0.040369 |
| 1.5 | 33 | 1.723545 | 1.968896 | 2.434141 | 0.380412 | 0.446089 | 0.564123 |
| 1 | 189 | 1.305353 | 1.444097 | 1.746676 | 0.082290 | 0.095813 | 0.120068 |
| 1 | 300 | 1.205097 | 1.309280 | 1.548740 | 0.052106 | 0.060688 | 0.076090 |
| 1 | 500 | 1.126660 | 1.199284 | 1.374531 | 0.031372 | 0.036547 | 0.045838 |
| 1 | 33 | 1.957107 | 2.230755 | 2.735808 | 0.441741 | 0.512295 | 0.637903 |

**Inference, arrival-time sensitivity using (7) and (9):** for N=189, K=128, e=1:

| First target hit | Grid time t_j | Expected remaining edits C_j | Probability of any departure |
|---|---:|---:|---:|
| Initialization | 1.000000 | 5.39663 | 0.99196 |
| After 25% of steps | 0.750003 | 0.0212441 | 0.00831677 |
| After 50% | 0.500005 | 0.00356764 | 0.00158776 |
| After 75% | 0.250007 | 0.000487868 | 0.000233104 |
| After about 90% | 0.101571 | 5.8779e-05 | 2.89532e-05 |

**Inference:** a token first becoming correct halfway through the linear run has only about 0.00357 further edits in expectation in this frozen, certain-target model. The larger H totals arise mainly from conditioning on correctness at the almost fully noisy starting point. More steps increase these finite-grid counts by resolving excursions that a coarse grid can skip; they do not make the count proportional to K. [Posterior and grid](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1126-L1155)

### Gaps

- These are controlled posterior calculations, not measurements of a trained model. Confidence and target predictions in a real model depend on the whole evolving sequence. No prediction-stability assumption is established by the paper or sampler.
- The requested alpha=1-t is idealized. The upstream noise class defaults to internal epsilon 0.001, giving alpha=1-0.999t, while the sampling endpoint is a separate epsilon 10^-5. The tables intentionally use the user's alpha=1-t^e exactly. [noise_schedule.py:134–153](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/noise_schedule.py#L134-L153), [factory:13–18](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/noise_schedule.py#L13-L18), [sampling grid](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1126-L1128)
- The inspected project snapshot uses these powers for **masking**, not uniform token replacement: `diffusion84.py:40–47,137–142`; its `sample` fills one column per iteration at lines 367–381. Thus the power-schedule tables analyze a proposed uniform-noise posterior with those powers, not the deployed sampler's actual edit rate. Pinned project source: [schedule](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/f9f4e94614c8758ca2cec3c8f365a788e546b13a/code/src/diffusion84.py#L40-L47), [masking](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/f9f4e94614c8758ca2cec3c8f365a788e546b13a/code/src/diffusion84.py#L137-L142), [sampling](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/f9f4e94614c8758ca2cec3c8f365a788e546b13a/code/src/diffusion84.py#L367-L381).

## At which part of the time grid do most changes happen?

### Takeaway

**Inference, (5) and (10):** departures from a token the model already endorses are concentrated near t=1. Larger exponents move the noise-driven re-editing earlier on the same equally spaced time grid. This statement concerns a certain, fixed predictor, not an uncertain learned sampler. [Posterior](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L369-L380), [grid](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1126-L1145)

### Cited findings

- The sampler is uniform in t, not in alpha or in log noise. [diffusion.py:1126–1145](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1126-L1145)
- Eq. 25 explicitly contains both alpha_s-alpha_t and a uniform off-diagonal contribution. [Appendix A](https://arxiv.org/html/2412.10193v1#A1)

### Inferences

**Inference, summing the terms of (10) in four equal groups:** the following percentages locate actual post-first-hit edits R for N=189, K=128. Time decreases left to right. All rows sum to 100% before rounding.

| Exponent e | First 32 steps, t≈1 to .75 | Next 32, .75 to .5 | Next 32, .5 to .25 | Last 32, .25 to 10^-5 |
|---:|---:|---:|---:|---:|
| 16 | 99.451% | 0.548% | 0.001% | 0.000% |
| 4 | 85.690% | 11.857% | 2.304% | 0.149% |
| 1.5 | 68.183% | 18.219% | 9.406% | 4.192% |
| 1 | 61.294% | 18.690% | 11.616% | 8.400% |

**Inference, late-time expansion of (5):** when t is small and the step is locally fine, s≈t and d≈e*t^(e-1)*dt, so

```text
A ≈ e*(N-1)/N^2 * t^(2e-1)*dt.                       (11)
```

The probability of leaving a correct token decays especially quickly for e=16. But when the current token is wrong, (2) with c=0 gives

```text
B ≈ e*dt/t                                           (12)
```

for dt≪t≪1. Corrections can therefore remain likely late in the run. At the final finite step, use (2), not (12), because dt/t need not be small. The exact final B approaches 1 as s approaches zero. [Eq. 25](https://arxiv.org/html/2412.10193v1#A1)

**Inference, endpoint behavior:** t=1 is special because alpha_t=0. Its certain-current departure probability is A_0=(N-1)/N*s_0^e. For N=189, K=128, e=1, A_0≈0.98694. Yet only 1/189 of uniform initial tokens equal a preselected fixed target. This explains why a very high conditional initial departure probability can coexist with a small average count after the target's first hit. Source formulas are (5) and the [uniform prior](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L855-L862).

### Gaps

A changed model prediction can shift edits into later steps. Neither the time-share table nor the certain-target calculation establishes where a trained team's sampler actually revises tokens.

## Does classifier-free guidance raise or lower the change probability?

### Takeaway

**Inference, guided posterior odds:** gamma>1 has no universal direction. It lowers change probability when the conditional posterior favors staying more strongly than the unconditional posterior; it raises it when the conditional posterior favors alternatives more strongly. If the two posteriors agree, guidance has no effect. [Paper Eq. 8](https://arxiv.org/html/2412.10193v1#S3.SS1), [diffusion.py:1314–1328](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1314-L1328)

### Cited findings

- The uniform-noise branch computes the conditional and unconditional reverse posteriors separately, then applies a softmax to gamma*log q_cond+(1-gamma)*log q_uncond. It does not first mix clean-token logits and then compute one posterior. [diffusion.py:1314–1328](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1314-L1328)
- The paper's discrete classifier-free guidance factorizes this normalized geometric combination per position. [Section 3.1, Eq. 8](https://arxiv.org/html/2412.10193v1#S3.SS1)

### Inferences

**Inference:** let C_j and U_j denote the two posterior probabilities, i the current token, and gamma the guidance strength. Then

```text
G_j = C_j^gamma * U_j^(1-gamma) / sum_k C_k^gamma*U_k^(1-gamma)
J_gamma = 1-G_i
        = S_gamma/(1+S_gamma)
S_gamma = sum_(j!=i) (C_j/C_i)^gamma * (U_j/U_i)^(1-gamma). (13)
```

The necessary and sufficient criterion for raising change probability relative to ordinary conditional sampling is S_gamma>S_1. A sufficient condition for lowering it for every gamma>1 is C_j/C_i≤U_j/U_i for every alternative j; reverse all inequalities for a sufficient condition for raising it. [Mixing code](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1316-L1328)

**Inference, common confident cases from (1) and (13):**

- Both networks certain of the same current token: C=U, so J_gamma=A exactly. Guidance does not remove the posterior's random departures.
- Conditional network certain of the current token, unconditional network uniform over clean tokens: every off-diagonal conditional odds ratio is smaller. Guidance suppresses departures.
- Conditional network certain of a different token x, unconditional network uniform: the odds of x relative to the current token increase; odds of other alternatives relative to the current token are unchanged. Guidance increases total change probability.
- Both networks certain of the same different token: again C=U, so guidance does nothing. Confidence by itself does not establish a guidance effect.

All four are substitutions into [Eq. 25](https://arxiv.org/html/2412.10193v1#A1) and the [implemented posterior combination](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1314-L1328).

**Inference, numeric check of (13):** at t=.5, s=.49, N=189, with an unconditional uniform clean-token prediction:

| Conditional prediction | Change probability, gamma=1 | Change probability, gamma=2 |
|---|---:|---:|
| Certain of current token | 0.0001006001 | 0.0000005089 |
| Certain of a different token | 0.0391140160 | 0.8060495419 |

### Gaps

Without both conditional and unconditional predicted distributions, a guidance strength alone cannot identify an edit rate. Counts in the first section are unguided; applying them directly to gamma>1 would omit the normalization in (13).

### Reproducible numeric check

**Inference / verification:** the following uses NumPy and Python's standard library. It computes the full crossed table with cancellation-resistant t^e differences. A separate 1,000-case check against the full Eq. 25 vector, using random simplex predictions and valid alpha pairs, gave maximum normalization/change-formula error 3.34×10^-15. Runtime was under one second in this environment.

```python
import numpy as np

def grid(N, e, K):
    g = np.linspace(1., 1e-5, K+1)
    rt, rs = g[:-1]**e, g[1:]**e
    a, b, d = 1-rt, 1-rs, rt-rs
    A = d/b * (N-1)/N * rs/(N-(N-1)*rt)
    B = d/b * (1-rs/N)/rt
    E = d/b * (b+rs/N)/rt
    return A, B, E

for e in (16, 4, 1.5, 1):
    for N in (189, 300, 500, 33):
        for K in (84, 128, 256):
            A, B, E = grid(N, e, K)
            v, u, R = 1/N, 1-1/N, 0.
            for ai, bi, ei in zip(A, B, E):
                R += v*ai + (1-v-u)*bi
                u *= 1-ei
                v = v*(1-ai)+(1-v)*ei
            assert abs(v-(1-(1-1/N)*1e-5**e)) < 2e-12
            print(e, N, K, 'H=', A.sum(), 'R=', R)
```
