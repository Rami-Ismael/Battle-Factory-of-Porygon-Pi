---
created_at: 2026-08-19
updated_at: 2026-08-19
type: reference
title: Action masking (mask-out) in policy-gradient RL — why and how, with VGC-Bench's implementation
sources:
  - "🟢 Huang & Ontañón, A Closer Look at Invalid Action Masking in Policy Gradient Algorithms, FLAIRS 35 (2022), arXiv 2006.14171v3 — https://arxiv.org/abs/2006.14171"
  - "🟢 Zabounidis, Siegelmann, Qadri, Kim, Stepputtis, Sycara, Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms, RLJ 2026 — https://rlj.cs.umass.edu/2026/papers/Paper1.pdf"
  - "🟢 Angliss, Cui, Hu, Rahman, Stone, VGC-Bench, AAMAS 2026, arXiv 2506.10326v3 — https://arxiv.org/abs/2506.10326"
  - "🟢 vgc-bench/vgc_bench/src/policy.py (MaskedActorCriticPolicy) — https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/src/policy.py"
  - "🟢 vgc-bench/vgc_bench/src/policy_player.py, env.py, pyproject.toml — same repo"
  - "🟢 poke-env fork, branch vgc-bench: src/poke_env/environment/env.py and doubles_env.py — https://github.com/cameronangliss/poke-env/tree/vgc-bench"
  - "🟢 sb3-contrib: sb3_contrib/common/maskable/distributions.py, common/wrappers/action_masker.py — https://github.com/Stable-Baselines-Team/stable-baselines3-contrib"
  - "🟢 Huang's reference code, invalid_action_masking/ppo_10x10.py (CategoricalMasked) — https://github.com/vwxyzjn/invalid-action-masking"
tags:
  - reinforcement-learning
  - action-masking
  - policy-gradient
  - ppo
  - vgc-bench
  - poke-env
  - doubles
---
# Rami Todo

- [ ] **Q7** - **Entropy under masking**: PPO adds an entropy bonus to keep the policy exploring. Should entropy be computed over all 107 actions, or only the legal ones?
# Action masking in policy-gradient RL

Companion reference for the battle-policy work. Everything below was read from the paper PDFs / source files listed in the frontmatter; a fetch that failed is marked "not verified".

## 1. Why mask at all — three reasons

**(a) Scale: the penalty approach stops working as the invalid set grows.** Huang & Ontañón compare four regimes in Gym-μRTS (maps 4×4 → 24×24). Invalid-action penalty ($r_{invalid}\in\{0,-0.01,-0.1,-1\}$) reaches the 40-return target only on 4×4 and never on 10×10 or larger, while masking hits 40.00 on every map with $t_{solve}\approx 12\%$ of training: "Invalid action masking is shown to scale well as the number of invalid actions increases … Invalid action penalty … does not scale to larger maps. As the space of invalid action gets larger, sometimes it struggles to even find the very first reward." Also, "the hyper-parameter $r_{invalid}$ can be difficult to tune" ($r_{invalid}=-1$ "discouraging exploration … consistently the worst"). 🟢 Huang & Ontañón, Table 2 + "Evaluation Results". In VGC the action space is 107 per slot (2 slots ⇒ $107^2$ joint) and most of it is invalid on any given turn, so this is the regime that matters.

**(b) Valid action suppression — the mechanism behind (a).** Zabounidis et al. (RLJ 2026): "when an action is invalid at visited states, policy gradients decrease their probabilities; shared parameters propagate this decrease to unvisited states where those actions are valid, causing exponential suppression before the agent reaches them." Theorem 1: with shared features, after $T$ steps $\pi_T(a\mid s^*)\le e^{-K_T}/n$. "Entropy regularization is reactive rather than preventive". Masking kills this because invalid actions "receive zero probability and contribute zero gradient". 🟢 Zabounidis et al. §1, §3, §4. Pokémon analogue: Tera / a 4th move that is rarely legal gets driven to ~0 before the policy ever sees the turn where it wins.

**(c) It is theoretically free.** The masked gradient is still the policy gradient of a valid policy (Prop. 1 below), so nothing is traded for the gain. 🟢 Huang & Ontañón, "Masking Still Produces a Valid Policy Gradient".

## 2. How — masked softmax and the zero-gradient argument

Policy $\pi_\theta(\cdot\mid s)=\mathrm{softmax}(l(s))$. Masking is a **state-dependent** function applied to the logits:

$$\pi'_\theta(\cdot\mid s)=\mathrm{softmax}(\mathit{mask}(l(s))),\qquad \mathit{mask}(l(s))_i=\begin{cases}l_i & a_i\ \text{valid in } s\\ M & \text{otherwise}\end{cases}$$

with $M$ a large negative number ("e.g. $M=-1\times10^8$"). 🟢 Huang & Ontañón, Eq. 2–4 + Prop. 1 proof. Equivalently (Zabounidis Eq. 2) $\pi^{oracle}_\theta(a\mid s)=\exp(z_a)\,\nu(s,a)/\sum_j\exp(z_j)\,\nu(s,j)$ with validity indicator $\nu\in\{0,1\}$.

**Zero gradient.** Worked example, 4 actions, $a_2$ invalid: the unmasked policy gradient for sampling $a_0$ is $[0.75,-0.25,-0.25,-0.25]$; the masked one is $[0.67,-0.33,0.0000,-0.33]$ — "invalid action masking appears to do more than just 'renormalizing the probability distribution'; it in fact makes the gradient corresponding to the logits of the invalid action to zero." 🟢 Eq. 1, 5–6.

**Still a policy gradient (Prop. 1).** "$\mathit{mask}$ is either an identity function or a constant function for elements in the logits. Since these two kinds of functions are differentiable, $\pi'_\theta$ is differentiable to its parameters $\theta$ … which satisfies the assumption of policy gradient theorem (Sutton et al. 2000)." Note "$\mathit{mask}$ is *not* a piece-wise linear function" — it depends on $s$: $\mathit{mask}(s,x)\ne\mathit{mask}(s',x)$. 🟢 same section. Caveat: this is a theorem about the *update*, not about correctness of the mask — a wrong mask is a wrong policy.

**Two wrong ways** Huang & Ontañón measure: *naive masking* (sample from the masked dist, but compute the gradient from the unmasked one) — "KL divergence explodes", $t_{solve}$ up to 49% on 24×24; *mask removed at test time* — "still able to perform well to a certain degree … as the map size gets larger, its performance degrades and starts to execute more invalid actions", yet "significantly better than … invalid action penalty" (return 33.5 → 17.4 across maps vs 40 masked). 🟢 Table 2, "Evaluation Results".

Pseudo-code (the pattern every implementation uses):

```python
logits = policy_head(features)                       # [B, A]
masked = torch.where(mask.bool(), logits, -inf_or_-1e8)
dist   = Categorical(logits=masked)                  # softmax renormalises
a      = dist.sample(); logp = dist.log_prob(a)      # grad wrt masked logits = 0
# PPO ratio / entropy MUST be computed from `dist`, not from Categorical(logits)
```

## 3. VGC-Bench's implementation (doubles, two 107-way slots)

Paper: "The output of the Transformer encoder is then projected to the size of the action space by a linear layer into the logits before the softmax layer. The logits at the invalid actions are masked by $-\infty$. We also handle interdependent action constraints, such as ensuring that both Pokémon do not switch into the same replacement." Action space: "Each Pokémon has 107 available actions, which captures the full space of switching in benched Pokémon and using moves, where moves involve which move is being picked, the intended target, and whether or not the Pokémon is terastallizing (or using another available gimmick)"; team preview is "two joint 'switch-in' actions in a row". 🟢 VGC-Bench §4.1, §4.2.1 (no list of what makes an action invalid in the paper — that lives in code).

Code (`policy.py`, class `MaskedActorCriticPolicy(ActorCriticPolicy)` — plain `stable_baselines3==2.8.0`, no sb3-contrib, per `pyproject.toml` 🟢):

```python
def get_dist_from_logits(self, action_logits, mask, action=None):
    if action is not None:
        mask = self._update_mask(mask, action)
    mask = torch.where(mask == 1, 0, float("-inf"))
    distribution = self.action_dist.proba_distribution(action_logits + mask)
    assert isinstance(distribution, MultiCategoricalDistribution)
    return distribution
```

So the mask is a 0/1 vector of length $2\times107$, turned into an additive $\{0,-\infty\}$ vector and added to the logits before SB3's `MultiCategoricalDistribution` (two 107-way Categoricals). **Cross-slot masking** is autoregressive: `forward` samples slot 1, then rebuilds slot 2's distribution with `_update_mask(mask, actions[:, :1])` and overwrites `distribution.distribution[1]`; `evaluate_actions` does the same conditioned on the stored slot-1 action, so the PPO log-prob matches. `_update_mask` removes from the second half: action 0 (pass) if the ally passed but was not force-passed; the same switch index if the ally switched (1–6); and the whole mega / Z / dynamax / tera band (27–46 / 47–66 / 67–86 / 87–106) if the ally used that gimmick. 🟢 `policy.py`.

**Where `obs["action_mask"]` is produced** — located: poke-env fork, branch `vgc-bench`, `src/poke_env/environment/env.py`: `__setattr__` rewrites `observation_spaces` into `spaces.Dict({"observation": raw, "action_mask": Box(0,1,shape=(flatdim(action_space),),int64)})`, and `reset`/`step` return `{"observation": self.embed_battle(b), "action_mask": np.array(self.get_action_mask(b))}`. The mask itself is `DoublesEnv.get_action_mask` → `get_action_mask_individual(battle, pos)` in `doubles_env.py`: switch slots from `battle.available_switches[pos]` unless `battle.trapped[pos]`; move ids `7+5*i+j+2` for each of the first 4 known moves that is in `battle.available_moves[pos]` (PP / Choice lock / Disable are already filtered by Showdown's request) × `battle.get_possible_showdown_targets(move, active_mon)`; gimmick bands gated by `can_mega_evolve / can_z_move / can_dynamax / can_tera[pos]`; `[0]` (pass) only if `battle._wait` or the other slot is force-switching; `actions or [0]`. Inference path: `policy_player.py` `choose_move` builds the same dict with `DoublesEnv.get_action_mask(battle)`. 🟢 all three files.

## 4. Applying to the owner's poke-env DoublesEnv

- Inherit the fork's `DoublesEnv` (or port `get_action_mask_individual`): the 107 layout is `0` pass, `1–6` switch, `7+20g+5m+(t+2)` for move $m$, target $t\in\{-2..2\}$, gimmick $g\in\{0..4\}$ — Gen 9 ⇒ only $g\in\{0,4\}$ ever valid (`get_action_space_size(9)=107`). 🟢 `doubles_env.py`.
- Fainted slot: `active_mon is None` ⇒ only `switch_space`; 0 PP / Choice lock / Encore: handled by Showdown's `available_moves`; invalid targets: `get_possible_showdown_targets`; trapped: `battle.trapped[pos]` drops switches. Mask is therefore fully state-dependent, as Huang & Ontañón require.
- Keep the mask in both `forward` and `evaluate_actions` (else you get "naive masking"); keep it at deployment too (mask-removed degrades).
- Ally constraint: copy `_update_mask` — without it the two Categoricals are independent and can both switch to the same bench slot; `action_to_order` then raises "converted orders … are incompatible!" (or falls back to a random move with `strict=False`). 🟢 `doubles_env.py`.
- Entropy: SB3's `MultiCategoricalDistribution.entropy()` on $-\infty$ logits gives $p\log p\to0$ for masked entries (torch convention); VGC-Bench relies on this. Not verified numerically — check for NaN if you change the fill value.

## 5. Alternatives, and when not to mask

- **sb3-contrib MaskablePPO**: env exposes `action_masks()` (or wrap with `ActionMasker(env, fn)`); `MaskableCategorical.apply_masking` does `th.where(masks, logits, HUGE_NEG=-1e8)`, and `MaskableMultiCategoricalDistribution` splits a flat mask by `action_dims`. 🟢 source files. Docs page returned HTTP 429 — not verified; the source is.
- **CleanRL / Huang's `CategoricalMasked`**: `logits = torch.where(masks, logits, -1e8)`; entropy overridden to zero out masked terms. 🟢 `ppo_10x10.py` in the reference repo (cleanrl's `ppo_multidiscrete_mask.py` 404'd — not verified; it is a descendant of this class).
- **Penalty instead of mask**: only when you have no validity oracle. Zabounidis et al.'s fix for that case is *feasibility classification* — an auxiliary loss training the encoder to predict the mask, plus a "KL-balanced" weighting, so the learned predictor can replace the oracle at test time (Craftax PPO-Hybrid 48.8 vs 45.6 masked; 43.2 vs 43.9 unmasked when deployed without oracle). 🟢 RLJ 2026 §1. Showdown always gives the legal set, so this is not needed here.
- **Action-space shaping / elimination nets** (Kanervisto et al., Zahavy et al.): Huang & Ontañón note shaping "is shown to be potentially difficult to tune". 🟢 Related Work.

## 6. Questions to grill

1. Train-only vs test masking: VGC-Bench masks at both; Huang shows mask-removed loses ~50% return. Is there any reason to ever unmask (e.g. to see whether the net "learned legality")?
2. Autoregressive slot-2 mask: `forward` conditions slot 2 on slot 1 but slot 1 never sees slot 2 — is the ordering asymmetry a bias worth a second pass / symmetrised ordering?
3. Mask leakage in behaviour cloning: if BC is trained with the $-\infty$ mask, the cross-entropy is over legal moves only — then at RL time the policy has never had gradient on illegal logits, exactly the "no validity-discriminating features" situation Zabounidis describes. Does that matter when the oracle mask is always present? (Probably not — but ask.)
4. Fill value: VGC-Bench uses `-inf`, sb3-contrib / CleanRL use `-1e8`. Does `-inf` ever produce NaN in the SB3 entropy / KL computations with fp16?
5. `strict=False` fallback to a random move hides mask bugs — should training assert `strict=True`?
