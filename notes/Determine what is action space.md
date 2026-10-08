
1. The plan to make an agent the does well in vgc format the closest related work is [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] so we are copy their method which is "he action space itself is joint: the agent must select moves for both active Pokémon simultaneously, resulting in a large discrete output space where invalid moves (such as using a move with zero remaining power) are masked out during training. [1] ([[Citation]])"
2. Verdict 2026-08-18: keep VGC-Bench's 107-way per slot unchanged (poke-env DoublesEnv native), factored per slot with cross-slot masking; team preview folded in as two joint switch-ins (§4.1) — no separate 4-of-6 head. Detail in chat.[1][[Citation]]

- [ ] Find the PyTorch code for this — found 2026-08-22: slot 1 sampled, `_update_mask` strikes the pairs it forbids, slot 2 resampled — [MaskedActorCriticPolicy (policy.py L39)](https://github.com/cameronangliss/vgc-bench/blob/d79f953/vgc_bench/src/policy.py#L39), [act_len = 107 (utils.py L88)](https://github.com/cameronangliss/vgc-bench/blob/d79f953/vgc_bench/src/utils.py#L88)

| Gym action space          | SB3 distribution               | what the head emits                    |
| ------------------------- | ------------------------------ | -------------------------------------- |
| `Discrete(n)`             | `CategoricalDistribution`      | n logits, sample 1 index               |
| `MultiDiscrete([n₁, n₂])` | `MultiCategoricalDistribution` | n₁+n₂ logits, sample 1 index per chunk |
| `Box(low, high)`          | `DiagGaussianDistribution`     | mean and log-std, sample a real vector |
# Prompt

1. I want to determine that is the best form of action space of the pokemon vgc. Currently, my understanding of action spaces from vgc there is 107. We are using multiDiscrete there are the policy will make a decision from 107 option. Then afterward, then, multidiscrete make a decision from 107 option. We are using MutliDiscrete sb3.