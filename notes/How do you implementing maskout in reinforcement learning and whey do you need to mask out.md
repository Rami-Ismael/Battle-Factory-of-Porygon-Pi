1. I don't much about how to implement action masking you I expect to claude but i still reference to other method so I gonna copy the method from  [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] as it nearest related work "Invalid action masking**: During training, invalid actions (e.g., using a move with 0 remaining PP, switching to a fainted Pokémon, or targeting an invalid slot) are masked by setting their logits to −∞ in the policy network ." [1] [[Citation]]
2. Sources 2026-08-19: 🟢 Huang & Ontañón, *A Closer Look at Invalid Action Masking* (FLAIRS 2022, arXiv 2006.14171) · 🟢 Zabounidis et al. RLC 2026 (why not penalise) · code: vgc-bench `src/policy.py` `MaskedActorCriticPolicy.get_dist_from_logits`, mask from `obs["action_mask"]`
3. Research write-up (2026-08-19): `Teaching/reference/action-masking-in-policy-gradient-rl.md` — why/how, VGC-Bench code walk, poke-env DoublesEnv mask source, grill questions.

- [ ] Where is the code action masking
