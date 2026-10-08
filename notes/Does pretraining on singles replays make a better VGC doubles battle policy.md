---
created_at: 2026-08-18
updated_at: 2026-08-18
type: experiment
status: open
title: Does pretraining on singles replays make a better VGC doubles battle policy
one_line: pretrain the battle policy on Metamon's singles replay corpus, fine-tune on VGC Reg M-B doubles replays, compare against a doubles-only policy — the singles→doubles transfer test
goal: measure how much singles pretraining helps in doubles, and what it is worth in doubles data
reframed_2026-08-18: the BC stage is the initialisation of the self-play arms — measure the transfer at the end of self-play (pure self-play, fictitious play, double oracle), not at the BC checkpoint
init_arms: random · singles-only BC · doubles-only BC · singles→doubles BC
control_arm: doubles-only BC (VGC-Bench's baseline)
pretrain_source: Metamon replay corpus — 🟢 https://huggingface.co/jakegrigsby/metamon
moved_from: "[[Todo Section]] — checkbox of 2026-08-18, relocated 2026-08-18"
differentiation_levers: proposed in chat 2026-08-18 — data-scarcity curve, unified singles-as-masked-doubles encoding, rare-species bins, per-mechanic negative transfer, generation confound, 2×2 with data v0/v1
related:
  - "[[Behavior cloning a VGC battle policy from human Showdown replays]]"
  - "[[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]]"
  - "[[Why Self-Play is consider Multi-Agent]]"
tags:
  - battle-policy
  - transfer-learning
  - singles-to-doubles
  - metamon
  - showdown-replays
---
- [ ] 2026-08-18, new experiment: pretrain the battle policy on singles replays (Metamon corpus), then fine-tune on doubles; measure how much better it is on average in doubles than a model trained on doubles only — the singles→doubles transfer test



## Verify each differentiation claim

- [ ] 1. Data-scarcity claim: train doubles-only on 10 / 25 / 50 / 100 % of the doubles replays and singles-pretrained on the same slices, read where the two curves cross, and decide whether the pretraining is worth more doubles data than a Reg M-B scrape would give.
- [ ] 2. Unified-encoding claim: encode singles as doubles with the second slot masked (singles actions are a subset of the 107-way space), then transfer embeddings only / embeddings + backbone / everything, and decide which layers carry the transfer.
- [ ] 3. Rare-species claim: bin doubles win rate by each species' frequency in the doubles corpus and rerun the VGC-Bench §4.3 held-out-team test, then decide whether the gain sits in the rare bins.
- [ ] 4. Negative-transfer claim: track Protect, switch, spread-move and targeting rates against human doubles replays across fine-tuning, and decide whether singles habits hurt early and for how many steps.
- [ ] 5. Generation-confound claim: check the Metamon corpus format list, add a Gen 9 OU singles slice if it is Gen 1–4 only, run generation-matched vs mismatched, and decide which shift the experiment actually measures.
- [ ] 6. Stacking claim: run the 2×2 of {data v0, v1} × {no pretrain, singles pretrain} from [[Behavior cloning a VGC battle policy from human Showdown replays]], and decide whether more doubles data alone closes the gap.
- [ ] 7. Protocol: match doubles data and fine-tune steps across arms, report data-matched and compute-matched, rank arms in the cross-play + [[AlphaRank]] pool, and decide the pool before any run starts.
- [ ] 8. Downstream claim (reframed 2026-08-18): initialise each version of self-play (pure self-play, fictitious play, double oracle) from random, singles-only BC, doubles-only BC and singles→doubles BC, measure at several self-play budgets, and decide whether the init gap survives self-play or washes out.
