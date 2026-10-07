# Todo
- [ ] /goal Write down every method we have tried so far that we believe has failed, judged on a stated metric. A method is a whole process that comes from a paper, such as the cross-entropy method or Boltzmann selection; it is not a baseline. A change we made to a method is a modification, not a method: mention it only inside that method's entry, where it helps explain the result. For each method, give the detail of the Boltzmann example: what it is, why we tried it, what it was compared against, the result, why it failed, what we learned, and the verdict. Name the paper the method came from, labelled 🟢 or 🔒, and link its note in Papers/. Every score must say it is a win rate against the top-50 meta, with the behaviour-cloning policy on both sides, and give its battle count.
	- [ ] grill me
		- [ ] ❓ **Q3** - **Which paper does "the paper the method came from" mean?** Boltzmann selection is decades older than DiffUCO. The cross-entropy method goes back to Rubinstein, 1999. Options:(The paper you took it from explains why you tried it. The original gives it proper credit.)
		- [ ] ❓ **Q4** - **What if a method has no Papers/ note?** The cross-entropy method, for one, has none. Options:(write "no Papers/ note" in the entry and link the paper directly)
- [ ] Methods we have tried
	- [ ] Talked about cross entropy method
	- [ ] Boltzmann selection with an annealed temperature
	- [ ] Masked diffusion as a from-scratch generator (MDLM, NeurIPS 2024 🟢) just byitself
	- [ ] Classifier-free guidance on win rate (DDOM, ICML 2023 🟢; Schiff et al., ICLR 2025 🟢)

Full list (8 failed methods + 3 that didn't fail): code repo `docs/failed-methods.md`, 2026-09-28.
