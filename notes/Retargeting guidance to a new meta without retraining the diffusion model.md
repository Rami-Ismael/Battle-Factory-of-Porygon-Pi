---
created_at: 2026-09-27
updated_at: 2026-09-27
tags:
  - training-free-guidance
  - discrete-diffusion
  - meta-shift
  - surrogate-model
---
- goals is 
	- The current method is using classifer free guidance to base on the win rate of a fix pool of 50 top current meta pokemon we steer the direction to the places I was wondering we calculate the win rate base on simulated battle with a battle policy how make the win rate guidance base on a dynamic list of meta pokemon if we want to test on the top 100 current meta pokemon we have to retrained the diffusion model i think there should be a good academic name maybe dynamic meta team
- [ ] [[Does naive gradient work as guidance]]
- [ ] Test set-conditioned guidance, where a set encoder embeds the meta teams as a surrogate input, so a new meta needs no refit.(A set encoder (Deep Sets style) embeds the meta teams, and the surrogate takes that embedding as an input. Any meta then becomes just an input, so nothing is refitted. The broader problem is a **non-stationary objective**: the meta shifts, so $f$ moves.)
- [ ] [Constrained Black-Box Optimization with Rejection Sampling](https://deephyper.readthedocs.io/en/latest/examples/examples_bbo/plot_constrained_black_box_optimization.html)
