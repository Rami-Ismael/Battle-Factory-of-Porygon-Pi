# Todo

- [ ] [[Write up a source of error you notice on 2026-10-14 when my cross entropy method state the best pokemon team had pointless item]]
- [ ] /goal Write down every method we have tried so far that we believe has failed, judged on a stated metric. A method is a whole process that comes from a paper, such as the cross-entropy method or Boltzmann selection; it is not a baseline. A change we made to a method is a modification, not a method: mention it only inside that method's entry, where it helps explain the result. For each method, give the detail of the Boltzmann example: what it is, why we tried it, what it was compared against, the result, why it failed, what we learned, and the verdict. Name the paper the method came from, labelled 🟢 or 🔒, and link its note in Papers/. Every score must say it is a win rate against the top-50 meta, with the behaviour-cloning policy on both sides, and give its battle count.
	- [ ] grill me
		- [ ] ❓ **Q3** - **Which paper does "the paper the method came from" mean?** Boltzmann selection is decades older than DiffUCO. The cross-entropy method goes back to Rubinstein, 1999. Options:(The paper you took it from explains why you tried it. The original gives it proper credit.)
		- [ ] ❓ **Q4** - **What if a method has no Papers/ note?** The cross-entropy method, for one, has none. Options:(write "no Papers/ note" in the entry and link the paper directly)
- [ ] Methods we have tried
	- [ ] Talked about cross entropy method
	- [ ] Boltzmann selection with an annealed temperature
	- [ ] Masked diffusion as a from-scratch generator (MDLM, NeurIPS 2024 🟢) just byitself
	- [ ] Classifier-free guidance on win rate (DDOM, ICML 2023 🟢; Schiff et al., ICLR 2025 🟢)
	- [ ] It took me time to realize i need a held out meta pool as a form of 
	- [ ] Talked about how I prompt the diffusino model initial to been able to generate all the possible pokemon instead of limited section of the pokemon team i was shocked it the version three of my diffusion model with the ability to genereate pokemon team is still limited
	- [ ] [[Write up a source of error you notice on 2026-10-14 when my cross entropy method state the best pokemon team had pointless item]]
	- [ ] Write up stuff there a clear mis communication between me and claude implementation what are the solution to the fix problem z
		- [ ] Talked about the detail in where in you want to reimplement the work of diffusion-bbo the model interpret the work as a fine tuning steps not training steps
		- [ ] ANother thing the cross entropy method it will implement it will say this pokemon team is .5 when we have battle simulation the dsiagree wiht say .5 instead so that is a problem the cross entropy problem is just become a forward model nothing it will actually condition on  as you see the ask and got chart on the 2026-10-16 in @Inverse-model loop: train on all labels, ask from delivered. I am trying to say here is the model was trained on condition of ask then ask will be fixed in my fixed cross enetropy method therefore that why the ask and got chart in my method look stuck. 
		- [ ] I think the source of the error of the right graph where it does follow the ask and got chart for cross entropy method while diffusion bbo method does that is because we are doing 40 epoch of training please set research agent for this task 
		- [ ] the cross-entropy retraining only ever shows the model teams labelled 0.52–0.72. Asked for 0.3 or 0.5, values it never saw in that retraining, it falls back on “make teams like my elites”. The best-fit win rates measured earlier show it has lost the middle of the scale. The guidance paper only tests image classes, so it doesn’t cover a continuous win-rate condition or asks outside the training range.
		- [ ] The label were to noisy with 49 example so I trained with an example with 196 training examples you see much better improvement in score
	- [ ] [[Diffusion-BBO Diffusion-Based Inverse Modeling for Online Black-Box Optimization]]

Full list (8 failed methods + 3 that didn't fail): code repo `docs/failed-methods.md`, 2026-09-28.
