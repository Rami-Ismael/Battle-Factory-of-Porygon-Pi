# TWISTER predictions, written before opening the #297/#322 battle outcomes or any refusal record
Written 2026-11-15 (see file mtime / SHA in the report). Paper: Thimmaiah et al., arXiv 2609.35609.

What I had already seen (KNOWN, not blind): the Two Temperatures and Fang readings' best-of-k, Pass@k, non-top share,
line-up and order-gap numbers; one cell's metadata of stat_points_84_sched_linear_random.json (ask 1/24, seed 1:
70 tries, 6 rejected, legal 91.4%). No refusal record has been seen (none is stored).

Mapping. Our sampler fills one field per network pass. The network's conditional for a field is already
renormalised over the values the legal Pokemon construction tables allow given the visible fields (in training and
sampling). Random-order rows add the lookahead (a feasibility filter: keep a value only if every Pokemon whose species
is still masked keeps a candidate species). TWISTER's target = the plain sampler's path law conditioned on a valid team.
The lookahead sampler's law / target ∝ 1 / kept share, where kept share = prod over draws of (1 - lookahead-refused
share). So kept share is TWISTER's per-team correction weight (twisted-SMC incremental weights multiplied out).
Rejection by the Showdown validator is exact conditioning and adds no tilt.

| # | Paper claim | Prediction for our rows | Confidence |
|---|---|---|---|
| T1 | No bias when nothing is reconditioned against the constraint (vacuous constraint / one-step reveal) | Species-first order: the lookahead refuses nothing, kept share = 1 for every team, so species-first rows are trajectory-exact for the tables constraint | 99% |
| T2 | Step-exact composition tilts the path law (trajectory bias) | Random order: kept share varies across teams; median per-team refused share between 20% and 80%; effective sample size of the kept-share weights between 20% and 80% of the teams in an ask | 55% |
| T3 | Correcting the tilt changes outputs | Reweighting random-order teams by kept share moves mean win rate by < 0.01 and best-of-64 by < 0.02 (paired bootstrap intervals contain 0), and closes < 25% of the random-order vs species-first gap | 75% |
| T4 | (implied) tilted trajectories differ in quality | Spearman correlation between log kept share and team win rate: absolute value < 0.1 | 65% |
| T5 | Bias grows with how much the denoiser changes when reconditioned | Per-field-schedule random order refuses more (lower median kept share) than linear-schedule random order, because the per-field network predicts a species from its set worse (swap mismatch 1.25 vs 0.90 nats) | 60% |
| T6 | — | Refused share rises with guidance from 2 to 8 | 50% |
| T7 | Step-exact constrained decoding guarantees validity | Our lookahead is partial, so validator rejections remain: species-first first-try legality >= random-order first-try legality at every ask 13-17 | 70% |
| T8 | Masking schedule | TWISTER predicts nothing between linear and per-field under species-first: both have zero tilt; battle difference already KNOWN to be inside +-0.04 | n/a (KNOWN) |
