# Mask Lab: matched team-completion comparison

Open [index.html](index.html) directly in a browser. It is a self-contained,
completed snapshot; no API key, live service or experiment runtime is required.
[data.json](data.json) contains the displayed results and historical reference data.
Edit `template.html` or `data.json`, then run `python3 blog/mask-lab/rebuild.py`.
The rebuild retains sprite assets embedded in the checked-in page.

The current study uses 20 starters, seven masks per starter and three repeats:
420 outputs per method. Methods are Ling with reasoning off/on, Mercury with
reasoning off/low/medium/high, and the Pokémon diffusion ensemble of the
search-loop-trained model (F1) and matchup-matrix-augmented model (G2), using
their regulation-specific v3 checkpoints.

Three views intentionally use different denominators:

- Full set: all 420 attempts, legality, and descriptive win rates among each
  method's legal outputs. Illegal outputs are not fabricated battle losses.
- Shared cohort: 64 trial/repeat pairs legal for all seven methods, across 20
  starters; win rates weight starters equally.
- Pairwise: diffusion versus each LLM setting on their larger jointly legal
  subsets, with exploratory paired starter-bootstrap intervals.

Diffusion achieved 100% legality and a 9.24% full-set legal-team win rate.
On the shared cohort it scored 2.13%; Mercury high scored 2.29%, without a clear
paired advantage over diffusion. These compare complete systems on the frozen,
visible Champions M-B opponent pool, not equal-input neural architectures or
held-out/live-meta generalization. Diffusion uses constrained decoding and
non-neural Stat Point fills. Cached panel reuse and selection effects matter.

Mercury recorded $0.6902 across its four arms. Five missing cost records and
unknown original timeout charges are excluded, as are $0.0040 of preflight calls.
Diffusion has no API fee; local training/hardware/electricity costs are unmeasured.

The `allMethodComparison`, `matchedDiffusion`, `mercury` and `reasoning` keys in
`data.json` preserve the summaries and audit metadata. Raw paid completions and
per-panel records remain in ignored local `code/results/` study directories.
The study-specific analysis scripts require those saved studies; a fresh clone
can inspect this snapshot but cannot reproduce absent model checkpoints or raw
experiments without the runtime and datasets described in the code documentation.
