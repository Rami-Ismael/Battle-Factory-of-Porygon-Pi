# Actual masked-model inference replay

The visualization replays a real CPU run of the saved `temperature_p0.pt` checkpoint in the local `vgc-team-generator-pilot` repository. It does not run inference in the browser.

`record-trace.py` observes the original `hpsdiffusion.sample_constrained` sampler's 48 categorical draws, recording probabilities after legality masking. The checkpoint uses masked-diffusion training and dependency-order constrained decoding. This is not a continuous image-denoising trajectory. Stat Point spreads are subsequently sampled from same-species corpus sets, not predicted by the checkpoint.

The run used seed 91826, unconditional conditioning, temperature 1 and guidance 1. The first proposal passed the bundled M-B validator. Checkpoint and all 692 corpus hashes were checked against the training manifest. Replaying the untouched sampler at the same seed produced identical tokens. No matchup performance was evaluated.

- `trace.json`: actual decisions, states, probabilities, hashes and validation result.
- `generated-team.txt`: complete team paste.
- `real-diffusion.html`: self-contained interactive fragment with embedded trace and sprites.
- `preview.html`: standalone preview wrapper.

To record again from this directory:

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python record-trace.py
```

The recorder depends on the original local repository and bundled validator paths. Re-recording updates the JSON and paste; the embedded trace in the HTML must also be refreshed before displaying a new run.
