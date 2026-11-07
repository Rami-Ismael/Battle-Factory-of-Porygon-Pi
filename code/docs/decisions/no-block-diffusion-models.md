# No block diffusion models

- **status:** accepted
- **decided_on:** 2026-11-06 (first said 2026-11-05)
- **scope:** every text diffusion model this project trains, fine-tunes, samples from, compares against or costs.
- **decision:** text diffusion work uses fully bidirectional masked diffusion models only. Block diffusion language models (generate block by block, left to right, with diffusion inside each block: Fast-dLLM v2, BD3LM, SDAR) are not candidates, baselines or training recipes. The baseline for masked models is the 84-field discrete diffusion model.
- **reason:** the owner's two reasons. A block diffusion model "takes too much resource to train", and "inference is not the bottleneck", so block diffusion's speed advantage buys nothing here.
- **source_links:** "not interested in training a block diffusion language model as it takes too much resource to train and inference is not the bottleneck" ("Sharpening the pretrained text diffusion Wayfinder prompt", 2026-11-05 08:53); "We are not using block diffusion" ("Drawing the good-enough bar", 2026-11-06 20:40). [#179](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/179), [#185](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/185) and [#186](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/186) closed on 2026-11-06; [#160](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160) names the 84-field model as the baseline.
- **excluded_alternatives:** the Fast-dLLM v2 1.5B and Qwen2.5 3B block fine-tunes as #160's baseline; the block-against-bidirectional swap test; cheaper block diffusion training (E2D2 and similar in `code/docs/textdiff-speed-research.md`).
- **code:** `code/scripts/count_masked_diffusion_text.py` `CANDIDATES` lists masked models only. `code/src/textdiff*.py`, `code/scripts/textdiff_llmformat.py` and `code/docs/textdiff*.md` are the earlier Fast-dLLM v2 work, kept as a record and marked dropped.
- **verification:** `code/scripts/test_no_block_diffusion_candidates.py` (in `ci_tests.txt`) fails if a block diffusion model joins `CANDIDATES`.
- **superseded_by:** none.

Citing a block diffusion paper in a reading note is fine. "Block" in a sampler's unmasking order (the "Medium/Sharp block unmasking order" arms in `code/docs/experiment-names.md`) is a different thing and is unaffected.
