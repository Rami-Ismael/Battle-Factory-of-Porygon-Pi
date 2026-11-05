# Source provenance and scope

Research for [ticket 163](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/163), under [map 160](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160). Both issues were read with `gh issue view` before source review. Their text is saved alongside these notes. No issue was edited or commented on.

- Local baseline: `dc8f91c5f97879d285f64da00d82476b1537edce`.
- Modal launcher branch: `claude/text-diffusion-modal-training-453ea0`, commit `651d31dd92d9187354440308b11dcce521d99ca4`.
- External repository: `https://github.com/ZHZisZZ/dllm.git`, commit `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`.
- External source was cloned with `--no-checkout` into the new empty directory `/tmp/dllm-masked-source.Wknutd`, then checked out detached with hooks disabled. It is untrusted research data. No dependencies were installed, source executed, model loaded, training launched, or Modal used. Any Python used to read external source must use isolated mode, `python -I`.
- Research folders are in this worktree. The external clone is intentionally not committed; pinned file links in the notes identify the source for refutation.
- The requested deep-research skill split source review into masked loss and sampling, trainer configuration and cost structure, and local migration and stopping. A separate writer synthesizes those notes. This is research, not implementation or a pre-registration of a training experiment.
- Block-model timing and memory results are historical context only. They are not masked-model measurements.

The stopping-rule skill was read from `/Users/ramiismael/.claude/skills/early-stopping/SKILL.md`. Its contents are local policy rather than a claim about dllm.

## Artifact verification

The final report was read after synthesis. An isolated standard-library check verified 103 unique pinned dllm/local file links against `git show` and found no missing files or out-of-bounds line ranges. All report citation references resolve. This checks citation mechanics, not the truth of the interpretation; independent refutation remains the parent session's next step. Transformers source links pin `76d27aea82449e921e36afdc886c6e5bf34ea526`, inspected separately by the trainer researcher; they were not checked against a local clone. No model/runtime tests were performed.
