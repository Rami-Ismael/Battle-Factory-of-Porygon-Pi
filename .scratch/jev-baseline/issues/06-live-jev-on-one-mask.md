# 06: Live Jev on one mask

**What to build:** An adapter on TypeSafe's SDK implements the chooser interface. The key comes from the Keychain or the environment and is never printed; model, address and option limit are settings; SDK errors map to retry or stop. One real call on one mask is logged with its choice, confidence and probabilities.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] One live call succeeds and its full response is logged.
- [ ] The key appears in no log or file.

Source: `code/docs/jev-baseline-spec.md`.

**Result 2026-10-27:** one live call on meta mask `b90b461c39429bf2` (item, k=1) through `jev_grid.py live`: model `jev-1.14.0`, 0.44 s, 24,644 input and 1,357 output tokens reported ($0.00104). Jev picked Never-Melt Ice (confidence 0.58; Choice Scarf 0.33), the starter's own item, which the meta list in the state contains. The team is legal. The full response is in `~/vgc-data/jev/meta-live.jsonl`; no key appears in it. The token estimate was recalibrated from 3 to 2.3 characters per token (measured 2.54).
