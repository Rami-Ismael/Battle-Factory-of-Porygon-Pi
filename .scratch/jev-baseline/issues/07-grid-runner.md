# 07: Grid runner: four runs per mask, resumable, priced

**What to build:** A command runs Jev over a grid's masks: the top pick plus three runs sampled from Jev's probabilities, each run asking its own questions. Records go to a resumable per-grid log. A dry run prints masks, runs, questions, estimated tokens and dollars per grid. A maximum-call limit and a small live smoke test exist.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] The dry run prints counts for all three grids without calling Jev.
- [ ] A five-mask live run writes the expected records.
- [ ] An interrupted run resumes without duplicate records.

Source: `code/docs/jev-baseline-spec.md`.

**Result 2026-10-27:** dry runs: 1,080 pairs, 12,720 questions and about $13.4 per grid (random, held-out, meta). Smoke run on the meta grid (`run --grid meta --limit 5`, five Stat Points masks, 20 pairs, 60 calls): all 20 legal, 1,382,380 input tokens reported against 1,413,887 estimated (2% high), $0.058. A rerun found 0 pairs left to send. 20 of 60 answers were the starter's own spread; confidence median 0.30 (0.06 to 0.82); every mask's sampled runs differ from its top pick. Results: `~/vgc-data/jev/meta.jsonl`.
