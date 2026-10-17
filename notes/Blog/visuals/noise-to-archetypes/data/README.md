# Completions behind the archetype-diffusion trace

Recorded 2026-10-11. `completions.json` holds each individual completion ("future") that `../../archetype-diffusion/trace.json` only summarises, for trace records 0, 246, 298, 86, 294, 59, 158 and for the real team MB607. `mb607-record.json` is MB607 in trace.json's `records[i]` schema, plus `order`, `stepFormat` and one extra `meta` block.

## Schema of `completions.json`

- `meta`: checkpoint and file hashes, `seeds`, `parity` (true), `parityDetail`, `note`.
- `archetypes`, `centroids3d`, `centroids2d`, `map2d` (`xy = matrix @ xyz`, zero offset: the fit `build.py` makes, so pages need not refit it).
- `speciesTable`: sprite keys, as trace.json's `speciesId`. `vocabTable`: display strings for `f`.
- `shared0`: the 512 completions of the fully masked state. `p` (3D point), `lab` (0–3), `sp` (six `speciesTable` indexes).
- `teams[key]`: `name`, `fields` (48), `speciesId`, `label`, and `steps["1"…"41"]`, each with 24 × `p`, `lab`, `sp`, `ev`. `ev` is the slot of the Pokémon that triggers the label (lowest slot if several qualify), −1 for Other.
- Teams `"0"` and `"MB607"` also carry `f`: the 48 fields of each completion as `vocabTable` indexes. Natures in `f` come from the same sampler loop continued over steps 42–47 on separate generators; `""` is the vocabulary's token for a set with no nature line. Stat Points are not sampled.

**MB607.** Slots stay in file order; moves inside a slot are in the encoder's alphabetical order, as in every team the model trained on. All 48 fields are in the vocabulary and pass the sampler's constraints. Label: Weather, from Ninetales-Alola's Snow Warning (revealed at step 17). Trick Room outranks Weather in the rule, and most completions add Trick Room until Mimikyu's last move is revealed, so Weather holds all 24 only from step 38.

**Seeds.** The recorder's own: reference 20260929, main and shared state 20260928, record *i* 20260928 + 1 + *i*. New: MB607 20261011; natures in `f` 20261012 (team 0) and 20261013 (MB607).

**Rerun** (about 55 s, eight spawned single-thread processes; the check needs numpy only):

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python record-completions.py
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python check-completions.py
```

**Acceptance, 2026-10-11: all pass at tolerance 1e-4.** Worst difference from trace.json over steps 1–41, recomputed from the rounded file, as mean / covariance / shares: record 0 6.3e-5 / 3.3e-5 / 3.3e-5; 246 5.8e-5 / 8.9e-6 / 3.3e-5; 298 5.0e-5 / 1.6e-5 / 3.3e-5; 86 5.8e-5 / 1.6e-5 / 3.3e-5; 294 6.7e-5 / 1.3e-5 / 3.3e-5; 59 8.8e-5 / 1.8e-5 / 3.3e-5; 158 6.3e-5 / 1.9e-5 / 3.3e-5. Those are rounding only: before the file is rounded, every summary row equals trace.json's exactly. Step 0 is 178/153/83/98 of 512, the centroids match exactly, all 300 main teams regenerate field for field, and two runs wrote identical files.
