# Browser team validator

`showdown-validator.js` exposes `globalThis.PokemonTeamValidator` in a window or classic Web Worker:

```js
importScripts('./runtime/showdown-validator.js');
const errors = PokemonTeamValidator.validateTeam(showdownPaste);
// [] means valid. Nonempty array contains the pinned validator's messages.
```

The format is fixed to `gen9championsvgc2026regmb`. This is actual local validation, not a lookup of the example's recorded result. It needs no backend, network requests, or filesystem after the bundle loads. Prefer loading inside the sampler worker to avoid blocking the article UI. Internal errors produce a nonempty error array; they never count as valid.

## Provenance and scope

The 41 bundled CommonJS modules are unchanged compiled Pokémon Showdown validator/data modules from the project's pinned runtime (`913da3602a3aa1db79f9fdc5d5222eaf8d39569d`), plus its `ts-chacha20` 1.2.0 dependency. Source-map comments are omitted. `validator-manifest.json` records module hashes and bundle hash. `SHOWDOWN-LICENSE` and `TS-CHACHA20-LICENSE` accompany these sources.

The small browser wrapper supplies an in-memory CommonJS resolver, POSIX path normalization, the pinned mod-directory listing, and recursive strict equality for the acyclic plain species records used by Dex. It does not implement battles, random-team generation, general Node APIs, or other formats. The wrapper adds no legality rules and does not remove rules from the format. The full base and Champions data tables used by the validator are embedded.

## Rebuild and verification

```sh
NODE_PATH=/path/to/ts-chacha20/node_modules node build-validator.cjs /path/to/pinned/showdown/runtime
NODE_PATH=/path/to/ts-chacha20/node_modules node test-validator.cjs /path/to/pinned/showdown/runtime
```

The test expects this figure at `http://127.0.0.1:8765/` and Playwright plus Chrome. Set `PLAYWRIGHT_PATH` if the bundled dependency path differs. An optional local Reg MB corpus supplies additional cases when present.

`validator-parity.json` records exact error-array equality between native Node validation and browser validation with networking disabled. The 66 checked cases include 55 corpus teams plus a known-valid team and ten invalid cases: empty/oversize team, unknown/restricted species, unknown/unlearnable moves, incompatible ability, duplicate species/items, and excess Stat Points. All passed. Classic Worker `importScripts` was separately checked against the known-valid team. These checks cover representative behavior, not every possible team.
