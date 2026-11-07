# Ask grid verification map

The maintained list of user-facing journeys through the 84-field ask grid. Read this index, then use the matching feature file as the recipe.

| Feature | File | Proves |
| --- | --- | --- |
| Smoke grid | [`smoke-grid.md`](smoke-grid.md) | Sampling, battling and the ask-versus-got error work together on one checkpoint. |
| Preflight | [`preflight.md`](preflight.md) | A checkpoint is fit to battle: round trip, no training copies, conditioning on the ask, controls in range. |
| Error comparison | [`error-comparison.md`](error-comparison.md) | Saved arms are judged by the ask-versus-got error against a reference arm. |

## Baseline preconditions

- `verify.sh doctor` prints `DOCTOR PASS`.
- `verify.sh launch` printed a `RUN` folder whose `seed.log` ends with `top50 opponents: 49`.
- `MATCHUP_DB` points at `RUN/matchup.sqlite` for every command that battles. `verify.sh` sets it.

## Proof and skip reporting

- Cite the PASS and FAIL lines from `RUN/drive.out` and the numbers printed beside them.
- A battle proof includes the private database's battle count before and after.
- Report a journey you could not reach with the command tried and the unmet precondition. A journey proven through a different entry point is not proven.
