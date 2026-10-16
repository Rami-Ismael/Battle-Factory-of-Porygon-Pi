# Run queue

Training and battle runs that are written and checked but not yet started. Each entry names the
pre-registration, the command, what must exist first, and the cost. Run in order; move an entry to
its pre-registration's results section when it finishes.

| # | Run | Pre-registration | Command | Needs first | Cost |
|---|---|---|---|---|---|
| 1 | Smoke test of the Diffusion-BBO-rule loop, both groups | [cem-dbbo-loop.md](cem-dbbo-loop.md) | `CEM_SMOKE=1 CEM_GROUP=dbbo python cem_dbbo.py run` then the same with `CEM_GROUP=scratch` | the version-3 checkpoints; `regmb-v3/remapped.npz` and `synthetic.npy` (from `scripts/retrain_regmb.py prepare --version v3`) | minutes |
| 2 | Re-steered foundation on every scored team (`cem_dbbo`) | [cem-dbbo-loop.md](cem-dbbo-loop.md) | `CEM_GROUP=dbbo python battle_supervisor.py cem_dbbo.py run --then final` | entry 1 clean; the Diffusion-BBO acquisition checked against §4 of the paper | per generation as `cem_v3`; up to 10 generations plus 16 × 192 final battles |
| 3 | From-scratch network on every scored team (`cem_scratch`) | [cem-dbbo-loop.md](cem-dbbo-loop.md) | `CEM_GROUP=scratch python battle_supervisor.py cem_dbbo.py run --then final` | entry 1 clean | the most expensive: a full training run every generation; read generation 0's minutes in `results/cem_scratch.json` before letting it continue, lower `CEM_SCRATCH_EPOCHS` if needed |
| 4 | Paid run of the DSPy structure experiment | [dspy-structure-experiment.md](dspy-structure-experiment.md) | see that document | pending since 2026-10-14 (README) | API spend |

Optional after 2: the fixed-multiplier control, `CEM_W=1.2 CEM_GROUP=dbbo` (its own result file,
`results/cem_dbbo_w1.2.json`).

All commands run from `src/` with `/tmp/vgc-pilot/.venv/bin/python`.
