# Foul Play for doubles

Foul Play ([pmariglia/foul-play](https://github.com/pmariglia/foul-play), first in Gen 9 OU at the PokeAgent Challenge) plays singles only: its search engine, [poke-engine](https://github.com/pmariglia/poke-engine), models one active Pokémon per side. This is its decision loop rebuilt on Pokémon Showdown's own simulator so it plays VGC 2026 Reg M-B doubles: open team sheets, bring 4 of 6, Mega Evolution. The source reading behind it is in [foul-play-source-reading.md](foul-play-source-reading.md) (foul-play `8f541f8`, poke-engine `f7b147f`).

## One decision

| Step | Foul Play (singles) | This port (doubles) |
|---|---|---|
| What it knows | its request plus its side's protocol | the same: `View` reads only its own player stream, never the omniscient log |
| Speed | narrows a foe's Speed from move order | the same: two moves in one epoch with equal priority bound the foe's raw Speed stat |
| Sampled battles | N copies with the foe's sets drawn from team datasets and usage stats | N copies; with open team sheets only Stat Points are hidden, drawn from VGCPastes spreads filtered by the Speed bound; unrevealed brought Pokémon drawn by team-preview affinity (BSS mode's rule) |
| Search on each | poke-engine MCTS: decoupled UCB1 per side, one node per iteration, leaf `sigmoid(0.0125 * (eval - root eval))` | the same, on a Showdown `Battle` rebuilt from the view, with UCB statistics kept per active slot |
| Choice | add root visit shares over samples, keep options within 75% of the best, draw one in proportion | the same, slot by slot |
| Team preview | BSS mode searches lead and back | searches the 90 doubles options (2 leads, 2 back) for both sides |

## What changed for doubles, and why

1. **UCB statistics per slot.** A side's joint action in doubles has about 100 options (4 moves × targets × Mega, plus switches, for each of two slots). poke-engine runs on the order of 100,000 iterations a second in Rust; Showdown's simulator in JavaScript manages about 1,000. Keeping statistics per slot lets each slot's ~12 options be visited enough. A slot's pick excludes what its partner took: the same switch-in, a second Mega, two passes when a switch is required.
2. **Open-loop tree.** poke-engine branches on every chance outcome. Here each iteration replays the simulator from the root with fresh randomness, and a child is keyed by the joint choice plus which Pokémon are in and who must switch, so a faint opens a new branch while damage rolls share one.
3. **Showdown is the engine.** No second implementation of doubles mechanics: the sampled battle is a real `Battle` (pinned runtime in `~/vgc-data/battle-evaluations/runtime`) with HP, status, boosts, volatiles, items, Mega forme, field and side conditions written in from the view, and durations from Showdown's own `durationCallback`.
4. **Evaluation.** poke-engine's `genx/evaluate.rs`, summed over all four brought Pokémon per side, with boosts and volatiles for both actives and side conditions weighted by turns left. Terastallization terms are dropped (Champions has none).

## Baselines

- **Behaviour cloning** (`bc`): the project's battle policy, `/tmp/bc_100.zip`, played through poke-env as the fast evaluator plays it (stochastic, actor only).
- **Flat Monte Carlo** (`montecarlo`): on the same sampled battles and time budget, draw both sides' first choices at random, play both sides at random to the end, credit the result to every slot option drawn, and pick the best mean per slot.
- **Random** (`random`): a uniformly random legal choice per slot.

## Checks

`src/foul_play_doubles_audit.cjs` plays random battles and, at every decision, compares the rebuilt battle with the real one field by field. On 98 battles of top-50 meta teams (2,586 decisions, about 100,000 checks), everything a player can see matched except 6 Protect counters in mid-turn switch decisions (Parting Shot or U-turn after a Protect). `scripts/test_foul_play_doubles.py` runs a short audit and a short Foul Play battle.

## Run

```sh
cd code
/tmp/vgc-pilot/.venv/bin/python src/foul_play_doubles_battles.py spreads          # data/foul-play-doubles-spreads.json
/tmp/vgc-pilot/.venv/bin/python src/foul_play_doubles_battles.py match foulplay bc --pairs 100 --out DIR
```

A matchup X vs Y plays each ordered pair of distinct top-50 meta teams (A, B) twice, X with A against Y with B and then Y with A against X with B, with the same slot shuffle in both games. Team strength and side cancel, so X's share of points (win 1, tie 0.5) above 0.5 measures the policy. `summary.json` gives the share, a Wilson interval over battles, and a paired interval over team pairs.

## Results, 2026-11-23

Chart: [reports/foul-play-doubles/index.html](../reports/foul-play-doubles/index.html). Raw battles: `~/vgc-data/foul-play-doubles/2026-11-23/`. Each row is 100 pairs of top-50 meta teams, 200 battles; the interval is 95% over team pairs.

| Search budget per decision | Opponent | Foul Play's points of 200 | Share (95% interval) | Search iterations per decision |
|---|---|---:|---|---:|
| 4 samples × 1 s | behaviour cloning | 135 | 0.675 (0.61–0.74) | 1,520 |
| 1 sample × 1 s | behaviour cloning | 125 | 0.625 (0.56–0.69) | 340 |
| 4 samples × 0.25 s | behaviour cloning | 99 | 0.495 (0.42–0.57) | 563 |
| 4 samples × 1 s | flat Monte Carlo at the same budget | 104 | 0.52 (0.45–0.59) | 1,446 |
| 4 samples × 0.25 s | flat Monte Carlo at the same budget | 110 | 0.55 (0.48–0.62) | 382 |
| 4 samples × 0.25 s | random play | 173 | 0.865 (0.82–0.91) | 379 |

Flat Monte Carlo at 4 samples × 0.25 s against behaviour cloning: 102 of 200 (0.51, 0.44–0.58).

- Foul Play beats behaviour cloning once it searches about 1 s per decision; at 1 s split four ways it does not.
- At equal time, one deep tree beats four shallow samples. Open team sheets leave only Stat Points hidden, so depth is worth more than averaging over samples here.
- Foul Play and flat Monte Carlo were not told apart at either budget.
- 2 of about 30,000 Foul Play choices were rejected ("Fake Out needs a target", a request without a move target); fixed after the runs. 8 switches into a trapped slot were refused and re-chosen, as Showdown hides trapping until tried.

## Known gaps

- A switch request in the middle of a turn (U-turn, Parting Shot, Eject Button) is searched as if the turn ended there; the rest of that turn's queue is not rebuilt.
- Sleep and Toxic counters are not tracked; the rebuilt battle starts them fresh.
- The Speed bound skips turns with Quick Claw, After You, Quash or Instruct, and movers with Speed abilities it does not model (Unburden, Quick Feet, Protosynthesis, Quark Drive).
- The Stat Points dataset holds out the 49 meta teams that are the opponents here, so a foe's exact spread is never looked up; species seen only on those teams fall back to a default spread. Since 2026-11-25 (#418) it also holds out the held-out evaluation teams (ranks 51-100): 504 pool teams, down from 573, and its `_excluded` record lists both sets. The PR #415 benchmarks ran on the 573-team file. `check_spreads()` rebuilds the table from the pool and refuses a file that differs.

## Rust Regulation M-B doubles search engine (`code/rust/poke-engine-doubles`)

Upstream `poke-engine` (`f7b147f`) already supports Pokémon Champions singles (`feature = "champions"`, Stat Points `8*sp - 4`, sleep max 2 turns, freeze thaw 25%, paralysis 12.5%, Champions Mega Evolutions), but hardcodes one active Pokémon per side across ~25k lines in `src/genx/`. To get native search throughput for VGC 2026 Reg M-B doubles while keeping Pokémon Showdown as the ground-truth live battle simulator, `code/rust/poke-engine-doubles` implements a compact stack-copyable 2-vs-2 doubles simulator and search daemon fed by exact Reg M-B dex tables exported from the pinned Showdown runtime (`code/data/regmb-dex.json`: 1,517 species, 938 moves, 583 items, 91 Champions Mega Evolutions), including native 90-option bring-4 Team Preview search.

- **Throughput vs. Showdown JS:** Decoupled-UCT MCTS **605,184 iterations/s** vs. 594/s (**1,019× faster**); joint-action progressive-widening PUCT **243,072/s**; UCT with random playouts **143,616/s**; Flat Monte Carlo **158,816/s** vs. 84/s (**1,891× faster**).
- **Differential audit against Showdown** (`code/scripts/audit_rust_vs_showdown.cjs`): 120 real top-50 Reg M-B doubles turns (960 Pokémon HP checks) give **96.35% faint agreement** and **0.0456 mean absolute HP fraction error** (including Showdown damage rolls, crits, and misses).

## Battle noise comparison (2026-11-23)

Chart: [reports/foul-play-doubles/noise.html](../reports/foul-play-doubles/noise.html). Raw runs: `~/vgc-data/foul-play-doubles/2026-11-23-noise/`. Self-play on 24 fixed top-50 × top-50 meta team pairs, 8 games per pair (192 battles per policy), slots shuffled per game (except the fixed-slot control). For reference, all 1,176 top-50 × top-50 cells stored in `~/vgc-data/matchup_regmb.sqlite` for the current stochastic behaviour-cloning policy (`policy_id = 8`, 8 games/cell = 9,408 battles) have within-pair variance **0.2336** (1-game SD 0.4833), true matchup SD 0.1257, Matchup ICC **0.063**, and 49-game label SE 0.0690.

| Policy | Within-pair var | 1-game SD | True matchup SD | Matchup ICC | 49-game label SE | Split-half $r$ |
|---|---:|---:|---:|---:|---:|---:|
| Behaviour cloning, deterministic, fixed slot order (control) | 0.1726 | 0.4155 | 0.2850 | 0.320 | 0.0594 | 0.524 |
| Rust Reg M-B Flat Monte Carlo (30k iters) | 0.1823 | 0.4270 | 0.2676 | 0.282 | 0.0610 | 0.635 |
| Rust Reg M-B UCT with random playouts (30k iters) | 0.1868 | 0.4322 | 0.2562 | 0.260 | 0.0617 | 0.526 |
| Rust Reg M-B Foul Play, deterministic argmax (30k iters) | 0.1935 | 0.4398 | 0.2392 | 0.228 | 0.0628 | 0.626 |
| Rust Reg M-B UCT with random playouts (100k iters, 1 sample) | 0.1987 | 0.4457 | 0.2332 | 0.215 | 0.0637 | 0.490 |
| Rust Reg M-B UCT with random playouts (4 × 25k iters) | 0.2009 | 0.4482 | 0.2197 | 0.194 | 0.0640 | 0.375 |
| Rust Reg M-B Foul Play, 75% stochastic vote (30k iters) | 0.2068 | 0.4548 | 0.2148 | 0.182 | 0.0650 | 0.460 |
| Behaviour cloning, deterministic argmax (shuffled) | 0.2068 | 0.4548 | 0.2106 | 0.177 | 0.0650 | 0.507 |
| Rust Reg M-B Foul Play, 2-turn rollout + doubles eval (30k iters) | 0.2091 | 0.4572 | 0.2093 | 0.173 | 0.0653 | 0.401 |
| Showdown JS Foul Play, deterministic argmax (400 ms) | 0.2113 | 0.4597 | 0.2035 | 0.164 | 0.0657 | 0.592 |
| Showdown JS Flat Monte Carlo (400 ms) | 0.2083 | 0.4564 | 0.1883 | 0.145 | 0.0652 | 0.389 |
| Rust Reg M-B Foul Play, doubles evaluator (30k iters) | 0.2083 | 0.4564 | 0.1825 | 0.138 | 0.0652 | 0.483 |
| Showdown JS Foul Play, 2-turn rollout + doubles eval (400 ms) | 0.2165 | 0.4653 | 0.1808 | 0.131 | 0.0665 | 0.448 |
| Behaviour cloning, stochastic (current baseline, shuffled) | 0.2225 | 0.4717 | 0.1710 | 0.116 | 0.0674 | 0.196 |
| Showdown JS Foul Play, doubles evaluator (400 ms) | 0.2180 | 0.4669 | 0.1679 | 0.115 | 0.0667 | 0.325 |
| Showdown JS UCT with random playouts (400 ms) | 0.2292 | 0.4787 | 0.1516 | 0.091 | 0.0684 | 0.597 |
| Showdown JS Foul Play, 75% stochastic vote (400 ms) | 0.2277 | 0.4772 | 0.1473 | 0.087 | 0.0682 | 0.158 |
| Rust Reg M-B Foul Play, joint-action PUCT (30k iters) | 0.2232 | 0.4725 | 0.1414 | 0.082 | 0.0675 | 0.183 |
| Showdown JS Foul Play, joint-action PUCT (400 ms) | 0.2366 | 0.4864 | 0.1235 | 0.061 | 0.0695 | -0.039 |

## Head-to-head policy trade-offs in Rust (2026-11-23)

Chart: [reports/foul-play-doubles/tradeoffs.html](../reports/foul-play-doubles/tradeoffs.html). Raw runs: `~/vgc-data/foul-play-doubles/2026-11-23-h2h/`. Each row is 100 mirrored pairs (200 battles) on top-50 meta teams with sides swapped on identical slot shuffles.

| Budget | First policy | Opponent | First policy share (95% paired CI) |
|---|---|---|---:|
| 30k iters | Rust Foul Play, deterministic argmax | Behaviour cloning, stochastic (`bc`) | **0.900** (0.86–0.94) |
| 30k iters | Rust Foul Play, joint-action PUCT | Behaviour cloning, stochastic (`bc`) | **0.875** (0.83–0.92) |
| 30k iters | Rust Foul Play, 2-turn rollout + doubles eval | Behaviour cloning, stochastic (`bc`) | **0.870** (0.82–0.92) |
| 30k iters | Rust UCT with random playouts | Behaviour cloning, stochastic (`bc`) | **0.860** (0.82–0.90) |
| 30k iters | Rust Foul Play, doubles evaluator | Behaviour cloning, stochastic (`bc`) | **0.845** (0.80–0.89) |
| 30k iters | Rust Flat Monte Carlo | Behaviour cloning, stochastic (`bc`) | **0.845** (0.80–0.89) |
| 30k iters | Rust Foul Play, deterministic argmax | Behaviour cloning, deterministic (`bc_det`) | **0.810** (0.75–0.87) |
| 30k iters | Rust UCT with random playouts | Behaviour cloning, deterministic (`bc_det`) | **0.810** (0.76–0.86) |
| 30k iters | Rust Flat Monte Carlo | Behaviour cloning, deterministic (`bc_det`) | **0.785** (0.73–0.84) |
| 30k iters | Rust UCT with random playouts | Rust Flat Monte Carlo | **0.730** (0.67–0.79) |
| 30k iters | Rust Foul Play, deterministic argmax | Rust Flat Monte Carlo | **0.605** (0.55–0.66) |
| 30k iters | Rust Foul Play, deterministic argmax | Rust UCT with random playouts | 0.430 (0.36–0.50) |
| 30k iters | Rust Foul Play, deterministic argmax | Rust Foul Play, doubles evaluator | 0.450 (0.39–0.51) |
| 200 ms wall | Rust UCT with random playouts | Rust Foul Play, deterministic argmax | 0.515 (0.44–0.59) |
| 200 ms wall | Rust UCT with random playouts | Rust Foul Play, doubles evaluator | 0.535 (0.48–0.59) |
| 200 ms wall | Rust UCT with random playouts | Rust Foul Play, joint-action PUCT | 0.440 (0.37–0.51) |

- **Best offline labeler overall:** **Rust UCT with random playouts (30k iters)** and **Rust Foul Play deterministic argmax (30k iters)** dominate both criteria simultaneously. Both crush stochastic behaviour cloning (`0.860` and `0.900`) and deterministic behaviour cloning (`0.810` each), beat Flat Monte Carlo strictly above 0.5 (`0.730` and `0.605`), and double to quadruple matchup signal-to-noise (ICC `0.260` and `0.228` vs. `bc`'s `0.116` on the same 24 pairs and `0.063` in SQLite).
- **Per-iteration vs. wall-clock trade-off:** Per iteration, UCT with full random playouts beats `poke-engine`'s static heuristic (`0.570` vs. `0.430`), because 4v4 doubles games are short (~6–8 turns) and full playouts avoid singles-heuristic bias. At equal wall-clock time (200 ms), static-heuristic Foul Play runs ~3–4× more iterations and ties UCT rollout (`0.515` [0.44–0.59]), while joint-action PUCT reaches ~50k iterations and edges ahead head-to-head (`0.560` vs. UCT rollout) at the cost of much higher self-play variance (`ICC = 0.082`).
- **Noise floor:** Scaling Rust UCT rollout from 30k to 100k iterations (or 4 × 25k determinizations) keeps within-pair variance near `0.19–0.20` (1-game SD `~0.43–0.45`), showing that 30k iterations already saturates search noise; the residual variance is inherent Pokémon RNG and open-team-sheet lead/spread uncertainty.
