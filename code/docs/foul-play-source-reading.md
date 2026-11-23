# How Foul Play chooses a move

Foul Play samples several complete battles consistent with what it has seen, runs poke-engine's simultaneous-move MCTS on each, and votes with the root visit shares. Every layer assumes one active Pokémon per side, so a doubles version has to replace the engine, the state tracker and the inference, which is what [foul-play-doubles.md](foul-play-doubles.md) does.

Sources: [pmariglia/foul-play](https://github.com/pmariglia/foul-play) at `8f541f8` (2026-10-03) and [pmariglia/poke-engine](https://github.com/pmariglia/poke-engine) at `f7b147f` (2026-10-11), read 2026-11-23. Paths below are inside those two repositories.

## The decision loop

- `find_best_move` asks the mode for a number of sampled battles and a search time each (`foul-play/fp/search/main.py:58`, `:64`), prepares the samples, and runs one MCTS per sample in a process pool (`main.py:47`).
- `select_move_from_mcts_results` adds each sample's root visit shares, weighted by the sample's chance (1/N), keeps the options within 75% of the best (`main.py:38`) and draws one in proportion (`main.py:16`).
- Defaults: 100 ms of search per sample and one sample (`foul-play/fp/config.py:120`, `:126`); the competition entry raised both.

## The search (poke-engine)

- Decoupled UCT: each side picks its own option by UCB1 with exploration term `sqrt(2 ln N / n)` (`poke-engine/src/mcts.rs:219`, `:224`), and the pair of picks indexes the child.
- One new node per iteration. Chance outcomes are explicit children weighted by probability, and damage rolls branch only at the first two plies (`mcts.rs:149`).
- No rollouts: a leaf scores `sigmoid(0.0125 * (evaluate(state) - evaluate(root)))` (`mcts.rs:17`, `:19`, `:196`).
- `evaluate` (`poke-engine/src/genx/evaluate.rs:159`) sums, per side, 30 per Pokémon alive (`:7`), 100 × HP fraction, status penalties, 10 for holding an item, the active Pokémon's boosts, substitute, confusion and leech seed, and side conditions weighted by turns left.

## Sampling the opponent

- Each sample deep-copies the battle and fills every opponent Pokémon's set (`foul-play/fp/search/standard_battles.py:296`).
- A set comes first from team datasets (whole public teams), skipped a quarter of the time once a move is known so samples vary (`standard_battles.py:178`), then from a team-dataset set with moves drawn separately (`:57`), then from Smogon usage combinations (`:204`). Revealed moves, items and abilities filter every source.
- In Battle Stadium Singles (bring 3 of 6), team preview is searched too. The opponent's preview visit shares become "affinities" (`foul-play/fp/search/bss.py:21`), and later samples drop unrevealed opponent Pokémon with weight 1/affinity until three remain (`bss.py:36`, `:45`).

## Inference from the log

- `check_speed_ranges` (`foul-play/fp/battle/inference.py:133`) sets a minimum or maximum Speed on the opponent's active Pokémon from which of two equal-priority moves went first, ignoring items (the item is applied when a set is checked against the range). It skips the whole turn if anyone switched (`:161`), could not move, hit itself in confusion, or had Quick Claw, Quick Draw or Custap Berry activate.
- `check_choicescarf` (`inference.py:344`) infers a Choice Scarf when the opponent outsped a Pokémon its sampled sets cannot outspeed.
- Both require one or two moves in the turn (`inference.py:187`, `:359`): one per side.

## Singles-only assumptions

| Assumption | Where | What doubles needs |
|---|---|---|
| One active Pokémon per side | `poke-engine/src/state.rs:1056`, `:1057` (`Side.active_index`); `foul-play/fp/battle/state.py:207`, `:209` (`Battler.active`) | two active slots, spread moves, targets, ally interactions |
| A side's options are one flat list | `mcts.rs` UCB per side | joint options for two slots (about 100), or statistics per slot |
| A turn has at most two moves | `inference.py:187`, `:359` | up to four moves; compare pairs inside a stretch with no Speed change |
| Any switch voids the turn's Speed check | `inference.py:161` | doubles turns often contain a switch; compare only moves after it |
| Team preview picks 3 (BSS) or a lead | `bss.py:74` | 2 leads and 2 back from 6: 90 options per side |
| Mechanics are poke-engine's Rust reimplementation | `poke-engine/src/genx/*` | no doubles mechanics exist there; use Showdown's simulator |
