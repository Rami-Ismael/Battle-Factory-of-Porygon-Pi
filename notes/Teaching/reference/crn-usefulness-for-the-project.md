---
created_at: 2026-08-20
updated_at: 2026-08-20
type: research-report
verdict: "CRN idea for team-vs-team comparison: MED (opponent-draw pairing works, needs no seed support); the paper's X_DD theorem for this pipeline today: LOW (guards a rollout-planning setting the project does not have)"
---

# Is the CRN paper (Yadav et al., RLJ 2026) actually useful here?

Question: is "Using Common Random Numbers for Simulation-based Planning with Rollouts" (Yadav, Maliakkal, Khadilkar & Kalyanakrishnan) actually useful for this project, or only plausibly-adjacent? The owner's claimed use: "run team A and team B against the same opponent under the same PRNG seeds; directly cuts the battle count."

Short answer: the same-opponent half of that sentence is real and free; the same-PRNG-seeds half mostly does not survive contact with how Showdown consumes randomness, and poke-env cannot set the seed anyway.

## 1. Showdown seed support (verified in source)

**(a) A battle CAN be started with a fixed seed — at the BattleStream/CLI level.** `sim/SIMULATOR.md` documents the `>start OPTIONS` message: "`seed` - an array of four numbers representing a seed for the random number generator (defaults to a random seed)" 🟢 [SIMULATOR.md, "seed" bullet under `>start OPTIONS`]. `sim/battle-stream.ts` (`case 'start':`) parses that JSON and passes it to `new Battle(options)`; `sim/battle.ts` declares `seed?: PRNGSeed; // PRNG seed` in its options (line 67) and does `this.prng = options.prng || new PRNG(options.seed || undefined)` (line 224). 🟢 Note the SIMULATOR.md wording is stale: a four-number array is the legacy Gen-5 seed; the current native seed is a string `` `${'sodium' | 'gen5' | number},${string}` `` (`sim/prng.ts` line 18), and the `PRNG` constructor joins arrays "compat for old inputlogs".

**(b) How the PRNG is consumed: one serial stream per battle, advanced call-by-call.** `sim/prng.ts` 🟢: default is `SodiumRNG` — a ChaCha20-based drop-in for libsodium's `randombytes_buf_deterministic`; each `next()` encrypts a zero buffer, keeps 32 bytes as the next internal seed and emits 4 bytes as the output (lines 198–208). The Gen-5 64-bit LCG (`a = 0x5D588B656C078965`, `c = 0x00269EC3`, `m = 2^64`) survives only for old-seed compatibility (lines 235–295). Every random event in a battle — damage rolls, accuracy, secondaries, `sample`, `randomChance` — draws from this single self-chaining stream via `battle.random(...)` wrappers (`sim/battle.ts` lines 347–356). **Speed ties consume the same stream**: `prng.shuffle` is "A Fisher-Yates shuffle. This is how the game resolves speed ties" (`sim/prng.ts` lines 145–150), called from `sim/battle.ts` line 456. Team order / lead selection do NOT consume it — they are player choices (`>p1 team 123456` in SIMULATOR.md's example). There is also a `>reseed` message that re-seeds mid-battle (`sim/battle-stream.ts` lines 113–116).

**(c) The load-bearing implication: shared seeds desynchronize as soon as the battles diverge.** There is no per-event keying (no counter-based RNG keyed by turn/event); draws are consumed in whatever order events fire. Two battles with different player-1 teams make different numbers and orders of PRNG calls from the first turn (different moves → different accuracy/damage/secondary rolls; different speed-tie shuffles), after which "corresponding" events in the two battles read different positions in the stream — and ChaCha20 output at different stream positions is effectively independent. So a shared seed aligns randomness only up to the first divergent PRNG call — which for two different candidate teams can precede even turn 1 (switch-in speed ties, ability order). The honest engineering statement: **fixed seeds buy reproducibility (replayable input logs — that is the stated purpose in the `prng.ts` header comment), not cross-team variance coupling.** `>reseed` at turn boundaries could re-align stream *position*, but still would not key draws to events within a turn.

## 2. poke-env seed control (verified in source)

**No battle-seed control exists in poke-env today.** poke-env creates battles as a *player* over websocket: `/challenge username, format_` (`src/poke_env/ps_client/ps_client.py` line 130), `/search format_` (line 305), team via `/utm packed_team` (line 330) 🟢 — none of these chat commands carry a seed. A repo-wide grep for "seed" finds only Leech Seed/Bullet Seed and the Gymnasium `reset(seed=...)` in `src/poke_env/environment/env.py` (lines 470–474), which seeds *numpy* for the env wrapper, not the Showdown battle PRNG. Server-side, `RoomBattleOptions` does have `seed?: PRNGSeed` (`server/room-battle.ts` line 490, forwarded into `>start` at lines 575–580) 🟢, but no websocket command sets it *at battle creation*. One correction from the verification pass: mid-battle, the admin-gated `/editbattle reseed [seed]` chat command (`server/chat-commands/admin.ts` lines 1666–1687, requires the `forcewin` permission — Administrator/Host groups in `config/config-example.js`) reaches `battle.resetRNG` over websocket 🟢 — usable on this project's *self-hosted* server where the owner is admin, though it re-seeds a running battle rather than setting its initial seed, and re-aligning stream position still does not re-align events across diverged states. **Initial-seed control therefore requires bypassing poke-env**: drive `sim/battle-stream.ts` / the `pokemon-showdown simulate-battle` CLI directly with `>start {"formatid":..., "seed":...}`, or modify the server to pass a seed into `RoomBattleOptions`.

## 3. The always-available coupling (needs no simulator support)

Pairing the **opponent draw** — battle *i* of team A and battle *i* of team B face the same sampled team from the collected meta-team list weighted by frequency, played by the same opponent battle policy — is classical paired comparison. The variance argument is one identity, quoted in the paper's own Proposition 1 proof: "var(Y−Z) = var(Y) + var(Z) − 2cov(Y,Z)" 🟢 [arXiv 2605.04732v1, §1.2] — sharing the opponent draw makes the two win-rate estimates positively correlated (a hard opponent is hard for both), so the difference estimator's variance drops by 2·cov. The classical reference the paper leans on is Glasserman & Yao (1992), "Some Guidelines and Guarantees for Common Random Numbers", *Management Science* 38, 884–908 🔒 (no free copy found; cited here via the Yadav paper's bibliography, not read). **This channel needs nothing from the paper and nothing from Showdown — it is stratified/paired sampling of opponents, implementable in the poke-env harness today.** Two implementation caveats from the verification pass: (i) the opponent battle policy's own decision randomness should be seeded too — Showdown's built-in `RandomPlayerAI` carries its *own* PRNG separate from the battle's (`sim/tools/random-player-ai.ts` line 27) 🟢, and a learned policy's action sampling likewise — otherwise unseeded opponent-decision noise stays unpaired; (ii) the opponent's team-preview selection is conditioned on *your* team, so pairing delivers the same opponent team and policy, not the same four-selection — that difference is signal about the candidates, not a defect of the pairing.

## 4. Scope match against the paper's actual contribution

From the paper 🟢 (arXiv 2605.04732v1, fetched in full):

- Three estimators of value difference: X_I (independent runs), X_D (fully shared randomness), X_DD (shared up to depth d, then shared *rollout* continuation).
- **Proposition 1**: X_D has NO guarantee — there exist MDPs and policy pairs where var(X_D) > var(X_I) (negative covariance construction).
- **Theorem 2**: var(X_DD) ≤ var(X_I) on *all* MDPs, strict under mild conditions — but only when "policies π₁, π₂ agree after d steps". That precondition is the whole theorem. It arises "quite naturally in rollout-based planning": two root actions followed by the *same* rollout policy. Two different candidate teams never satisfy it — their induced policies differ at every turn.
- Empirics: "the advantage of (depth-)dependence is pronounced when the number of simulations is small"; the dependent scheme "outperforms the independent variant and is indistinguishable from the depth-dependent variant" on their tasks; the UCT deployment (Ludo) passes only the root action to determine the seed, which "does not find theoretical support" but helps empirically.

**What transfers NOW:** (i) the opponent-draw pairing of §3 — real, cheap, classical, not this paper's contribution; (ii) seed discipline for *reproducibility* via BattleStream (input logs, regression-testing the harness); (iii) the qualitative lesson that coupling matters most at small per-comparison battle budgets — relevant whenever candidate teams are compared by simulated battles under a tight budget. **What transfers only IF a UCT/rollout-lookahead battle-policy arm is ever added** (none exists — the arms are behaviour cloning and PPO self-play variants, all policy-gradient over a 107-way action space): X_DD itself, coupling root-action comparisons *within one decision inside one battle*. And even then, applying it on Showdown would require an event-keyed sampling model the simulator's serial stream does not provide.

## Adversarial verification pass (2026-08-20, second agent)

A second agent independently tried to *refute* §1–§2 against the same repositories (files fetched: `sim/prng.ts`, `sim/battle.ts`, `sim/battle-stream.ts`, `sim/state.ts`, `sim/tools/random-player-ai.ts`, `server/room-battle.ts`, `server/chat-commands/admin.ts`, `server/user-groups.ts`, `config/config-example.js`; poke-env `env.py`, `ps_client.py` + repo-wide grep).

- **§1 CONFIRMED.** Single self-chaining `SodiumRNG` stream per battle, no per-event keying, no child/per-context PRNGs anywhere in `sim/`; permanent decoupling after the first divergent call stands. Nuances folded in above: separate streams exist (`RandomPlayerAI`'s own PRNG; random-format team generation) but none rescues cross-team coupling, and `Battle.resetRNG` / `>reseed` lets a harness driving `BattleStream` re-align stream *position* per turn — without re-aligning events across diverged states.
- **§2 poke-env side CONFIRMED; server-side clause corrected.** The original "nothing a websocket player sends can set it" was overstated: the admin-gated `/editbattle reseed` exists (correction now in §2). The narrow claim — no user input populates `RoomBattleOptions.seed` at creation — held under full grep.
- **§3 inference holds**, with the two caveats added above.

The verdicts below are unchanged by verification.

## Verdict

- **(a) The CRN *idea* for team-vs-team battle comparisons — MED.** Pairing the opponent draw (same sampled meta team, same opponent policy, ideally same battle index) is a genuine, zero-cost variance reduction the harness should adopt; but it is textbook paired comparison, the "same PRNG seed" channel adds almost nothing because Showdown's single serial stream desynchronizes at the first divergence, and poke-env cannot set seeds without bypassing it.
- **(b) The paper's *specific theoretical contribution* (X_DD, Theorem 2) for this pipeline as it exists today — LOW.** The theorem's precondition (compared policies identical beyond depth d) is satisfied by no comparison this project currently makes; it becomes relevant only if a rollout/UCT battle-policy arm is added, which is not planned.

The owner's TODO claim should be corrected to: "run team A and team B against the same *opponent draws*" — that part cuts the battle count for a given comparison precision; "under the same PRNG seeds" does not, and is not currently possible through poke-env.

## Sources (all fetched 2026-08-20)

- 🟢 [sim/SIMULATOR.md (smogon/pokemon-showdown, master)](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md)
- 🟢 [sim/prng.ts](https://github.com/smogon/pokemon-showdown/blob/master/sim/prng.ts) · 🟢 [sim/battle.ts](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle.ts) · 🟢 [sim/battle-stream.ts](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle-stream.ts) · 🟢 [server/room-battle.ts](https://github.com/smogon/pokemon-showdown/blob/master/server/room-battle.ts)
- 🟢 [poke-env master tarball: src/poke_env/ps_client/ps_client.py, src/poke_env/environment/env.py](https://github.com/hsahovic/poke-env)
- 🟢 [Using Common Random Numbers for Simulation-based Planning with Rollouts by Yadav, Maliakkal, Khadilkar & Kalyanakrishnan (arXiv 2605.04732v1 HTML)](https://arxiv.org/html/2605.04732v1) — RLJ 2026 [Paper52](https://rlj.cs.umass.edu/2026/papers/Paper52.pdf)
- 🔒 Glasserman & Yao (1992), "Some Guidelines and Guarantees for Common Random Numbers", *Management Science* 38(6), 884–908 — no free copy located; cited via the Yadav paper's bibliography only.
