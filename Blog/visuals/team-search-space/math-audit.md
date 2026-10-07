# Independent audit of the team-space counts

Audited 2026-09-18 by the main agent and an independent subagent.

## Verdict

No arithmetic error was found in the stored restricted construction model. All seven stage totals were independently reproduced. The original headline is unsuitable as an unqualified answer to “how many legal teams are possible”: it counts a restricted set of configurations, not the complete legal search space.

Use this wording: **“Approximately 4.17 × 10^111 configurations in a restricted, frozen Champions M-B construction model.”** Do not describe it as a verified live-game total or a proved live-game lower bound.

## Independently checked arithmetic

- Six distinct species: C(208,6) = 104,579,959,848.
- Garchomp four-move sets: C(58,4) = 424,270.
- Held-item assignments: sum from k=0 to 6 of C(6,k) × 148!/(148−k)! = 9,889,059,373,669. This correctly excludes duplicate nonempty items while permitting multiple members with no item.
- Fully spent Stat Point spreads: C(71,5) − 6 C(38,5) + 15 C(5,5) = 10,008,272. This counts integer allocations totaling 66 with each of six stats between 0 and 32.
- Species-specific form/ability/move weights are summed within each National Dex species. The degree-six coefficient of their generating polynomial chooses six different species; this correctly avoids duplicate species and does not multiply a generic average moveset count across six slots.
- Final count: 4.174175575433252568… × 10^111.

The new `audit-counts.py` uses Newton identities rather than the original coefficient dynamic program. It uses Python's standard library, performs no writes, and checks every stored stage and the partially spent SP variant. Run:

```sh
python3 Blog/visuals/team-search-space/audit-counts.py
```

## Assumptions that change the interpretation

1. **Exactly 66 SP is a restriction, not a requirement to spend the entire budget.** The validator also accepts partially spent budgets. Allowing totals from 0 through 66 gives 136,663,185 allocations per member. Keeping all other model assumptions, this yields approximately 2.71 × 10^118 configurations. This still is not a complete legal-team census.
2. **Exactly four moves excludes other legal configurations.** Ditto is the explicit exception. A one-move Garchomp with Protect was accepted during this audit. Fewer moves therefore cannot simply be omitted when claiming to count every legal configuration.
3. **Gender is not varied.** The model also excludes roster/move ordering and cosmetic distinctions, and uses a single neutral alignment representative. These are conventions about what constitutes a different candidate, not conclusions established by combinatorics. Order should only be canonicalized if the chosen pilot/mechanics justify it.
4. **208 species and 232 forms are the retained extracted domain.** Direct 9M learnsets were used. These figures are not independently established here as the complete official eligible-species/form inventory.
5. **Legality of combinations requires more than legality of individual choices.** The extractor validates single moves, then counts their four-move subsets. The independent agent inspected active rules and found no complex bans, custom checkCanLearn override, or required move/item/ability conditions among retained species. All retained move sources are 9M. This supports the product model; it is not an exhaustive live-game proof.
6. **Configurations are not strategies.** Different parameter settings may have the same behavior or win rate. A known opposing team defines the objective; it does not itself change this feasibility count.

## Additional validator checks

The main agent validated every retained species–ability–item combination (83,440 cases), using the first four retained moves (or the sole move), Hardy, level 50, and SP 32/32/2/0/0/0. There were no errors or unexpected species/ability rewrites. This broadens the original item test, which checked only the first ability. It does not exhaust moveset interactions.

Garchomp / Rough Skin / Protect / no item / Hardy / female / level 50 was also accepted at total SP 0, 1, 32, 64, 65 and 66, distributing points into stats in order with a 32 cap. This directly demonstrates legal configurations omitted by the fully spent/four-move convention.

## Sources and boundaries

- [Official frozen M-B rules](https://champions-news.pokemon-home.com/en/page/776.html) establish the item restriction and eligible-Pokémon mechanism; the published period ended September 9, 2026. This audit concerns the frozen project format, not the current regulation.
- [Pinned Champions rules](https://github.com/smogon/pokemon-showdown/blob/913da3602a3aa1db79f9fdc5d5222eaf8d39569d/data/mods/champions/rulesets.ts).
- Local `../search-loop/runtime/showdown-validator.js`: `validateStats` enforces the 32-per-stat cap and 66-total ceiling, rather than equality to 66. `../search-loop/runtime/VALIDATOR.md` records provenance.
- Original `extract.cjs`, `render.py`, `domain.json`, `counts.json` and README were inspected. Existing outputs were preserved for comparison; this audit does not silently replace their definitions or claim a newly exhaustive count.
