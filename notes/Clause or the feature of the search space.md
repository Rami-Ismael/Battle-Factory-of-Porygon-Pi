---
created_at: 2026-08-17
updated_at: 2026-08-20
type: reference
regulation: Pokémon Champions VGC 2026 Regulation Set M-B
counts_pulled_on: 2026-08-17
sources:
  - https://rotompicks.com/en/pokemon/
  - https://rotompicks.com/en/items/
  - https://rotompicks.com/en/moves/
  - https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_in_Pok%C3%A9mon_Champions
  - https://champsdex.com/posts/pokemon-champions-ev-iv-stats-guide-2026/
  - https://game8.co/games/Pokemon-Champions/archives/538683
tags:
  - clauses
  - regulation-mb
  - search-space
  - vgc
  - combinatorics
---
**

1. You can use only legal move for the pokemon
	*Range —* **1 of 1–3**. Across the 231 forms: 24 have a single ability, 87 have two, 120 have three. **186** distinct abilities in the pool.
2. **One Mega Evolution per battle** — you may *hold* several Mega Stones, you may only Mega Evolve one Pokémon per battle. Reg M-B has no Terastallization. Unlike 1 and 2 this removes **nothing** from the team space: it binds on the battle policy at selection time, not on team building. See [[Regulation M-B — the Mega Evolution constraint]]
	*Range —* **75** Mega Stones — half the item list — across **73** Dex numbers; Charizard `#0006` and Raichu `#0026` have two Megas each. There are 306 battle forms (231 buildable + 75 Mega), but you never *pick* a Mega form: you pick the base form and the stone.
3. EV
	*Range —* **there are no EVs in Champions, and no IVs either.** EVs are replaced by **Stat Points**: 66 per Pokémon, capped at 32 per stat, and 1 SP is flatly +1 to that stat at level 50 — no divide-by-four. IVs do not exist as a variable; every Pokémon is treated as 31 across the board. Natures are renamed **Stat Alignments**, and there are **21**, not 25


- [x] Determine the range for each clause in the item for example — done: a range line under each clause above, and the tables below rebuilt from pulled data instead of estimates
	- [x] What is the total number of pokemon at the moment — done: **208 species / 231 buildable forms / 306 battle forms** — which one you want depends on the clause, see the table below
- [ ] Grill me on the aspect
- [x] Determine the number of pokemon on average in a regulation — done: **220 buildable forms**, but the average is the wrong statistic — see *How the pool moves between regulations*

## Total number of Pokémon — three answers, not one

The Species Clause binds on **Dex number**, but a team slot is filled by a **form**, so "how many Pokémon" splits three ways:

| What | Count | Why it's the one you want |
|---|---|---|
| Distinct National Dex numbers | **208** | The alphabet the all-different constraint runs over |
| Buildable forms | **231** | The actual domain of one team slot |
| Battle forms | **306** | 231 + 75 Megas — the number of matchup profiles $f$ eventually has to price |

208 species + 75 Megas is exactly Bulbapedia's count of the Champions roster, and 231 + 75 = 306 is RotomPicks' — the two sources agree once you know which thing each is counting. Two other sites disagree (StrataDex 228, MetaVGC 224); both look like snapshots taken before a mid-regulation roster update, and neither publishes a form-vs-species breakdown you can check. **Unresolved, and low stakes** — the pipeline searches a top-30 cut, so pool size only sets a ceiling.

## How the pool moves between regulations

Champions has run two regulation sets, so an "average pool size" averages two numbers that are **monotonically increasing** — the roster has only ever grown:

| | M-A | M-B | Δ | Mean |
|---|---|---|---|---|
| Distinct Dex numbers | 186 | 208 | +22 | 197 |
| Buildable forms | 209 | 231 | +22 | **220** |
| Mega forms | 61 | 75 | +14 | 68 |
| Battle forms | 270 | 306 | +36 | 288 |
| Legal items | 117 | 148 | +31 | 132.5 |
| Legal moves | 481 | 486 | +5 | 483.5 |

The +31 items decompose as +15 hold items and +16 Mega Stones, which matches the announced "15 new held items, 16 new Mega Evolutions" — two independently reported numbers agreeing, so the item counts are trustworthy.

**So the honest answer to "on average" is: don't use the average.** With $n{=}2$ and a roster that has only grown, 220 is a midpoint between a past and a present, not a value any future regulation will sit near. If you need a planning number, use the **current** pool (231) and assume the next set is bigger.

## Species layer — choosing which 6 species (the sets come after, at the set layer)

| Axis | Range | Size | Source |
|---|---|---|---|
| Species (Dex numbers) | 6 distinct from 208, unordered (Species Clause) | $\binom{208}{6} =$ **104,579,959,848** | exact |
| Species → forms | each of the 6 realized as one of its buildable forms | **193,744,198,529** — 1.85× the line above, all of it from the 17 multi-form families | exact |
| The searched cut | 6 from a 30-species candidate pool | **593,775** | exact, [[Decision — enumerate species subsets, don't optimise]] *(note deleted 2026-08-18, no replacement)* |
| Items across the team | all-different, `None` exempt | **9,889,059,373,669** assignments to a fixed 6 (of which $P(148,6) =$ 9,484,150,515,840 have all six holding something) | exact |

*Retired 2026-08-18 — "the searched cut" row: $\binom{30}{6}$ is a correct count but no longer the searched object; the owner deleted the enumeration decision. The search unit is a full team, i.e. the Legal / Viable rows below.*

## Per species — the competitive set

| Axis | Range | Size | Source |
|---|---|---|---|
| Moves | 4 distinct, unordered, from that form's movepool (1–102, median 60) | $\binom{60}{4} =$ **487,635** median · $\binom{102}{4} =$ **4,249,575** ceiling · **160,688,362** summed over all 231 forms | exact |
| Stat Points | 6 stats, ≤32 each, ≤66 total, +1 per point | **136,663,185** spreads (10,008,272 of which spend all 66) | exact |
| IVs | do not exist — everything is 31 | **1** | exact |
| Stat Alignment (nature) | 21 legal | **21** | exact |
| Ability | the form's legal abilities | **1–3**; 24 forms have 1, 87 have 2, 120 have 3 | exact |
| Item | one held item or none — all-different across the team (Item Clause); Mega Stones live here | **149** per slot (148 + none) | exact |
| Tera type | not a dimension in Reg M-B — no Terastallization | 1 | exact, [[Regulation M-B — the Mega Evolution constraint]] |
| Level | flat rules, fixed at 50 | 1 | exact |

Per-form competitive sets, before items: **3.5 × 10¹⁵** at the median form, **3.7 × 10¹⁶** at the widest. Garchomp specifically: 2.3 × 10¹⁵, or 3.4 × 10¹⁷ once the item choice is counted.

## Three totals

| What you count | Full pool | Top-30 cut | Source |
|---|---|---|---|
| **Legal** teams — every configuration the rules permit | **2.6 × 10¹¹⁸** | ~1.1 × 10¹¹² | exact for the full pool; the cut uses the median form as a proxy |
| **Viable** teams — ~20 real move sets, ~20 spreads people run, ~40 items, 2 abilities, 3 alignments | **1.0 × 10⁴¹** | **3.1 × 10³⁵** | every input estimated |
| **Species-only** — the space the species layer searches; sets follow at the set layer | 193,744,198,529 | **593,775** | exact, [[Decision — enumerate species subsets, don't optimise]] *(note deleted 2026-08-18, no replacement)* |

*Retired 2026-08-18 — the Species-only row as "the space the species layer searches": the enumeration decision was deleted; there is no species layer. The Viable row is the searched space's honest order of magnitude.*

The Viable row is the owner's own estimate, re-run on Champions inputs: the "4 IV configs" factor drops to **1** (there are no IVs), which is the only reason it moved. 2,400 realistic sets per species is the load-bearing guess in it, and nothing has measured that number.

## What changed when the real numbers came in

The estimates this note carried before were mainline Gen 9 numbers, and Champions breaks three of them outright:

- **The set layer is ~10⁷ times smaller than a Gen 9 estimate says.** 510 EVs at ≤252 per stat give 6,137,312,896 *distinct stat outcomes*; 66 SP at ≤32 give 136,663,185 — **45× fewer** — and the IV dimension, worth another 1.07 × 10⁹ on its own, is gone entirely. Deleting IVs is by far the bigger of the two cuts.
- **The item pool is half what a mainline count suggests** (148, not ~300), and **75 of the 148 are Mega Stones**. Six Pokémon drawing distinct items from a 148-item pool where half the pool is only usable by the 73 Mega-capable species is a real constraint, not a formality.
- **21 Stat Alignments, not 25 natures with 5 neutral.** Champions already did the dedup.

This is the ammunition for the hub's open item *"the explosion comes from what fills a slot is asserted and never computed"* — it is computed now, and the answer is that the slot layer is still astronomically large (10¹⁵ per form) but four orders of magnitude less so than the Gen 9 framing implied.

## Caveats on the counts

- **The move total is contested.** RotomPicks lists 486 legal moves; MetaVGC says 502. The per-form movepools above come from RotomPicks' own learner index (14,065 form–move pairs, covering all 231 forms), so they are internally consistent whichever headline total is right.
- **Everything here is one pull, 2026-08-17.** Champions patches its roster mid-regulation — the most likely explanation for the 224/228/231 spread — so re-pull before quoting any of this in something published.
- **The "no item" option is inferred, not quoted.** The regulations say each Pokémon *may* hold an item and that no two may hold the same one. No source explicitly confirms that a team of six item-less Pokémon is legal.
- **One source only for the 21 Stat Alignments.** The 66/32 Stat Point numbers are confirmed twice (ChampsDex, Game8); the alignment count is ChampsDex alone.

Sources (all 🟢 openable): [RotomPicks — legal Pokémon](https://rotompicks.com/en/pokemon/) · [items](https://rotompicks.com/en/items/) · [moves](https://rotompicks.com/en/moves/) · [Regulation M-A list](https://rotompicks.com/en/m-a/pokemon/) · [Bulbapedia — List of Pokémon in Pokémon Champions](https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_in_Pok%C3%A9mon_Champions) · [ChampsDex — EVs, IVs & Stats](https://champsdex.com/posts/pokemon-champions-ev-iv-stats-guide-2026/) · [Game8 — What Are Stat Points?](https://game8.co/games/Pokemon-Champions/archives/538683) · [Victory Road — Champions regulations](https://victoryroad.pro/champions-regulations/)

Vocabulary: [[CONTEXT]] · [[Regulation M-B — the Mega Evolution constraint]] · Hub:
[[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]]


# Todo 

- [ ] Create a markdown table of all the features of the search spaces for example the size of the pokemon team, number of pokemon and what are the restriction  and determine the range of it also how many pokemon are avaiblibale for each regulation write it down a temp 
	- [ ] Here is the list for now
		- [ ] Species 
			- [ ] **Species Clause** — ✅ true. No two Pokémon on your team may share a National Dex number. This is by _species_, not form, so Rotom-Wash + Rotom-Heat is illegal, as is Landorus-Therian + Landorus-Incarnate.
			- [ ] Moves clauesse - *Range —* **486** moves exist in the format, but the constraint is per-form: 4 distinct moves from that form's own movepool. Movepool size runs **1 → 102**, median **60** (Q1 51, Q3 70). The floor is Ditto — Transform only, so it has **no** legal 4-move set at all; the ceiling is Gallade at 102.
			- [ ] Number of pokemon in regulation in current meta games which 2026-09-21 is **231 buildable forms** (208 distinct species)
			- [ ] The number of pokemon item in regulation in current meta game which is **148 legal held items** — 149 choices per slot counting "none", all-different across the team, however in the current meta game does not have a support vest which increase special damage when added will change the metagames
			- [ ] The number of pokemon moves is **486 legal moves**; each Pokémon picks 4 from _its own form's_ pool (median 60, max 102) however each metagames pokemon can change the available pokemon moves for examples https://www.youtube.com/watch?v=Ci2XIZZnsXI is is will change if you have search dependences on set of variable it need to quckly change as fast possible
			- [ ] `{the number of chess moves}` isn't a standard quantity. Four different things get called that: 20 legal opening moves, ~35 average branching factor, **4.8 × 10⁴⁴ legal positions**, **~10¹²⁰ possible games** (the Shannon number).
			- [ ] Each pokemon are allowed to have one ability that ability that is allowed to pokemon to be attacked too *Range —* **1 of 1–3**. Across the 231 forms: 24 have a single ability, 87 have two, 120 have three. **186** distinct abilities in the pool. 
		- [ ] Items
			- [ ] The total number of items in pokemon
			- [ ] The number of item in the current regulation, what is the average number of item use per regulation 
			- [ ] I want a barchart of the number of item that is aviaible for each regulation
		- [ ] I found the paper i was talking about the show the math of the number of possible pokemon tema combinatino **VGC-Bench: Towards Mastering Diverse Team Strategies in Competitive Pokémon** by Angliss, Cui, Hu, Rahman and Stone (AAMAS 2026
- [ ] Provide me a definition of clause in the context of
- [x] 2026-08-20, swept RLC 2026 for papers on this note's search-space structure — done 2026-08-20: all five candidates fell to LOW on full-PDF checks; the surviving pointer for a huge pool under a finite budget is 🟢 [Bayati et al. 2020](https://arxiv.org/abs/2002.10121) — rows in [[Todo Section]]
- [ ] https://www.smogon.com/forums/threads/how-many-possible-teams-are-there-in-pokemon-a-comprehensive-and-detailed-answer.3648997/