# Pokémon team-construction search space

Created 2026-09-18 for the introduction/search-space discussion. This is a **computed conservative construction model of a frozen simulator domain**, not a verified census of every team available in the live game, a count of strategies, or a count of strong counter-teams.

## Figures

- `team-search-space.png`, `.svg`, `.pdf`: integrated decision tree and cumulative growth chart.
- `decision-tree.png`, `.svg`, `.pdf`: the branching example by itself.
- `search-space-growth.png`, `.svg`, `.pdf`: the cumulative count by itself.
- `domain.json`: extracted per-form abilities, moves, items, source provenance, and validation results.
- `counts.json`: exact integer counts and log10 values, computed without floating-point arithmetic until plotting.
- `extract.cjs`, `render.py`: reproducible extraction, counting, verification, and rendering.

## Scope and source

Format: `gen9championsvgc2026regmb`. Showdown revision: `913da3602a3aa1db79f9fdc5d5222eaf8d39569d`.

The input is the existing, unchanged [project validator bundle](../search-loop/runtime/VALIDATOR.md), whose manifest records the provenance of all bundled modules. The extractor exposes internal read-only APIs in memory; it does not change the saved bundle. This is the project's frozen Regulation M-B instance, not a claim that M-B remains the current regulation.

Primary references:

- [Official M-B regulation](https://champions-news.pokemon-home.com/en/page/776.html): eligible-species mechanism, Item Clause, and one Mega Evolution per battle.
- [Pinned Champions learnsets](https://github.com/smogon/pokemon-showdown/blob/913da3602a3aa1db79f9fdc5d5222eaf8d39569d/data/mods/champions/learnsets.ts).
- [Pinned team validator](https://github.com/smogon/pokemon-showdown/blob/913da3602a3aa1db79f9fdc5d5222eaf8d39569d/sim/team-validator.ts).
- [Pinned Champions rules](https://github.com/smogon/pokemon-showdown/blob/913da3602a3aa1db79f9fdc5d5222eaf8d39569d/data/mods/champions/rulesets.ts).

## What is counted

Six distinct National Dex species, each realized as one modeled form, with:

- One eligible ability.
- One of 148 eligible held items or no item. Nonempty items cannot repeat; no-item choices can repeat. A held Mega Stone on a nonmatching species is accepted by this frozen validator and is counted, even though it will not enable that species to Mega Evolve.
- Exactly four distinct unordered moves from the form's explicit `9M` learnset, or its sole move for Ditto. Only individually validated species–ability–move choices are retained. Inherited/event sources are excluded. This deliberately omits some legal configurations rather than multiplying a global move count across all species.
- One of 21 stat alignments (20 modifying alignments plus one neutral representative).
- Exactly 66 integer Stat Points over six stats, each between 0 and 32 inclusive.

Roster order and move order are not counted. The model takes one representative ordering; it does not claim that all pilots/mechanics are order-invariant. Gender variations, cosmetic choices, nicknames, and other presentation fields are not multiplied into the count. IVs and Tera types are not variable dimensions in this Champions instance. Mega battle forms are not independently selected as roster members: their stones are included in the item layer.

The extraction retains 208 species and 232 form representatives with explicit eligible moves. This differs from the older vault note's 231-form scrape: it is a separately computed, pinned simulator domain and is not presented as a correction to the historical website snapshot. The headline 4.17 × 10^111 is a lower-bound *model*, conditional on the pinned implementation and the separability of the explicit `9M` options. Finite validation tests do not exhaustively prove that every counted combination is legal in the live game.

## Counting method

Let f range over the retained forms of National Dex species s, and a over its eligible abilities. Let m(f,a) be the number of retained moves for that form and ability. A form contributes C(m,4) move sets, except Ditto, which contributes one.

Group form weights by species, then use the coefficient of x^6 in the product over species of (1 + w_s x). This selects six different species and counts the alternative forms/configurations without enumerating all teams. The dynamic program stores seven arbitrary-precision integers and updates degrees 6 down to 1 for each species.

At successive stages, the species weight w_s is:

1. One for species-only combinations.
2. The number of retained forms for that species.
3. The sum of eligible ability counts across its forms.
4. The sum over its forms and abilities of C(m(f,a),4), with the Ditto exception.

The held-item factor for six distinct members is:

`sum_{k=0..6} C(6,k) P(148,k) = 9,889,059,373,669`.

Choose which k members hold something; injectively assign k different held items to them. Other members hold nothing. This is not 149^6, which would allow duplicate held items.

Multiply by 21^6 for alignments and S^6 for stat spreads, where:

`S = [x^66](1 + x + ... + x^32)^6 = 10,008,272`.

The spread count is independently checked by dynamic programming and inclusion–exclusion. The six-spread multiplier is a sixth power, not a multiplication by six. The diagram's “Repeat for all six members” means that the same kinds of choices must be made for each member, not that every species has Garchomp's move count.

## Rounded stage totals

| Stage | Configurations |
|---|---:|
| Six species | 1.05 × 10^11 |
| Choose forms | 1.99 × 10^11 |
| Add abilities | 3.93 × 10^13 |
| Assign held items | 3.89 × 10^26 |
| Choose movesets | 4.84 × 10^61 |
| Choose alignments | 4.15 × 10^69 |
| Allocate all 66 Stat Points | **4.17 × 10^111** |

The horizontal coordinate is log10(configurations); a difference of 20 units means a factor of 10^20. Bar lengths should not be interpreted as ordinary ratios of team counts. Species-specific counts are summed with the dynamic program; no median-movepool approximation is used.

If unspent Stat Points are also allowed, the bounded allocation count is 136,663,185 per member and the same construction model expands to approximately **2.71 × 10^118**. This sensitivity is recorded in `counts.json` but deliberately omitted from the main figure to keep one clear counting convention.

## Verification

- 69,655 validator calls in the extraction run, including per-move/ability checks, all representative-form/item choices, and 500 fixed-seed complete teams.
- 64 rejected move/ability cases excluded (the unavailable Greninja Battle Bond variant).
- Zero item errors across the tested representative-form/item combinations.
- Zero errors for the 500 generated complete teams with randomized species, abilities, items, moves, alignments, and fully spent Stat Points.
- Species-group dynamic programming checked against direct enumeration on a toy domain.
- Item assignment formula checked against direct enumeration for small pools and team sizes.
- Stat-spread dynamic programming checked against inclusion–exclusion.
- Example Garchomp movesets checked against the extracted move pool.
- Figures rendered and visually inspected; PNG, editable-text SVG, and vector PDF exports supplied.

These checks support reproducibility and representative consistency; they do not turn a combinatorial model into an exhaustive validation of more than 10^111 teams.

## Reproduce

From this vault project's root:

```sh
node Blog/visuals/team-search-space/extract.cjs
python3 Blog/visuals/team-search-space/render.py
```

Rendering requires matplotlib and Pillow. The extraction requires only Node's built-in modules plus the already bundled validator. All numeric outputs and source records are local; the rendering step downloads nothing.

## Suggested figure caption

**Full configuration dominates species selection.** (A) A partial decision tree for Garchomp illustrates how one species branches into abilities, held items, and four-move sets; each path also requires an alignment and a Stat Point allocation. Only selected branches are shown. (B) Cumulative counts from a conservative model of the pinned Pokémon Champions Regulation M-B domain, using species-specific options, Species Clause, Item Clause, and exactly 66 Stat Points per member. The model grows from approximately 10^11 species combinations to 4.17 × 10^111 configured teams. The axis is logarithmic. Counts describe configurations under the stated model, not distinct strategies, viable teams, or a verified census of the live game's complete legal space.
