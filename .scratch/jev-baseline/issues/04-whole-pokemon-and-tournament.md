# 04: Whole-Pokémon masks with the species tournament

**What to build:** A hidden Pokémon is filled species first through the tournament (random groups sized by the 255-option limit and the token budget, one winner per group, then a final between the winners), then ability, item, four moves, Stat Alignment and Stat Points in that order. The random split is recorded. Masks hiding up to all six Pokémon work.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] With the fake chooser, every whole-Pokémon mask on the three grids yields a team the Showdown validator accepts.
- [ ] No question sends more than 255 options or more than 32k tokens.
- [ ] The final question offers exactly the group winners, and the split is in the record.

Source: `code/docs/jev-baseline-spec.md`.
