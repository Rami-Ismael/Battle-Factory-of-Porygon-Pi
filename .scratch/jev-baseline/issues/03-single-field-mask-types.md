# 03: Every single-field mask type

**What to build:** Masks that hide abilities, items, moves, Stat Alignment or Stat Points are filled end to end with the fake chooser, one question per hidden field in slot order, each answer written in before the next question. Stat Points come from a menu of spreads seen for that species, with generic spreads for species that have none. Each field type has its own instruction template and JSON option descriptions.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] With the fake chooser, every mask of these task types on the random, held-out and meta grids yields a team the Showdown validator accepts.
- [ ] A later question's state contains every earlier answer.
- [ ] Every question fits the 32k-token limit.

Source: `code/docs/jev-baseline-spec.md`.
