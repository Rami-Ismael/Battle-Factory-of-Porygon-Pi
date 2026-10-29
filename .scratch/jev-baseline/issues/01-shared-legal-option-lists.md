# 01: Shared legal-option lists (prefactor)

**What to build:** One function returns the legal option names for a hidden field of a partial team (species, ability, item, move, Stat Alignment), using the same rules the DSPy experiment's legal-choice types use. The DSPy experiment builds its typed choices from it, so both arms share one source of rules.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The DSPy structure tests pass unchanged.
- [ ] The new function has its own tests on a known partial team: a move list excludes the slot's other moves, an item list excludes teammates' items, a species list excludes families already on the team.
- [ ] Ruff passes with complexity 10 or less.

Source: `code/docs/jev-baseline-spec.md`.
