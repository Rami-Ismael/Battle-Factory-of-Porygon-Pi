# 02: One item mask, end to end, no network

**What to build:** For one single-item mask, a command prints the full Jev request (state with format, rules, the top-50 meta teams from Ling's configuration, the team so far, and the field asked about; one Choice question with legal items, each described by a JSON object) and its token count. A fake chooser answers it, the answer is written into the team, and the Showdown validator accepts the filled team. This ticket also adds the chooser interface and the fake chooser that later tickets use.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] The printed request matches the structure on the ideas page's "What Jev is sent" section.
- [ ] The token count is printed and checked against the 32k-per-question limit.
- [ ] With the fake chooser, the filled team passes Showdown validation.
- [ ] No network call is made.

Source: `code/docs/jev-baseline-spec.md`.
