1. The **VGC Google Sheet Repository**
# Work on this 
- The VGCPastes Google Sheets database — the community-maintained paste tracker for real VGC play. Per regulation (Reg MA, MB, …), it pulls only the "Champions" sheets, i.e., teams credited to players who won/placed at official events, plus a separate featured/ subset the curators highlight (typically meta-relevant or tournament-winning teams).
- Quality gates before a team enters the pool:
	- Row must have full EV spreads listed ("EVs: yes") and exactly 6 Pokémon
	- Every team is validated against Pokémon Showdown's official legality checker (TeamValidator running validate-teams-batch.js) — illegal builds are dropped
	- Normalization fixes paste artifacts: nicknames stripped, As One/Urshifu/Ogerpon forms disambiguated, event-mon IV locks enforced, Level 50 forced, comments removed
