---
created_at: 2026-08-14
updated_at: 2026-09-01
url: https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/home
framework: https://gitlab.com/DracoStriker/pokemon-vgc-engine
research_docs: https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/Documentation/Research
page_2025: https://cog2025.inesc-id.pt/vgc-ai-competition/
page_2026: https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/VGC-AI-Competition/2026-Edition
listing_2026: https://cog2026.org/competitions
discord: https://discord.gg/GwKHqXpdjf
organizer: Simão Reis (simao.reis@vortex-colab.com)
venue: IEEE CoG — see [[IEEE CoG (Conference on Games)]]
edition: 4th Edition (listed for CoG 2026)
research_page: https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/Documentation/Research
team_size: 3 Pokémon selected from a generated roster (per the ToG 2023 paper) — not 6
tracks: Battle; Championship; Balance (Meta + Rule, since 2025). "Teambuild" in the CoG 2025 Logistics text = Championship (resolved 2026-08-18)
status: 4th Edition (2026) started 2026-02-09, submissions closed 2026-07-15 per wiki 2026-Edition; 2025 wiki deadline was 2025-07-13
type: competition
relevance: high
tags:
  - pokemon
  - competitive-battling
  - teambuilding
  - competition
  - metagame
  - benchmark
  - relevance/high
---
# ToDo 

- [x] only keep the section from backlink only sibiling — done: Backlink section now holds only the sibling link
- [x] find releavnt paper from this their wiki https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/Documentation/Research — done 2026-08-18: all 9 tabulated under "Every paper on their Research wiki"; #1 (ToG 2023) is the one that matters most
- [ ] Go through paper
	- [ ] Deep Team Synthesis: Opponent-Conditioned Latent Space Mapping for the Pokémon VGC AI Competition
	- [ ] Architecting Meta-Game Diversity: Novel Objective Formalizations for the VGC AI Benchmark
## Every paper on their Research wiki

All 9 entries on [Documentation/Research](https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/Documentation/Research), fetched 2026-08-18. Verdicts are quoted from the numbered list below, not re-derived.

| #   | Paper                                                                                             | Venue         | Free copy                                                                                                                            | Relevance                                         |
| --- | ------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| 1   | An Adversarial Approach for Automated Pokémon Team Building and Meta-Game Balance                 | IEEE ToG 2023 | 🟢 [wiki PDF](https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/uploads/d96a4fdacf01666d071072db3ccb9232/FINAL_VERSION.PDF) | HIGHEST                                           |
| 2   | Deep Team Synthesis: Opponent-Conditioned Latent Space Mapping for the Pokémon VGC AI Competition | IEEE CoG 2026 | TBR                                                                                                                                  | HIGH                                              |
| 3   | Architecting Meta-Game Diversity: Novel Objective Formalizations for the VGC AI Benchmark         | IEEE CoG 2026 | TBR                                                                                                                                  | HIGH                                              |
| 4   | A New Rules Balance Track for the Pokémon VGC AI Competition 2.0                                  | IEEE CoG 2025 | 🟢 [wiki PDF](https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/uploads/f8c31a49136ddc46803e707c0adcfbc8/paper_119.pdf)     | MED-HIGH                                          |
| 5   | VGC AI Competition — A New Model of Meta-Game Balance AI Competition                              | IEEE CoG 2021 | 🟢 [wiki PDF](https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/uploads/2b17bd8650d944df9eec773c3af8a5b9/paper_6.pdf)       | MED-HIGH                                          |
| 6   | Bilevel Entropy based Mechanism Design for Balancing Meta in Video Games                          | AAMAS 2023    | 🟢 [TAMU PDF](https://people.engr.tamu.edu/guni/pistar/Papers/AAMAS23-meta.pdf)                                                      | LOW                                               |
| 7   | Competitive Deep Reinforcement Learning over a Pokémon Battling Simulator                         | ICARSC 2020   | 🔒 [DOI](https://doi.org/10.1109/ICARSC49921.2020.9096092)                                                                           | MED                                               |
| 8   | A Self-Play Policy Optimization Approach to Battling Pokémon                                      | IEEE CIG 2019 | 🔒 [DOI](https://doi.org/10.1109/CIG.2019.8848014) (free copy cited in [[PokeAgent Challenge (NeurIPS 2025)]])                       | MED                                               |
| 9   | Showdown AI Competition                                                                           | IEEE CIG 2017 | 🟢 [CIG PDF](http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf)                                                         | ❌ LOW — owner verdict 2026-08-26: bad paper, skip |

Rows 1–5 are the organisers' own section; 6–9 are its "Other Publications" list. #10 below (HCI-Games 2024) is **not** on the wiki.

### Not on the wiki — papers that cite the organisers' work

The wiki is only their own output. Semantic Scholar (queried 2026-08-18) lists 14 works citing #5 and 8 citing #1; these are the ones about *team composition* rather than balancing. Sorted by relevance; graded from abstracts (TMLR one from full text) 2026-08-18.

| Paper | Venue | Copy | Relevance | Why |
|---|---|---|---|---|
| VGC-Bench: Towards Mastering Diverse Team Strategies in Competitive Pokémon | AAMAS 2026 | 🟢 [arXiv 2506.10326](https://arxiv.org/abs/2506.10326) | HIGH | graded in [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] |
| Pokémon GO Team Optimization: A Comparative Study of Classic Metaheuristic Algorithms | J. Interactive Systems 2026 | 🟢 [PDF](https://journals-sol.sbc.org.br/index.php/jis/article/download/6773/3893) | MED-HIGH | *"generate a three-Pokémon counter-team that maximizes performance against a given rival team under simulated battles"* — my problem's shape, but one rival team not a meta, k=3, and metaheuristics I don't need (593,775 subsets enumerate) |
| Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis | TMLR 2024 | 🟢 [arXiv 2408.17180](https://arxiv.org/abs/2408.17180) | MED | see paragraph below |
| A Framework for Predicting the Impact of Game Balance Changes Through Meta Discovery | IEEE ToG 2024 | 🟢 [arXiv 2409.07340](https://arxiv.org/abs/2409.07340) | MED | RL discovers the dominant teams on Pokémon Showdown after a balance patch — team-space search on my stack, but the designer's question |
| An Empirical Analysis of the Validity of Competitive Pokémon Rule Sets | EXAG @ AIIDE 2024 | not found | LOW | rule-set validity, not team building |

*Retired 2026-09-01 — the Pokémon GO row's "metaheuristics I don't need (593,775 subsets enumerate)": the enumeration decision was deleted 2026-08-18, the search unit is a full team, and the elite-refit loop already running is one. The k=3 / one-rival-team objections stand.*

**TMLR 2024, graded MED.** Their win-value model is *Bradley-Terry strength + a learned "counter table"* on the residual (Siamese net, compositions vector-quantised into a small codebook, deterministic VQ so counter[cluster(A), cluster(B)] is a lookup). That is a competing model class for the matchup predictor: BT alone is transitive, the counter table is the intransitive part — the same job my pairwise terms do ([[Decision — the matchup predictor is ridge with pairwise terms]]). Worth citing as the comparison baseline. Everything else — Top-D Diversity, Top-B Balance, AoE2 / Hearthstone / Brawl Stars / LoL — is the designer's balance question, not mine. No Pokémon anywhere in it.

### From the competition's own group (Reis et al.)

0. **Deep Team Synthesis: Opponent-Conditioned Latent Space Mapping for the Pokémon VGC AI Competition** — Simão Reis, Fernando Alves. **IEEE CoG 2026.** TBR — wiki lists title + authors only (checked 2026-08-18); no abstract, no PDF. **HIGH, provisional** — the title says *team generation conditioned on the opponent*, i.e. my counter-team problem on their framework.
1. **Architecting Meta-Game Diversity: Novel Objective Formalizations for the VGC AI Benchmark** — Simão Reis, Rita Novais, A. Lucas Martins, Fernando Alves. **IEEE CoG 2026.** TBR — title + authors only on the wiki, nothing on arXiv or the CoG site (checked 2026-08-18). **HIGH** — "novel objective formalizations" is my open question about how to *write* the objective; "diversity" is my off-meta/creative-teams question.
2. **A New Rules Balance Track for the Pokémon VGC AI Competition 2.0** — Simão Reis, A. Lucas Martins, Rita Novais, Fernando Alves. **IEEE CoG 2025.** 🟢 [Free accepted manuscript](https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/uploads/f8c31a49136ddc46803e707c0adcfbc8/paper_119.pdf) · 🔒 [DOI](https://doi.org/10.1109/CoG64752.2025.11114412). **MED-HIGH** — describes "Competition 2.0" and a new Rules Balance track; the current authority on what the tracks actually are.
3. **VGC AI Competition — A New Model of Meta-Game Balance AI Competition** — Simão Reis, Luís Paulo Reis, Nuno Lau. **IEEE CoG 2021.** 🟢 [ieee-cog.org PDF](https://ieee-cog.org/2021/assets/papers/paper_6.pdf) · 🟢 [wiki copy](https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/uploads/2b17bd8650d944df9eec773c3af8a5b9/paper_6.pdf) · 🔒 [DOI](https://doi.org/10.1109/CoG52621.2021.9618985). **MED-HIGH** — the original definition of the competition. Superseded on track structure by #4.

### Adjacent, listed on the same page

6. **Bilevel Entropy based Mechanism Design for Balancing Meta in Video Games** — **AAMAS 2023.** 🟢 [PDF](https://people.engr.tamu.edu/guni/pistar/Papers/AAMAS23-meta.pdf) · [ACM](https://dl.acm.org/doi/10.5555/3545946.3598887). **LOW (downgraded 2026-08-17)** — bilevel optimisation for meta-balance. It was graded on bilevel being the formal shape of "I optimise, then the opponent re-optimises," a framing the project dropped; what remains is *game balancing* from the designer's seat, the opposite of my problem. Guni Sharon also has a paper in the SoCS 2026 list.
7. **Competitive Deep Reinforcement Learning over a Pokémon Battling Simulator** — Simões, Reis, Lau, Reis. **ICARSC 2020.** 🔒 [DOI](https://doi.org/10.1109/ICARSC49921.2020.9096092). **MED** — battling, not team building.
8. **Showdown AI Competition** — **CIG 2017.** 🟢 [PDF](http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf) · 🔒 [DOI](https://doi.org/10.1109/CIG.2017.8080435). ~~**LOW-MED** — the older Showdown-based competition; historical context.~~ *Owner verdict 2026-08-26: bad paper — skip.* Its only content is random / one-turn-lookahead / minimax baseline battle policies and their win rates — common knowledge, nothing to extract.
9. **Enhancing Pokémon VGC Player Performance: Deep RL and Neuroevolution** — Rodriguez, Villanueva, Baldeón (PUCP). **HCI-Games 2024 @ HCII 2024.** 🔒 [DOI](https://doi.org/10.1007/978-3-031-60692-2_19), no free version. **MED** — same-*domain*, not same-*framework*; not listed on the group's own research page.

- [ ] Read #1 (ToG 2023) before writing my own max-coverage formulation — it may already answer several hub open questions, or show they were the wrong questions
- [ ] Watch for #2 and #3 to be released after CoG 2026 (2026-09-01/04)

[[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] used to sit here; it's an AAMAS 2026 paper and lives in [[AAMAS (Autonomous Agents and Multiagent Systems)]].

## My own notes

1. Framework research docs: <https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/Documentation/Research>

## Backlink

sibling: [[PokeAgent Challenge (NeurIPS 2025)]]
