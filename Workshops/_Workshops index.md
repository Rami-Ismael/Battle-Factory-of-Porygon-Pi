---
created_at: 2026-08-14
updated_at: 2026-08-27
type: index
tags:
  - workshop
  - index
  - moc
---

Each venue has its own note with website, candidate papers, and a relevance verdict. Ranked most → least relevant to *"search the VGC team space for the best counter-team."*

**Organised by conference (2026-08-18):** each conference is a top-level heading and a folder; the workshops and competitions it hosts sit under it. Where a conference has no note of its own (NeurIPS, RLC) the heading is the folder only.

## NeurIPS
*No conference note — NeurIPS itself is a publication venue, not a reading target; the workshops are what matter.*

| Relevance   | Workshop                                          | Why (one line)                                                                                                        |
| ----------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 🟢 High     | [[PokeAgent Challenge (NeurIPS 2025)]]            | The domain itself — competitive Pokémon AI, 20M battle trajectories, baselines                                        |
| 🟡 Med-High | [[EvoRobust (NeurIPS 2026)]]                      | Quality-Diversity = my "creative / off-meta diverse teams" question, formalized                                       |
| 🔴 Low      | [[DiffCoALG (NeurIPS 2025)]]                      | *Retired 2026-08-18* — program checked, all neural CO / routing / MIS; nothing on subset selection against a meta     |
| 🔴 Low      | [[DynaFront (NeurIPS 2026)]]                      | *Retired 2026-08-18* — checked all 46 papers of the 2025 edition, none touch discrete team search                     |
| 🔴 Low      | [[MLxOR 2026 (NeurIPS 2026)]]                     | *Downgraded from Med 2026-08-18* — 245 papers of 2025 edition scanned, none on subset selection; 2 adjacent reads kept |
| 🔴 Low      | [[OPT 2026 — Optimization for ML (NeurIPS 2026)]] | Continuous NN-training optimizers — wrong branch (I need discrete)                                                    |
| 🔴 Low      | [[RL4XS (NeurIPS 2026)]]                          | *Retired 2026-08-18* — inaugural edition, no papers yet; physical-lab RL, no sim-to-real gap for me                    |

## RLC (Reinforcement Learning Conference)
*No conference note.*

| Relevance | Workshop | Why (one line) |
|---|---|---|
| 🟡 Med | [[Finding the Frame (RLC 2024)]] | Problem-*framing* — the set-cover→max-coverage reframe, principled |
| 🟢 High | [[RLVG (RLC 2026)]] | RL × video games — the RLC workshop this project's shape fits; its 2026 papers yield FootsiesGym (HIGH) + Generals.io self-play (MED) |

## IEEE CoG (Conference on Games)
| Relevance | Note | Why (one line) |
|---|---|---|
| 🟡 Med-High | [[IEEE CoG (Conference on Games)]] — *the conference* | The venue that hosts it — valuable as a *publication target*, not as reading |
| 🟢 High | ↳ [[VGC AI Competition (IEEE CoG)]] — *competition* | **The closest thing to my project that already exists** — Championship track = team selection, scored, with an open-source engine |
| 🟡 Med-High | ↳ [[Strategy Card Game AI Competition (IEEE CoG)]] — *competition, ended 2022* | Agents build a deck (pick ≤30 of 120 cards, ≤2 copies) then play it; scored by mirrored round-robin win rate on fixed seeds; IEEE CEC 2019 + CoG 2019–2022 |
| 🟡 Med | ↳ [[Hearthstone AI Competition (IEEE CoG)]] — *competition, ended 2020* | User-Created-Deck track: submit agent + own 30-card deck vs an unknown field; round-robin, ≥100 games per matchup, avg win rate; IEEE CIG 2018 → CoG 2020 |

## FDG (Foundations of Digital Games)
| Relevance | Note | Why (one line) |
|---|---|---|
| 🟠 Low-Med | [[FDG 2026 (Foundations of Digital Games)]] — *the conference* | *Swept 2026-08-24* — "all 124 papers screened; none touches team/deck search against a meta; two method-adjacent MEDs" |

## AAMAS (Autonomous Agents and Multiagent Systems)
| Relevance | Note | Why (one line) |
|---|---|---|
| 🟡 Med-High | [[AAMAS (Autonomous Agents and Multiagent Systems)]] | It published [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] (AAMAS 2026) — a venue with a paper in my exact domain; **only live deadline found** (abstract 2026-10-01) |

## SoCS (Symposium on Combinatorial Search)
| Relevance | Note | Why (one line) |
|---|---|---|
| 🟡 Med-High | [[SoCS (Symposium on Combinatorial Search)]] | The combinatorial-search field itself; 2 of 2026's papers are ML-guided MILP / **binary quadratic** solving |

## CP (Principles and Practice of Constraint Programming)
| Relevance | Note | Why (one line) |
|---|---|---|
| 🟠 Low-Med | [[CP (Constraint Programming conference)]] | Modelling vocabulary only — *retired 2026-08-18 from Med-High: the clauses stopped being search constraints ([[Decision — enumerate species subsets, don't optimise]] *(note deleted 2026-08-18, no replacement)*)* |

## CPAIOR (CP-AI-OR integration)
| Relevance | Note | Why (one line) |
|---|---|---|
| 🔴 Low | [[CPAIOR (CP-AI-OR integration)]] | **Retired as a venue 2026-08-18** — 2025 + 2026 accepted lists (69 papers) scanned, none highly relevant; the note survives only as home of the 4-paper decision-focused-learning reading list (Wilder AAAI 2019, Demirović CPAIOR 2019, Mandi JAIR 2024, SPO) |

## EvoStar (EvoApplications)
| Relevance | Note | Why (one line) |
|---|---|---|
| 🟠 Low-Med | [[EvoApplications 2026 (EvoStar)]] | *Swept 2026-08-24* — "Every paper is paywalled, so nothing can clear the read-the-PDF bar — no HIGH. Best is Seals & Tauritz on hall-of-fame deceptive fitness (MED)." |

## ICLR (International Conference on Learning Representations)
*No conference note.*

| Relevance | Note | Why (one line) |
|---|---|---|
| 🟠 Low-Med | [[ReALM-GEN (ICLR 2026)]] | *Swept 2026-08-27* — "Two of 84 papers touch a black-box reward on discrete outputs": Clean-Sample Markov Chain (MED), Trust-Region Noise Search (LOW); both assume the reward is free |

## AIIDE (AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment)
*The conference itself stays screened 🔴 Low (list at bottom); the folder exists for its workshop.*

| Relevance | Note | Why (one line) |
|---|---|---|
| 🔴 Low | [[EXAG 2026 (Experimental AI in Games)]] | *Swept 2026-08-24* — "Swept ~180 papers 2014–2024: experimental/creative games AI — procedural content generation, narrative, co-creativity. Two survivors, both MED; nothing HIGH." |

*The non-NeurIPS conferences above were added 2026-08-14 from an online search for venues beyond the ML workshops. These are **conferences, not workshops** — bigger, older, and in several cases the field of origin for the techniques I'm using. The VGC AI Competition row is the single most important find in this vault so far.*

### Screened and *not* given a note
Checked in the same search, judged not worth their own note — recorded so I don't re-search them:
- **AIIDE 2026** (2026-11-09/13, Belo Horizonte, Brazil — [site](https://sites.google.com/view/aiide2026/)) — AAAI's game-AI conference, but weighted toward procedural content generation, narrative and entertainment AI. CoG covers the same ground closer to my problem. **LOW.**
- **GECCO 2026** (2026-07-13/17, San José, Costa Rica — [site](https://gecco-2026.sigevo.org/HomePage)) — evolutionary computation. I expected a quality-diversity workshop and there **isn't one in 2026**; the closest are *Decomposition Techniques in Evolutionary Optimization* and *Quantum Optimization* (the latter is QUBO-adjacent, since QUBO is the annealer-native form). Quality-diversity is better served by [[EvoRobust (NeurIPS 2026)]]. **LOW-MED.**
- **LION 20** (2026-06-15/19, Milan — [site](https://lion20.org/)) — Learning and Intelligent OptimizatioN; on-theme (ML ∩ optimisation ∩ hard problems) but small, and CPAIOR covers the same intersection with more rigour. **LOW-MED.**

Added 2026-08-19, from a search for pick-a-subset-to-beat-a-field competitions:
- **Tales of Tribute AI Competition** (IEEE CoG 2024–) — deck-*builder* genre: the deck is acquired during play, not picked before the match, so not my shape. **LOW.**
- **MOBA drafting (Dota 2 / LoL)** — no AI competition exists anywhere (searched 2026-08-19); only papers (DraftRec, GAE) and hobby tools. Nothing to add.
- **Fantasy-sports lineups (DFS)** — no academic AI competition; FanDuel/DraftKings contests are real-money human contests that papers attack with ML + MILP. Nothing to add.

## Verdict on "check if each workshop is relevant"
- **Actually pursue:** **[[VGC AI Competition (IEEE CoG)]]** (my exact problem, already benchmarked — the new #1, it supplies the evaluation harness the others don't), PokeAgent (battling domain). ~~DiffCoALG (method)~~ — retired 2026-08-18.
- **Worth a skim for one idea each:** SoCS (the two ML+MILP/QUBO papers), the DFL reading list parked in the CPAIOR note (venue itself retired 2026-08-18), EvoRobust (quality-diversity for diverse teams), MLxOR (one paper: bandits with ML surrogate rewards — the matchup predictor as proxy for $f$), Finding the Frame (how to state the objective).
- **Reference, not reading:** CP (vocabulary for my clauses and symmetry breaking), AAMAS (for [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]], and as a submission target).
- **Deprioritize:** DynaFront, OPT 2026 and RL4XS (branch/domain mismatch — do not spend time here for team search), AIIDE / GECCO / LION (see screened list above).

## Notes

- 2026-09-06: [[SPIGM 2026 — paper-by-paper relevance screening]] — all 210 workshop-site papers plus PI-LNO from ICML screened. Added 18 readings (4 HIGH, 14 MED, one reward-deferred); 7 held for evidence gaps and 186 skipped.
- 2026-09-06: [[EIML 2026 — paper-by-paper relevance screening]] — ICML 2026, second Epistemic Intelligence workshop. All 79 accepted entries recorded: 59 skipped on current project fit, 20 insufficiently verified. No papers added to the reading queue; exact evidence and access limitations recorded per paper. **Reassessment later that day:** two MED readings added (Space-sampled Value Decay; Distributional Energy-Based Models), leaving 57 skips and 20 unverified. The initial threshold conflated reading value with immediate implementation fit; the reassessment supersedes the initial zero-addition result.
- 2026 workshops are **upcoming** as of 2026-08-14: dedicated program/papers often not public yet. Deadlines pulled from the hub's [aiworkshoptracker](https://aiworkshoptracker.com/conference/neurips/) snapshot — verify on each official site before relying on them.
- **Almost every 2026 conference deadline in the table above has already passed** (SoCS March, CoG March, CPAIOR/CP/GECCO/LION all held by July). For submitting, the targets are the **2027** editions — except AAMAS 2027, which is open until 2026-10-08.
- The NeurIPS 2026 list (102 workshops, [announced 2026-08-10](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/)) was scanned via that announcement post rather than read line by line — **treat the NeurIPS coverage here as good but not proven exhaustive.**
- Backlink: [[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]]

- 2026-09-06 follow-up: [[SPIGM 2026 — full-paper reassessment]] — full reads of all 18 previously accepted papers; revised recommendations: 6 MED, 9 deferred, 3 not recommended. Supersedes the initial accepted-paper grades; original screening preserved.

- 2026-09-06 owner correction: SPIGM shortlist reduced to **4 MED, 9 deferred, 5 not recommended**. Training is not a current problem; exact duplicate-team removal already exists. #181 and #173 no longer recommended; see [[SPIGM 2026 — full-paper reassessment]].

- 2026-09-06 metric-relevance correction: SPIGM now **3 MED, 9 deferred, 6 not recommended**. #183 Hacking Generative Perplexity withdrawn: its text-metric counterexample adds too little beyond the existing full-team/composition diversity plan. See [[SPIGM 2026 — full-paper reassessment]].
