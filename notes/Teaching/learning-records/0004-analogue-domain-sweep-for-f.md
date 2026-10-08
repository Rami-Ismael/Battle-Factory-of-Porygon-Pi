---
created_at: 2026-08-19
status: active
---

# Analogue-domain sweep for $f$ — what exists and what doesn't

The 2026-08-18 todo ("find papers that apply weighted maximum coverage or a learned pairwise objective over a k-subset…") is answered in `RESOURCES.md` § "analogue-domain papers graded for $f$": 26 verified papers, 6 HIGH / 16 MED / 4 LOW, every link fetched and venue confirmed.

Load-bearing findings:

- **$f$'s exact conjunction is unpublished.** No paper combines weighted coverage of an opponent-team distribution with k-subset selection, and none fits a pairwise objective from battle results and then searches the subset (BOCS / FMQA run that loop in materials design). Teaching consequence: there is no single primary source — the two halves get taught separately, Williamson–Shmoys for coverage (already the spine) and Haugh & Singal for the opponent-distribution + quadratic-subset half. Project consequence: the formulation itself is a defensible novelty claim.
- **Haugh & Singal (Management Science 2021) is the next lesson's primary source.** Dirichlet-multinomial over opponents' lineups + binary quadratic programs over the roster is structurally the meta + $Q_{ij}$ pipeline.
- The MOBA drafting papers (The Art of Drafting, JueWuDraft) grade HIGH on the learn-then-search shape, but the 2026-08-18 sweep's caveat stands and is quoted in the resource entries: sequential pick/ban ≠ VGC's simultaneous blind 6-pick.
- One qualifying paper was excluded per the owner's banned-topics note (Synergy Graphs), despite fitting the rubric.
- **A parallel session answered the same todo the same day** (its full report is in its own chat; its pointer sits on the todo line). Its two finds this sweep missed are folded into `RESOURCES.md` after re-verification: Skowron & Faliszewski (JAIR 2017) — approval-based Chamberlin–Courant ≡ weighted MaxCover, so **multiwinner voting is the mathematical home of formulation (a)** and a candidate lesson topic (meta teams as voters, the team as the committee); and Neoh et al. (arXiv 2605.09588) — coverage estimated from binary/pairwise feedback, then greedy *(retired 2026-08-19 — the owner rejected this paper as not relevant)*. Both sweeps independently reached the same negative verdict, which strengthens it.
