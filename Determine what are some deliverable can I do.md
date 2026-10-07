---
created_at: 2026-08-17
updated_at: 2026-08-22
tags:
  - deliverables
  - portfolio
  - dissemination
---

# Todo

- [x] Use the goal skills to Collect all the relevant pokemon github repo that apply Ai on Pokemon see their readme to get insipiration what should a good readme has, store all the relevant github repo under her as checkbox I also want to rank the repo bases on the quality of the readme
- [ ] The goal is to  build a Hugging Face Docker Space where I play a VGC battle in the browser against a `poke-env` `RandomPlayer`. The space will not interaction it the actualy website slogon. This is possible because we can clone the smogon/pokemon-showdown and run locally. Serve as normal UI for people have played pokemon know hot it work.
- [ ] I wished they go in detail how we are deploying a trained policy and a desire pokemon team to pokemon showdown to see how well it perform on the ladder "Ladder validation report ($N$ games, pipeline team vs. baseline, confidence intervals)"
	- [ ] [[Create a public repository for this project with a runnable README]]
- [ ] Grab all the link special link on https://github.com/caymansimpson/reuniclusVGC to add them on this details
- [ ] Replace this section "Ladder validation report ($N$ games, pipeline team vs. baseline, confidence intervals)" with a new section that discuss how to build your vgc agent to be a ladder 
- [ ] Replace this section "Workshop paper → arXiv preprint regardless of acceptance" with literature review
- [ ] Determine what abalation experiment I cand do 
- [ ] go through ranked deliberable create a list of modification
	- [ ] Determine
		- [ ] Recorded talk (15–20 min) or poster — UTD club, PyData, local meetup
			- [ ] Create a youtube video for this
			- [ ] Determine a local meetup for this
- [ ] Creaet an Rl model that can beat a different band of il model 60 percet 
- [ ] **PufferLib throughput A/B post.** Battles/sec, vectorizer-only vs full loop, with the honest finding you already have in the findings note (the wrapper alone buys you none of the PPO improvements). The PufferLib community will amplify a real benchmark, and your #4 finding — no public trace of a Showdown integration anywhere in the repo — is worth publishing on its own.
# Prompt
- I am intersted in display my work to employer in the future and my peers. I want to create a certain amount of develiverable for this personal project this is what i have so far. I want to create more what are the thing I can do 


1. I can tweet about this project
2. Show a final tweet about this project 
3. Show off my pokemon in regional that 
4. Have my pokemon be in the top selection in pokemon showdown 
5. A workshop paper 
6. Blogs
	1. Daily
	2. Weekly
	3. Monthly
	4. Final Blog

## Ranked deliverables (Claude, 2026-08-17)

Blunt read of the list above: tweets are *distribution*, not deliverables; one regional or one "top selection" is a coin flip, not evidence; daily blogs will die and leave a visible graveyard. The paper is the highest-value item already on the list. Value is rated for the two audiences named above — employers and peers — plus academics where the workshop paper is concerned.

| #   | Deliverable                                                                                                                         | Audience             | Effort       | Value       | Depends on                    | Notes                                                                 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ----------- | ----------------------------- | --------------------------------------------------------------------- |
| 1   | [[Create a public repository for this project with a runnable README]] (`pip install`, one command → top-$k$ teams for Reg Set M-B) | Employers, peers     | Med          | ★★★★★       | Working pipeline              | The thing a hiring manager actually opens                             |
| 2   | Interactive demo (Streamlit / Gradio / HF Space): meta-team list → top-$k$ teams + win-rate matrix                                  | Everyone             | Med          | ★★★★★       | #1                            | Most shareable artifact; what the tweet links to                      |
| 3   | Ladder validation report ($N$ games, pipeline team vs. baseline, confidence intervals)                                              | Employers, academics | Med          | ★★★★★       | Pipeline + Showdown account   | The one number quoted everywhere                                      |
| 4   | Workshop paper → arXiv preprint regardless of acceptance                                                                            | Academics, employers | High         | ★★★★★       | #3, #12                       | Pick the top venue in [[_Workshops index]] whose deadline you can hit |
| 5   | Final blog post                                                                                                                     | Everyone             | Med          | ★★★★☆       | #1–#4                         | Narrative wrapper around the results                                  |
| 6   | Released dataset with DOI (HF / Zenodo): Monte Carlo evaluation battles + fitted $Q_{ij}$                                           | Academics, peers     | Low          | ★★★★☆       | Monte Carlo evaluation stage  | Citable contribution independent of the search result                 |
| 7   | Ablation table ($f$ with/without pairwise terms; team-list-weighted vs. uniform meta)                                               | Academics, employers | Low          | ★★★★☆       | $f$ trained                   | Cheap; turns "built a thing" into "understand the thing"              |
| 8   | Pipeline diagram + $Q_{ij}$ heat-map                                                                                                | Everyone             | Low          | ★★★★☆       | $f$ trained                   | One visual reused in paper, README, blog, talk                        |
| 9   | Recorded talk (15–20 min) or poster — UTD club, PyData, local meetup                                                                | Peers, employers     | Med          | ★★★☆☆       | #8                            | The recording is the deliverable                                      |
| 10  | Decisions / lessons-learned write-up (what was retired and why)                                                                     | Peers, employers     | Low          | ★★★☆☆       | Existing `Decision — …` notes | Already half-written in this vault                                    |
| 11  | Portfolio one-pager (site or PDF) + resume bullet + LinkedIn post                                                                   | Employers            | Low          | ★★★☆☆       | #1–#4                         | Packaging of everything above                                         |
| 12  | Public matchup-predictor benchmark (held-out accuracy vs. baselines)                                                                | Academics, peers     | Med          | ★★★☆☆       | #6                            | Lets others compare against $f$                                       |
| 13  | pip-installable package / CLI (`vgc-search --reg M-B --top 20`)                                                                     | Peers, employers     | Low          | ★★★☆☆       | #1                            | Signals you finish work                                               |
| 14  | Weekly (or when-something-happened) dev logs                                                                                        | Peers                | Low, ongoing | ★★☆☆☆       | —                             | Cut daily; abandoned logs look worse than none                        |
| 15  | Team result at a regional                                                                                                           | Everyone             | High         | ★★☆☆☆       | #3                            | Great story, weak evidence — one tournament is a coin flip            |
| 16  | Team in top Showdown selection                                                                                                      | Peers                | Low          | ★★☆☆☆       | #3                            | Noisy; Elo/GXE over $N$ games (#3) is the stronger claim              |
| 17  | Tweets during / final tweet                                                                                                         | Everyone             | Low          | ★☆☆☆☆ alone | #2                            | Distribution, not a deliverable — always link to something            |
| 18  | Extended abstract / student poster (CoG, AAAI/NeurIPS workshop)                                                                     | Academics            | Med          | fallback    | #4 slips                      | Backup if the full workshop paper misses its deadline                 |

**If you only do five:** #1 → #2 → #3 → #4 → #5. Everything else is either input to those or packaging of them.

Related: the hub's 2026-08-15 "What is the deliverable?" item settled the *ladder* (tweet → blog → model on HuggingFace → paper); this table is the wider menu around that ladder. The HuggingFace artifact there is $f$ itself, which is #6/#12 here.

**Owner's decision (2026-08-17):** daily blog stays, and each day's tweet is based on that day's post. The blog is a working log first — it tracks what happened and what was done — and a public artifact second, so the "abandoned logs look worse than none" objection to #14 is withdrawn: the log is written for the owner regardless of readership, and it becomes the raw timeline for the final blog (#5) and the paper (#4). Row #17 still holds — the tweet links to the day's post.

**2026-08-22 — second pass (in chat):** rows 1–3, 6, 7, 12 all wait on a search pipeline that doesn't exist yet. Shippable now, all battle-policy side: VGC-Bench reproduction report, cross-play/Alpha-Rank arena, upstream PRs (action masking, Showdown seeds), PufferLib throughput A/B.