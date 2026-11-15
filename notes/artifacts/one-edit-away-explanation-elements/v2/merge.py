"""Merge the 2026-11-13 "Draft Version two" pass into the site's element files.

One subagent handled each checkbox of the owner's checklist and wrote its result into this folder.
This script copies those results over the site's element files, attaches the owner's own note to each
element word for word, and writes the files the page needs for the handed-back questions and the
versions for other sites. Run from anywhere:  python3 merge.py
"""
import json
import re
from pathlib import Path

here = Path(__file__).parent
site = here.parent
load = lambda path: json.loads(Path(path).read_text())
save = lambda path, value: Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")

# The owner's notes under each checkbox, exactly as he typed them. None means he listed the element with no note.
OWNER_NOTES = {
    "01-contrast-reader-default": ["Define what is the reader default"],
    "02-looks-like-but-differs": None,
    "03-borrow-sibling-method": None,
    "04-everyday-picture": None,
    "05-neighbouring-field-term": ["Epstatis is a good on"],
    "06-design-decisions": [
        "The starting point of the pokemon is [[How did we create the top 50 placement pokemon team we consider top meta threat to atleast counter and show my method is effective]] a random pokemon team is mostly perform terrible.  We can see how senetive the strong pokemon. What does it take to make a good competitive pokemon team",
        "What can we say about nosie"],
    "07-historical-reasons": ["etc"],
    "09-implications": ["etc"],
    "10-specific-examples": ["etc"],
    "08-technical-constraints": ["One team has 24 move slots, and a Pokemon in this format can learn about 61 moves on average, so a single team has roughly 1,400 one-move neighbours."],
    "11-bigger-picture": [],
    "12-field-history": ['The origin of the method comes from "Comparing neighbours one edit apart is an old way to measure how rough a search space is. Weinberger proposed it in 1990: check how well one point\'s score predicts the score of a point one edit away."'],
    "14-reasons-justifications": ["González-Duque et al., NeurIPS 2024, [A survey and benchmark of high-dimensional Bayesian optimization of discrete sequences](https://arxiv.org/abs/2406.04739), Section 4.2.1"],
    "15-counter-examples": [],
}
WEINBERGER = ("Weinberger, E. 1990. Correlated and uncorrelated fitness landscapes and how to tell the difference. "
              "Biological Cybernetics 63(5), 325-336. https://doi.org/10.1007/BF00202749")
WEINBERGER_ANSWER = ("Answered 2026-11-13: the full citation is " + WEINBERGER + ". The abstract says his experiments used \"a single family of landscapes "
                     "(Kauffman's 'N-k model')\", so the sentence can end \"on Kauffman's N-k model of fitness landscapes from evolutionary biology\". "
                     "His measure is \"the autocorrelation of fitnesses obtained from a random walk on the landscape\", and its value at one step is the "
                     "correlation between a point and a neighbour one edit away, so your sentence is accurate as written. The full text is behind a paywall, "
                     "so this rests on the abstract and on NAS-Bench-101, Section 3.3. Report: v2/q-weinberger-citation.md")


def answers(element):
    """An element's direct answers to the owner's note, as a list."""
    answer = element.pop("owner_answer", None)
    return [] if answer is None else answer if isinstance(answer, list) else [answer]


def weinberger(element):
    """Replace the 'citation is being checked' gap with the citation subagent's answer."""
    element["gaps"] = [gap for gap in element["gaps"] if "Weinberger" not in gap or "citation" not in gap] + [WEINBERGER_ANSWER]
    for candidate in element["candidates"]:
        if any("Weinberger" in fact["claim"] + fact["source"] for fact in candidate["facts"]) or "Weinberger" in candidate["sentence"]:
            candidate["facts"].append({"claim": "Full citation, each field checked against Crossref and the publisher's page. He tested the measure on Kauffman's N-k model.", "source": WEINBERGER})


for path in sorted(here.glob("[0-9][0-9]-*.json")):
    if path.stem.startswith("19-"):
        continue
    element = load(path)
    element["owner_answers"] = answers(element)
    if path.stem in ("08-technical-constraints", "12-field-history", "14-reasons-justifications"):      # these three count candidates from zero; the page counts from one
        element["applied"] = re.sub(r"candidate (\d)", lambda found: f"candidate {int(found.group(1)) + 1}", element["applied"])
    if path.stem in OWNER_NOTES:
        element["owner_note"] = OWNER_NOTES[path.stem]
    if path.stem in ("03-borrow-sibling-method", "12-field-history"):
        weinberger(element)
    save(site / path.name, element)

# Design decisions: one subagent took the starting point, another took noise. Both halves go on one page.
design = load(site / "06-design-decisions.json")
start, noise = load(here / "06a-design-starting-point.json"), load(here / "06b-design-noise.json")
design.update(candidates=start["candidates"] + noise["candidates"], gaps=start["gaps"] + noise["gaps"], picked=None,
              owner_answers=answers(start) + answers(noise), applied=start["applied"] + " " + noise["applied"],
              owner_note=OWNER_NOTES["06-design-decisions"])
save(site / "06-design-decisions.json", design)

# His paragraph on what the project taught him becomes its own element under "Admit opinion and perspective".
stake = load(here / "19-mathematics-statement.json")
idea = stake["source_for_the_idea"]
checks = " ".join(f'{number}. "{claim["sentence"]}" {claim["check"]}' for number, claim in enumerate(stake["claims"], 1))
save(site / "17-what-the-project-taught-me.json", {
    "action": "Admit opinion and perspective",
    "element": "The author's own view",
    "what_it_does": "It tells the reader, in the author's own voice, why the work was worth doing and what it taught him.",
    "when_to_use": stake["where_it_fits"],
    "sentence_pattern": "As I understand it, x. This project ...",
    "dllm_example": None,
    "anchor_in_draft": "It interesting problem it require to solve the problem with different method.",
    "extra": {"label": "Your paragraph checked against the project, sentence by sentence", "text": checks},
    "candidates": stake["candidates"],
    "gaps": [item["question"] for item in stake["questions"]],
    "picked": None,
    "owner_note": [stake["owner_text"]],
    "owner_answers": [{"label": "A mathematician who said it", "text": f'{idea["verdict"]} {idea["author"]}, {idea["work"]}, {idea["link"]}. '
                       + " ".join(f'"{quote["text"]}" {quote["section"]}.' for quote in idea["quotes"][:2])}],
    "applied": "Your paragraph is kept as you wrote it. Each sentence was checked against what the project built and ran. One claim does not hold as written: reinforcement learning was studied and planned, never trained.",
})

# The four questions he handed back, each with its verdict and the sources that carry it.
save(site / "answers.json", [
    {"question": "Do you have a source saying that a benchmark's smoothness decides which optimisation methods look good on it?",
     "verdict": "The post can say: how rugged a test problem is can change which search method looks best on it. No source supports \"smoothness decides the ranking\".",
     "sources": ["Li, Arnold and others 2024, Results and Figure 2c, https://doi.org/10.1101/2024.10.24.619774: on 16 protein landscapes the gain of model-guided search over greedy hill climbing grows with ruggedness. A preprint, and the order never flips.",
                 "Santoni, Doerr and others 2024, Section 6, https://arxiv.org/abs/2303.00890: on the BBOB functions one Bayesian optimisation method wins on single-basin functions and another on functions with many basins.",
                 "Pelikan 2008, Section 4.5, https://arxiv.org/abs/0801.3111: the ranking reverses as the interaction count k grows. Its hierarchical Bayesian optimization algorithm is an estimation-of-distribution algorithm, so do not let it read as Bayesian optimisation."],
     "report": "v2/q-smoothness-decides-methods.md"},
    {"question": "What is the full Weinberger citation, and what landscape did he first measure neighbour correlation on?",
     "verdict": "Kauffman's N-k model. Your sentence is accurate as written, and can end \"on Kauffman's N-k model of fitness landscapes from evolutionary biology\".",
     "sources": [WEINBERGER + ". Full text is behind a paywall, so the answer rests on the abstract.",
                 "Ying and others 2019, NAS-Bench-101, Section 3.3, credits Weinberger 1990 for the random-walk autocorrelation and is free to read."],
     "report": "v2/q-weinberger-citation.md"},
    {"question": "Why are academic optimisation benchmarks mostly smooth?",
     "verdict": "Say only that the benchmarks you measured are smooth, and name them. Papers give a reason why benchmarks are simple and synthetic, cost and isolating one difficulty. None says designers pick smooth functions.",
     "sources": ["Your own table of ten tasks: five stay smooth and five are rough, at least near good points. Only LABS-50 and DRD2 turn rough near good points. notes/Teaching/reference/bo-benchmark-derived-vgc-research-comparison.md",
                 "Kononova and others 2025, Section 1, https://arxiv.org/abs/2511.12264: academic suites are \"deliberately constructed to isolate algorithmic phenomena\".",
                 "Stanton and others 2024, Section 2.2, https://arxiv.org/abs/2407.00236: closed-form test functions \"are so easy to solve that they are mostly used to catch major bugs\".",
                 "Hansen and others, COCO, Section 2, https://arxiv.org/abs/1603.08785: the suite models ruggedness on purpose."],
     "report": "v2/q-why-benchmarks-smooth.md"},
    {"question": "Does the effect of one move depend on the other five Pokemon?",
     "verdict": "Keep it as a hypothesis, and do not cite the interaction result. That result swapped whole Pokemon in retired battles. No saved data scores the same move edit on the same set on two teams under the current setup.",
     "sources": ["The rerun holds one near match, picked after looking: Sylveon's Detect replaced by Giga Impact moved one team up 5.8 points and another down 5.1, about 2.7 times its noise. The two sets' Stat Points differ.",
                 "A measurement that would settle it: 48 sets that recur on two teams with different teammates, one identical move edit each, 192 scores, 73,728 battles, about 29 minutes. 49 of 168 sets in the top 50 recur on two or more teams.",
                 "Szendro and others 2013, Section 1, https://arxiv.org/abs/1202.4378, names the idea: \"sign epistasis occurs when mutations are beneficial in some genetic backgrounds, but not in others.\""],
     "report": "v2/q-move-depends-on-teammates.md"},
])

# Versions for other sites. Two corrections to the subagent's drafts, then the character counts are taken again.
social = load(here / "social.json")
for version in social["versions"]:
    version["posts"] = [post.replace("Third post on why Pokemon team building is a hard optimisation problem.",
                                     "Third post in my build-in-public series. This one covers one reason Pokemon team building is a hard optimisation problem.")
                        for post in version["posts"]]
    version["characters"] = [len(post) for post in version["posts"]]
    text = " ".join(version["posts"])
    if "reinforcement learning" in text:
        version["cautions"].append("The closing names reinforcement learning. The project studied it and never trained it, and every battle uses a behaviour-cloning policy. Keep the word only as something you studied.")
    if "either count" in text:
        version["cautions"].append("\"Bigger on either count\" is clear for tournament-ready teams. For search-found teams it holds only narrowly on the stricter count.")
    assert not re.search(r"[—;()]|baseline|meta teams|threshold|5-point", text), version["id"]
save(site / "social.json", social)
# Element 16 had no checkbox in this pass. Three of its lines still leaned on retired battles or an unmeasured claim.
scope = load(site / "16-one-why-question.json")
scope["candidates"][0]["sentence"] = scope["candidates"][0]["sentence"].replace(
    "why can a single legal edit to a Pokemon team break hill climbing, when a single step on the benchmarks optimisation papers use mostly does not?",
    "how far does one legal edit move a Pokemon team's win rate, and is that further than one step moves a standard optimisation benchmark?")
scope["candidates"][1]["facts"][0] = {
    "claim": "The rerun gave each of 40 starting teams 18 legal edits of six kinds and replayed each unchanged team three times, with every score taken over 384 battles against the 50 top-placing teams under one behaviour-cloning policy.",
    "source": "code/docs/ruggedness2.md, last sections"}
scope["gaps"] = [gap for gap in scope["gaps"] if not gap.startswith(("The sources measure only two benchmarks", "Should the post use the 8-team pilot", "Answered 2026-11-13"))] + [
    "Answered 2026-11-13: say that the benchmarks you measured are smooth and name them. No source says benchmarks are mostly smooth. See the four questions you handed back.",
    "Answered 2026-11-13: use neither the 8-team pilot nor the 400,000-battle study. Both ran in listed slot order and no longer count. Use the rerun of 2026-11-13."]
save(site / "16-one-why-question.json", scope)

result = load(site / "rerun-result.json")
result["checked"] = ("Run matched its plan, 880 scores of 384 battles. Read without a size cutoff: a whole-Pokemon swap and a move edit are above zero from both groups of starting teams; "
                     "item and ability edits are above zero from tournament-ready teams only; stat alignment and Stat Points edits are not shown above zero. "
                     "As registered before the run, with a size cutoff that was dropped on 2026-11-13, only the whole-Pokemon swap passed both tests, so the reading here was chosen after the result. "
                     "Against one Pest Control step, taken as a share of the spread between benchmark starting points of its own kind, only the whole-Pokemon swap from tournament-ready teams is larger. "
                     "The 24 tournament-ready starting teams are a fresh draw, not September's.")
save(site / "rerun-result.json", result)


# ---------------------------------------------------------------- the owner's decision of 2026-11-13 on the benchmark spread
# The subagents wrote while the choice was open. He then chose each kind of benchmark starting point on its own spread,
# so the step is 0.37 and 0.32, the pooled 0.14 is not used, and only the whole-Pokemon swap from tournament-ready teams reads larger.
DECIDED = ("Answered 2026-11-13: you chose each kind of benchmark starting point on its own spread. One Pest Control step is then 0.37 from random points and 0.32 from "
           "hill-climbed points. Only the whole-Pokemon swap from tournament-ready teams reads larger than a step. Move, item and ability edits read undecided.")
NOTE = " Decided 2026-11-13: the two kinds of starting point are not pooled, so the step is 0.37 and 0.32 and the pooled 0.14 is not used."
EXACT = [
    ("changing one station moves its score by 0.14 to 0.37 of the spread between its points depending on how that spread is counted. Swapping one Pokemon on a tournament-ready team moved win rate by 0.65 of the spread between teams, larger than the benchmark step on either count.",
     "changing one station moves its score by 0.32 to 0.37 of the spread between its starting points. Swapping one Pokemon on a tournament-ready team moved win rate by 0.65 of the spread between teams, larger than the benchmark step."),
    ("One Pest Control step moves the score by 0.14 to 0.37 of that benchmark's spread between points, depending on how the spread is counted.",
     "One Pest Control step moves the score by 0.32 to 0.37 of the spread between that benchmark's starting points."),
    ("There a step moves the score by 0.14 of the spread between its points, or about a third when each kind of starting point is measured on its own.",
     "There a step moves the score by about a third of the spread between its starting points, 0.37 from random points and 0.32 from hill-climbed ones."),
    ("meets much larger steps when its step is a Pokemon swap", "meets steps about twice as large when its step is a Pokemon swap"),
    ("Against one step on the Pest Control benchmark, the whole-Pokemon swap is larger on either way of counting. For move, item and ability edits, the comparison depends on which spread the benchmark step is divided by, and that choice is still open.",
     "Against one step on the Pest Control benchmark, only the whole-Pokemon swap on a tournament-ready team is shown larger. Move, item and ability edits cannot be told apart from that step."),
    ("That step is 0.14 of the benchmark's spread pooled over random and hill-climbed starting points, and 0.37 and 0.32 when each kind of starting point uses its own spread. The whole-Pokemon swap is larger on either count. For move, item and ability edits, the comparison depends on which spread you divide by, and I have not settled that.",
     "That step is 0.37 of the spread between the benchmark's random starting points and 0.32 between its hill-climbed ones. I keep the two kinds apart because their average scores differ. The whole-Pokemon swap on tournament-ready teams is larger than that step. Move, item and ability edits are undecided against it."),
    ("one step is 0.14 to 0.37 of its own spread, depending on how that spread is counted. One Pokemon swap is a bigger step than that on either count.",
     "one step is 0.32 to 0.37 of the spread between its own starting points. One Pokemon swap is a bigger step than that."),
    ("is 0.14 to 0.37 of its own spread, depending on how you count it.", "is 0.32 to 0.37 of the spread between its own starting points."),
    ("So one Pokemon swap is a bigger step than the benchmark's on either count.", "So one Pokemon swap is a bigger step than the benchmark's."),
    ("is 0.14 to 0.37 of its own spread depending on how it is counted. The swap is bigger on either count.", "is 0.32 to 0.37 of the spread between its own starting points. The swap is bigger."),
    (", which is 0.14 to 0.37 depending on how its spread is counted.", ", which is 0.32 to 0.37 of the spread between its own starting points."),
    ("Do not extend this comparison to move, item or ability edits while the open decision on the benchmark spread stands.", "Move, item and ability edits read undecided against the step."),
    ("For move, item and ability edits that comparison depends on which spread the benchmark step is divided by, so no candidate says they are larger or smaller.",
     "Move, item and ability edits read undecided against the step, so no candidate says they are larger or smaller."),
    ("and the open decision on which spread to use is still yours.", "and you chose each kind's own spread on 2026-11-13."),
    ("Whether a move edit is larger than one Pest Control step depends on the open choice of spread in FACTS.md, so this section should not say it is or is not.",
     "A move edit reads undecided against one Pest Control step, so this section should not say it is larger or smaller."),
    ("and it is tied to the open decision in FACTS.md", "and you chose each kind's own spread on 2026-11-13, which gives 0.93 and 0.95"),
    ("0.14 to 0.37", "0.32 to 0.37"),
]
used = set()


def decide(text, where):
    for old, new in EXACT:
        if old in text: text = text.replace(old, new); used.add(old)
    if where == "gap" and re.search(r"^(Open: )?[Ww]hich spread|open decision|is still your open decision", text): return DECIDED
    if where in ("fact", "extra") and re.search(r"0\.14\b|0\.138|[Oo]pen (choice|decision)", text) and "Decided 2026-11-13" not in text:
        text = re.sub(r" For move, item and ability edits the comparison (with the benchmark step )?depends on (the|an|that) open choice( of spread)?(, so this candidate says nothing about them)?\.?", " Move, item and ability edits read undecided against it.", text)
        text = re.sub(r" Which spread to use is an open decision\.?", "", text)
        text = re.sub(r"Open decision, raised 2026-11-13 and not yet answered: which spread the benchmark step is divided by\. ", "", text).rstrip()
        text = text + ("" if text.endswith(".") else ".") + NOTE
    if where == "fact": text = re.sub(r" on either count( of the benchmark's spread)?", "", text)
    if where == "source": text = re.sub(r"[Oo]pen decision(,? raised 2026-11-13)?", "Decision on the benchmark spread", text)
    return text


for path in sorted(site.glob("[0-9][0-9]-*.json")):
    element = load(path)
    for candidate in element["candidates"]:
        candidate["sentence"] = decide(candidate["sentence"], "sentence")
        for fact in candidate["facts"]: fact["claim"], fact["source"] = decide(fact["claim"], "fact"), decide(fact["source"], "source")
    gaps = [decide(gap, "gap") for gap in element["gaps"]]
    element["gaps"] = [gap for number, gap in enumerate(gaps) if gap not in gaps[:number]]
    if element.get("extra"): element["extra"]["text"] = decide(element["extra"]["text"], "extra")
    for answer in element.get("owner_answers", []): answer["text"] = decide(answer["text"], "sentence")
    save(path, element)
social = load(site / "social.json")
social["claim"] = decide(social["claim"], "sentence")
for version in social["versions"]:
    version["hooks"] = [decide(hook, "sentence") for hook in version["hooks"]]
    version["posts"] = [decide(post, "sentence") for post in version["posts"]]
    for shot in version.get("shots", []): shot.update({key: decide(value, "sentence") for key, value in shot.items() if isinstance(value, str)})
    version["characters"] = [len(post) for post in version["posts"]]
    version["cautions"] = [c for c in version["cautions"] if "frame 7 or frame 8" not in c and "either count" not in c]
    if version["id"] in ("linkedin", "x", "bluesky", "reddit-ml", "short-video"):
        version["cautions"].append("Frames 7 and 8 of the storyboard were corrected on 2026-11-13 and can be attached. The benchmark comparison is shown for tournament-ready teams. From search-found teams the whole-Pokemon swap misses it narrowly.")
save(site / "social.json", social)
unused = [old[:50] for old, _ in EXACT if old not in used]
print("decision applied; replacements that matched nothing:", unused)


# ---------------------------------------------------------------- the top-50 study finished on 2026-11-14 (issue #317)
TOP50 = [
    ("Editing the 49 distinct top-50 teams themselves is queued as issue #317 and has no result.",
     "Editing the 49 distinct top-50 teams themselves ran on 2026-11-13 and 2026-11-14, issue #317. A whole-Pokemon swap moved a top-50 team by 4.91 points, a move edit by 3.04 and an item edit by 3.12, all above zero."),
    ("Your note says the starting point is the top 50, but the finished rerun started from 24 tournament-ready teams outside the top 50 and 16 search-found teams, and the top 50 are its opponents. Editing the top 50 themselves is issue #317, not yet run. Should the post describe the starting point as tournament-ready teams now and add the top 50 when #317 finishes?",
     "Your note says the starting point is the top 50. The study that edits the top 50 themselves has now run, issue #317: whole Pokemon 4.91 points, move 3.04, item 3.12, all above zero. The earlier rerun started from tournament-ready and search-found teams. Which starting point should the post lead with?"),
    ("The top-50 study, issue #317, keeps the same seed design and has not started. Should it give each team its own seeds before it runs, so luck is not shared between teams?",
     "Answered 2026-11-14: the top-50 study ran with the same seed design. A check found no luck shared between an edit and the replay on its seed. It did find that measuring every edit against one first score shifts all of a team's edits together, so the next study should compare against several unchanged scores."),
    ("No study under the current rules has edited the top 50 teams yet. That is the next study: the same six kinds of edit on the 49 distinct top-50 teams, 413,952 battles, planned and not started.",
     "So we ran it again under the current rules: the same six kinds of edit on the 49 distinct top-50 teams, 413,952 battles, finished on 2026-11-14."),
    ("The top-50 study, GitHub issue #317, has not started and has no result.", "The top-50 study, GitHub issue #317, ran on 2026-11-13 and 2026-11-14 with 413,952 battles, as planned."),
    ("The next study, queued with no result yet, applies the same six edit kinds to the 49 distinct top-50 teams, 18 edits each, for 413,952 battles, planned at about 2.7 hours.",
     "The top-50 study applied the same six edit kinds to the 49 distinct top-50 teams, 18 edits each, for 413,952 battles. It was planned at about 2.7 hours and took 2.1."),
    ("Open: when the top-50 study in issue #317 finishes, should candidate 1 add whether stat alignment and Stat Points stay below the line on the top-placing teams themselves? No result exists yet.",
     "Answered 2026-11-14: on the top-50 teams themselves stat alignment and Stat Points both read 0.00 points, with intervals up to 1.81 and 1.01, and both read smaller than a Pest Control step. Candidate 1 can add that."),
    ("Queued, no result yet", "Result of the top-50 study"),
]
hit = set()
for path in sorted(site.glob("[0-9][0-9]-*.json")):
    text = path.read_text()
    for old, new in TOP50:
        old_json, new_json = json.dumps(old, ensure_ascii=False)[1:-1], json.dumps(new, ensure_ascii=False)[1:-1]
        if old_json in text: text = text.replace(old_json, new_json); hit.add(old)
    path.write_text(text)
print("top-50 result applied; replacements that matched nothing:", [old[:50] for old, _ in TOP50 if old not in hit])
print("merged", len(list(site.glob("[0-9][0-9]-*.json"))), "elements,", len(social["versions"]), "versions")
