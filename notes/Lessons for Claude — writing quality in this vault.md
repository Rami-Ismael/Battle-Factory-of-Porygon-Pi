---
created_at: 2026-08-14
updated_at: 2026-10-21
type: guidance
audience: claude
status: living
source: mistakes made in this vault on 2026-08-14
tags:
  - writing
  - note-quality
  - feedback
  - sourcing
---

Every rule below comes from something that actually went wrong in this vault on 2026-08-14, with
the receipt attached. Generic writing advice would itself be the slop this vault bans.

Ordered by severity: things that made notes **wrong**, then things that made them **empty**, then
things that made them **inconsistent**.

---

## Accuracy

### 1. Never quote from a search-result summary. Fetch the page.

**What happened.** The verdict "IEEE CoG is my #1 venue" rested on a quoted line: *"the AI Pokémon
Trainer must choose the best team against the current game meta."* That sentence does not exist.
The real page says *"...to choose the best team against the current game like Video Game
Championships."* The quote came from a search engine's paraphrase, presented in the note as
verbatim source text.

**The rule.** A search result is a pointer, never a source. If it goes inside quotation marks, it
was fetched from the page it's attributed to — no exceptions, no matter how confident the summary
sounded.

### 2. The word your verdict rests on must appear in the source.

**What happened.** "Meta" was the load-bearing word — the entire relevance argument was "this
competition is about beating the *meta*, which is my project." The source never used it. The idea
survived; the evidence for it didn't.

**The rule.** Before writing a verdict, find the single word or phrase it depends on and confirm
that word is in the source. If it's yours, say so in the note: *"that gloss is mine."*

### 3. Verify counts before asserting them.

**What happened.** The note claimed the competition had three tracks. The page names two, and
mentions a third term only in a logistics aside. The "three" came from stitching two sources
together without checking either.

**The rule.** Numbers — how many tracks, papers, editions, authors — are checkable in seconds and
are the easiest thing to be caught wrong on. Check, or write "unclear" and say why.

### 4. When you fix a load-bearing claim, fix every copy of it.

**What happened.** The bad quote had already been copied into `_Workshops index` and into the hub
in two places. Fixing the source note left three wrong copies behind; only a grep caught them.

**The rule.** Before writing a striking claim into a note, know that it will get copied into the
index and the hub. When it changes, grep the vault for the distinctive phrase and fix every
instance in the same pass.

---

## Substance — sentences that carry nothing

### 5. Never open by restating the title.

**What happened.** The note read:

> `# IEEE CoG — Conference on Games`
> The IEEE Conference on Games — an academic venue for AI applied to games people actually play.

The heading says it, then the first sentence says it again. This is the single most common tic in
machine-written notes: the opening line is a warm-up lap, not information.

**The rule.** The first sentence must contain something the title and frontmatter do not. If it
can't, delete it and start with the first real claim.

### 6. Don't say the same fact twice in different words.

**What happened.** In the same four lines: "a venue for AI applied to games people **actually
play**" and then the CFP "does not accept game-theory papers unless applied to games people
**would play**." That is one fact wearing two costumes.

**The rule.** After drafting a paragraph, list its distinct facts. If the count is lower than the
sentence count, cut until they match.

### 7. Cut "this is relevant to you" when it changes no decision.

**What happened.** *"That's a useful filter for me: I'm not doing abstract game theory, I'm doing
VGC."* It reads as a relevance signal, but nothing follows from it — the owner was never going to
submit an abstract game-theory paper, so the "filter" filtered nothing. It's flattery dressed as
analysis.

**The rule.** For every "this matters because…" ask *what would I do differently if it were
false?* No answer → delete the sentence.

### 8. State rules positively.

**What happened.** *"does not accept game-theory papers unless they are applied to games people
would play"* — a negative plus an exception, which the reader has to unwind to recover a simple
fact. The positive form is one clause: **"CoG only wants work applied to real games."**

**The rule.** Prefer "X only accepts Y" to "X does not accept Z unless Y." Reserve negation for
when the prohibition itself is the point.

### 9. Write "none" instead of padding a section.

**What happened (this one went right — codifying it).** After splitting the competition out, the
venue note's "Papers that could interest me (≤3)" section had no papers left. The tempting move
was to list any three CoG papers to fill the template. Instead: *"None from the venue at large"*
plus one line explaining why.

**The rule.** A template heading is not a quota. An honest empty section is information — it says
"I looked and there wasn't anything." Padding destroys that signal and costs the reader time.

The cap itself was removed on 2026-08-14 — the owner judged "≤3" to be a limiting early idea. The
section is now just "Papers that could interest me": list everything relevant, no floor, no
ceiling. The anti-padding rule stands; it was never about the number.

---

## Provenance

### 12. Where you file a thing is a claim about it. Look up the venue; never inherit it from the note you happen to be writing.

**What happened.** **[[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]]** was listed under "Papers that could interest me (≤3)" inside a
note about the VGC AI Competition at IEEE CoG. In this vault that section means *papers from this
venue*. [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] is an **AAMAS 2026** paper — nothing to do with IEEE or the competition. I had
even written "not part of this competition; independent" one line below, but **placement beat the
disclaimer**: anyone scanning the note reads it as an IEEE/competition paper, and the owner did.

Why it happened, step by step — the chain matters more than the error:
1. I found it in a search *about the competition*, so it surfaced only because it shares the token
   "VGC". Co-occurrence in a result list is not a relationship.
2. I recorded arXiv and OpenReview links and treated those as provenance. **They are hosting
   locations, not venues** — an arXiv ID says nothing about peer review or where a paper appeared.
3. I filed it where I was already typing. The open file decided the filing.
4. The "≤3" template created pressure to produce a third item.

**The rule.** A paper's venue is a fact to be looked up (arXiv's comments/journal-ref field, or
the OpenReview venue field), never inferred from surroundings. If a paper is worth citing but
comes from elsewhere, either move it to the note for its real venue or set it visibly apart from
the venue's own list — never as an unnumbered peer in that list. And: **a disclaimer does not undo
a placement.** If you need a sentence to explain why something is where it is, it is in the wrong
place.

### 13. Cite what the reader can actually open, and label the access.

**What happened.** I cited a paper as a bare [Springer](https://link.springer.com/chapter/10.1007/978-3-031-60692-2_19)
link. The owner has **no institutional access** — so the citation was, for him, a paywall with a
title on it. I had treated "I found a DOI" as "I gave him the paper." They are not the same thing.

**The trap in the obvious fix.** The instinct is "cite the IEEE version instead." **IEEE Xplore is
also paywalled** — swapping one publisher DOI for another buys nothing. What actually works is
knowing where free copies live. In this case the conference's *own site* hosts every paper:
`ieee-cog.org/2021/assets/papers/paper_6.pdf`. Free, official, permanent — and I had already seen
this exact pattern in the owner's PokeAgent note (an `ieee-cog.org/2019/papers/…` link) without
drawing the general rule from it.

**The rule — check for a free copy in this order, and stop at the first hit:**
1. **arXiv** — free by construction.
2. **The conference's own website** — CoG, NeurIPS, ICML, AAAI (via `ojs.aaai.org`), OpenReview and
   most workshops post free PDFs even when a publisher also sells the paper.
3. **Author or lab page** — academics post their own PDFs.
4. **Institutional repository** — university CRIS/repository systems.
5. **Publisher DOI** — last resort, and then **label it 🔒**.

Never present a paywalled link as *the* link. Mark every citation 🟢 (openable) or 🔒 (paywalled),
and when only 🔒 exists say so plainly, so the reader knows the cost before clicking rather than
after. **Springer, IEEE Xplore, ScienceDirect and ACM DL links are paywalled by default** — assume
it unless the page says open access.

### 14. Grading a recurring venue means checking the edition that already happened.

**What happened.** I graded AAMAS **MEDIUM** on its 2027 call for papers and wrote "no paper list
to give" — while **AAMAS 2026** had been held in May 2026 in Paphos, with a full program including
the single most domain-relevant paper in this whole scan. I looked only at the edition with an
open deadline, because I was thinking about *submitting* rather than *reading*. AAMAS is now
MEDIUM-HIGH, and the upgrade came from the user catching an unrelated mistake, not from my search.

**The rule.** For any recurring venue, check the **most recent completed** edition's accepted
papers before assigning relevance. Upcoming editions tell you about deadlines; past editions tell
you what the venue actually publishes — which is what relevance means.

## Consistency

### 10. One verdict, stated the same way everywhere.

**What happened.** The CoG note said the venue "ranks alongside PokeAgent." The index said it was
"the new #1." Both written the same day, by me, about the same thing.

**The rule.** A relevance grade lives in one place and is quoted, not re-derived, everywhere else.
When a grade changes, the note, the index and the hub change together.

### 11. Apply a convention fully or not at all.

**What happened.** The owner's cleanup instruction said workshop notes should have no "What it is"
section. Every note kept one, and then a single note dropped it — leaving the vault inconsistent
in a *new* way rather than fixing it.

**The rule.** Half-applied conventions are worse than either alternative, because they destroy the
reader's ability to trust the pattern. Either sweep all the notes or leave them all alone, and
surface the decision instead of quietly picking one file.

---

## Shape

### 15. Prefer the work-list shape of [[PokeAgent Challenge (NeurIPS 2025)]].

Owner's preferred form (2026-08-18): verdict + metadata in frontmatter, no opener; body = lettered stages (Reading → Methods → Baselines → Synthesis) of checkboxes; each bullet carries a number and ends in a decision; method summary in the heading; unclear passages quoted verbatim as tasks.

## The one-line test

Before a paragraph ships, ask of each sentence: **what does the reader now know that they didn't
know from the title, the frontmatter, and the previous sentence?** If the answer is nothing,
delete the sentence. The paragraph that triggered this note failed that test four times in a row.

## Backlink

Hub: [[Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team]] ·
the note that prompted it: [[IEEE CoG (Conference on Games)]] ·
the sourcing failure in detail: [[VGC AI Competition (IEEE CoG)]]
