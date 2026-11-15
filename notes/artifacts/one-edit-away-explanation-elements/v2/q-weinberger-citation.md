Verdict: the paper is Weinberger, E. (1990). Correlated and uncorrelated fitness landscapes and how to tell the difference. Biological Cybernetics 63(5), 325–336, https://doi.org/10.1007/BF00202749. He tested his method on Kauffman's N-k model.
His measure is the autocorrelation of scores along a random walk. Its value at one step is the correlation between a point and a random neighbour one edit away, so Rami's sentence is accurate as written and needs no correction.

## The "on w" clause

> ..., on Kauffman's N-k model of fitness landscapes from evolutionary biology.

The full history sentence then reads:

> Comparing neighbours one edit apart is an old way to measure how rough a search space is. Weinberger proposed it in 1990 on Kauffman's N-k model of fitness landscapes from evolutionary biology: check how well one point's score predicts the score of a point one edit away.

The abstract writes "N-k model". Most later papers write "NK model". Either spelling is fine.

## Full citation, every field checked

| Field | Value | Source opened |
|---|---|---|
| Author | E. Weinberger, Department of Biochemistry and Biophysics, University of Pennsylvania. His address at publication was the Max-Planck-Institut für biophysikalische Chemie, Göttingen | Springer article page, https://link.springer.com/article/10.1007/BF00202749 |
| Title | Correlated and uncorrelated fitness landscapes and how to tell the difference | Crossref record, https://api.crossref.org/works/10.1007/BF00202749 |
| Journal | Biological Cybernetics | Crossref |
| Volume and issue | 63, issue 5 | Crossref |
| Pages | 325–336 | Crossref and Springer |
| Date | September 1990. Received 30 September 1989, accepted 7 March 1990 | Crossref and Springer |
| DOI | 10.1007/BF00202749 | Crossref and Springer |
| Access | Paywalled. Springer shows a preview and sells the PDF | Springer article page |

## What Weinberger measured, and on what

The full text is paywalled. I read the abstract on the Springer page. My fetch tool returns quotes of at most about 125 characters, so the abstract below is quoted in pieces, not as one passage.

| Claim | Exact quote | Where |
|---|---|---|
| The setting is fitness landscapes from evolutionary biology | "The properties of multi-peaked “fitness landscapes” have attracted attention in a wide variety of fields, including" ... "evolutionary biology." | Abstract, sentence 1 |
| The measure is a random-walk autocorrelation | "A central role in this discussion is played by the autocorrelation of fitnesses obtained from a random walk on the landscape." | Abstract |
| A class of landscapes where the autocorrelation decays exponentially | "a condition (satisfied by a wide class of landscapes we call AR(1) landscapes) under which the average autocorrelation approximates a decaying exponential." | Abstract, sentence joined from two returned pieces |
| The landscapes he tested it on | "We illustrate some aspects of our method with computer experiments based on a single family of landscapes (Kauffman's" ... "“N-k model”), that is shown to be a generic AR(1) landscape." | Abstract, sentence joined from two returned pieces |
| What it is for | "the “tuning” of combinatorial optimization algorithms, and in modelling in the experimental sciences." | Abstract, end of a sentence |

A later paper that is free to read and states the method in its own words is NAS-Bench-101, Ying et al. 2019, https://arxiv.org/abs/1902.09635, Section 3.3 "Locality".

| Claim | Exact quote |
|---|---|
| What "one edit" means there | "We define “closeness” in terms of edit-distance: the smallest number of changes required to turn one architecture into another; one change entails flipping the operation at a vertex or the presence/absence of an edge." |
| The method, credited to Weinberger | "A popular measure of locality is the random-walk autocorrelation (RWA), defined as the autocorrelation of the accuracies of points visited as we perform a long walk of random changes through the space (Weinberger, 1990; Stadler, 1996)." |
| Its bibliography entry | "Weinberger, E. Correlated and uncorrelated fitness landscapes and how to tell the difference. Biological cybernetics, 63(5):325–336, 1990." |

### Random-walk autocorrelation compared with one-step neighbour correlation

Weinberger's measure is a function of the lag s. It is the correlation between the score at step t of a random walk and the score at step t+s. Each step of the walk moves to a random neighbour, which is one edit. At s = 1 the two points are a point and a randomly chosen neighbour one edit away. So the value at one step is the neighbour correlation: how well one point's score predicts the score of a point one edit away. Weinberger used the whole curve. On AR(1) landscapes the curve decays as ρ(1) to the power s, so the one-step value fixes the rest of the curve and gives a correlation length.

Rami's sentence describes the one-step value, which is the core of the method, so it is accurate. Three small caveats:

- "Proposed" follows the usual credit, which NAS-Bench-101 also gives. I could not read the full text to check whether Weinberger built on earlier correlation measures, for example by Kauffman.
- The idea that one step of a walk on the N-k model changes one gene comes from the standard N-k setup. I could not confirm that wording in Weinberger's paywalled text.
- "Predicts" means linear correlation. The method measures correlation and does not fit a prediction model.

## Where the project's notes cite Weinberger

| File | What it says | Correct? |
|---|---|---|
| `notes/Teaching/reference/bo-benchmarks-vs-vgc-full-team-search-unit.md`, section "Smoothness — added 2026-09-28", bullet "Method: neighbour correlation". This is the earliest mention. The file entered git on 2026-10-08 in commit `12e816f0`, and the section is dated 2026-09-28 | "How strongly a point's score predicts the score of a point one edit away (Weinberger 1990). NAS-Bench-101 §3.3 measures exactly this for its own space (random-walk autocorrelation, where one edit flips one operation or one edge)" | Yes. The author and year are right, and NAS-Bench-101 Section 3.3 does credit Weinberger 1990. Its only gap is the missing full reference, now given above |
| `SITE/12-field-history.json`, `SITE/03-borrow-sibling-method.json`, `SITE/v2/03-borrow-sibling-method.json`, `SITE/14-reasons-justifications.json` | All take the citation from the reference note above | Yes, for the same reason |

`code/docs` has no mention of Weinberger.

## What I could not verify

- The paper's body text, including the exact definition of a walk step on the N-k model and any earlier work it cites. It is paywalled.
- The Weinberger 1988 working paper at IIASA that may come before the journal article. https://pure.iiasa.ac.at/6466 returned an access-denied page.
- Stadler 1996, "Landscapes and their correlation functions", https://doi.org/10.1007/BF01165154. I read only its preprint abstract, https://www.tbi.univie.ac.at/papers/Abstracts/95-07-067abs.html, which mentions "the correlation of a time series sampled along a random walk". I found no free full text.
