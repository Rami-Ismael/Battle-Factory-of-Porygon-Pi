# DIMES citation corpus: method and coverage

Retrieved 2026-09-08, approximately 23:45 UTC. Citation counts are a dated snapshot, not a quality score.

## Seed identity

The [NeurIPS 2022 proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/a3a7387e49f4de290c23beea2dfcdc75-Abstract-Conference.html) identify DIMES by Ruizhong Qiu, Zhiqing Sun, and Yiming Yang, DOI `10.52202/068431-1851`. Its [arXiv version](https://arxiv.org/abs/2210.04123) is `2210.04123`.

Semantic Scholar resolved `ARXIV:2210.04123` to `570a4a186f348ebad33816afc82c16acabd07122`, reporting 175 citations. The DOI lookup returned not found. Several first/retry requests returned HTTP 429; those responses remain saved rather than concealed. The successful arXiv lookup is `semantic-scholar-seed-retry.json`. A subsequent exact-title `/paper/search/match` query independently returned the same canonical ID, arXiv ID, DOI `10.48550/arXiv.2210.04123`, and 175 citations (`semantic-scholar-title-match.json`); no alternate matching seed was returned.

## Complete indexed direct-citation retrieval

Called [Semantic Scholar citations API](https://api.semanticscholar.org/graph/v1/paper/570a4a186f348ebad33816afc82c16acabd07122/citations?fields=title,year,externalIds,citationCount,url,abstract&limit=1000&offset=0). `semantic-scholar-citations-page-0.json` preserves the complete response. It returned offset 0, 175 records, and no next-page pointer: all 175 citations reported by this seed record were fetched. All 175 had distinct paper IDs. The flattened copy is `semantic-scholar-all-citing.json`.

Applied `citingPaper.citationCount >= 2` to the citing paper's own count, yielding **111 Semantic Scholar records** (`semantic-scholar-qualified.json`). Stable audit IDs S001–S111 sort descending by count, preserving response order for ties. Exactly 64 records have counts 0 or 1 and were excluded (`semantic-scholar-excluded.json`). This is an index-edge screen: citation presence was established by the index, not by manually inspecting every bibliography. Missing or erroneous index edges remain possible. The API supplied abstracts for 97 of 111 qualifying records, as well as external identifiers and author/OA metadata.

## Independent index cross-check

[OpenAlex exact-title search](https://api.openalex.org/works?search=DIMES%3A%20A%20Differentiable%20Meta%20Solver%20for%20Combinatorial%20Optimization%20Problems&per-page=10) found two exact seed-title records:

- `W4304701342`: arXiv DOI, 16 citing works.
- `W7133224898`: conference DOI, 14 citing works.

Fetched `filter=cites:SEED&per-page=200&cursor=*` for both; each returned all its reported records in one page with no continuation. Their union is 30 distinct OpenAlex IDs, 11 of which have `cited_by_count >= 2`. Raw responses and the complete/filtered sets are retained as `openalex-*` JSON files.

Nine qualifying OpenAlex works match Semantic Scholar qualifying titles/DOIs. Two extra qualifying works are retained in `openalex-supplement-qualified.json`:

- O001, **GLOP: Learning Global Partition and Local Construction for Solving Large-Scale Routing Problems in Real-Time**: 38 OpenAlex citations; DOI `10.1609/aaai.v38i18.30009`.
- O002, **DOGE-Train: Discrete Optimization on GPU with End-to-End Training**: 3 OpenAlex citations; DOI `10.1609/aaai.v38i18.30048`.

Do not mix these counts with Semantic Scholar counts without labeling the source, or sum them. Their presence demonstrates why a single index should not be described as exhaustive worldwide coverage.

## Version duplicates

The 111 Semantic Scholar IDs are bibliographic records, not necessarily 111 distinct works. Preserve all rows for audit, but avoid duplicate reading assignments:

- S001 and S033 share arXiv `2401.02051` (one record encodes it in its DOI): confirmed same work/version family.
- S035 and S063 share arXiv `2405.01906` (one record encodes it in its DOI): confirmed same work/version family.
- S096 and S101 share arXiv `2502.12188` (one record encodes it in its DOI): confirmed same work/version family.
- S103 is confirmed as the earlier title of S096/S101 by [arXiv v2](https://arxiv.org/abs/2502.12188v2): exact title plus all five matching authors; current v4 uses the S096 title. The S103 record itself lacks an external ID.
- S080 differs from S022 only in the final word Learning/Learner; S080 lacks authors, year, and external ID. [The original OpenReview submission](https://openreview.net/notes/edits/attachment?id=FPegqx6woH&name=pdf) uses the exact Learning title and anonymous authors; [author-published code](https://github.com/naver/goal-co/) connects the project to the final Learner title. Treat this as a strongly supported version duplicate for reading purposes, while retaining the malformed raw row.

Thus there are **113 qualifying source records** including the two supplements, **109 after four confirmed version merges**, or **108 reading entries when the probable GOAL title-version merge is also accepted**. Keep those distinctions explicit; never sum duplicate citation counts.

## Claim boundary

This is complete pagination of the retrieved Semantic Scholar seed record, plus complete pagination of both exact-title OpenAlex seed records, as retrieved on the date above. It is **not proof of all citing papers in existence**, nor a claim that all versions/citations are correctly merged by either index. Literature added after retrieval, missing references, citation-index errors, and papers indexed under unidentified versions can change the result. Re-run the saved API URLs to refresh.

## CADO threshold check

`semantic-scholar-excluded.json` contains arXiv `2602.08210`, **CADO: From Imitation to Cost Minimization for Heatmap-based Solvers in Combinatorial Optimization**, paper ID `80f8325a0cdd6fcfe9ecb3df99cc450efa783808`, with **0 citations**. Two malformed CADO alternate-title records also have 0. It fails the requested threshold in this snapshot; method relevance does not override eligibility.

The machine-readable mapping is `canonical-inventory.json`. Its `mergeStatus` preserves GOAL as probable rather than treating a malformed identifier-free record as conclusively resolved.
