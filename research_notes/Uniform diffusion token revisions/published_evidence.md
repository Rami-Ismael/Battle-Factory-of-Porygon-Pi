# Published evidence of repeated token edits

## Do published trajectories show a position changing twice or more?

### Takeaway

| Paper | Evidence relevant to repeated edits | What can be concluded |
| --- | --- | --- |
| Uniform-noise diffusion language models, Schiff et al., UDLM | Explicit mechanism and capability claims; quality ablations | Repeated edits are permitted; no position-resolved empirical trajectory found. |
| Structured discrete diffusion, Austin et al., D3PM | Actual uniform-noise snapshots, Figs. 12–13 | **Inference:** all three character examples in Fig. 13 have 177/256 positions with at least two observed changes. |
| Score entropy discrete diffusion, Lou et al., SEDD | Uniform-model quality curve and final sample galleries | No position-resolved reverse trajectory found. |
| Uniform/Gaussian diffusion duality, Sahoo et al., Duo | Five-position schematic, Fig. 5(c) | **Inference:** four positions change at least twice, two change three times. Illustrative, not an empirical rate. |
| Generalized interpolating discrete diffusion, von Rütte et al., GIDD | Correction curves and before/after examples; verbal report of oscillation | No full token trajectory; oscillation supports repeated edits in a separate correction loop. |

Sources for the table, including limits of negative findings: [Schiff v3, §4 p. 5, §5.2 Table 7 p. 10, Apps. E–F pp. 31–46](https://arxiv.org/pdf/2412.10193v3); [Austin v3, App. B.3 Figs. 12–13 pp. 30–31](https://arxiv.org/pdf/2107.03006v3); [Lou v3, App. D Fig. 2 p. 17 and sample galleries Figs. 4, 7 pp. 20, 23](https://arxiv.org/pdf/2310.16834v3); [Sahoo v3, Fig. 5 p. 22 and App. A.7 pp. 22–23](https://arxiv.org/pdf/2506.10892v3); [von Rütte v2, §5.4 p. 8, Fig. 3 p. 7, App. C pp. 15–16](https://arxiv.org/pdf/2503.04482v2).

### Cited findings

This extends `00_earlier_note_revision_count.md`, which I read before inspecting the papers. I inspected the latest PDFs returned by arXiv on 2026-11-09: Schiff v3, Austin v3, Lou v3, Duo v3, GIDD v2. Page numbers below are printed page numbers, also their one-based PDF page numbers. These versions include revisions later than the first-submission years in the question. The reviewed material was the full PDFs, including appendices and sample galleries. [Schiff](https://arxiv.org/pdf/2412.10193v3), [Austin](https://arxiv.org/pdf/2107.03006v3), [Lou](https://arxiv.org/pdf/2310.16834v3), [Sahoo](https://arxiv.org/pdf/2506.10892v3), [von Rütte](https://arxiv.org/pdf/2503.04482v2).

- Austin Fig. 12 contains two LM1B uniform-noise trajectories; Fig. 13 contains three text8 uniform-noise trajectories. Both show states labelled t = 999, 800, 600, 400, 200, 0 from models using 1,000 diffusion steps. Text8 is character-level, including spaces. LM1B uses an 8,192-entry SentencePiece vocabulary. These are sparse snapshots, not all sampler steps. [Austin, §5.1 p. 6, §5.2 p. 7, App. B.3 pp. 30–31](https://arxiv.org/pdf/2107.03006v3#page=30).
- Duo Fig. 5 contrasts autoregressive generation, masked diffusion, uniform-state diffusion and the deterministic discrete trajectory used for distillation. The caption says uniform-state tokens can visit multiple intermediate states. It supplies no empirical sample provenance, sampler-step labels between the endpoints, or edit-rate estimate. Treat it as a schematic. [Sahoo, Fig. 5 p. 22, App. A.7 pp. 22–23](https://arxiv.org/pdf/2506.10892v3#page=22).
- GIDD Fig. 3 gives aggregate correction/quality curves. Tables 1 and 7 show before/after replacements; App. I shows uncorrected and corrected final samples. None supplies the intermediate replacement order. Its correction procedure is additional to ancestral denoising. [von Rütte, §3.4 p. 5, Fig. 3 p. 7, Tables 1 and 7 pp. 2, 18, App. I pp. 30–35](https://arxiv.org/pdf/2503.04482v2).
- SEDD's uniform-model Fig. 2 is generative perplexity versus computation. Its uniform sample galleries, Figs. 4 and 7, are final texts, not successive states of one sample. [Lou, App. D.2 pp. 17–18, Figs. 4 and 7 pp. 20, 23](https://arxiv.org/pdf/2310.16834v3#page=17).

### Inferences

#### Counting convention and reproducibility

**Inference:** For displayed states in generation order, I count a change at position i whenever its value differs between adjacent displayed snapshots. Initial random characters are included as the first observed state; departure from that state counts once. A position is re-edited when its observed change count is at least two. A return to a previous value counts as another change. Spaces count as tokens for text8. This is my operational convention, not a metric introduced by these authors. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

**Inference:** Every change between two snapshots establishes at least one actual token change somewhere inside that interval. Equal snapshots do not establish that the token stayed unchanged between them. Therefore every count below is a lower bound on actual changes, and no exact individual sampler-step timestamp can be recovered. In particular, t=999 is the first displayed state, not proof of the sampler's complete initial state. [Austin, Figs. 12–13 pp. 30–31](https://arxiv.org/pdf/2107.03006v3#page=31).

I counted Fig. 13 myself by extracting its PDF text and drawn space markers, expanding typographic ligatures, and joining wrapped lines without adding layout spaces. Each of the 18 states reconstructs to exactly 256 characters drawn only from a–z and space. I visually inspected the rendered figure, including its space glyphs. Ordinary PDF text extraction drops leading/trailing space markers and would misalign these counts.

The accompanying `count_published_trajectories.py` reproduces the extraction with `pypdf`. `published_trajectory_counts.json` contains all 18 states and, for **every position**, its six values, observed count, and changed time intervals. Indexing is one-based from the left, including all spaces. These files are researcher-produced evidence, not author-released measurements. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

#### Actual character trajectories: full displayed-state counts

**Inference, my count:** Each row below describes one complete displayed six-state trajectory, ordered top to bottom in Fig. 13. The histogram gives the number of positions exhibiting each observed change count. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

| Example | 0 changes | 1 change | 2 changes | 3 changes | 4 changes | 5 changes | Positions with ≥2 changes | Total observed change events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top | 4 | 75 | 106 | 65 | 6 | 0 | 177/256 | 506 |
| Middle | 2 | 77 | 125 | 46 | 6 | 0 | 177/256 | 489 |
| Bottom | 2 | 77 | 110 | 58 | 8 | 1 | 177/256 | 508 |

**Inference, my count:** The following are changed-position totals per displayed interval. An interval contributes at most once per position, regardless of unobserved intermediate edits. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

| Example | 999→800 | 800→600 | 600→400 | 400→200 | 200→0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top | 243 | 117 | 90 | 45 | 11 |
| Middle | 245 | 119 | 75 | 36 | 14 |
| Bottom | 243 | 125 | 78 | 48 | 14 |

**Inference, my count:** Especially clear repeated-edit witnesses follow. `_` means one space. Values always correspond to t = 999, 800, 600, 400, 200, 0. The exact change intervals follow by comparing adjacent entries. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

| Example / position | Six displayed values | Observed changes | Changed intervals |
| --- | --- | ---: | --- |
| Top / 53 | r → z → m → l → n → n | 4 | 999→800; 800→600; 600→400; 400→200 |
| Top / 75 | v → b → c → x → b → b | 4 | 999→800; 800→600; 600→400; 400→200 |
| Top / 11 | k → k → k → k → i → w | 2 | 400→200; 200→0 |
| Middle / 45 | a → x → v → x → o → o | 4 | 999→800; 800→600; 600→400; 400→200 |
| Middle / 170 | f → k → e → e → m → _ | 4 | 999→800; 800→600; 400→200; 200→0 |
| Bottom / 27 | u → z → v → v → y → e | 4 | 999→800; 800→600; 400→200; 200→0 |
| Bottom / 96 | d → t → e → _ → w → y | 5 | Every displayed interval |

**Inference:** Top position 75 and middle position 45 revisit earlier values, so a first visit to a value does not establish settlement. Top position 11 stays visually unchanged through t=400, then changes twice within the last 400 reverse-time units. These are direct counterexamples to treating early stability as irreversible commitment. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

**Inference, my count:** All positions that change in the last displayed interval also changed in an earlier displayed interval. Their exact indices are:

- Top: 11, 29, 37, 46, 118, 120, 135, 173, 237, 246, 256.
- Middle: 17, 31, 48, 97, 108, 119, 142, 151, 160, 170, 185, 215, 216, 217.
- Bottom: 9, 27, 42, 47, 48, 96, 118, 123, 127, 150, 169, 210, 212, 243.

These establish actual late re-editing within 200→0, but cannot locate the changes inside that interval. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

#### Duo's schematic: every position counted

**Inference, my count:** Fig. 5(c) has five displayed rows. I label them r0–r4 from top to bottom. Only r0 and r4 have time labels, t=1 and t=0; r1–r3 are not identifiable numerical sampler steps. [Sahoo, Fig. 5(c) p. 22](https://arxiv.org/pdf/2506.10892v3#page=22).

| Position | r0 → r1 → r2 → r3 → r4 | Changes | Changed row intervals |
| --- | --- | ---: | --- |
| 1 | Really → Really → Really → Dual → Dual | 1 | r2→r3 |
| 2 | Through → Number → Other → Other → Diffusion | 3 | r0→r1; r1→r2; r3→r4 |
| 3 | Little → Building → Building → For → For | 2 | r0→r1; r2→r3 |
| 4 | Good → Good → Think → Think → Discrete | 2 | r1→r2; r3→r4 |
| 5 | Always → Question → Date → Date → Data | 3 | r0→r1; r1→r2; r3→r4 |

**Inference:** Thus four of five positions are re-edited; two change three times; three positions still change in the last illustrated interval. This is a statement about a drawing, not a measured 80% re-edit rate or evidence about real sampler timing. Fig. 5(d), by contrast, illustrates a distillation target with at most one flip per position. Appendix A.7 explicitly distinguishes that target construction from learned Duo sampling without irreversible carry-over. [Sahoo, Fig. 5 and App. A.7 pp. 22–23](https://arxiv.org/pdf/2506.10892v3#page=22).

### Gaps

- Fig. 12's LM1B strings are detokenized SentencePiece outputs with changing line wraps. I did not reconstruct the exact original subword token IDs or count positions from word alignment. Its evolving text is qualitative evidence, but I do not claim a whole-figure per-token count. Fig. 13 supplies an unambiguous character-position answer for the same paper. [Austin, §5.2 p. 7, Figs. 12–13 pp. 30–31](https://arxiv.org/pdf/2107.03006v3).
- No position-resolved trajectory was located for Schiff or Lou, and no full replacement trace was located for GIDD. This is an inspection result for the specified versions, not proof that no such data exists outside the papers. [Schiff, Figs. 1–6 and Apps. E–F](https://arxiv.org/pdf/2412.10193v3); [Lou, Figs. 1–14 and App. D](https://arxiv.org/pdf/2310.16834v3); [von Rütte, Fig. 3, Apps. C and I](https://arxiv.org/pdf/2503.04482v2).

## Which papers claim continuous editing or error correction, and what supports the claim?

### Takeaway

Schiff and Duo explicitly frame uniform-noise sampling as repeatedly editable and self-correcting. GIDD supplies stronger empirical evidence for useful correction, but in a separate correction pass. Austin supplies actual repeated-edit snapshots. None of these observations alone measures how often an ordinary reverse sampler corrects an error rather than merely changes a token. [Schiff, §4 p. 5 and Table 7 p. 10](https://arxiv.org/pdf/2412.10193v3); [Sahoo, §§2.1, 5.2 pp. 3, 9](https://arxiv.org/pdf/2506.10892v3); [von Rütte, §5.4 p. 8](https://arxiv.org/pdf/2503.04482v2); [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3).

### Cited findings

- Schiff's abstract uses “continuously edit their outputs”; the introduction links repeated editing to correcting errors. Section 4 explicitly notes a token can change again after its first change. Table 7 supports the authors' explanation through faster-sampling image quality: at 128 steps, uniform-noise FID is 30.48 versus masked-model 64.09; at 1,024 steps, 26.70 versus 27.94. These are quality measurements, not revision counts. Appendix E.3 and Table 35 repeat the recovery-from-mistakes interpretation for faster image sampling, without tracing edits. [Schiff, abstract/introduction p. 1, §4 p. 5, §5.2/Table 7 p. 10, App. E.3 p. 34/Table 35 p. 45](https://arxiv.org/pdf/2412.10193v3#page=10).
- Duo describes continual updates and natural self-correction in §§1–2.1. Section 5.2 attributes its low-step advantage to fixing earlier errors in later denoising steps; Figs. 3–4 and 9 measure final quality/diversity versus computation. Figure 5 and App. A.7 explain repeated updates but do not measure their frequency. [Sahoo, pp. 2–3, §5.2 pp. 9–10, Fig. 9 p. 29, Fig. 5/App. A.7 pp. 22–23](https://arxiv.org/pdf/2506.10892v3).
- GIDD's separate fixed-point correction selects a replacement for one position per iteration. Figure 3 reports changes to up to 10% of tokens; §5.4 reports generative perplexity improving 214→93.3 and self-accuracy 62.0%→73.5% for its larger model with 20% uniform noise. Table 4 adds model-judged quality; Tables 1/7 and App. I give before/after examples. These show useful correction, without reporting per-position multiplicity. [von Rütte, §3.4 p. 5, Fig. 3/Table 4 p. 7, §5.4 p. 8, Table 7 p. 18, App. I pp. 30–35](https://arxiv.org/pdf/2503.04482v2).
- GIDD App. C explicitly reports oscillation between two or more states with similar self-accuracy and uses early-stopping patience 32. This is the clearest verbal observation of repeated state revisitation in this source set, confined to correction. [von Rütte, App. C pp. 15–16](https://arxiv.org/pdf/2503.04482v2#page=15).
- I found no equivalent explicit repeated-edit/error-correction claim in SEDD's own discussion. Its evidence relevant here is uniform-transition theory, quality versus computation, and final samples. Austin's directly relevant evidence is the trajectory gallery rather than a quantified self-correction claim. [Lou, §§3–6 pp. 3–8, App. D pp. 17–30](https://arxiv.org/pdf/2310.16834v3); [Austin, App. B.3 pp. 27–33](https://arxiv.org/pdf/2107.03006v3#page=27).

### Inferences

- **Inference:** GIDD's reported oscillation necessarily entails repeated edits of at least one position if a sequence returns to a previous state through single-position replacements. It does not identify which positions, how often, or at what reverse-noise level. [von Rütte, §3.4 p. 5, App. C pp. 15–16](https://arxiv.org/pdf/2503.04482v2).
- **Inference:** Do not reinterpret GIDD's Fig. 3 as a histogram of ordinary denoising revisions. Nor does a single before/after replacement establish that its position changed twice. [von Rütte, §5.4 p. 8 and Fig. 3 p. 7](https://arxiv.org/pdf/2503.04482v2).
- **Inference:** Editability is a transition-kernel property; repeated edits are a trajectory property; improvement is a quality property. The papers provide different evidence for each. The actual D3PM character changes need not all be semantically beneficial, and Duo's drawing cannot estimate an empirical correction rate. [Schiff, §4 p. 5](https://arxiv.org/pdf/2412.10193v3); [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3); [Sahoo, Fig. 5 p. 22](https://arxiv.org/pdf/2506.10892v3).

### Gaps

No paper in this inspected set reports a population distribution of per-position repeat counts, repeated-edit probability conditional on an earlier edit, or successful-correction fraction during ordinary uniform-noise reverse sampling. Figures and endpoint quality results do not supply those missing measurements. [Schiff](https://arxiv.org/pdf/2412.10193v3), [Austin](https://arxiv.org/pdf/2107.03006v3), [Lou](https://arxiv.org/pdf/2310.16834v3), [Sahoo](https://arxiv.org/pdf/2506.10892v3), [von Rütte](https://arxiv.org/pdf/2503.04482v2).

## Do most edits occur early, and do sequences settle before sampling ends?

### Takeaway

**Inference:** Austin's three displayed character trajectories show sharply declining changed-position counts as noise falls, while still showing re-edits in the final displayed interval. They support “more visible changes early, some re-editing late” for those examples. I found no general measured settling-time claim for ordinary uniform-noise reverse sampling in the five papers. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31); other inspected sources and limits above.

### Cited findings

- SEDD §5.1 describes approximately t·d altered tokens under its forward-noise schedule. This concerns distance from clean data at a noise level, not the time distribution of reverse edits. [Lou, §5.1 p. 5](https://arxiv.org/pdf/2310.16834v3#page=5).
- Schiff's explicit early-decoding observation concerns masked diffusion under strong classifier-based guidance, not uniform-noise settling. [Schiff, App. E.2 p. 33](https://arxiv.org/pdf/2412.10193v3#page=33).
- Duo's one-flip deterministic target is a distillation construction, not an empirical settling claim about its learned sampler. Its Greedy-Tail method changes the final transition to greedy decoding; it does not report that all tokens were already final. [Sahoo, §3.3 p. 5, App. A.7 p. 23](https://arxiv.org/pdf/2506.10892v3).
- GIDD App. D's quality plateau around 128–256 denoising steps compares complete runs with different computational budgets. It is not a time-resolved within-run settlement measurement. Correction-loop stability is also qualified by the oscillation report. [von Rütte, Apps. C–D pp. 15–17, Fig. 6 p. 17](https://arxiv.org/pdf/2503.04482v2).

### Inferences

**Inference, my count:** Across Austin Fig. 13's three examples, the first two displayed intervals, 999→800 and 800→600, contain 1,092 of 1,503 observed changes, about 72.7%. They span about 40% of the displayed reverse-time range. The last interval has 39 observed changes, all at positions already changed earlier. This establishes front-loaded *snapshot differences*, not the fraction of all hidden sampler transitions. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

**Inference, my count:** The following table groups positions by their last *observed* change interval. “None” means identical across all six snapshots, which still permits hidden excursions. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

| Example | None | 999→800 | 800→600 | 600→400 | 400→200 | 200→0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top | 4 | 69 | 60 | 70 | 42 | 11 |
| Middle | 2 | 74 | 71 | 60 | 35 | 14 |
| Bottom | 2 | 70 | 63 | 62 | 45 | 14 |

**Inference:** By t=600, 133, 147 and 135 positions respectively have reached their final value and never visibly leave it in the remaining snapshots. By t=200, those counts are 245, 242 and 242. These are retrospective statements about sparse observations; they are upper bounds on actual uninterrupted settlement by those times. None of the three whole sequences is visibly settled at t=200, because each differs at t=0. [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3#page=31).

### Gaps

- No full 1,000-step trace is printed in Austin's figures, so exact last-change steps, hidden returns and the real early/late event ratio remain unknown. [Austin, Figs. 12–13 pp. 30–31](https://arxiv.org/pdf/2107.03006v3).
- No cited source establishes an irreversible first commitment or a dataset-level settling-time distribution for an ordinary uniform-noise token. A statement that most positions *appear stable at sampled checkpoints* should not be upgraded to a guarantee that they stay unchanged between checkpoints. [Schiff, §4 p. 5](https://arxiv.org/pdf/2412.10193v3); [Austin, Fig. 13 p. 31](https://arxiv.org/pdf/2107.03006v3); [Sahoo, App. A.7 pp. 22–23](https://arxiv.org/pdf/2506.10892v3).
