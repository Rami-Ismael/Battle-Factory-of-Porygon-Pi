# Mask Lab: see what changes, measure what helps

Open [index.html](index.html) directly in a browser. It is a self-contained
snapshot: no API key, live service or experiment runtime is required.
[data.json](data.json) holds the displayed results. Edit `template.html` or
`data.json`, then run `python3 blog/mask-lab/rebuild.py`. The rebuild keeps the
sprites embedded in the checked-in page and adds any opponent sprite it lacks
from `notes/Blog/visuals/search-loop/sprites/`.

The page asks one thing: when part of a random legal Reg M-B team is hidden and
a writer fills it, does the completed team battle better? It is laid out in the
order the questions were asked:

0. Explorer: the masked team once, every writer's fill stacked under it with
   the validator's reason on rejection, and the 49-cell threat constellation
   drawn as each opponent's six sprites. It opens on the reasoning studies
   (20 starting teams, 7 mask families, 3 repeats, 7 writers); a toggle
   switches to the earlier 5-starter, 270-mask, 4-writer grid. The two are
   never pooled: the earlier grid's Ling arms are prompt settings, not the
   reasoning switch.
1. Ling or diffusion? Legality bars and paired battle differences.
2. Does reasoning help Ling? Paired legality and battle deltas, discordant
   masks, legality by mask family.
3. Does more reasoning help Mercury? Legality, cost and reasoning tokens by
   effort, paired deltas against off, rejection reasons by effort.
4. Do the writers keep writing the same thing? Distinct fills, largest share
   and meta-pool share per writer and mask family, for either study.
5. Verdict, then the method notes.

A panel is 50 battles: 49 distinct meta opponents, one counted twice (checked
against the per-opponent bits in `data.json`).

The current study uses 20 starters, seven masks per starter and three repeats:
420 outputs per method. Methods are Ling with reasoning off/on, Mercury with
reasoning off/low/medium/high, and the Pokémon diffusion ensemble of the
search-loop-trained model (F1) and matchup-matrix-augmented model (G2), using
their regulation-specific v3 checkpoints.

Three views intentionally use different denominators:

- Full set: all 420 attempts, legality, and descriptive win rates among each
  method's legal outputs. Illegal outputs are not fabricated battle losses.
- Shared cohort: 64 trial/repeat pairs legal for all seven methods, across 20
  starters; win rates weight starters equally.
- Pairwise: diffusion versus each LLM setting on their larger jointly legal
  subsets, with exploratory paired starter-bootstrap intervals.

Diffusion achieved 100% legality and a 9.24% full-set legal-team win rate.
On the shared cohort it scored 2.13%; Mercury high scored 2.29%, without a clear
paired advantage over diffusion. These compare complete systems on the frozen,
visible Champions M-B opponent pool, not equal-input neural architectures or
held-out/live-meta generalization. Diffusion uses constrained decoding and
non-neural Stat Point fills. Cached panel reuse and selection effects matter.

Mercury recorded $0.6902 across its four arms. Five missing cost records and
unknown original timeout charges are excluded, as are $0.0040 of preflight calls.
Diffusion has no API fee; local training/hardware/electricity costs are unmeasured.

The `allMethodComparison`, `matchedDiffusion`, `mercury` and `reasoning` keys in
`data.json` preserve the summaries and audit metadata. Raw paid completions and
per-panel records remain in ignored local `code/results/` study directories.

The `study20` key holds the per-trial records of the reasoning studies, which
the explorer draws. `export_study20.py` writes it from the three saved study
directories (`reasoning-expanded-20261012`, `mercury-reasoning-20261013`,
`diffusion-matched-20261013`); the 5-starter keys (`starts`, `tasks`, `results`,
`originals`, `arms`, `opponents`) are left as they were. Inside `study20`:

- `starts`: the 20 starting teams; `trials`: the 140 masks (`s{starter}-{family}-{k}`)
  with their masked paths in the explorer's `team[i].field` form.
- `results["{trial}/r{repeat}"][writer]`: each of the seven writers' fill
  (`fill[k]` is the value written at `paths[k]`, a whole slot for a
  whole-Pokémon mask) with `win` = wins / 50 and `opp`, or on rejection its
  `stage` (`parse`, `response`, `schema`, `validator`) and the validator's
  message. `cost` is the recorded API charge and `rt` the reported reasoning
  tokens.
- `opp` is 49 characters in `opponents` order, like the 5-starter records, but
  each character is the number of battles won against that opponent, because
  the twice-counted team (weight 2) is battled twice here and can be won once.
  The twice-counted team is a different meta team from the earlier grid's.
- `untouched` is empty: the 20 starting teams were never battled on their own,
  so the explorer shows no untouched baseline for this study.
The study-specific analysis scripts require those saved studies; a fresh clone
can inspect this snapshot but cannot reproduce absent model checkpoints or raw
experiments without the runtime and datasets described in the code documentation.
