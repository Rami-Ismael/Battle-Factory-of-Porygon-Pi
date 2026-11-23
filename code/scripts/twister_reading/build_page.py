"""One report page for the TWISTER reading: every section opens with a figure. Charts come from figures.py (the same
options its chart pages use). Writes <out>/index.html (full document) and <out>/artifact.html (content only).
Usage: build_page.py <data dir> <out dir>"""
import html, json, os, runpy, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
os.environ["TWISTER_DATA"] = str(DATA)
sys.argv = [str(HERE / "figures.py"), str(DATA), str(OUT)]
CH = runpy.run_path(str(HERE / "figures.py"), run_name="__main__")["CHARTS"]
T = json.load(open(DATA / "tilt_analysis.json"))
esc = html.escape


def chart(key):
    c = CH[key]
    return (f'<figure class="fig" id="f-{key}"><figcaption><b>{esc(c["title"])}</b><span>{esc(c["sub"])}</span></figcaption>'
            f'<div class="chart" data-chart="{key}" style="height:{c["height"]}px"></div>'
            + (f'<details><summary>How this was measured</summary><p>{esc(c["note"])}</p></details>' if c["note"] else "") + '</figure>')


def g(row, gd=2): return T.get(f"{row}|g{gd}", {})
def tval(row, gd, beta, metric="mean"):
    return next(c[metric]["point"] for c in g(row, gd)["tempered"] if c["beta"] == beta)

LIN, PF = "sched_linear_random", "sched_per_field_random"
lin2, pf2 = g(LIN), g(PF)
ess = lambda r: sum(r["ess_share_by_ask"].values()) / len(r["ess_share_by_ask"])

# ---------- 1. scope ------------------------------------------------------------------------------------------------
SCOPE = [  # claim, lane, carries over?, assumption
    ("Enforcing a constraint step by step, even exactly, tilts samples away from the model's own probabilities over valid outputs",
     "decoding", "yes", "Our random-order rows: the lookahead and same-pass legality checks remove probability at each draw"),
    ("No tilt for a constraint that forbids nothing, or a one-step reveal", "decoding", "yes",
     "Species-first rows: the legality tables are inside the trained model, and one field per pass leaves nothing to re-score"),
    ("The tilt is a product of per-step ratios, so it can be corrected step by step (twisted SMC)", "decoding", "test",
     "Weights computable for our sampler; the correction is measured here by reweighting, not run as SMC"),
    ("TWISTER keeps validity (JSON: 100% parse, 98-99% schema)", "decoding", "no", "We already redraw until Showdown's validator accepts"),
    ("Exact for regular-language constraints via an automaton", "decoding", "weak",
     "Our rules (learnsets, Species and Item Clause) are a finite check over 84 fields, not a token automaton"),
]
MARK = {"yes": ("carries over", "ok"), "no": ("does not carry over", "no"), "test": ("measured here", "warn"), "weak": ("weakly", "warn")}
lane = lambda which: "".join(f'<div class="claim {MARK[c][1]}"><div class="claim-t">{esc(t)}</div><div class="claim-m"><span class="chip {MARK[c][1]}">{MARK[c][0]}</span> {esc(a)}</div></div>'
                             for t, l, c, a in SCOPE if l == which)
PIPE = [("Training loss", "values the legal Pokémon construction tables forbid get no probability (elbo_terms)", "model"),
        ("Each network pass", "same tables mask before the softmax (Model84._logprobs)", "model"),
        ("Several fields drawn from one pass", "random order only: later draws see earlier ones' legality", "tilt"),
        ("Lookahead", "random order only: a set value must leave its Pokémon a possible species", "tilt"),
        ("Showdown validator", "illegal teams redrawn: exact conditioning, no tilt", "exact")]
PK = {"model": ("part of the model", "ok"), "tilt": ("tilts samples", "no"), "exact": ("exact", "ok")}
pipe = "".join(f'<div class="pstep {PK[k][1]}"><b>{esc(a)}</b><span>{esc(b)}</span><span class="chip {PK[k][1]}">{PK[k][0]}</span></div>' for a, b, k in PIPE)
scope_fig = f'''<figure class="fig"><figcaption><b>All five claims are about decoding; the tilt they describe only exists in our random-order rows</b>
<span>Left: the paper's claims and whether each carries over. Right: where our legality rules act, from training to the validator</span></figcaption>
<div class="lanes"><div class="lane"><h4>Decoding claims</h4>{lane("decoding")}
<div class="claim gap"><div class="claim-t">Masking schedule (linear t, or species t^16, ability and item t^4, moves t^1.5, Stat Alignment and Stat Points t)</div>
<div class="claim-m"><span class="chip no">never varied</span> The paper trains nothing; it uses six released models</div></div></div>
<div class="lane"><h4>Where our legality rules act</h4><div class="pipe">{pipe}</div></div></div></figure>'''

# ---------- 2. predictions -------------------------------------------------------------------------------------------
def pf(x): return f"{x:.1f}"
PRED = [
    ("T1", "Species-first rows carry no tilt", 99, "holds", "median 0.000 nats removed per team, max 0.08 (3,456 fresh teams, both schedules, guidance 2, 4, 8)"),
    ("T2", "Random order: median refused share 20-80%, effective teams 20-80%", 55, "fails",
     f"refused share 1.000 (median {pf(-lin2['median_log_w'])} nats linear, {pf(-pf2['median_log_w'])} per-field); effective teams {ess(lin2)*100:.0f}% and {ess(pf2)*100:.0f}%"),
    ("T3", "Correcting the tilt moves mean win rate < 0.01, best of 64 < 0.02", 75, "partly",
     f"mean win rate +{lin2['mean']['gap']:.2f} linear, +{pf2['mean']['gap']:.2f} per-field (on about 2% effective teams); best of 64 {lin2['best@64']['gap']:+.3f} and {pf2['best@64']['gap']:+.3f} at guidance 2, lower at 4 and 8"),
    ("T4", "Tilt and win rate uncorrelated (|Spearman| < 0.1)", 65, "fails",
     f"Spearman within ask +{lin2['spearman_within_ask']:.2f} linear, +{pf2['spearman_within_ask']:.2f} per-field: less-bent teams win more"),
    ("T5", "Per-field random order refuses more than linear", 60, "holds",
     f"median {pf(-pf2['median_log_w'])} against {pf(-lin2['median_log_w'])} nats per team; same order on fresh teams (13.4 against 11.8)"),
    ("T6", "Refused mass rises with guidance 2 to 8", 50, "holds",
     f"linear {pf(-lin2['median_log_w'])}, {pf(-g(LIN, 4)['median_log_w'])}, {pf(-g(LIN, 8)['median_log_w'])} nats; per-field {pf(-pf2['median_log_w'])}, {pf(-g(PF, 4)['median_log_w'])}, {pf(-g(PF, 8)['median_log_w'])}"),
    ("T7", "Species-first first-try legality >= random order at every ask 13-17", 70, "holds",
     "species-first 100% at every ask and guidance; random order 80% to 93%"),
    ("T8", "The paper says nothing about linear against per-field under species-first", 0, "cannot decide", "both rows have zero tilt; battle gap already inside ±0.04 (KNOWN)"),
]
VK = {"holds": "ok", "fails": "no", "cannot decide": "warn", "partly": "warn"}
def pred_rows():
    return "".join(f'<div class="pred"><span class="pn">{n}</span><span class="pt">{esc(t)}</span>'
                   f'<span class="pc"><span class="bar"><i style="width:{c}%"></i></span><em>{c if c else "n/a"}{"%" if c else ""}</em></span>'
                   f'<span class="chip {VK.get(v, "warn")}">{esc(v)}</span><span class="pm">{esc(m)}</span></div>' for n, t, c, v, m in PRED)

# ---------- 5. weaknesses --------------------------------------------------------------------------------------------
WEAK = [
    ("Compute matching", "“Although we report average generation times, our implementation is not optimized for performance.”", "appendix-eval.tex:41",
     "TWISTER with 1 or 4 particles takes 1.4 to 2.1 times DINGO's time, and no method gets TWISTER's budget. Fair: give the uncorrected decoder and rejection sampling the same denoiser calls (8 times the samples for 8 particles) and plot distance against calls.",
     "8 × 10,000 constrained samples per point; about 8 × the paper's sampling time"),
    ("Seeds", "“% We run all experiments once.” (commented out of the compiled paper)", "appendix-eval.tex:32",
     "The compiled text gives no run count or seed. The only uncertainty is a split-half noise floor (method-eval.tex:9). Fair: three seeds per point with bootstrap intervals on the distance.",
     "3 × the sampling; about 0 extra for the intervals"),
    ("The judge", "“We use pairwise total variation distance to measure the similarity between the two distributions.”", "method-eval.tex:7",
     "Distance over token classes on a toy constraint ('a.*'), 3 of 6 models shown; JSON is scored only for validity, with an unnamed checker and no quality measure. Fair: all six models, and one task where the tilt could change answer quality.",
     "the other three models' points already exist; one quality benchmark"),
    ("Text against algorithm", "Main-text Algorithm 1 has its resampling step commented out (alg-twister.tex:68-84)",
     "method-algorithm.tex:16 · appendix-algorithm.tex:14",
     "The prose and the appendix algorithm resample when the effective sample size drops below a threshold; the main algorithm never resamples, and no threshold value is reported.",
     "one sentence and one number"),
    ("Baselines", "“The original code is not publicly available.”", "appendix-eval.tex:23, 25",
     "Both baselines are the authors' reimplementations, and TWISTER's own code is promised “upon acceptance” (mandatory.tex:11).", "release the code"),
]
weak = "".join(f'<div class="w"><h4>{esc(h)}</h4><blockquote>{esc(q)}<cite>{esc(c)}</cite></blockquote><p>{esc(f)}</p><p class="cost">Cost of the fair version: {esc(k)}</p></div>' for h, q, c, f, k in WEAK)

# ---------- 6. decoding rule -----------------------------------------------------------------------------------------
OURS = """# code/src/diffusion84.py @ 787fac31 (species-first rows)
for step, c in enumerate(ORDER):                        # :588  one field per pass
    lc = log_softmax(logits.masked_fill(~tables_ok, -1e4))  # :524  tables inside the model
    lc = g * lc + (1 - g) * lc_unguided                 # :592  guidance
    legal, plain, kept, fallback = _legal_parts(x, c)   # :593  [MASK], empty first move
    lc = lc.masked_fill(~legal, -inf) / temperature     # :595
    v = multinomial(softmax(lc))                        # :596  draw, temperature 1

# code/src/remask84.py @ 787fac31 (random-order rows)
reveal = masked & (rand < odds)                         # :223  each field unmasks independently
lp = _logprobs(x, ask); lpn = _logprobs(x, none)        # :69   ONE pass for all revealed fields
for c in targets:                                       # in ORDER, within the pass
    lc = g * lp[c] + (1 - g) * lpn[c]                   # :74
    legal = _legal_parts(x, c, lookahead=True)          # :75   sees this pass's earlier draws
    v = multinomial(softmax(lc.masked_fill(~legal)))    # :79"""
LLADA = """# ML-GSAI/LLaDA generate.py @ 9182493720
logits_with_noise = add_gumbel_noise(logits, temperature)   # :174
x0 = argmax(logits_with_noise)                               # :175  temperature 0 = argmax
if remasking == 'low_confidence':                            # :180
    x0_p = softmax(logits)[x0]                               # :182
elif remasking == 'random':                                  # :184
    x0_p = rand()                                            # :185
confidence = where(mask_index, x0_p, -inf)                   # :192
select = topk(confidence, k=num_transfer_tokens)             # :196
# no constraint, no legality mask anywhere in the sampler"""
TWIST = """# TWISTER, algorithms/alg-twister.tex (no code released)
for t in 0 .. T-1:                                      # :18
  for each particle i:
    x[t+1] ~ step-exact FFBS kernel(. | x[t])           # :21-26  exact constrained mean-field draw
    Zhat = clamped partition sum(x[t+1] | x[t])         # :27-31  valid mass, old denoiser
    Z    = partition sum(x[t+1] | x[t+1])               # :34-41  valid mass, re-queried denoiser
    w[i] *= Z / Zhat                                    # :53-60  re-freezing correction
  normalise w                                           # :62-66
  # resampling when ESS < threshold: commented out     # :68-84"""
dec_fig = f'''<figure class="fig"><figcaption><b>Three samplers, three places the constraint can act</b>
<span>Ours masks inside the model and, in random order, again at each draw; LLaDA has no constraint; TWISTER reweights particles by how much valid mass changed</span></figcaption>
<div class="codes"><div><h4>Ours (masks, no correction)</h4><pre>{esc(OURS)}</pre></div><div><h4>LLaDA (no constraint)</h4><pre>{esc(LLADA)}</pre></div><div><h4>TWISTER (corrects)</h4><pre>{esc(TWIST)}</pre></div></div></figure>'''

# ---------- 7. disagreements -----------------------------------------------------------------------------------------
DIS = [  # paper, what it blames for distortion, evidence, code, kind
    ("TWISTER (Thimmaiah et al. 2609.35609)", "Composing exact per-step constrained draws: valid mass is scored by one denoiser and re-scored by the next", "TVD on a toy constraint, 3 models, no seeds", "promised on acceptance", "self"),
    ("Dang & Ermon (2607.07026)", "Not discussed; exact per-step sampling is presented as the fix for DINGO's MAP", "validity and accuracy, LLaDA-8B and Dream-7B; 3 seeds for runtime only", "none linked", "warn"),
    ("DINGO (Suresh et al. 2505.23061)", "Says it is “provably distribution-preserving” (main.tex:91) while picking the highest-probability string (dp_block argmax)", "greedy runs, no seeds", "“Code is provided” (supplement); no public repo", "no"),
    ("Dang et al. (2606.01926)", "Locally constrained decoding's “greedy and myopic enforcement” (problemsetup.tex:33)", "accuracy against particles; no divergence measured; 3 seeds for latency", "MhDang/gelatwo @ d974a70b9c", "ok"),
    ("Grammar-Aligned Decoding (Park et al. 2405.21047)", "Grammar-constrained decoding treats every future as equally completable (problem.tex:253)", "KL to target against samples, Mistral-7B, 36 problems, no seeds", "ebmoon/transformers-GAD @ bc062c68e0", "ok"),
    ("Loula et al. (2504.13139)", "“greedy approximations that badly distort the distribution” (conference.tex:296)", "log Z − KL estimate, 4 domains, bootstrap 95% intervals", "probcomp/genlm-control @ e8e39c2040", "ok"),
    ("Mündler et al. (2508.10111)", "Not discussed; its completion mode “cannot rely on the model's probability distribution” (discussion.tex:12)", "validity and pass@1, 4 seeds, 95% intervals; appendix says seeds 0 to 4", "eth-sri/constrained-diffusion @ 0908291ae5 (runner: seeds [0,1,2,3])", "warn"),
    ("LAVE (2602.00612)", "Not discussed; optimises validity and time", "syntactic@k, functional@k; run count not stated", "THU-Agent/LAVE @ b4800d7ab3 (built on Mündler's codebase)", "warn"),
    ("DConstrain (2610.30920)", "Argues bias with a toy example (98.78% against 90%), measures none", "BFCL, JSONSchemaBench; seed 0, one run", "none linked; cites TWISTER, never runs it", "warn"),
]
dis = "".join(f'<div class="drow"><div class="dq">{esc(p)}</div><div class="dcells"><div class="dcell {k}"><b>blames</b>{esc(b)}</div>'
              f'<div class="dcell"><b>evidence</b>{esc(e)}</div><div class="dcell"><b>code</b>{esc(c)}</div></div></div>' for p, b, e, c, k in DIS)
SETTLE = ("The experiment that would settle it for any of these: the same model and constraint, the uncorrected decoder against "
          "the corrected one against rejection sampling, at matched denoiser calls, with seeds, scored by distance to the rejection "
          "samples and by task quality. None of the nine papers runs all four parts.")

# ---------- 8. adoption ---------------------------------------------------------------------------------------------
ADOPT = [("2024-05", "Grammar-Aligned Decoding", "the bias named, for autoregressive models"), ("2025-04", "Loula et al. SMC", "SMC to correct it, autoregressive"),
         ("2025-05", "DINGO", "first diffusion constrained decoder"), ("2025-08", "Mündler et al.", "context-free grammars, diffusion"),
         ("2026-02", "LAVE", "lookahead then verify"), ("2026-06", "Dang et al.", "tractable proposals, autoregressive"),
         ("2026-07", "Dang & Ermon", "step-exact sampling, diffusion"), ("2026-09", "TWISTER", "corrects step-exact sampling"),
         ("2026-10", "DConstrain", "the one citer: cites, does not run")]
adopt = "".join(f'<div class="tl{" me" if n == "TWISTER" else ""}"><span class="td">{d}</span><b>{esc(n)}</b><span>{esc(w)}</span></div>' for d, n, w in ADOPT)

# ---------- 9. experiments -------------------------------------------------------------------------------------------
EXP = [  # name, code id, what changes, prediction, result that changes the blog, sampling, battles
    ("Exact random order by rejection", "rand_plain_reject", "Random order without the lookahead; redraw until Showdown accepts. Draws exactly TWISTER's target.",
     "Mean win rate rises toward species-first (about 0.45, from the corrected curve); best of 64 stays near 0.59",
     "Interval on mean win rate against species-first contains 0: random order's loss is the sampler's legality tilt, not the order. Stays 0.05 or more below: the order costs strength on its own",
     "about 100 tries per legal team (acceptance 0.8% to 1.1%, estimated from the weights): about 15 h per row at asks 13-17, about 3 h at ask 15 alone", "47,040 per row (about 10 min)",
     "Ran 2026-11-17 (#358): mean win rate 0.410 against species-first 0.490, -0.079 [-0.094, -0.064]; wins back 49% [41%, 57%] of the gap, not the 84% reweighting estimated. Acceptance 1.52% of 63,295 tries."),
    ("Random order with twisted SMC", "rand_twist_k8", "8 partial teams per final team; after each pass, weight by the kept share and resample when fewer than 4 are effective.",
     "Between the as-drawn and the exact row; closer to exact as K grows", "Same as above, at one twelfth of rejection's sampling",
     "8 times the sampling: about 80 min per row; 32 particles about 5 h", "47,040 per row"),
    ("Two-top-values gap order", "order_margin", "Fill next the field whose top two values are furthest apart (LLaDA's margin variant).",
     "If confident fields are species, tilt near 0 and win rate near species-first; if they are moves, tilt as in random order", "A tilt-free order other than species-first would show the tilt, not the order, decides",
     "1 times: about 10 min per row", "47,040 per row"),
    ("Position temperature 1 on confidence order", "order_conf_tpos1", "Choose the next field in proportion to its top-value confidence (Two Temperatures' TLC).",
     "Tilt between species-first and random order", "Tilt and mean win rate should move together across the three orders; if not, the tilt is not the cause",
     "1 times: about 10 min per row", "47,040 per row"),
]
exps = "".join(f'<div class="w"><h4>{esc(n)} <code>{c}</code></h4><p>{esc(w)}</p><p><b>Prediction:</b> {esc(pr)}</p><p><b>Changes the blog if:</b> {esc(ch)}</p><p class="cost">Sampling: {esc(sm)} · Battles: {esc(bt)}</p>'
               + (f'<p><span class="chip warn">ran</span> {esc(e[0])}</p>' if e else "") + '</div>' for n, c, w, pr, ch, sm, bt, *e in EXP)
EXPS = (f'<figure class="fig"><figcaption><b>Four sampling-only decoding rules on the same checkpoints; the first has since run (#358)</b><span>Each with common random numbers: '
        f'the grid seeds (7301 to 7303) and the same 49 opponents, so paired bootstrap intervals apply; asks 13/24 to 17/24</span></figcaption><div class="exps">{exps}</div></figure>')

# ---------- 11. vocabulary -------------------------------------------------------------------------------------------
VOC = [
    ("Constrained decoding", "Drawing outputs while forcing them to obey a rule; here, a legal Showdown team.", "f-01_paper_figure1"),
    ("Tilt (trajectory bias)", "How far the sampler's chance of a team drifts from the model's own chance among legal teams.", "f-02_tilt_per_team"),
    ("Correction weight", "Model's probability of the team over the sampler's, along the same draws; exp of minus the removed log mass.", "f-02_tilt_per_team"),
    ("Removed mass", "Probability the sampler took away at one draw before renormalising.", "f-03_tilt_by_field"),
    ("Step-exact decoder", "Draws all fields revealed at a step from the model, restricted exactly to completable values.", "f-01_paper_figure1"),
    ("Twisted SMC", "Many partial teams (particles) drawn side by side, each reweighted every step by a correction factor (the twist).", "f-05_tempered_correction"),
    ("Effective sample size", "How many equally weighted teams a weighted set is worth.", "f-05_tempered_correction"),
    ("Doob target", "The model's own sampler, conditioned on ending legal: what rejection sampling draws.", "f-04_win_by_tilt_decile"),
    ("Total variation distance", "Largest gap in probability two distributions give any set of outcomes; the paper's bias measure.", "f-01_paper_figure1"),
    ("Paired bootstrap interval", "Range from redrawing opponents and teams together for both rows, so shared luck cancels.", "f-05_tempered_correction"),
]
voc = "".join(f'<a class="v" href="#{f}"><b>{esc(t)}</b><span>{esc(d)}</span></a>' for t, d, f in VOC)
GLOSS_T = [
    ("Tilt", "How far a sampler's chance of drawing a team drifts from the trained model's own chance among legal teams, because legality rules removed probability while drawing. Random order with the lookahead: median 13.6 nats per team (linear schedule) and 15.2 (per-field), guidance 2; species-first order: 0.",
     "Thimmaiah et al. 2026, §3.4 (Proposition Re-Freezing Tilt); the paper also calls it trajectory bias."),
    ("Correction weight", "A team's probability under the trained model's own per-pass distributions over its probability under the sampler, along the same draws: the product over draws of the kept share. About 2% of random-order teams are effective under the full weights.",
     "Thimmaiah et al. 2026, §4.1 (Feynman-Kac potentials); the weights here are this project's (tilt_weights.py)."),
    ("Removed mass", "The probability a legality rule takes away at one draw before the sampler renormalises. Random order loses most at moves (8.8 nats per team) and abilities (5.4), per-field schedule, guidance 2.",
     "This project, legality_skew.py and tilt_weights.py, 2026-11-15; the paper's clamped partition sum (§3.2) is the kept share's counterpart."),
    ("Step-exact decoder", "A sampler that draws all the fields revealed at a step from the model's distributions, restricted exactly to values that can still end legal. Our lookahead is a partial version of it.",
     "Dang & Ermon 2026 (arXiv 2607.07026); named in Thimmaiah et al. 2026, §3.2."),
    ("Twisted sequential Monte Carlo", "Draw many partial teams side by side; after each step reweight each by a correction factor (the twist) and resample, so the finished set follows the target. Not run here.",
     "Thimmaiah et al. 2026, §2.3 and §4.2."),
]
gl = "".join(f'<div class="w"><h4>{esc(t)}</h4><p>{esc(d)}</p><p class="cost"><i>Origin</i>: {esc(o)}</p></div>' for t, d, o in GLOSS_T)
GLOSS = f'<figure class="fig"><figcaption><b>Proposed GLOSSARY.md entries: you pick which go in</b><span>Mechanism, my numbers, and an Origin line with section numbers</span></figcaption><div class="weak">{gl}</div></figure>'
REL_T = [
    ("Dang et al. 2026, tractable proposals (2606.01926)", "closest fit", "ok",
     "Names our three samplers: masking only what is illegal now (our plain sampler), a mask that knows whether the rule can still be met (our lookahead), and a proposal that also knows how likely it is to be met. Needs no LLM in principle; its probability proposal is an HMM distilled from model samples, which our model can supply. Experiments: Llama-3.1-8B and GPT-2 Large only.",
     "github.com/MhDang/gelatwo @ d974a70b9c, opened by hand: README names LCD, GCD and P-GCD; package has hmm/, nfa/, decoding/"),
    ("Loula et al. 2025, SMC control (2504.13139)", "machinery", "ok",
     "General SMC with potentials over any left-to-right generator; the particle and resampling code is what experiment 2 needs. Experiments: Llama 1B to 70B, no small model.",
     "github.com/probcomp/genlm-control @ e8e39c2040, opened by hand"),
    ("Grammar-Aligned Decoding 2024 (2405.21047)", "idea only", "warn",
     "Learns, from repeated samples, how likely each prefix is to end legal, instead of a 0-or-1 lookahead. Autoregressive only; Mistral-7B; no seeds.",
     "github.com/ebmoon/transformers-GAD @ bc062c68e0, opened by hand"),
    ("TWISTER itself (2609.35609)", "theory only", "warn",
     "The tilt and its correction carry over exactly; the automaton machinery does not (our rules are a finite check over 84 fields). Smallest model: OWT-130M.",
     "no code (promised on acceptance)"),
    ("Mündler et al., LAVE, DConstrain, DINGO, Dang & Ermon", "low fit", "no",
     "Text grammars, token automata and JSON schemas over 7-8B diffusion LLMs; none measures the tilt; none runs a small model.",
     "Mündler: eth-sri/constrained-diffusion @ 0908291ae5; LAVE: THU-Agent/LAVE @ b4800d7ab3; others none"),
]
rel = "".join(f'<div class="w"><h4>{esc(n)} <span class="chip {k}">{esc(f)}</span></h4><p>{esc(d)}</p><p class="cost">Code: {esc(c)}</p></div>' for n, f, k, d, c in REL_T)
REL = f'<figure class="fig"><figcaption><b>Which related papers fit an 84-field model</b><span>Each verified against its full LaTeX text by a Haiku reader; every code link opened by hand on GitHub</span></figcaption><div class="weak">{rel}</div><p class="fine">One Haiku reader said Dang et al. link no code; the paper links gelatwo in the introduction footnote (sections/1.introduction.tex:17). Corrected by hand.</p></figure>'
TB = json.load(open(DATA / "tables_analysis.json"))
tb_lin, tb_pf = TB["sched_linear_species_first|g2"], TB["sched_species_first|g2"]
drawn = tval(LIN, 2, 0.0); est = tval(LIN, 2, 1.0); SFM, REJM = 0.490, 0.410
x = lambda v: f"{100 * (v - 0.30) / 0.22:.1f}%"
TEASER = f'''<figure class="teaser" aria-label="Teaser: where legality rules act and what each costs">
<div class="tz-flow">
<div class="tz ok"><h4>1. The tables, inside the model</h4><span class="where">Training loss and every network pass, both orders</span>
<span class="num">{-tb_lin["median_log_w"]:.0f}–{-tb_pf["median_log_w"]:.0f}<small>nats removed per team</small></span>
<span class="num">+{tb_pf["spearman_within_ask"]:.2f}–{tb_lin["spearman_within_ask"]:.2f}<small>tie to win rate</small></span>
<span class="chip ok">part of the model: no tilt</span></div>
<div class="tz no"><h4>2. The lookahead</h4><span class="where">Random order only: a set value must leave its Pokémon a possible species</span>
<span class="num">{-lin2["median_log_w"]:.1f}–{-pf2["median_log_w"]:.1f}<small>nats removed per team</small></span>
<span class="num">+{pf2["spearman_within_ask"]:.2f}–{lin2["spearman_within_ask"]:.2f}<small>tie to win rate</small></span>
<span class="chip no">tilts: bent teams lose</span></div>
<div class="tz ok"><h4>3. Showdown's validator</h4><span class="where">Illegal teams redrawn, both orders</span>
<span class="num">0<small>nats: exact conditioning</small></span>
<span class="chip ok">no tilt</span></div></div>
<div class="tz-line"><span class="lbl">Mean win rate, linear schedule, guidance 2: random order with the lookahead, then drawn exactly from the model's own legal teams (#358), then species-first. The lookahead's tilt is the orange half of the gap.</span>
<div class="tz-track"><i style="left:{x(drawn)};width:calc({x(REJM)} - {x(drawn)});background:var(--no);opacity:.75"></i><i style="left:{x(REJM)};width:calc({x(SFM)} - {x(REJM)});background:var(--ink2);opacity:.35"></i></div>
<div class="tz-ends"><span>random order {100 * drawn:.1f}%</span><span>exact target {100 * REJM:.1f}% (49% of the gap)</span><span>species-first {100 * SFM:.1f}%</span></div></div>
</figure>'''
PROSE = dict(
    LEDE=(f"Yes, though not about the schedule itself. Species-first order, the only order the schedule comparison needs, has no tilt: "
          f"it draws teams from the model's own distribution. Random order needs the lookahead, which bends every team by a median "
          f"{pf(-lin2['median_log_w'])} to {pf(-pf2['median_log_w'])} nats, and the more a team is bent, the more it loses. Drawing exactly from the "
          f"model's own legal teams (#358) wins back half of random order's win-rate gap to species-first, not the {round(100 * (est - drawn) / (SFM - drawn))}% "
          f"reweighting estimated. The legal Pokémon construction tables remove far more, but that mass was never trained, and it predicts nothing."),
    P_SCOPE=("The paper never trains anything and never varies a masking schedule. What carries over is its accounting: any sampler that "
             "removes probability at each draw and renormalises draws from a different distribution than the model's own, restricted to legal "
             "outputs. In our code, the tables of legal Pokémon construction are inside the model (training and every pass), so removing what they "
             "forbid is not a tilt. The lookahead and the same-pass legality checks of random order are."),
    PRED_TITLE="Four predictions hold, three fail or partly fail, one cannot be decided",
    P_TILT=(f"Every one of the 135 grid cells was redrawn with its own seed and gave the stored teams bit for bit, so each team's correction weight "
            f"sits beside its 49 stored battle outcomes. On the random-order rows the weight and the win rate move together: Spearman +{lin2['spearman_within_ask']:.2f} "
            f"(linear) and +{pf2['spearman_within_ask']:.2f} (per-field) within asks, at guidance 2. Full correction leaves about 2% effective teams, so read the dose "
            f"curve's direction rather than its end point; the decile chart rests on about 96 teams per point. A weight does not say why a team is "
            f"weak: teams whose sets were drawn long before their species are both the most bent and, from the order results, the weakest. "
            f"The exact test on one Pokémon shows the mechanism: the lookahead keeps any ability some Fake Out user has, without weighting it by how "
            f"much of the model's next move stays legal, so 17% of the probability moves. The tables cut {-tb_lin['median_log_w']:.0f} to {-tb_pf['median_log_w']:.0f} "
            f"nats per team, but the training loss masks the same values, so the cut removes untrained mass and tracks win rate at only "
            f"+{tb_lin['spearman_within_ask']:.2f} and +{tb_pf['spearman_within_ask']:.2f}. Rejection without the lookahead, run since (#358), gives 0.410: "
            f"49% of the gap, against {round(100 * (est - drawn) / (SFM - drawn))}% from reweighting, whose 1-2% effective teams let a few heavy teams set the estimate."),
    P_JOINT=("TWISTER takes the model's per-field predictions as fixed and asks what the sampler does with them; it never checks that they come "
             "from one joint distribution (refuter, S11). Ours do not: the same held-out team scores 18.2 nats better species-first than moves-first "
             "on the per-field checkpoint and 12.3 on the linear one. So the tilt-corrected target is itself order-specific: a corrected random order "
             "is the model's own random-order distribution, not the species-first one."),
    P_RULE=("LLaDA's released sampler has no constraint at all. Ours puts the rules inside the model and then, in random order only, removes more "
            "at each draw without correcting. TWISTER is the only one of the three that corrects, and it releases no code. About two filled fields in five "
            "are not the model's top value in every row, so the samplers differ in what they remove, not in how greedy they are."),
    P_ADOPT=("Semantic Scholar lists one citer on 2026-11-15, DConstrain (arXiv 2610.30920), which mentions TWISTER in its introduction and related "
             "work and runs it nowhere. No released sampler changed its default: LLaDA's generate.py (9182493720) has no constraint, and the most-used "
             "constrained diffusion code, Mündler et al.'s, enforces validity and redraws without correcting. Nothing has replaced TWISTER because "
             "nothing has used it; the methods around it are shown faster or more often valid, never closer to the model's distribution at matched compute."),
)

PAGE = Path(HERE / "page_template.html").read_text()
body = PAGE
for k, v in dict(TEASER=TEASER, SCOPE_FIG=scope_fig, PREDS=pred_rows(), WEAK=weak, DEC_FIG=dec_fig, DIS=dis, SETTLE=esc(SETTLE), ADOPT=adopt, VOC=voc,
                EXPS=EXPS, REL=REL, GLOSS=GLOSS, **{k2: esc(v2) for k2, v2 in PROSE.items()}).items():
    body = body.replace("{{" + k + "}}", v)
for k in CH: body = body.replace("{{CHART:" + k + "}}", chart(k))
assert "{{" not in body, body[body.index("{{"):body.index("{{") + 40]
body = body.replace("__CH__", json.dumps({k: dict(option=v["option"]) for k, v in CH.items()}, ensure_ascii=False).replace("</", "<\\/"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "artifact.html").write_text(body)
(OUT / "index.html").write_text("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
print("page written:", OUT / "index.html")
