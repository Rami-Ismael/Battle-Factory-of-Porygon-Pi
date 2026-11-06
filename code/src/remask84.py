"""Remasking on the all-teams 84-field model (docs/remask-84-fields.md): the least-likely rule, the paper's rules, and random decoding.

The control, "No remasking", is the existing grid in results/stat_points_84_gen84_all_ep21.json, same asks, seeds and scorer.
Every arm keeps species-first order, guidance 2, temperature 1, rule masks and the Stat Alignment repair.

    python remask84.py check | probes | gen | battle | report      species-first arms
    python remask84.py random_decoding                        the sampling-only legality probe (REMASK84_SMOKE=1: one batch)
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ask_grid84 as A                                   # also sets up sys.path for the modules below
import diffusion84 as M
import matchup_db as MDB
from team_diversity import slots_from_paste

ARMS = {
    "remask_ll_10": "Least-likely 10% remasked, 84 passes",
    "remask_conf_loop": "Confidence-weighted 0.3 fields remasked, 210 passes",
    "remask_even_loop": "Even-odds 0.5 fields remasked, 210 passes",
}
ASK_NUMERATORS, SEEDS, KEEP = [4, 8, 12, 16], [1, 2, 3], A.KEEP
LOOP_START = 63                                          # alpha 0.75: the paper's t_on 0.25 on 84 fields
LOOP_PASSES = 126                                        # the paper's T = 4L has 3T/8 loop steps; L = 84
SIGMA_CONF = 0.3                                         # paper: 0.9 x min(1, (1 - alpha) / alpha) at alpha 0.75
ETA_CAP = 0.008                                          # the paper's best MAUVE setting on text (its Table 5)
LEAST_LIKELY = 0.10
OUT = A.RES / "remask84.json"
CONTROL = A.RES / "stat_points_84_gen84_all_ep21.json"


def least_likely(conf, fraction):
    """[B, COLS]: per team, round(fraction x filled), rounded stochastically, of its filled fields drawn with the lowest probability."""
    filled = conf.isfinite()
    k = torch.floor(fraction * filled.sum(1) + torch.rand(len(conf), device=conf.device)).long()
    columns = torch.arange(conf.shape[1], device=conf.device).expand_as(conf)
    rank = torch.empty_like(columns).scatter_(1, conf.argsort(1), columns)
    return (rank < k[:, None]) & filled


def odds_by_confidence(conf):
    """[B, COLS]: the paper's ReMDM-conf: odds sigma x exp(-p) / sum(exp(-p)) over filled fields, drawn independently."""
    filled = conf.isfinite()
    weight = torch.exp(-conf.masked_fill(~filled, 0.0)) * filled
    return torch.rand_like(conf) < SIGMA_CONF * weight / weight.sum(1, keepdim=True).clamp_min(1e-9)


def even_odds(conf):
    """[B, COLS]: the paper's ReMDM-cap: every filled field remasked with odds eta, independently."""
    return (torch.rand_like(conf) < ETA_CAP) & conf.isfinite()


def fill_pass(w, x, conf, targets, due, redo, ask_w, null_w, temperature=1.0, guidance=A.GUIDANCE, lookahead=False):
    """One network pass: draws each target field in ORDER, using one set of logits, so legality, and the lookahead when
    it is on (lookahead.py), see each earlier draw."""
    n = len(x)
    t = (1.0 - (x != 0).sum(1).float() / M.COLS).to(x.device)
    lp = w._logprobs(x, t, ask_w); lpn = w._logprobs(x, t, null_w)
    everyone = torch.ones(n, dtype=torch.bool, device=x.device)
    for c in targets:
        rows = everyone if c in due else redo[:, c]
        if not rows.any(): continue
        lc = guidance * w._column(lp, c) + (1 - guidance) * w._column(lpn, c)
        lc = lc.masked_fill(~w._legal(x, c, lookahead), -float("inf")) / temperature
        p = torch.softmax(lc, -1)
        v = torch.multinomial(p, 1).squeeze(1)
        conf[rows, c] = p.gather(1, v[:, None]).squeeze(1)[rows]
        x[rows, c] = (v + 1 if c >= M.STAT0 else v)[rows]    # a Stat Point output index is its value


def run_schedule(arm, step, conf, none):
    if arm == "remask_ll_10":
        for i, c in enumerate(M.ORDER):
            step([c], least_likely(conf, LEAST_LIKELY) if i else none)
    elif arm is None:
        for c in M.ORDER: step([c], none)
    else:
        rule = odds_by_confidence if arm == "remask_conf_loop" else even_odds
        for c in M.ORDER[:LOOP_START]: step([c], none)
        for _ in range(LOOP_PASSES): step([], rule(conf))
        for c in M.ORDER[LOOP_START:]: step([c], none)


@torch.no_grad()
def sample(w, n, ask, arm, stats=None):
    """n teams. arm None: the plain decode, one field per pass. remask_ll_10: remask before every pass. The others: fill 63 fields,
    loop LOOP_PASSES passes that only redraw, then fill the last 21."""
    ask_w = torch.full((n,), ask, device=M.DEV); null_w = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    conf = torch.full((n, M.COLS), float("inf"), device=M.DEV)
    stats = stats if stats is not None else {}
    stats.setdefault("passes", 0); stats.setdefault("redrawn", 0); stats.setdefault("changed", 0); stats.setdefault("redrawn_species", 0)
    none = torch.zeros_like(x, dtype=torch.bool)

    def step(new, redo):
        old = x.clone(); x[redo] = 0; conf[redo] = float("inf")
        targets = sorted(set(new) | set(redo.any(0).nonzero().flatten().tolist()), key=M.ORDER.index)
        if not targets: return
        fill_pass(w, x, conf, targets, set(new), redo, ask_w, null_w)
        stats["passes"] += 1; stats["redrawn"] += int(redo.sum()); stats["changed"] += int((redo & (old != x)).sum())
        stats["redrawn_species"] += int(redo[:, ::8][:, :6].sum())

    run_schedule(arm, step, conf, none)
    if (x == 0).any(): raise RuntimeError("a field is still masked after the last pass")
    return x


class Armed:
    """What ask_grid84.draw needs: sample() and to_paste(), for one arm."""
    def __init__(self, w, arm): self.w, self.arm, self.stats = w, arm, {}

    def sample(self, n, ask=None, **_):
        return sample(self.w, n, ask, self.arm, self.stats)

    def to_paste(self, row, repair=False): return self.w.to_paste(row, repair=repair)


def load(): return json.load(open(OUT)) if OUT.exists() else {"cells": {}}


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def species_of(paste): return sorted(s["species"] for s in slots_from_paste(paste))


def team_measures(pastes, stat_points):
    """Diversity over the whole group of teams: distinct line-ups, effective species, top species share, and the Stat Points check."""
    lineups = {tuple(species_of(p)) for p in pastes}
    counts = np.array(list(Counter(s for p in pastes for s in species_of(p)).values()), float)
    share = counts / counts.sum()
    totals = [sum(e) if isinstance(e, (list, tuple)) else sum(e.values()) for team in stat_points for _, e in team]
    return dict(teams=len(pastes), lineups=len(lineups), effective_species=float(np.exp(-(share * np.log(share)).sum())),
                top_species_share=float(share.max()), spend_66=float(np.mean([x == 66 for x in totals])), mean_stat_points=float(np.mean(totals)))


def check():
    """Passing no arm must reproduce the deployed sampler bit for bit."""
    w = A.load_model()
    torch.manual_seed(1); a = w.sample(16, ask=0.5, guidance=A.GUIDANCE)
    torch.manual_seed(1); b = sample(w, 16, 0.5, None)
    assert bool((a == b).all()), "plain path differs from Model84.sample"
    print("REMASK84_CHECK PASS: no-remasking path is bit-identical to Model84.sample", flush=True)


def probes():
    """Round trip and copies for each arm; a failed round trip stops the queue."""
    w = A.load_model(); out = load(); d = np.load(A.DATA); train = d["all_X"][~d["all_val"]].astype(np.int64); pf = {}
    for arm in ARMS:
        torch.manual_seed(A.SEED0 + 1)
        rows = sample(w, 96, 8 / 24, arm).cpu().numpy()
        pf[arm] = dict(round_trip=A.round_trip(w, rows[:20]), copies=A.copies(rows, train)[1])
        print(arm, json.dumps(pf[arm]), flush=True)
    out["preflight"] = pf; save(out)
    if not all(p["round_trip"].get("passed") for p in pf.values()): raise SystemExit("REMASK84_PROBES FAILED")
    print("REMASK84_PROBES PASS", flush=True)


def gen(seeds=SEEDS):
    out = load(); w = A.load_model()
    out.update(arms=ARMS, asks=ASK_NUMERATORS, checkpoint_sha256=hashlib.sha256(A.checkpoint_path().read_bytes()).hexdigest())
    for arm in ARMS:
        for seed in seeds:
            for k in ASK_NUMERATORS:
                tag = f"{arm}_a{k:02d}of24_s{seed}"
                if tag in out["cells"]: continue
                armed = Armed(w, arm); t0 = time.time()
                kept, stats, counts = A.draw(armed, k / 24, seed)
                out["cells"][tag] = dict(arm=arm, ask=k / 24, numerator=k, seed=seed, **counts, pastes=kept, stat_points=stats, sampler=armed.stats,
                                         seconds=time.time() - t0, measures=team_measures(kept, stats) if kept else None)
                save(out); m = out["cells"][tag]["measures"] or {}
                print(f"  {tag:34s} kept {len(kept):3d} legal {counts['legal']:6.1%} line-ups {m.get('lineups', 0):3d} eff {m.get('effective_species', 0):5.1f} "
                      f"66 {m.get('spend_66', 0):.0%} {time.time() - t0:5.0f}s", flush=True)


def battle():
    A.require_shuffled(); con = MDB.connect(); out = load()
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=A.BATTLES_PER_OPPONENT, seed=A.SEED0 + c["seed"], origin=f"remask84:{tag}")
        if not A.complete(sc): print(f"  {tag} INCOMPLETE: left for the next run", flush=True); continue
        c["y"] = [s["score"] for s in sc]; save(out); print(f"  {tag:34s} win {np.mean(c['y']):.3f}", flush=True)


RANDOM_ARMS = {
    "rand_plain": "No remasking, 84 passes",
    "rand_remask_conf": "Confidence-weighted 0.3 fields remasked, 210 passes",
    "rand_remask_even": "Even-odds 0.5 fields remasked, 210 passes",
}
RANDOM_TRIES = 384


@torch.no_grad()
def sample_random(w, n, ask, arm, stats, guidance=A.GUIDANCE, temperature=1.0, lookahead=False):
    """Random decoding as the paper does it: each masked field unmasks independently, and a remasked field goes back to the
    pool until the unmasking odds pick it. Within one pass the drawn fields go in ORDER, so legality sees earlier draws.
    ask None: no condition. lookahead: draw only values that leave every Pokémon whose species is masked a legal species;
    a remasked field counts as masked again."""
    ask_w = torch.full((n,), M.NAN if ask is None else ask, device=M.DEV); null_w = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    conf = torch.full((n, M.COLS), float("inf"), device=M.DEV)

    def step(unmask_odds, rule=None):
        masked = x == 0
        odds = unmask_odds(masked) if callable(unmask_odds) else unmask_odds
        reveal = masked & (torch.rand_like(conf) < odds)
        remask = rule(conf) if rule is not None else torch.zeros_like(masked)      # chosen from the team the network saw
        targets = sorted(reveal.any(0).nonzero().flatten().tolist(), key=M.ORDER.index)
        if targets:
            fill_pass(w, x, conf, targets, set(), reveal, ask_w, null_w, temperature, guidance, lookahead); stats["passes"] = stats.get("passes", 0) + 1
        x[remask] = 0; conf[remask] = float("inf")
        stats["remasked"] = stats.get("remasked", 0) + int(remask.sum())

    def fill(passes, start, end):                       # filled share moves from start to end, linear in alpha as the paper's schedule
        for i in range(passes):
            a_t, a_s = start + (end - start) * i / passes, start + (end - start) * (i + 1) / passes
            step(1.0 if i == passes - 1 and end == 1.0 else (a_s - a_t) / (1 - a_t))

    if arm == "rand_plain":
        fill(M.COLS, 0.0, 1.0)
    else:
        share = LOOP_START / M.COLS
        fill(LOOP_START, 0.0, share)
        if arm == "rand_remask_even":
            for _ in range(LOOP_PASSES): step(ETA_CAP * share / (1 - share), even_odds)    # the paper's sigma alpha / (1 - alpha)
        else:
            # The paper gives masked fields zero remasking odds, which never unmasks them while alpha is held; these odds keep the count level.
            for _ in range(LOOP_PASSES): step(lambda m: SIGMA_CONF / m.sum(1, keepdim=True).clamp_min(1), odds_by_confidence)
        fill(M.COLS - LOOP_START, share, 1.0)
    if (x == 0).any(): raise RuntimeError("a field is still masked after the last pass")
    return x


class RandomDecodingArm:
    """Random decoding as an arm object for ask_grid84.draw: sample() and to_paste(). arm: a RANDOM_ARMS key.
    lookahead: the switch of sample_random. sample() uses the guidance weight and temperature it is given and refuses
    any other setting."""
    def __init__(self, w, arm="rand_plain", lookahead=False):
        if arm not in RANDOM_ARMS: raise ValueError(f"arm must be one of {sorted(RANDOM_ARMS)}")
        self.w, self.arm, self.lookahead, self.stats = w, arm, lookahead, {}

    def sample(self, n, ask=None, guidance=1.0, temperature=1.0):
        return sample_random(self.w, n, ask, self.arm, self.stats, guidance=guidance, temperature=temperature, lookahead=self.lookahead)

    def to_paste(self, row, repair=False): return self.w.to_paste(row, repair=repair)


class SpeciesFirstArm:
    """Species-first decoding (Model84.sample) with the lookahead switch, as an arm object for ask_grid84.draw."""
    def __init__(self, w, lookahead=False): self.w, self.lookahead = w, lookahead

    def sample(self, n, ask=None, guidance=1.0, temperature=1.0):
        return self.w.sample(n, ask=ask, guidance=guidance, temperature=temperature, lookahead=self.lookahead)

    def to_paste(self, row, repair=False): return self.w.to_paste(row, repair=repair)


def random_decoding():
    """Sampling only: every try is counted, with no stop at a number of legal teams. Sets the try cap and battle budget later."""
    smoke = bool(os.environ.get("REMASK84_SMOKE"))
    path = A.RES / f"remask84_random_decoding{'_smoke' if smoke else ''}.json"
    out = json.load(open(path)) if path.exists() else {"cells": {}}
    out.update(arms=RANDOM_ARMS, asks=ASK_NUMERATORS, tries_per_cell=RANDOM_TRIES, checkpoint_sha256=hashlib.sha256(A.checkpoint_path().read_bytes()).hexdigest())
    w = A.load_model(); val = A.Validator()
    tries, seeds, asks = (96, [1], [8]) if smoke else (RANDOM_TRIES, SEEDS, ASK_NUMERATORS)
    try:
        for arm in RANDOM_ARMS:
            for seed in seeds:
                for k in asks:
                    tag = f"{arm}_a{k:02d}of24_s{seed}"
                    if tag in out["cells"]: continue
                    torch.manual_seed(A.SEED0 + seed); t0, stats = time.time(), {}
                    legal, rows, stat_points, reasons, rejected, held, tried = [], [], [], Counter(), 0, 0, 0
                    while tried < tries:
                        for r in sample_random(w, 96, k / 24, arm, stats).cpu().numpy():
                            tried += 1
                            p, slots = w.to_paste(r, repair=True); err = val(p)
                            if err is not None: rejected += 1; reasons[A.reason(err)] += 1
                            elif A.HO.is_heldout_text(p): held += 1
                            else: legal.append(p); rows.append(r.tolist()); stat_points.append([(s["species"], s["evs"]) for s in slots])
                    share = (tried - rejected) / tried
                    out["cells"][tag] = dict(arm=arm, ask=k / 24, numerator=k, seed=seed, tried=tried, rejected=rejected, heldout_dropped=held, legal=share, reasons=reasons.most_common(10),
                                             pastes=legal, rows=rows, stat_points=stat_points, sampler=stats, seconds=time.time() - t0, measures=team_measures(legal, stat_points) if legal else None)
                    tmp = path.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, path)
                    print(f"  {tag:36s} legal {share:6.1%} of {tried} · {time.time() - t0:5.0f}s · {reasons.most_common(1)}", flush=True)
    finally:
        val.close()
    picked = [np.array(r) for c in out["cells"].values() for r in c["rows"]][:20]
    out["round_trip"] = A.round_trip(w, picked) if picked else dict(skipped="no legal team")
    out["summary"] = {arm: float(np.mean([c["legal"] for c in out["cells"].values() if c["arm"] == arm])) for arm in RANDOM_ARMS if any(c["arm"] == arm for c in out["cells"].values())}
    tmp = path.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, path)
    print("REMASK84_RANDOM_DONE", json.dumps(out["summary"]), json.dumps(out["round_trip"]), flush=True)


def slope(asks, wins):
    return float(np.polyfit(asks, wins, 1)[0])


def accumulation(cells, orders=200, step=16):
    """Species accumulation curve, as in ecology: distinct line-ups, and line-ups at or above 0.531 on these 49 battles, as teams are added in random order."""
    teams = [(tuple(species_of(p)), y) for c in cells for p, y in zip(c["pastes"], c.get("y") or [None] * len(c["pastes"]))]
    rng = np.random.default_rng(0); sizes = list(range(step, len(teams) + 1, step)); seen, strong = np.zeros(len(sizes)), np.zeros(len(sizes))
    for _ in range(orders):
        order = rng.permutation(len(teams))
        for j, n in enumerate(sizes):
            pick = [teams[i] for i in order[:n]]
            seen[j] += len({t for t, _ in pick}); strong[j] += len({t for t, y in pick if y is not None and round(y, 4) >= 0.531})
    return dict(teams=sizes, lineups=(seen / orders).round(2).tolist(), strong_lineups_unconfirmed=(strong / orders).round(2).tolist())


def control_cells():
    cells = json.load(open(CONTROL))["cells"]
    return {f"plain_a{k:02d}of24_s{s}": cells[f"gen84_all_ep21_a{k:02d}of24_s{s}"] for k in ASK_NUMERATORS for s in SEEDS}


def report():
    out = load(); rows = {}
    plain = control_cells()
    for tag, c in plain.items():
        c["arm"] = "plain"; c["measures"] = team_measures(c["pastes"], c["stat_points"])
    cells = list(plain.values()) + list(out["cells"].values())
    for arm in ["plain", *ARMS]:
        rows[arm] = []
        for k in ASK_NUMERATORS:
            at = [c for c in cells if c["arm"] == arm and c["numerator"] == k]
            if not at: continue
            ys = [np.mean(c["y"]) for c in at if c.get("y")]
            meas = lambda f: float(np.mean([c["measures"][f] for c in at if c["measures"]]))
            rows[arm].append(dict(numerator=k, win=float(np.mean(ys)) if ys else None, win_by_seed=[float(y) for y in ys], legal=float(np.mean([c["legal"] for c in at])),
                                  lineups=meas("lineups"), effective_species=meas("effective_species"), spend_66=meas("spend_66")))
        print(arm, json.dumps(rows[arm]), flush=True)
    fits = {}
    for arm, row in rows.items():
        done = [r for r in row if r["win"] is not None]
        if len(done) == len(ASK_NUMERATORS):                    # a partial grid would label a lower ask as the top ask
            keep =[r for r in done if r["numerator"] != 12]    # 12/24 selects a different team population
            fits[arm] = dict(slope=slope([r["numerator"] / 24 for r in done], [r["win"] for r in done]), slope_without_12=slope([r["numerator"] / 24 for r in keep], [r["win"] for r in keep]),
                             top_ask_win=done[-1]["win"], line=A.line_measures(done), accumulation=accumulation([c for c in cells if c["arm"] == arm]))
        print(arm, json.dumps({k: v for k, v in fits.get(arm, {}).items() if k != "accumulation"}), flush=True)
    out["summary"] = rows; out["fits"] = fits; save(out)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "check": check()
    elif what == "probes": probes()
    elif what == "random_decoding": random_decoding()
    elif what == "gen": gen()
    elif what == "battle": battle()
    elif what == "report": report()
    else: raise SystemExit(__doc__)
