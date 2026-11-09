"""The 23-ask battle protocol for an 84-field model (issues #61 and #62; docs/stat-points-84-fields.md, "Battles").

Asks 1/24 to 23/24, seeds 1-3, guidance 2, temperature 1, rule masks on, species first, Stat Alignment repaired to match
the moves. Each cell draws until 64 teams pass Showdown and are not held-out teams, at most 1,280 tries. Each team plays
one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled.
Fields the model generates: all 84, except that the repair replaces its Stat Alignment.

    AG84_MODEL=gen84_all_ep21 python ask_grid84.py preflight | gen | battle | report      (AG84_SMOKE=1: plumbing check)

AG84_MODEL is a name in MODELS, or the checkpoint name train84.py --label-noise printed, with no row to add:
    AG84_MODEL=stat84_all_teams_es3_cap80_labelnoise1over48 python ask_grid84.py gen
    AG84_MODEL=stat84_labelled_noise_byepoch_es3_cap80_labelnoise1over48_ep03 python ask_grid84.py gen

Run preflight with MATCHUP_DB pointing at a private database. Finished cells are skipped, so every step resumes.
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L                                     # before diffusion84: its imports put /tmp/vgc-pilot/src first
sys.path.insert(0, str(HERE))
import asked_vs_got as X
import diffusion84 as M
from checkpoints import load_checkpoint
from ask_line import headline, line_measures
from label_noise import flagged_checkpoint, flagged_row
import heldout as HO
import matchup_db as MDB
import team84 as T
import top50 as T50
from corpus import parse_team_text
from hps_generate import Validator
from team_diversity import slots_from_paste
from team_diversity import summary as diversity_summary

MODELS = {
    "gen84_all_ep21": dict(checkpoint="stat84_all_teams_ep21", reader="Generated, all teams, 21 epochs", epochs=dict(all_teams=21),
                           stopped="owner stopped training at epoch 21 of a cap of 80; validation loss was still falling"),
    "gen84_all_noise_ep21": dict(checkpoint="stat84_all_teams_noise_ep21", reader="Label noise ±1/48, all teams, 21 epochs", epochs=dict(all_teams=21),
                                 stopped="planned stop at epoch 21, to match the run without label noise"),
    "gen84_same": dict(checkpoint="stat84_same_teams", reader="Generated, same teams, 80 and 60 epochs",
                       epochs=dict(loop_teams=80, battle_records=60), stopped="fixed epoch counts, as planned"),
}
BYEPOCH = "stat84_labelled_noise_byepoch_es3_cap80"
for _ep in range(1, 81):
    MODELS[f"gen84_noise_byepoch_ep{_ep:02d}"] = dict(checkpoint=f"{BYEPOCH}/{BYEPOCH}_ep{_ep:02d}", reader=f"Epoch {_ep}, labelled teams, label noise",
                                                      epochs=dict(labelled_teams=_ep), stopped="one training, a checkpoint per epoch", labelled=True)
for _t, _r in (("01", "0.1"), ("03", "0.3")):
    MODELS[f"gen84_ddom_tau{_t}"] = dict(checkpoint=f"stat84_labelled_noise_ddom_tau{_t}_ep21", reader=f"Tau {_r} weights, 21 epochs",
                                         epochs=dict(labelled_teams=21), stopped="planned stop at epoch 21, to match the control", labelled=True)
for _arm, _r in (("subset_10_39", "10 to 39 opponents per label"), ("subset_1_39", "1 to 39 opponents per label"),
                 ("subset_39", "All 39 opponents per label")):    # docs/subset-size-labels.md
    MODELS[f"gen65_{_arm}"] = dict(checkpoint=f"stat65_{_arm}", reader=_r, epochs=dict(shuffled_table="early stopping, cap 80, patience 3"),
                                   stopped="early stopping on validation rows", table65=True)
MODELS["ft_hindsight_relabel"] = dict(checkpoint="stat84_labelled_noise_byepoch_ep21_ft_hindsight_relabel_es3_cap40",   # issue #106, src/hindsight_relabel.py
                                      reader="Hindsight-relabeled fine-tune, epoch 21", epochs=dict(labelled_teams=21, relabelled_fine_tune="early stopping"),
                                      stopped="epoch 21 fine-tuned on its own battled teams, labelled with measured win rate; best pass by validation loss", labelled=True)
MODEL = os.environ.get("AG84_MODEL", "gen84_all_ep21")
if MODEL not in MODELS and flagged_checkpoint(MODEL):
    MODELS[MODEL] = flagged_row(MODEL, load_checkpoint(L.CK / f"{flagged_checkpoint(MODEL)}.pt", map_location="cpu")["info"])
SMOKE = bool(os.environ.get("AG84_SMOKE"))
GUIDANCE, KEEP, BATCH, MAX_TRIES, SEED0, BATTLES_PER_OPPONENT = 2.0, 64, 96, 1280, 7300, 1
ASK_NUMERATORS, SEEDS = list(range(1, 24)), [1, 2, 3]
if os.environ.get("AG84_SEEDS"): SEEDS = [int(s) for s in os.environ["AG84_SEEDS"].split(",")]
if SMOKE: KEEP, BATCH, MAX_TRIES, ASK_NUMERATORS, SEEDS = 4, 8, 64, [12], [1]
DIVERSITY_SUBSETS, SUBSET_SIZE = 200, 5
OFF_GRID_ASKS = [0.3, 0.531, 0.6, 0.7]                   # not multiples of 1/24; the 48-field runs battled these too
NUDGE_NUMERATORS, NUDGES = (6, 8, 10, 11, 12, 13, 14, 16), (0.0, 0.002, -0.002, 1 / 48)
if SMOKE: OFF_GRID_ASKS, NUDGE_NUMERATORS, NUDGES = [0.3], (11,), (0.0, 0.002)
RES = Path.home() / "vgc-data/results"
OUT = RES / f"stat_points_84_{MODEL}{'_smoke' if SMOKE else ''}.json"
DATA = X.RES / "lossdata84.npz"                          # as train84
UNIFORM_LEGAL = Path.home() / "vgc-data/teams/hps_reg_mb_100k.jsonl"


def load_out():
    return json.load(open(OUT)) if OUT.exists() else {"model": MODEL, "cells": {}}


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def tag_of(numerator, seed): return f"{MODEL}_a{numerator:02d}of24_s{seed}"


def checkpoint_path(): return L.CK / f"{MODELS[MODEL]['checkpoint']}.pt"


def load_model():
    return M.Model84.load(checkpoint_path())


def reason(err):
    """A rejection's first line with its Pokémon, move and item names removed, so counts group by kind of error."""
    return re.sub(r"\s+", " ", re.sub(r"[^:]*?(can't|cannot|has|is|must|needs)\b", r"\1", str(err).split("\n")[0])).strip()[:80]


def subset_diversity(pastes, seed):
    """Mean over random 5-team subsets: effective species, distinct line-ups and top-species share."""
    teams = [slots_from_paste(p) for p in pastes]
    species = [sorted(s["species"] for s in t) for t in teams]
    rng = np.random.default_rng(seed); eff, lineups, top = [], [], []
    size = min(SUBSET_SIZE, len(species))
    for _ in range(DIVERSITY_SUBSETS):
        pick = rng.choice(len(species), size, replace=False)
        counts = np.array(list(Counter(x for i in pick for x in species[i]).values()), float)
        share = counts / counts.sum()
        eff.append(float(np.exp(-(share * np.log(share)).sum()))); top.append(float(share.max()))
        lineups.append(len({tuple(species[i]) for i in pick}))
    return dict(effective_species=float(np.mean(eff)), distinct_lineups=float(np.mean(lineups)), top_species_share=float(np.mean(top)))


def draw(w, ask, seed):
    """Sample until KEEP teams pass Showdown and are not held-out teams, counting every try and every rejection."""
    torch.manual_seed(SEED0 + seed)
    val = Validator(); kept, stats, tried, rejected, heldout, reasons = [], [], 0, 0, 0, Counter()
    try:
        while len(kept) < KEEP and tried < MAX_TRIES:
            for r in w.sample(BATCH, ask=ask, guidance=GUIDANCE).cpu().numpy():
                tried += 1
                p, slots = w.to_paste(r, repair=True)
                err = val(p)
                if err is not None: rejected += 1; reasons[reason(err)] += 1
                elif HO.is_heldout_text(p): heldout += 1
                else: kept.append(p); stats.append([(s["species"], s["evs"]) for s in slots])
                if len(kept) >= KEEP or tried >= MAX_TRIES: break
    finally:
        val.close()
    return kept, stats, dict(tried=tried, rejected=rejected, heldout_dropped=heldout, legal=(tried - rejected) / max(tried, 1),
                             reasons=reasons.most_common(10))


def require_same_checkpoint(out):
    """Teams already drawn belong to the checkpoint that wrote them: never file them under different weights."""
    sha = hashlib.sha256(checkpoint_path().read_bytes()).hexdigest()
    if (out.get("cells") or out.get("off_grid")) and out.get("checkpoint_sha256") not in (None, sha):
        raise SystemExit(f"{MODEL}: this results file holds teams from another checkpoint; move it before drawing more")
    return sha


def gen():
    out = load_out(); sha = require_same_checkpoint(out)
    out.update(reader=MODELS[MODEL]["reader"], epochs_trained=MODELS[MODEL]["epochs"], stopped=MODELS[MODEL]["stopped"],
               checkpoint=checkpoint_path().name, checkpoint_sha256=sha,
               guidance=GUIDANCE, keep=KEEP, max_tries=MAX_TRIES, seed0=SEED0)
    save(out)
    w = load_model()
    for seed in SEEDS:
        for k in ASK_NUMERATORS:
            t = tag_of(k, seed)
            if t in out["cells"]: continue
            kept, stats, counts = draw(w, k / 24, seed)
            out["cells"][t] = dict(ask=k / 24, numerator=k, seed=seed, **counts, pastes=kept, stat_points=stats,
                                   div=subset_diversity(kept, SEED0 + seed) if kept else None)
            save(out)
            print(f"  {t:32s} kept {len(kept):3d} · legal {counts['legal']:6.1%} · eff species "
                  f"{(out['cells'][t]['div'] or {}).get('effective_species', 0):.1f}", flush=True)


def require_shuffled():
    if MDB.SLOT_ORDER != "shuffled": raise SystemExit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")


def complete(scores):
    return all(s["score"] is not None and s["coverage"] == s["columns"] and s["battles"] >= BATTLES_PER_OPPONENT * s["columns"] for s in scores)


def record_scores(cell, scores):
    cell["y"] = [s["score"] for s in scores]; cell["battles"] = sum(s["battles"] for s in scores)
    cell["team_ids"] = [s["team_id"] for s in scores]


def battle():
    require_shuffled(); con = MDB.connect(); out = load_out()
    for t, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=BATTLES_PER_OPPONENT, seed=SEED0 + c["seed"], origin=f"statpoints84:{t}")
        if not complete(sc): print(f"  {t} INCOMPLETE: left for the next run", flush=True); continue
        record_scores(c, sc); save(out); print(f"  {t:32s} win {np.mean(c['y']):.3f}", flush=True)


def offgrid():
    """Generate and battle the asks that are not multiples of 1/24, kept apart from the 23-ask grid."""
    require_shuffled(); con = MDB.connect(); out = load_out(); require_same_checkpoint(out); w = load_model(); cells = out.setdefault("off_grid", {})
    for seed in SEEDS:
        for ask in OFF_GRID_ASKS:
            t = f"{MODEL}_a{ask}_s{seed}"
            if "y" in cells.get(t, {}): continue
            if t not in cells:
                kept, stats, counts = draw(w, ask, seed)
                cells[t] = dict(ask=ask, seed=seed, **counts, pastes=kept, stat_points=stats); save(out)
            sc = MDB.score_pastes(con, cells[t]["pastes"], per_cell=BATTLES_PER_OPPONENT, seed=SEED0 + seed, origin=f"statpoints84:{t}")
            if not complete(sc): print(f"  {t} INCOMPLETE: left for the next run", flush=True); continue
            record_scores(cells[t], sc); save(out); print(f"  {t:36s} win {np.mean(cells[t]['y']):.3f}", flush=True)


def nudge():
    """Share of Pokémon with fewer than four moves at k/24 and just beside it: sampling only, no battles."""
    out = load_out(); w = load_model(); E = w.V.stoi["move"][""]; rows = {}
    for k in NUDGE_NUMERATORS:
        for d in NUDGES:
            r = w.sample(16 if SMOKE else 256, ask=k / 24 + d, guidance=GUIDANCE, seed=SEED0).cpu().numpy()
            moves = r[:, :T.COLS48].reshape(len(r), 6, -1)[:, :, 3:7]
            rows[f"{k}/24{d:+.4f}"] = float((moves == E).any(-1).mean())
            print(f"  ask {k}/24 {d:+.4f}: {rows[f'{k}/24{d:+.4f}']:.3f}", flush=True)
    out["nudge"] = rows; save(out)


def win_with_interval(y):
    """Mean win rate with a 95% interval over teams (each team's battles form one cluster)."""
    se = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else float("nan")
    return dict(teams=len(y), win=float(y.mean()), lo=float(y.mean() - 1.96 * se), hi=float(y.mean() + 1.96 * se))


def mean_distance(cells):
    return float(diversity_summary([slots_from_paste(p) for c in cells for p in c["pastes"]])["mean_distance"])


def off_grid_table(cells):
    """Per ask off the grid: win rate, line-ups among all that ask's teams, and mean slot-matched distance."""
    rows = []
    for ask in sorted({c["ask"] for c in cells}):
        at = [c for c in cells if c["ask"] == ask]
        y = np.array([v for c in at for v in c.get("y", []) if v is not None], float)
        if not len(y): continue
        d = diversity_summary([slots_from_paste(p) for c in at for p in c["pastes"]])
        rows.append(dict(ask=ask, **win_with_interval(y), distinct_lineups=d["distinct_lineups"], mean_distance=float(d["mean_distance"])))
    return rows


def ask_table(cells):
    """Per ask: win rate with a 95% interval, legality, diversity, and mean slot-matched distance."""
    rows = []
    for k in sorted({c["numerator"] for c in cells}):
        at = [c for c in cells if c["numerator"] == k]
        y = np.array([v for c in at for v in c.get("y", []) if v is not None], float)
        if not len(y): continue
        rows.append(dict(numerator=k, ask=k / 24, **win_with_interval(y), mean_distance=mean_distance(at),
                         legal=sum(c["tried"] - c["rejected"] for c in at) / max(sum(c["tried"] for c in at), 1),
                         **{m: float(np.mean([c["div"][m] for c in at if c["div"]])) for m in ("effective_species", "distinct_lineups", "top_species_share")}))
    return rows


def stat_point_measures(cells, corpus_sets):
    slots = [(T.norm(sp), ev) for c in cells for team in c["stat_points"] for sp, ev in team]   # corpus_sets holds normalised species
    if not slots: return {}
    key = lambda ev: tuple(sorted(ev.items()))
    per_species = {}
    for sp, ev in slots: per_species.setdefault(sp, set()).add(key(ev))
    moves = [len(s["moves"]) for c in cells for p in c["pastes"] for s in slots_from_paste(p)]
    return dict(pokemon=len(slots), share_spending_66=float(np.mean([sum(ev.values()) == 66 for _, ev in slots])),
                mean_stat_points_spent=float(np.mean([sum(ev.values()) for _, ev in slots])),
                share_under_four_moves=float(np.mean([m < 4 for m in moves])),
                mean_distinct_sets_per_species=float(np.mean([len(v) for v in per_species.values()])),
                share_matching_corpus_set=float(np.mean([(sp, key(ev)) in corpus_sets for sp, ev in slots])))


def corpus_stat_sets(V):
    sets = set()
    for team in X.setup()[0]:
        try: row = T.encode(V, team)
        except ValueError: continue
        for s in T.decode(V, row): sets.add((s["species"], tuple(sorted(s["evs"].items()))))
    return sets


def report():
    out = load_out(); cells = list(out["cells"].values()); w = load_model()
    rows = ask_table(cells); reasons = Counter()
    for c in cells: reasons.update(dict(c["reasons"]))
    summary = dict(asks=rows, off_grid=off_grid_table(list(out.get("off_grid", {}).values())), line=line_measures(rows), stat_points=stat_point_measures(cells, corpus_stat_sets(w.V)),
                   legal=sum(c["tried"] - c["rejected"] for c in cells) / max(sum(c["tried"] for c in cells), 1),
                   tries=sum(c["tried"] for c in cells), teams=sum(len(c["pastes"]) for c in cells),
                   battles=sum(c.get("battles", BATTLES_PER_OPPONENT * 49 * len(c.get("y", []))) for c in cells), rejections=reasons.most_common(10),
                   win_rate_definition="against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, one battle per opponent")
    out["summary"] = summary; save(out)
    print(f"{out['reader']} ({MODEL}): {summary['teams']} teams, {summary['battles']} battles, legal {summary['legal']:.1%}")
    print(f"{'ask':>8s} {'win':>6s} {'95% interval':>17s} {'legal':>7s} {'eff species':>12s} {'line-ups':>9s} {'top share':>10s}")
    for r in rows:
        print(f"{r['numerator']:>5d}/24 {r['win']:6.3f}   [{r['lo']:.3f}, {r['hi']:.3f}] {r['legal']:7.1%} {r['effective_species']:12.1f} {r['distinct_lineups']:9.2f} {r['top_species_share']:10.3f}")
    line = summary["line"]
    print(headline(line))
    print("line:", json.dumps(line)); print("Stat Points:", json.dumps(summary["stat_points"]))
    print("AG84_REPORT_DONE")


# ---------------------------------------------------------------- preflight (run-experiment step 2)
def slot_key(s):
    return (s["species"], s["ability"], s["item"], tuple(sorted(s["moves"])), tuple(sorted(s["evs"].items())))


def round_trip(w, rows):
    """Every generated field reaches the paste: re-read the paste, compare everything except the repaired Stat Alignment."""
    bad = 0
    for r in rows:
        p, _ = w.to_paste(r, repair=True)
        back = T.decode(w.V, T.encode(w.V, parse_team_text(p)))
        if sorted(map(str, map(slot_key, T.decode(w.V, r)))) != sorted(map(str, map(slot_key, back))): bad += 1
    return dict(teams=len(rows), mismatched=bad, passed=bad == 0)


def shares(field, slot_lists, top=10):
    c = Counter(x for team in slot_lists for s in team for x in (s[field] if field == "moves" else [s[field]]))
    n = sum(c.values()); return {k: round(v / n, 4) for k, v in c.most_common(top)}


def separation(V, train_keys):
    held = 0; overlap = 0
    for path in HO.files():
        team = parse_team_text(Path(path).read_text())
        try: row = T.encode(V, team)
        except ValueError: continue
        held += 1; overlap += team_key(row) in train_keys
    return dict(heldout_teams_encoded=held, overlap_with_training=overlap, passed=overlap == 0)


def team_key(row):
    return b"".join(sorted(slot_bytes(row)))


def slot_bytes(row):
    """Six slot keys: each Pokémon's 8 fields plus its 6 Stat Points."""
    r = np.asarray(row, np.int64); a, b = r[:T.COLS48].reshape(6, -1), r[T.COLS48:].reshape(6, -1)
    return [np.concatenate([a[i], b[i]]).tobytes() for i in range(6)]


def copies(rows, train):
    keys = {team_key(r) for r in train}
    index = {}
    for i, r in enumerate(train):
        for sb in slot_bytes(r): index.setdefault(sb, []).append(i)
    shared = []
    for r in rows[:64]:
        c = Counter(j for sb in set(slot_bytes(r)) for j in index.get(sb, []))
        shared.append(max(c.values()) if c else 0)
    identical = sum(team_key(r) in keys for r in rows)
    return keys, dict(generated=len(rows), identical_to_training=identical, share_identical=identical / len(rows),
                      mean_shared_slots_nearest=float(np.mean(shared)), max_shared_slots_nearest=int(max(shared)))


def controls():
    """Battle known teams through the real scorer in a private database."""
    if "MATCHUP_DB" not in os.environ: raise SystemExit("run preflight with MATCHUP_DB pointing at a private database")
    require_shuffled(); con = MDB.connect(); val = Validator()
    try:
        with con:
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):
                tid = MDB.add_team(con, Path(T50.resolve(t["id"])).read_text(), "vgcpastes", val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) DO UPDATE SET weight = weight + 1", (tid,))
    finally:
        val.close()
    meta = [Path(T50.resolve(t["id"])).read_text() for t in json.load(open(T50.TOP50))[:6]]
    uniform = [json.loads(line)["team"] for _, line in zip(range(6), open(UNIFORM_LEGAL))]
    y = lambda ps, o: [s["score"] for s in MDB.score_pastes(con, ps, per_cell=1, seed=SEED0, origin=o)]
    a, b = y(meta, "preflight:meta"), y(uniform, "preflight:uniform")
    return dict(top50_win=float(np.mean(a)), uniform_legal_win=float(np.mean(b)), teams_each=6,
                passed=bool(0.35 < np.mean(a) < 0.65 and np.mean(b) < 0.12))


def preflight():
    w = load_model(); V = w.V; out = load_out(); pf = {}
    if MODELS[MODEL].get("table65"):                        # trained on the shuffled-slot table (src/train65.py)
        import train65
        d, tr_rows, _, _ = train65.load(); tr = d["X"][tr_rows].astype(np.int64)
    else:
        d = np.load(DATA); tr_rows = ~d["all_val"] & (~np.isnan(d["all_W"]) if MODELS[MODEL].get("labelled") else True)
        tr = d["all_X"][tr_rows].astype(np.int64)
    torch.manual_seed(SEED0 + 1)
    rows = w.sample(256, ask=0.5, guidance=GUIDANCE).cpu().numpy()
    pf["round_trip"] = round_trip(w, rows[:20])
    pf["coverage"] = dict(vocabulary="regulation tables directly", n_train=(w.info or {}).get("n_train"), dropped_by_tables=(w.info or {}).get("dropped_by_tables"),
                          explanation="teams the rule tables reject are dropped before training")
    gen_slots = [T.decode(V, r) for r in rows]
    rng = np.random.default_rng(0); real_slots = [T.decode(V, tr[i]) for i in rng.choice(len(tr), 256, replace=False)]
    pf["read_teams"] = dict(pastes=[w.to_paste(r, repair=True)[0] for r in rows[:20]],
                            **{f"{f}_generated": shares(f, gen_slots) for f in ("species", "item", "ability", "moves")},
                            **{f"{f}_training": shares(f, real_slots) for f in ("species", "item", "ability", "moves")})
    pf["training_smoke"] = dict(skipped="the checkpoint is already trained; the loss curve is in the training log and the tracer covers the pipeline")
    pf["controls"] = controls()
    pf["invariance"] = dict(skipped="no new scorer; every battle shuffles slots and require_shuffled refuses any other order")
    pf["repeat"] = dict(skipped="scorer unchanged since the line-up repeats run; the seed reaches the schedule through score_pastes as there")
    keys, pf["copies"] = copies(rows, tr)
    pf["separation"] = separation(V, keys)
    torch.manual_seed(SEED0); low = w.sample(128, ask=0.1, guidance=GUIDANCE).cpu().numpy()
    torch.manual_seed(SEED0); high = w.sample(128, ask=0.7, guidance=GUIDANCE).cpu().numpy()
    a, b = shares("species", [T.decode(V, r) for r in low], 8), shares("species", [T.decode(V, r) for r in high], 8)
    pf["condition"] = dict(share_teams_changed=float(np.mean((low != high).any(1))), species_low_ask=a, species_high_ask=b, passed=bool((low != high).any(1).mean() > 0.5),
                           note="direction is measured by the grid itself")
    out["preflight"] = pf; save(out)
    print(json.dumps({k: {x: y for x, y in v.items() if x != "pastes"} for k, v in pf.items()}, indent=1)[:6000])
    print("\n\n---\n\n".join(pf["read_teams"]["pastes"]))
    print("AG84_PREFLIGHT_DONE", "PASS" if all(v.get("passed", True) for v in pf.values()) else "FAIL")


if __name__ == "__main__":
    {"preflight": preflight, "gen": gen, "battle": battle, "offgrid": offgrid, "nudge": nudge, "report": report}[sys.argv[1]]()
