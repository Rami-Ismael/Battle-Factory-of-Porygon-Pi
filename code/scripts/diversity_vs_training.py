"""Is the 80-epoch labelled model's output as diverse as its training teams? (plan: notes/Diversity.md and its page)

Five samples of n = 1,088 teams each (17 asks x 64 teams), every team as its 84 fields:
  model    the saved ask-grid teams of gen84_noise_byepoch_ep80, asks 1/24 to 17/24, one sample per seed (1, 2, 3)
  train    training teams (lossdata84 all_X, labelled, not in the 2% validation split), 64 per ask group, a team's
           group being the ask nearest its label, so the label mix matches the model's; one sample per seed
  copied   one training team repeated n times (low anchor: every measure at its minimum)
  random   n random legal teams from hps_generate.one_team (random legal sets, 66 Stat Points placed at random),
           each checked by Showdown (high anchor)
  val      labelled validation-split teams, the same 64 per ask group (real teams the model never trained on)
Noise floor: 20 pairs of disjoint training samples drawn the same way; a measure's floor is the 2.5% to 97.5% range of
the 20 differences. The model differs from its training teams on a measure when all three seed differences share a
sign and each lies outside that range.

Measures per team level (line-up = six species; sets = species, ability, item, nature, moves, 48 fields;
Stat Points = 36 fields; team = all 84). Similarity of two teams at a level is the cosine of their feature sets, every
feature tied to its species, so the same item on two different species is not shared:
  vendi      exp(entropy of the eigenvalues of K/n): the effective number of different teams, 1 to n (Friedman & Dieng)
  dcscore    trace(softmax(K/0.1)) row-wise, 1 to n (Zhu et al. 2025); tau fixed before any run
  distance   1 - mean off-diagonal similarity (embedding diversity, Tevet & Berant 2021)
plus exact counts (distinct and effective line-ups, species, species pairs, sets, Stat Points per Pokémon, teams) and
novelty against the full training set (share of teams whose line-up, 48 fields or 84 fields appear in it).

Added 2026-11-11 (issue #264), on every sample and every floor pair:
  train_all  training teams drawn at random over all win rates (labelled, not validation), one sample per seed
  builds per species at EQUAL_USES uses, also on the species both sides use that often (builds_shared_species)
  Novelty: share of teams 0, 1, 2 or 3+ species swaps from the closest training line-up
  Nearest training distance: fields out of 84 that differ from the closest training team, slots matched
  archetype shares, setters only (ability, move, or a mega stone whose mega form has the setting ability)
A sample drawn from the training teams leaves each team out of its own reference once.

    python diversity_vs_training.py run [--out PATH]      (with /tmp/vgc-pilot/.venv/bin/python)
"""
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
RES = Path.home() / "vgc-data/results"
GEN = RES / "stat_points_84_gen84_noise_byepoch_ep80.json"
OUT = RES / "diversity_vs_training_2026-11-11.json"     # the 11-09 result is diversity_vs_training.json
EQUAL_USES = 30          # every model seed has 36 to 40 species used at least 30 times out of 6,528 slots
TABLE = Path.home() / "Documents/code/vgc-team-generator-pilot/results/lossdata84.npz"
TABLE_SHA = "02463cbd812499e7bd94ffb4faff505318f86e58d2fc41ba57476d21b03e11c7"
GRID, ASKS, PER_ASK, SEEDS, FLOOR_PAIRS, TAU, SEED0 = 24, range(1, 18), 64, (1, 2, 3), 20, 0.1, 20261109
LEVELS = ("lineup", "sets", "statpoints", "team")
STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


# ---------------------------------------------------------------- measures (pure: a team is six slot dicts)
def features(team, level):
    """Species-anchored features of one team at one level."""
    f = set()
    for s in team:
        sp = s["species"]
        if level in ("lineup", "sets", "team"): f.add((sp,))
        if level in ("sets", "team"):
            f |= {(sp, "ability", s["ability"]), (sp, "item", s["item"]), (sp, "nature", s["nature"])}
            f |= {(sp, "move", m) for m in s["moves"]}
        if level in ("statpoints", "team"):
            f |= {(sp, st, s["evs"][st]) for st in STATS}
    return f


def kernel(teams, level):
    """n x n cosine similarity of the teams' feature sets."""
    fs = [features(t, level) for t in teams]
    ids = {}
    X = np.zeros((len(fs), len({x for f in fs for x in f})))
    for i, f in enumerate(fs):
        X[i, [ids.setdefault(x, len(ids)) for x in f]] = 1
    X /= np.sqrt(X.sum(1, keepdims=True))
    return np.clip(np.round(X @ X.T, 12), 0, 1)


def vendi(K):
    lam = np.clip(np.linalg.eigvalsh(K / len(K)), 0, None)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def dcscore(K, tau=TAU):
    Z = (K - K.max(1, keepdims=True)) / tau
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return float(np.trace(P))


def mean_distance(K):
    n = len(K)
    return float(1 - (K.sum() - np.trace(K)) / (n * (n - 1)))


def effective(counter):
    c = np.array(list(counter.values()), float); p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def lineup(t): return tuple(sorted(s["species"] for s in t))
def set_key(s): return (s["species"], s["ability"], s["item"], s["nature"], tuple(sorted(s["moves"])))
def statpoints_key(s): return (s["species"], *tuple(s["evs"][st] for st in STATS))
def key48(t): return tuple(sorted(set_key(s) for s in t))
def key84(t): return tuple(sorted(set_key(s) + statpoints_key(s)[1:] for s in t))


def counts(teams):
    lu = Counter(lineup(t) for t in teams)
    sp = Counter(s["species"] for t in teams for s in t)
    pairs = Counter(p for t in teams for p in itertools.combinations(lineup(t), 2))
    sets = Counter(set_key(s) for t in teams for s in t)
    statpoints = Counter(statpoints_key(s) for t in teams for s in t)
    return dict(distinct_lineups=len(lu), effective_lineups=effective(lu), largest_lineup_share=max(lu.values()) / len(teams),
                distinct_species=len(sp), effective_species=effective(sp),
                distinct_pairs=len(pairs), effective_pairs=effective(pairs),
                distinct_sets=len(sets), sets_per_species=len(sets) / len(sp), effective_sets=effective(sets),
                distinct_statpoints=len(statpoints), effective_statpoints=effective(statpoints),
                top_lineup=list(lu.most_common(1)[0][0]), top_lineup_count=lu.most_common(1)[0][1],
                distinct_teams48=len({key48(t) for t in teams}), distinct_teams84=len({key84(t) for t in teams}))


def measure(teams):
    out = counts(teams)
    for lv in LEVELS:
        K = kernel(teams, lv)
        out.update({f"vendi_{lv}": vendi(K), f"dcscore_{lv}": dcscore(K), f"distance_{lv}": mean_distance(K)})
    return out


def novelty(teams, ref):
    n = len(teams)
    return dict(lineup_in_train=sum(lineup(t) in ref["lineup"] for t in teams) / n,
                sets48_in_train=sum(key48(t) in ref["k48"] for t in teams) / n,
                team84_in_train=sum(key84(t) in ref["k84"] for t in teams) / n)


def species_swaps(teams, ref_lineups, exclude_self=False):
    """Share of teams at each count of species swaps to the nearest reference line-up (six minus shared species).

    ref_lineups must keep its duplicates (one entry per reference team, never a set). With exclude_self, a reference
    line-up identical to the team's own counts only if it appears at least twice (one copy is the team itself).
    A line-up never repeats a species (species clause), so shared species = size of the intersection.
    """
    copies = Counter(ref_lineups)
    refs = list(copies)
    species = sorted({sp for ref in refs for sp in ref} | {s["species"] for t in teams for s in t})
    col = {sp: j for j, sp in enumerate(species)}
    R = np.zeros((len(refs), len(species)), np.float32)
    for i, ref in enumerate(refs): R[i, [col[sp] for sp in ref]] = 1
    row_of = {ref: i for i, ref in enumerate(refs)}
    buckets = Counter()
    for start in range(0, len(teams), 256):
        chunk = teams[start:start + 256]
        Q = np.zeros((len(chunk), len(species)), np.float32)
        for i, t in enumerate(chunk): Q[i, [col[s["species"]] for s in t]] = 1
        shared = Q @ R.T
        for i, t in enumerate(chunk):
            own = row_of.get(lineup(t))
            if exclude_self and own is not None and copies[refs[own]] == 1: shared[i, own] = -1
            swaps = 6 - max(int(np.rint(shared[i].max())), 0) if len(refs) else 6
            buckets["swaps_3plus" if swaps >= 3 else f"swaps_{swaps}"] += 1
    n = len(teams)
    return {name: buckets[name] / n for name in ("swaps_0", "swaps_1", "swaps_2", "swaps_3plus")}


# A set stores its base ability, so a mega stone whose mega form has a setting ability is listed as an item: in the
# training teams 4,925 of 5,202 Charizard holding Charizardite Y store Blaze, not Drought (sample of 20,000, 11-11).
# Mega abilities from code/data/pokedex.json; Meganiumite's Mega Sol is left out (it sets no weather on the field).
ARCHETYPE_SETTERS = {
    "sun": dict(abilities=("Drought", "Orichalcum Pulse", "Desolate Land"), moves=("Sunny Day",), items=("Charizardite Y",)),
    "rain": dict(abilities=("Drizzle", "Primordial Sea"), moves=("Rain Dance",), items=()),
    "sand": dict(abilities=("Sand Stream", "Sand Spit"), moves=("Sandstorm",), items=("Tyranitarite",)),
    "snow": dict(abilities=("Snow Warning",), moves=("Snowscape", "Chilly Reception", "Hail"),
                 items=("Abomasite", "Froslassite")),
    "electric terrain": dict(abilities=("Electric Surge", "Hadron Engine"), moves=("Electric Terrain",),
                             items=("Raichunite X",)),
    "grassy terrain": dict(abilities=("Grassy Surge", "Seed Sower"), moves=("Grassy Terrain",), items=()),
    "misty terrain": dict(abilities=("Misty Surge",), moves=("Misty Terrain",), items=()),
    "psychic terrain": dict(abilities=("Psychic Surge",), moves=("Psychic Terrain",), items=()),
    "Trick Room": dict(abilities=(), moves=("Trick Room",), items=()),
    "Tailwind": dict(abilities=(), moves=("Tailwind",), items=()),
}


def comparable(name):
    """Lowercased name with every character that is not a letter or digit removed."""
    return "".join(c for c in name.lower() if c.isalnum())


def archetype_shares(teams):
    """Share of teams in each archetype (a team with any setting ability or move), plus "none"."""
    setters = {arch: ({comparable(x) for x in parts["abilities"] + parts["moves"]}, {comparable(x) for x in parts["items"]})
               for arch, parts in ARCHETYPE_SETTERS.items()}
    hits_per_team = Counter()
    for t in teams:
        hits = set()
        for s in t:
            ability, item, moves = comparable(s["ability"]), comparable(s["item"]), {comparable(m) for m in s["moves"]}
            for arch, (words, stones) in setters.items():
                if ability in words or moves & words or item in stones: hits.add(arch)
        hits_per_team.update(hits or {"none"})
    n = len(teams)
    return {name: hits_per_team[name] / n for name in [*ARCHETYPE_SETTERS, "none"]}


def uses_per_species(teams):
    return Counter(s["species"] for t in teams for s in t)


def builds_at_equal_uses(teams, uses, rng, draws=20, species=None):
    """Distinct builds (set_key) among `uses` sets drawn without replacement, per species used at least `uses` times.

    Averaged over `draws` repeats and then over species, so species with more uses are not favoured. Species are
    visited in sorted order, so the result is reproducible for a given rng. `species` restricts the average to a
    given list, so two samples can be compared on the same species (otherwise the species mix differs between them).
    """
    sets_of = {}
    for t in teams:
        for s in t: sets_of.setdefault(s["species"], []).append(set_key(s))
    allowed = set(sets_of) if species is None else set(species) & set(sets_of)
    qualifying = [sp for sp in sorted(allowed) if len(sets_of[sp]) >= uses]
    if not qualifying: return dict(builds_per_species=float("nan"), species_counted=0)
    per_species = []
    for sp in qualifying:
        sets = sets_of[sp]
        distinct = [len({sets[i] for i in rng.choice(len(sets), uses, replace=False)}) for _ in range(draws)]
        per_species.append(np.mean(distinct))
    return dict(builds_per_species=float(np.mean(per_species)), species_counted=len(qualifying))


# ---------------------------------------------------------------- data
def modules():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(SRC))
    import lossdown  # noqa: F401  puts /tmp/vgc-pilot/src first again
    sys.path.insert(0, str(SRC))


def nearest_ask(w):
    out = np.full(w.shape, -1, np.int64); ok = np.isfinite(w)
    out[ok] = np.floor(w[ok] * GRID + 0.5).astype(np.int64)
    return out


def stratified(groups, rng, exclude=()):
    """PER_ASK row indices per ask group, without replacement, avoiding `exclude`."""
    pick = []
    for k in ASKS:
        pool = np.setdiff1d(groups[k], exclude) if len(exclude) else groups[k]
        pick.append(rng.choice(pool, min(PER_ASK, len(pool)), replace=False))
    return np.concatenate(pick)


def load():
    modules()
    import team84 as T
    from corpus import parse_team_text
    if hashlib.sha256(TABLE.read_bytes()).hexdigest() != TABLE_SHA: raise SystemExit("lossdata84.npz changed")
    V = T.regulation_vocab()
    z = np.load(TABLE)
    X, W, val = z["all_X"], z["all_W"], z["all_val"]
    ask = nearest_ask(W)
    train = np.flatnonzero(~val & np.isfinite(W)); vrows = np.flatnonzero(val & np.isfinite(W))
    tgroups = {k: train[ask[train] == k] for k in ASKS}; vgroups = {k: vrows[ask[vrows] == k] for k in ASKS}
    dec = lambda i: T.decode(V, X[i])
    for i in train[:200]:                                    # decode/encode round trip on real rows
        if not np.array_equal(T.encode(V, dec(i)), X[i].astype(np.int64)): raise SystemExit(f"row {i} does not round-trip")

    def from_paste(p):
        return T.decode(V, T.encode(V, parse_team_text(p)))

    gen = json.load(open(GEN))
    model, dropped = {}, Counter()
    for s in SEEDS:
        teams = []
        for k in ASKS:
            c = gen["cells"][f"gen84_noise_byepoch_ep80_a{k:02d}of24_s{s}"]
            if len(c["pastes"]) != PER_ASK: raise SystemExit(f"cell a{k}s{s} has {len(c['pastes'])} teams")
            for p in c["pastes"]:
                try: teams.append(from_paste(p))
                except ValueError: dropped[f"model_s{s}"] += 1
        model[s] = teams
    return T, V, X, train, tgroups, vgroups, dec, model, dropped


def random_teams(V, T, n, seed):
    import hps_generate as H
    rng = np.random.default_rng(seed); tables = H.build_tables(); val = H.Validator(); out, tries, outside = [], 0, 0
    try:
        while len(out) < n and tries < 40 * n:
            tries += 1
            sl = H.one_team(tables, rng)
            if sl is None or val("\n\n".join(H.slot_to_text(s) for s in sl) + "\n") is not None: continue
            try: out.append(T.decode(V, T.encode(V, sl)))
            except ValueError: outside += 1                # a field outside the Regulation M-B vocabulary
    finally:
        val.close()
    return out, dict(tries=tries, outside_vocab=outside)


def added_measures(teams, rows, from_training, ref_lineups, ref_rows, rng):
    """The 11-11 additions for one sample, flat: builds per species at equal uses, Novelty shares, Nearest training
    distance summary, archetype shares. from_training: the sample was drawn from the reference, so each team is
    left out of its own reference once (exclude_self, exclude_identical_once)."""
    from nearest_training_distance import nearest_training_distance, summarize_nearest_training_distance
    b = builds_at_equal_uses(teams, EQUAL_USES, rng)
    out = dict(builds_per_species=b["builds_per_species"], builds_species_counted=b["species_counted"])
    out.update({f"novelty_{k}": v for k, v in species_swaps(teams, ref_lineups, exclude_self=from_training).items()})
    dist = nearest_training_distance(rows, ref_rows, exclude_identical_once=from_training)
    out.update({f"nearest_{k}": v for k, v in summarize_nearest_training_distance(dist).items()})
    out.update({f"archetype_{k}": v for k, v in archetype_shares(teams).items()})
    return out


def builds_on_shared_species(a, b, rng):
    """Builds per species at EQUAL_USES for two samples, over the species both use at least EQUAL_USES times."""
    ua, ub = uses_per_species(a), uses_per_species(b)
    common = sorted(sp for sp in ua if ua[sp] >= EQUAL_USES and ub.get(sp, 0) >= EQUAL_USES)
    ba, bb = builds_at_equal_uses(a, EQUAL_USES, rng, species=common), builds_at_equal_uses(b, EQUAL_USES, rng, species=common)
    return ba["builds_per_species"], bb["builds_per_species"], len(common)


def setter_vocabulary(V, teams_by_sample):
    """Which listed setters exist in the Regulation M-B vocabulary, and how often a mega stone holder stores the
    mega form's ability."""
    vocab = {k: {comparable(x) for x in V.itos[k]} for k in ("ability", "move", "item")}
    listed = {arch: {kind: {x: comparable(x) in vocab[dict(abilities="ability", moves="move", items="item")[kind]]
                            for x in parts[kind]} for kind in parts} for arch, parts in ARCHETYPE_SETTERS.items()}
    mega_ability = {"charizarditey": "drought", "tyranitarite": "sandstream", "abomasite": "snowwarning",
                    "froslassite": "snowwarning", "raichunitex": "electricsurge"}
    stored = {}
    for name, teams in teams_by_sample.items():
        held = Counter(); same = Counter()
        for t in teams:
            for s in t:
                stone = comparable(s["item"])
                if stone in mega_ability:
                    held[stone] += 1; same[stone] += comparable(s["ability"]) == mega_ability[stone]
        stored[name] = {st: dict(holders=held[st], stores_mega_ability=same[st]) for st in sorted(held)}
    return dict(listed_in_vocabulary=listed, mega_stone_holders=stored)


# ---------------------------------------------------------------- run
def run(out_path=OUT):
    T, V, X, train, tgroups, vgroups, dec, model, dropped = load()
    n = len(ASKS) * PER_ASK
    rng = np.random.default_rng(SEED0)
    rng_added = np.random.default_rng(SEED0 + 1111)     # the additions draw here, so the 11-09 samples are unchanged
    ref = dict(lineup=set(), k48=set(), k84=set())
    ref_lineups = []
    for i in train:
        t = dec(i); ref["lineup"].add(lineup(t)); ref["k48"].add(key48(t)); ref["k84"].add(key84(t))
        ref_lineups.append(lineup(t))
    ref_rows = X[train].astype(np.int64)
    print(f"training reference: {len(train)} teams, {len(ref['lineup'])} line-ups", flush=True)

    samples, rows_of = {f"model_s{s}": model[s] for s in SEEDS}, {}
    for s in SEEDS:
        idx = stratified(tgroups, rng); samples[f"train_s{s}"] = [dec(i) for i in idx]; rows_of[f"train_s{s}"] = idx
    pick = rng.choice(train); samples["copied"] = [dec(pick)] * n; rows_of["copied"] = np.full(n, pick)
    idx = stratified(vgroups, rng); samples["val"] = [dec(i) for i in idx]; rows_of["val"] = idx
    samples["random"], rinfo = random_teams(V, T, n, SEED0)
    for s in SEEDS:                                     # training teams drawn over all win rates, not ask-matched
        idx = np.random.default_rng(SEED0 + 100 + s).choice(train, n, replace=False)
        samples[f"train_all_s{s}"] = [dec(i) for i in idx]; rows_of[f"train_all_s{s}"] = idx
    from_training = {k for k in samples if k.startswith("train") or k == "copied"}
    out = dict(n=n, tau=TAU, sizes={k: len(v) for k, v in samples.items()}, dropped=dict(dropped), random=rinfo,
               equal_uses=EQUAL_USES,
               self_excluded=dict(samples=sorted(from_training), note=(
                   "Rows drawn from the training teams leave each team out of its own reference once: Novelty "
                   "(species_swaps exclude_self) and Nearest training distance (exclude_identical_once). The 11-09 "
                   "novelty block does not exclude.")),
               measures={}, novelty={}, floor={}, verdict={}, verdict_all_win_rates={}, builds_shared_species={})
    for k, teams in samples.items():
        rows = X[rows_of[k]].astype(np.int64) if k in rows_of else np.stack([T.encode(V, t) for t in teams])
        out["measures"][k] = measure(teams)
        out["measures"][k].update(added_measures(teams, rows, k in from_training, ref_lineups, ref_rows, rng_added))
        if not k.startswith("train"): out["novelty"][k] = novelty(teams, ref)
        m = out["measures"][k]
        print(f"{k:12s} n={len(teams)} vendi team {m['vendi_team']:.1f} line-up {m['vendi_lineup']:.1f} builds/species "
              f"{m['builds_per_species']:.2f} nearest {m['nearest_mean']:.1f}", flush=True)
    for s in SEEDS:
        for other in (f"train_s{s}", f"train_all_s{s}"):
            bm, bt, c = builds_on_shared_species(samples[f"model_s{s}"], samples[other], rng_added)
            out["builds_shared_species"][f"model_s{s}_vs_{other}"] = dict(model=bm, training=bt, diff=bm - bt, species=c)
    out["setter_vocabulary"] = setter_vocabulary(V, {k: samples[k] for k in ("model_s1", "train_s1", "train_all_s1")})

    diffs = []
    for p in range(FLOOR_PAIRS):
        a = stratified(tgroups, rng); b = stratified(tgroups, rng, exclude=a)
        ta, tb = [dec(i) for i in a], [dec(i) for i in b]
        ma, mb = measure(ta), measure(tb)
        ma.update(added_measures(ta, X[a].astype(np.int64), True, ref_lineups, ref_rows, rng_added))
        mb.update(added_measures(tb, X[b].astype(np.int64), True, ref_lineups, ref_rows, rng_added))
        d = {m: ma[m] - mb[m] for m in ma if not isinstance(ma[m], list)}
        x, y, _ = builds_on_shared_species(ta, tb, rng_added); d["builds_shared_species"] = x - y
        diffs.append(d)
        print(f"floor pair {p + 1}/{FLOOR_PAIRS}", flush=True)
    verdicts(out, diffs)
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {out_path}")


def verdicts(out, diffs):
    """Noise floor per measure from the training pair differences, then the model's verdict against the ask-matched
    training rows (verdict) and the all-win-rate training rows (verdict_all_win_rates)."""
    for m in diffs[0]:
        v = np.array([d[m] for d in diffs])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out["floor"][m] = dict(lo=float(lo), hi=float(hi), sd=float(v.std(ddof=1)))
        for key, ref_row in (("verdict", "train_s"), ("verdict_all_win_rates", "train_all_s")):
            if m == "builds_shared_species":
                d = [out["builds_shared_species"][f"model_s{s}_vs_{ref_row}{s}"]["diff"] for s in SEEDS]
            else:
                d = [out["measures"][f"model_s{s}"][m] - out["measures"][f"{ref_row}{s}"][m] for s in SEEDS]
            outside = all(x < lo or x > hi for x in d); same = all(x > 0 for x in d) or all(x < 0 for x in d)
            out[key][m] = dict(diffs=d, differs=bool(outside and same), sign=int(np.sign(np.mean(d))) if same else 0)
            if m != "builds_shared_species":
                c, r, t = (out["measures"][k][m] for k in ("copied", "random", f"{ref_row}1"))
                out[key][m]["anchors_ordered"] = bool(min(c, r) < t < max(c, r))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "run" or len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--out"):
        raise SystemExit(__doc__)
    run(Path(args[2]) if len(args) == 3 else OUT)
