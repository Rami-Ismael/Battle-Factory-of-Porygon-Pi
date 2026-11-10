"""Diversity of a set of six-slot teams, for reporting beside win rate in ablations (no model imports).

Owner rule 2026-10-27: an ablation compares its metric with diversity. The loop's own per-generation `div`
record (diversity_loops.diversity) counts line-ups, but the taboo list caps each line-up at two copies per
generation, so line-up counts sit near their cap in every arm. These measures are not capped by the taboo list:

- distinct_lineups      distinct sorted species six-sets
- distinct_species      species used anywhere
- effective_species     exp(Shannon entropy) of species frequency over all slots
- top_species_share     share of teams that carry the most common species (named in top_species)
- mean_distance         mean pairwise distance in fields out of 48: species, ability, item and nature compared
                        directly, the four moves as a set, the six slots matched to minimise the total
                        (computed on a fixed random subsample of `pairs_n` teams when the set is larger)

    from team_diversity import summary, slots_from_paste
    summary([slots_from_paste(p) for p in pastes])
"""
import itertools, math
from collections import Counter
import numpy as np
from corpus import parse_team_text, norm

PERMS = np.array(list(itertools.permutations(range(6))))


def slots_from_paste(paste):
    """Six slot dicts (species, ability, item, nature, moves) from a Showdown paste; None if it doesn't parse to six.
    A blank line starts a slot, so an item-less Pokémon after the first keeps its own slot."""
    t = parse_team_text(paste, blank_ends_slot=True)
    if len(t) != 6: return None
    return [dict(species=norm(s["species"]), ability=norm(s.get("ability") or ""), item=norm(s.get("item") or ""),
                 nature=norm(s.get("nature") or ""), moves=[norm(m) for m in s.get("moves", []) if m]) for s in t]


def _encode(teams):
    ids = {}
    tok = lambda k, v: ids.setdefault((k, v), len(ids) + 1)
    T = np.full((len(teams), 6, 8), -1, np.int64)
    for i, t in enumerate(teams):
        for s, slot in enumerate(t):
            T[i, s, :4] = [tok(k, slot[k]) for k in ("species", "ability", "item", "nature")]
            ms = sorted({tok("move", m) for m in slot["moves"] if m})[:4]
            T[i, s, 4:4 + len(ms)] = ms
    return T


def _team_distances(a, B):
    """a [6,8], B [m,6,8] -> [m] slot-matched field distances (out of 48)."""
    c = (a[None, :, None, :4] != B[:, None, :, :4]).sum(-1)                      # [m,6,6]
    am, bm = a[None, :, None, 4:, None], B[:, None, :, None, 4:]
    overlap = ((am == bm) & (am >= 0)).any(-1).sum(-1)
    na, nb = (a[:, 4:] >= 0).sum(-1), (B[..., 4:] >= 0).sum(-1)
    c = c + np.maximum(na[None, :, None], nb[:, None, :]) - overlap
    return c[:, np.arange(6)[None, :], PERMS].sum(-1).min(-1)


def mean_distance(teams, pairs_n=128, seed=0):
    teams = [t for t in teams if t]
    if len(teams) > pairs_n:
        idx = np.random.default_rng(seed).choice(len(teams), pairs_n, replace=False)
        teams = [teams[i] for i in sorted(idx)]
    T = _encode(teams); d = []
    for i in range(len(T) - 1): d.append(_team_distances(T[i], T[i + 1:]))
    return float(np.concatenate(d).mean()) if d else None


def summary(teams, pairs_n=128, seed=0):
    teams = [t for t in teams if t]
    n = len(teams)
    if not n: return dict(n=0)
    sp = Counter(s["species"] for t in teams for s in t)
    tot = sum(sp.values()); H = -sum(c / tot * math.log(c / tot) for c in sp.values())
    top, _ = sp.most_common(1)[0]
    return dict(n=n, distinct_lineups=len({tuple(sorted(s["species"] for s in t)) for t in teams}),
                distinct_species=len(sp), effective_species=math.exp(H), top_species=top,
                top_species_share=sum(any(s["species"] == top for s in t) for t in teams) / n,
                mean_distance=mean_distance(teams, pairs_n, seed))
