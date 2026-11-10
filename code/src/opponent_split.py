"""The frozen split of the 49 distinct meta teams into 39 training opponents and 10 validation opponents (issue #117).

Rule, agreed 2026-11-05: species replacement distance = 6 - shared species (forms distinct; order, sets and Stat Points
ignored). Teams at distance 0 or 1 are linked, and a family is a connected component. A fixed seed shuffles the families
and keeps each whole family that still fits until exactly 10 teams are chosen. No model result is read.

Canonical input order: results/top50_evs.json order, with the duplicate paste MB494 dropped (top50.DUPLICATES).
Species normalization: temperature_experiment.paste_species on the team file.

    /tmp/vgc-pilot/.venv/bin/python src/opponent_split.py   -> docs/opponent-split.json (refuses to change an existing draw)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import top50 as T50
from temperature_experiment import paste_species

SEED = 0
N_VALIDATION = 10
MANIFEST = HERE.parent / "docs" / "opponent-split.json"


def distinct_ids():
    return [i for i in T50.ids() if i not in T50.DUPLICATES]


def distance(a, b):
    return 6 - len(set(a) & set(b))


def families(species):
    """Connected components at distance <= 1, each a sorted list of indices; listed by their first member."""
    n, parent = len(species), list(range(len(species)))
    def root(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if distance(species[i], species[j]) <= 1: parent[root(i)] = root(j)
    groups = {}
    for i in range(n): groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def draw(fams, seed=SEED, k=N_VALIDATION):
    """Whole families, in a seeded random order, kept while they fit, until exactly k teams."""
    chosen, left = [], k
    for f in np.random.default_rng(seed).permutation(len(fams)):
        if len(fams[f]) <= left: chosen.append(int(f)); left -= len(fams[f])
        if left == 0: return sorted(chosen)
    raise ValueError(f"no union of whole families holds exactly {k} teams")


def build():
    ids = distinct_ids()
    species = [paste_species(Path(T50.resolve(i)).read_text()) for i in ids]
    fams = families(species)
    val_fams = draw(fams)
    val = sorted(ids[i] for f in val_fams for i in fams[f])
    return dict(
        seed=SEED, n_validation=N_VALIDATION, input_order=ids, duplicates_dropped=T50.DUPLICATES,
        normalization="temperature_experiment.paste_species; distance = 6 - shared species; link at distance <= 1",
        draw="families shuffled by numpy default_rng(seed).permutation; each whole family kept while it fits",
        families=[[ids[i] for i in f] for f in fams], family_sizes=sorted((len(f) for f in fams), reverse=True),
        species={i: list(s) for i, s in zip(ids, species)},
        validation=val, training=[i for i in ids if i not in val])


def load(path=MANIFEST):
    m = json.load(open(path))
    assert len(m["validation"]) == N_VALIDATION and len(m["training"]) == 49 - N_VALIDATION
    return m


def main():
    m = build()
    if MANIFEST.exists():
        old = load()
        if (old["validation"], old["training"]) != (m["validation"], m["training"]):
            raise SystemExit(f"{MANIFEST} holds a different draw; the split is frozen, so it is never redrawn")
        print("split unchanged"); return
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps({k: m[k] for k in ("family_sizes", "validation")}, indent=1))


if __name__ == "__main__":
    main()
