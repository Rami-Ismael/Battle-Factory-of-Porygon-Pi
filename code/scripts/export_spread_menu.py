"""Write data/regmb-spread-menu.json: the Stat Point spreads Jev may choose from, per species.

    /tmp/vgc-pilot/.venv/bin/python scripts/export_spread_menu.py

A species' menu is every distinct spread seen on it in the training corpus (corpus.load_corpus,
which already leaves out the held-out teams ranked 51 to 100) and in the top-50 meta teams,
sorted by value. GENERIC is the menu for a species with no spread on record, and every species'
menu also carries it. Every spread keeps each stat at 32 or less and the total at 66 or less.
"""
import json

import repo_paths
import corpus
import jev_fill
from mb_prompt import norm

OUT = repo_paths.ROOT / 'data/regmb-spread-menu.json'
STATS = jev_fill.STATS
GENERIC = [
    {'HP': 2, 'Atk': 32, 'Def': 0, 'SpA': 0, 'SpD': 0, 'Spe': 32},    # fast physical attacker
    {'HP': 2, 'Atk': 0, 'Def': 0, 'SpA': 32, 'SpD': 0, 'Spe': 32},    # fast special attacker
    {'HP': 32, 'Atk': 32, 'Def': 2, 'SpA': 0, 'SpD': 0, 'Spe': 0},    # slow physical attacker
    {'HP': 32, 'Atk': 0, 'Def': 2, 'SpA': 32, 'SpD': 0, 'Spe': 0},    # slow special attacker
    {'HP': 32, 'Atk': 0, 'Def': 17, 'SpA': 0, 'SpD': 17, 'Spe': 0},   # bulky support
]


def legal(spread: dict[str, int]) -> bool:
    return all(0 <= spread[k] <= 32 for k in STATS) and sum(spread.values()) <= 66


def key(spread: dict[str, int]) -> tuple[int, ...]:
    return tuple(spread[k] for k in STATS)


def main() -> None:
    seen: dict[str, set[tuple[int, ...]]] = {}
    teams, _ = corpus.load_corpus()
    pipeline = jev_fill.load_pipeline()
    meta = [row['team'] for row in pipeline.load_teams(str(jev_fill.PIPELINE / jev_fill.ling_config()['meta_teams']))]
    for mon in [m for t in teams for m in t]:
        seen.setdefault(norm(mon['species']), set()).add(key({k: int(mon['evs'].get(k, 0)) for k in STATS}))
    for mon in [m for t in meta for m in t]:
        seen.setdefault(norm(mon['species']), set()).add(key({k: int(mon['statPoints'].get(k, 0)) for k in STATS}))
    generic = {key(s) for s in GENERIC}
    menu = {}
    for species, spreads in seen.items():
        rows = [dict(zip(STATS, values, strict=True)) for values in sorted(spreads | generic)]
        menu[species] = [s for s in rows if legal(s)]
    out = {'generic': GENERIC, 'species': dict(sorted(menu.items()))}
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    sizes = sorted(len(v) for v in menu.values())
    print(f'{len(menu)} species; menu sizes min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}; wrote {OUT}')


if __name__ == '__main__':
    main()
