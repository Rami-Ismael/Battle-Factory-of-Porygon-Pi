"""Write dose-pools.json: legal and illegal option counts per category on the three grids' masks.

    /tmp/vgc-pilot/.venv/bin/python build_dose_pools.py

For the first hidden field of every one-field mask: the legal list from legal_options, and the pool
of illegal options the dose curve can draw from: every vocabulary value of that field minus the
legal list. Moves: ones the species cannot learn or the slot already has. Abilities: ones not legal
for the species. Items: a teammate's item (Item Clause) for most species, but nearly every item for a
Mega form, whose only legal item is its own stone. Species: a family already on the team (Species
Clause) or a species whose slot could not be finished (Ditto, a Mega whose stone is taken). Stat
Alignment has no illegal option.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'code/scripts'))
import repo_paths  # noqa: E402,F401
import jev_fill as J  # noqa: E402
import legal_options as L  # noqa: E402
from mb_prompt import norm, regulation  # noqa: E402

FIELD = {'moves': 'moves', 'abilities': 'ability', 'items': 'item', 'pokemon': 'species', 'natures': 'nature'}


def counts(task):
    reg, team = regulation(), task['masked_team']
    path = J.parse_path(task['masked_paths'][0])
    i, field = path[0], FIELD[task['task']]
    if field == 'species':
        legal = L.fillable_species_options(team)
        return len(legal), len([s for s in reg['values']['species'] if s]) - len(legal)
    p = [i, field] + ([path[2]] if field == 'moves' else [])
    legal = L.options(team, p)
    if field == 'nature':
        return len(legal), 0
    pool = {'moves': 'move', 'ability': 'ability', 'item': 'item'}[field]
    vocab = {v for v in reg['values'][pool] if v}
    return len(legal), len(vocab) - len({norm(x) for x in legal})


out = {}
for grid in J.GRIDS:
    for t in J.tasks(grid):
        if t['k'] != 1 or t['task'] not in FIELD:
            continue
        lg, il = counts(t)
        slot = t['masked_team'][J.parse_path(t['masked_paths'][0])[0]]
        out.setdefault(t['task'], []).append({'grid': grid, 'task_id': t['id'], 'species': slot['species'] if slot else None,
                                              'legal': lg, 'illegal_pool': il})
(HERE / 'dose-pools.json').write_text(json.dumps(out, indent=1) + '\n')
for k, rows in out.items():
    lg = sorted(r['legal'] for r in rows); il = sorted(r['illegal_pool'] for r in rows)
    print(f"{k:9} n={len(rows):2} legal {lg[0]}-{lg[-1]} (median {lg[len(lg)//2]})  illegal pool {il[0]}-{il[-1]} (median {il[len(il)//2]})")
