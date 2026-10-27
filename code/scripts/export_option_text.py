"""Write data/regmb-option-text.json: effect text of every Reg M-B item and ability, and the 25 Stat Alignments.

    /tmp/vgc-pilot/.venv/bin/python scripts/export_option_text.py

The repository's data/items.json and data/abilities.json hold only the mod's few overrides, so the
text comes from the pinned simulator's built text tables (dist/data/text). Only names in
data/regmb-vocabulary.json are kept. The Jev baseline reads this file to describe item and
ability options.
"""
import json
import subprocess
from typing import Any

import repo_paths
from mb_prompt import regulation
from showdown_validator import SHOWDOWN

OUT = repo_paths.ROOT / 'data/regmb-option-text.json'
DUMP = ("const t = require(process.argv[1] + '/dist/data/text/items.js').ItemsText;"
        "const a = require(process.argv[1] + '/dist/data/text/abilities.js').AbilitiesText;"
        "const n = require(process.argv[1] + '/dist/data/natures.js').Natures;"
        "process.stdout.write(JSON.stringify({item: t, ability: a, nature: n}));")
STAT_NAMES = {'hp': 'HP', 'atk': 'Atk', 'def': 'Def', 'spa': 'SpA', 'spd': 'SpD', 'spe': 'Spe'}


def short_text(entry: dict[str, Any]) -> str:
    return entry.get('shortDesc') or entry.get('desc') or ''


def main() -> None:
    tables = json.loads(subprocess.run(['node', '-e', DUMP, str(SHOWDOWN)], capture_output=True, text=True,
                                       check=True).stdout)
    reg = regulation()
    out: dict[str, Any] = {}
    for field in ('item', 'ability'):
        wanted = [v for v in reg['values'][field] if v]
        out[field] = {reg['names'].get(v, v): short_text(tables[field].get(v, {})) for v in wanted}
        missing = [name for name, text in out[field].items() if not text]
        print(f'{field}: {len(out[field])} names, {len(missing)} without text {missing[:5]}')
    out['nature'] = {n['name']: {'raises': STAT_NAMES.get(n.get('plus')), 'lowers': STAT_NAMES.get(n.get('minus'))}
                     for n in tables['nature'].values()}
    print(f"nature: {len(out['nature'])} Stat Alignments")
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + '\n')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
