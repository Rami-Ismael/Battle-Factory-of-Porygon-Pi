"""Export the 20-starter reasoning studies' per-trial records into data.json.

Reads the three saved study directories (Ling reasoning off/on, Mercury
off/low/medium/high, Pokémon diffusion F1 + G2 v3) and writes them under the
`study20` key of data.json. The existing 5-starter keys are left untouched.

    python3 blog/mask-lab/export_study20.py [RESULTS_DIR]

RESULTS_DIR is the `code/results/` directory holding
reasoning-expanded-20261012, mercury-reasoning-20261013 and
diffusion-matched-20261013. The paid completions stay where they are; only
the fields the explorer draws are copied.

Record shapes (all keyed so the page can reuse the 5-starter explorer code):

- `starts[s].team[i]`: species, item, ability, nature, moves, statPoints.
- `trials[]`: id (`s{starter}-{family}-{k}`), s, f, k, paths. Paths use the
  explorer's string form: `team[i]`, `team[i].item`, `team[i].ability`,
  `team[i].nature`, `team[i].statPoints`, `team[i].moves[j]`.
- `results["{trial}/r{repeat}"][writer]`: `{ok: true, fill, win, opp, cost,
  rt}` or `{ok: false, stage, err, cost, rt}`. `fill[k]` is the value written
  at `paths[k]` (a slot object for a whole Pokémon). `win` is wins / 50.
  `opp` is 49 characters in `opponents` order; each character is the number of
  battles won against that opponent (0 or 1, and 0, 1 or 2 for the opponent
  with weight 2). `cost` is the recorded API charge in USD; `rt` the reported
  reasoning tokens. `stage` is `parse`, `response`, `schema` or `validator`.
- `untouched[s]`: the starting team's own panel, if one was ever battled.
"""
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = Path.home() / '.codex/worktrees/519e' / (
    'Using Advance method search in the whole search of possible vgc pokemon '
    'format to find the right counter to a pokemon team') / 'code/results'
RESULTS = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_RESULTS

STUDIES = {
    'ling': 'reasoning-expanded-20261012',
    'mercury': 'mercury-reasoning-20261013',
    'diffusion': 'diffusion-matched-20261013',
}
WRITERS = [
    dict(key='ling@off', kind='Ling', short='Ling 3.0 Flash · reasoning off', study=STUDIES['ling'], arm='off', run='off'),
    dict(key='ling@on', kind='Ling', short='Ling 3.0 Flash · reasoning on', study=STUDIES['ling'], arm='on', run='on'),
    dict(key='mercury@off', kind='Mercury', short='Mercury 2.5 · reasoning off', study=STUDIES['mercury'], arm='off', run='off'),
    dict(key='mercury@low', kind='Mercury', short='Mercury 2.5 · reasoning low', study=STUDIES['mercury'], arm='low', run='low'),
    dict(key='mercury@medium', kind='Mercury', short='Mercury 2.5 · reasoning medium', study=STUDIES['mercury'], arm='medium', run='medium'),
    dict(key='mercury@high', kind='Mercury', short='Mercury 2.5 · reasoning high', study=STUDIES['mercury'], arm='high', run='high'),
    dict(key='diffusion@v3', kind='Diffusion', short='Pokémon diffusion v3 (F1 + G2)', study=STUDIES['diffusion'], arm='v3', run=''),
]
FAMILIES = {
    'pokemon': 'Whole Pokémon', 'mixed': 'Mixed', 'move': 'Moves', 'item': 'Items',
    'ability': 'Abilities', 'alignment': 'Stat Alignment', 'stats': 'Stat Points',
}
REPEATS = 3
STATS = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']
DOUBLED_OPPONENT = 641  # the one panel opponent battled twice in these studies


def slot(s):
    return dict(species=s['species'], item=s['item'], ability=s['ability'], nature=s['nature'],
                moves=list(s['moves']), statPoints={k: s['evs'].get(k, 0) for k in STATS})


def path_str(p):
    i = p[0]
    if len(p) == 1:
        return f'team[{i}]'
    if p[1] == 'evs':
        return f'team[{i}].statPoints'
    if p[1] == 'moves':
        return f'team[{i}].moves[{p[2]}]'
    return f'team[{i}].{p[1]}'


def value_at(team, p):
    s = team[p[0]]
    if len(p) == 1:
        return slot(s)
    if p[1] == 'evs':
        return {k: s['evs'].get(k, 0) for k in STATS}
    if p[1] == 'moves':
        return s['moves'][p[2]]
    return s[p[1]]


def stage_of(err):
    if re.search(r"Expecting|Unterminated|Extra data|Invalid control|JSONDecodeError", err):
        return 'parse'
    if err.strip("'") in ('choices',):
        return 'response'
    if re.search(r"duplicate or unmasked path|wrong number of fills|truncated/non-stop|invalid slot fields|"
                 r"expected exactly|slot names must be strings|four move strings|Stat Points require", err):
        return 'schema'
    return 'validator'


def main():
    data = json.loads((HERE / 'data.json').read_text())
    opp_species = {o['team_id']: o['species'] for o in data['opponents']}

    manifest = json.loads((RESULTS / STUDIES['ling'] / 'off/r0/manifest.json').read_text())
    starts = manifest['starts']
    trials = manifest['trials']
    opp_ids = [o['team_id'] for o in manifest['opponents']]
    assert opp_ids == [o['team_id'] for o in data['opponents']], 'opponent order differs from the 5-starter export'
    opp_index = {tid: i for i, tid in enumerate(opp_ids)}

    # every run must share the same starters, masks and opponents
    for w in WRITERS:
        for r in range(REPEATS):
            run = RESULTS / w['study'] / w['run'] / f'r{r}'
            m = json.loads((run / 'manifest.json').read_text())
            assert m['starts'] == starts and m['trials'] == trials and \
                [o['team_id'] for o in m['opponents']] == opp_ids, f'{run} does not match the shared design'

    results = {}
    stages = collections.Counter()
    legal = collections.Counter()
    doubled = collections.Counter()
    for t in trials:
        for r in range(REPEATS):
            rec = {}
            for w in WRITERS:
                run = RESULTS / w['study'] / w['run'] / f'r{r}'
                c = json.loads((run / 'completions' / f'{t["id"]}.json').read_text())
                resp = c.get('response') or {}
                usage = resp.get('usage') or {}
                out = {'ok': bool(c.get('valid'))}
                if w['kind'] != 'Diffusion':
                    out['cost'] = usage.get('cost')
                    rt = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
                    if rt:
                        out['rt'] = rt
                    if c.get('attempts', 1) != 1:
                        out['attempts'] = c['attempts']
                if out['ok']:
                    legal[w['key']] += 1
                    out['fill'] = [value_at(c['team'], p) for p in t['paths']]
                    prefix = 'diffusion-' if w['kind'] == 'Diffusion' else 'llm-'
                    lab = json.loads((run / 'labels' / f'{prefix}{t["id"]}.json').read_text())
                    assert lab['battles'] == 50 and len(lab['results']) == 49, f'{run} {t["id"]} panel is not 50 battles'
                    bits = ['0'] * 49
                    wins = 0
                    for team_id, opp, w_, l_, ties in lab['results']:
                        assert team_id == lab['team_id'] and ties == 0
                        if w_ + l_ == 2:
                            doubled[opp] += 1
                        bits[opp_index[opp]] = str(w_)
                        wins += w_
                    assert wins == lab['wins']
                    out['win'] = wins / 50
                    out['opp'] = ''.join(bits)
                else:
                    err = c.get('error') or ('finish: ' + str((resp.get('choices') or [{}])[0].get('finish_reason')))
                    out['stage'] = stage_of(err)
                    out['err'] = err
                    stages[(w['key'], out['stage'])] += 1
                rec[w['key']] = out
            results[f'{t["id"]}/r{r}'] = rec

    assert set(doubled) == {DOUBLED_OPPONENT}, f'unexpected doubled opponents {doubled}'
    opponents = [dict(team_id=tid, weight=2.0 if tid == DOUBLED_OPPONENT else 1.0, species=opp_species[tid]) for tid in opp_ids]

    study20 = {
        'label': 'Reasoning studies · 20 starting teams · 420 masks per writer',
        'studies': STUDIES,
        'policy': data['policy'],
        'battles': 50,
        'opponents': opponents,
        'writers': [{k: v for k, v in w.items() if k != 'run'} for w in WRITERS],
        'families': FAMILIES,
        'repeats': REPEATS,
        'starts': [dict(team=[slot(s) for s in st]) for st in starts],
        'trials': [dict(id=t['id'], s=t['start'], f=t['task'], k=t['k'], paths=[path_str(p) for p in t['paths']]) for t in trials],
        'untouched': {},
        'results': results,
    }
    data['study20'] = study20
    (HERE / 'data.json').write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False))

    print(f'{len(results)} masks × {len(WRITERS)} writers exported to data.json["study20"]')
    for w in WRITERS:
        print(f"  {w['key']:16} legal {legal[w['key']]}/{len(results)}")
    print('rejection stages:', dict(sorted(stages.items())))
    print(f'doubled opponent: team {DOUBLED_OPPONENT} (index {opp_index[DOUBLED_OPPONENT]}); untouched panels: {len(study20["untouched"])}')


if __name__ == '__main__':
    main()
