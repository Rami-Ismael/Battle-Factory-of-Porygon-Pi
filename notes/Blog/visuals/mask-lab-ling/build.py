#!/usr/bin/env python3
"""Build the Mask Lab page from the Ling masked-completion run.

Inputs (team_completion_pipeline/runs/prompt-ab/): eval2.tasks.json, eval2.jsonl, candidates.json and,
once battles have run, panels.json. Output: index.html (one file, sprites inlined) and data.json.
Missing panels are shown on the page as "not scored yet", never as zero.
"""
import base64
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2] / 'team_completion_pipeline/runs/prompt-ab'
SPRITES = HERE.parent / 'search-loop/sprites'
ITEMS = HERE.parent / 'item-sprites'
ITEM_IDS = {f.stem[5:] for f in (HERE.parent / 'item-sprites').glob('item-*.png')}
ARMS = ['baseline@4096', 'baseline@16384', 'regmb@16384']
FAMILIES = ['stats', 'items', 'moves', 'pokemon', 'abilities', 'natures']
STATS = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']


def norm(text):
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def get_path(team, path):
    m = re.fullmatch(r'team\[(\d+)\](?:\.(\w+)(?:\[(\d+)\])?)?', path)
    i, field, j = int(m.group(1)), m.group(2), m.group(3)
    mon = team[i]
    if field is None:
        return mon
    return mon[field] if j is None else mon[field][int(j)]


def slim(value):
    """Whole Pokémon donors carry level, ivs and slot; the page shows only these fields."""
    if isinstance(value, dict) and 'species' in value:
        return {k: value[k] for k in ('species', 'item', 'ability', 'nature', 'statPoints', 'moves') if k in value}
    return value


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def main():
    saved = json.loads((RUN / 'eval2.tasks.json').read_text())
    rows = [json.loads(l) for l in (RUN / 'eval2.jsonl').read_text().splitlines()]
    cands = {c['label']: c for c in json.loads((RUN / 'candidates.json').read_text())['candidates']}
    panels_path = RUN / 'panels.json'
    panels = json.loads(panels_path.read_text()) if panels_path.exists() else None
    opp_ids = [o['team_id'] for o in panels['opponents']] if panels else []

    def panel(label):
        p = panels['panels'].get(label) if panels else None
        if not p or p.get('score') is None:
            return None
        by = p['by_opponent']
        bits = ''.join(('1' if by[str(o)][0] > 0 else '0') if str(o) in by else '-' for o in opp_ids)
        return {'win': round(p['score'], 4), 'opp': bits}

    starts = [{'id': s['id'], 'team': s['team']} for s in saved['starting_teams']]
    start_index = {s['id']: i for i, s in enumerate(starts)}
    tasks, results = [], {}
    arm_rows = {(r['arm'], r['task_id']): r for r in rows if r['status'] != 'transport'}
    for t in saved['tasks']:
        tasks.append({'id': t['id'], 's': start_index[t['source']], 'f': t['task'], 'k': t['k'], 'sl': t['slots']})
        res = {'arms': {}}
        for arm in ARMS:
            r = arm_rows.get((arm, t['id']))
            if r is None:
                res['arms'][arm] = None
                continue
            usage = r.get('usage') or {}
            entry = {'ok': r['status'] == 'legal', 'cost': round(usage.get('cost') or 0, 6)}
            if entry['ok']:
                entry['fill'] = [slim(get_path(r['team'], p)) for p in t['masked_paths']]
                entry.update(panel(f"ling:{arm}:{t['id']}") or {})
            else:
                entry['stage'] = r.get('stage')
                entry['err'] = (r.get('error') or '')[:110]
            res['arms'][arm] = entry
        rand = cands.get(f"random:{t['id']}")
        if rand and rand['team']:
            res['random'] = dict({'fill': [slim(get_path(rand['team'], p)) for p in t['masked_paths']]},
                                 **(panel(f"random:{t['id']}") or {}))
        results[t['id']] = res
    originals = {}
    for s in starts:
        originals[s['id']] = panel(f"orig:{s['id']}")

    # Sprites actually used by the page, inlined so the file works anywhere.
    species, items = set(), set()
    def collect(team):
        for m in team:
            if m:
                species.add(norm(m['species']))
                if m.get('item'):
                    items.add(norm(m['item']))
    for s in starts:
        collect(s['team'])
    for t in tasks:
        r = results[t['id']]
        fills = [r['random']['fill']] if r.get('random') else []
        fills += [a['fill'] for a in r['arms'].values() if a and a.get('fill')]
        for fill in fills:
            for value in fill:
                if isinstance(value, dict) and 'species' in value:
                    collect([value])
                elif isinstance(value, str) and norm(value) in ITEM_IDS:
                    items.add(norm(value))
    sprites = {}
    for sid in sorted(species):
        f = SPRITES / f'{sid}.png'
        if f.exists():
            sprites[sid] = 'data:image/png;base64,' + base64.b64encode(f.read_bytes()).decode()
    for iid in sorted(items):
        f = ITEMS / f'item-{iid}.png'
        if f.exists():
            sprites['item-' + iid] = 'data:image/png;base64,' + base64.b64encode(f.read_bytes()).decode()

    arm_summary = {}
    for arm in ARMS:
        got = [r for (a, _), r in arm_rows.items() if a == arm]
        legal = sum(r['status'] == 'legal' for r in got)
        arm_summary[arm] = {'n': len(got), 'legal': legal, 'ci': wilson(legal, len(got)) if got else None}
    first = RUN / 'summary-eval.json'
    run1 = None
    if first.exists():
        s = json.loads(first.read_text())
        run1 = {a: {'n': s['arms'][a]['n'], 'legal': s['arms'][a]['legal']} for a in ARMS}
    data = {
        'opponents': panels['opponents'] if panels else [],
        'policy': panels['policy'] if panels else 'not battled yet',
        'starts': starts, 'tasks': tasks, 'results': results, 'originals': originals,
        'arms': arm_summary, 'run1': run1, 'scored': bool(panels),
    }
    (HERE / 'data.json').write_text(json.dumps(data, separators=(',', ':')))
    html = (HERE / 'template.html').read_text()
    html = html.replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    html = html.replace('/*SPRITES*/{}', json.dumps(sprites, separators=(',', ':')))
    (HERE / 'index.html').write_text(html)
    scored = sum(1 for t in tasks for a in results[t['id']]['arms'].values() if a and a.get('win') is not None)
    print(f'{len(tasks)} tasks, {len(sprites)} sprites, {scored} scored Ling completions, '
          f'index.html {len(html) // 1024} KB')


if __name__ == '__main__':
    main()
