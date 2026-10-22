#!/usr/bin/env python3
"""Build the Mask Lab page from the Ling masked-completion run.

Inputs (team_completion_pipeline/runs/prompt-ab/): eval2.tasks.json, eval2.jsonl, candidates.json and,
once battles have run, panels.json. Output: index.html (one file, sprites inlined) and data.json.
Missing panels are shown on the page as "not scored yet", never as zero.
"""
import base64
import argparse
import hashlib
from datetime import datetime, timezone
import gzip
import json
import math
import re
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2] / 'team_completion_pipeline/runs/prompt-ab'
SPRITES = HERE.parent / 'search-loop/sprites'
ITEMS = HERE.parent / 'item-sprites'
ITEM_IDS = {f.stem[5:] for f in (HERE.parent / 'item-sprites').glob('item-*.png')}
ARMS = ['baseline@4096', 'baseline@16384', 'regmb@16384', 'diffusion@v3']
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reasoning-audit', type=Path, help='Import the completed expanded study audit; never launch experiments')
    parser.add_argument('--mercury-study', type=Path, help='Import an explicitly timestamped Mercury checkpoint; no paid calls')
    parser.add_argument('--diffusion-study', type=Path, help='Import matched Pokémon diffusion results')
    parser.add_argument('--comparison', type=Path, help='Import the seven-method common-cohort comparison')
    args = parser.parse_args()
    if args.comparison:
        (HERE/'all-method-comparison.json').write_bytes(args.comparison.read_bytes())
    if args.diffusion_study:
        checkpoint=json.loads((args.diffusion_study/'status.json').read_text())
        checkpoint.pop('rows',None)
        checkpoint['design']=json.loads((args.diffusion_study/'design.json').read_text())
        audit_path=args.diffusion_study/'final-audit.json'
        checkpoint['audit']=json.loads(audit_path.read_text()) if checkpoint['complete'] and audit_path.exists() else None
        (HERE/'matched-diffusion-results.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
    if args.mercury_study:
        root = args.mercury_study
        experiment = json.loads((root/'experiment.json').read_text())
        assert experiment['model'] == 'inception/mercury-2.5'
        stats, observations, configs = {}, {}, []
        for entry in experiment['runs']:
            folder = root/entry['path']
            manifest = json.loads((folder/'manifest.json').read_text())
            configs.append(json.loads((folder/'battle-config.json').read_text()))
            assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == entry['manifest_sha256']
            a = stats.setdefault(entry['arm'], dict(returned=0, legal=0, scored=0, planned=0, cost=0, cost_coverage=0, prompt_tokens=0, cached_tokens=0, reasoning_tokens=0, cache_coverage=0))
            for trial in manifest['trials']:
                a['planned'] += 1
                path = folder/'completions'/(trial['id']+'.json')
                if not path.exists(): continue
                row = json.loads(path.read_text())
                assert row['request_hash'] == entry['request_hashes'][trial['id']]
                a['returned'] += 1
                a['legal'] += bool(row['valid'])
                label_path = folder/'labels'/('llm-'+trial['id']+'.json')
                label = json.loads(label_path.read_text()) if label_path.exists() else None
                if label is not None:
                    assert row['valid'] and label['battles'] == 50 and 0 <= label['wins'] <= 50
                win = label['wins']/50 if label else None
                a['scored'] += win is not None
                a['win_sum'] = a.get('win_sum', 0) + (win or 0)
                observations[(entry['arm'], entry['replicate'], trial['id'])] = (trial['start'], bool(row['valid']), win)
                usage = row.get('response',{}).get('usage',{})
                a['cost'] += usage.get('cost') or 0
                a['cost_coverage'] += usage.get('cost') is not None
                a['prompt_tokens'] += usage.get('prompt_tokens') or 0
                cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                a['cached_tokens'] += cached or 0
                a['cache_coverage'] += cached is not None
                a['reasoning_tokens'] += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
        assert all(c == configs[0] for c in configs), 'Battle configurations differ'
        def interval(rows):
            grouped = defaultdict(list)
            for starter, delta in rows:
                grouped[starter].append(delta)
            values = [sum(v)/len(v) for _,v in sorted(grouped.items())]
            result = dict(pairs=len(rows), starters=len(values), mean=sum(values)/len(values) if values else None, low=None, high=None)
            if len(values) >= 2:
                rng = random.Random(experiment['seed'])
                draws = sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(4000))
                result.update(low=draws[100], high=draws[3899])
            return result
        interim = {}
        for arm in stats:
            if arm == 'off': continue
            legality, battles = [], []
            for (level, rep, trial), (starter, valid, win) in observations.items():
                control = observations.get(('off',rep,trial))
                if level != arm or control is None: continue
                legality.append((starter,int(valid)-int(control[1])))
                if valid and control[1] and win is not None and control[2] is not None:
                    battles.append((starter,win-control[2]))
            interim[arm] = dict(legality_delta=interval(legality),win_rate_delta_both_legal=interval(battles))
        final = json.loads((root/'status.json').read_text())
        complete = final.get('complete', False) and all(a['returned']==a['planned'] and a['scored']==a['legal'] for a in stats.values())
        checkpoint = dict(model=experiment['model'], updated_at=datetime.now(timezone.utc).isoformat(), complete=complete, arms=stats,
                          final={k:final[k] for k in ['arms','versus_off']} if complete else None,
                          audit=json.loads((root/'final-audit.json').read_text()) if complete and (root/'final-audit.json').exists() else None,
                          generation=json.loads((root/'generation-status.json').read_text()),
                          interim=interim, preflight=json.loads((root/'preflight-verification.json').read_text()))
        (HERE/'mercury-results.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
    snapshot = HERE / 'reasoning-results.json'
    if args.reasoning_audit:
        audit_bytes = args.reasoning_audit.read_bytes()
        audit = json.loads(audit_bytes)
        assert audit['missing'] == audit['pending'] == 0
        assert audit['configs_identical'] and audit['request_hashes_verified'] == 840
        assert all(audit['arms'][a]['returned'] == 420 and audit['arms'][a]['legal'] == audit['arms'][a]['scored'] for a in ('off', 'on'))
        snapshot.write_text(json.dumps({'study': 'reasoning-expanded-20261012', 'source_sha256': hashlib.sha256(audit_bytes).hexdigest(), 'audit': audit}, indent=2) + '\n')
    saved = json.loads((RUN / 'eval2.tasks.json').read_text())
    rows = [json.loads(l) for l in (RUN / 'eval2.jsonl').read_text().splitlines()]
    cands = {c['label']: c for c in json.loads((RUN / 'candidates.json').read_text())['candidates']}
    drows_path = RUN / 'diffusion_rows.json'
    drows = json.loads(drows_path.read_text()) if drows_path.exists() else {}
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
            if arm == 'diffusion@v3':
                cand = cands.get(f"diffusion:{t['id']}")
                if cand and cand['team']:
                    e = {'ok': True, 'cost': 0, 'fill': [slim(get_path(cand['team'], p)) for p in t['masked_paths']]}
                    e.update(panel(f"diffusion:{t['id']}") or {})
                else:
                    e = {'ok': False, 'cost': 0, 'stage': 'decoder', 'err': (drows.get(t['id'], {}).get('error') or 'not run')[:110]}
                res['arms'][arm] = e
                continue
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
        got = [results[t['id']]['arms'][arm] for t in tasks if results[t['id']]['arms'][arm]]
        legal = sum(1 for g in got if g['ok'])
        arm_summary[arm] = {'n': len(got), 'legal': legal, 'ci': wilson(legal, len(got)) if got else None}
    first = RUN / 'summary-eval.json'
    run1 = None
    if first.exists():
        s = json.loads(first.read_text())
        run1 = {a: {'n': s['arms'][a]['n'], 'legal': s['arms'][a]['legal']} for a in ARMS if a in s['arms']}
    data = {
        'opponents': panels['opponents'] if panels else [],
        'policy': panels['policy'] if panels else 'not battled yet',
        'starts': starts, 'tasks': tasks, 'results': results, 'originals': originals,
        'arms': arm_summary, 'run1': run1, 'scored': bool(panels),
    }
    if snapshot.exists():
        data['reasoning'] = json.loads(snapshot.read_text())
    if (HERE/'mercury-results.json').exists():
        data['mercury'] = json.loads((HERE/'mercury-results.json').read_text())
    if (HERE/'matched-diffusion-results.json').exists():
        data['matchedDiffusion']=json.loads((HERE/'matched-diffusion-results.json').read_text())
    if (HERE/'all-method-comparison.json').exists():
        data['allMethodComparison']=json.loads((HERE/'all-method-comparison.json').read_text())
    (HERE / 'data.json').write_text(json.dumps(data, separators=(',', ':')))
    html = (HERE / 'template.html').read_text()
    packed = base64.b64encode(gzip.compress(json.dumps(data, separators=(',', ':')).encode(), 9)).decode()
    html = html.replace('/*DATAGZ*/', packed)
    html = html.replace('/*SPRITES*/{}', json.dumps(sprites, separators=(',', ':')))
    (HERE / 'index.html').write_text(html)
    scored = sum(1 for t in tasks for a in results[t['id']]['arms'].values() if a and a.get('win') is not None)
    print(f'{len(tasks)} tasks, {len(sprites)} sprites, {scored} scored Ling completions, '
          f'index.html {len(html) // 1024} KB')


if __name__ == '__main__':
    main()
