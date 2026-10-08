#!/usr/bin/env python3
"""Run the project's v3 masked-diffusion decoder on the same masks Ling was given.

    python3 mask_lab_diffusion.py prepare [--limit N]   # manifest in the decoder's format
    python3 mask_lab_diffusion.py generate              # F1 + G2 v3 ensemble, ask 0.5, guidance 2, temperature 1
    python3 mask_lab_diffusion.py collect               # add valid completions to candidates.json

The decoder is scripts/diffusion_baseline_v3.py in the code repo and is used unchanged, with the settings the
owner selected on 2026-10-08 (profile: code/data/diffusion-baseline-profile.json). It sees the known categorical
fields only: no Stat Points and no opponent pastes. Masked Stat Points are copied from a same-species corpus
spread or drawn as a random legal spread, so the Stat Points family is not a neural completion. No LLM calls.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/prompt-ab'
WORK = Path.home() / 'vgc-data/mask_lab_diffusion'
REPO = Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot')
PY = '/tmp/vgc-pilot/.venv/bin/python'
MEMBERS = ['F1_final_medium_regmb_v3', 'G2_family_matrix_noprtrain_regmb_v3']
SEED = 20261008
STATS = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']


def to_decoder_team(team):
    return [{'species': m['species'], 'item': m['item'], 'ability': m['ability'], 'nature': m['nature'],
             'moves': list(m['moves']), 'evs': {k: int(m['statPoints'].get(k, 0)) for k in STATS}} for m in team]


def from_decoder_team(team):
    return [{'slot': i + 1, 'species': s['species'], 'item': s['item'], 'ability': s['ability'], 'nature': s['nature'],
             'moves': list(s['moves']), 'statPoints': dict(s['evs'])} for i, s in enumerate(team)]


def paths_for(task):
    kind, slots = task['task'], task['slots']
    field = {'stats': 'evs', 'items': 'item', 'abilities': 'ability', 'natures': 'nature'}
    if kind == 'pokemon':
        return [[s] for s in slots]
    if kind == 'moves':
        return [[i // 4, 'moves', i % 4] for i in slots]
    return [[s, field[kind]] for s in slots]


def prepare(args):
    saved = json.loads((RUN / 'eval2.tasks.json').read_text())
    index = {s['id']: i for i, s in enumerate(saved['starting_teams'])}
    trials = [{'id': t['id'], 'start': index[t['source']], 'task': t['task'], 'k': t['k'], 'paths': paths_for(t)}
              for t in saved['tasks']]
    if args.limit:
        trials = trials[:args.limit]
    manifest = {'schema': 1, 'format': saved['format'], 'seed': SEED,
                'starts': [to_decoder_team(s['team']) for s in saved['starting_teams']], 'trials': trials}
    (WORK / 'baseline').mkdir(parents=True, exist_ok=True)
    (WORK / 'baseline/manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f'manifest: {len(manifest["starts"])} starting teams, {len(trials)} trials -> {WORK / "baseline"}')


def generate(args):
    cmd = ['nice', '-n', '10', PY, 'diffusion_baseline_v3.py', 'generate', '--baseline', str(WORK / 'baseline'),
           '--output', str(WORK / 'out'), '--members', *MEMBERS]
    sys.exit(subprocess.run(cmd, cwd=REPO / 'scripts').returncode)


def collect(args):
    cands_path = RUN / 'candidates.json'
    cands = json.loads(cands_path.read_text())['candidates']
    tasks = {t['id']: t for t in json.loads((RUN / 'eval2.tasks.json').read_text())['tasks']}
    cands = [c for c in cands if c['kind'] != 'diffusion']
    stats = {'valid': 0, 'invalid': 0, 'unsupported': 0}
    rows = {}
    for f in sorted((WORK / 'out/completions').glob('*.json')):
        row = json.loads(f.read_text())
        rows[row['trial']] = {'valid': row['valid'], 'supported': row.get('supported', True),
                              'error': (row.get('error') or '')[:160], 'seconds': row.get('seconds')}
        if row['valid']:
            team = from_decoder_team(row['team'])
            cands.append({'label': f"diffusion:{row['trial']}", 'kind': 'diffusion', 'task_id': row['trial'],
                          'source': tasks[row['trial']]['source'], 'team': team})
            stats['valid'] += 1
        elif row.get('supported', True):
            stats['invalid'] += 1
        else:
            stats['unsupported'] += 1
    sys.path.insert(0, str(HERE))
    import mask_lab_prepare as M
    for c in cands:
        if c['kind'] == 'diffusion':
            c['paste'] = M.to_paste(c['team'])
    cands_path.write_text(json.dumps({'candidates': cands}, indent=1) + '\n')
    (RUN / 'diffusion_rows.json').write_text(json.dumps(rows, indent=1) + '\n')
    print('diffusion completions', stats, '| candidates now', len(cands))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--limit', type=int)
    sub.add_parser('generate'); sub.add_parser('collect')
    a = ap.parse_args()
    {'prepare': prepare, 'generate': generate, 'collect': collect}[a.cmd](a)
