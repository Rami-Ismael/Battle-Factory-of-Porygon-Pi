#!/usr/bin/env python3
"""Candidate teams for the Mask Lab page: untouched originals, random legal fills, legal Ling completions.

Reads runs/prompt-ab/eval2.jsonl and eval2.tasks.json (written by prompt_ab.py with saved teams).
A random legal fill redraws every masked value uniformly from values the format allows
(moves from the species' Reg M-B learnset, abilities from its dex entry, items from the items seen
in the datasets, natures from all 25, Stat Points as a random 66-point spread, whole Pokémon from
random legal sets), checks the team with the Showdown validator, and redraws up to 50 times.
Writes runs/prompt-ab/candidates.json. No model calls and no battles.
"""
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/vgc-pilot/src')
import corpus  # noqa: E402  (dex entries and the 25 natures)
import pipeline as P  # noqa: E402
import prompt_ab as A  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/prompt-ab'
LEARNSET = json.loads((Path.home() / '.local/share/vgc-pilot-runtime/files/learnset_true.json').read_text())
STATS = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']
ARMS = ['baseline@4096', 'baseline@16384', 'regmb@16384']


def to_paste(team):
    blocks = []
    for mon in team:
        lines = [mon['species'] + (f" @ {mon['item']}" if mon.get('item') else ''),
                 f"Ability: {mon['ability']}", 'Level: 50']
        spread = ' / '.join(f'{mon["statPoints"][k]} {k}' for k in STATS if mon['statPoints'].get(k))
        if spread:
            lines.append(f'EVs: {spread}')
        lines.append(f"{mon['nature']} Nature")
        lines += [f'- {move}' for move in mon['moves']]
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks) + '\n'


def random_spread(rng):
    spread = {k: 0 for k in STATS}
    for _ in range(66):
        spread[rng.choice([k for k in STATS if spread[k] < 32])] += 1
    return spread


def move_names(ids, config):
    """Display names for move ids, from the simulator's own data."""
    if not ids:
        return {}
    script = ("const {Dex}=require(process.argv[1]+'/dist/sim');const ids=JSON.parse(process.argv[2]);"
              "console.log(JSON.stringify(Object.fromEntries(ids.map(i=>[i,Dex.moves.get(i).name]))))")
    out = subprocess.run(['node', '-e', script, config['showdown_runtime'], json.dumps(sorted(ids))],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def random_fill(config, task, donors, items, rng, tries=50):
    kind, slots = task['task'], task['slots']
    for attempt in range(1, tries + 1):
        team = copy.deepcopy(task['original'])
        if kind == 'stats':
            for s in slots:
                team[s]['statPoints'] = random_spread(rng)
        elif kind == 'natures':
            for s in slots:
                team[s]['nature'] = rng.choice(corpus.NATURES)
        elif kind == 'items':
            for s in slots:
                # A Mega form can only legally hold its own stone; that is a property of the species.
                required = (corpus.dex_entry(team[s]['species']) or {}).get('requiredItem')
                if required:
                    team[s]['item'] = required
                    continue
                held = {m['item'] for i, m in enumerate(team) if i != s}
                team[s]['item'] = rng.choice([x for x in items if x not in held])
        elif kind == 'abilities':
            for s in slots:
                options = list((corpus.dex_entry(team[s]['species']) or {}).get('abilities', {}).values())
                if not options:
                    return None, attempt
                team[s]['ability'] = rng.choice(options)
        elif kind == 'moves':
            by_mon = {}
            for i in slots:
                by_mon.setdefault(i // 4, []).append(i % 4)
            for mon, positions in by_mon.items():
                legal = [m for m in LEARNSET.get(corpus.norm(team[mon]['species']), []) if m != '[MASK]']
                taken = {corpus.norm(m) for j, m in enumerate(team[mon]['moves']) if j not in positions}
                legal = [m for m in legal if m not in taken]
                if len(legal) < len(positions):
                    return None, attempt
                for j, move in zip(positions, rng.sample(legal, len(positions))):
                    team[mon]['moves'][j] = move
        elif kind == 'pokemon':
            for s in slots:
                kept = [m for i, m in enumerate(team) if i != s and m is not None]
                options = [d for d in donors if d['species'] not in {m['species'] for m in kept}
                           and (not d['item'] or d['item'] not in {m['item'] for m in kept})]
                team[s] = dict(copy.deepcopy(rng.choice(options)), slot=s + 1)
        try:
            P.validate_team(config, team)
            return team, attempt
        except ValueError:
            continue
    return None, tries


def main():
    config = A.load_config(str(HERE / 'benchmark.ling.json'))
    saved = P.read_json(RUN / 'eval2.tasks.json')
    tasks = {t['id']: t for t in saved['tasks']}
    rows = [json.loads(line) for line in (RUN / 'eval2.jsonl').read_text().splitlines()]
    pool = P.load_teams(config['starting_teams'])
    donors = [mon for row in pool for mon in row['team']]
    items = sorted({m['item'] for m in donors if m['item']} |
                   {m['item'] for row in P.load_teams(config['meta_teams']) for m in row['team'] if m['item']})

    candidates = []
    for start in saved['starting_teams']:
        candidates.append({'label': f"orig:{start['id']}", 'kind': 'original', 'source': start['id'], 'team': start['team']})
    drawn = {}
    for task in saved['tasks']:
        rng = random.Random(f"maskLab:{task['id']}")
        team, attempts = random_fill(config, task, donors, items, rng)
        drawn[task['id']] = (team, attempts)
    ids = {m for team, _ in drawn.values() if team for mon in team for m in mon['moves']
           if m and m == m.lower() and ' ' not in m}
    names = move_names(ids, config)
    for task in saved['tasks']:
        team, attempts = drawn[task['id']]
        if team:
            for mon in team:
                mon['moves'] = [names.get(m, m) for m in mon['moves']]
        candidates.append({'label': f"random:{task['id']}", 'kind': 'random', 'task_id': task['id'],
                           'source': task['source'], 'team': team, 'draws': attempts})
    for row in rows:
        if row['status'] == 'legal':
            candidates.append({'label': f"ling:{row['arm']}:{row['task_id']}", 'kind': 'ling', 'arm': row['arm'],
                               'task_id': row['task_id'], 'source': row['source'], 'team': row['team']})
    for c in candidates:
        c['paste'] = to_paste(c['team']) if c['team'] else None
    # Every candidate must pass the real validator before it is allowed into a battle.
    bad = 0
    for c in candidates:
        if c['team'] is None:
            continue
        try:
            P.validate_team(config, c['team'])
        except ValueError as error:
            c['team'], c['paste'], c['invalid'] = None, None, str(error)[:200]
            bad += 1
    P.save_json(RUN / 'candidates.json', {'candidates': candidates})
    kinds = {k: sum(1 for c in candidates if c['kind'] == k and c['team']) for k in ('original', 'random', 'ling')}
    print('candidates ready', kinds, '| random fills that failed in 50 draws:',
          sum(1 for c in candidates if c['kind'] == 'random' and not c['team']),
          '| failed final validation:', bad)


if __name__ == '__main__':
    main()
