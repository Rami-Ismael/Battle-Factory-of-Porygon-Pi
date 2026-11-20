"""Battle the small masked models' legal held-out grid teams from today's in-place prompt (#171), as the Jev and
Mercury rows were battled (scripts/jev_rebattle.py). Pre-registration: code/docs/inplace-prompt-heldout-battles.md.

Each model's teams are its first legal draw per mask after #169's validator loop (inplace_prompt_grid.py redraw,
then teams): D/inplace_header/<key>/teams.json. Every team plays the Mask Lab's 49 top-50 teams paste for paste,
then the held-out teams (ranks 51-100) minus the five unmasked teams (45), 8 battles per opponent, the
behaviour-cloning policy on both sides, six Pokemon shuffled every battle, into a private database. Two controls
share the database: the five unmasked teams at 32 battles per opponent (they set the same-mask comparison and must
land near their 47.6% from the Jev re-battle), and five uniform-legal teams from hps_reg_mb_100k.jsonl (near 0).

    /tmp/vgc-pilot/.venv/bin/python scripts/inplace_prompt_battle.py battle --stage controls   # needs /tmp/vgc-pilot
    /tmp/vgc-pilot/.venv/bin/python scripts/inplace_prompt_battle.py check-controls          # exit 1 on a miss
    /tmp/vgc-pilot/.venv/bin/python scripts/inplace_prompt_battle.py battle [--dry-run]
    /tmp/vgc-pilot/.venv/bin/python scripts/inplace_prompt_battle.py report

report writes D/battles-report.json: per model and opponent set, usable win rate over the 270 masks (a mask with no
legal team scores 0) with 95% intervals from resampling masks and from resampling the five unmasked teams, the
win rate when legal, and on the same masks the model minus the unmasked team it was cut from, minus the random
legal fill and minus Jev's top pick (those two from ~/vgc-data/jev/rebattle-heldout.json, same design).
"""
import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import jev_rebattle as J

D = Path.home() / 'vgc-data/inplace_prompt_grid'
DB = D / 'battles.sqlite'
OUT = D / 'battles-report.json'
PROMPT = 'inplace_header'
MODELS = ('diffugpt_s', 'diffugpt_m', 'qwen25c', 'qwen3')
UNIFORM = Path.home() / 'vgc-data/teams/hps_reg_mb_100k.jsonl'
PER_CELL, UNMASKED_PER_CELL, SEED, BATCH, CONTROLS = 8, 32, 20261120, 25, 5
REFERENCE = Path.home() / 'vgc-data/jev/rebattle-heldout.json'


def model_teams(key: str) -> dict:
    return json.loads((D / PROMPT / key / 'teams.json').read_text())


def uniform_controls(n: int = CONTROLS, seed: int = SEED) -> list[str]:
    """n uniform-legal teams, drawn with a fixed seed from the first 10,000 rows."""
    rows = [json.loads(line)['team'] for _, line in zip(range(10_000), UNIFORM.open())]
    return random.Random(seed).sample(rows, n)


def unmasked() -> dict[str, str]:
    """{source: paste} for the five unmasked teams every mask is cut from."""
    cands = json.loads((J.GRID / 'candidates.json').read_text())['candidates']
    return {c['source']: c['paste'] for c in cands if c['kind'] == 'original'}


def stat_points_full(paste: str) -> tuple[int, int]:
    """(Pokemon spending all 66 Stat Points, Pokemon) in a paste."""
    totals = [sum(int(n) for n in re.findall(r'(\d+) \w+', line)) for line in paste.splitlines()
              if line.startswith('EVs:')]
    return sum(t == 66 for t in totals), 6


def battle(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(J.SRC))
    import matchup_db as M
    import top50 as T50
    from showdown_validator import Validator
    con = M.connect(DB)
    pid = M.get_policy(con, note='in-place prompt held-out grid battles (#171), slots shuffled, 2026-11-20')
    assert '+slots-shuffled' in con.execute('SELECT row_policy FROM policy WHERE policy_id=?', (pid,)).fetchone()[0]
    val = Validator()
    try:
        with con:
            fills = {}
            for key in MODELS:
                for task_id, t in model_teams(key)['teams'].items():
                    team_id = M.add_team(con, t['paste'], f'inplace-171:{key}', val)[0]
                    legal, error = con.execute('SELECT legal, legal_error FROM team WHERE team_id=?', (team_id,)).fetchone()
                    if legal != 1:
                        raise SystemExit(f'{key} {task_id}: Showdown rejects the stored paste: {error}')
                    fills[(key, task_id)] = team_id
            sources = {s: M.add_team(con, p, 'inplace-171:unmasked', val)[0] for s, p in unmasked().items()}
            uniform = [M.add_team(con, p, 'inplace-171:uniform', val)[0] for p in uniform_controls()]
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):          # as matchup_db.seed: a team listed twice counts twice
                tid = M.add_team(con, Path(T50.resolve(t['id'])).read_text(), 'vgcpastes', val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                            'DO UPDATE SET weight = weight + 1', (tid,))
        hcols = [t for t in M.seed_heldout(con, val) if t not in set(sources.values())]
    finally:
        val.close()
    cols = M.set_members(con, 'top50')
    lab = sqlite3.connect('file:' + str(Path.home() / 'vgc-data/mask_lab_matrix.sqlite') + '?mode=ro&immutable=1', uri=True)
    want = {h for (h,) in lab.execute("SELECT canon_hash FROM team JOIN team_set USING (team_id) WHERE set_name='top50'")}
    got = {h for (h,) in con.execute(f"SELECT canon_hash FROM team WHERE team_id IN ({','.join('?' * len(cols))})", cols)}
    if want != got:
        raise SystemExit(f"top-50 columns differ from the Mask Lab's: {len(want ^ got)} pastes")
    rows = sorted(set(fills.values()) - set(uniform) - set(sources.values()))
    main_rows = sorted(set(sources.values()))
    n = ((len(rows) + len(uniform)) * PER_CELL + len(main_rows) * UNMASKED_PER_CELL) * (len(cols) + len(hcols))
    print(f'policy {pid} · {len(fills)} legal fills, {len(set(fills.values()))} distinct · {len(main_rows)} unmasked · '
          f'{len(uniform)} uniform-legal · {len(cols)} top-50 and {len(hcols)} held-out opponents · <= {n} battles',
          flush=True)
    if args.dry_run:
        con.close()
        return
    t0, done = time.time(), 0
    controls = [(main_rows, cols, UNMASKED_PER_CELL, 0, 'unmasked v top-50'),
                (uniform, cols, PER_CELL, 30000, 'uniform-legal v top-50')]
    plan = controls if args.stage == 'controls' else [
        *controls, (main_rows, hcols, UNMASKED_PER_CELL, 40000, 'unmasked v held-out'),
        (uniform, hcols, PER_CELL, 45000, 'uniform-legal v held-out'),
        (rows, cols, PER_CELL, 0, 'top-50'), (rows, hcols, PER_CELL, 50000, 'held-out')]
    for group, opponents, per_cell, offset, label in plan:   # evaluation=True: the unmasked teams are held-out teams
        for i in range(0, len(group), BATCH):
            done += M.ensure(con, pid, group[i:i + BATCH], opponents, per_cell, seed=SEED + offset + i,
                             note=f'inplace_prompt_battle {label}', evaluation=True)
            print(f'{label} rows {min(i + BATCH, len(group))}/{len(group)} · {done} battles · '
                  f'{time.time() - t0:.0f}s', flush=True)
    con.close()


def check_controls(_args: argparse.Namespace) -> int:
    """The scorer's controls against the top-50 teams, before any fill is battled: the five unmasked teams within
    3 points of their Jev re-battle mean (a fresh database and seed: the repeat probe), and the uniform-legal teams
    below 10%. Exit 1 when either misses."""
    sys.path.insert(0, str(J.SRC))
    import matchup_db as M
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    pid = con.execute("SELECT max(policy_id) FROM policy WHERE row_policy LIKE '%+slots-shuffled'").fetchone()[0]
    canon = {h: t for t, h in con.execute('SELECT team_id, canon_hash FROM team')}
    tid = lambda paste: canon[M.canon_hash(M.canonical_paste(paste))]
    cols = {t for (t,) in con.execute("SELECT team_id FROM team_set WHERE set_name='top50'")}
    scores = J.team_scores(con, pid, cols, 'top50')
    ours = {s: scores[tid(p)][0] for s, p in unmasked().items()}
    jev = json.loads(REFERENCE.read_text())['sets']['top50']['unmasked']
    uniform = [scores[tid(p)][0] for p in uniform_controls()]
    gap = statistics.mean(ours.values()) - statistics.mean(jev.values())
    out = {'unmasked': ours, 'unmasked_jev_rebattle': jev, 'gap': round(gap, 4), 'uniform_legal': uniform,
           'uniform_mean': round(statistics.mean(uniform), 4)}
    out['pass'] = abs(gap) <= 0.03 and out['uniform_mean'] < 0.10
    (D / 'preflight').mkdir(parents=True, exist_ok=True)
    (D / 'preflight' / 'battle-controls.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)
    return 0 if out['pass'] else 1


def report(_args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(J.SRC))
    import matchup_db as M
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    pid = con.execute("SELECT max(policy_id) FROM policy WHERE row_policy LIKE '%+slots-shuffled'").fetchone()[0]
    canon = {h: t for t, h in con.execute('SELECT team_id, canon_hash FROM team')}
    tid = lambda paste: canon[M.canon_hash(M.canonical_paste(paste))]
    sources = {s: tid(p) for s, p in unmasked().items()}
    uniform = [tid(p) for p in uniform_controls()]
    tasks = {t['id']: t for t in json.loads((J.GRID / 'eval2.tasks.json').read_text())['tasks']}
    kinds = sorted({t['task'] for t in tasks.values()})
    reference = json.loads(REFERENCE.read_text())['sets']
    opponents = {'top50': ('top50', {t for (t,) in con.execute("SELECT team_id FROM team_set WHERE set_name='top50'")}),
                 'heldout': ('heldout51_100', {t for (t,) in con.execute(
                     "SELECT team_id FROM team_set WHERE set_name='heldout51_100'")} - set(sources.values()))}
    teams = {key: model_teams(key) for key in MODELS}
    out = {'date': time.strftime('%Y-%m-%d'), 'policy_id': pid, 'battles_per_opponent': PER_CELL,
           'unmasked_battles_per_opponent': UNMASKED_PER_CELL,
           'policy': 'bc_100.zip on both sides, six Pokemon shuffled every battle', 'sets': {}}
    for name, (set_name, cols) in opponents.items():
        scores = J.team_scores(con, pid, cols, set_name)
        ref = reference[name]
        res = {'opponents': len(cols),
               'controls': {'unmasked': {s: scores[t][0] for s, t in sources.items()},
                            'unmasked_jev_rebattle': ref['unmasked'],
                            'uniform_legal': [scores[t][0] for t in uniform]}, 'models': {}}
        for key in MODELS:
            fill = {i: scores[tid(t['paste'])][0] for i, t in teams[key]['teams'].items()}
            row = {'legal': len(fill), 'legal_first_try': teams[key]['legal_first_try'],
                   'legal_by_draw': teams[key]['legal_by_draw'], 'draws_made': teams[key]['draws_made'], 'kinds': {}}
            for kind in [*kinds, 'all']:
                ids = [i for i, t in tasks.items() if kind == 'all' or t['task'] == kind]
                k = {'masks': len(ids), 'legal': sum(i in fill for i in ids),
                     'usable': J.interval([(tasks[i]['source'], fill.get(i, 0)) for i in ids], 'mean')}
                legal_ids = [i for i in ids if i in fill]
                if legal_ids:
                    k['when_legal'] = round(statistics.mean(fill[i] for i in legal_ids), 4)
                for other, get in (('unmasked', lambda i: res['controls']['unmasked'][tasks[i]['source']]),
                                   ('random', lambda i: ref['scores']['random'].get(i)),
                                   ('jev', lambda i: ref['scores']['jev'].get(i))):
                    pairs = [(tasks[i]['source'], fill[i] - get(i)) for i in legal_ids if get(i) is not None]
                    if len(pairs) >= 2:
                        k[f'minus_{other}'] = J.interval(pairs, 'diff')
                row['kinds'][kind] = k
            full = [stat_points_full(t['paste']) for t in teams[key]['teams'].values()]
            row['stat_points_all_66'] = [sum(a for a, _ in full), sum(b for _, b in full)]
            row['recovered_unmasked_team'] = sum(t['recovered'] for t in teams[key]['teams'].values())
            res['models'][key] = row
        out['sets'][name] = res
        c = res['controls']
        print(f"\nagainst {name} ({len(cols)} teams): unmasked " +
              ', '.join(f'{s} {v:.3f} (Jev re-battle {c["unmasked_jev_rebattle"][s]:.3f})' for s, v in c['unmasked'].items()) +
              f"; uniform-legal {', '.join(f'{v:.3f}' for v in c['uniform_legal'])}")
        for key, row in res['models'].items():
            a = row['kinds']['all']
            print(f"{key:11} legal {row['legal']:>3}/270 usable {a['usable']['mean']:.3f} "
                  f"[{a['usable']['ci_teams'][0]:.3f}, {a['usable']['ci_teams'][1]:.3f}] when legal "
                  f"{a.get('when_legal', float('nan')):.3f}" +
                  ''.join(f" · minus {o} {a[f'minus_{o}']['diff']:+.3f} [{a[f'minus_{o}']['ci_masks'][0]:+.3f}, "
                          f"{a[f'minus_{o}']['ci_masks'][1]:+.3f}]" for o in ('unmasked', 'random', 'jev')
                          if f'minus_{o}' in a))
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print('wrote', OUT)
    return out


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('battle')
    b.add_argument('--dry-run', action='store_true', help='add the teams, count the battles')
    b.add_argument('--stage', choices=('controls', 'all'), default='all',
                   help='controls: only the unmasked and uniform-legal teams against the top-50 teams')
    sub.add_parser('check-controls')
    sub.add_parser('report')
    a = p.parse_args()
    if a.command == 'check-controls':
        sys.exit(check_controls(a))
    battle(a) if a.command == 'battle' else report(a)
