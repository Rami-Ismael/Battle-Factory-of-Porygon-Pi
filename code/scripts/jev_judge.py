"""Experiment 6: Jev as the judge. Can Jev predict a finished team's win rate without battling it?

Every distinct team the three grids battled (all arms, the originals included) is shown to Jev with
the same format, rules and top-50 meta list the fills get, and two questions:

- band, a Score question with 10 win-rate bands, at most what a Score question allows. The bands
  follow the plan on the Jev Picks a Team page (5-point steps through the middle), except the ends:
  the random grid's teams mostly win under 10%, so "below 25%" splits at 10% and the top two merge.
- percent, a Choice question with labels "0" to "100", read as the win rate in percent.

Each turns into one predicted win rate: band as the probability-weighted band midpoint, percent as
the probability-weighted label. The battle win rate is the share of battles won against the 49 top-50
meta teams, one battle each, behaviour-cloning policy on both sides. One line per team goes to
~/vgc-data/jev/judge-<grid>.jsonl; a rerun skips teams already judged.

    /tmp/vgc-pilot/.venv/bin/python scripts/jev_judge.py run --grid all --dry-run
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_judge.py run --grid all --workers 6 --max-calls 6000
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_judge.py report
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import statistics
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jev_fill
import jev_run

GRIDS = ('heldout', 'meta', 'random')
BANDS = ['below 10%', '10 to 25%', '25 to 30%', '30 to 35%', '35 to 40%', '40 to 45%', '45 to 50%',
         '50 to 55%', '55 to 60%', '60% or more']
MIDPOINTS = [0.05, 0.175, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.65]
WIN_RATE = ('Each team plays one battle against each team in "meta", with the same battle policy on both sides. '
            'What share of those battles will the team in "team" win?')


def results_path(grid: str) -> Path:
    return jev_run.DATA / f'judge-{grid}.jsonl'


def team_id(paste: str) -> str:
    return hashlib.sha256(paste.encode()).hexdigest()[:16]


def teams(grid: str) -> list[dict[str, Any]]:
    """Every distinct battled team on a grid (one per paste), with its battle win rate and the labels that share it."""
    run_dir = jev_fill.GRIDS[grid]
    panels = json.loads((run_dir / 'panels.json').read_text())['panels']
    out: dict[str, dict[str, Any]] = {}
    for c in json.loads((run_dir / 'candidates.json').read_text())['candidates']:
        score = (panels.get(c['label']) or {}).get('score')
        if score is None or not c['paste']:
            continue
        t = out.setdefault(c['paste'], {'id': team_id(c['paste']), 'team': c['team'], 'win_rate': score, 'labels': []})
        t['labels'].append(c['label'])
    return list(out.values())


def questions() -> dict[str, jev_fill.Question]:
    return {'band': {'type': 'score', 'instructions': WIN_RATE + ' Pick the band.',
                     'criteria': {str(i): b for i, b in enumerate(BANDS)}},
            'percent': {'type': 'choice', 'instructions': WIN_RATE + ' Answer in percent.',
                        'criteria': {str(i): None for i in range(101)}}}


def state(team: jev_fill.Team, meta: list[str]) -> dict[str, Any]:
    return {'format': jev_fill.FORMAT_NAME, 'rules': jev_fill.RULES, 'meta': meta, 'team': team,
            'asking_about': 'win rate'}


def predict(answers: dict[str, jev_fill.Answer]) -> dict[str, float]:
    band, percent = answers['band']['probabilities'], answers['percent']['probabilities']
    return {'band': sum(p * MIDPOINTS[int(k)] for k, p in band.items()) / sum(band.values()),
            'percent': sum(p * int(k) for k, p in percent.items()) / sum(percent.values()) / 100}


def pending(grid: str) -> list[dict[str, Any]]:
    path = results_path(grid)
    done = {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
    return [t for t in teams(grid) if t['id'] not in done]


def estimate(todo: dict[str, list[dict[str, Any]]], meta: list[str]) -> int:
    return sum(jev_fill.estimate_tokens([state(t['team'], meta), q]) for ts in todo.values() for t in ts
               for q in questions().values())


def execute(grid: str, todo: list[dict[str, Any]], make: Callable[[], jev_fill.Chooser], meta: list[str],
            workers: int = 6) -> dict[str, int]:
    """Judge each team with one request per question, appending a line per team as it finishes."""
    lock, local, counts = threading.Lock(), threading.local(), {'judged': 0, 'transport': 0}
    path = results_path(grid)
    path.parent.mkdir(parents=True, exist_ok=True)

    def one(t: dict[str, Any]) -> None:
        if not hasattr(local, 'chooser'):
            local.chooser = make()
        s = state(t['team'], meta)
        try:
            answers = {name: local.chooser.choose(s, {name: q})[name] for name, q in questions().items()}
        except jev_fill.TransportError:
            with lock:
                counts['transport'] += 1
            return
        row = {'id': t['id'], 'grid': grid, 'labels': t['labels'], 'win_rate': t['win_rate'],
               'answers': answers, 'predicted': predict(answers)}
        with lock, path.open('a') as f:
            f.write(json.dumps(row) + '\n')
            counts['judged'] += 1
            if counts['judged'] % 100 == 0:
                print(f"{grid}: {counts['judged']}/{len(todo)} judged", flush=True)

    with cf.ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    return counts


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(v):                                  # ties share their mean rank
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def report() -> dict[str, Any]:
    out: dict[str, Any] = {}
    rows_all: list[dict[str, Any]] = []
    for grid in GRIDS:
        path = results_path(grid)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        rows_all += rows
        out[grid] = {'teams': len(rows), **{v: round(spearman([r['predicted'][v] for r in rows], [r['win_rate'] for r in rows]), 3)
                                             for v in ('band', 'percent') if len(rows) > 2}}
    out['pooled'] = {'teams': len(rows_all), **{v: round(spearman([r['predicted'][v] for r in rows_all], [r['win_rate'] for r in rows_all]), 3)
                                               for v in ('band', 'percent') if len(rows_all) > 2}}
    (jev_run.DATA / 'judge-report.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--grid', choices=[*GRIDS, 'all'], default='all')
    r.add_argument('--dry-run', action='store_true')
    r.add_argument('--max-calls', type=int, default=0, help='refuse to start above this many calls')
    r.add_argument('--workers', type=int, default=6)
    sub.add_parser('report')
    a = ap.parse_args()
    if a.cmd == 'report':
        print(json.dumps(report(), indent=1))
        return
    meta = jev_fill.meta_pastes()
    todo = {g: pending(g) for g in (GRIDS if a.grid == 'all' else (a.grid,))}
    calls = 2 * sum(map(len, todo.values()))
    tokens = estimate(todo, meta)
    print(f"{', '.join(f'{g} {len(t)}' for g, t in todo.items())} teams to judge, {calls} calls, "
          f"~{tokens:,} input tokens, ~${tokens * 0.042 / 1e6:.2f} at $0.042/M (estimate)", flush=True)
    if a.dry_run or not calls:
        return
    if a.max_calls and calls > a.max_calls:
        raise SystemExit(f'{calls} calls is over --max-calls {a.max_calls}')
    import jev_client
    choosers: list[jev_client.JevChooser] = []

    def make() -> jev_client.JevChooser:
        c = jev_client.JevChooser(jev_run.DATA / 'judge-calls.jsonl')
        choosers.append(c)
        return c
    for grid, ts in todo.items():
        print(f'{grid}: {execute(grid, ts, make, meta, a.workers)}', flush=True)
    spent = sum(c.dollars() for c in choosers)
    print(f"{sum(c.usage['calls'] for c in choosers)} calls, ${spent:.4f}", flush=True)
    for c in choosers:
        c.close()


if __name__ == '__main__':
    main()
