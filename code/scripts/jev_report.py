"""Jev's candidates on a Mask Lab grid: collect them, then report their win rate beside the other arms.

collect appends every legal Jev fill from ~/vgc-data/jev/<grid>.jsonl to the grid's candidates.json as
kind 'jev', label 'jev:<run>:<mask id>', after a timestamped backup. It never rebuilds the file, which
holds every other arm's fills, and skips labels already there. The grid's own battle step then battles
them:

    MASK_LAB_STARTERS=meta /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout_starters.py battle

(heldout is the default; an unknown value is refused). Before 2026-10-27 this branch's copy ignored the
setting and battled the held-out grid; it now matches the grid queue's copy panel for panel.

report reads the grid's panels.json. Win rate is the share of battles a team wins against the 49
top-50 meta teams, one battle each, the behaviour-cloning policy on both sides, listed slot order.
For every mask Jev filled it compares Jev's top-pick team and the mean of its three sampled teams with
the original team, the random legal fill, and each other arm that filled the same mask, using only
masks where both teams are legal. It prints one table per mask type and writes
~/vgc-data/jev/<grid>-report.json; the grid's own report.json is left alone.
"""
import json
import shutil
import statistics
import time
from typing import Any

import jev_fill
import jev_run

OTHER_ARMS = ('random', 'diffusion', 'textdiff', 'textdiff3b', 'llm_regmb', 'llm_baseline', 'llm_mb-original-v1',
              'mercury', 'ling')
# Ling ran three prompt versions on the random grid; compare the regmb one, the prompt Mercury got.
ARM_PREFIX = {'ling': 'ling:regmb@16384'}


def collect(grid: str) -> dict[str, int]:
    path = jev_fill.GRIDS[grid] / 'candidates.json'
    cands = json.loads(path.read_text())['candidates']
    have = {c['label'] for c in cands}
    rows = [json.loads(line) for line in jev_run.results_path(grid).read_text().splitlines() if line.strip()]
    new = []
    for r in rows:
        label = f"jev:{r['run']}:{r['task_id']}"
        if r['status'] == 'legal' and label not in have:
            new.append({'label': label, 'kind': 'jev', 'run': r['run'], 'task_id': r['task_id'], 'source': r['source'],
                        'team': r['team'], 'paste': jev_fill.team_paste(r['team'])})
            have.add(label)
    if new:
        shutil.copy(path, path.with_name(f"candidates.before-jev-{time.strftime('%Y%m%d-%H%M%S')}.json"))
        path.write_text(json.dumps({'candidates': cands + new}, indent=1) + '\n')
    return {'rows': len(rows), 'appended': len(new), 'candidates': len(cands) + len(new)}


def arm_score(scores: dict[str, float | None], arm: str, task_id: str) -> float | None:
    """An arm's win rate on one mask; an arm with several prompt versions is read at the one ARM_PREFIX names."""
    prefix = ARM_PREFIX.get(arm, arm) + ':'
    hits = [v for k, v in scores.items() if k.startswith(prefix) and k.endswith(':' + task_id)]
    assert len(hits) <= 1, f'{arm} has {len(hits)} scores on {task_id}; name its version in ARM_PREFIX'
    return hits[0] if hits else None


def mask_rows(grid: str) -> list[dict[str, Any]]:
    panels = json.loads((jev_fill.GRIDS[grid] / 'panels.json').read_text())['panels']
    scores = {label: p.get('score') for label, p in panels.items()}
    out = []
    for task in jev_fill.tasks(grid):
        top = scores.get(f"jev:top:{task['id']}")
        samples = [s for run in jev_run.SAMPLED if (s := scores.get(f"jev:{run}:{task['id']}")) is not None]
        if top is None and not samples:
            continue
        row = {'task_id': task['id'], 'task': task['task'], 'k': task['k'], 'source': task['source'],
               'original': scores.get(f"orig:{task['source']}"), 'jev_top': top,
               'jev_sampled': statistics.mean(samples) if samples else None, 'sampled_n': len(samples)}
        row.update({arm: arm_score(scores, arm, task['id']) for arm in OTHER_ARMS})
        out.append(row)
    return out


def paired(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    diffs = [r[a] - r[b] for r in rows if r[a] is not None and r[b] is not None]
    return {'n': len(diffs), 'mean': statistics.mean(diffs) if diffs else None}


def report(grid: str) -> dict[str, Any]:
    rows = mask_rows(grid)
    groups: dict[str, Any] = {}
    for task_type in ('pokemon', 'items', 'moves', 'abilities', 'natures', 'stats'):
        rs = [r for r in rows if r['task'] == task_type]
        if not rs:
            continue
        refs = ['original', *[arm for arm in OTHER_ARMS if any(r[arm] is not None for r in rs)]]
        groups[task_type] = {'masks': len(rs),
                             'jev_top_mean': statistics.mean([r['jev_top'] for r in rs if r['jev_top'] is not None] or [0]),
                             **{f'top_minus_{ref}': paired(rs, 'jev_top', ref) for ref in refs},
                             **{f'sampled_minus_{ref}': paired(rs, 'jev_sampled', ref) for ref in refs[:2]}}
    out = {'grid': grid, 'win_rate': 'share of battles won against the 49 top-50 meta teams, one battle each, '
                                     'behaviour-cloning policy on both sides, listed slot order',
           'groups': groups, 'masks': rows}
    (jev_run.DATA / f'{grid}-report.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


def fmt(cell: dict[str, Any]) -> str:
    return '–' if cell['mean'] is None else f"{cell['mean']:+.3f} ({cell['n']})"


def print_report(out: dict[str, Any]) -> None:
    print(f"Jev on the {out['grid']} grid. Win rate: {out['win_rate']}. Cells: mean difference (masks compared).")
    for task_type, g in out['groups'].items():
        refs = [k.removeprefix('top_minus_') for k in g if k.startswith('top_minus_')]
        print(f"\n{task_type}: {g['masks']} masks, Jev top-pick mean win rate {g['jev_top_mean']:.3f}")
        print('| | ' + ' | '.join(refs) + ' |')
        print('|---|' + '---:|' * len(refs))
        print('| Jev top − | ' + ' | '.join(fmt(g[f'top_minus_{r}']) for r in refs) + ' |')
        sampled = [fmt(g[f'sampled_minus_{r}']) if f'sampled_minus_{r}' in g else '' for r in refs]
        print('| Jev sampled − | ' + ' | '.join(sampled) + ' |')
