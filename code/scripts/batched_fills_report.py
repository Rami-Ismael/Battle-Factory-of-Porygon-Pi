"""Batched fills against batch-1 fills of the same masks (#443): legal counts per model and prompt, paired by mask
(McNemar's exact test), and fills per second and cost per 1,000 fills on the Modal L4.

The batch-1 side is #239's run (one mask per forward pass, ~/vgc-data/inplace_prompt_grid/modal-l4-239); the batched
side is the same four prompts, masks, mask seeds and sampler setting filled batch at a time
(~/vgc-data/inplace_prompt_grid/modal-l4-443). Within noise means McNemar's two-sided p is at least ALPHA for that
model and prompt; the owner dropped token-for-token agreement with batch 1 (2026-11-25), so legal counts are the
check. Speed counts sampling time only, as each row records it (the batch's time shared out over its rows); model
loading and Showdown are left out of both sides.

    python code/scripts/batched_fills_report.py [--before D] [--after D] [--out D]
"""
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import inplace_prompt_grid as G

HOME = Path.home() / 'vgc-data/inplace_prompt_grid'
BEFORE, AFTER = HOME / 'modal-l4-239', HOME / 'modal-l4-443'
PROMPTS = ('inplace_header', 'inplace_instructions', 'inplace_header_example', 'inplace_example')
L4_DOLLARS_PER_HOUR = 0.80          # Modal's L4 price, as the runners' timeouts state it
ALPHA = 0.05
LABELS = {'diffugpt_s': 'DiffuGPT small', 'diffugpt_m': 'DiffuGPT medium', 'qwen25c': 'Masked Qwen2.5 Coder',
          'qwen3': 'Masked Qwen3'}


def speed(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Fills per second of sampling time and dollars per 1,000 fills at the L4's price."""
    seconds = sum(r['seconds'] for r in rows)
    per_second = len(rows) / seconds if seconds else 0.0
    dollars = 1000 / per_second / 3600 * L4_DOLLARS_PER_HOUR if per_second else 0.0
    return {'fills': len(rows), 'sampling_seconds': seconds, 'fills_per_second': per_second,
            'dollars_per_1000': dollars}


def batch_size(rows: Sequence[dict[str, Any]]) -> int:
    sizes = {r.get('batch_size', 1) for r in rows}
    if len(sizes) != 1:
        raise ValueError(f'rows filled at batch sizes {sorted(sizes)}')
    return sizes.pop()


def one_pair(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    """One model and prompt: the masks both sides filled, their legal counts, the paired test and each side's speed."""
    shared = {r['id'] for r in before} & {r['id'] for r in after}
    before = [r for r in before if r['id'] in shared]
    after = [r for r in after if r['id'] in shared]
    test = G.paired(before, after)
    return {'masks': len(shared), 'legal_batch1': sum(r['legal'] for r in before),
            'legal_batched': sum(r['legal'] for r in after), **test, 'within_noise': test['p_value'] >= ALPHA,
            'batch1': speed(before) | {'batch': batch_size(before)},
            'batched': speed(after) | {'batch': batch_size(after)}}


def report(before_dir: Path = BEFORE, after_dir: Path = AFTER) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for key in G.SMALL_FOUR:
        for prompt in PROMPTS:
            paths = [d / prompt / key / 'fills.jsonl' for d in (before_dir, after_dir)]
            if all(p.exists() for p in paths):
                models.setdefault(key, {})[prompt] = one_pair(G.read_fills(paths[0]), G.read_fills(paths[1]))
    for key, prompts in models.items():
        rows = {side: [r for p in prompts for d in (before_dir if side == 'batch1' else after_dir,)
                       for r in G.read_fills(d / p / key / 'fills.jsonl')] for side in ('batch1', 'batched')}
        prompts['all prompts'] = {side: speed(rows[side]) for side in rows}
    return {'before': str(before_dir), 'after': str(after_dir), 'alpha': ALPHA,
            'l4_dollars_per_hour': L4_DOLLARS_PER_HOUR, 'models': models,
            'all_within_noise': all(v['within_noise'] for m in models.values() for p, v in m.items()
                                    if p != 'all prompts')}


def table(result: dict[str, Any]) -> str:
    lines = ['| Model | Prompt | Masks | Legal, batch 1 | Legal, batched | Gained / lost (McNemar p) | '
             'Fills per second, batch 1 → batched (batch size) |', '|---|---|---|---|---|---|---|']
    for key, prompts in result['models'].items():
        for prompt, v in prompts.items():
            if prompt == 'all prompts':
                continue
            lines.append(f"| {LABELS[key]} | {G.PROMPTS[prompt].name} | {v['masks']} | {v['legal_batch1']} | "
                         f"{v['legal_batched']} | {v['gained']} / {v['lost']} (p = {v['p_value']:.2f}) | "
                         f"{v['batch1']['fills_per_second']:.2f} → {v['batched']['fills_per_second']:.2f} "
                         f"({v['batched']['batch']}) |")
    lines += ['', '| Model | Fills per second, batch 1 | Fills per second, batched | Speed-up | '
              '$ per 1,000 fills, batch 1 | $ per 1,000 fills, batched |', '|---|---|---|---|---|---|']
    for key, prompts in result['models'].items():
        one, many = prompts['all prompts']['batch1'], prompts['all prompts']['batched']
        up = many['fills_per_second'] / one['fills_per_second'] if one['fills_per_second'] else 0.0
        lines.append(f"| {LABELS[key]} | {one['fills_per_second']:.2f} | {many['fills_per_second']:.2f} | {up:.1f}× | "
                     f"{one['dollars_per_1000']:.2f} | {many['dollars_per_1000']:.2f} |")
    return '\n'.join(lines)


def charts(result: dict[str, Any], out: Path) -> list[Path]:
    """Two ECharts pages: legal fills per model and prompt at batch 1 and batched, and fills per second per model."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from echarts_chart import write_chart
    cells = [(key, p) for key, prompts in result['models'].items() for p in prompts if p != 'all prompts']
    names = [f"{LABELS[k]}\n{G.PROMPTS[p].name}" for k, p in cells]
    bar = {'type': 'bar', 'emphasis': {'focus': 'series'}, 'label': {'show': True, 'position': 'top'}}
    legal = {'tooltip': {'trigger': 'axis'}, 'legend': {'top': 0},
             'grid': {'left': 48, 'right': 16, 'bottom': 96, 'top': 40, 'containLabel': True},
             'xAxis': {'type': 'category', 'data': names, 'axisLabel': {'interval': 0, 'rotate': 40, 'fontSize': 11}},
             'yAxis': {'type': 'value', 'name': 'legal first-try fills'},
             'series': [bar | {'name': 'one mask per forward pass', 'color': '#8a8984',
                               'data': [result['models'][k][p]['legal_batch1'] for k, p in cells]},
                        bar | {'name': 'batched', 'color': '#2a6fdb',
                               'data': [result['models'][k][p]['legal_batched'] for k, p in cells]}]}
    keys = list(result['models'])
    fast = {'tooltip': {'trigger': 'axis'}, 'legend': {'top': 0},
            'grid': {'left': 48, 'right': 16, 'bottom': 40, 'top': 40, 'containLabel': True},
            'xAxis': {'type': 'category', 'data': [LABELS[k] for k in keys]},
            'yAxis': {'type': 'value', 'name': 'fills per second'},
            'series': [bar | {'name': 'one mask per forward pass', 'color': '#8a8984',
                              'label': {'show': True, 'position': 'top', 'formatter': 'js:p => p.value.toFixed(2)'},
                              'data': [result['models'][k]['all prompts']['batch1']['fills_per_second'] for k in keys]},
                       bar | {'name': 'batched', 'color': '#2a6fdb',
                              'label': {'show': True, 'position': 'top', 'formatter': 'js:p => p.value.toFixed(2)'},
                              'data': [result['models'][k]['all prompts']['batched']['fills_per_second']
                                       for k in keys]}]}
    within = 'every model and prompt within McNemar noise' if result['all_within_noise'] else \
        'at least one model and prompt outside McNemar noise'
    return [write_chart(legal, out / 'batched-legal.html', title='Batched fills keep the legal counts',
                        subtitle=f'Same masks, seeds and sampler on a Modal L4; {within} (p ≥ {ALPHA})',
                        note='Legal: parses, keeps every visible value, Reg M-B names, and Showdown accepts it.',
                        height=480),
            write_chart(fast, out / 'batched-speed.html', title='Fills per second on a Modal L4',
                        subtitle='Sampling time only, all four prompts together; model loading left out')]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--before', type=Path, default=BEFORE)
    parser.add_argument('--after', type=Path, default=AFTER)
    parser.add_argument('--out', type=Path, default=None, help='default: --after')
    args = parser.parse_args(argv)
    out = args.out or args.after
    result = report(args.before, args.after)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'batched-report.json').write_text(json.dumps(result, indent=1) + '\n')
    (out / 'batched-report.md').write_text(table(result) + '\n')
    charts(result, out / 'figure')
    print(table(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
