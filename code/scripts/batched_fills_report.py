"""Batched fills against batch-1 fills of the same masks (#443): legal counts per model and prompt, paired by mask
(McNemar's exact test), and fills per second and cost per 1,000 fills on the Modal L4.

The batch-1 side is #239's run (one mask per forward pass, ~/vgc-data/inplace_prompt_grid/modal-l4-239); the batched
side is the same four prompts, masks, mask seeds and sampler setting filled batch at a time
(~/vgc-data/inplace_prompt_grid/modal-l4-443). Within noise means McNemar's two-sided p is at least ALPHA for that
model and prompt (McNemar finds no difference); the owner dropped token-for-token agreement with batch 1 (2026-11-25), so legal counts are the
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
    seconds = sum(row['seconds'] for row in rows)
    per_second = len(rows) / seconds if seconds else 0.0
    dollars = 1000 / per_second / 3600 * L4_DOLLARS_PER_HOUR if per_second else 0.0
    return {'fills': len(rows), 'sampling_seconds': seconds, 'fills_per_second': per_second,
            'dollars_per_1000': dollars}


def batch_size(rows: Sequence[dict[str, Any]]) -> int:
    sizes = {row.get('batch_size', 1) for row in rows}
    if len(sizes) != 1:
        raise ValueError(f'rows filled at batch sizes {sorted(sizes)}')
    return sizes.pop()


def matched(one_at_a_time: list[dict[str, Any]], batched: list[dict[str, Any]]) -> tuple[list, list]:
    """Both sides cut to the masks both filled."""
    shared = {row['id'] for row in one_at_a_time} & {row['id'] for row in batched}
    return [row for row in one_at_a_time if row['id'] in shared], [row for row in batched if row['id'] in shared]


def one_pair(one_at_a_time: list[dict[str, Any]], batched: list[dict[str, Any]]) -> dict[str, Any]:
    """One model and prompt: the masks both sides filled, their legal counts, the paired test and each side's speed.
    A p-value of at least ALPHA means McNemar's test finds no difference between the two batchings."""
    one_at_a_time, batched = matched(one_at_a_time, batched)
    test = G.paired(one_at_a_time, batched)
    return {'masks': len(one_at_a_time), 'legal_one_at_a_time': sum(row['legal'] for row in one_at_a_time),
            'legal_batched': sum(row['legal'] for row in batched), **test, 'no_difference': test['p_value'] >= ALPHA,
            'one_at_a_time': speed(one_at_a_time) | {'batch': batch_size(one_at_a_time)},
            'batched': speed(batched) | {'batch': batch_size(batched)}}


def report(before_dir: Path = BEFORE, after_dir: Path = AFTER) -> dict[str, Any]:
    """Every model and prompt pair; missing lists the pairs a side has no fills for, and the verdict needs all."""
    models: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key in G.SMALL_FOUR:
        sides: dict[str, list[dict[str, Any]]] = {'one_at_a_time': [], 'batched': []}
        for prompt in PROMPTS:
            paths = [directory / prompt / key / 'fills.jsonl' for directory in (before_dir, after_dir)]
            absent = [str(path) for path in paths if not path.exists()]
            if absent:
                missing += absent
                continue
            one_at_a_time, batched = matched(G.read_fills(paths[0]), G.read_fills(paths[1]))
            models.setdefault(key, {})[prompt] = one_pair(one_at_a_time, batched)
            sides['one_at_a_time'] += one_at_a_time
            sides['batched'] += batched
        if key in models:
            models[key]['all prompts'] = {side: speed(rows) for side, rows in sides.items()}
    pairs = [value for prompts in models.values() for prompt, value in prompts.items() if prompt != 'all prompts']
    return {'before': str(before_dir), 'after': str(after_dir), 'alpha': ALPHA,
            'l4_dollars_per_hour': L4_DOLLARS_PER_HOUR, 'models': models, 'missing': missing,
            'pairs': len(pairs), 'no_difference_anywhere': bool(pairs) and not missing
            and all(value['no_difference'] for value in pairs)}


def verdict(result: dict[str, Any]) -> str:
    if result['missing']:
        return f"incomplete: {len(result['missing'])} fills files missing"
    if result['no_difference_anywhere']:
        return f"McNemar finds no difference for any model and prompt (every p ≥ {ALPHA})"
    return f"McNemar finds a difference for at least one model and prompt (p < {ALPHA})"


def table(result: dict[str, Any]) -> str:
    lines = [f'Verdict: {verdict(result)}.', '',
             '| Model | Prompt | Masks | Legal, one mask per forward pass | Legal, batched | '
             'Gained / lost (McNemar p) | Fills per second, one at a time → batched (batch size) |',
             '|---|---|---|---|---|---|---|']
    for key, prompts in result['models'].items():
        for prompt, value in prompts.items():
            if prompt == 'all prompts':
                continue
            lines.append(f"| {LABELS[key]} | {G.PROMPTS[prompt].name} | {value['masks']} | "
                         f"{value['legal_one_at_a_time']} | {value['legal_batched']} | "
                         f"{value['gained']} / {value['lost']} (p = {value['p_value']:.2f}) | "
                         f"{value['one_at_a_time']['fills_per_second']:.2f} → "
                         f"{value['batched']['fills_per_second']:.2f} ({value['batched']['batch']}) |")
    lines += ['', '| Model | Fills per second, one at a time | Fills per second, batched | Speed-up | '
              '$ per 1,000 fills, one at a time | $ per 1,000 fills, batched |', '|---|---|---|---|---|---|']
    for key, prompts in result['models'].items():
        alone, batched = prompts['all prompts']['one_at_a_time'], prompts['all prompts']['batched']
        speedup = batched['fills_per_second'] / alone['fills_per_second'] if alone['fills_per_second'] else 0.0
        lines.append(f"| {LABELS[key]} | {alone['fills_per_second']:.2f} | {batched['fills_per_second']:.2f} | "
                     f"{speedup:.1f}× | {alone['dollars_per_1000']:.2f} | {batched['dollars_per_1000']:.2f} |")
    if result['missing']:
        lines += ['', 'Missing fills files:', *(f'- `{path}`' for path in result['missing'])]
    return '\n'.join(lines)


ONE_AT_A_TIME, BATCHED = 'one mask per forward pass', 'many masks per forward pass'


def charts(result: dict[str, Any], out: Path) -> list[Path]:
    """Two ECharts pages: legal fills per model and prompt under each batching, and fills per second per model."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from echarts_chart import write_chart
    cells = [(key, prompt) for key, prompts in result['models'].items() for prompt in prompts
             if prompt != 'all prompts']
    names = [f"{LABELS[key]}\n{G.PROMPTS[prompt].name}" for key, prompt in cells]
    bar = {'type': 'bar', 'emphasis': {'focus': 'series'}, 'label': {'show': True, 'position': 'top'}}
    legal = {'tooltip': {'trigger': 'axis'}, 'legend': {'top': 0},
             'grid': {'left': 48, 'right': 16, 'bottom': 96, 'top': 40, 'containLabel': True},
             'dataZoom': [{'type': 'inside'}, {'type': 'slider', 'bottom': 8, 'height': 16}],
             'xAxis': {'type': 'category', 'data': names, 'axisLabel': {'interval': 0, 'rotate': 40, 'fontSize': 11}},
             'yAxis': {'type': 'value', 'name': 'legal first-try fills'},
             'series': [bar | {'name': ONE_AT_A_TIME, 'color': '#8a8984',
                               'data': [result['models'][k][p]['legal_one_at_a_time'] for k, p in cells]},
                        bar | {'name': BATCHED, 'color': '#2a6fdb',
                               'data': [result['models'][k][p]['legal_batched'] for k, p in cells]}]}
    keys = list(result['models'])
    two_decimals = {'show': True, 'position': 'top', 'formatter': 'js:p => p.value.toFixed(2)'}
    fast = {'tooltip': {'trigger': 'axis'}, 'legend': {'top': 0},
            'grid': {'left': 48, 'right': 16, 'bottom': 40, 'top': 40, 'containLabel': True},
            'xAxis': {'type': 'category', 'data': [LABELS[key] for key in keys]},
            'yAxis': {'type': 'value', 'name': 'fills per second'},
            'series': [bar | {'name': ONE_AT_A_TIME, 'color': '#8a8984', 'label': two_decimals,
                              'data': [result['models'][k]['all prompts']['one_at_a_time']['fills_per_second']
                                       for k in keys]},
                       bar | {'name': BATCHED, 'color': '#2a6fdb', 'label': two_decimals,
                              'data': [result['models'][k]['all prompts']['batched']['fills_per_second']
                                       for k in keys]}]}
    title = ('Batching leaves the legal counts unchanged' if result['no_difference_anywhere']
             else 'Legal counts: one mask per forward pass against many')
    return [write_chart(legal, out / 'batched-legal.html', title=title,
                        subtitle=f'Same masks, seeds and sampler on a Modal L4; {verdict(result)}',
                        note=('Legal: the fill parses, keeps every visible value, writes only names allowed in '
                              'Regulation M-B (the format), and Showdown accepts the team.'),
                        height=480),
            write_chart(fast, out / 'batched-speed.html', title='Fills per second on a Modal L4',
                        subtitle='Sampling time only, over the masks both batchings filled; model loading left out')]


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
