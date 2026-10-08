#!/usr/bin/env python3
"""Summarise runs/prompt-ab/eval.jsonl: legal rate per arm and task, paired tests, failure types, SVG chart."""
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

ARMS = ['baseline@4096', 'baseline@16384', 'regmb@16384']
NAME = {'baseline@4096': ('Original prompt', '4,096-token cap'),
        'baseline@16384': ('Original prompt', '16,384-token cap'),
        'regmb@16384': ('Reg M-B prompt', '16,384-token cap')}
TASKS = [('stats', 'Stat Points'), ('items', 'Items'), ('moves', 'Moves'),
         ('pokemon', 'Whole Pokémon'), ('abilities', 'Abilities'), ('natures', 'Stat Alignment')]
FAILURES = [('json', 'Unparsable or cut-off JSON'), ('structure', 'Malformed team'),
            ('preservation', 'Changed a fixed field'), ('validator', 'Broke a Reg M-B rule')]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_bucket(message):
    found = set()
    for part in message.split(';'):
        if "can't learn" in part or 'invalid move' in part:
            found.add('move not legal for that species')
        elif "can't have" in part:
            found.add('ability not legal for that species')
        elif 'does not exist' in part:
            found.add('item not in the format')
        elif 'Stat Points' in part:
            found.add('Stat Points over a limit')
        elif 'Clause' in part or part.strip().startswith('(You have'):
            found.add('duplicate species or item')
        else:
            found.add('other: ' + part.strip()[:50])
    return found


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    low = min(b, c)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(rows, a, b_arm, seed=7, boots=4000):
    by = {arm: {r['task_id']: r for r in rows if r['arm'] == arm} for arm in (a, b_arm)}
    ids = sorted(set(by[a]) & set(by[b_arm]))
    x = [(by[a][i]['status'] == 'legal', by[b_arm][i]['status'] == 'legal') for i in ids]
    only_a = sum(1 for p, q in x if p and not q)
    only_b = sum(1 for p, q in x if q and not p)
    diffs = [(q - p) for p, q in x]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(boots))
    return {'n': len(ids), 'only_first': only_a, 'only_second': only_b,
            'diff': sum(diffs) / len(diffs), 'ci95': [means[int(.025 * boots)], means[int(.975 * boots)]],
            'mcnemar_p': exact_mcnemar(only_a, only_b)}


def svg(summary, path):
    W, H = 900, 660
    out = []
    add = out.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">')
    add('<style>:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--a0:#c3c2b7;--a1:#898781;--a2:#2a78d6}'
        '@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--a0:#62615c;--a1:#9d9c95;--a2:#3987e5}}'
        '.t{fill:var(--ink)}.s{fill:var(--ink2)}.m{fill:var(--muted)}.g{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}'
        '.c0{fill:var(--a0)}.c1{fill:var(--a1)}.c2{fill:var(--a2)}.w{stroke:var(--ink2);stroke-width:1.5;fill:none}'
        'text{font-size:12px}.h{font-size:19px;font-weight:600}.b{font-weight:600}.v{font-size:12px;font-weight:600}</style>')
    add(f'<rect width="{W}" height="{H}" fill="var(--surface)"/>')
    n = summary['arms'][ARMS[0]]['n']
    add('<text class="t h" x="40" y="38">Share of Ling completions that pass the Reg M-B validator</text>')
    add(f'<text class="s" x="40" y="58">One attempt per masked task · {n} tasks per arm from 5 random legal starting teams · bars show 95% Wilson intervals</text>')

    # chart 1: overall legal rate, horizontal bars
    x0, x1, top, step = 250, 830, 96, 54
    scale = lambda p: x0 + (x1 - x0) * p
    for t in (0, .25, .5, .75, 1):
        add(f'<line class="g" x1="{scale(t):.1f}" x2="{scale(t):.1f}" y1="{top - 14}" y2="{top + 3 * step - 12}"/>')
        add(f'<text class="m" x="{scale(t):.1f}" y="{top + 3 * step + 6}" text-anchor="middle">{int(t * 100)}%</text>')
    for i, arm in enumerate(ARMS):
        a = summary['arms'][arm]
        y = top + i * step
        title, sub = NAME[arm]
        add(f'<text class="t b" x="40" y="{y + 6}">{title}</text><text class="s" x="40" y="{y + 23}">{sub}</text>')
        w = scale(a['rate']) - x0
        r = min(4, w / 2)
        add(f'<path class="c{i}" d="M{x0},{y - 4} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{22 - 2 * r} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z">'
            f'<title>{title}, {sub}: {a["legal"]}/{a["n"]} legal ({a["rate"]:.1%}); 95% interval {a["ci"][0]:.1%} to {a["ci"][1]:.1%}</title></path>')
        lo, hi = scale(a['ci'][0]), scale(a['ci'][1])
        add(f'<path class="w" d="M{lo:.1f},{y + 7} H{hi:.1f} M{lo:.1f},{y + 1} v12 M{hi:.1f},{y + 1} v12"/>')
        add(f'<text class="t v" x="{hi + 8:.1f}" y="{y + 11}">{a["rate"] * 100:.0f}% <tspan class="m" font-weight="400">({a["legal"]}/{a["n"]})</tspan></text>')

    # chart 2: by task, grouped columns
    gx0, gx1, gy0, gy1 = 70, 860, 372, 600
    ys = lambda p: gy1 - (gy1 - gy0) * p
    add(f'<text class="t b" x="40" y="{gy0 - 36}">Legal rate by task</text>')
    lx = 200
    for i, arm in enumerate(ARMS):
        add(f'<rect class="c{i}" x="{lx}" y="{gy0 - 45}" width="10" height="10" rx="2"/>')
        short = {'baseline@4096': 'Original prompt, 4,096 cap', 'baseline@16384': 'Original prompt, 16,384 cap',
                 'regmb@16384': 'Reg M-B prompt, 16,384 cap'}[arm]
        add(f'<text class="s" x="{lx + 15}" y="{gy0 - 36}">{short}</text>')
        lx += {0: 185, 1: 195, 2: 0}[i]
    p, k = summary['paired']['prompt_effect (baseline@16384 -> regmb@16384)'], summary['paired']['token_effect (baseline@4096 -> baseline@16384)']
    pt = lambda v: f'{v * 100:+.0f}'
    add(f'<text class="s" x="40" y="{top + 3 * step + 34}">Same 16,384 cap, prompt only: <tspan class="t b">{pt(p["diff"])} points</tspan> (paired 95% interval {pt(p["ci95"][0])} to {pt(p["ci95"][1])}). '
        f'Raising the cap alone: {pt(k["diff"])} points (interval {pt(k["ci95"][0])} to {pt(k["ci95"][1])}).</text>')
    for t in (0, .25, .5, .75, 1):
        add(f'<line class="{"ax" if t == 0 else "g"}" x1="{gx0}" x2="{gx1}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>')
        add(f'<text class="m" x="{gx0 - 8}" y="{ys(t) + 4:.1f}" text-anchor="end">{int(t * 100)}%</text>')
    gw = (gx1 - gx0) / len(TASKS)
    bw, gap = 28, 4
    for j, (task, label) in enumerate(TASKS):
        cx = gx0 + gw * (j + .5)
        left = cx - (3 * bw + 2 * gap) / 2
        for i, arm in enumerate(ARMS):
            cell = summary['by_task'][arm][task]
            bx = left + i * (bw + gap)
            h = (gy1 - gy0) * cell['rate']
            if h > 0.5:
                r = min(4, h)
                add(f'<path class="c{i}" d="M{bx:.1f},{gy1} v{-(h - r):.1f} a{r},{r} 0 0 1 {r},{-r} h{bw - 2 * r} a{r},{r} 0 0 1 {r},{r} v{h - r:.1f} z">'
                    f'<title>{NAME[arm][0]}, {NAME[arm][1]} · {label}: {cell["legal"]}/{cell["n"]} legal ({cell["rate"]:.0%})</title></path>')
            add(f'<text class="{"t v" if i == 2 else "m"}" x="{bx + bw / 2:.1f}" y="{ys(cell["rate"]) - 5:.1f}" text-anchor="middle">{cell["rate"] * 100:.0f}</text>')
        add(f'<text class="t" x="{cx:.1f}" y="{gy1 + 20}" text-anchor="middle">{label}</text>')
        add(f'<text class="m" x="{cx:.1f}" y="{gy1 + 35}" text-anchor="middle">n = {summary["by_task"][ARMS[0]][task]["n"]} per arm</text>')
    add('</svg>')
    Path(path).write_text('\n'.join(out) + '\n')


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/prompt-ab/eval.jsonl')
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    transport = collections.Counter(r['arm'] for r in rows if r['status'] == 'transport')
    rows = [r for r in rows if r['status'] != 'transport']
    summary = {'arms': {}, 'by_task': {}, 'failures': {}, 'rule_types': {}, 'cost': {}, 'transport_excluded': dict(transport)}
    for arm in ARMS:
        mine = [r for r in rows if r['arm'] == arm]
        legal = sum(r['status'] == 'legal' for r in mine)
        summary['arms'][arm] = {'n': len(mine), 'legal': legal, 'rate': legal / len(mine) if mine else 0,
                                'ci': wilson(legal, len(mine))}
        summary['by_task'][arm] = {}
        for task, _ in TASKS:
            sub = [r for r in mine if r['task'] == task]
            k = sum(r['status'] == 'legal' for r in sub)
            summary['by_task'][arm][task] = {'n': len(sub), 'legal': k, 'rate': k / len(sub) if sub else 0}
        stages = collections.Counter(r['stage'] for r in mine if r['status'] == 'illegal')
        summary['failures'][arm] = {key: stages.get(key, 0) for key, _ in FAILURES}
        rules = collections.Counter()
        for r in mine:
            if r['stage'] == 'validator':
                rules.update(rule_bucket(r['error']))
        summary['rule_types'][arm] = dict(rules.most_common())
        usage = [r['usage'] for r in mine if r.get('usage')]
        summary['cost'][arm] = {
            'dollars': round(sum(u.get('cost', 0) for u in usage), 4), 'calls_with_usage': len(usage),
            'mean_prompt_tokens': round(sum(u['prompt_tokens'] for u in usage) / max(1, len(usage))),
            'mean_completion_tokens': round(sum(u['completion_tokens'] for u in usage) / max(1, len(usage))),
            'mean_reasoning_tokens': round(sum((u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
                                               for u in usage) / max(1, len(usage)))}
    summary['paired'] = {'prompt_effect (baseline@16384 -> regmb@16384)': paired(rows, 'baseline@16384', 'regmb@16384'),
                         'token_effect (baseline@4096 -> baseline@16384)': paired(rows, 'baseline@4096', 'baseline@16384'),
                         'both (baseline@4096 -> regmb@16384)': paired(rows, 'baseline@4096', 'regmb@16384')}
    structure = collections.Counter(r['error'][:70] for r in rows if r['stage'] == 'structure')
    summary['structure_errors'] = dict(structure.most_common(6))
    save = src.with_name(f'summary-{src.stem}.json')
    save.write_text(json.dumps(summary, indent=2) + '\n')
    svg(summary, src.with_name(f'legal-rate-{src.stem}.svg'))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
