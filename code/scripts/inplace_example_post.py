"""Figures and issue comments for #239: one fixed example team in the in-place prompt, on both bases.

Each model fills four prompts in one Modal container: no example team or the example team, on the header-only prompt
or the task-statement prompt. The example prompts fill only the masks that fit under 1,024 tokens, so every pair is
compared on the masks both its prompts fill (inplace_prompt_grid.shared_ids). Figures, storyboarded in
code/docs/inplace-prompt-example.md:
    model    one model: legal first-try fills, the four prompts as a 2 x 2 matrix (example team down, base across)
    copies   every model: the share of filled values that repeat the example team, with and without it in the prompt
    compare  every model: legal first-try fills under each of the four prompts

    figure --dir D --key K --out F           the model matrix
    figure --dir D --copies --out F          the copy matrix
    figure --dir D --compare --out F         the four-prompt matrix
    comment --dir D --key K --url U          one model's comment; U holds {name} and {theme}
    comment --dir D --compare --url U        the comment comparing every model

Run with any Python that has the repo's code/scripts and code/src on its path.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import inplace_prompt_grid as G
from echarts_chart import write_chart
from inplace_prompt_post import LABELS, TASK_NAMES, picture

BASES = (('inplace_header', 'inplace_header_example', 'Header only'),
         ('inplace_instructions', 'inplace_example', 'Task statement'))
ARMS = {'inplace_header': 'No example team, header only', 'inplace_instructions': 'No example team, task statement',
        'inplace_header_example': 'Example team, header only', 'inplace_example': 'Example team, task statement'}
ALPHA = 0.025               # two bases per model, Bonferroni (pre-registration, "Contract")
REFERENCE = Path.home() / 'vgc-data/inplace_prompt_grid/modal-l4'      # #238's L4 fills of the two prompts without it
COPY_FIELDS = ('species', 'item', 'ability', 'nature', 'moves', 'statPoints')
COPY_NAMES = {'species': 'species', 'item': 'item', 'ability': 'ability', 'nature': 'Stat Alignment', 'moves': 'move',
              'statPoints': 'Stat Point'}
LOW, HIGH = '#e8f1fd', '#0d366b'            # dataviz sequential blue, as #238's rules figure
LOW_DARK, HIGH_DARK = '#1b2a3d', '#cde2fb'
SURFACE = {'#fcfcfb': '#161615', LOW: LOW_DARK, HIGH: HIGH_DARK}
SETTING_NOTE = ('One first-try fill per mask, 128 steps, temperature 1.0, mask seeds from run seed 2026, all four '
                'prompts filled in one Modal L4 container per model. Legal means the fill parses as a team, keeps every '
                'shown value, writes only Reg M-B names and Showdown accepts it.')


def rows_for(directory: Path, key: str, prompt: str, keep: frozenset[str] | None = None) -> list[dict[str, Any]]:
    rows = G.read_fills(directory / prompt / key / 'fills.jsonl')
    return rows if keep is None else [r for r in rows if r['id'] in keep]


def pair(directory: Path, key: str, before: str, after: str) -> dict[str, Any]:
    """One base's comparison for one model, on the masks both prompts fill, with each side's copy counts."""
    keep = G.shared_ids(before, after)
    a, b = rows_for(directory, key, before, keep), rows_for(directory, key, after, keep)
    out = G.compare_rows(a, b)
    out['copies'] = {'before': G.example_copies(a), 'after': G.example_copies(b)}
    out['masks'] = len(a)
    out['peak_gib'] = {'before': G.summarize(a)['peak_gib'], 'after': G.summarize(b)['peak_gib']}
    return out


def change_words(c: dict[str, Any]) -> str:
    before, after = c['legal']['before'], c['legal']['after']
    verb = 'raises' if after > before else 'lowers' if after < before else 'leaves'
    return (f"{verb} legal fills from {before} to {after}" if verb != 'leaves'
            else f"leaves legal fills at {before}")


def headline(key: str, pairs: dict[str, dict[str, Any]]) -> str:
    h, s = pairs['inplace_header'], pairs['inplace_instructions']
    return (f"{LABELS[key]}: the example team {change_words(h)} on the header-only prompt "
            f"and {change_words(s).replace('legal fills ', '')} on the task-statement prompt")


def pairs_for(directory: Path, key: str) -> dict[str, dict[str, Any]]:
    return {before: pair(directory, key, before, after) for before, after, _ in BASES}


def verdict(p: dict[str, Any]) -> str:
    return 'significant' if p['p_value'] < ALPHA else 'within noise'


def model_figure(directory: Path, key: str, out: Path) -> Path:
    """2 x 2: base across, example team down; each cell the legal count of its own prompt on the pair's masks."""
    pairs = pairs_for(directory, key)
    data = []
    for x, (before, after, base) in enumerate(BASES):
        c = pairs[before]
        p = c['paired']
        for y, (side, prompt) in enumerate((('before', before), ('after', after))):
            n = c['legal'][side]
            label = f"{n} of {c['masks']}"
            if side == 'after':
                label += f"\n+{p['gained']} / −{p['lost']}, p = {p['p_value']:.2g}"
            data.append({'value': [x, y, n], 'label': {'formatter': label,
                                                       'color': '#fff' if n > 18 else '#0b0b0b'},
                         'tip': f"{ARMS[prompt]}<br>{label.replace(chr(10), '<br>')}"})
    peak = max(c['legal'][s] for c in pairs.values() for s in ('before', 'after'))
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 150, 'right': 16, 'top': 40, 'bottom': 16},
        'xAxis': {'type': 'category', 'position': 'top', 'data': [b for _, _, b in BASES],
                  'axisLabel': {'fontSize': 13}},
        'yAxis': {'type': 'category', 'data': ['No example team', 'Example team'], 'inverse': True,
                  'axisLabel': {'fontSize': 13}},
        'visualMap': {'show': False, 'min': 0, 'max': max(peak, 1), 'dimension': 2, 'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data, 'label': {'show': True, 'fontSize': 14, 'lineHeight': 20},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 3, 'borderRadius': 4},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    masks = ', '.join(f"{b.lower()}: {pairs[x]['masks']}" for x, _, b in BASES)
    note = (f'Legal first-try fills out of the masks both prompts of a column fill ({masks}); with the example team '
            'some masks pass the 1,024-token limit and are left out of that column. +gained / −lost counts masks legal '
            f'only with or only without the example; McNemar exact p, significant below {ALPHA}. ' + SETTING_NOTE)
    return write_chart(option, out, title=headline(key, pairs),
                       subtitle='Held-out grid, one fixed example team (MB756) before the masked team (#239)',
                       note=note, height=300, dark=SURFACE)


def copy_share(copies: dict[str, dict[str, int]], field: str) -> tuple[int, int]:
    """Filled values that repeat the example: any of its values of the field for a name, the same place for a Stat
    Point number."""
    c = copies.get(field, {'same_place': 0, 'same_field': 0, 'filled': 0})
    return (c['same_place'] if field == 'statPoints' else c['same_field']), c['filled']


def copies_figure(directory: Path, out: Path) -> Path:
    """Rows: each model with the example team, then without it (the chance rate); columns: fields. Header-only base."""
    keys = [k for k in G.SMALL_FOUR if (directory / 'inplace_header_example' / k / 'fills.jsonl').exists()]
    rows = [(k, side) for k in keys for side in ('after', 'before')]
    pairs = {k: pair(directory, k, 'inplace_header', 'inplace_header_example') for k in keys}
    data = []
    for y, (k, side) in enumerate(rows):
        for x, field in enumerate(COPY_FIELDS):
            hit, of = copy_share(pairs[k]['copies'][side], field)
            share = 100 * hit / of if of else None
            data.append({'value': [x, y, None if share is None else round(share, 1)],
                         'tip': f"{LABELS[k]}, {'with' if side == 'after' else 'without'} the example<br>"
                                f"{COPY_NAMES[field]}: {hit} of {of} filled values",
                         'label': {'color': '#fff' if share and share > 55 else '#0b0b0b'}})
    names = [f"{LABELS[k]}\n{'with the example' if side == 'after' else 'without (chance)'}" for k, side in rows]
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 210, 'right': 8, 'top': 34, 'bottom': 10},
        'xAxis': {'type': 'category', 'position': 'top', 'data': [COPY_NAMES[f] for f in COPY_FIELDS],
                  'axisLabel': {'interval': 0, 'fontSize': 12}},
        'yAxis': {'type': 'category', 'data': names, 'inverse': True,
                  'axisLabel': {'fontSize': 11, 'lineHeight': 14}},
        'visualMap': {'show': False, 'min': 0, 'max': 100, 'dimension': 2, 'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data,
                    'label': {'show': True, 'fontSize': 12,
                              'formatter': 'js:p => p.value[2] == null ? "–" : `${Math.round(p.value[2])}%`'},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2, 'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    top = max(keys, key=lambda k: copy_share(pairs[k]['copies']['after'], 'moves')[0]
              / max(1, copy_share(pairs[k]['copies']['after'], 'moves')[1]))
    note = ("Share of non-blank filled values that repeat the example team: for a name, any of the example's six "
            "values of that field (24 for moves); for a Stat Point number, the example's number in the same Pokémon and "
            "stat. The row without the example is the same count on the header-only prompt's fills of the same masks, "
            f"the rate by chance. Header-only base, {pairs[keys[0]]['masks']} masks. " + SETTING_NOTE)
    m_hit, m_of = copy_share(pairs[top]['copies']['after'], 'moves')
    title = (f"{LABELS[top]} copies the example team most: {m_hit} of {m_of} filled moves are one of its moves")
    return write_chart(option, out, title=title, subtitle='Which filled values repeat the example team (#239)',
                       note=note, height=110 + 44 * len(rows), dark=SURFACE)


def compare_figure(directory: Path, out: Path) -> Path:
    """Rows: models; columns: the four prompts; each cell the legal count of that prompt on its own masks."""
    keys = [k for k in G.SMALL_FOUR if all((directory / p / k / 'fills.jsonl').exists() for p in ARMS)]
    data = []
    for y, k in enumerate(keys):
        for x, prompt in enumerate(ARMS):
            rows = rows_for(directory, k, prompt)
            n = sum(r['legal'] for r in rows)
            data.append({'value': [x, y, n], 'label': {'formatter': f'{n} of {len(rows)}',
                                                       'color': '#fff' if n > 18 else '#0b0b0b'},
                         'tip': f"{LABELS[k]}<br>{ARMS[prompt]}: {n} of {len(rows)} masks legal"})
    pairs = {k: pairs_for(directory, k) for k in keys}
    moved = [k for k in keys if any(c['paired']['p_value'] < ALPHA for c in pairs[k].values())]
    gains = [k for k in moved if any(c['paired']['p_value'] < ALPHA and c['paired']['gained'] > c['paired']['lost']
                                     for c in pairs[k].values())]
    if not moved:
        title = 'The example team changes no model\'s legal fills beyond noise, on either base'
    elif gains:
        title = f"The example team raises legal fills for {', '.join(LABELS[k] for k in gains)}"
    else:
        title = f"The example team lowers legal fills for {', '.join(LABELS[k] for k in moved)}"
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 190, 'right': 8, 'top': 48, 'bottom': 10},
        'xAxis': {'type': 'category', 'position': 'top', 'data': [a.replace(', ', ',\n') for a in ARMS.values()],
                  'axisLabel': {'interval': 0, 'fontSize': 12, 'lineHeight': 15}},
        'yAxis': {'type': 'category', 'data': [LABELS[k] for k in keys], 'inverse': True, 'axisLabel': {'fontSize': 12}},
        'visualMap': {'show': False, 'min': 0, 'max': max([d['value'][2] for d in data] + [1]), 'dimension': 2,
                      'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data, 'label': {'show': True, 'fontSize': 13},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2, 'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Legal first-try fills of each prompt on its own masks: 270 without the example team, 257 with it on the '
            'header-only prompt and 189 with it on the task-statement prompt, where the rest pass the 1,024-token '
            'limit. The paired tests (in the table below) use the masks both prompts fill. ' + SETTING_NOTE)
    return write_chart(option, out, title=title, subtitle='Held-out grid, the four prompts per model (#239)', note=note,
                       height=120 + 56 * len(keys), dark=SURFACE)


def same_as_238(directory: Path, key: str) -> dict[str, tuple[int, int, int, int]]:
    """Per prompt without the example: masks whose fill repeats #238's L4 fill token for token, masks, and each
    run's legal count."""
    out = {}
    for prompt in ('inplace_header', 'inplace_instructions'):
        mine = {r['id']: r for r in rows_for(directory, key, prompt)}
        path = REFERENCE / prompt / key / 'fills.jsonl'
        if not path.exists():
            continue
        ref = {r['id']: r for r in G.read_rows(path)}
        shared = set(mine) & set(ref)
        out[prompt] = (sum(mine[i]['slot_ids'] == ref[i]['slot_ids'] for i in shared), len(shared),
                       sum(mine[i]['legal'] for i in shared), sum(ref[i]['legal'] for i in shared))
    return out


def copy_line(c: dict[str, Any]) -> str:
    parts = []
    for field in COPY_FIELDS:
        hit, of = copy_share(c['copies']['after'], field)
        base, base_of = copy_share(c['copies']['before'], field)
        if of:
            parts.append(f"{COPY_NAMES[field]} {hit}/{of} (without it {base}/{base_of})")
    return '; '.join(parts)


def real_line(directory: Path, key: str, before: str, after: str) -> str:
    keep = G.shared_ids(before, after)
    a, b = (G.summarize(rows_for(directory, key, p, keep))['real_values'] for p in (before, after))
    return ', '.join(f"{G.FIELD_NAMES[f]} {a[f]['real']} → {b[f]['real']} of {b[f]['of']}" for f in b)


def model_comment(directory: Path, key: str, url: str) -> str:
    pairs = pairs_for(directory, key)
    lines = [f'## {headline(key, pairs)}', '',
             picture(url, f'example-{key}', headline(key, pairs) + '. Legal first-try fills: ' + '; '.join(
                 f"{base.lower()} {pairs[b]['legal']['before']} without the example, {pairs[b]['legal']['after']} "
                 f"with it, of {pairs[b]['masks']} masks" for b, _, base in BASES) + '.'), '']
    for before, after, base in BASES:
        c = pairs[before]
        p = c['paired']
        tasks = ', '.join(f"{TASK_NAMES[t]} {v['legal']}" for t, v in c['legal_by_task']['after'].items() if v['legal'])
        lines += [f"**{base} base** ({c['masks']} masks, `{before}` → `{after}`)",
                  f"- **Legal:** {c['legal']['before']} → {c['legal']['after']}; {p['gained']} gained, {p['lost']} lost "
                  f"(McNemar exact p = {p['p_value']:.3g}, {verdict(p)} at {ALPHA}). Legal with the example by what the "
                  f"mask hides: {tasks or 'none'}.",
                  f"- **Copies of the example** (filled values that repeat it): {copy_line(c)}.",
                  f"- **Real Reg M-B values filled:** {real_line(directory, key, before, after)}. Parses as JSON "
                  f"{c['json_parses']['before']} → {c['json_parses']['after']}.",
                  '- **Most common reasons with the example:** '
                  + '; '.join(f'{r} ({n})' for r, n in c['reasons']['after'][:4]) + '.',
                  f"- **Cost:** prompt {c['prompt_tokens_mean']['before']:.0f} → {c['prompt_tokens_mean']['after']:.0f} "
                  f"tokens on average, {c['seconds_per_team']['before']:.2f} → {c['seconds_per_team']['after']:.2f} s "
                  f"per team, peak {c['peak_gib']['after']:.2f} GiB (ceiling 17.76).", '']
    probe = json.loads((directory / 'preflight' / f'modal-gpu-{key}.json').read_text())
    same = same_as_238(directory, key)
    repeat = '; '.join(f"`{p}` {a}/{n} token for token, legal {x} here and {y} in #238" for p, (a, n, x, y) in same.items())
    meta = json.loads((directory / 'inplace_example' / key / 'run.json').read_text())
    where = str(directory).replace(str(Path.home()), '~')
    lines += [f"**Probes before the fill** (on a `{probe['prompt']}` mask): attention sees later tokens "
              f"{probe['sees_later_tokens']}; same seed twice {probe['repeat_same_tokens']:.0%} of slot tokens the same; "
              f"on {probe['device']}.",
              f"**Against #238's L4 fills of the prompts without the example** (another container): {repeat}.", '',
              f"<details><summary>Setting</summary>\n\nExample team MB756 (`code/data/inplace-example-team.json`), "
              f"{meta['candidate']['repo']} at `{meta['candidate']['revision'][:8]}`, {meta['setting']}, torch "
              f"{meta['environment']['torch']}, transformers {meta['environment']['transformers']}. One fill per mask, "
              f"batch size 1. Grid file SHA-256 `{meta['tasks_sha256'][:12]}`. Fills: `{where}/<prompt>/{key}/fills.jsonl`. "
              "Pre-registration: `code/docs/inplace-prompt-example.md`.\n\n</details>"]
    return '\n'.join(lines) + '\n'


def compare_comment(directory: Path, url: str) -> str:
    keys = [k for k in G.SMALL_FOUR if all((directory / p / k / 'fills.jsonl').exists() for p in ARMS)]
    pairs = {k: pairs_for(directory, k) for k in keys}
    alt = '; '.join(f"{LABELS[k]}: " + ', '.join(f"{ARMS[p].lower()} {sum(r['legal'] for r in rows_for(directory, k, p))}"
                                                  for p in ARMS) for k in keys)
    lines = ['## The four prompts on every model', '', picture(url, 'example-compare', alt), '', '',
             picture(url, 'example-copies', 'Share of filled values that repeat the example team, per model and field, '
                     'with the example and without it.'), '',
             f'Each row pairs fills made in the same Modal container, on the masks both prompts fill. Significant '
             f'means McNemar exact p below {ALPHA} (two bases per model). Copies count filled moves that are one of the '
             "example's 24 moves, with the same count on the prompt without it in brackets.", '',
             '| Model | Base | Masks | Legal, without → with | Gained / lost (p) | Parses as JSON | Moves copied | '
             'Seconds per team |', '|---|---|---|---|---|---|---|---|']
    for k in keys:
        for before, _, base in BASES:
            c = pairs[k][before]
            p = c['paired']
            hit, of = copy_share(c['copies']['after'], 'moves')
            was, was_of = copy_share(c['copies']['before'], 'moves')
            lines.append(f"| {LABELS[k]} | {base} | {c['masks']} | {c['legal']['before']} → {c['legal']['after']} | "
                         f"{p['gained']} / {p['lost']} ({p['p_value']:.3g}) | {c['json_parses']['before']} → "
                         f"{c['json_parses']['after']} | {hit}/{of} ({was}/{was_of}) | "
                         f"{c['seconds_per_team']['before']:.2f} → {c['seconds_per_team']['after']:.2f} |")
    common = G.shared_ids(*ARMS)
    lines += ['', f"**On the {len(common)} masks all four prompts fill** (exploratory, the example × statement "
              'interaction): ' + '; '.join(f"{LABELS[k]} " + ' / '.join(
                  str(sum(r['legal'] for r in rows_for(directory, k, p, common))) for p in ARMS) for k in keys)
              + ' legal, in the order ' + ', '.join(ARMS.values()).lower() + '.',
              '', 'Pre-registration and prediction: `code/docs/inplace-prompt-example.md`. Fills: '
              f"`{str(directory).replace(str(Path.home()), '~')}/<prompt>/<model>/fills.jsonl`."]
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('figure', 'comment'):
        p = sub.add_parser(name)
        p.add_argument('--dir', type=Path, required=True)
        p.add_argument('--key', choices=G.SMALL_FOUR)
        p.add_argument('--copies', action='store_true')
        p.add_argument('--compare', action='store_true')
        p.add_argument('--out', type=Path)
        p.add_argument('--url')
    args = parser.parse_args()
    if args.command == 'figure':
        if args.copies:
            copies_figure(args.dir, args.out)
        elif args.compare:
            compare_figure(args.dir, args.out)
        else:
            model_figure(args.dir, args.key, args.out)
        return 0
    print(compare_comment(args.dir, args.url) if args.compare else model_comment(args.dir, args.key, args.url), end='')
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
