"""Figures and issue comments for the held-out grid runs (#237), from inplace_prompt_grid.py's fills.

Two figures, storyboarded in code/docs/inplace-prompt-heldout-baseline.md:
    grid     one model: every one of the 270 masks as a cell (5 unmasked teams x 54 cells), colored by how far its
             first-try fill got through the four checks
    matrix   every model that has run: masks per outcome, one row per model

    figure --dir D --key K --out F            the grid page for one model
    figure --dir D --matrix --out F           the matrix page
    comment --dir D --key K --url U           one model's comment; U holds {name} and {theme}
    comment --dir D --matrix --url U          the comment comparing every model
    figure|comment --dir D --key K --redraw   #171: one model's grid after redraws
    figure|comment --dir D --battles          #171: usable win rate of every arm, from battles-report.json
    figure|comment ... --prompt P             a later prompt's grid or model comment (#238 to #240); its model
                                              comment adds the paired change from today's prompt
    figure|comment --dir D --compare --prompt P   #238: every model's legal and parsing count, today's prompt
                                              against P, and the ticket's before-and-after table

Run with any Python that has the repo's code/scripts and code/src on its path.
"""
import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import inplace_prompt_grid as G
from echarts_chart import write_chart

LABELS = {'diffugpt_s': 'DiffuGPT small', 'diffugpt_m': 'DiffuGPT medium', 'qwen25c': 'Masked Qwen2.5 Coder 0.5B',
          'qwen3': 'Masked Qwen3 0.6B'}
# Outcomes in the order a fill gets through the checks; the value is how far it got.
OUTCOMES = ('parses', 'visible', 'regmb', 'showdown', None)
OUTCOME_LABELS = {'parses': 'Not a team: the JSON does not parse', 'visible': 'Overwrote a value the mask showed',
                  'regmb': 'Wrote a name that is not Reg M-B', 'showdown': 'Showdown rejects the team',
                  None: 'Legal on the first try'}
LIGHT = ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b')     # dataviz ordinal blue, validated 2026-11-18
DARK = ('#184f95', '#2a78d6', '#5598e7', '#86b6ef', '#cde2fb')
CELL_ORDER = [(kind, k) for kind in ('items', 'abilities', 'natures', 'stats', 'pokemon') for k in range(1, 7)]
CELL_ORDER += [('moves', k) for k in range(1, 25)]
TASK_NAMES = {'items': 'item', 'abilities': 'ability', 'natures': 'Stat Alignment', 'stats': 'Stat Points',
              'pokemon': 'whole Pokémon', 'moves': 'move'}
BAR = 3
BEFORE = 'inplace_header'
TICKETS = {'inplace_header': '#237', 'inplace_instructions': '#238'}
PROMPT_TEXT = {'inplace_header': "#168's text", 'inplace_instructions': "#168's text plus #238's task statement"}


def rows_for(directory: Path, key: str, prompt: str = 'inplace_header') -> list[dict[str, Any]]:
    return G.read_rows(directory / prompt / key / 'fills.jsonl')


def cell_label(kind: str, k: int) -> str:
    return f'{TASK_NAMES[kind]} ×{k}'


def headline(key: str, s: dict[str, Any]) -> str:
    verdict = 'passes' if s['legal'] >= BAR else 'misses'
    return f"{LABELS[key]}: {s['legal']} of {s['masks']} masks legal on the first try, so it {verdict} the bar of {BAR}"


def grid_figure(rows: list[dict[str, Any]], key: str, out: Path, prompt: str = BEFORE) -> Path:
    sources = sorted({r['source'] for r in rows})
    data = []
    for r in rows:
        x = CELL_ORDER.index((r['task'], r['k']))
        y = sources.index(r['source'])
        reason = r['reasons'][0] if r['reasons'] else ''
        fill = ' | '.join(v[:24] for v in r['values'][:4])
        tip = (f"{r['source']}, {cell_label(r['task'], r['k'])}<br><b>{OUTCOME_LABELS[r['failed']]}</b>"
               + (f'<br>{html.escape(reason)}' if reason else '') + f'<br>Filled: <code>{html.escape(fill)}</code>')
        data.append({'value': [x, y, OUTCOMES.index(r['failed'])], 'tip': tip})
    s = G.summarize(rows)
    pieces = [{'value': i, 'label': OUTCOME_LABELS[o], 'color': LIGHT[i]} for i, o in enumerate(OUTCOMES)]
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 70, 'right': 250, 'top': 30, 'bottom': 120},
        'xAxis': {'type': 'category', 'data': [cell_label(*c) for c in CELL_ORDER], 'splitArea': {'show': False},
                  'axisLabel': {'rotate': 60, 'fontSize': 10, 'interval': 0}, 'name': 'What the mask hides, and how many',
                  'nameLocation': 'middle', 'nameGap': 105},
        'yAxis': {'type': 'category', 'data': sources, 'name': 'Unmasked team', 'nameLocation': 'end'},
        'visualMap': {'type': 'piecewise', 'dimension': 2, 'pieces': pieces[::-1], 'orient': 'vertical', 'top': 40,
                      'right': 0, 'itemGap': 10, 'textStyle': {'fontSize': 12, 'width': 200, 'overflow': 'break'}},
        'series': [{'type': 'heatmap', 'data': data, 'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2,
                                                                    'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Each cell is one mask: an unmasked team (rank 51 to 100) with some values hidden, filled once by the '
            f'model with the prompt "{G.PROMPTS[prompt].name}". The four checks run in order: the filled JSON parses as a '
            'team; every value the mask showed reads back unchanged; every filled name is a Reg M-B species, item, '
            'ability, Stat Alignment or move; Showdown accepts the team. A cell shows the first check it failed. '
            f"128 steps, temperature 1.0, mask seeds from run seed 2026. Showdown alone accepts {s['showdown_accepts']} of 270.")
    return write_chart(option, out, title=headline(key, s),
                       subtitle=f'Held-out grid, 270 masks, first try only ({TICKETS[prompt]})',
                       note=note, height=460, dark=dict(zip(LIGHT, DARK, strict=True)) | {'#fcfcfb': '#161615'})


def matrix_figure(results: dict[str, list[dict[str, Any]]], out: Path) -> Path:
    keys = list(results)
    counts = {k: Counter(r['failed'] for r in rows) for k, rows in results.items()}
    data = [{'value': [j, i, counts[k][o]], 'label': {'color': '#fff' if counts[k][o] > 120 else '#0b0b0b'}}
            for i, k in enumerate(keys) for j, o in enumerate(OUTCOMES)]
    passed = [LABELS[k] for k in keys if counts[k][None] >= BAR]
    title = ('No small model passes the bar of 3 legal masks with today\'s prompt' if not passed
             else f"{', '.join(passed)} pass{'es' if len(passed) == 1 else ''} the bar of 3 legal masks with today's prompt")
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => `${p.value[2]} of 270 masks`'},
        'grid': {'left': 190, 'right': 16, 'top': 20, 'bottom': 90},
        'xAxis': {'type': 'category', 'data': [OUTCOME_LABELS[o] for o in OUTCOMES], 'position': 'bottom',
                  'axisLabel': {'interval': 0, 'width': 120, 'overflow': 'break'}},
        'yAxis': {'type': 'category', 'data': [LABELS[k] for k in keys], 'inverse': True},
        'visualMap': {'show': False, 'min': 0, 'max': 270, 'dimension': 2, 'inRange': {'color': ['#cde2fb', '#0d366b']}},
        'series': [{'type': 'heatmap', 'data': data, 'label': {'show': True, 'fontSize': 14,
                                                               'formatter': 'js:p => p.value[2]'},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2, 'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Masks out of 270, by the first check each first-try fill failed, in the order the checks run. '
            'Header-only in-place prompt, held-out grid, 128 steps, temperature 1.0, mask seeds from run seed 2026.')
    return write_chart(option, out, title=title, subtitle='Where the first-try fills stop (#237)', note=note,
                       height=120 + 70 * len(keys), dark={'#fcfcfb': '#161615', '#cde2fb': '#184f95', '#0d366b': '#cde2fb'})


def picture(url: str, name: str, alt: str) -> str:
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{url.format(name=name, theme="dark")}">'
            f'<img src="{url.format(name=name, theme="light")}" width="900" alt="{alt}"></picture>')


def model_comment(directory: Path, key: str, url: str, prompt: str = BEFORE) -> str:
    rows = rows_for(directory, key, prompt)
    s = G.summarize(rows)
    lo, hi = s['legal_interval']
    meta = json.loads((directory / prompt / key / 'run.json').read_text())
    probe = json.loads(G.gpu_preflight_path(directory, key, prompt).read_text())
    steps = Counter(r['failed'] for r in rows)
    top = Counter(reason for r in rows for reason in r['reasons']).most_common(6)
    real = s['real_values']
    by_task = ', '.join(f"{TASK_NAMES[t]} {v['legal']}/{v['of']}" for t, v in s['legal_by_task'].items())
    alt = f"{headline(key, s)}. Of 270 masks: " + '; '.join(f'{OUTCOME_LABELS[o]}, {steps[o]}' for o in OUTCOMES) + '.'
    field_line = ', '.join(f"{G.FIELD_NAMES[f]} {v['real']}/{v['of']}" for f, v in real.items())
    change = []
    if prompt != BEFORE:
        c = G.compare_rows(rows_for(directory, key, BEFORE), rows)
        p = c['paired']
        change = [f"- **Against today's prompt** ({TICKETS[BEFORE]}, the same masks and mask seeds): legal "
                  f"{c['legal']['before']} → {c['legal']['after']}; {p['gained']} masks turned legal and {p['lost']} "
                  f"stopped being legal (McNemar exact p = {p['p_value']:.3g}). Parses as JSON "
                  f"{c['json_parses']['before']} → {c['json_parses']['after']}; seconds per team "
                  f"{c['seconds_per_team']['before']:.1f} → {c['seconds_per_team']['after']:.1f}."]
    lines = [f'## {headline(key, s)}', '', picture(url, f'grid-{prompt}-{key}' if prompt != BEFORE else f'grid-{key}', alt),
             '', *change,
             f"- **Legal on the first try:** {s['legal']}/270, exact 95% interval {lo:.2%} to {hi:.2%}. Legal means all "
             "four checks pass: parses as a team, keeps every shown value, writes only Reg M-B names, Showdown accepts.",
             f"- **Showdown alone accepts** {s['showdown_accepts']}/270. It lets junk Stat Alignments through, which is "
             'why the Reg M-B check runs first (pre-registration, "Departures").',
             f"- **Parses as JSON:** {s['json_parses']}/270; as a six-Pokémon team: {s['reads_as_team']}/270.",
             f"- **Filled values that are real Reg M-B values:** {field_line}. Distinct real species: "
             f"{s['distinct_real_species']}.",
             f"- **Legal by what the mask hides:** {by_task}.",
             '- **Most common reasons:** ' + '; '.join(f'{reason} ({n})' for reason, n in top) + '.',
             f"- **Copies:** {s['recovered_unmasked_team']} masks refilled exactly as the unmasked team; "
             f"{s['names_copied_from_visible']} filled names repeat a name shown in the same text.",
             f"- **Cost:** prompt {s['prompt_tokens_mean']:.0f} tokens on average (most {s['prompt_tokens_max']}), "
             f"{s['seconds_per_team']:.1f} s per team, {s['seconds_total'] / 3600:.2f} h in all, peak "
             f"{s['peak_gib']:.2f} GiB.",
             f"- **Probes before the fill:** attention sees later tokens {probe['sees_later_tokens']}; same seed twice "
             f"{probe['repeat_same_tokens']:.0%} of slot tokens the same; "
             + (f"reverse run order {probe['reverse_order_same_tokens']:.0%} the same." if 'reverse_order_same_tokens'
                in probe else f"on {probe.get('device', 'the GPU')}; run order cannot matter, since each mask's seed "
                "comes from its id (CPU test)."),
             '',
             f"<details><summary>Setting</summary>\n\nPrompt `{prompt}` ({PROMPT_TEXT[prompt]}, ask `unknown`), "
             f"{meta['candidate']['repo']} at `{meta['candidate']['revision'][:8]}`, {meta['setting']}, torch "
             f"{meta['environment']['torch']}, transformers {meta['environment']['transformers']}. One fill per mask, "
             f"batch size 1. Grid file SHA-256 `{meta['tasks_sha256'][:12]}`. Fills: "
             f"`~/vgc-data/inplace_prompt_grid/{prompt}/{key}/fills.jsonl`. Pre-registration: "
             f"`code/docs/{PREREGISTRATION[prompt]}`.\n\n</details>"]
    return '\n'.join(lines) + '\n'


PREREGISTRATION = {'inplace_header': 'inplace-prompt-heldout-baseline.md',
                   'inplace_instructions': 'inplace-prompt-instructions.md'}
AFTER_LIGHT, AFTER_DARK = '#2a78d6', '#86b6ef'     # the prompt measured; today's prompt is the grey
BEFORE_LIGHT, BEFORE_DARK = '#b5b5b0', '#5c5c58'


def compare_results(directory: Path, prompt: str) -> dict[str, dict[str, Any]]:
    return {k: G.compare_rows(rows_for(directory, k, BEFORE), rows_for(directory, k, prompt)) for k in G.SMALL_FOUR
            if all((directory / p / k / 'fills.jsonl').exists() for p in (BEFORE, prompt))}


def compare_headline(results: dict[str, dict[str, Any]], prompt: str) -> str:
    up = [k for k, c in results.items() if c['legal']['after'] > c['legal']['before']]
    down = [k for k, c in results.items() if c['legal']['after'] < c['legal']['before']]
    name = G.PROMPTS[prompt].name.split(',')[0]
    if not up and not down:
        return f'{name}: no model changes its legal count'
    parts = []
    if up:
        parts.append(f"more legal masks for {len(up)} of {len(results)} models")
    if down:
        parts.append(f"fewer for {', '.join(LABELS[k] for k in down)}")
    return f"{name}: {' and '.join(parts)}"


def compare_figure(results: dict[str, dict[str, Any]], prompt: str, out: Path) -> Path:
    """Two panels, one row per model: masks legal on the first try, and masks whose fill parses as JSON, today's
    prompt (grey) against the prompt measured (blue)."""
    keys = list(results)
    names = [LABELS[k] for k in keys]
    series = []
    for panel, measure in enumerate(('legal', 'json_parses')):
        for side, color, label in (('before', BEFORE_LIGHT, G.PROMPTS[BEFORE].name.split(',')[0]),
                                   ('after', AFTER_LIGHT, G.PROMPTS[prompt].name.split(',')[0])):
            series.append({'type': 'bar', 'name': label, 'xAxisIndex': panel, 'yAxisIndex': panel, 'color': color,
                           'data': [results[k][measure][side] for k in keys], 'barGap': '10%',
                           'label': {'show': True, 'position': 'right', 'fontSize': 12},
                           'emphasis': {'focus': 'series'}})
    option = {
        'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
        'legend': {'top': 0, 'data': [series[0]['name'], series[1]['name']]},
        'grid': [{'left': 190, 'right': '54%', 'top': 60, 'bottom': 40},
                 {'left': '56%', 'right': 40, 'top': 60, 'bottom': 40}],
        'title': [{'text': 'Legal on the first try', 'left': 190, 'top': 30, 'textStyle': {'fontSize': 13}},
                  {'text': 'Parses as JSON', 'left': '56%', 'top': 30, 'textStyle': {'fontSize': 13}}],
        'xAxis': [{'type': 'value', 'gridIndex': 0, 'max': 270, 'name': 'masks of 270', 'nameLocation': 'middle',
                   'nameGap': 26},
                  {'type': 'value', 'gridIndex': 1, 'max': 270, 'name': 'masks of 270', 'nameLocation': 'middle',
                   'nameGap': 26}],
        'yAxis': [{'type': 'category', 'gridIndex': 0, 'data': names, 'inverse': True},
                  {'type': 'category', 'gridIndex': 1, 'data': names, 'inverse': True, 'axisLabel': {'show': False}}],
        'series': series,
    }
    note = ('The same 270 held-out masks and mask seeds under both prompts, one first-try fill each, 128 steps, '
            'temperature 1.0, run seed 2026. Legal means the fill parses as a team, keeps every shown value, writes '
            'only Reg M-B names and Showdown accepts it.')
    return write_chart(option, out, title=compare_headline(results, prompt),
                       subtitle=f'{G.PROMPTS[BEFORE].name} against {G.PROMPTS[prompt].name} ({TICKETS[prompt]})',
                       note=note, height=150 + 60 * len(keys),
                       dark={AFTER_LIGHT: AFTER_DARK, BEFORE_LIGHT: BEFORE_DARK})


def same_tokens(rows: list[dict[str, Any]], other: list[dict[str, Any]]) -> int:
    """Masks whose fill is the same token for token in both runs."""
    by_id = {r['id']: r['slot_ids'] for r in other}
    return sum(r['slot_ids'] == by_id.get(r['id']) for r in rows)


def compare_comment(directory: Path, prompt: str, url: str) -> str:
    results = compare_results(directory, prompt)
    alt = '; '.join(f"{LABELS[k]}: legal {c['legal']['before']} then {c['legal']['after']} of 270, parses as JSON "
                    f"{c['json_parses']['before']} then {c['json_parses']['after']}" for k, c in results.items())
    lines = [f'## {compare_headline(results, prompt)}', '', picture(url, f'compare-{prompt}', alt), '',
             f"Each row pairs {TICKETS[BEFORE]}'s fills of today's prompt with this prompt's fills of the same 270 masks "
             'and mask seeds. Gained counts masks legal only with the new prompt, lost those legal only with today\'s; '
             "McNemar's exact test runs on those two counts.", '',
             '| Model | Legal, first try | Gained / lost (p) | Parses as JSON | Real species | Seconds per team | '
             'Most common reasons, before → after |', '|---|---|---|---|---|---|---|']
    for k, c in results.items():
        p = c['paired']
        top = {side: '; '.join(f'{r} ({n})' for r, n in c['reasons'][side][:2]) for side in ('before', 'after')}
        lines.append(f"| {LABELS[k]} | {c['legal']['before']} → {c['legal']['after']} | {p['gained']} / {p['lost']} "
                     f"({p['p_value']:.3g}) | {c['json_parses']['before']} → {c['json_parses']['after']} | "
                     f"{c['real_species']['before']} → {c['real_species']['after']} of {c['real_species']['of']} | "
                     f"{c['seconds_per_team']['before']:.1f} → {c['seconds_per_team']['after']:.1f} | "
                     f"{top['before']} → {top['after']} |")
    laptop = {k: G.read_rows(G.OUT / BEFORE / k / 'fills.jsonl') for k in results
              if directory.resolve() != G.OUT.resolve() and (G.OUT / BEFORE / k / 'fills.jsonl').exists()}
    if laptop:
        lines += ['', f"**The device, with the prompt held fixed.** Header-only fills here against {TICKETS[BEFORE]}'s "
                  'laptop fills of the same masks and seeds: ' + '; '.join(
                      f"{LABELS[k]} {same_tokens(rows_for(directory, k, BEFORE), laptop[k])}/270 token for token, "
                      f"legal {sum(r['legal'] for r in rows_for(directory, k, BEFORE))} here and "
                      f"{sum(r['legal'] for r in laptop[k])} on the laptop" for k in laptop) + '.']
    lines += ['', f"Pre-registration and prediction: `code/docs/{PREREGISTRATION[prompt]}`. Fills: "
              f"`{str(directory).replace(str(Path.home()), '~')}/<prompt>/<model>/fills.jsonl`."]
    return '\n'.join(lines) + '\n'


def matrix_comment(directory: Path, url: str) -> str:
    results = {k: rows_for(directory, k) for k in G.SMALL_FOUR if (directory / 'inplace_header' / k / 'fills.jsonl').exists()}
    summaries = {k: G.summarize(rows) for k, rows in results.items()}
    passed = [k for k, s in summaries.items() if s['legal'] >= BAR]
    verdict = ('**The claim is refuted:** no small model reaches 3 legal masks of 270 with today\'s prompt.' if not passed
               else '**The claim holds** for ' + ', '.join(LABELS[k] for k in passed) + '.')
    alt = '; '.join(f"{LABELS[k]} {s['legal']} legal of 270" for k, s in summaries.items())
    lines = ['## The starting point: today\'s in-place prompt on the held-out grid, all four small models', '',
             picture(url, 'matrix', alt), '', verdict,
             'Prediction written before any fill: all four miss, 0 to 2 legal each.', '',
             '| Model | Legal, first try | 95% interval | Showdown alone accepts | Parses as a team | Real species |',
             '|---|---|---|---|---|---|']
    for k, s in summaries.items():
        lo, hi = s['legal_interval']
        sp = s['real_values'].get('species', {'real': 0, 'of': 0})
        lines.append(f"| {LABELS[k]} | {s['legal']}/270 | {lo:.2%} to {hi:.2%} | {s['showdown_accepts']}/270 | "
                     f"{s['reads_as_team']}/270 | {sp['real']}/{sp['of']} |")
    lines += ['', 'Each later prompt (#238 to #240) is paired with this row on the same masks and mask seeds.']
    return '\n'.join(lines) + '\n'


# ---------- #171: redraws and battles ----------

DRAW_LABELS = ('Legal on the first try', 'Legal on redraw 1', 'Legal on redraw 2', 'Legal on redraw 3',
               'Legal on redraw 4', 'No legal team in 5 draws')
NONE_LIGHT, NONE_DARK = '#e4e4e1', '#2c2c2a'
REFERENCE_ROWS = (('jev', 'Jev, top pick'), ('random', 'Random legal fill'), ('mercury-off', 'Mercury, no reasoning'))


def final_for(directory: Path, key: str) -> dict[str, dict[str, Any]]:
    out = directory / 'inplace_header' / key
    return G.final_draws(out / 'fills.jsonl', out / 'redraws.jsonl')


def draw_class(row: dict[str, Any]) -> int:
    return row['draw'] if row['legal'] else 5


def redraw_headline(key: str, final: dict[str, dict[str, Any]]) -> str:
    first = sum(r['legal'] and r['draw'] == 0 for r in final.values())
    legal = sum(r['legal'] for r in final.values())
    return f'{LABELS[key]}: {legal} of {len(final)} masks legal within 5 draws, up from {first} on the first try'


def redraw_figure(final: dict[str, dict[str, Any]], key: str, out: Path) -> Path:
    rows = list(final.values())
    sources = sorted({r['source'] for r in rows})
    ramp = (LIGHT[4], LIGHT[3], LIGHT[2], LIGHT[1], LIGHT[0], NONE_LIGHT)
    data = []
    for r in rows:
        c = draw_class(r)
        reason = '' if r['legal'] else (r['reasons'][0] if r['reasons'] else '')
        tip = (f"{r['source']}, {cell_label(r['task'], r['k'])}<br><b>{DRAW_LABELS[c]}</b>"
               + (f'<br>Last draw: {html.escape(reason)}' if reason else ''))
        data.append({'value': [CELL_ORDER.index((r['task'], r['k'])), sources.index(r['source']), c], 'tip': tip})
    pieces = [{'value': i, 'label': label, 'color': ramp[i]} for i, label in enumerate(DRAW_LABELS)]
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 70, 'right': 250, 'top': 30, 'bottom': 120},
        'xAxis': {'type': 'category', 'data': [cell_label(*c) for c in CELL_ORDER], 'splitArea': {'show': False},
                  'axisLabel': {'rotate': 60, 'fontSize': 10, 'interval': 0}, 'name': 'What the mask hides, and how many',
                  'nameLocation': 'middle', 'nameGap': 105},
        'yAxis': {'type': 'category', 'data': sources, 'name': 'Unmasked team', 'nameLocation': 'end'},
        'visualMap': {'type': 'piecewise', 'dimension': 2, 'pieces': pieces, 'orient': 'vertical', 'top': 40,
                      'right': 0, 'itemGap': 10, 'textStyle': {'fontSize': 12, 'width': 200, 'overflow': 'break'}},
        'series': [{'type': 'heatmap', 'data': data, 'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2,
                                                                    'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Each cell is one mask. A mask whose first try failed the four checks was drawn again, up to 4 more '
            'times, each draw with its own seed, stopping at the first legal draw (#169). The legend gives the draw at '
            'which each mask turned legal. '
            'Header-only in-place prompt, 128 steps, temperature 1.0.')
    dark = dict(zip(LIGHT, DARK, strict=True)) | {'#fcfcfb': '#161615', NONE_LIGHT: NONE_DARK}
    return write_chart(option, out, title=redraw_headline(key, final), subtitle='Held-out grid, 270 masks, '
                       'after redraws (#171)', note=note, height=460, dark=dark)


def redraw_comment(directory: Path, key: str, url: str) -> str:
    final = final_for(directory, key)
    out = directory / 'inplace_header' / key
    redraws = G.read_rows(out / 'redraws.jsonl') if (out / 'redraws.jsonl').exists() else []
    probe = json.loads((directory / 'preflight' / f'gpu-{key}.json').read_text())
    by_draw = Counter(draw_class(r) for r in final.values())
    kinds = Counter(r['task'] for r in final.values() if r['legal'])
    legal_redraws = [r for r in redraws if r['legal']]
    alt = redraw_headline(key, final) + '. ' + '; '.join(f'{DRAW_LABELS[i]}: {by_draw[i]}' for i in range(6)) + '.'
    lines = [f'## {redraw_headline(key, final)}', '', picture(url, f'redraw-{key}', alt), '',
             '- **Legal by draw:** ' + ', '.join(f'{DRAW_LABELS[i].lower()} {by_draw[i]}' for i in range(6)) + '.',
             '- **Legal by what the mask hides:** ' + ', '.join(f'{TASK_NAMES[t]} {kinds[t]}/{30 if t != "moves" else 120}'
                                                               for t in TASK_NAMES) + '.',
             f"- **Draws made:** {len(final) + len(redraws)} in all ({len(redraws)} redraws), "
             f"{sum(r['seconds'] for r in redraws) / 3600:.2f} h of redraws.",
             f"- **Copies:** {sum(r['recovered'] for r in legal_redraws)} legal redraws give back the unmasked team.",
             f"- **Redraws differ:** a redraw's seed kept {probe['redraw_same_tokens']:.0%} of slot tokens on three "
             'probe masks, so a redraw is a new draw.',
             '', 'Battles follow once all four models are redrawn. Pre-registration: '
             '`code/docs/inplace-prompt-heldout-battles.md`.']
    return '\n'.join(lines) + '\n'


def battle_report(directory: Path) -> dict[str, Any]:
    return json.loads((directory / 'battles-report.json').read_text())


def reference_usable() -> dict[str, dict[str, Any]]:
    ref = json.loads((Path.home() / 'vgc-data/jev/rebattle-heldout.json').read_text())
    return ref['sets']['top50']['kinds']['all']['usable']


def battle_headline(rep: dict[str, Any]) -> str:
    models = rep['sets']['top50']['models']
    best = max(models, key=lambda k: models[k]['kinds']['all']['usable']['mean'])
    mercury = reference_usable()['mercury-off']['mean']
    v = models[best]['kinds']['all']['usable']['mean']
    side = 'below' if v < mercury else 'at or above'
    return f'The best small model, {LABELS[best]}, keeps {v:.1%} usable win rate, {side} Mercury with no reasoning ({mercury:.1%})'


def battle_figure(rep: dict[str, Any], out: Path) -> Path:
    ref = reference_usable()
    models = rep['sets']['top50']['models']
    rows = [(label, ref[a], False) for a, label in REFERENCE_ROWS]
    rows += [(LABELS[k], m['kinds']['all']['usable'], True) for k, m in models.items()]
    rows.sort(key=lambda r: r[1]['mean'], reverse=True)
    data = [{'value': round(u['mean'] * 100, 1), 'itemStyle': {'color': LIGHT[3] if ours else NONE_LIGHT},
             'tip': f"{label}<br>{u['mean']:.1%}, 95% interval {u['ci_teams'][0]:.1%} to {u['ci_teams'][1]:.1%}"}
            for label, u, ours in rows]
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 200, 'right': 60, 'top': 10, 'bottom': 50},
        'xAxis': {'type': 'value', 'min': 0, 'max': 50, 'name': 'Usable win rate against the top-50 teams, %',
                  'nameLocation': 'middle', 'nameGap': 30},
        'yAxis': {'type': 'category', 'data': [r[0] for r in rows], 'inverse': True},
        'series': [{'type': 'bar', 'data': data, 'barWidth': '60%', 'label': {'show': True, 'position': 'right',
                                                                               'formatter': 'js:p => p.value.toFixed(1) + "%"'}}],
    }
    note = ('Usable win rate: over all 270 masks of the held-out grid, a mask with no legal team scores 0. Blue: the '
            'small masked models, header-only in-place prompt, up to 5 draws per mask. Grey: reference rows from the '
            '2026-10-29 re-battle. Behaviour-cloning policy on both sides, slots shuffled, 8 battles per opponent.')
    return write_chart(option, out, title=battle_headline(rep), subtitle='Held-out grid battles (#171)', note=note,
                       height=60 + 46 * len(rows), dark={LIGHT[3]: DARK[3], NONE_LIGHT: '#5a5a57'})


def battle_comment(directory: Path, url: str) -> str:
    rep = battle_report(directory)
    top, held = rep['sets']['top50'], rep['sets']['heldout']
    c = top['controls']
    mercury = reference_usable()['mercury-off']['mean']
    holds = all(m['kinds']['all']['usable']['mean'] < mercury for m in top['models'].values())
    verdict = ('**The claim holds:** every small model keeps less usable win rate than Mercury with no reasoning '
               f'({mercury:.1%}).' if holds else '**The claim is refuted:** a small model reaches Mercury with no '
               f'reasoning ({mercury:.1%}).')

    def diff(k: dict[str, Any], other: str) -> str:
        d = k.get(f'minus_{other}')
        return '–' if d is None else f"{d['diff'] * 100:+.1f} [{d['ci_masks'][0] * 100:+.1f}, {d['ci_masks'][1] * 100:+.1f}]"
    lines = [f'## {battle_headline(rep)}', '', picture(url, 'battles', battle_headline(rep)), '', verdict,
             'Prediction written before any redraw: every model below 6%, far below Mercury with no reasoning.', '',
             '| Model | Legal within 5 draws | Usable, top-50 | 95% interval | When legal | Usable, held-out | '
             'Minus unmasked team, points | Minus random legal fill | Minus Jev | All 66 Stat Points spent |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for key, m in top['models'].items():
        a, h = m['kinds']['all'], held['models'][key]['kinds']['all']
        sp = m['stat_points_all_66']
        lines.append(f"| {LABELS[key]} | {m['legal']}/270 | {a['usable']['mean']:.1%} | "
                     f"{a['usable']['ci_teams'][0]:.1%} to {a['usable']['ci_teams'][1]:.1%} | "
                     f"{a.get('when_legal', float('nan')):.1%} | {h['usable']['mean']:.1%} | {diff(a, 'unmasked')} | "
                     f"{diff(a, 'random')} | {diff(a, 'jev')} | {sp[0]}/{sp[1]} |")
    lines += ['', 'Minus columns: on the masks the model filled legally, its win rate minus the other arm\'s on the '
              'same masks, in points, with a 95% interval from resampling masks.', '',
              f"**Controls.** Unmasked teams here {sum(c['unmasked'].values()) / 5:.1%} against "
              f"{sum(c['unmasked_jev_rebattle'].values()) / 5:.1%} in the Jev re-battle; uniform-legal teams "
              f"{sum(c['uniform_legal']) / len(c['uniform_legal']):.1%}.", '',
              '<details><summary>Setting</summary>\n\n' + rep['policy'] + f", {rep['battles_per_opponent']} battles "
              f"per opponent ({rep['unmasked_battles_per_opponent']} for the unmasked teams), {top['opponents']} top-50 "
              f"and {held['opponents']} held-out opponents. Report: `~/vgc-data/inplace_prompt_grid/battles-report.json`. "
              'Pre-registration: `code/docs/inplace-prompt-heldout-battles.md`.\n\n</details>']
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('figure', 'comment'):
        p = sub.add_parser(name)
        p.add_argument('--dir', type=Path, default=G.OUT)
        p.add_argument('--key', choices=G.SMALL_FOUR)
        p.add_argument('--matrix', action='store_true')
        p.add_argument('--redraw', action='store_true', help='with --key: the grid after redraws (#171)')
        p.add_argument('--battles', action='store_true', help='usable win rate of every arm (#171)')
        p.add_argument('--out', type=Path)
        p.add_argument('--url')
        p.add_argument('--prompt', choices=G.PROMPTS, default=BEFORE)
        p.add_argument('--compare', action='store_true', help=f'#238: --prompt against {BEFORE}, every model')
    args = parser.parse_args()
    if args.command == 'figure' and args.compare:
        compare_figure(compare_results(args.dir, args.prompt), args.prompt, args.out)
    elif args.command == 'comment' and args.compare:
        print(compare_comment(args.dir, args.prompt, args.url), end='')
    elif args.command == 'figure' and args.battles:
        battle_figure(battle_report(args.dir), args.out)
    elif args.command == 'figure' and args.redraw:
        redraw_figure(final_for(args.dir, args.key), args.key, args.out)
    elif args.command == 'comment' and args.battles:
        print(battle_comment(args.dir, args.url), end='')
    elif args.command == 'comment' and args.redraw:
        print(redraw_comment(args.dir, args.key, args.url), end='')
    elif args.command == 'figure':
        if args.matrix:
            results = {k: rows_for(args.dir, k) for k in G.SMALL_FOUR
                       if (args.dir / 'inplace_header' / k / 'fills.jsonl').exists()}
            matrix_figure(results, args.out)
        else:
            grid_figure(rows_for(args.dir, args.key, args.prompt), args.key, args.out, args.prompt)
    else:
        print(matrix_comment(args.dir, args.url) if args.matrix
              else model_comment(args.dir, args.key, args.url, args.prompt), end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
