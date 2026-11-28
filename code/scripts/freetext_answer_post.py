"""Figures and issue comments for #240: free-text answer instead of in-place mask slots on the held-out grid.

Each small model fills two paired arms (inplace_header, freetext_answer) in one Modal L40S container on all 270
held-out masks. iLLaDA 8B Instruct fills four arms (inplace_header, freetext_answer, inplace_header_example,
freetext_header_example) in one Modal L40S container on all 270 masks.

Figures, storyboarded in code/docs/freetext-answer-prompt.md:
    model        one model: legal on first try beside JSON parse share (plus 2x2 on iLLaDA)
    compare      all five models: first-try legal fills beside JSON parse share, in-place vs free-text
    illada       iLLaDA 8B Instruct: 2x2 matrix (layout x example team) with gained/lost and p-value
    limits       answer-region length-limit hit share (hit_length_limit) by model and task type
    rules        full 17-rule audit matrix across all five models and arms

    figure --dir D --key K --out F
    figure --dir D --compare --out F
    figure --dir D --illada --out F
    figure --dir D --limits --out F
    figure --dir D --rules --out F
    comment --dir D --key K --url U [--audit A]
    comment --dir D --compare --url U
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import inplace_prompt_grid as G
from echarts_chart import write_chart
from inplace_prompt_post import TASK_NAMES, picture

LABELS = {'diffugpt_s': 'DiffuGPT small', 'diffugpt_m': 'DiffuGPT medium', 'qwen25c': 'Masked Qwen2.5 Coder 0.5B',
          'qwen3': 'Masked Qwen3 0.6B', 'illada': 'iLLaDA 8B Instruct'}
SHORT = {'diffugpt_s': 'DiffuGPT S', 'diffugpt_m': 'DiffuGPT M', 'qwen25c': 'Qwen Coder', 'qwen3': 'Qwen3',
         'illada': 'iLLaDA 8B'}
TASK_ORDER = ('items', 'abilities', 'natures', 'stats', 'moves', 'pokemon')
ARM_LABELS = {'inplace_header': 'In-place slots, header only',
              'freetext_answer': 'Free-text answer, header only',
              'inplace_header_example': 'In-place slots, example team',
              'freetext_header_example': 'Free-text answer, example team'}
SHORT_ARM = {'inplace_header': 'in-place', 'freetext_answer': 'free-text',
             'inplace_header_example': 'ex in-place', 'freetext_header_example': 'ex free-text'}

INPLACE_LIGHT, INPLACE_DARK = '#b5b5b0', '#5c5c58'
FREETEXT_LIGHT, FREETEXT_DARK = '#2a78d6', '#86b6ef'
EX_INPLACE_LIGHT, EX_INPLACE_DARK = '#8b8b85', '#787872'
EX_FREETEXT_LIGHT, EX_FREETEXT_DARK = '#166534', '#4ade80'
BAR_DARK = {INPLACE_LIGHT: INPLACE_DARK, FREETEXT_LIGHT: FREETEXT_DARK,
            EX_INPLACE_LIGHT: EX_INPLACE_DARK, EX_FREETEXT_LIGHT: EX_FREETEXT_DARK}

LOW, HIGH = '#e8f1fd', '#0d366b'
LOW_DARK, HIGH_DARK = '#1b2a3d', '#cde2fb'
SURFACE = {'#fcfcfb': '#161615', LOW: LOW_DARK, HIGH: HIGH_DARK}

SETTING_NOTE = ('All 270 held-out masks, one first-try fill per mask, 128 steps, temperature 1.0, mask seeds from '
                'run seed 2026, paired arms filled in one Modal L40S container per model at batch cap 8. '
                'Legal means the fill parses into a six-Pokémon team, keeps every shown value, writes only Reg M-B '
                'names, and Showdown accepts it.')


def rows_for(directory: Path, key: str, prompt: str) -> list[dict[str, Any]]:
    return G.read_fills(directory / prompt / key / 'fills.jsonl')


def alpha_for(key: str) -> float:
    return 0.025 if key == 'illada' else 0.05


def arms_for(directory: Path, key: str) -> list[str]:
    base = ['inplace_header', 'freetext_answer']
    if key == 'illada' and all((directory / p / key / 'fills.jsonl').exists()
                               for p in ('inplace_header_example', 'freetext_header_example')):
        return [*base, 'inplace_header_example', 'freetext_header_example']
    return base


def change_phrase(c: dict[str, Any], baseline: str, alpha: float) -> str:
    before, after = c['legal']['before'], c['legal']['after']
    p = c['paired']
    if p['p_value'] >= alpha:
        return f'matches {baseline} within noise ({before} → {after}, p = {p["p_value"]:.2g})'
    verb = 'raises' if after > before else 'lowers'
    return f'{verb} legal fills over {baseline} ({before} → {after}, p = {p["p_value"]:.2g})'


def model_headline(directory: Path, key: str) -> str:
    alpha = alpha_for(key)
    c_hdr = G.compare_rows(rows_for(directory, key, 'inplace_header'), rows_for(directory, key, 'freetext_answer'))
    phrase = change_phrase(c_hdr, 'in-place header only', alpha)
    if key == 'illada' and all((directory / p / key / 'fills.jsonl').exists()
                               for p in ('inplace_header_example', 'freetext_header_example')):
        c_ex = G.compare_rows(rows_for(directory, key, 'inplace_header_example'),
                              rows_for(directory, key, 'freetext_header_example'))
        ex_phrase = change_phrase(c_ex, 'in-place example team', alpha)
        return f'{LABELS[key]}: free-text {phrase}; {ex_phrase}'
    return f'{LABELS[key]}: free-text {phrase}'


def model_figure(directory: Path, key: str, out: Path) -> Path:
    """Two-panel figure for one model: first-try legal fills (left) beside JSON parse share (right)."""
    arms = arms_for(directory, key)
    rows = {p: rows_for(directory, key, p) for p in arms}
    s = {p: G.summarize(r) for p, r in rows.items()}
    colors = {'inplace_header': INPLACE_LIGHT, 'freetext_answer': FREETEXT_LIGHT,
              'inplace_header_example': EX_INPLACE_LIGHT, 'freetext_header_example': EX_FREETEXT_LIGHT}
    x_labels = [SHORT_ARM[p] for p in arms]
    legal_data = [{'value': s[p]['legal'], 'itemStyle': {'color': colors[p]},
                   'tip': f"{ARM_LABELS[p]}<br>Legal: {s[p]['legal']}/{s[p]['masks']} ({s[p]['legal_share']:.1%})"}
                  for p in arms]
    parse_data = [{'value': round(100 * s[p]['json_parses'] / s[p]['masks'], 1), 'itemStyle': {'color': colors[p]},
                   'tip': f"{ARM_LABELS[p]}<br>Parses as JSON: {s[p]['json_parses']}/{s[p]['masks']}<br>"
                          f"Reads as team: {s[p]['reads_as_team']}/{s[p]['masks']}"}
                  for p in arms]
    max_legal = max([s[p]['legal'] for p in arms] + [10])
    legal_top = ((int(max_legal * 1.25) + 4) // 5) * 5
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': [{'left': 48, 'right': '56%', 'top': 56, 'bottom': 56},
                 {'left': '54%', 'right': 18, 'top': 56, 'bottom': 56}],
        'title': [{'text': 'Legal on the first try (of 270)', 'left': 44, 'top': 16, 'textStyle': {'fontSize': 13}},
                  {'text': 'Parses as JSON (% of 270)', 'left': '53.5%', 'top': 16, 'textStyle': {'fontSize': 13}}],
        'xAxis': [{'type': 'category', 'gridIndex': 0, 'data': x_labels, 'axisLabel': {'interval': 0, 'fontSize': 11}},
                  {'type': 'category', 'gridIndex': 1, 'data': x_labels, 'axisLabel': {'interval': 0, 'fontSize': 11}}],
        'yAxis': [{'type': 'value', 'gridIndex': 0, 'min': 0, 'max': legal_top},
                  {'type': 'value', 'gridIndex': 1, 'min': 0, 'max': 100, 'axisLabel': {'formatter': '{value}%'}}],
        'series': [
            {'type': 'bar', 'name': 'Legal', 'xAxisIndex': 0, 'yAxisIndex': 0, 'data': legal_data, 'barWidth': '50%',
             'label': {'show': True, 'position': 'top', 'fontSize': 12, 'formatter': 'js:p => `${p.value}/270`'},
             'emphasis': {'focus': 'self'}},
            {'type': 'bar', 'name': 'Parses', 'xAxisIndex': 1, 'yAxisIndex': 1, 'data': parse_data, 'barWidth': '50%',
             'label': {'show': True, 'position': 'top', 'fontSize': 12,
                       'formatter': 'js:p => `${p.value.toFixed(0)}%`'},
             'emphasis': {'focus': 'self'}},
        ],
    }
    flag = (' Flagged: DiffuGPT small was trained on 512-token texts; free-text prompts run 510–724 tokens.'
            if key == 'diffugpt_s' else '')
    c_hdr = G.compare_rows(rows['inplace_header'], rows['freetext_answer'])
    p_hdr = c_hdr['paired']
    note = (f"Header-only pair: {c_hdr['legal']['before']} → {c_hdr['legal']['after']} legal "
            f"(+{p_hdr['gained']} / −{p_hdr['lost']}, McNemar p = {p_hdr['p_value']:.3g}, α = {alpha_for(key)}). "
            f"Free-text hit length limit: {s['freetext_answer']['hit_length_limit']}/270.{flag} " + SETTING_NOTE)
    subtitle = ('Held-out grid, free-text answer vs in-place slots (#240)'
                + (' — flagged: 512-token training window' if key == 'diffugpt_s' else ''))
    return write_chart(option, out, title=model_headline(directory, key), subtitle=subtitle, note=note,
                       height=360, dark=BAR_DARK)


def compare_figure(directory: Path, out: Path) -> Path:
    """All five models (plus iLLaDA example pair): legal fills (left) beside JSON parse share (right)."""
    rows_spec: list[tuple[str, str, str, str]] = []
    for key in G.ALL_MODELS:
        if all((directory / p / key / 'fills.jsonl').exists() for p in ('inplace_header', 'freetext_answer')):
            tag = ' *' if key == 'diffugpt_s' else ''
            rows_spec.append((key, 'inplace_header', 'freetext_answer', f'{LABELS[key]}{tag} (header)'))
        if key == 'illada' and all((directory / p / key / 'fills.jsonl').exists()
                                   for p in ('inplace_header_example', 'freetext_header_example')):
            rows_spec.append((key, 'inplace_header_example', 'freetext_header_example',
                              f'{LABELS[key]} (example team)'))
    labels = [r[3] for r in rows_spec]
    comp = [G.compare_rows(rows_for(directory, k, b), rows_for(directory, k, a)) for k, b, a, _ in rows_spec]
    max_legal = max([max(c['legal']['before'], c['legal']['after']) for c in comp] + [10])
    legal_top = ((int(max_legal * 1.25) + 4) // 5) * 5
    series = []
    for panel, measure in enumerate(('legal', 'json_parses')):
        for side, color, name in (('before', INPLACE_LIGHT, 'In-place slots'),
                                  ('after', FREETEXT_LIGHT, 'Free-text answer')):
            vals = [c[measure][side] if measure == 'legal'
                    else round(100 * c[measure][side] / c[measure]['of'], 1) for c in comp]
            fmt = 'js:p => `${p.value}`' if measure == 'legal' else 'js:p => `${p.value.toFixed(0)}%`'
            series.append({'type': 'bar', 'name': name, 'xAxisIndex': panel, 'yAxisIndex': panel,
                           'color': color, 'data': vals, 'barGap': '12%',
                           'label': {'show': True, 'position': 'right', 'fontSize': 11, 'formatter': fmt},
                           'emphasis': {'focus': 'series'}})
    gained = [r[3] for r, c in zip(rows_spec, comp, strict=True)
              if c['paired']['p_value'] < alpha_for(r[0]) and c['legal']['after'] > c['legal']['before']]
    title = (f"Free-text answer raises legal fills for {', '.join(gained)}" if gained
             else 'Free-text answer raises no model above in-place slots')
    option = {
        'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
        'legend': {'top': 0, 'data': ['In-place slots', 'Free-text answer']},
        'grid': [{'left': 225, 'right': '52%', 'top': 56, 'bottom': 44},
                 {'left': '56%', 'right': 40, 'top': 56, 'bottom': 44}],
        'title': [{'text': 'Legal on the first try (of 270)', 'left': 225, 'top': 28, 'textStyle': {'fontSize': 13}},
                  {'text': 'Parses as JSON (% of 270)', 'left': '56%', 'top': 28, 'textStyle': {'fontSize': 13}}],
        'xAxis': [{'type': 'value', 'gridIndex': 0, 'min': 0, 'max': legal_top,
                   'name': 'masks of 270', 'nameLocation': 'middle', 'nameGap': 26},
                  {'type': 'value', 'gridIndex': 1, 'min': 0, 'max': 100,
                   'name': '% of 270 masks', 'nameLocation': 'middle', 'nameGap': 26}],
        'yAxis': [{'type': 'category', 'gridIndex': 0, 'data': labels, 'inverse': True, 'axisLabel': {'fontSize': 11}},
                  {'type': 'category', 'gridIndex': 1, 'data': labels, 'inverse': True, 'axisLabel': {'show': False}}],
        'series': series,
    }
    note = ('*DiffuGPT small is flagged (512-token training window; free-text runs 510–724 tokens). ' + SETTING_NOTE)
    return write_chart(option, out, title=title, subtitle='All five models, paired arms on 270 held-out masks (#240)',
                       note=note, height=160 + 54 * len(rows_spec), dark=BAR_DARK)


def illada_matrix_figure(directory: Path, out: Path) -> Path:
    """iLLaDA 8B Instruct 2x2 matrix: layout (in-place vs free-text) down, base (header vs example) across."""
    grid = (('inplace_header', 'inplace_header_example'), ('freetext_answer', 'freetext_header_example'))
    rows = {p: rows_for(directory, 'illada', p) for row in grid for p in row}
    s = {p: G.summarize(r) for p, r in rows.items()}
    c_hdr = G.compare_rows(rows['inplace_header'], rows['freetext_answer'])
    c_ex = G.compare_rows(rows['inplace_header_example'], rows['freetext_header_example'])
    peak = max([m['legal'] for m in s.values()] + [1])
    data = []
    for y, row in enumerate(grid):
        for x, code in enumerate(row):
            m = s[code]
            text = f"{m['legal']} / {m['masks']}"
            if y == 1:
                p = (c_hdr if x == 0 else c_ex)['paired']
                text += f"\n+{p['gained']}/−{p['lost']} (p={p['p_value']:.2g})"
            data.append({'value': [x, y, m['legal']],
                         'label': {'formatter': text, 'color': '#fff' if m['legal'] >= peak * 0.55 else '#0b0b0b'},
                         'tip': f"{ARM_LABELS[code]}<br>Legal: {m['legal']}/{m['masks']}<br>"
                                f"Parses: {m['json_parses']}/{m['masks']}, team: {m['reads_as_team']}/{m['masks']}"})
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 170, 'right': 20, 'top': 44, 'bottom': 16},
        'xAxis': {'type': 'category', 'position': 'top', 'data': ['Header only', 'Example team on header only'],
                  'axisLabel': {'interval': 0, 'fontSize': 12}},
        'yAxis': {'type': 'category', 'data': ['In-place slots', 'Free-text answer'], 'inverse': True,
                  'axisLabel': {'fontSize': 12}},
        'visualMap': {'show': False, 'min': 0, 'max': peak, 'dimension': 2, 'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data, 'label': {'show': True, 'fontSize': 13, 'lineHeight': 19},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2, 'borderRadius': 4},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('iLLaDA 8B Instruct 2 × 2 design (Bonferroni α = 0.025 across the two bases). Bottom row shows paired '
            'gained / lost and McNemar exact p against the in-place cell directly above it. ' + SETTING_NOTE)
    return write_chart(option, out, title=model_headline(directory, 'illada'),
                       subtitle='iLLaDA 8B Instruct: layout × example team on all 270 held-out masks (#240)',
                       note=note, height=250, dark=SURFACE)


def limits_figure(directory: Path, out: Path) -> Path:
    """Answer-region length-limit hit share (%) by model/arm (rows) and task type (columns)."""
    entries: list[tuple[str, str, str]] = []
    for k in G.ALL_MODELS:
        if (directory / 'freetext_answer' / k / 'fills.jsonl').exists():
            entries.append((k, 'freetext_answer', f'{SHORT[k]} (header)'))
        if k == 'illada' and (directory / 'freetext_header_example' / k / 'fills.jsonl').exists():
            entries.append((k, 'freetext_header_example', f'{SHORT[k]} (example)'))
    data = []
    for y, (k, prompt, label) in enumerate(entries):
        rows = rows_for(directory, k, prompt)
        for x, task in enumerate(TASK_ORDER):
            sub = [r for r in rows if r['task'] == task]
            hits = sum(bool(r.get('hit_length_limit')) for r in sub)
            pct = round(100.0 * hits / len(sub), 1) if sub else 0.0
            data.append({'value': [x, y, pct],
                         'label': {'color': '#fff' if pct >= 55 else '#0b0b0b'},
                         'tip': f'{label}, {TASK_NAMES[task]}<br>Hit limit: {hits}/{len(sub)} ({pct:.1f}%)'})
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 155, 'right': 16, 'top': 42, 'bottom': 16},
        'xAxis': {'type': 'category', 'position': 'top', 'data': [TASK_NAMES[t] for t in TASK_ORDER],
                  'axisLabel': {'interval': 0, 'fontSize': 11}},
        'yAxis': {'type': 'category', 'data': [e[2] for e in entries], 'inverse': True, 'axisLabel': {'fontSize': 11}},
        'visualMap': {'show': False, 'min': 0, 'max': 100, 'dimension': 2, 'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data,
                    'label': {'show': True, 'fontSize': 11, 'formatter': 'js:p => `${Math.round(p.value[2])}%`'},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2, 'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Share of free-text answers that ran to the end of the answer region without emitting an end token '
            '(hit_length_limit). Lower means the model stopped cleanly with its end token. ' + SETTING_NOTE)
    return write_chart(option, out, title='How often free-text answers hit the answer-region length limit',
                       subtitle='Share without an end token, by model and hidden field type (#240)',
                       note=note, height=130 + 46 * len(entries), dark=SURFACE)


def rules_figure(directory: Path, out: Path) -> Path:
    """Full 17-rule break matrix across every completed model and paired arm."""
    columns: list[tuple[str, str]] = []
    for k in G.ALL_MODELS:
        for p in arms_for(directory, k):
            if (directory / p / k / 'fills.jsonl').exists():
                columns.append((k, p))
    counts = {col: G.rule_counts(rows_for(directory, col[0], col[1])) for col in columns}
    data = []
    for y, rule in enumerate(G.RULES):
        for x, col in enumerate(columns):
            c = counts[col][rule]
            share = c['broken'] / c['of'] if c['of'] else None
            data.append({'value': [x, y, None if share is None else round(100 * share, 1)],
                         'tip': f"{LABELS[col[0]]}, {ARM_LABELS[col[1]]}<br><b>{rule}</b><br>"
                                f"{c['broken']} of {c['of']} fills",
                         'label': {'color': '#fff' if share and share > 0.55 else '#0b0b0b'}})
    col_labels = [f"{SHORT[k]}\n{SHORT_ARM[p]}" for k, p in columns]
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 200, 'right': 8, 'top': 54, 'bottom': 10},
        'xAxis': {'type': 'category', 'position': 'top', 'data': col_labels,
                  'axisLabel': {'interval': 0, 'fontSize': 9, 'lineHeight': 12}},
        'yAxis': {'type': 'category', 'data': list(G.RULES), 'inverse': True, 'axisLabel': {'fontSize': 11}},
        'visualMap': {'show': False, 'min': 0, 'max': 100, 'dimension': 2, 'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data,
                    'label': {'show': True, 'fontSize': 9,
                              'formatter': 'js:p => p.value[2] == null ? "–" : `${Math.round(p.value[2])}`'},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 1, 'borderRadius': 2},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Percent of first-try fills breaking each rule, out of fills where it could be judged, on all 270 masks. '
            + SETTING_NOTE)
    return write_chart(option, out, title='Which rules the first-try fills break, in-place vs free-text',
                       subtitle='Full rule audit matrix across all five models (#240)', note=note,
                       height=130 + 30 * len(G.RULES), dark=SURFACE)


def model_comment(directory: Path, key: str, url: str, audit_path: Path | None = None) -> str:
    arms = arms_for(directory, key)
    rows = {p: rows_for(directory, key, p) for p in arms}
    s = {p: G.summarize(r) for p, r in rows.items()}
    c_hdr = G.compare_rows(rows['inplace_header'], rows['freetext_answer'])
    p_hdr = c_hdr['paired']
    alt = f"{model_headline(directory, key)}. Header only: in-place {s['inplace_header']['legal']}/270, free-text {s['freetext_answer']['legal']}/270."
    lines = [f'## {model_headline(directory, key)}', '', picture(url, f'freetext-{key}', alt), '']
    if key == 'diffugpt_s':
        lines += ['> **Flagged:** DiffuGPT small was trained on 512-token texts (Gong et al. 2024, Table 4). '
                  'Its free-text prompts run 510–724 tokens, pushing almost every mask past 512.', '']
    lines += [
        '| Arm (`code ID`) | Legal, first try | Parses as JSON | Reads as team | Real species | Hit length limit | Mean prompt tokens | Seconds / team |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for p in arms:
        m = s[p]
        sp = m['real_values'].get('species', {'real': 0, 'of': 0})
        hit = f"{m['hit_length_limit']}/{m['masks']} ({m['hit_length_limit_share']:.1%})" if 'hit_length_limit' in m else '–'
        lines.append(f"| {ARM_LABELS[p]} (`{p}`) | {m['legal']}/{m['masks']} ({m['legal_share']:.1%}) | "
                     f"{m['json_parses']}/{m['masks']} | {m['reads_as_team']}/{m['masks']} | "
                     f"{sp['real']}/{sp['of']} | {hit} | {m['prompt_tokens_mean']:.0f} | {m['seconds_per_team']:.2f} |")
    lines += [
        '',
        f"- **Header-only pair** (`inplace_header` → `freetext_answer`): {c_hdr['legal']['before']} → {c_hdr['legal']['after']} "
        f"legal; +{p_hdr['gained']} gained, −{p_hdr['lost']} lost (McNemar exact p = {p_hdr['p_value']:.3g}, α = {alpha_for(key)}).",
    ]
    if 'freetext_header_example' in rows:
        c_ex = G.compare_rows(rows['inplace_header_example'], rows['freetext_header_example'])
        p_ex = c_ex['paired']
        lines.append(f"- **Example-team pair** (`inplace_header_example` → `freetext_header_example`): "
                     f"{c_ex['legal']['before']} → {c_ex['legal']['after']} legal; +{p_ex['gained']} gained, "
                     f"−{p_ex['lost']} lost (McNemar exact p = {p_ex['p_value']:.3g}, α = {alpha_for(key)}).")
    tasks_ft = ', '.join(f"{TASK_NAMES[t]} {v['legal']}/{v['of']}"
                         for t, v in s['freetext_answer']['legal_by_task'].items() if v['legal'])
    lines.append(f"- **Legal under `freetext_answer` by mask type:** {tasks_ft or 'none'}.")
    if 'freetext_header_example' in s:
        tasks_ex = ', '.join(f"{TASK_NAMES[t]} {v['legal']}/{v['of']}"
                             for t, v in s['freetext_header_example']['legal_by_task'].items() if v['legal'])
        lines.append(f"- **Legal under `freetext_header_example` by mask type:** {tasks_ex or 'none'}.")
    lines.append('- **Most common failure reasons (`freetext_answer`):** '
                 + '; '.join(f'{r} ({n})' for r, n in c_hdr['reasons']['after'][:4]) + '.')
    if s['freetext_answer'].get('had_code_fence'):
        lines.append(f"- **Markdown code fences unwrapped (`freetext_answer`):** {s['freetext_answer']['had_code_fence']}/270 "
                     f"(strict JSON parses without unwrapping: {s['freetext_answer']['json_parses_strict']}/270).")
    probe = json.loads((directory / 'preflight' / f'modal-gpu-{key}.json').read_text())
    meta = json.loads((directory / 'freetext_answer' / key / 'run.json').read_text())
    lines.append(f"- **Probes & memory** (on `{probe['prompt']}`): sees later tokens {probe['sees_later_tokens']}; "
                 f"repeat {probe['repeat_same_tokens']:.0%}; padding pad-leak {probe['padding']['pad_leak_max_diff']:.2g}; "
                 f"batch {meta['setting']['batch']} on {probe['device']}, peak {s['freetext_answer']['peak_gib']:.2f} GiB.")
    if audit_path is not None and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        parts = []
        for p in arms:
            a = audit.get(f'{p}/{key}')
            if a:
                bad = (len(a['rejudge_mismatch']) + len(a['legal_but_broken'])
                       + len(a['illegal_but_clean']) + len(a['step_without_rule']))
                parts.append(f"`{p}` {a['fills']} fills, {bad} audit issues")
        if parts:
            lines.append('- **Rule audit (`inplace_rule_audit.py`):** ' + '; '.join(parts) + '.')
    where = str(directory).replace(str(Path.home()), '~')
    lines += [
        '',
        f"<details><summary>Setting</summary>\n\n{meta['candidate']['repo']} at `{meta['candidate']['revision'][:8]}`, "
        f"{meta['setting']}, torch {meta['environment']['torch']}, transformers {meta['environment']['transformers']}. "
        f"Grid file SHA-256 `{meta['tasks_sha256'][:12]}`. Fills: `{where}/<prompt>/{key}/fills.jsonl`. "
        'Pre-registration: `code/docs/freetext-answer-prompt.md`.\n\n</details>',
    ]
    return '\n'.join(lines) + '\n'


def compare_comment(directory: Path, url: str) -> str:
    pairs: list[tuple[str, str, str, str]] = []
    for k in G.ALL_MODELS:
        if all((directory / p / k / 'fills.jsonl').exists() for p in ('inplace_header', 'freetext_answer')):
            tag = ' (flagged: 512 window)' if k == 'diffugpt_s' else ''
            pairs.append((k, 'inplace_header', 'freetext_answer', f'{LABELS[k]}{tag}, header only'))
        if k == 'illada' and all((directory / p / k / 'fills.jsonl').exists()
                                 for p in ('inplace_header_example', 'freetext_header_example')):
            pairs.append((k, 'inplace_header_example', 'freetext_header_example',
                          f'{LABELS[k]}, example team'))
    lines = [
        '## All five models: free-text answer vs in-place mask slots on the held-out grid',
        '',
        picture(url, 'freetext-compare', 'First-try legal fills and JSON parse share across all five models, in-place slots vs free-text answer.'),
        '',
        picture(url, 'freetext-illada-2x2', 'iLLaDA 8B Instruct 2x2 matrix of layout (in-place vs free-text) by example team (header only vs example team).'),
        '',
        picture(url, 'freetext-limits', 'Share of free-text answers hitting the answer-region length limit by model and task type.'),
        '',
        picture(url, 'freetext-rules', 'Full 17-rule break matrix across all five models and arms on 270 masks.'),
        '',
        'Each pair was filled in one Modal L40S container on all 270 held-out masks with identical batching and mask '
        'seeds. Significance: α = 0.05 on each small model, Bonferroni α = 0.025 on iLLaDA 8B Instruct (two bases).',
        '',
        '| Model & base | Legal, first try | Gained / lost (McNemar p) | Parses as JSON | Reads as team | Real species | Hit length limit | Seconds / team |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for k, b, a, label in pairs:
        rb, ra = rows_for(directory, k, b), rows_for(directory, k, a)
        c, sa = G.compare_rows(rb, ra), G.summarize(ra)
        p = c['paired']
        lines.append(
            f"| {label} | {c['legal']['before']} → {c['legal']['after']} | +{p['gained']}/−{p['lost']} ({p['p_value']:.3g}) | "
            f"{c['json_parses']['before']} → {c['json_parses']['after']} | "
            f"{c['reads_as_team']['before']} → {c['reads_as_team']['after']} | "
            f"{c['real_species']['before']} → {c['real_species']['after']} / {c['real_species']['of']} | "
            f"{sa['hit_length_limit']}/270 ({sa['hit_length_limit_share']:.0%}) | "
            f"{c['seconds_per_team']['before']:.2f} → {c['seconds_per_team']['after']:.2f} |"
        )
    where = str(directory).replace(str(Path.home()), '~')
    lines += ['', 'Pre-registration and prediction: `code/docs/freetext-answer-prompt.md`. '
              f'Fills: `{where}/<prompt>/<model>/fills.jsonl`.']
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('figure', 'comment'):
        p = sub.add_parser(name)
        p.add_argument('--dir', type=Path, required=True)
        p.add_argument('--key', choices=G.ALL_MODELS)
        p.add_argument('--compare', action='store_true')
        p.add_argument('--illada', action='store_true')
        p.add_argument('--limits', action='store_true')
        p.add_argument('--rules', action='store_true')
        p.add_argument('--out', type=Path)
        p.add_argument('--url')
        p.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.command == 'figure':
        if args.compare:
            compare_figure(args.dir, args.out)
        elif args.illada:
            illada_matrix_figure(args.dir, args.out)
        elif args.limits:
            limits_figure(args.dir, args.out)
        elif args.rules:
            rules_figure(args.dir, args.out)
        else:
            model_figure(args.dir, args.key, args.out)
        return 0
    print(compare_comment(args.dir, args.url) if args.compare
          else model_comment(args.dir, args.key, args.url, args.audit), end='')
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
