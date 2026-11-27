"""Figures and issue comments for #445: three team-building rules spelled out in the in-place prompt, against a
neutral length-control line and the header-only prompt.

Each model fills all three prompts in one Modal L40S container on all 270 held-out masks, with identical batching
and mask seeds. Figures, storyboarded in code/docs/inplace-prompt-rules.md:
    model        one model: first-try legal fills beside the three named rule-break shares, across all three arms
    compare      every model: first-try legal fills under the three arms
    three-rules  every model: the three named rule-break shares across the three arms
    rules        every model: the full 17-rule matrix across the three arms

    figure --dir D --key K --out F             one model's figure
    figure --dir D --compare --out F           the all-models legality matrix
    figure --dir D --three-rules --out F       the three named rules matrix
    figure --dir D --rules --out F             the full rule matrix
    comment --dir D --key K --url U            one model's comment; U holds {name} and {theme}
    comment --dir D --compare --url U          the summary comment across every model

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
from inplace_prompt_post import LABELS, SHORT, TASK_NAMES, picture

ARMS = (('inplace_header', 'No rules spelled out'),
        ('inplace_filler', 'Neutral line of the same length'),
        ('inplace_rules', 'Rules spelled out'))
ARM_NAMES = dict(ARMS)
SHORT_ARMS = {'inplace_header': 'none', 'inplace_filler': 'filler', 'inplace_rules': 'rules'}
NAMED_RULES = ('item used more than once', 'Stat Points over 66', 'a Stat Point over 32')
ALPHA = 0.025  # two primary comparisons per model (rules vs filler, rules vs header), Bonferroni

HEADER_LIGHT, HEADER_DARK = '#b5b5b0', '#5c5c58'
FILLER_LIGHT, FILLER_DARK = '#d9913b', '#e5a858'
RULES_LIGHT, RULES_DARK = '#2a78d6', '#86b6ef'
ARM_COLORS = {'inplace_header': HEADER_LIGHT, 'inplace_filler': FILLER_LIGHT, 'inplace_rules': RULES_LIGHT}
BAR_DARK = {HEADER_LIGHT: HEADER_DARK, FILLER_LIGHT: FILLER_DARK, RULES_LIGHT: RULES_DARK}

LOW, HIGH = '#e8f1fd', '#0d366b'
LOW_DARK, HIGH_DARK = '#1b2a3d', '#cde2fb'
SURFACE = {'#fcfcfb': '#161615', LOW: LOW_DARK, HIGH: HIGH_DARK}

SETTING_NOTE = ('All 270 held-out masks, one first-try fill per mask, 128 steps, temperature 1.0, mask seeds from '
                'run seed 2026, all three arms filled in one Modal L40S container per model with identical batching. '
                'Legal means the fill parses as a team, keeps every shown value, writes only Reg M-B names and '
                'Showdown accepts it.')


def rows_for(directory: Path, key: str, prompt: str) -> list[dict[str, Any]]:
    return G.read_fills(directory / prompt / key / 'fills.jsonl')


def model_stats(directory: Path, key: str) -> dict[str, Any]:
    rows = {code: rows_for(directory, key, code) for code, _ in ARMS}
    summaries = {code: G.summarize(r) for code, r in rows.items()}
    return {
        'rows': rows,
        'summaries': summaries,
        'vs_filler': G.compare_rows(rows['inplace_filler'], rows['inplace_rules']),
        'vs_header': G.compare_rows(rows['inplace_header'], rows['inplace_rules']),
        'filler_vs_header': G.compare_rows(rows['inplace_header'], rows['inplace_filler']),
    }


def rule_pct(summary: dict[str, Any], rule: str) -> tuple[float | None, int, int]:
    c = summary['rules'][rule]
    if not c['of']:
        return None, 0, 0
    return 100.0 * c['broken'] / c['of'], c['broken'], c['of']


def verdict(p: dict[str, Any]) -> str:
    return 'significant' if p['p_value'] < ALPHA else 'within noise'


def change_phrase(c: dict[str, Any], baseline: str) -> str:
    before, after = c['legal']['before'], c['legal']['after']
    p = c['paired']
    if p['p_value'] >= ALPHA:
        return f'matches {baseline} within noise ({before} → {after}, p = {p["p_value"]:.2g})'
    verb = 'raises' if after > before else 'lowers'
    return f'{verb} legal fills over {baseline} ({before} → {after}, p = {p["p_value"]:.2g})'


def headline(key: str, stats: dict[str, Any]) -> str:
    content = change_phrase(stats['vs_filler'], 'the neutral line')
    total = change_phrase(stats['vs_header'], 'header only')
    return f'{LABELS[key]}: rules spelled out {content}; {total}'


def model_figure(directory: Path, key: str, out: Path) -> Path:
    """Two-panel figure for one model: first-try legal fills beside the three named rule-break shares."""
    stats = model_stats(directory, key)
    s = stats['summaries']
    series = []
    # Left panel: legal fills per arm (xAxisIndex 0, yAxisIndex 0)
    legal_data = [{'value': s[code]['legal'], 'itemStyle': {'color': ARM_COLORS[code]},
                   'tip': f"{label}<br>Legal: {s[code]['legal']} of {s[code]['masks']} "
                          f"({s[code]['legal_share']:.1%})"}
                  for code, label in ARMS]
    series.append({
        'type': 'bar', 'name': 'Legal fills', 'xAxisIndex': 0, 'yAxisIndex': 0,
        'data': legal_data, 'barWidth': '50%',
        'label': {'show': True, 'position': 'top', 'fontSize': 12,
                  'formatter': 'js:p => `${p.value}/270`'},
        'emphasis': {'focus': 'self'},
    })
    # Right panel: three named rules (% of parsed fills), one bar series per arm
    for code, label in ARMS:
        vals = []
        for rule in NAMED_RULES:
            pct, broken, of = rule_pct(s[code], rule)
            vals.append({
                'value': 0.0 if pct is None else round(pct, 1),
                'tip': f"{label}<br><b>{rule}</b>: {broken} of {of} parsed fills "
                       + (f'({pct:.1f}%)' if pct is not None else '(not judged)'),
            })
        series.append({
            'type': 'bar', 'name': label, 'xAxisIndex': 1, 'yAxisIndex': 1,
            'color': ARM_COLORS[code], 'data': vals, 'barGap': '12%',
            'label': {'show': True, 'position': 'top', 'fontSize': 11,
                      'formatter': 'js:p => `${p.value.toFixed(0)}%`'},
            'emphasis': {'focus': 'series'},
        })
    max_legal = max([s[code]['legal'] for code, _ in ARMS] + [10])
    legal_top = ((int(max_legal * 1.2) + 4) // 5) * 5
    max_rule = max([rule_pct(s[code], r)[0] or 0.0 for code, _ in ARMS for r in NAMED_RULES] + [15.0])
    rule_top = min(100, ((int(max_rule * 1.25) + 9) // 5) * 5)
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'legend': {'top': 0, 'data': [label for _, label in ARMS]},
        'grid': [{'left': 48, 'right': '63%', 'top': 72, 'bottom': 56},
                 {'left': '44%', 'right': 18, 'top': 72, 'bottom': 56}],
        'title': [{'text': 'Legal on the first try (of 270)', 'left': 44, 'top': 36, 'textStyle': {'fontSize': 13}},
                  {'text': 'Three named rules broken (% of parsed fills)', 'left': '43.5%', 'top': 36,
                   'textStyle': {'fontSize': 13}}],
        'xAxis': [
            {'type': 'category', 'gridIndex': 0, 'data': ['No rules', 'Neutral line', 'Rules'],
             'axisLabel': {'interval': 0, 'fontSize': 11}},
            {'type': 'category', 'gridIndex': 1,
             'data': ['Item used\nmore than once', 'Stat Points\nover 66', 'A Stat Point\nover 32'],
             'axisLabel': {'interval': 0, 'fontSize': 11, 'lineHeight': 14}},
        ],
        'yAxis': [
            {'type': 'value', 'gridIndex': 0, 'min': 0, 'max': legal_top},
            {'type': 'value', 'gridIndex': 1, 'min': 0, 'max': rule_top,
             'axisLabel': {'formatter': '{value}%'}},
        ],
        'series': series,
    }
    flag = (' Flagged: DiffuGPT small was trained on 512-token texts; adding 39 tokens moves 6 of 30 Stat Points '
            'masks (221 of 270 overall) past 512.' if key == 'diffugpt_s' else '')
    cf, ch = stats['vs_filler']['paired'], stats['vs_header']['paired']
    note = (f"Content effect (rules vs neutral line): +{cf['gained']} / −{cf['lost']}, McNemar p = {cf['p_value']:.3g}. "
            f"Total effect (rules vs header only): +{ch['gained']} / −{ch['lost']}, p = {ch['p_value']:.3g} "
            f"(Bonferroni α = {ALPHA}).{flag} " + SETTING_NOTE)
    subtitle = ('Held-out grid, three team-building rules vs neutral length control (#445)'
                + (' — flagged: 512-token training window' if key == 'diffugpt_s' else ''))
    return write_chart(option, out, title=headline(key, stats), subtitle=subtitle, note=note,
                       height=380, dark=BAR_DARK)


def compare_figure(directory: Path, out: Path) -> Path:
    """Matrix of models x 3 arms: first-try legal fills of 270."""
    keys = [k for k in G.SMALL_FOUR if all((directory / p / k / 'fills.jsonl').exists() for p, _ in ARMS)]
    all_stats = {k: model_stats(directory, k) for k in keys}
    peak = max([all_stats[k]['summaries'][code]['legal'] for k in keys for code, _ in ARMS] + [1])
    data = []
    for y, k in enumerate(keys):
        for x, (code, label) in enumerate(ARMS):
            s = all_stats[k]['summaries'][code]
            n = s['legal']
            text = f"{n} of {s['masks']}"
            if code == 'inplace_rules':
                pf = all_stats[k]['vs_filler']['paired']
                text += f"\nvs filler +{pf['gained']}/−{pf['lost']}"
            data.append({'value': [x, y, n], 'label': {'formatter': text,
                                                       'color': '#fff' if n >= peak * 0.55 else '#0b0b0b'},
                         'tip': f"{LABELS[k]}<br>{label}: {n} of {s['masks']} masks legal"})
    moved_content = [k for k in keys if all_stats[k]['vs_filler']['paired']['p_value'] < ALPHA]
    if not moved_content:
        title = 'Spelling out three team-building rules moves no model beyond its length control'
    else:
        gains = [k for k in moved_content if all_stats[k]['vs_filler']['paired']['gained']
                 > all_stats[k]['vs_filler']['paired']['lost']]
        if gains:
            title = f"Spelling out three rules raises legal fills beyond the length control for {', '.join(LABELS[k] for k in gains)}"
        else:
            title = f"Spelling out three rules lowers legal fills against the length control for {', '.join(LABELS[k] for k in moved_content)}"
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 195, 'right': 12, 'top': 44, 'bottom': 12},
        'xAxis': {'type': 'category', 'position': 'top', 'data': [label for _, label in ARMS],
                  'axisLabel': {'interval': 0, 'fontSize': 12}},
        'yAxis': {'type': 'category', 'data': [LABELS[k] + (' *' if k == 'diffugpt_s' else '') for k in keys],
                  'inverse': True, 'axisLabel': {'fontSize': 12}},
        'visualMap': {'show': False, 'min': 0, 'max': max([d['value'][2] for d in data] + [1]), 'dimension': 2,
                      'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data, 'label': {'show': True, 'fontSize': 12, 'lineHeight': 17},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2, 'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Legal first-try fills of 270 held-out masks. *DiffuGPT small was trained on 512-token texts; +39 tokens '
            'moves 6 of 30 Stat Points masks (221 of 270 overall) past 512. ' + SETTING_NOTE)
    return write_chart(option, out, title=title, subtitle='Held-out grid, three arms per model (#445)',
                       note=note, height=130 + 62 * len(keys), dark=SURFACE)


def three_rules_figure(directory: Path, out: Path) -> Path:
    """Matrix of the three named rules x (models x 3 arms): % of parsed fills breaking each rule."""
    keys = [k for k in G.SMALL_FOUR if all((directory / p / k / 'fills.jsonl').exists() for p, _ in ARMS)]
    columns = [(k, code) for k in keys for code, _ in ARMS]
    counts = {(k, code): G.rule_counts(rows_for(directory, k, code)) for k, code in columns}
    raw_shares = [100.0 * counts[col][rule]['broken'] / counts[col][rule]['of']
                  for rule in NAMED_RULES for col in columns if counts[col][rule]['of']]
    peak = max([*raw_shares, 10.0])
    data = []
    for y, rule in enumerate(NAMED_RULES):
        for x, col in enumerate(columns):
            c = counts[col][rule]
            share = 100.0 * c['broken'] / c['of'] if c['of'] else None
            data.append({'value': [x, y, None if share is None else round(share, 1)],
                         'tip': f"{LABELS[col[0]]}, {ARM_NAMES[col[1]]}<br><b>{rule}</b><br>"
                                f"{c['broken']} of {c['of']} parsed fills",
                         'label': {'color': '#fff' if share and share >= peak * 0.55 else '#0b0b0b'}})
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 190, 'right': 10, 'top': 64, 'bottom': 12},
        'xAxis': [{'type': 'category', 'position': 'top', 'data': [SHORT_ARMS[p] for _, p in columns],
                   'axisLabel': {'interval': 0, 'fontSize': 10}, 'axisTick': {'show': False}},
                  {'type': 'category', 'position': 'top', 'offset': 22,
                   'data': [SHORT[k] + ('*' if k == 'diffugpt_s' else '') for k in keys],
                   'axisLabel': {'interval': 0, 'fontSize': 12, 'fontWeight': 'bold'},
                   'axisTick': {'show': True, 'length': 30}, 'axisLine': {'show': False}}],
        'yAxis': {'type': 'category', 'data': list(NAMED_RULES), 'inverse': True, 'axisLabel': {'fontSize': 12}},
        'visualMap': {'show': False, 'min': 0, 'max': peak, 'dimension': 2, 'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data,
                    'label': {'show': True, 'fontSize': 11,
                              'formatter': 'js:p => p.value[2] == null ? "–" : `${p.value[2].toFixed(1)}%`'},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 2, 'borderRadius': 3},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Share of parsed fills breaking each of the three named rules. Columns per model: none = header only, '
            'filler = neutral line of the same token count, rules = three rules spelled out. *DiffuGPT small is '
            'flagged (512-token training window). ' + SETTING_NOTE)
    return write_chart(option, out, title='Share of parsed fills breaking each of the three named rules',
                       subtitle='Held-out grid, 270 masks per column (#445)', note=note,
                       height=240, dark=SURFACE)


def rules_figure(directory: Path, out: Path) -> Path:
    """Full 17-rule matrix across all four models x three arms."""
    keys = [k for k in G.SMALL_FOUR if all((directory / p / k / 'fills.jsonl').exists() for p, _ in ARMS)]
    columns = [(k, code) for k in keys for code, _ in ARMS]
    counts = {(k, code): G.rule_counts(rows_for(directory, k, code)) for k, code in columns}
    data = []
    for y, rule in enumerate(G.RULES):
        for x, col in enumerate(columns):
            c = counts[col][rule]
            share = c['broken'] / c['of'] if c['of'] else None
            data.append({'value': [x, y, None if share is None else round(100 * share, 1)],
                         'tip': f"{LABELS[col[0]]}, {ARM_NAMES[col[1]]}<br><b>{rule}</b><br>"
                                f"{c['broken']} of {c['of']} fills",
                         'label': {'color': '#fff' if share and share > 0.55 else '#0b0b0b'}})
    option = {
        'tooltip': {'trigger': 'item', 'formatter': 'js:p => p.data.tip'},
        'grid': {'left': 200, 'right': 8, 'top': 64, 'bottom': 10},
        'xAxis': [{'type': 'category', 'position': 'top', 'data': [SHORT_ARMS[p] for _, p in columns],
                   'axisLabel': {'interval': 0, 'fontSize': 10}, 'axisTick': {'show': False}},
                  {'type': 'category', 'position': 'top', 'offset': 22, 'data': [SHORT[k] for k in keys],
                   'axisLabel': {'interval': 0, 'fontSize': 12, 'fontWeight': 'bold'},
                   'axisTick': {'show': True, 'length': 30}, 'axisLine': {'show': False}}],
        'yAxis': {'type': 'category', 'data': list(G.RULES), 'inverse': True, 'axisLabel': {'fontSize': 11}},
        'visualMap': {'show': False, 'min': 0, 'max': 100, 'dimension': 2, 'inRange': {'color': [LOW, HIGH]}},
        'series': [{'type': 'heatmap', 'data': data,
                    'label': {'show': True, 'fontSize': 9,
                              'formatter': 'js:p => p.value[2] == null ? "–" : `${Math.round(p.value[2])}`'},
                    'itemStyle': {'borderColor': '#fcfcfb', 'borderWidth': 1, 'borderRadius': 2},
                    'emphasis': {'itemStyle': {'borderColor': '#0b0b0b', 'borderWidth': 1}}}],
    }
    note = ('Percent of first-try fills breaking each rule, out of the fills where it could be judged, on all 270 '
            'masks. Columns per model: none = header only, filler = neutral line of the same token count, rules = '
            'three rules spelled out. One fill can break several rules. ' + SETTING_NOTE)
    return write_chart(option, out, title='Which rules the first-try fills break, by model and arm',
                       subtitle='Full rule audit matrix, 270 masks per column (#445)', note=note,
                       height=130 + 30 * len(G.RULES), dark=SURFACE)


def _fmt_rule(summary: dict[str, Any], rule: str) -> str:
    pct, broken, of = rule_pct(summary, rule)
    return '–' if pct is None else f'{broken}/{of} ({pct:.1f}%)'


def model_comment(directory: Path, key: str, url: str, audit_path: Path | None = None) -> str:
    stats = model_stats(directory, key)
    s = stats['summaries']
    cf, ch = stats['vs_filler'], stats['vs_header']
    pf, ph = cf['paired'], ch['paired']
    alt = (f"{headline(key, stats)}. Legal of 270: no rules {s['inplace_header']['legal']}, "
           f"neutral line {s['inplace_filler']['legal']}, rules spelled out {s['inplace_rules']['legal']}.")
    lines = [f'## {headline(key, stats)}', '', picture(url, f'rules-{key}', alt), '']
    if key == 'diffugpt_s':
        lines += ['> **Flagged:** DiffuGPT small was trained on 512-token texts (Gong et al. 2024, Table 4). Adding 39 '
                  'tokens moves 6 of 30 Stat Points masks (and 221 of 270 masks overall) past 512. Both `inplace_rules` '
                  'and `inplace_filler` share this exact shift, so rules vs neutral line isolates content from position.',
                  '']
    lines += [
        '| Arm (`code ID`) | Legal, first try | Parses as JSON | Parses as team | Item used >1× | Stat Points >66 | A Stat Point >32 | Mean prompt tokens | Seconds / team |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for code, label in ARMS:
        m = s[code]
        lines.append(f"| {label} (`{code}`) | {m['legal']}/270 ({m['legal_share']:.1%}) | "
                     f"{m['json_parses']}/270 | {m['reads_as_team']}/270 | "
                     f"{_fmt_rule(m, NAMED_RULES[0])} | {_fmt_rule(m, NAMED_RULES[1])} | "
                     f"{_fmt_rule(m, NAMED_RULES[2])} | {m['prompt_tokens_mean']:.0f} | "
                     f"{m['seconds_per_team']:.2f} |")
    tasks_rules = ', '.join(f"{TASK_NAMES[t]} {v['legal']}" for t, v in s['inplace_rules']['legal_by_task'].items()
                            if v['legal'])
    probe = json.loads((directory / 'preflight' / f'modal-gpu-{key}.json').read_text())
    meta = json.loads((directory / 'inplace_rules' / key / 'run.json').read_text())
    where = str(directory).replace(str(Path.home()), '~')
    audit_note = ''
    if audit_path is not None and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        parts = []
        for code, _ in ARMS:
            a = audit.get(f'{code}/{key}')
            if a:
                bad = (len(a['rejudge_mismatch']) + len(a['legal_but_broken'])
                       + len(a['illegal_but_clean']) + len(a['step_without_rule']))
                parts.append(f"`{code}` {a['fills']} fills, {bad} audit issues")
        if parts:
            audit_note = '- **Rule audit (`inplace_rule_audit.py`):** ' + '; '.join(parts) + '.'
    lines += [
        '',
        f"- **Content effect** (`inplace_rules` vs `inplace_filler`): {cf['legal']['before']} → {cf['legal']['after']} "
        f"legal; +{pf['gained']} gained, −{pf['lost']} lost (McNemar exact p = {pf['p_value']:.3g}, "
        f"{verdict(pf)} at α = {ALPHA}).",
        f"- **Total effect** (`inplace_rules` vs `inplace_header`): {ch['legal']['before']} → {ch['legal']['after']} "
        f"legal; +{ph['gained']} gained, −{ph['lost']} lost (McNemar exact p = {ph['p_value']:.3g}, "
        f"{verdict(ph)} at α = {ALPHA}).",
        f"- **Position / length effect alone** (`inplace_filler` vs `inplace_header`): "
        f"{stats['filler_vs_header']['legal']['before']} → {stats['filler_vs_header']['legal']['after']} legal "
        f"(+{stats['filler_vs_header']['paired']['gained']} / −{stats['filler_vs_header']['paired']['lost']}, "
        f"p = {stats['filler_vs_header']['paired']['p_value']:.3g}).",
        f"- **Legal with rules spelled out by what the mask hides:** {tasks_rules or 'none'}.",
        '- **Most common failure reasons with rules spelled out:** '
        + '; '.join(f'{r} ({n})' for r, n in ch['reasons']['after'][:4]) + '.',
        f"- **Probes before the fill** (on a `{probe['prompt']}` mask): attention sees later tokens "
        f"{probe['sees_later_tokens']}; same seed twice {probe['repeat_same_tokens']:.0%}; "
        f"padding top-1 same {probe['padding']['top1_same']:.0%}; batch {meta['setting']['batch']} on "
        f"{probe['device']}, peak {s['inplace_rules']['peak_gib']:.2f} GiB.",
    ]
    if audit_note:
        lines.append(audit_note)
    lines += [
        '',
        f"<details><summary>Setting</summary>\n\n{meta['candidate']['repo']} at `{meta['candidate']['revision'][:8]}`, "
        f"{meta['setting']}, torch {meta['environment']['torch']}, transformers "
        f"{meta['environment']['transformers']}. Grid file SHA-256 `{meta['tasks_sha256'][:12]}`. "
        f"Fills: `{where}/<prompt>/{key}/fills.jsonl`. Pre-registration: `code/docs/inplace-prompt-rules.md`.\n\n</details>",
    ]
    return '\n'.join(lines) + '\n'


def compare_comment(directory: Path, url: str) -> str:
    keys = [k for k in G.SMALL_FOUR if all((directory / p / k / 'fills.jsonl').exists() for p, _ in ARMS)]
    all_stats = {k: model_stats(directory, k) for k in keys}
    alt = '; '.join(f"{LABELS[k]}: " + ', '.join(f"{ARM_NAMES[p].lower()} {all_stats[k]['summaries'][p]['legal']}"
                                                 for p, _ in ARMS) for k in keys)
    lines = [
        '## All four small models: three team-building rules spelled out vs neutral length control and header only',
        '',
        picture(url, 'rules-compare', alt),
        '',
        picture(url, 'rules-three', 'Share of parsed fills breaking item used more than once, Stat Points over 66, '
                'and a Stat Point over 32 across all four models and three arms.'),
        '',
        picture(url, 'rules-matrix', 'Full 17-rule break matrix across all four models and three arms on 270 masks.'),
        '',
        f'Each model filled all three arms in one Modal L40S container on all 270 held-out masks with identical '
        f'batching and mask seeds. Significant means McNemar exact p < {ALPHA} (two comparisons per model, Bonferroni).',
        '',
        '| Model | No rules | Neutral line | Rules spelled out | Content (rules vs filler, p) | Total (rules vs header, p) | Item >1× (none / filler / rules) | SP >66 (none / filler / rules) | SP >32 (none / filler / rules) |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for k in keys:
        st = all_stats[k]
        s = st['summaries']
        pf, ph = st['vs_filler']['paired'], st['vs_header']['paired']

        def triple(rule: str) -> str:
            parts = []
            for code, _ in ARMS:
                pct, _, _ = rule_pct(s[code], rule)
                parts.append('–' if pct is None else f'{pct:.1f}%')
            return ' / '.join(parts)

        name = LABELS[k] + (' (flagged: 512 window)' if k == 'diffugpt_s' else '')
        lines.append(
            f"| {name} | {s['inplace_header']['legal']}/270 | {s['inplace_filler']['legal']}/270 | "
            f"{s['inplace_rules']['legal']}/270 | +{pf['gained']}/−{pf['lost']} ({pf['p_value']:.3g}) | "
            f"+{ph['gained']}/−{ph['lost']} ({ph['p_value']:.3g}) | "
            f"{triple(NAMED_RULES[0])} | {triple(NAMED_RULES[1])} | {triple(NAMED_RULES[2])} |"
        )
    lines += [
        '',
        'Pre-registration and prediction: `code/docs/inplace-prompt-rules.md`. '
        f"Fills: `{str(directory).replace(str(Path.home()), '~')}/<prompt>/<model>/fills.jsonl`.",
    ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('figure', 'comment'):
        p = sub.add_parser(name)
        p.add_argument('--dir', type=Path, required=True)
        p.add_argument('--key', choices=G.SMALL_FOUR)
        p.add_argument('--compare', action='store_true')
        p.add_argument('--three-rules', action='store_true')
        p.add_argument('--rules', action='store_true')
        p.add_argument('--out', type=Path)
        p.add_argument('--url')
        p.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.command == 'figure':
        if args.compare:
            compare_figure(args.dir, args.out)
        elif args.three_rules:
            three_rules_figure(args.dir, args.out)
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
