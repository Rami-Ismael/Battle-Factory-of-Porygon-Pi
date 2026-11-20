"""Figures for notes/research_notes/masked-text-diffusion-samplers-by-model-2026-11-18.md.

1. sampler-settings.html: each fine-tune model's own sampler, setting by setting, against the shared sampler we run.
2. diffugpt-step-sweep.html: the released DiffuGPT small and medium checkpoints under their authors' sampler at
   16 to 256 steps (UNIFUSION, arXiv 2607.24507v1, Appendix C.3 table, five seeds of 1,024 samples each).

Every value is copied from the sources the note cites; nothing here is measured.

    python code/scripts/sampler_by_model_figures.py [--out DIR]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from echarts_chart import write_chart  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / 'notes' / 'research_notes' / 'figures' / 'samplers-by-model-2026-11-18'

SETTINGS = ['Fill order', 'Masks left after each step', 'Temperature', 'Top-p', 'Block length', 'Tokens per step',
            'Right shift']
# One row per sampler: (name, [(cell text, differs from ours?)]). Ours is the reference row (None).
OURS = ('Ours (every model)', [('most confident first', None), ('exactly linear', None), ('1.0', None),
                                 ('off', None), ('whole text', None), ('about 2.7', None),
                                 ('per model', None)])
ROWS = [
    ('DiffuGPT, evaluation', [('most confident first', False), ('at most linear', True),
                                               ('1.0', False), ('off', False), ('whole text', False),
                                               ('1', True), ('yes', False)]),
    ('DiffuGPT, inference script', [('random', True), ('random', True),
                                              ('0.95', True), ('0.9', True), ('whole text', False),
                                              ('8 or 16', True), ('yes', False)]),
    ('Qwen3, model card', [('most confident first', False), ('exactly linear', False), ('0', True),
                                          ('off', False), ('64', True), ('1', True),
                                          ('no', False)]),
    ('Qwen3, dLLM evaluation', [('most confident first', False), ('exactly linear', False),
                                              ('0', True), ('off', False), ('256', True),
                                              ('1', True), ('no', False)]),
]
SAME, DIFFERS, REFERENCE = '#e6e5e0', '#2f6fdb', '#ffffff'

STEPS = [16, 32, 64, 128, 256]
# UNIFUSION Appendix C.3 table: (a) GPT-2-large perplexity, (b) unigram entropy in nats; mean of five seeds.
PERPLEXITY = {'DiffuGPT small': [494.491, 271.210, 185.117, 144.980, 117.334],
              'DiffuGPT medium': [535.982, 229.408, 147.867, 111.603, 90.645]}
ENTROPY = {'DiffuGPT small': [5.5106, 5.4427, 5.3821, 5.3333, 5.2792],
           'DiffuGPT medium': [5.9229, 5.8319, 5.7685, 5.7099, 5.6472]}
COLOR = {'DiffuGPT small': '#2f6fdb', 'DiffuGPT medium': '#c2571a'}


def settings_option() -> dict:
    # Samplers run across, settings down, so the matrix stays readable at phone width.
    columns = [OURS, *ROWS]
    names = [name for name, _ in columns]
    data = []
    for x, (name, cells) in enumerate(columns):
        for r, (text, differs) in enumerate(cells):
            y = len(SETTINGS) - 1 - r
            fill = REFERENCE if differs is None else DIFFERS if differs else SAME
            ink = '#fefefe' if differs else '#1f1f1e'
            data.append({'value': [x, y, 0 if differs is None else int(differs) + 1], 'text': text,
                         'itemStyle': {'color': fill, 'borderColor': '#6f6e69' if differs is None else fill},
                         'label': {'color': ink}})
    return {
        'tooltip': {'formatter': 'js:p => p.data.text'},
        'grid': {'left': 84, 'right': 4, 'top': 44, 'bottom': 8},
        'xAxis': {'type': 'category', 'data': names, 'position': 'top', 'axisTick': {'show': False},
                  'axisLine': {'show': False}, 'axisLabel': {'interval': 0, 'width': 56, 'fontSize': 10.5,
                                                             'overflow': 'break', 'color': '#1f1f1e',
                                                             'fontWeight': 'bold'}},
        'yAxis': {'type': 'category', 'data': SETTINGS[::-1], 'axisTick': {'show': False}, 'axisLine': {'show': False},
                  'axisLabel': {'color': '#52514e', 'width': 80, 'overflow': 'break', 'fontSize': 11}},
        'series': [{'type': 'heatmap', 'data': data, 'itemStyle': {'borderWidth': 3},
                    'label': {'show': True, 'formatter': 'js:p => p.data.text', 'width': 54, 'overflow': 'break',
                              'fontSize': 10.5},
                    'emphasis': {'disabled': True}}],
    }


def sweep_option() -> dict:
    grids = [{'left': 56, 'right': '54%', 'top': 64, 'bottom': 44}, {'left': '56%', 'right': 16, 'top': 64, 'bottom': 44}]
    axis = {'type': 'log', 'logBase': 2, 'min': 16, 'max': 256, 'name': 'sampler steps', 'nameLocation': 'middle',
            'nameGap': 26}
    series = []
    for name in PERPLEXITY:
        for g, values in enumerate((PERPLEXITY[name], ENTROPY[name])):
            series.append({'type': 'line', 'name': name, 'xAxisIndex': g, 'yAxisIndex': g, 'color': COLOR[name],
                           'symbolSize': 7, 'emphasis': {'focus': 'series'},
                           'data': [[s, v] for s, v in zip(STEPS, values)],
                           })
    return {
        'tooltip': {'trigger': 'axis'},
        'legend': {'top': 0},
        'grid': grids,
        'xAxis': [{**axis, 'gridIndex': 0}, {**axis, 'gridIndex': 1}],
        'yAxis': [{'type': 'value', 'gridIndex': 0, 'name': 'perplexity (lower = more fluent)', 'min': 0},
                  {'type': 'value', 'gridIndex': 1, 'name': 'unigram entropy, nats (higher = more varied)',
                   'scale': True}],
        'series': series,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    write_chart(settings_option(), args.out / 'sampler-settings.html',
                title="Each model's own sampler differs from ours in two to five settings",
                subtitle="Blue cells differ from the shared sampler we run on every masked model (first column). Grey cells match "
                         "it. DiffuGPT rows hold for small and medium; Qwen3 is Masked Qwen3 0.6B.",
                note="Sources: DiffuLLaMA repository at c17e897f (trainer.py generate_samples, model.py, "
                     "inf_diffugpt.py); Qwen3-0.6B-diffusion-mdlm-v0.1 card at c8d24a3f; dllm at ca176752 "
                     "(examples/a2d/mdlm/eval.sh). Ours: code/scripts/masked_diffusion_memory_bench.py defaults "
                     "(128 steps) on the #170 benchmark text, about 350 masks, so about 2.7 tokens per step; the "
                     "right shift follows each model's training. Authors' tokens per step: steps equal masked tokens, "
                     "except the standalone script's 64 steps over 512 or 1,024 tokens. 'At most linear': committed tokens take part in the ranking "
                     "but are never remasked, so fewer masks can remain than the linear count. 'Random': each mask is revealed with probability 1/(t+1).",
                height=470, dark={'#e6e5e0': '#3a3936'})
    write_chart(sweep_option(), args.out / 'diffugpt-step-sweep.html',
                title='More steps make DiffuGPT text more fluent and less varied',
                subtitle="Released DiffuGPT checkpoints, authors' sampler, unconditional text at temperature 1.0",
                note='UNIFUSION (arXiv 2607.24507v1), Appendix C.3 table: mean of five seeds of 1,024 samples, '
                     '512 tokens (small) or 1,024 (medium); perplexity scored by GPT-2 large. Standard deviations '
                     'are at most 7.9 perplexity and 0.007 nats, too small to draw. Free text, not teams.',
                height=380)
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
