"""Time a small masked model's batched fill on one Modal GPU, under each precision setting (#443 follow-up).

#443's batched fills of DiffuGPT medium on a Modal L4 took the same seconds per fill as #239's one mask per forward
pass (1.23 s against 1.26 s on the header-only prompt). This probe answers two questions with the fill code that run
froze (INPLACE_CODE_DIR, default ~/vgc-data/inplace_prompt_grid/run443):

    1. Which precision the fills ran at: the weights' dtype and the TF32 and autocast switches as the fill leaves
       them, then the same fill timed with each switch changed (float16 as run, TF32 on, bfloat16 weights, float32
       weights with TF32 off and on, float32 weights under bfloat16 autocast).
    2. What a larger GPU buys: the same fills on the GPU named in PROBE_GPU.

Every timing fills the masks whose header-only text is nearest the grid's median length, so batches of different
sizes read alike. Each forward is timed between two torch.cuda.synchronize calls; the rest of a fill's time is the
sampling loop. Nothing is judged and no result of #443 is touched: one JSON per GPU goes to PROBE_RESULTS_DIR
(default ~/vgc-data/inplace_prompt_grid/gpu-probe).

Before any Modal launch: state its estimated cost and credit left (`modal billing summary`). One GPU at a time:

    PROBE_GPU=L4 modal run code/scripts/gpu_speed_probe_modal.py --key diffugpt_m
    PROBE_GPU=L40S modal run code/scripts/gpu_speed_probe_modal.py --key illada --full 1,2,4,8,16,32 --no-variants
    PROBE_GPU=L40S modal run code/scripts/gpu_speed_probe_modal.py --key diffugpt_m
    python code/scripts/gpu_speed_probe_modal.py report --key diffugpt_m
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HOME = Path.home() / 'vgc-data/inplace_prompt_grid'
FROZEN = Path(os.environ.get('INPLACE_CODE_DIR', HOME / 'run443'))
RESULTS = Path(os.environ.get('PROBE_RESULTS_DIR', HOME / 'gpu-probe'))
TASKS = FROZEN / 'data/eval2.tasks.json'
GPU = os.environ.get('PROBE_GPU', 'L4')
# Dollars per hour, modal.com/pricing read 2026-11-25 (per-second prices x 3,600).
PRICE_PER_HOUR = {'T4': 0.59, 'L4': 0.80, 'A10': 1.10, 'L40S': 1.95, 'A100-40GB': 2.10, 'A100-80GB': 2.50,
                  'RTX-PRO-6000': 3.03, 'H100': 3.95, 'H200': 4.54, 'B200': 6.25}
KEYS = ('diffugpt_s', 'diffugpt_m', 'qwen25c', 'qwen3', 'illada')
PROMPT = os.environ.get('PROBE_PROMPT', 'inplace_header')   # the prompt whose fills are timed
TIMEOUT_SECONDS = 20 * 60              # 20-minute cap: $0.27 on an L4, $1.32 on an H100
TIMEOUT_BY_KEY = {'illada': 45 * 60}   # 15 GB of weights to load, and every forward pass costs 16 x Masked Qwen3's
HF_CACHE_VOLUME = 'vgc-hf-cache'
REMOTE_CODE = '/root/code'
IMAGE_PACKAGES = ('torch==2.15.0', 'transformers==4.57.3', 'accelerate==1.15.0', 'safetensors==0.8.0',
                  'numpy==2.4.6', 'huggingface_hub==0.36.2')
PACKAGED_FILES = ('scripts/inplace_prompt_grid.py', 'scripts/masked_diffusion_text.py',
                  'scripts/masked_diffusion_memory_bench.py', 'scripts/count_masked_diffusion_text.py',
                  'scripts/masked_diffusion_sampler.py', 'scripts/jev_fill.py', 'scripts/mask_paste.py',
                  'data/regmb-vocabulary.json', 'data/pokedex.json', 'data/inplace-example-team.json')
AS_RUN = 'as the fills ran'
SHORT_STEPS = 8                        # forwards per precision timing; the full fills use the sampler's own steps
# (name, weights dtype or None to leave the loaded weights, TF32 switches, bfloat16 autocast)
VARIANTS = ((AS_RUN, None, False, False),
            ('as run, TF32 on', None, True, False),
            ('as run, bfloat16 autocast', None, False, True),
            ('bfloat16 weights', 'bfloat16', False, False),
            ('float32 weights, TF32 off', 'float32', False, False),
            ('float32 weights, TF32 on', 'float32', True, False),
            ('float32 weights, bfloat16 autocast', 'float32', False, True))


def median_rows(order: list[str], batch: int) -> list[str]:
    """The batch ids centred on the median of order (task ids from the shortest text to the longest)."""
    if not 1 <= batch <= len(order):
        raise ValueError(f'batch {batch} of {len(order)} masks')
    start = (len(order) - batch) // 2
    return order[start:start + batch]


def fitting(tasks: list[dict[str, Any]], build: Any) -> list[dict[str, Any]]:
    """The tasks whose text build can make: a text over the model's token limit is not a fill the grid makes."""
    kept = []
    for task in tasks:
        try:
            build(task)
        except Exception:
            continue
        kept.append(task)
    return kept


def probe_remote(key: str, tasks: list[dict[str, Any]], batches: list[int], full_batches: list[int],
                 variants: bool = True) -> dict[str, Any]:
    """Container side: the precision switches as the fill leaves them, a short fill per (precision, batch), and a
    whole fill per batch in full_batches at the precision the fills ran at. variants False times only the whole fills,
    as iLLaDA needs: its float32 weights alone would take 28 GiB."""
    sys.path.insert(0, f'{REMOTE_CODE}/scripts')
    from dataclasses import replace

    import torch

    import inplace_prompt_grid as G
    import masked_diffusion_memory_bench as B
    from masked_diffusion_sampler import resolve_sampler
    candidate, tokenizer, model, family = G.load_candidate(key, 'cuda')
    active = resolve_sampler(candidate.sampler, None)
    start_id = G.start_token_id(tokenizer, active)
    row_forward = B.make_row_forward(model, family, right_shift=active.right_shift)
    weights: dict[str, int] = {}
    for p in model.parameters():
        weights[str(p.dtype)] = weights.get(str(p.dtype), 0) + p.numel()
    found = {'weights': weights, 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
             'cudnn_tf32': torch.backends.cudnn.allow_tf32,
             'float32_matmul_precision': torch.get_float32_matmul_precision(),
             'autocast_on': torch.is_autocast_enabled('cuda')}
    tasks = fitting(tasks, lambda t: G.canvas(t, G.PROMPTS[PROMPT], tokenizer, candidate.mask_id, start_id))
    order = G.length_order(tasks, G.PROMPTS[PROMPT], tokenizer, candidate.mask_id, start_id)
    by_id = {t['id']: t for t in tasks}
    texts = {i: G.canvas(by_id[i], G.PROMPTS[PROMPT], tokenizer, candidate.mask_id, start_id) for i in order}
    autocast = {'on': False}
    clock = {'seconds': 0.0, 'calls': 0, 'tokens': 0}

    def timed(x: Any, attention: Any, keep: Any) -> Any:
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast['on']):
            out = row_forward(x, attention, keep)
        torch.cuda.synchronize()
        clock['seconds'] += time.perf_counter() - begin
        clock['calls'] += 1
        clock['tokens'] += int(attention.sum())
        return out

    def one_fill(batch: int, steps: int | None) -> dict[str, Any]:
        if batch > len(order):
            return {'batch': batch, 'status': f'only {len(order)} masks'}
        ids = median_rows(order, batch)
        chosen = [texts[i] for i in ids]
        setting = active if steps is None else replace(active, steps=steps)
        rows, slots = [c['ids'] for c in chosen], [c['slots'] for c in chosen]
        seeds = [G.mask_seed(G.SETTING['seed'], i) for i in ids]
        generators = lambda: [torch.Generator(device='cuda').manual_seed(s) for s in seeds]
        torch.cuda.empty_cache()
        try:
            B.fill_rows(timed, rows, slots, candidate.mask_id, replace(active, steps=1), generators(), 'cuda')
            clock.update(seconds=0.0, calls=0, tokens=0)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            B.fill_rows(timed, rows, slots, candidate.mask_id, setting, generators(), 'cuda')
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
        except Exception as error:
            if not B.is_over_ceiling(error):
                return {'batch': batch, 'status': f'{type(error).__name__}: {str(error)[:200]}'}
            return {'batch': batch, 'status': 'over memory'}
        return {'batch': batch, 'status': 'ok', 'forwards': clock['calls'], 'seconds': seconds,
                'forward_seconds': clock['seconds'], 'tokens': clock['tokens'],
                'tokens_per_second': clock['tokens'] / clock['seconds'],
                'seconds_per_fill': seconds / batch, 'forward_share': clock['seconds'] / seconds,
                'mean_tokens': sum(map(len, rows)) / batch, 'mean_masks': sum(map(len, slots)) / batch,
                'peak_gib': torch.cuda.max_memory_allocated() / 2**30}

    result: dict[str, Any] = {'gpu_asked': GPU, 'device': torch.cuda.get_device_name(),
                              'memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
                              'torch': str(torch.__version__), 'candidate': key, 'prompt': PROMPT, 'masks_that_fit': len(order),
                              'sampler_setting': active.to_dict(), 'found': found, 'short_steps': SHORT_STEPS,
                              'full': [], 'variants': []}
    for batch in full_batches:
        result['full'].append(one_fill(batch, None))
    for name, dtype, tf32, cast in (VARIANTS if variants else ()):
        if dtype is not None:
            model.to(getattr(torch, dtype))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32
        autocast['on'] = cast
        result['variants'].append({'name': name, 'weights': str(next(model.parameters()).dtype), 'tf32': tf32,
                                   'autocast': cast, 'fills': [one_fill(b, SHORT_STEPS) for b in batches]})
    return result


def read_results(key: str, directory: Path | None = None) -> list[dict[str, Any]]:
    """Every GPU's result for key, cheapest GPU first."""
    found = [json.loads(p.read_text()) for p in sorted((directory or RESULTS).glob(f'*-{key}.json'))]
    return sorted(found, key=lambda r: PRICE_PER_HOUR.get(r['gpu_asked'], 0.0))


def full_fill(result: dict[str, Any], batch: int | None = None) -> dict[str, Any] | None:
    """The whole fill at batch (the largest that ran when None), or None when none ran."""
    ok = [f for f in result['full'] if f['status'] == 'ok' and batch in (None, f['batch'])]
    return max(ok, key=lambda f: f['batch']) if ok else None


def dollars_per_thousand(result: dict[str, Any], fill: dict[str, Any]) -> float:
    return PRICE_PER_HOUR[result['gpu_asked']] / 3600 * fill['seconds_per_fill'] * 1000


def report(results: list[dict[str, Any]]) -> str:
    """One table of whole fills per GPU and one of precision settings per GPU, as Markdown."""
    lines = ['| GPU | Price per hour | Fills per forward pass | Seconds per fill | Dollars per 1,000 fills | '
             'Share of time in the model | Peak memory |', '|---|---|---|---|---|---|---|']
    for r in results:
        for f in r['full']:
            if f['status'] != 'ok':
                lines.append(f"| {r['device']} | ${PRICE_PER_HOUR[r['gpu_asked']]:.2f} | {f['batch']} | "
                             f"{f['status']} | | | |")
                continue
            lines.append(f"| {r['device']} | ${PRICE_PER_HOUR[r['gpu_asked']]:.2f} | {f['batch']} | "
                         f"{f['seconds_per_fill']:.3f} | ${dollars_per_thousand(r, f):.3f} | "
                         f"{f['forward_share']:.0%} | {f['peak_gib']:.1f} GiB |")
    if not any(r['variants'] for r in results):         # --no-variants: no precision timings to show
        return '\n'.join(lines)
    lines += ['', '| GPU | Precision | Fills per forward pass | Real tokens per second through the model | '
              'Against the fills\' own setting |', '|---|---|---|---|---|']
    for r in results:
        if not r['variants']:
            continue
        base = {f['batch']: f['tokens_per_second'] for f in r['variants'][0]['fills'] if f['status'] == 'ok'}
        for v in r['variants']:
            for f in v['fills']:
                if f['status'] != 'ok':
                    lines.append(f"| {r['device']} | {v['name']} | {f['batch']} | {f['status']} | |")
                    continue
                against = f"{f['tokens_per_second'] / base[f['batch']]:.2f}x" if f['batch'] in base else ''
                lines.append(f"| {r['device']} | {v['name']} | {f['batch']} | {f['tokens_per_second']:,.0f} | "
                             f"{against} |")
    return '\n'.join(lines)


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (modal.Image.debian_slim(python_version='3.11').pip_install(*IMAGE_PACKAGES)
             .env({'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PROBE_GPU': GPU, 'PROBE_PROMPT': PROMPT}))
    for _rel in PACKAGED_FILES:
        image = image.add_local_file(FROZEN / _rel, f'{REMOTE_CODE}/{_rel}')
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
    app = modal.App('gpu-speed-probe')

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_SECONDS, volumes={'/root/.cache/huggingface': hf_cache})
    def probe(key: str, tasks: list, batches: list, full_batches: list, variants: bool = True) -> str:
        # JSON text, not objects: the laptop has no torch, so it can unpickle neither torch's version string nor a
        # torch error
        try:
            return json.dumps(probe_remote(key, tasks, batches, full_batches, variants))
        except Exception:
            import traceback
            return json.dumps({'error': traceback.format_exc()})

    @app.local_entrypoint()
    def main(key: str = 'diffugpt_m', batches: str = '1,32', full: str = '1,142', variants: bool = True) -> None:
        """batches: fills per forward pass for the precision timings; full: for the whole fills; --no-variants
        skips the precision timings."""
        if GPU not in PRICE_PER_HOUR:
            raise SystemExit(f'PROBE_GPU={GPU}: no price recorded; add it to PRICE_PER_HOUR')
        tasks = json.loads(TASKS.read_bytes())['tasks']
        print(f'{key}: probing Modal {GPU} at ${PRICE_PER_HOUR[GPU]:.2f} an hour', flush=True)
        begin = time.time()
        timed = probe.with_options(timeout=TIMEOUT_BY_KEY.get(key, TIMEOUT_SECONDS),
                                   volumes={'/root/.cache/huggingface': hf_cache})
        result = json.loads(timed.remote(key, tasks, [int(b) for b in batches.split(',') if b],
                                         [int(b) for b in full.split(',')], variants))
        if 'error' in result:
            raise SystemExit(f'{key} on {GPU}: THE PROBE FAILED\n{result["error"]}')
        result['wall_minutes'] = (time.time() - begin) / 60
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f'{GPU}-{key}.json').write_text(json.dumps(result, indent=1) + '\n')
        print(json.dumps(result['found']), flush=True)
        print(report([result]), flush=True)
        print(f'{key} on {GPU}: {result["wall_minutes"]:.1f} min', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('report')
    r.add_argument('--key', required=True, choices=KEYS)
    args = parser.parse_args()
    print(report(read_results(args.key)), flush=True)
