"""Redraw today's in-place prompt on the held-out grid on a Modal GPU, then judge the draws on the laptop (#171).

#171's queue redraws every mask that failed its first try up to 4 more times (#169) on the laptop's GPU. This
moves the drawing for one model to a Modal L4 with the same seeds (inplace_prompt_grid.redraw_seed) and the same
sampler setting. Showdown does not run on Modal, so the container cannot tell a legal draw and makes all 4 redraws
of each failing mask, batch at a time (#443): the largest batch that fits the GPU, every draw with its own seed;
--batch N fixes the size, and --batch 1 draws one at a time as #171 did. Each row streams back to
D/inplace_header/<key>/redraws-modal-raw.jsonl as it is drawn, so a stopped run resumes. The code the container runs
is the frozen copy in INPLACE_CODE_DIR (default D/run443, the first with batched fills; #171 ran from D/run171).

merge then judges every raw row on the laptop with the frozen judge (inplace_prompt_grid.judge: parses, keeps the
visible values, Reg M-B names, Showdown) and appends to redraws.jsonl only the draws #169's loop would have made:
up to and including each mask's first legal draw. The queue's own redraw step then finds nothing left to draw.

Before any Modal launch: state its estimated cost and credit left (`modal billing summary`).

    modal run code/scripts/inplace_prompt_redraw_modal.py --key qwen25c [--limit 3]
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/inplace_prompt_redraw_modal.py merge --key qwen25c
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

DIRECTORY = Path.home() / 'vgc-data/inplace_prompt_grid'
FROZEN = Path(os.environ.get('INPLACE_CODE_DIR', DIRECTORY / 'run443'))      # #171 ran from run171
TASKS = Path.home() / ('Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of '
                       'possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline/'
                       'runs/heldout-starters/'
                       'eval2.tasks.json')
PROMPT = 'inplace_header'
REDRAWS = 4
RAW = 'redraws-modal-raw.jsonl'
REMOTE_CODE = '/root/code'
GPU = 'L4'
TIMEOUT_SECONDS = 3 * 60 * 60          # 3-hour cap, about $2.40 at L4's $0.80 an hour
HF_CACHE_VOLUME = 'vgc-hf-cache'
DEFERRED = 'deferred: Showdown judges this draw on the laptop'
IMAGE_PACKAGES = ('torch==2.15.0', 'transformers==4.57.3', 'accelerate==1.15.0', 'safetensors==0.8.0',
                  'numpy==2.4.6', 'huggingface_hub==0.36.2')
PACKAGED_FILES = ('scripts/inplace_prompt_grid.py', 'scripts/masked_diffusion_text.py',
                  'scripts/masked_diffusion_memory_bench.py', 'scripts/count_masked_diffusion_text.py',
                  'scripts/masked_diffusion_sampler.py', 'scripts/jev_fill.py', 'scripts/mask_paste.py',
                  'data/regmb-vocabulary.json', 'data/pokedex.json', 'data/inplace-example-team.json')
VERDICT_KEYS = frozenset({'json_parses', 'reads_as_team', 'visible_kept', 'regmb_names', 'showdown_errors',
                          'showdown_accepts', 'legal', 'failed', 'reasons', 'not_regmb', 'recovered',
                          'copied_visible', 'paste', 'rules'})


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def pending(tasks: Iterable[dict[str, Any]], first: list[dict[str, Any]], done: list[dict[str, Any]],
            redraws: int = REDRAWS) -> list[tuple[dict[str, Any], int]]:
    """(task, draw) still to draw: every redraw of each mask whose first try failed, minus the draws already in done,
    and none for a mask that done already shows legal."""
    legal = {r['id']: r['legal'] for r in first}
    drawn = {(r['id'], r['draw']) for r in done}
    stopped = {r['id'] for r in done if r.get('legal')}
    return [(task, draw) for task in tasks if not legal[task['id']] and task['id'] not in stopped
            for draw in range(1, redraws + 1) if (task['id'], draw) not in drawn]


def loop_keeps(judged: Iterable[dict[str, Any]], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The judged draws #169's loop would have made: each mask's draws in order, up to and including its first legal
    draw (counting the draws already in existing), and none already in existing."""
    first_legal = {}
    for r in existing:
        if r['legal']:
            first_legal[r['id']] = min(r['draw'], first_legal.get(r['id'], r['draw']))
    have = {(r['id'], r['draw']) for r in existing}
    kept = []
    for r in sorted(judged, key=lambda r: (r['id'], r['draw'])):
        if (r['id'], r['draw']) in have or r['draw'] > first_legal.get(r['id'], REDRAWS + 1):
            continue
        kept.append(r)
        if r['legal']:
            first_legal[r['id']] = r['draw']
    return kept


def draw_remote(key: str, tasks: list[dict[str, Any]], first: list[dict[str, Any]], done: list[dict[str, Any]],
                limit: int | None, batch: int | None = None, gpu: str = 'cuda') -> Iterator[str]:
    """Container side: a determinism probe line, then one raw row per (mask, draw) as a JSON line, batch at a time
    (#443): the largest batch that fits the GPU at the longest text, or batch if given. Each draw keeps its own seed
    (inplace_prompt_grid.redraw_seed)."""
    sys.path.insert(0, f'{REMOTE_CODE}/scripts')
    import torch

    import inplace_prompt_grid as G
    import masked_diffusion_memory_bench as B
    candidate, tokenizer, model, family = G.load_candidate(key, gpu)
    active = candidate.sampler
    start_id = G.start_token_id(tokenizer, active)
    row_forward = B.make_row_forward(model, family, right_shift=active.right_shift)
    names, prompt = G.vocabulary(), G.PROMPTS[PROMPT]
    todo = pending(tasks[:limit], first, done)
    device = f'cuda {torch.cuda.get_device_name()}' if gpu == 'cuda' else gpu
    memory = torch.cuda.max_memory_allocated if gpu == 'cuda' else (lambda: 0)
    deferred = lambda _paste: DEFERRED
    if not todo:
        return
    texts = {t['id']: G.canvas(t, prompt, tokenizer, candidate.mask_id, start_id) for t, _ in todo}
    longest, most_masks, _ = G.extreme_texts(texts.values())
    sizing = (G.choose_batch(row_forward, longest, candidate.mask_id, active, gpu, most_masks=most_masks,
                             cap=len(todo))
              if batch is None else {'batch': batch, 'ladder': [], 'chosen_by': 'given'})
    task, draw = todo[0]
    seed = G.redraw_seed(G.SETTING['seed'], task['id'], draw)
    first_draw, second_draw = G.repeat_draw(task, seed, prompt, tokenizer, row_forward, candidate.mask_id, deferred,
                                            names, gpu, start_id, active, sizing['batch'])
    yield json.dumps({'probe': {'device': device, 'torch': torch.__version__, 'mask': task['id'],
                                'repeat_same_tokens': G.token_share(first_draw['slot_ids'], second_draw['slot_ids']),
                                'pending': len(todo), 'batching': sizing}})
    rank = {i: n for n, i in enumerate(G.length_order([t for t, _ in todo], prompt, tokenizer, candidate.mask_id,
                                                      start_id))}
    ordered = sorted(todo, key=lambda pair: (pair[1], rank[pair[0]['id']]))
    for chunk in G.batches(ordered, sizing['batch']):
        draws = [(t, G.redraw_seed(G.SETTING['seed'], t['id'], d)) for t, d in chunk]
        rows = G.fill_batch(draws, prompt, tokenizer, row_forward, candidate.mask_id, deferred, names, gpu,
                            start_id, memory=memory, sampler=active,
                            batch_size=sizing['batch'])
        for row, (_task, d) in zip(rows, chunk, strict=True):
            yield json.dumps(row | {'draw': d, 'device': device}, ensure_ascii=False)


def resume_batch(rows: Iterable[dict[str, Any]], batch: int | None) -> int | None:
    """The batch size a launch must use: the one its raw rows were drawn at (rows from before #443 count as 1), the
    given size, or None (find it) when nothing is drawn yet. Refuses a different given size and mixed sizes."""
    sizes = {r.get('batch_size', 1) for r in rows}
    if len(sizes) > 1:
        raise SystemExit(f'raw rows drawn at batch sizes {sorted(sizes)}; one run keeps one size')
    if sizes and batch is not None and batch not in sizes:
        raise SystemExit(f'raw rows were drawn at batch {sizes.pop()}, not {batch}; resume at that size')
    return sizes.pop() if sizes else batch


def resume_batching(raw: list[dict[str, Any]], probe: dict[str, Any]) -> dict[str, Any]:
    """The redraws' batching for run.json: the size every raw row was drawn at, with the probe's ladder when the
    probe chose that size."""
    size = resume_batch(raw, None)
    recorded = probe.get('batching') or {}
    return recorded if recorded.get('batch') == size else {'batch': size, 'ladder': [], 'chosen_by': 'rows'}


def merge(key: str, directory: Path = DIRECTORY) -> dict[str, int]:
    """Judge the raw Modal rows on the laptop and append the ones #169's loop keeps to redraws.jsonl."""
    sys.path[:0] = [str(FROZEN / 'scripts'), str(FROZEN / 'src')]
    import count_masked_diffusion_text as C
    import inplace_prompt_grid as G
    import masked_diffusion_memory_bench as B
    from mask_paste import team_paste
    from showdown_validator import Validator
    out = directory / PROMPT / key
    raw, existing = read_rows(out / RAW), read_rows(out / 'redraws.jsonl')
    candidate = next(c for c in C.CANDIDATES if c.key == key)
    tokenizer = B.load_tokenizer(B.candidate_info(candidate))
    start_id = tokenizer.bos_token_id if candidate.start_token else None
    tasks = {t['id']: t for t in G.grid_tasks()}
    names, prompt, validate = G.vocabulary(), G.PROMPTS[PROMPT], Validator()
    judged = []
    try:
        for row in raw:
            task = tasks[row['id']]
            c = G.canvas(task, prompt, tokenizer, candidate.mask_id, start_id)
            verdict = G.judge(task, c['built'], row['values'], validate, names)
            team = verdict.pop('team', None)
            verdict |= G.copies(task, c['built'], row['values'])
            new = {k: v for k, v in row.items() if k not in VERDICT_KEYS} | verdict
            if new['legal']:
                new['paste'] = team_paste(team)
            judged.append(new)
    finally:
        validate.close()
    kept = loop_keeps(judged, existing)
    with (out / 'redraws.jsonl').open('a') as handle:
        for r in kept:
            handle.write(json.dumps(r, ensure_ascii=False) + '\n')
    meta_path = out / 'run.json'
    if meta_path.exists() and raw:
        probe_path = out / 'modal-probe.json'
        probe = json.loads(probe_path.read_text()).get('probe', {}) if probe_path.exists() else {}
        sizing = resume_batching(raw, probe)
        meta_path.write_text(json.dumps(G.record_batching(json.loads(meta_path.read_text()), sizing, redraw=True),
                                        indent=1) + '\n')
    final = G.final_draws(out / 'fills.jsonl', out / 'redraws.jsonl')
    return {'raw': len(raw), 'appended': len(kept), 'legal_raw': sum(r['legal'] for r in judged),
            'legal_after_redraws': sum(r['legal'] for r in final.values()), 'masks': len(final)}


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (modal.Image.debian_slim(python_version='3.11').pip_install(*IMAGE_PACKAGES)
             .env({'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}))
    for _rel in PACKAGED_FILES:
        image = image.add_local_file(FROZEN / _rel, f'{REMOTE_CODE}/{_rel}')
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
    app = modal.App('inplace-prompt-redraws')

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_SECONDS, volumes={'/root/.cache/huggingface': hf_cache})
    def draw(key: str, tasks: list, first: list, done: list, limit: int | None,
             batch: int | None = None) -> Iterator[str]:
        try:
            yield from draw_remote(key, tasks, first, done, limit, batch)
        finally:
            hf_cache.commit()

    @app.local_entrypoint()
    def main(key: str, limit: int = 0, batch: int = 0) -> None:
        """batch 0 (the default) takes the largest batch that fits the GPU; any other value fixes it."""
        out = DIRECTORY / PROMPT / key
        tasks = json.loads(TASKS.read_text())['tasks']
        first = read_rows(out / 'fills.jsonl')
        done = read_rows(out / 'redraws.jsonl') + read_rows(out / RAW)
        left = pending(tasks[:limit or None], first, done)
        size = resume_batch(read_rows(out / RAW), batch or None)
        print(f'{key}: {len(left)} draws to make on Modal {GPU}', flush=True)
        if not left:
            return
        begin, made = time.time(), 0
        with (out / RAW).open('a') as handle:
            for line in draw.remote_gen(key, tasks, first, done, limit or None, size):
                if line.startswith('{"probe"'):
                    print(line, flush=True)
                    (out / 'modal-probe.json').write_text(line + '\n')
                    continue
                handle.write(line + '\n')
                handle.flush()
                made += 1
                if made % 50 == 0:
                    print(f'{made}/{len(left)} draws, {(time.time() - begin) / 60:.1f} min', flush=True)
        print(f'{key}: {made} draws in {(time.time() - begin) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=['merge'])
    parser.add_argument('--key', required=True, choices=['qwen25c', 'qwen3'])
    args = parser.parse_args()
    print(json.dumps(merge(args.key)), flush=True)
