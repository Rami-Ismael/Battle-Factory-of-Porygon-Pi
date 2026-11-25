"""Fill the held-out grid's 270 masks first try on a Modal GPU, under several prompts in one container, then judge
the fills on the laptop (#238, #239).

#238 measures the instructions prompt against today's prompt on the same masks and mask seeds. Owner, 2026-11-25:
run on Modal, not the laptop GPU. A CUDA generator draws different noise from an MPS one for the same seed, so the
container fills both prompts, today's (inplace_header) and the one measured: the pair then differs in the prompt
alone. #237's laptop fills stay the reference row; the report also counts how many of today's-prompt fills match
them token for token.

Each mask is drawn exactly as inplace_prompt_grid.fill_grid draws it (fill_one, mask_seed, the candidate's sampler
setting). Showdown does not run on Modal, so every draw streams back as a raw row to
D/<prompt>/<key>/fills-modal-raw.jsonl, and a stopped run resumes. merge judges each raw row on the laptop with
inplace_prompt_grid.judge (parses, keeps the visible values, Reg M-B names, Showdown) into D/<prompt>/<key>/fills.jsonl.

#239's example prompts fill only the masks whose text fits under 1,024 tokens for all four models. Those lists are
computed on the laptop (inplace_prompt_grid.fitting_ids) and frozen in FROZEN/data/fitting-ids.json; the entrypoint
fills each prompt's list and merge refuses a raw file that is not exactly that list.

D is INPLACE_RESULTS_DIR (default ~/vgc-data/inplace_prompt_grid/modal-l4; #239 uses modal-l4-239), apart from #237's
laptop fills. The code the container runs is the
frozen copy in INPLACE_CODE_DIR (default ~/vgc-data/inplace_prompt_grid/run239; #238 ran from run238).

Before any Modal launch: state its estimated cost and credit left (`modal billing summary`).

    modal run code/scripts/inplace_prompt_fill_modal.py --key qwen3 --prompts inplace_header,inplace_instructions [--limit 3]
    modal run code/scripts/inplace_prompt_fill_modal.py --key qwen3 \
        --prompts inplace_header,inplace_instructions,inplace_header_example,inplace_example
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/inplace_prompt_fill_modal.py merge --key qwen3 --prompt inplace_instructions
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

HOME = Path.home() / 'vgc-data/inplace_prompt_grid'
DIRECTORY = Path(os.environ.get('INPLACE_RESULTS_DIR', HOME / 'modal-l4'))     # #239: modal-l4-239, so both
# prompts without the example are refilled in its containers rather than read from #238's rows
FROZEN = Path(os.environ.get('INPLACE_CODE_DIR', HOME / 'run239'))
TASKS = FROZEN / 'data/eval2.tasks.json'     # the held-out grid, copied out of the iCloud vault into the frozen run
FITTING = FROZEN / 'data/fitting-ids.json'   # {prompt: [mask ids]} for the prompts that do not fit every mask (#239)
TASKS_SHA256 = '596b3adc19abea7df6e82d61775b9efdf35dd54b74c77623d49165b0a31e754b'      # #237's grid file
KEYS = ('diffugpt_s', 'diffugpt_m', 'qwen25c', 'qwen3')
RAW = 'fills-modal-raw.jsonl'
REMOTE_CODE = '/root/code'
GPU = 'L4'
TIMEOUT_SECONDS = 2 * 60 * 60          # 2-hour cap per model, about $1.60 at L4's $0.80 an hour
HF_CACHE_VOLUME = 'vgc-hf-cache'
DEFERRED = 'deferred: Showdown judges this fill on the laptop'
IMAGE_PACKAGES = ('torch==2.15.0', 'transformers==4.57.3', 'accelerate==1.15.0', 'safetensors==0.8.0',
                  'numpy==2.4.6', 'huggingface_hub==0.36.2')
PACKAGED_FILES = ('scripts/inplace_prompt_grid.py', 'scripts/masked_diffusion_text.py',
                  'scripts/masked_diffusion_memory_bench.py', 'scripts/count_masked_diffusion_text.py',
                  'scripts/masked_diffusion_sampler.py', 'scripts/jev_fill.py', 'scripts/mask_paste.py',
                  'data/regmb-vocabulary.json', 'data/pokedex.json', 'data/inplace-example-team.json')
VERDICT_KEYS = frozenset({'json_parses', 'reads_as_team', 'visible_kept', 'regmb_names', 'showdown_errors',
                          'showdown_accepts', 'legal', 'failed', 'reasons', 'not_regmb', 'recovered',
                          'copied_visible', 'paste', 'rules'})


def grid_tasks() -> list[dict[str, Any]]:
    """The 270 held-out masks from the frozen copy, refused unless it is #237's grid file byte for byte."""
    import hashlib
    data = TASKS.read_bytes()
    if hashlib.sha256(data).hexdigest() != TASKS_SHA256:
        raise ValueError(f'{TASKS} is not #237\'s grid file (an empty iCloud placeholder reads as e3b0c442...)')
    tasks = json.loads(data)['tasks']
    if len(tasks) != 270:
        raise ValueError(f'{TASKS}: {len(tasks)} masks, not 270')
    return tasks


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def fitting(path: Path | None = None) -> dict[str, set[str]]:
    """The frozen mask list of each prompt that does not fit every mask; a prompt not in it fills all 270."""
    path = path or FITTING
    return {p: set(ids) for p, ids in json.loads(path.read_text()).items()} if path.exists() else {}


def pending(prompts: Iterable[str], tasks: list[dict[str, Any]], done: dict[str, set[str]],
            only: dict[str, set[str]] | None = None) -> list[tuple[str, dict]]:
    """(prompt, task) still to fill, prompt by prompt in the order given, minus the masks already drawn and, for a
    prompt in only, the masks outside its list."""
    only = only or {}
    return [(p, t) for p in prompts for t in tasks
            if t['id'] not in done.get(p, set()) and (p not in only or t['id'] in only[p])]


def fill_remote(key: str, prompts: list[str], tasks: list[dict[str, Any]], done: dict[str, list[str]],
                limit: int | None, only: dict[str, list[str]] | None = None) -> Iterator[str]:
    """Container side: one probe line, then one raw row per (prompt, mask) as a JSON line."""
    sys.path.insert(0, f'{REMOTE_CODE}/scripts')
    import torch

    import inplace_prompt_grid as G
    import masked_diffusion_memory_bench as B
    from masked_diffusion_sampler import resolve_sampler
    fill = B.fill

    def synced(*args: Any, **kwargs: Any) -> Any:          # time the draw, not just its launch
        result = fill(*args, **kwargs)
        torch.cuda.synchronize()
        return result
    B.fill = synced
    candidate, tokenizer, model, family = G.load_candidate(key, 'cuda')
    active = resolve_sampler(candidate.sampler, None)
    start_id = G.start_token_id(tokenizer, active)
    forward_for = lambda s: B.make_forward(model, family, G.SETTING['scope'], s, right_shift=active.right_shift)
    names = G.vocabulary()
    todo = pending(prompts, tasks[:limit], {p: set(ids) for p, ids in done.items()},
                   {p: set(ids) for p, ids in (only or {}).items()})
    device = f'cuda {torch.cuda.get_device_name()}'

    def one(prompt: str, task: dict[str, Any]) -> dict[str, Any]:
        row = G.fill_one(task, G.PROMPTS[prompt], tokenizer, forward_for, candidate.mask_id, lambda _paste: DEFERRED,
                         names, G.mask_seed(G.SETTING['seed'], task['id']), 'cuda', start_id,
                         memory=torch.cuda.max_memory_allocated, sampler=active)
        return row | {'device': device}

    probe: dict[str, Any] = {'device': device, 'torch': torch.__version__, 'candidate': key, 'pending': len(todo),
                             'sampler_setting': active.to_dict()}
    if todo:
        prompt, task = todo[0]
        c = G.canvas(task, G.PROMPTS[prompt], tokenizer, candidate.mask_id, start_id)
        probe['sees_later_tokens'] = B.sees_later_tokens(model, family, torch.tensor([c['ids']], device='cuda'),
                                                         torch.tensor(c['slots'], device='cuda'),
                                                         right_shift=active.right_shift)
        a, b = one(prompt, task), one(prompt, task)
        same = [x == y for sa, sb in zip(a['slot_ids'], b['slot_ids'], strict=True) for x, y in zip(sa, sb)]
        probe |= {'mask': task['id'], 'prompt': prompt, 'repeat_same_tokens': sum(same) / len(same)}
        probe['pass'] = bool(probe['sees_later_tokens']) and probe['repeat_same_tokens'] >= 0.9
    yield json.dumps({'probe': probe})
    for prompt, task in todo:
        yield json.dumps(one(prompt, task), ensure_ascii=False)


def run_meta(prompt: str, key: str, probe: dict[str, Any]) -> dict[str, Any]:
    """run.json for one prompt and model, in the shape inplace_prompt_grid.run writes, with the Modal device."""
    import hashlib
    sys.path[:0] = [str(FROZEN / 'scripts'), str(FROZEN / 'src')]
    import count_masked_diffusion_text as C
    import inplace_prompt_grid as G
    candidate = C.candidate_by_key(key)
    tasks = grid_tasks()
    meta = G.run_meta(G.PROMPTS[prompt], candidate, tasks, tasks_sha256=hashlib.sha256(TASKS.read_bytes()).hexdigest())
    meta['setting']['device'] = probe['device']
    meta['environment'] |= {'torch': probe['torch'], 'device': probe['device'], 'judged_on': 'laptop'}
    return meta


def merge(key: str, prompt: str, directory: Path = DIRECTORY) -> dict[str, Any]:
    """Judge the raw Modal rows on the laptop into fills.jsonl, in grid order; refuses a raw file that is not
    exactly one row per mask."""
    sys.path[:0] = [str(FROZEN / 'scripts'), str(FROZEN / 'src')]
    import count_masked_diffusion_text as C
    import inplace_prompt_grid as G
    import masked_diffusion_memory_bench as B
    from mask_paste import team_paste
    from showdown_validator import Validator
    out = directory / prompt / key
    raw = read_rows(out / RAW)
    tasks = grid_tasks()
    only = fitting().get(prompt)
    if only is not None:
        if only != set(G.fitting_ids(prompt) or ()):
            raise ValueError(f'{FITTING}: the frozen list for {prompt} is not the one fitting_ids gives')
        tasks = [t for t in tasks if t['id'] in only]
    ids = [r['id'] for r in raw]
    if sorted(ids) != sorted(t['id'] for t in tasks):
        raise ValueError(f'{out / RAW}: {len(ids)} rows, {len(set(ids))} distinct masks, not the {len(tasks)} masks')
    others = sorted({str(r.get('prompt')) for r in raw} - {prompt})
    if others:
        raise ValueError(f'{out / RAW}: rows filled with {others}, not {prompt}')
    candidate = C.candidate_by_key(key)
    tokenizer = B.load_tokenizer(B.candidate_info(candidate))
    start_id = G.start_token_id(tokenizer, candidate.sampler)
    by_id = {r['id']: r for r in raw}
    names, validate = G.vocabulary(), Validator()
    judged = []
    try:
        for task in tasks:
            row = by_id[task['id']]
            c = G.canvas(task, G.PROMPTS[prompt], tokenizer, candidate.mask_id, start_id)
            verdict = G.judge(task, c['built'], row['values'], validate, names)
            team = verdict.pop('team', None)
            verdict |= G.copies(task, c['built'], row['values'])
            new = {k: v for k, v in row.items() if k not in VERDICT_KEYS} | verdict
            if new['legal']:
                new['paste'] = team_paste(team)
            judged.append(new)
    finally:
        validate.close()
    (out / 'fills.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in judged))
    probe = json.loads((directory / 'preflight' / f'modal-gpu-{key}.json').read_text())
    G.gpu_preflight_path(directory, key, prompt).write_text(json.dumps(probe, indent=1) + '\n')
    (out / 'run.json').write_text(json.dumps(run_meta(prompt, key, probe), indent=1) + '\n')
    return {'key': key, 'prompt': prompt, 'masks': len(judged), 'legal': sum(r['legal'] for r in judged)}


def same_as_laptop(key: str, directory: Path = DIRECTORY, laptop: Path = HOME) -> dict[str, int]:
    """How many of today's-prompt fills on Modal match #237's laptop fill of the same mask token for token, and
    each run's legal count: the device's effect with the prompt held fixed."""
    modal_rows = {r['id']: r for r in read_rows(directory / 'inplace_header' / key / 'fills.jsonl')}
    laptop_rows = {r['id']: r for r in read_rows(laptop / 'inplace_header' / key / 'fills.jsonl')}
    shared = set(modal_rows) & set(laptop_rows)
    return {'masks': len(shared), 'same_tokens': sum(modal_rows[i]['slot_ids'] == laptop_rows[i]['slot_ids']
                                                      for i in shared),
            'legal_modal': sum(modal_rows[i]['legal'] for i in shared),
            'legal_laptop': sum(laptop_rows[i]['legal'] for i in shared)}


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
    app = modal.App('inplace-prompt-fills')

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_SECONDS, volumes={'/root/.cache/huggingface': hf_cache})
    def fill(key: str, prompts: list, tasks: list, done: dict, limit: int | None,
             only: dict | None = None) -> Iterator[str]:
        try:
            yield from fill_remote(key, prompts, tasks, done, limit, only)
        finally:
            hf_cache.commit()

    @app.local_entrypoint()
    def main(key: str, prompts: str = 'inplace_header,inplace_instructions', limit: int = 0) -> None:
        order = prompts.split(',')
        tasks = grid_tasks()
        done = {p: [r['id'] for r in read_rows(DIRECTORY / p / key / RAW)] for p in order}
        only = {p: ids for p, ids in fitting().items() if p in order}
        missing = [p for p in order if p.endswith('_example') and p not in only]
        if missing:
            raise SystemExit(f'{FITTING}: no frozen mask list for {missing}')
        left = pending(order, tasks[:limit or None], {p: set(ids) for p, ids in done.items()}, only)
        print(f'{key}: {len(left)} fills to make on Modal {GPU}', flush=True)
        if not left:
            return
        handles = {}
        for p in order:
            (DIRECTORY / p / key).mkdir(parents=True, exist_ok=True)
            handles[p] = (DIRECTORY / p / key / RAW).open('a')
        begin, made = time.time(), 0
        try:
            for line in fill.remote_gen(key, order, tasks, done, limit or None, {p: sorted(i) for p, i in only.items()}):
                row = json.loads(line)
                if 'probe' in row:
                    print(line, flush=True)
                    path = DIRECTORY / 'preflight' / f'modal-gpu-{key}.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(row['probe'], indent=1) + '\n')
                    if row['probe'].get('pass') is False:
                        raise SystemExit(f'{key}: A MODEL PROBE FAILED, {row["probe"]}')
                    continue
                handles[row['prompt']].write(line + '\n')
                handles[row['prompt']].flush()
                made += 1
                if made % 50 == 0:
                    print(f'{made}/{len(left)} fills, {(time.time() - begin) / 60:.1f} min', flush=True)
        finally:
            for h in handles.values():
                h.close()
        print(f'{key}: {made} fills in {(time.time() - begin) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('merge')
    m.add_argument('--key', required=True, choices=KEYS)
    m.add_argument('--prompt', required=True)
    s = sub.add_parser('same-as-laptop')
    s.add_argument('--key', required=True, choices=KEYS)
    args = parser.parse_args()
    print(json.dumps(merge(args.key, args.prompt) if args.command == 'merge' else same_as_laptop(args.key)), flush=True)
