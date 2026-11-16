"""Rule out the sampler as the cause of the masked models' bad text (#236).

The memory benchmark (#170, masked_diffusion_memory_bench.py) fills text with its own float32 copy of the step of
dLLM's MDLMSampler.infill, and shifts the logits right by one for DiffuGPT, DiffuLLaMA and Dream. Three checks:

1. Same sampler. One masked Qwen text filled by dLLM's own MDLMSampler.infill (dllm at ca176752, installed in its
   own environment) and by ours, both on the CPU in float32 at temperature 0 with the same step count. Both are
   deterministic there, so the fills match token for token unless two tokens tie. Every canvas each sampler saw
   is kept, so a difference is traced to the step where the two part. dLLM's own step schedule is also compared
   with ours over many mask counts and step counts.
2. Right shift. The text holds the same three Pokémon twice, once visible and once with one value hidden; the
   hidden value's answer is the value above it. Each value of the second copy is filled alone at temperature 0,
   one token per step, with and without the right shift. The setting a model was trained with copies far more
   often. The masked Qwen models, which are not shifted, are the control.
3. Values stay inside their quotes. The share of filled hidden values with no '"', ':' or ',' in them, per model,
   over every text the memory benchmark saved. A share near 0 for every model would point at the fixed-length
   slots rather than at the models.

    dllm --canvas F --out O [--steps N] [--block-size N]   check 1, dLLM's side: run with the dllm environment's python
    schedule --out O                                        check 1, dLLM's schedule against ours (dllm environment)
    ours --canvas F --out O --scope all|slots [--steps N]   check 1, our side (venv-dlm)
    compare A B                                             check 1, differing tokens and the first step they part
    copy --key K --out O [--device mps]                     check 2 for one candidate (venv-dlm, laptop GPU)
    quotes --dir D                                          check 3 as a Markdown table, from the benchmark's runs
"""
import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

import masked_diffusion_memory_bench as B
import masked_diffusion_text as D
from mask_paste import json_pieces, value_text

DLLM_COMMIT = 'ca176752fbceec49c6b4777a2c18ae88e4eb10ed'
LEAKS = re.compile('[":,]')
FIELDS = tuple('move' if p[0][1] == 'moves' else p[0][1] for p in json_pieces([D.PROBE]) if not isinstance(p, str))
COPY_LABELS = ('Team: ', '\nSame team: ')
COPY_POKEMON = 3                # three Pokémon per copy keep two copies under GPT-2's 1,024 positions


# ---------- check 3: reading the hidden values back ----------

def slot_runs(slots: Sequence[int]) -> list[list[int]]:
    """Slot positions grouped into runs of consecutive positions: one run per hidden value."""
    runs: list[list[int]] = []
    for n in slots:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


def values_from_ids(slots: Sequence[int], filled_ids: Sequence[int], decode: Callable[[list[int]], str]) -> list[str]:
    """Each hidden value as its own tokens decode, when the filled token ids were saved."""
    return [decode([filled_ids[n] for n in run]) for run in slot_runs(slots)]


def values_in_text(canvas_text: str, mask_token: str, filled: str) -> list[str] | None:
    """Each hidden value of a filled text, found between the fixed text around it; None when the fixed text did
    not survive. A value that wrote the text that follows it is cut at the first place the rest still matches."""
    pieces = re.split(f'(?:{re.escape(mask_token)})+', canvas_text)
    match = re.fullmatch('(.*?)'.join(map(re.escape, pieces)), filled, re.S)
    return list(match.groups()) if match else None


def stays_inside(value: str) -> bool:
    """True when a filled value holds no quote, colon or comma, so it cannot have written JSON structure."""
    return not LEAKS.search(value)


def benchmark_values(canvas: dict[str, Any], run: dict[str, Any], decode: Callable[[list[int]], str],
                     mask_token: str) -> tuple[list[list[str]], int]:
    """Every filled text's hidden values in one benchmark result, and how many texts could not be read."""
    if 'filled_ids' in run:
        return [values_from_ids(canvas['slots'], ids, decode) for ids in run['filled_ids']], 0
    read = [values_in_text(canvas['text'], mask_token, text) for text in run.get('filled', [])]
    return [v for v in read if v is not None], sum(v is None for v in read)


def quote_rows(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for key in B.ORDER:
        path = directory / 'canvas' / f'{key}.json'
        runs = sorted((directory / 'runs' / key).glob('*.json'))
        if not path.exists() or not runs:
            continue
        canvas = json.loads(path.read_text())
        tokenizer = B.load_tokenizer(canvas['candidate'])
        texts, unreadable = [], 0
        for p in runs:
            values, bad = benchmark_values(canvas, json.loads(p.read_text()), tokenizer.decode, tokenizer.mask_token)
            texts += values
            unreadable += bad
        flat = [(FIELDS[i % len(FIELDS)], v) for values in texts for i, v in enumerate(values)]
        species = [v for f, v in flat if f == 'species']
        rows.append({'key': key, 'label': canvas['candidate']['label'], 'texts': len(texts), 'unreadable': unreadable,
                     'values': len(flat), 'inside': sum(stays_inside(v) for _, v in flat),
                     'species': len(species), 'species_inside': sum(stays_inside(v) for v in species),
                     'by_field': {f: [sum(stays_inside(v) for g, v in flat if g == f), sum(g == f for g, _ in flat)]
                                  for f in dict.fromkeys(FIELDS)}})
    return rows


def quote_table(rows: list[dict[str, Any]]) -> str:
    lines = ['| Model | Texts | Hidden values | Inside their quotes | Species values inside |', '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['label']} | {r['texts']} | {r['values']:,} | {r['inside'] / r['values']:.1%} | "
                     f"{r['species_inside'] / r['species']:.1%} |")
    return '\n'.join(lines)


# ---------- check 1: the same text through dLLM's sampler and ours ----------

def recording(forward: B.Forward) -> tuple[B.Forward, list[torch.Tensor]]:
    """forward that keeps a copy of every canvas it is given: the state before each sampling step."""
    states: list[torch.Tensor] = []

    def recorded(x: torch.Tensor) -> torch.Tensor:
        states.append(x.detach().cpu().clone())
        return forward(x)
    return recorded, states


def differences(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Two samplers' canvases, initial state first and final last: the tokens their final canvases differ in, and
    the first step after which they differ (None when every state matches)."""
    first = next((n for n, (x, y) in enumerate(zip(a, b)) if list(x) != list(y)), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return {'differing_tokens': sum(x != y for x, y in zip(a[-1], b[-1])), 'first_step': first,
            'steps': max(len(a), len(b)) - 1}


def dllm_fill(canvas: dict[str, Any], steps: int, block_size: int | None) -> dict[str, Any]:
    """dLLM's own MDLMSampler.infill on the canvas, CPU float32, temperature 0, every canvas kept. Needs dllm."""
    from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    info = canvas['candidate']
    tokenizer = AutoTokenizer.from_pretrained(info['repo'], revision=info['revision'])
    assert tokenizer.mask_token_id == info['mask_id'], (tokenizer.mask_token, tokenizer.mask_token_id)
    model = AutoModelForMaskedLM.from_pretrained(info['repo'], revision=info['revision'], trust_remote_code=True,
                                                 dtype=torch.float32).eval()
    config = MDLMSamplerConfig(steps=steps, temperature=0.0, block_size=block_size, return_dict=True)
    start = time.perf_counter()
    out = MDLMSampler(model=model, tokenizer=tokenizer).infill([canvas['ids']], config)
    return _fill_result('dLLM MDLMSampler.infill', canvas, steps, [h[0].tolist() for h in out.histories],
                        tokenizer.decode, time.perf_counter() - start, block_size=block_size, dllm=DLLM_COMMIT)


def ours_fill(canvas: dict[str, Any], scope: str, steps: int) -> dict[str, Any]:
    """The benchmark's fill on the canvas, CPU float32, temperature 0, every canvas kept."""
    info, family = canvas['candidate'], canvas['family']
    tokenizer = B.load_tokenizer(info)
    model = B.load_model(info, family, 'native', 'cpu').float()
    x, slots = torch.tensor([canvas['ids']]), torch.tensor(canvas['slots'])
    forward, states = recording(B.make_forward(model, family, scope, slots))
    start = time.perf_counter()
    final = B.fill(forward, x, slots, info['mask_id'], steps, 0.0)
    return _fill_result(f'ours, logits scored at {scope}', canvas, steps, [s[0].tolist() for s in [*states, final]],
                        tokenizer.decode, time.perf_counter() - start, scope=scope)


def _fill_result(sampler: str, canvas: dict[str, Any], steps: int, states: list[list[int]],
                 decode: Callable[[list[int]], str], seconds: float, **extra: Any) -> dict[str, Any]:
    return {'sampler': sampler, 'candidate': canvas['candidate'], 'steps': steps, 'temperature': 0.0,
            'device': 'cpu', 'precision': 'float32', 'environment': B.environment(), 'seconds': seconds,
            'states': states, 'filled': decode(states[-1]), 'values': values_from_ids(canvas['slots'], states[-1], decode),
            **extra}


# ---------- check 2: does the right shift copy a visible answer ----------

def copy_cases(team: list[dict[str, Any]], tokenizer: Any, mask_id: int) -> list[dict[str, Any]]:
    """The team written twice, then one case per value of the second copy with that value's own tokens masked.
    Each case holds the masked token ids, the masked positions and the answer's ids. A value whose tokens cross
    its quotes is left out."""
    pieces = json_pieces(team)
    rendered, where = D.render(pieces, lambda _, value: value_text(value))
    text = COPY_LABELS[0] + rendered + COPY_LABELS[1] + rendered
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, spans = list(encoded['input_ids']), [tuple(o) for o in encoded['offset_mapping']]
    second = len(COPY_LABELS[0]) + len(rendered) + len(COPY_LABELS[1])
    cases = []
    for path, (a, b) in where.items():
        try:
            positions = D.tokens_within(spans, second + a, second + b, D.value_label(path))
        except ValueError:
            continue
        if not positions:
            continue
        masked = list(ids)
        for n in positions:
            masked[n] = mask_id
        cases.append({'path': D.value_label(path), 'value': text[second + a:second + b], 'ids': masked,
                      'positions': positions, 'answer': [ids[n] for n in positions]})
    return cases


def fill_case(full_logits: B.Forward, case: dict[str, Any], mask_id: int, shift: bool,
              device: str = 'cpu') -> list[int]:
    """One case filled at temperature 0, one token per step, with or without the right shift."""
    x = torch.tensor([case['ids']], device=device)
    slots = torch.tensor(case['positions'], device=device)
    forward = (lambda t: B.shift_right(full_logits(t))) if shift else full_logits
    return B.fill(forward, x, slots, mask_id, len(case['positions']), 0.0)[0].tolist()


def copied(filled: Sequence[int], case: dict[str, Any]) -> tuple[bool, float]:
    """Whether the whole answer was copied, and the share of its tokens that were."""
    got = [filled[n] for n in case['positions']]
    same = sum(g == a for g, a in zip(got, case['answer']))
    return same == len(case['answer']), same / len(case['answer'])


def copy_teams() -> list[list[dict[str, Any]]]:
    """Halves of the first two distinct meta-grid teams: Pokémon 1-3 and 4-6 of each."""
    import jev_fill
    teams = list({json.dumps(t['original']): t['original'] for t in jev_fill.tasks('meta')}.values())[:2]
    return [team[i:i + COPY_POKEMON] for team in teams for i in (0, COPY_POKEMON)]


def copy_check(key: str, out: Path, device: str) -> dict[str, Any]:
    """Check 2 for one candidate. Cases already in out are kept, so a killed run resumes."""
    canvas = json.loads((Path.home() / 'vgc-data/masked_memory_bench/canvas' / f'{key}.json').read_text())
    info, family = canvas['candidate'], canvas['family']
    tokenizer = B.load_tokenizer(info)
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(1.0)
    model = B.load_model(info, family, 'native', device)
    result = json.loads(out.read_text()) if out.exists() else {
        'candidate': info, 'family': family, 'trained_shifted': family in B.SHIFTED, 'device': device,
        'precision': B.dtype_name(family), 'environment': B.environment(), 'cases': []}
    done = {(c['team'], c['path']) for c in result['cases']}
    for t, team in enumerate(copy_teams()):
        for case in copy_cases(team, tokenizer, info['mask_id']):
            if (t, case['path']) in done:
                continue
            assert len(case['ids']) <= 1024, len(case['ids'])
            row = {'team': t, 'path': case['path'], 'value': case['value'], 'tokens': len(case['positions'])}
            for shift in (True, False):
                filled = fill_case(lambda x: B.full_logits(model, family, x), case, info['mask_id'], shift, device)
                exact, share = copied(filled, case)
                row['shift' if shift else 'no_shift'] = {
                    'exact': exact, 'token_share': share,
                    'got': tokenizer.decode([filled[n] for n in case['positions']])}
            result['cases'].append(row)
            out.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    result['summary'] = {name: {'exact': sum(c[name]['exact'] for c in result['cases']),
                                'tokens': sum(c[name]['token_share'] * c['tokens'] for c in result['cases'])}
                         for name in ('shift', 'no_shift')}
    result['summary']['cases'] = len(result['cases'])
    result['summary']['answer_tokens'] = sum(c['tokens'] for c in result['cases'])
    out.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    return result


# ---------- the command line ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    theirs = sub.add_parser('dllm')
    ours = sub.add_parser('ours')
    for p in (theirs, ours):
        p.add_argument('--canvas', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--steps', type=int, default=128)
    theirs.add_argument('--block-size', type=int, default=None, help='default: one window over the whole text')
    ours.add_argument('--scope', choices=B.SCOPES, required=True)
    sub.add_parser('schedule').add_argument('--out', type=Path, required=True)
    pair = sub.add_parser('compare')
    pair.add_argument('a', type=Path)
    pair.add_argument('b', type=Path)
    copy = sub.add_parser('copy')
    copy.add_argument('--key', choices=B.ORDER, required=True)
    copy.add_argument('--out', type=Path, required=True)
    copy.add_argument('--device', default='mps')
    sub.add_parser('quotes').add_argument('--dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ('dllm', 'ours'):
        canvas = json.loads(args.canvas.read_text())
        result = (dllm_fill(canvas, args.steps, args.block_size) if args.command == 'dllm'
                  else ours_fill(canvas, args.scope, args.steps))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"{result['sampler']}: {result['steps']} steps, {len(result['states']) - 1} taken, "
              f"{result['seconds']:.0f} s", flush=True)
    elif args.command == 'schedule':
        from dllm.core.samplers.utils import get_num_transfer_tokens
        from dllm.core.schedulers import LinearAlphaScheduler
        scheduler, mismatches, total = LinearAlphaScheduler(), [], 0
        for masks in range(1, 513):
            for steps in (8, 16, 32, 64, 128, 256, 512):
                theirs_counts = get_num_transfer_tokens(torch.ones(1, masks, dtype=torch.bool), steps, scheduler)
                theirs_list = [int(n) for n in theirs_counts[0] if n > 0]
                total += 1
                if theirs_list != B.schedule(masks, steps):
                    mismatches.append({'masks': masks, 'steps': steps, 'dllm': theirs_list,
                                       'ours': B.schedule(masks, steps)})
        args.out.write_text(json.dumps({'dllm': DLLM_COMMIT, 'settings': total, 'mismatches': mismatches}) + '\n')
        print(f'schedule: {len(mismatches)} of {total} settings differ', flush=True)
    elif args.command == 'compare':
        a, b = (json.loads(p.read_text()) for p in (args.a, args.b))
        d = differences(a['states'], b['states'])
        print(f"{a['sampler']} against {b['sampler']}: {d['differing_tokens']} differing tokens, "
              f"first differing step {d['first_step']}")
    elif args.command == 'copy':
        s = copy_check(args.key, args.out, args.device)['summary']
        print(f"{args.key}: {s['cases']} values; copied whole with the shift {s['shift']['exact']}, "
              f"without {s['no_shift']['exact']}", flush=True)
    else:
        rows = quote_rows(args.dir)
        print(quote_table(rows))
        print(json.dumps(rows))
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
