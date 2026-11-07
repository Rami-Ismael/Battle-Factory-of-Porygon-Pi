"""Count the in-place masked diffusion text (#199, #200) for every Mask Lab mask, with every candidate's
tokenizer (#201). Nothing is sent to a model: each mask is built by masked_diffusion_text.in_place_text and
tokenized.

The in-place text is the whole training row. The model fills its mask tokens where they sit, so there is
no separate answer to add: the text's length is the prompt plus the answer. For each candidate the command
reports the longest text, the mask it came from, and how many masks run over 1,024 tokens, the length
Masked Qwen3 0.6B was fine-tuned at (#162).

Every candidate of #162 loads its own tokenizer files at the revision #162 pinned. Two load around a
defect #162 found, never by guessing:
    Dream 7B     its tokenizer class is remote code, so its own vocabulary files load into the fast Qwen2
                 tokenizer class instead; no code from the repository runs.
    LLaDA 8B     the mask token loads as None, so it is set to <|mdm_mask|>, id 126336 as #162 found.
Each mask token must be one token with the id #162 found. A tokenizer that fails to load, or whose mask
token differs, is reported as not counted, and a mask the builder refuses is listed with its reason. No
count is ever estimated.

Each candidate counts with the win-rate ask whose text is longest in its tokenizer (one of 0.00 to 1.00
and unknown), so no mask's real row can be longer than the count.

    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/count_masked_diffusion_text.py

It writes every mask's counts to artifacts/masked-diffusion-prompt/mask-counts.json and prints the
per-candidate summary as a Markdown table, the one posted on #168.
"""
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jev_fill
import masked_diffusion_text as D

LIMIT = 1024                        # Masked Qwen3 0.6B was fine-tuned at 1,024 tokens (#162)
NO_LIMIT = 10**9                    # build every mask in full, then compare with LIMIT
GRIDS = ('random', 'heldout', 'meta')
GRID_NAMES = {'random': 'random grid', 'heldout': 'held-out grid', 'meta': 'meta grid'}
ASKS: tuple[float | None, ...] = (None, *(n / 100 for n in range(101)))
OUT = Path(__file__).resolve().parents[2] / 'artifacts/masked-diffusion-prompt/mask-counts.json'
TASK_NAMES = {'items': 'items', 'abilities': 'abilities', 'moves': 'moves', 'natures': 'Stat Alignments',
              'stats': 'Stat Points', 'pokemon': 'whole Pokémon'}


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    repo: str
    revision: str
    mask_id: int                    # the mask token id #162 found
    mask_token: str | None = None   # set when the tokenizer does not load it (LLaDA)
    qwen2_files: bool = False       # load the repo's vocabulary into Qwen2TokenizerFast (Dream)


CANDIDATES = (                      # the seven of #162, at the revisions it pinned
    Candidate('qwen3', 'Masked Qwen3 0.6B', 'dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1',
              'c8d24a3f4adaeef46881b450e1bf7d1005203bd7', 151669),
    Candidate('qwen25c', 'Masked Qwen2.5 Coder 0.5B', 'dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1',
              'a284e895a6248256baf2f60502e54aba61b24c1a', 151665),
    Candidate('diffugpt_s', 'DiffuGPT small', 'diffusionfamily/diffugpt-s',
              '391d9850148c0beda355a8874bea958a40164c74', 10541),
    Candidate('diffugpt_m', 'DiffuGPT medium', 'diffusionfamily/diffugpt-m',
              'ce62f89724018390198ded702c4926791cf17c21', 50257),
    Candidate('diffullama', 'DiffuLLaMA 7B', 'diffusionfamily/diffullama',
              '31b22850c7b397d6948ea04076099b9ff7110adc', 811),
    Candidate('dream', 'Dream 7B', 'Dream-org/Dream-v0-Base-7B',
              '6572adb5535263e4d1a337b56942ba48b6dee2a9', 151666, qwen2_files=True),
    Candidate('llada', 'LLaDA 8B', 'GSAI-ML/LLaDA-8B-Base',
              '0f2787f2d87eac5eed8a087d5ecd24277e6255b2', 126336, mask_token='<|mdm_mask|>'),
)


def load_tokenizer(candidate: Candidate) -> Any:
    """The candidate's own tokenizer at its pinned revision. Raises if its mask token is not the one #162 found."""
    from transformers import AutoTokenizer, Qwen2TokenizerFast
    cls = Qwen2TokenizerFast if candidate.qwen2_files else AutoTokenizer
    tokenizer = cls.from_pretrained(candidate.repo, revision=candidate.revision)
    if candidate.mask_token:
        tokenizer.mask_token = candidate.mask_token
    check_mask_token(tokenizer, candidate)
    return tokenizer


def check_mask_token(tokenizer: Any, candidate: Candidate) -> None:
    ids = tokenizer(tokenizer.mask_token, add_special_tokens=False).input_ids if tokenizer.mask_token else []
    if ids != [candidate.mask_id]:
        raise ValueError(f'mask token {tokenizer.mask_token!r} tokenizes as {ids}, not [{candidate.mask_id}] as #162 found')


def text_tokens(built: D.InPlaceText) -> int:
    return sum(built.part_tokens.values())


def longest_ask(tokenizer: Any, task: dict[str, Any]) -> float | None:
    """The ask whose in-place text is longest; the first such ask on a tie."""
    return max(ASKS, key=lambda ask: text_tokens(D.in_place_text(task, ask, tokenizer, NO_LIMIT)))


def mask_name(row: dict[str, Any]) -> str:
    """'meta grid MB158, 1 items hidden (team[3].item)' style, for a reader of #168."""
    if row['task'] == 'pokemon':
        hidden = 'all six Pokémon hidden' if row['k'] == 6 else f"{row['k']} whole Pokémon hidden"
    else:
        hidden = f"{row['k']} {TASK_NAMES.get(row['task'], row['task'])} hidden"
    paths = row['masked_paths']
    shown = f" ({', '.join(paths)})" if len(paths) <= 3 and row['task'] != 'pokemon' else ''
    return f"{GRID_NAMES[row['grid']]} {row['source']}, {hidden}{shown}"


def count_candidate(tokenizer: Any, tasks: Iterable[tuple[str, dict[str, Any]]], limit: int) -> dict[str, Any]:
    """One tokenizer over every task: each mask's tokens and mask tokens, the longest mask and the masks over limit."""
    tasks = list(tasks)
    ask = longest_ask(tokenizer, tasks[0][1])
    tokens: dict[str, dict[str, list[int]]] = {}        # grid -> mask id -> [text tokens, mask tokens]
    refused, largest = [], None
    for grid, task in tasks:
        where = {'grid': grid, 'id': task['id'], 'source': task['source'], 'task': task['task'], 'k': task['k'],
                 'masked_paths': task['masked_paths']}
        try:
            built = D.in_place_text(task, ask, tokenizer, NO_LIMIT)
        except ValueError as error:
            refused.append({**where, 'reason': str(error)})
            continue
        n = text_tokens(built)
        tokens.setdefault(grid, {})[task['id']] = [n, sum(s.length for s in built.slots)]
        if largest is None or n > largest['tokens']:
            largest = {**where, 'tokens': n, 'parts': built.part_tokens}
    counts = [n for grid in tokens.values() for n, _ in grid.values()]
    if largest:
        largest |= {'name': mask_name(largest), 'ties': counts.count(largest['tokens'])}
    return {'counted': True, 'ask': D.ask_text(ask), 'slot_lengths': D.slot_lengths(tokenizer),
            'masks': len(counts), 'refused': refused, 'over_limit': sum(n > limit for n in counts),
            'largest': largest, 'smallest': min(counts, default=None), 'tokens': tokens}


def count(candidates: Iterable[Candidate], tasks: list[tuple[str, dict[str, Any]]],
          load: Callable[[Candidate], Any] = load_tokenizer, limit: int = LIMIT) -> dict[str, Any]:
    """Every candidate's counts over every task. A tokenizer that fails to load is not counted, never estimated."""
    out = []
    for candidate in candidates:
        head = {'key': candidate.key, 'label': candidate.label, 'repo': candidate.repo, 'revision': candidate.revision}
        try:
            tokenizer = load(candidate)
        except Exception as error:      # any load failure: report it, count nothing
            out.append({**head, 'counted': False, 'reason': f'{type(error).__name__}: {error}'[:400]})
            continue
        out.append({**head, 'mask_token': tokenizer.mask_token, **count_candidate(tokenizer, tasks, limit)})
    return {'limit': limit, 'version': D.VERSION, 'grids': sorted({g for g, _ in tasks}, key=GRIDS.index),
            'masks': len(tasks), 'candidates': out}


def summary(report: dict[str, Any]) -> str:
    limit = report['limit']
    lines = [f"| Candidate | Longest text (tokens) | Its mask | Masks over {limit:,} | Masks counted | Ask counted |",
             '|---|---|---|---|---|---|']
    for c in report['candidates']:
        if not c['counted']:
            lines.append(f"| {c['label']} | not counted | {c['reason']} | | 0 of {report['masks']} | |")
            continue
        big = c['largest']
        refused = f", {len(c['refused'])} refused" if c['refused'] else ''
        ties = f", tied with {big['ties'] - 1} other masks" if big['ties'] > 1 else ''
        lines.append(f"| {c['label']} | {big['tokens']:,} | {big['name']}{ties} | {c['over_limit']} | "
                     f"{c['masks']} of {report['masks']}{refused} | `{c['ask']}` |")
    return '\n'.join(lines)


def all_tasks(grids: Iterable[str] = GRIDS) -> list[tuple[str, dict[str, Any]]]:
    return [(grid, task) for grid in grids for task in jev_fill.tasks(grid)]


def main() -> None:
    tasks = all_tasks()
    report = count(CANDIDATES, tasks)
    for c in report['candidates']:
        state = (f"{c['masks']} masks, longest {c['largest']['tokens']}, {c['over_limit']} over {LIMIT}"
                 if c['counted'] else 'not counted: ' + c['reason'])
        print(f"{c['label']}: {state}", file=sys.stderr)
    OUT.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'wrote {OUT}', file=sys.stderr)
    print(summary(report))


if __name__ == '__main__':
    main()
