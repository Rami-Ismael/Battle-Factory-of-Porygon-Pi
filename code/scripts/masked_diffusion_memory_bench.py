"""Peak GPU memory of each masked diffusion candidate on the laptop, per sampler setting (#170).

The bar is code/docs/generation-memory-spec.md: a generation run may use at most
torch.mps.recommended_max_memory() (17.76 GiB on the M4 Pro). Each run sets
torch.mps.set_per_process_memory_fraction(1.0) before the model loads, so a run that goes over stops with an
out-of-memory error instead of swapping. Peak is the largest torch.mps.driver_allocated_memory() read after the
model loads and after every sampling step. Time is recorded, never judged.

The text is the approved in-place text (#168, masked_diffusion_text.in_place_text) for the longest mask: all six
Pokémon hidden, with the win-rate ask whose text is longest in the candidate's tokenizer (#201). The whole row is
fixed in length, so no generation length is set and nothing follows the team to trim.

The sampler is the step of dLLM's MDLMSampler.infill (ZHZisZZ/dllm at ca176752, the sampler #165 found runs
correctly on the masked Qwen models), written here because dllm is not installed in venv-dlm and its step
schedule and Gumbel noise use float64, which the laptop's GPU refuses:
    one window over the whole text; at each step the model scores the text, each mask draws a token by
    Gumbel-max at the given temperature, and the most confident masks are committed (low-confidence remasking).
    The number committed per step follows dLLM's linear schedule, computed in Python integers on the CPU.
    The Gumbel noise is float32 in log form, argmax(logits - T * log(-log u)), which draws from the same
    distribution as dLLM's exp(logits) / (-log u)^T and cannot overflow.
Temperature and step count are always explicit (dLLM's defaults are 0, which draws one team n times, and 128).

Two logit scopes per candidate:
    all     the model scores every position, as dLLM does
    slots   only the mask positions are scored: logits_to_keep on the masked Qwen models (#170's comment); the
            output layer on those positions for DiffuGPT, DiffuLLaMA and Dream; a hook on LLaDA's final norm
DiffuGPT, DiffuLLaMA and Dream predict position i from position i - 1, so their logits shift right by one, as
their own samplers do.

Two weight precisions:
    native  the precision the candidate ships in (float16 for DiffuGPT, bfloat16 for the rest)
    int8    every linear layer's weight rounded to int8 with one scale per output row, multiplied by torch's
            _weight_int8pack_mm; activations, embeddings and norms stay in the native precision. The model loads
            on the CPU and each layer moves to the GPU once rounded, so the full-precision weights never reach it
            (LLaDA 8B's do not load there: transformers asks the GPU for one 14.93 GiB buffer and it refuses).

    prepare --dir D                 build each candidate's text with its tokenizer, into D/canvas/<key>.json
    run --canvas F --batch B ...    one setting in this process, written to --out; exit 3 when over the ceiling
    ladder --dir D --key K          batch sizes 1, 2, 4, ... per logit scope, one process each, until one goes over
    agree --canvas F --out O        int8 against native on the first step's mask predictions: how often the most
                                    likely token is the same, and the KL divergence (native on the CPU)
    report --dir D                  the results as a Markdown table
    fetch / drop --dir D --key K    download the candidate's weights if the disk has room; delete them after
"""
import argparse
import contextlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

ORDER = ('diffugpt_s', 'diffugpt_m', 'qwen25c', 'qwen3', 'diffullama', 'dream', 'llada')   # smallest first
FAMILY = {'diffugpt_s': 'gpt2', 'diffugpt_m': 'gpt2', 'qwen25c': 'a2d', 'qwen3': 'a2d', 'diffullama': 'llama',
          'dream': 'dream', 'llada': 'llada'}
SHIFTED = {'gpt2', 'llama', 'dream'}        # logits at i - 1 predict token i
STORED_DTYPE = {'gpt2': 'float16'}          # DiffuGPT ships float16; every other candidate ships bfloat16
SCOPES = ('all', 'slots')
PRECISIONS = ('native', 'int8')
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
OVER = 3                                    # exit code of a run that went over the ceiling
SAMPLER = ('in-place confidence-order fill: the step of dLLM MDLMSampler.infill (ca176752), one window, '
           'low-confidence remasking, linear schedule, float32 Gumbel-max')
GIB = 2**30


# ---------- the sampler: pure torch, so the tests run on the CPU ----------

def schedule(masks: int, steps: int) -> list[int]:
    """Tokens committed per step under dLLM's linear schedule, zeros dropped: step j commits
    round(remaining / (steps - j)), rounding half to even as torch.round does."""
    counts, remaining = [], masks
    for j in range(steps):
        n = min(round(remaining / (steps - j)), remaining)
        remaining -= n
        if n:
            counts.append(n)
    assert sum(counts) == masks
    return counts


def gumbel_argmax(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> torch.Tensor:
    """argmax of logits plus temperature-scaled Gumbel noise, in float32; plain argmax at temperature 0."""
    if temperature == 0:
        return logits.argmax(-1)
    u = torch.rand(logits.shape, generator=generator, device=logits.device, dtype=torch.float32)
    u.clamp_(min=torch.finfo(torch.float32).tiny).log_().neg_().log_().mul_(-temperature).add_(logits)
    return u.argmax(-1)


Forward = Callable[[torch.Tensor], torch.Tensor]    # canvas [B, L] -> logits [B, L or slots, V]


@torch.no_grad()                            # as dLLM's samplers: no activations kept for gradients
def fill(forward: Forward, canvas: torch.Tensor, slots: torch.Tensor, mask_id: int, steps: int, temperature: float,
         generator: torch.Generator | None = None, on_step: Callable[[int], None] = lambda _: None) -> torch.Tensor:
    """Fill the mask tokens at the slot positions of canvas [B, L] in place, one forward per step. forward returns
    logits for every position (scope all) or for the slot positions only, in slot order (scope slots), already
    shifted for a model that predicts from the previous position."""
    x = canvas.clone()
    for j, k in enumerate(schedule(len(slots), steps)):
        logits = forward(x).float()
        x0 = gumbel_argmax(logits, temperature, generator)
        p = torch.softmax(logits, -1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        del logits
        if x0.shape[1] != len(slots):
            x0, p = x0[:, slots], p[:, slots]
        current = x[:, slots]
        p = torch.where(current == mask_id, p, -math.inf)
        chosen = p.topk(k, dim=1).indices
        x[:, slots] = current.scatter(1, chosen, x0.gather(1, chosen))
        on_step(j)
    return x


def shift_right(logits: torch.Tensor) -> torch.Tensor:
    """Position i's prediction from position i - 1, as Dream's and DiffuLLaMA's samplers shift."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def is_over_ceiling(error: BaseException) -> bool:
    return isinstance(error, (RuntimeError, torch.OutOfMemoryError)) and 'out of memory' in str(error).lower()


# ---------- the text each candidate fills ----------

def candidate_info(candidate: Any) -> dict[str, Any]:
    return {'key': candidate.key, 'label': candidate.label, 'repo': candidate.repo, 'revision': candidate.revision,
            'mask_id': candidate.mask_id, 'mask_token': candidate.mask_token, 'qwen2_files': candidate.qwen2_files}


def build_canvas(candidate: Any, tokenizer: Any, task: dict[str, Any], grid: str) -> dict[str, Any]:
    """The in-place text for task with the candidate's longest ask, as token ids with the slot positions."""
    import count_masked_diffusion_text as C
    import masked_diffusion_text as D
    ask = C.longest_ask(tokenizer, task)
    built = D.in_place_text(task, ask, tokenizer, C.LIMIT)
    ids = tokenizer(built.text, add_special_tokens=False).input_ids
    slots = [n for s in built.slots for n in range(s.first_token, s.first_token + s.length)]
    assert all(ids[n] == candidate.mask_id for n in slots) and ids.count(candidate.mask_id) == len(slots)
    return {'candidate': candidate_info(candidate), 'family': FAMILY[candidate.key], 'version': built.version,
            'mask': {'grid': grid, 'id': task['id'], 'source': task['source'], 'task': task['task'], 'k': task['k']},
            'ask': D.ask_text(ask), 'text': built.text, 'ids': ids, 'slots': slots, 'tokens': len(ids)}


def prepare(directory: Path) -> None:
    import count_masked_diffusion_text as C
    import jev_fill
    task = next(t for t in jev_fill.tasks('meta') if t['task'] == 'pokemon' and t['k'] == 6)
    out = directory / 'canvas'
    out.mkdir(parents=True, exist_ok=True)
    for candidate in sorted(C.CANDIDATES, key=lambda c: ORDER.index(c.key)):
        canvas = build_canvas(candidate, C.load_tokenizer(candidate), task, 'meta')
        (out / f'{candidate.key}.json').write_text(json.dumps(canvas, ensure_ascii=False) + '\n')
        print(f"{candidate.label}: {canvas['tokens']} tokens, {len(canvas['slots'])} masks, ask {canvas['ask']}")


# ---------- loading each family on the laptop's GPU ----------

def dtype_name(family: str) -> str:
    return STORED_DTYPE.get(family, 'bfloat16')


def load_tokenizer(info: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer, Qwen2TokenizerFast
    cls = Qwen2TokenizerFast if info['qwen2_files'] else AutoTokenizer
    tokenizer = cls.from_pretrained(info['repo'], revision=info['revision'])
    if info['mask_token']:
        tokenizer.mask_token = info['mask_token']
    return tokenizer


def load_diffugpt(info: dict[str, Any], dtype: torch.dtype, device: str = 'mps') -> Any:
    """DiffuGPT is HKUNLP's DiscreteDiffusionModel wrapper saved whole: embed_tokens, denoise_model (GPT-2 without
    its token table) and a head tied to embed_tokens. Rebuilt here as GPT2LMHeadModel; the caller makes attention
    bidirectional with an all-zero 4D mask."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import GPT2Config, GPT2LMHeadModel
    weights = load_file(hf_hub_download(info['repo'], 'model.safetensors', revision=info['revision']))
    config = GPT2Config.from_pretrained(info['repo'], revision=info['revision'])
    config.vocab_size = weights['embed_tokens.weight'].shape[0]
    config._attn_implementation = 'sdpa'
    model = GPT2LMHeadModel(config)
    state = {'transformer.wte.weight': weights.pop('embed_tokens.weight')}
    state |= {'transformer.' + k.removeprefix('denoise_model.'): v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith('attn.bias') and k != 'lm_head.weight']
    if missing or unexpected:
        raise ValueError(f'DiffuGPT weights do not match GPT-2: missing {missing[:5]}, unexpected {unexpected[:5]}')
    model.tie_weights()
    return model.to(device=device, dtype=dtype)


MAIN_ONLY_IMPORTS = {'dllm'}               # the masked Qwen files import dllm only under __name__ == '__main__'


@contextlib.contextmanager
def ignoring_main_only_imports() -> Iterator[None]:
    """Transformers checks every import named anywhere in a remote modeling file, so the masked Qwen files fail
    without dllm installed although loading never runs the block that imports it."""
    from transformers import dynamic_module_utils as dm
    original = dm.get_imports
    dm.get_imports = lambda filename: [m for m in original(filename) if m not in MAIN_ONLY_IMPORTS]
    try:
        yield
    finally:
        dm.get_imports = original


class Int8Linear(torch.nn.Module):
    """A linear layer whose weight is int8 with one scale per output row (symmetric, absolute maximum to 127)."""

    def __init__(self, linear: torch.nn.Linear, device: str | torch.device) -> None:
        super().__init__()
        w = linear.weight.detach().float()
        scales = w.abs().amax(1).clamp(min=torch.finfo(torch.float32).tiny) / 127
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_buffer('weight', (w / scales[:, None]).round().clamp(-127, 127).to(torch.int8).to(device))
        self.register_buffer('scales', scales.to(linear.weight.dtype).to(device))
        self.register_buffer('bias', None if linear.bias is None else linear.bias.detach().to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch._weight_int8pack_mm(x.reshape(-1, self.in_features), self.weight, self.scales.to(x.dtype))
        y = y.reshape(*x.shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias


def quantize_linears(model: torch.nn.Module, device: str | torch.device) -> int:
    """Replace every linear layer by an Int8Linear on device, one at a time so only one layer's float copy exists at
    once. A layer that shares its weight with an embedding table is left alone. Returns the number replaced."""
    tied = {m.weight.data_ptr() for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and m.weight.data_ptr() not in tied]
    for name in names:
        parent, _, child = name.rpartition('.')
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, child, Int8Linear(getattr(owner, child), device))
    return len(names)


def load_model(info: dict[str, Any], family: str, precision: str = 'native', device: str = 'mps') -> Any:
    """The candidate on device; int8 always ends on the GPU, after loading on the CPU and rounding there."""
    from transformers import AutoModel, AutoModelForMaskedLM, LlamaForCausalLM
    dtype = getattr(torch, dtype_name(family))
    first = device if precision == 'native' else 'cpu'
    common = {'revision': info['revision'], 'dtype': dtype, 'device_map': first}
    if family == 'gpt2':
        model = load_diffugpt(info, dtype, first)
    elif family == 'llama':
        model = LlamaForCausalLM.from_pretrained(info['repo'], attn_implementation='sdpa', **common)
    elif family == 'a2d':
        with ignoring_main_only_imports():
            model = AutoModelForMaskedLM.from_pretrained(info['repo'], trust_remote_code=True, **common)
    elif family == 'dream':                 # its eager attention cannot read the 'full' mask its sampler passes
        model = AutoModel.from_pretrained(info['repo'], trust_remote_code=True, attn_implementation='sdpa', **common)
    else:                                   # llada: its own remote code at the pinned revision
        model = AutoModel.from_pretrained(info['repo'], trust_remote_code=True, **common)
    if precision == 'int8':
        quantize_linears(model, 'mps')
        model.to('mps')
    return model.eval()


def bidirectional_mask(model: Any, x: torch.Tensor) -> torch.Tensor:
    """An all-zero additive 4D mask: every position sees every other, as DiffuGPT and DiffuLLaMA's wrapper does."""
    return torch.zeros(x.shape[0], 1, x.shape[1], x.shape[1], dtype=model.dtype, device=x.device)


def hidden_states(model: Any, family: str, x: torch.Tensor) -> torch.Tensor:
    """The last layer's output before the output layer, for the families whose head this script calls itself."""
    if family == 'gpt2':
        return model.transformer(input_ids=x, attention_mask=bidirectional_mask(model, x)).last_hidden_state
    if family == 'llama':
        return model.model(input_ids=x, attention_mask=bidirectional_mask(model, x)).last_hidden_state
    return model.model(input_ids=x, attention_mask='full')[0]       # dream


def full_logits(model: Any, family: str, x: torch.Tensor) -> torch.Tensor:
    if family in ('gpt2', 'llama'):
        return model(input_ids=x, attention_mask=bidirectional_mask(model, x)).logits
    if family == 'dream':
        return model(x, 'full', None).logits
    return model(input_ids=x).logits


def make_forward(model: Any, family: str, scope: str, slots: torch.Tensor) -> Forward:
    """The sampler's forward for one family and logit scope. Scope slots scores only the positions that predict the
    slots: logits_to_keep on the masked Qwen models, the output layer on those positions for DiffuGPT, DiffuLLaMA
    and Dream, and a hook on LLaDA's final norm, whose forward computes its logits inside."""
    shifted = family in SHIFTED
    if scope == 'all':
        return lambda x: shift_right(full_logits(model, family, x)) if shifted else full_logits(model, family, x)
    keep = slots - 1 if shifted else slots
    if family == 'a2d':
        return lambda x: model(input_ids=x, logits_to_keep=keep).logits
    if family == 'llada':
        model.model.transformer.ln_f.register_forward_hook(lambda _m, _i, out: out.index_select(1, keep))
        return lambda x: model(input_ids=x).logits
    return lambda x: model.lm_head(hidden_states(model, family, x).index_select(1, keep))


@torch.no_grad()
def sees_later_tokens(model: Any, family: str, canvas: torch.Tensor, slots: torch.Tensor) -> bool:
    """Whether the first slot's prediction changes when the text's last token changes: false for a causal model."""
    forward = make_forward(model, family, 'all', slots)
    other = canvas.clone()
    other[0, -1] = (int(canvas[0, -1]) + 1) % 1000
    first = int(slots[0])
    a, b = forward(canvas)[0, first].float(), forward(other)[0, first].float()
    return bool((a - b).abs().max() > 0)


# ---------- one setting, one process ----------

class Meter:
    """GPU memory read after load and after each step; peak is the largest driver allocation read."""

    def __init__(self) -> None:
        self.readings: list[dict[str, Any]] = []

    def read(self, when: str) -> None:
        torch.mps.synchronize()
        self.readings.append({'when': when, 'driver': torch.mps.driver_allocated_memory(),
                              'current': torch.mps.current_allocated_memory(), 't': time.perf_counter()})

    @property
    def peak(self) -> int:
        return max((r['driver'] for r in self.readings), default=0)


def environment() -> dict[str, Any]:
    import transformers
    return {'torch': torch.__version__, 'transformers': transformers.__version__, 'python': platform.python_version(),
            'macos': platform.mac_ver()[0], 'machine': platform.machine()}


def precision_label(family: str, precision: str) -> str:
    return dtype_name(family) if precision == 'native' else f'int8 weights, {dtype_name(family)} activations'


def run(canvas: dict[str, Any], batch: int, scope: str, steps: int, temperature: float, seed: int,
        precision: str = 'native') -> dict[str, Any]:
    """One sampler setting. The memory fraction is set before anything touches the GPU."""
    torch.mps.set_per_process_memory_fraction(1.0)
    ceiling = torch.mps.recommended_max_memory()
    info, family = canvas['candidate'], canvas['family']
    out = {'candidate': info, 'family': family, 'batch': batch, 'scope': scope, 'steps': steps,
           'temperature': temperature, 'seed': seed, 'sampler': SAMPLER,
           'precision': precision_label(family, precision), 'weights': precision,
           'prompt_tokens': canvas['tokens'], 'mask_tokens': len(canvas['slots']), 'ask': canvas['ask'],
           'mask': canvas['mask'], 'version': canvas['version'], 'ceiling': ceiling, 'environment': environment(),
           'committed_per_step': schedule(len(canvas['slots']), steps)}
    meter, start = Meter(), time.perf_counter()
    tokenizer = load_tokenizer(info)    # first: it asks the Hub about the repo, and a dropped connection after sampling lost a run
    try:
        model = load_model(info, family, precision)
        meter.read('loaded')
        out['load_seconds'] = time.perf_counter() - start
        x = torch.tensor([canvas['ids']], device='mps')
        slots = torch.tensor(canvas['slots'], device='mps')
        out['sees_later_tokens'] = sees_later_tokens(model, family, x, slots)
        meter.read('checked')
        generator = torch.Generator(device='mps').manual_seed(seed)
        begin = time.perf_counter()
        filled = fill(make_forward(model, family, scope, slots), x.repeat(batch, 1), slots, info['mask_id'], steps,
                      temperature, generator, on_step=lambda j: meter.read(f'step {j + 1}'))
        out['sample_seconds'] = time.perf_counter() - begin
        out['seconds_per_team'] = out['sample_seconds'] / batch
        out['filled'] = [tokenizer.decode(row.tolist()) for row in filled.cpu()]
        out['status'] = 'fits'
    except Exception as error:
        if not is_over_ceiling(error):
            raise
        out['status'], out['error'] = 'over ceiling', str(error)[:600]
    out['peak'] = meter.peak
    out['readings'] = [{k: v for k, v in r.items() if k != 't'} for r in meter.readings]
    out['fits_under_ceiling'] = out['status'] == 'fits' and out['peak'] <= ceiling
    return out


def agreement(reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """Log-probabilities [slots, V] from two precisions: how often the most likely token is the same, and the
    KL divergence from the reference, mean and largest over slots, in nats."""
    kl = (reference.exp() * (reference - other)).sum(-1)
    return {'top1_same': float((reference.argmax(-1) == other.argmax(-1)).float().mean()),
            'kl_mean': float(kl.mean()), 'kl_max': float(kl.max())}


@torch.no_grad()
def first_step(model: Any, family: str, canvas: dict[str, Any], device: str) -> torch.Tensor:
    """Log-probabilities at every mask on the unfilled text, the first step every sampler takes, back on the CPU."""
    x = torch.tensor([canvas['ids']], device=device)
    slots = torch.tensor(canvas['slots'], device=device)
    return torch.log_softmax(make_forward(model, family, 'slots', slots)(x)[0].float(), -1).cpu()


def agree(canvas: dict[str, Any]) -> dict[str, Any]:
    """int8 against native on the first step. Native runs on the CPU because LLaDA 8B does not load on the GPU;
    the two never hold memory at once."""
    import gc
    info, family = canvas['candidate'], canvas['family']
    start = time.perf_counter()
    model = load_model(info, family, 'native', 'cpu')
    reference = first_step(model, family, canvas, 'cpu')
    del model
    gc.collect()
    native_seconds = time.perf_counter() - start
    model = load_model(info, family, 'int8')
    quantized = first_step(model, family, canvas, 'mps')
    return {'candidate': info, 'precision': precision_label(family, 'int8'), 'reference': f'{dtype_name(family)} on CPU',
            'mask_tokens': len(canvas['slots']), 'native_seconds': native_seconds,
            'environment': environment(), **agreement(reference, quantized)}


def result_path(directory: Path, key: str, scope: str, batch: int) -> Path:
    return directory / 'runs' / key / f'{scope}-batch{batch:03d}.json'


def ladder(directory: Path, key: str, launch: Callable[[list[str]], int], batches: Sequence[int] = BATCHES,
           scopes: Iterable[str] = SCOPES, extra: Sequence[str] = ()) -> dict[str, list[int]]:
    """Each scope's batch sizes in order, one process each, stopping at the first that does not fit. A setting that
    already has a result is not run again."""
    fitted: dict[str, list[int]] = {}
    for scope in scopes:
        fitted[scope] = []
        for batch in batches:
            path = result_path(directory, key, scope, batch)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                launch(['run', '--canvas', str(directory / 'canvas' / f'{key}.json'), '--batch', str(batch),
                        '--scope', scope, '--out', str(path), *extra])
            result = json.loads(path.read_text()) if path.exists() else {}
            if result.get('status') != 'fits' or not result.get('fits_under_ceiling', True):
                break
            fitted[scope].append(batch)
    return fitted


def crashed_settings(directory: Path, key: str, fitted: dict[str, list[int]], batches: Sequence[int] = BATCHES,
                     scopes: Iterable[str] = SCOPES) -> list[str]:
    """The setting where each scope's ladder stopped without a result file: a crash, not the ceiling."""
    crashed = []
    for scope in scopes:
        rest = [b for b in batches if b not in fitted[scope]]
        if rest and not result_path(directory, key, scope, rest[0]).exists():
            crashed.append(f'{scope} batch {rest[0]}')
    return crashed


def subprocess_launch(arguments: list[str]) -> int:
    return subprocess.call([sys.executable, str(Path(__file__).resolve()), *arguments])


# ---------- weights on a nearly full disk ----------

FILES = ['*.safetensors', '*.json', '*.py', '*.txt', '*.model', '*.jinja']     # never the .bin duplicates
MARGIN = 4 * 10**9                          # bytes left free after a download


def fetch(info: dict[str, Any]) -> None:
    """Download the candidate's files at its pinned revision, refusing when the disk would keep under MARGIN free."""
    import shutil
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    meta = HfApi().model_info(info['repo'], revision=info['revision'], files_metadata=True)
    weights = [s for s in meta.siblings if s.rfilename.endswith('.safetensors')]
    missing = sum(s.size or 0 for s in weights
                  if not isinstance(try_to_load_from_cache(info['repo'], s.rfilename, revision=info['revision']), str))
    free = shutil.disk_usage(Path.home()).free
    if free - missing < MARGIN:
        raise SystemExit(f"{info['label']}: {missing / 1e9:.2f} GB to download, {free / 1e9:.2f} GB free")
    snapshot_download(info['repo'], revision=info['revision'], allow_patterns=FILES)
    print(f"{info['label']}: weights present ({missing / 1e9:.2f} GB downloaded)", flush=True)


def drop(info: dict[str, Any]) -> None:
    """Delete the candidate's downloaded weight files, keeping its tokenizer and code, to make room for the next."""
    from huggingface_hub import scan_cache_dir
    freed = 0
    for repo in scan_cache_dir().repos:
        if repo.repo_id != info['repo']:
            continue
        for revision in repo.revisions:
            for f in revision.files:
                if f.file_name.endswith('.safetensors'):
                    freed += f.size_on_disk
                    f.blob_path.unlink(missing_ok=True)
                    f.file_path.unlink(missing_ok=True)
    print(f"{info['label']}: deleted {freed / 1e9:.2f} GB of weights", flush=True)


# ---------- the report ----------

def gib(n: int) -> str:
    return f'{n / GIB:.2f}'


def report(directory: Path) -> str:
    rows = [json.loads(p.read_text()) for p in sorted((directory / 'runs').glob('*/*.json'))]
    rows.sort(key=lambda r: (ORDER.index(r['candidate']['key']), r['scope'], r['batch']))
    lines = ['| Candidate | Weights | Logits scored | Batch | Peak GiB | Under 17.76 GiB | Seconds per team | '
             'Text tokens | Masks |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        seconds = f"{r['seconds_per_team']:.1f}" if 'seconds_per_team' in r else '—'
        scope = 'every position' if r['scope'] == 'all' else 'mask positions only'
        lines.append(f"| {r['candidate']['label']} | {r['precision']} | {scope} | {r['batch']} | {gib(r['peak'])} | "
                     f"{'yes' if r['fits_under_ceiling'] else 'no, ' + r['status']} | {seconds} | "
                     f"{r['prompt_tokens']} | {r['mask_tokens']} |")
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare').add_argument('--dir', type=Path, required=True)
    one = sub.add_parser('run')
    one.add_argument('--canvas', type=Path, required=True)
    one.add_argument('--batch', type=int, required=True)
    one.add_argument('--scope', choices=SCOPES, required=True)
    one.add_argument('--out', type=Path, required=True)
    for p in (one, many := sub.add_parser('ladder')):
        p.add_argument('--steps', type=int, default=128)
        p.add_argument('--temperature', type=float, default=1.0)
        p.add_argument('--seed', type=int, default=2026)
        p.add_argument('--precision', choices=PRECISIONS, default='native')
    check = sub.add_parser('agree')
    check.add_argument('--canvas', type=Path, required=True)
    check.add_argument('--out', type=Path, required=True)
    many.add_argument('--dir', type=Path, required=True)
    many.add_argument('--key', choices=ORDER, required=True)
    many.add_argument('--batches', default=','.join(map(str, BATCHES)))
    sub.add_parser('report').add_argument('--dir', type=Path, required=True)
    for name in ('fetch', 'drop'):
        weights = sub.add_parser(name)
        weights.add_argument('--dir', type=Path, required=True)
        weights.add_argument('--key', choices=ORDER, required=True)
    args = parser.parse_args(argv)
    if args.command in ('fetch', 'drop'):
        info = json.loads((args.dir / 'canvas' / f'{args.key}.json').read_text())['candidate']
        (fetch if args.command == 'fetch' else drop)(info)
    elif args.command == 'prepare':
        prepare(args.dir)
    elif args.command == 'run':
        result = run(json.loads(args.canvas.read_text()), args.batch, args.scope, args.steps, args.temperature,
                     args.seed, args.precision)
        args.out.write_text(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"{result['candidate']['label']} {result['precision']} {args.scope} batch {args.batch}: "
              f"{result['status']}, peak {gib(result['peak'])} GiB of {gib(result['ceiling'])}", flush=True)
        return 0 if result['status'] == 'fits' else OVER
    elif args.command == 'agree':
        result = agree(json.loads(args.canvas.read_text()))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result) + '\n')
        print(f"{result['candidate']['label']} int8 against {result['reference']}: most likely token the same at "
              f"{result['top1_same']:.1%} of {result['mask_tokens']} masks, KL mean {result['kl_mean']:.4f} "
              f"max {result['kl_max']:.4f} nats", flush=True)
    elif args.command == 'ladder':
        extra = ['--steps', str(args.steps), '--temperature', str(args.temperature), '--seed', str(args.seed),
                 '--precision', args.precision]
        fitted = ladder(args.dir, args.key, subprocess_launch, [int(b) for b in args.batches.split(',')], extra=extra)
        batches = [int(b) for b in args.batches.split(',')]
        crashed = crashed_settings(args.dir, args.key, fitted, batches)
        if crashed:
            print(f'{args.key}: crashed without a result at {", ".join(crashed)}; fitted {fitted}', flush=True)
            return 1
        print(f'{args.key}: batch sizes that fit {fitted}', flush=True)
    else:
        print(report(args.dir))
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
