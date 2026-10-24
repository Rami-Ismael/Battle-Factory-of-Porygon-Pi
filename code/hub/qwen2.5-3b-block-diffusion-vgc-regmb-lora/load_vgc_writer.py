"""Load and sample Rami/qwen2.5-3b-block-diffusion-vgc-regmb-lora (Built with Qwen; Qwen RESEARCH LICENSE AGREEMENT).

The model is Fast-dLLM v2's block-diffusion architecture (code from Efficient-Large-Model/Fast_dLLM_v2_1.5B, Apache-2.0,
pinned revision) carrying Qwen/Qwen2.5-3B-Instruct's weights, with this LoRA adapter merged: the construction it was
trained on. generate_batch writes many teams per call; with n=1, greedy decoding and exact=True it reproduces the
released sampler token for token.

    tok, model = load("cuda")          # or "mps"
    ids = prompt_ids(tok, None, "cuda")  # target win rate unknown; or e.g. 0.5
    teams = [tok.decode(t, skip_special_tokens=True) for t in generate_batch(model, ids, 32)]

Source: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi (code/src/textdiff_port.py, code/src/textdiff.py).
"""
import torch
import torch.nn.functional as F

ARCH, ARCH_REV = "Efficient-Large-Model/Fast_dLLM_v2_1.5B", "25093b6f63300adfd57f72145083c8a528fe4f16"
BASE, ADAPTER = "Qwen/Qwen2.5-3B-Instruct", "Rami/qwen2.5-3b-block-diffusion-vgc-regmb-lora"
MASK_ID = 151665
SYSTEM = "You write Pokémon VGC teams as Showdown pastes."
PROMPT = "Format: gen9championsvgc2026regmb. Six Pokémon. Target win rate vs top meta teams: {ask}."


def load(device="cuda", dtype=torch.bfloat16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(ARCH, revision=ARCH_REV, trust_remote_code=True)
    arch = AutoModelForCausalLM.from_pretrained(ARCH, revision=ARCH_REV, trust_remote_code=True, dtype=dtype)
    src = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype)
    cfg = type(arch.config).from_dict({**src.config.to_dict(), "bd_size": arch.config.bd_size, "architectures": arch.config.architectures,
                                       "auto_map": arch.config.auto_map, "model_type": arch.config.model_type})
    model = type(arch)(cfg).to(dtype); del arch
    missing, unexpected = model.load_state_dict(src.state_dict(), strict=False); del src
    assert not unexpected and all(k == "lm_head.weight" for k in missing), (missing, unexpected)
    model.tie_weights()
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    return tok, model.to(device).eval()


def prompt_ids(tok, ask, device="cuda"):
    """ask: a target win rate (two decimals in the prompt) or None for 'unknown'."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT.format(ask="unknown" if ask is None else f"{ask:.2f}")}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)


@torch.no_grad()
def generate_batch(model, prompt_ids, n, max_new_tokens=320, threshold=0.9, block_size=32, small_block_size=8,
                   temperature=1.0, top_p=1.0, block_cache=False, exact=False, stop_token=151645):
    """The released block sampler (modeling.py generate) for n samples of one prompt in one batch. The released loop is
    single-row in practice: it samples row 0 only and stops every row at the first stop token it sees. Here each row
    samples, unmasks (every token above the threshold, and at least its most confident one) and stops on its own; a
    stopped row rides along unchanged until every row has stopped. The forward calls are the released ones: the prompt
    prefill, one per denoising step (the whole 32-token block, or with block_cache the 8-token sub-block against the
    block's cached keys and values), and the call that commits a finished block and drafts the next block's first
    token. exact: softmax over the bf16 logits as released (for the token-for-token check); default fp32.
    Returns a list of n token lists (generated tokens, cut after the stop token)."""
    bs, small = block_size, small_block_size
    dev = prompt_ids.device
    x = prompt_ids.expand(n, -1).clone(); original = x.shape[1]
    rows = torch.arange(n, device=dev)
    past = None
    if original > bs:
        out = model.forward(input_ids=x[:, :original // bs * bs], use_cache=True, update_past_key_values=True, block_size=bs)
        past = out.past_key_values
        if original % bs == 0: x = torch.cat([x, out.logits[:, -1:].argmax(-1)], 1)

    def stopped(x):                                     # the released check, per row: a stop token with no mask before it
        gen = x[:, original:]; hit = gen == stop_token
        if gen.shape[1] == 0: return hit.any(1)
        first = torch.where(hit.any(1), hit.int().argmax(1), gen.shape[1])
        masked_before = ((gen == MASK_ID) & (torch.arange(gen.shape[1], device=dev)[None] < first[:, None])).any(1)
        return hit.any(1) & ~masked_before

    def sample(lg):                                     # (n, k, V) -> tokens (n, k) and their probabilities
        if temperature <= 0:
            p = F.softmax(lg, -1); x1 = p.argmax(-1)
        else:
            p = F.softmax(lg / temperature, -1)
            if top_p < 1.0:
                ps, ix = p.sort(-1, descending=True); keep = ps.cumsum(-1) - ps < top_p
                p = torch.zeros_like(p).scatter(-1, ix, ps * keep); p = p / p.sum(-1, keepdim=True)
            x1 = torch.multinomial(p.flatten(0, 1), 1).view(p.shape[:2])
        return x1, p.gather(-1, x1[..., None]).squeeze(-1)

    shift = lambda lg: torch.cat([lg[:, :1], lg[:, :-1]], 1)
    for _ in range(max_new_tokens // bs):
        if bool(stopped(x).all()): break
        x = torch.cat([x, torch.full((n, bs - x.shape[1] % bs), MASK_ID, dtype=torch.long, device=dev)], 1)
        bpast = None
        for sb in range(bs // small):
            start = -bs + sb * small; end = None if sb == bs // small - 1 else start + small
            while True:
                m = x[:, start:end] == MASK_ID
                need = m.any(1) & ~stopped(x)
                if not bool(need.any()): break
                if block_cache and bpast is not None and not bool((x[:, -bs + sb * small] == MASK_ID).any()):
                    out = model.forward(input_ids=x[:, start:end], use_cache=True, past_key_values=past, update_past_key_values=False,
                                        use_block_cache=True, block_past_key_values=bpast, replace_position=sb * small)
                    lg = shift(out.logits)
                elif block_cache:
                    out = model.forward(input_ids=x[:, -bs:], use_cache=True, past_key_values=past, update_past_key_values=False,
                                        use_block_cache=True)
                    bpast = out.block_past_key_values; lg = shift(out.logits)[:, start:end]
                else:
                    out = model.forward(input_ids=x[:, -bs:], use_cache=True, past_key_values=past, update_past_key_values=False)
                    lg = shift(out.logits)[:, start:end]
                x1, conf = sample(lg if exact else lg.float())
                conf = torch.where(m, conf, torch.full_like(conf, -float("inf")))
                un = conf > threshold
                un[rows, conf.argmax(-1)] = True
                un &= m & need[:, None]
                seg = x[:, start:end]; seg[un] = x1[un]
        if bool(stopped(x).all()): break
        out = model.forward(input_ids=x[:, -bs:], use_cache=True, past_key_values=past, update_past_key_values=True, block_size=bs)
        past = out.past_key_values
        x = torch.cat([x, out.logits[:, -1:].argmax(-1)], 1)
    result = []
    for row in x[:, original:].tolist():
        result.append(row[:row.index(stop_token) + 1] if stop_token in row else row)
    return result
