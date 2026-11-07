"""PROTOTYPE for #168: the exact text the masked diffusion model receives under in-place filling.

    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python artifacts/masked-diffusion-prompt/build_inplace_text_prototype.py

Throwaway. It answers one question, "what will the model receive?", so the owner can approve a
concrete text before #199 builds the real one. It is not the #199 builder: no tests, no length
refusal, no reader for filled text.

The text follows the 2026-11-06 decision on #196: a three-line header (format, rules, win-rate ask),
then the team as compact JSON with each hidden value replaced by a fixed run of mask tokens. Slot
lengths are measured per tokenizer, at the longest Reg M-B value of that field type, inside JSON
quotes; never at the hidden value's own length. Each candidate tokenizer (#162 revisions) counts
every part. A token straddling a quote is counted and reported, never hidden.

Masks are real Mask Lab tasks on meta grid team MB158, one per field type, plus the whole-Pokémon
and all-six masks. Beside them: the same mask's Jev request (counted by Jev's server) and the old
text fine-tunes' prompt and team (src/textdiff.py).
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))
import jev_fill
from mask_paste import FORMAT_NAME, RULES, STATS
from mb_prompt import regulation

JEV_PAGE = ROOT / "notes/Blog/visuals/jev-picks-a-team/jev-picks-a-team.html"
EXAMPLE_ASK = 0.55
LIMIT = 1024
JEV_TOKENS = 24_644                    # meta grid MB158 item mask, Jev's server, 2026-10-27
FREE_TEXT_DRAFT = {"qwen3": 867, "qwen25c": 867, "dream": 867, "llada": 925, "diffugpt": 760,
                   "diffullama": 982}  # PR #217's free-text prompt plus answer, same mask
KEYS = ("species", "item", "ability", "nature", "moves", "statPoints")   # #199's key order
MASKS = [                              # (task id, short name): MB158 on the meta grid
    ("b90b461c39429bf2", "One item"),
    ("12df02000e1dc7de", "One ability"),
    ("1517709760661159", "One move"),
    ("8c31c95e4f47df9c", "One Stat Alignment"),
    ("0613d59de0c68499", "One set of Stat Points"),
    ("22d0b83c75aef17c", "One whole Pokémon"),
    ("e13544e8791a826d", "All six Pokémon"),
]
TOKENIZERS = [                         # same list and revisions as build_prompt_visual.py
    ("qwen3", "Masked Qwen3 0.6B", "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "c8d24a3f4adaeef46881b450e1bf7d1005203bd7"),
    ("qwen25c", "Masked Qwen2.5 Coder 0.5B", "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
     "a284e895a6248256baf2f60502e54aba61b24c1a"),
    ("dream", "Dream 7B (Qwen2.5 vocabulary)", "Qwen/Qwen2.5-7B", None),
    ("llada", "LLaDA 8B", "GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"),
    ("diffugpt", "DiffuGPT small and medium", "diffusionfamily/diffugpt-s", "391d9850148c0beda355a8874bea958a40164c74"),
    ("diffullama", "DiffuLLaMA 7B", "diffusionfamily/diffullama", "31b22850c7b397d6948ea04076099b9ff7110adc"),
]
FIELD = {"species": "species", "item": "item", "ability": "ability", "nature": "nature", "moves": "move"}


def header(ask: float | None) -> list[tuple[str, str]]:
    a = "unknown" if ask is None else f"{ask:.2f}"
    return [("format", f"Format: {FORMAT_NAME}\n"), ("rules", f"Rules: {RULES}\n"),
            ("ask", f"Target win rate vs top meta teams: {a}.\n")]


def mon_field(i: int, mon: dict, key: str, lit, value) -> None:
    if key == "moves":
        lit("[")
        for m in range(4):
            if m: lit(",")
            value(f"team[{i}].moves[{m}]", mon["moves"][m], "move")
        lit("]")
    elif key == "statPoints":
        lit("{")
        for s, stat in enumerate(STATS):
            lit(("," if s else "") + f'"{stat}":')
            value(f"team[{i}].statPoints.{stat}", mon["statPoints"][stat], "stat")
        lit("}")
    else:
        value(f"team[{i}].{key}", mon[key], key)


def segments(team: list, hidden: set) -> list[dict]:
    """The team as compact JSON, cut into literal text and slots. A slot is one hidden value."""
    out: list[dict] = []
    lit = lambda t: out.append({"k": "lit", "t": t})
    def value(path, v, kind):
        quoted = kind != "stat"
        if quoted: lit('"')
        if path in hidden: out.append({"k": "slot", "kind": kind, "path": path, "v": v})
        else: lit(str(v) if not quoted else json.dumps(v, ensure_ascii=False)[1:-1])
        if quoted: lit('"')
    lit("[")
    for i, mon in enumerate(team):
        lit("{" if i == 0 else ",{")
        for j, key in enumerate(KEYS):
            lit(("," if j else "") + f'"{key}":')
            mon_field(i, mon, key, lit, value)
        lit("}")
    lit("]")
    merged: list[dict] = []                          # join neighbouring literals
    for s in out:
        if s["k"] == "lit" and merged and merged[-1]["k"] == "lit": merged[-1]["t"] += s["t"]
        else: merged.append(dict(s))
    return merged


def hidden_paths(task: dict) -> set:
    paths = set()
    for text in task["masked_paths"]:
        for p in jev_fill.expand(jev_fill.parse_path(text), task["masked_team"]):
            if p[1] == "statPoints": paths |= {f"team[{p[0]}].statPoints.{s}" for s in STATS}
            elif p[1] == "moves": paths.add(f"team[{p[0]}].moves[{p[2]}]")
            else: paths.add(f"team[{p[0]}].{p[1]}")
    return paths


def in_quotes(tok, prefix: str, v: str, suffix: str) -> tuple[int, bool]:
    """Tokens of v inside its context, and whether any token straddles v's edges."""
    text = prefix + v + suffix
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    a, b = len(prefix), len(prefix) + len(v)
    inside = [s for s, e in enc.offset_mapping if s >= a and e <= b and e > s]
    cross = any(s < a < e or s < b < e for s, e in enc.offset_mapping)
    return len(inside), cross


def slot_lengths(tok) -> tuple[dict, dict, int]:
    reg = regulation()
    names = reg["names"]
    lens, longest, crossings, checked = {}, {}, 0, 0
    pools = {kind: sorted({names.get(x, x) for x in reg["values"][pool] if x}) for kind, pool in
             (("species", "species"), ("item", "item"), ("ability", "ability"), ("move", "move"), ("nature", "nature"))}
    pools["stat"] = [str(n) for n in range(33)]
    for kind, values in pools.items():
        best, arg = 0, None
        for v in values:
            n, cross = (in_quotes(tok, '{"HP":', v, ',') if kind == "stat"
                        else in_quotes(tok, f'{{"{kind}":"', json.dumps(v, ensure_ascii=False)[1:-1], '","'))
            crossings += cross; checked += 1
            if n > best: best, arg = n, v
        lens[kind], longest[kind] = best, arg
    return lens, longest, crossings


def count(tok, lens: dict, parts: list, segs: list) -> dict:
    n = lambda t: len(tok(t, add_special_tokens=False).input_ids)
    starts = len(tok("a").input_ids) - n("a")                      # start-of-text tokens the tokenizer adds
    lit = sum(n(s["t"]) for s in segs if s["k"] == "lit")
    slots = sum(lens[s["kind"]] for s in segs if s["k"] == "slot")
    head = {k: n(t) for k, t in parts}
    filled = "".join(t for _, t in parts) + "".join(s["t"] if s["k"] == "lit" else
                                                    (str(s["v"]) if s["kind"] == "stat" else s["v"]) for s in segs)
    whole = n(filled)
    pieces = sum(head.values()) + lit + sum(n(str(s["v"])) for s in segs if s["k"] == "slot")
    fills = [in_quotes(tok, '{"HP":', str(s["v"]), ',')[0] if s["kind"] == "stat" else
             in_quotes(tok, f'{{"{s["kind"]}":"', json.dumps(s["v"], ensure_ascii=False)[1:-1], '","')[0]
             for s in segs if s["k"] == "slot"]
    return {**head, "team": lit, "masks": slots, "start": starts, "fills": fills,
            "total": sum(head.values()) + lit + slots + starts, "seam": whole - pieces}


def old_prompt(task: dict, tok) -> int | None:
    """The old text fine-tunes' row: chat prompt plus the team in their paste form (src/textdiff.py)."""
    if not tok.chat_template: return None
    prompt = f"Format: gen9championsvgc2026regmb. Six Pokémon. Target win rate vs top meta teams: {EXAMPLE_ASK:.2f}."
    slots = []
    for m in task["original"]:
        lines = [f"{m['species']} @ {m['item']}", f"Ability: {m['ability']}", f"{m['nature']} Nature"]
        slots.append("\n".join(lines + [f"- {x}" for x in m["moves"]]))
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    return len(p) + len(tok("\n\n".join(slots) + "\n", add_special_tokens=False).input_ids) + 1


def main() -> None:
    by_id = {t["id"]: t for t in jev_fill.tasks("meta")}
    parts = header(EXAMPLE_ASK)
    masks = []
    for task_id, name in MASKS:
        t = by_id[task_id]
        hidden = hidden_paths(t)
        masks.append({"id": task_id, "name": name, "paths": t["masked_paths"], "segs": segments(t["original"], hidden),
                      "hiddenSlots": sorted({int(p[5]) for p in hidden})})
    original = by_id[MASKS[0][0]]["original"]
    tokenizers = []
    for key, label, repo, revision in TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        except Exception as error:                                  # never guess a count
            print(f"{repo}: not counted, {type(error).__name__}: {str(error)[:160]}")
            tokenizers.append({"key": key, "label": label, "repo": repo, "counts": None})
            continue
        lens, longest, crossings = slot_lengths(tok)
        counts = {m["id"]: count(tok, lens, parts, m["segs"]) for m in masks}
        tokenizers.append({"key": key, "label": label, "repo": repo, "revision": revision, "lens": lens,
                           "longest": longest, "crossings": crossings, "counts": counts,
                           "old": old_prompt(by_id[MASKS[0][0]], tok)})
        print(key, lens, "crossings", crossings, {m["name"]: counts[m["id"]]["total"] for m in masks})
    data = {"source": "MB158", "ask": EXAMPLE_ASK, "limit": LIMIT, "jev": JEV_TOKENS, "freeText": FREE_TEXT_DRAFT,
            "header": [{"key": k, "text": t} for k, t in parts], "masks": masks, "tokenizers": tokenizers,
            "team": [{"species": m["species"], "item": m["item"]} for m in original]}
    page = JEV_PAGE.read_text()
    assets = {}
    for name in ("SP", "IT"):
        i = page.find(f"const {name}=", page.find("const REQ="))
        assets[name] = json.loads(page[i + len(name) + 7:page.find(";\n", i)])
    assets = {"SP": {k: v for k, v in assets["SP"].items() if k in {m["species"] for m in original}},
              "IT": {k: v for k, v in assets["IT"].items() if k in {m["item"] for m in original}}}
    i = page.find("const JV_CSS = ")
    css = json.loads(page[i + len("const JV_CSS = "):page.find(";\n", i)])
    template = (HERE / "inplace-text-prototype.template.html").read_text()
    out = (template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
           .replace("/*ASSETS*/null", json.dumps(assets)).replace("/*JV_CSS*/", css))
    (HERE / "inplace-text-prototype.html").write_text(out)
    (HERE / "inplace-text-prototype.data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print("wrote", HERE / "inplace-text-prototype.html")


if __name__ == "__main__":
    main()
