"""Build prompt-anatomy.html: the masked diffusion prompt (#196) part by part, in the look of the
"What Jev is sent" figure on the Jev picks a team page.

    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python artifacts/masked-diffusion-prompt/build_prompt_visual.py

The prompt is a DRAFT of the free-text form #196 first specified. On 2026-11-06 the owner replaced it
with in-place filling (compact JSON, mask tokens at hidden values; see #196 and #199), so this page
records the superseded form. #201 swaps draft_prompt() for the #199 builder; the template only reads
DATA. The ask line copies the old text fine-tunes' wording (src/textdiff.py PROMPT). Token counts come
from each candidate's own tokenizer at the revision #162 pinned (tokenizer files only, no model code),
except Dream, counted with Qwen2.5-7B's tokenizer at its latest revision. A tokenizer that fails to
load is listed as not counted, never guessed.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))
import jev_fill

JEV_PAGE = ROOT / "notes/Blog/visuals/jev-picks-a-team/jev-picks-a-team.html"
TASK_ID = "b90b461c39429bf2"          # meta grid, unmasked team MB158, slot 4's item hidden: the Jev page's example
MARKER = "???"                        # draft; set in #168
EXAMPLE_ASK = 0.55                    # draft example; a real row carries its own label or "unknown"
SYSTEM = "You write Pokémon VGC teams as Showdown pastes."
LIMIT = 1024                          # masked Qwen3 0.6B was fine-tuned at 1,024 tokens (#162)
JEV_TOKENS = 24_644                   # the same mask's Jev request, counted by Jev's server on 2026-10-27
TOKENIZERS = [                        # (key, label, repo, revision): the seven candidates of #162 (DiffuGPT small and medium share one tokenizer)
    ("qwen3", "Masked Qwen3 0.6B", "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "c8d24a3f4adaeef46881b450e1bf7d1005203bd7"),
    ("qwen25c", "Masked Qwen2.5 Coder 0.5B", "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
     "a284e895a6248256baf2f60502e54aba61b24c1a"),
    ("diffugpt", "DiffuGPT small and medium", "diffusionfamily/diffugpt-s", "391d9850148c0beda355a8874bea958a40164c74"),
    ("dream", "Dream 7B (Qwen2.5 vocabulary)", "Qwen/Qwen2.5-7B", None),   # Dream's own tokenizer needs remote code
    ("llada", "LLaDA 8B", "GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"),
    ("diffullama", "DiffuLLaMA 7B", "diffusionfamily/diffullama", "31b22850c7b397d6948ea04076099b9ff7110adc"),
]


def masked_paste(team: jev_fill.Team, hidden: list[jev_fill.FieldPath]) -> str:
    """The team so far as paste text, with each hidden field written as MARKER."""
    shown = json.loads(json.dumps(team))
    for path in hidden:
        slot = shown[path[0]]
        if len(path) == 3:
            slot[path[1]][path[2]] = MARKER
        else:
            slot[path[1]] = MARKER
    return jev_fill.team_paste(shown).strip()


def ask_line(ask: float | None) -> str:
    """The ask line in the old text fine-tunes' wording (src/textdiff.py PROMPT), so a comparison changes only the model."""
    return f"Target win rate vs top meta teams: {'unknown' if ask is None else f'{ask:.2f}'}."


def draft_prompt(task: jev_fill.Task) -> dict:
    team = task["masked_team"]
    paths = [jev_fill.parse_path(p) for p in task["masked_paths"]]
    names = {"item": "item", "ability": "ability", "moves": "move", "nature": "Stat Alignment",
             "statPoints": "Stat Points", "species": "Pokémon"}
    asked = [f"{jev_fill.path_text(p)}: {team[p[0]]['species']}'s {names[p[1]]}" for p in paths]
    parts = [
        ("rules", "Format and rules", f"Format: {jev_fill.FORMAT_NAME}\nRules: {jev_fill.RULES}"),
        ("team", "The team so far", f"Team so far ({MARKER} marks a hidden field):\n{masked_paste(team, paths)}"),
        ("fields", "The fields to fill", "Fill:\n" + "\n".join(asked)),
        ("question", "The question", f"Write the whole team as Showdown paste text with every {MARKER} filled."),
        ("ask", "The win-rate ask", ask_line(EXAMPLE_ASK)),
    ]
    answer = jev_fill.team_paste(task["original"]).strip()
    return {"parts": parts, "answer": answer, "asked": asked}


def counts(repo: str, revision: str | None, system: str, parts: list, answer: str) -> dict | None:
    try:
        tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    except Exception as error:                      # never guess a count
        print(f"{repo}: {type(error).__name__}: {str(error)[:160]}")
        return None
    n = lambda text: len(tok(text, add_special_tokens=False).input_ids)
    user = "\n\n".join(text for _, _, text in parts)
    per = {key: n(text) for key, _, text in parts}
    chat = bool(tok.chat_template)
    if chat:
        whole = len(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            add_generation_prompt=True, tokenize=True))
    else:
        whole = n(system + "\n\n" + user + "\n\n")
    per["wrapper"] = whole - sum(per.values())          # system line, chat markers, joins
    per["answer"] = n(answer) + 1                       # + end-of-text
    per["_chat"] = chat
    return per


def main() -> None:
    task = next(t for t in jev_fill.tasks("meta") if t["id"] == TASK_ID)
    p = draft_prompt(task)
    tokenizers = []
    for key, label, repo, revision in TOKENIZERS:
        c = counts(repo, revision, SYSTEM, p["parts"], p["answer"])
        tokenizers.append({"key": key, "label": label, "repo": repo, "counts": c, "chat": bool(c and c.pop("_chat"))})
        print(key, "not cached" if c is None else f"{sum(c.values())} tokens: {c}")
    hidden = [jev_fill.parse_path(x) for x in task["masked_paths"]]
    data = {
        "mask": {"source": task["source"], "paths": task["masked_paths"], "grid": "meta"},
        "system": SYSTEM, "marker": MARKER, "ask": EXAMPLE_ASK, "limit": LIMIT, "jev_tokens": JEV_TOKENS,
        "parts": [{"key": k, "name": name, "text": text} for k, name, text in p["parts"]],
        "answer": p["answer"],
        "team": [{"species": m["species"], "item": m["item"],
                  "hidden": [h[1] for h in hidden if h[0] == i]} for i, m in enumerate(task["masked_team"])],
        "tokenizers": tokenizers,
    }
    page = JEV_PAGE.read_text()
    assets = {}
    for name in ("SP", "IT"):
        i = page.find(f"const {name}=", page.find("const REQ="))
        assets[name] = json.loads(page[i + len(name) + 7:page.find(";\n", i)])
    species = {m["species"] for m in task["original"]}
    items = {m["item"] for m in task["original"]}
    assets = {"SP": {k: v for k, v in assets["SP"].items() if k in species},
              "IT": {k: v for k, v in assets["IT"].items() if k in items}}
    i = page.find("const JV_CSS = ")
    css = json.loads(page[i + len("const JV_CSS = "):page.find(";\n", i)])
    template = (HERE / "prompt-anatomy.template.html").read_text()
    out = (template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
           .replace("/*ASSETS*/null", json.dumps(assets))
           .replace("/*JV_CSS*/", css))
    (HERE / "prompt-anatomy.html").write_text(out)
    (HERE / "prompt-anatomy.data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print("wrote", HERE / "prompt-anatomy.html")


if __name__ == "__main__":
    main()
