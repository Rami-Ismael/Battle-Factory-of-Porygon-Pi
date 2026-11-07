"""Build prompt-anatomy.html: the in-place text a masked diffusion model fills (#196, #199, #200), part by part,
in the look of the "What Jev is sent" figure on the Jev picks a team page.

    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/count_masked_diffusion_text.py
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python artifacts/masked-diffusion-prompt/build_prompt_visual.py

The text comes from the real builder, code/scripts/masked_diffusion_text.py, for the Jev page's example mask:
meta grid team MB158 with Alolan Ninetales's item hidden. Each candidate of #162 counts it with its own
tokenizer at its pinned revision, loaded by count_masked_diffusion_text.load_tokenizer; a tokenizer that
fails to load is listed as not counted, never guessed. The longest mask on any grid comes from the count
command's mask-counts.json, so run that first. The template reads only DATA and ASSETS.

The free-text draft this page first showed (prompt plus answer, 872 tokens with Masked Qwen3 0.6B) was
replaced on 2026-11-06 by in-place filling. DRAFT_TOTALS keeps its totals from #196 for comparison.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))
import count_masked_diffusion_text as C
import jev_fill
import masked_diffusion_text as D
from mask_paste import json_pieces, value_text

JEV_PAGE = ROOT / "notes/Blog/visuals/jev-picks-a-team/jev-picks-a-team.html"
TASK_ID = "b90b461c39429bf2"          # meta grid, unmasked team MB158, slot 4's item hidden: the Jev page's example
EXAMPLE_ASK = 0.55                    # an example; a real row carries its own label or "unknown"
JEV_TOKENS = 24_644                   # the same mask's Jev request, counted by Jev's server on 2026-10-27
DRAFT_TOTALS = {                      # the free-text draft's prompt plus answer for this mask, from #196's table
    "qwen3": 872, "qwen25c": 872, "diffugpt_s": 763, "diffugpt_m": 763, "diffullama": 985,
    "dream": 872, "llada": 928,       # Dream's draft row used Qwen2.5-7B's tokenizer
}
COUNTS = HERE / "mask-counts.json"


def segments(built: D.InPlaceText, filled: dict[str, str] | None = None) -> list[list[str]]:
    """The text cut into [text, kind] runs: rules, ask, team, and slot (or value, once filled)."""
    parts = D.header_parts(EXAMPLE_ASK)
    out = [[parts["format and rules"], "rules"], [parts["ask"], "ask"]]
    at = len(parts["format and rules"]) + len(parts["ask"])
    for slot in built.slots:
        out.append([built.text[at:slot.start], "team"])
        out.append([filled[slot.path], "value"] if filled else [built.text[slot.start:slot.end], "slot"])
        at = slot.end
    out.append([built.text[at:], "team"])
    return [s for s in out if s[0]]


def tokenizer_data(candidate: C.Candidate, task: dict, largest: dict | None) -> dict:
    head = {"key": candidate.key, "label": candidate.label, "repo": candidate.repo, "revision": candidate.revision,
            "draft": DRAFT_TOTALS[candidate.key]}
    try:
        tokenizer = C.load_tokenizer(candidate)
    except Exception as error:                      # never guess a count
        print(f"{candidate.repo}: not counted: {type(error).__name__}: {str(error)[:160]}")
        return {**head, "counts": None, "reason": f"{type(error).__name__}: {error}"[:300]}
    built = D.in_place_text(task, EXAMPLE_ASK, tokenizer, C.LIMIT)
    masks = sum(s.length for s in built.slots)
    real = {D.value_label(p[0]): value_text(p[1]) for p in json_pieces(task["original"]) if not isinstance(p, str)}
    counts = {"rules": built.part_tokens["format and rules"], "ask": built.part_tokens["ask"],
              "team": built.part_tokens["team"] - masks, "masks": masks}
    return {**head, "counts": counts, "total": sum(built.part_tokens.values()), "mask_token": tokenizer.mask_token,
            "slot_lengths": D.slot_lengths(tokenizer), "slots": [[s.path, s.length] for s in built.slots],
            "text": segments(built), "filled": segments(built, {s.path: real[s.path] for s in built.slots}),
            "largest": largest}


def main() -> None:
    task = next(t for t in jev_fill.tasks("meta") if t["id"] == TASK_ID)
    report = json.loads(COUNTS.read_text())
    largest = {c["key"]: c.get("largest") and {k: c["largest"][k] for k in ("tokens", "name", "ties")} | {
        "over": c["over_limit"], "masks": c["masks"]} for c in report["candidates"]}
    tokenizers = []
    for candidate in C.CANDIDATES:
        t = tokenizer_data(candidate, task, largest.get(candidate.key))
        tokenizers.append(t)
        print(candidate.key, "not counted" if t["counts"] is None else f"{t['total']} tokens: {t['counts']}")
    hidden = [jev_fill.parse_path(x) for x in task["masked_paths"]]
    data = {
        "mask": {"source": task["source"], "paths": task["masked_paths"], "grid": "meta"},
        "version": D.VERSION, "ask": EXAMPLE_ASK, "limit": C.LIMIT, "jev_tokens": JEV_TOKENS,
        "all_masks": report["masks"],
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
