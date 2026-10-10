#!/usr/bin/env python3
"""Build sprites.json (species id -> PNG data URI) and sprites-missing.txt
for the diversity-methods page. Reads results and sprite PNGs; writes only
into this script's folder."""
import base64
import json
import re
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent
VISUALS = HERE.parent
PRIMARY = VISUALS / "search-loop/sprites"
OUT = HERE
RESULTS = HOME / "Documents/code/vgc-team-generator-pilot/results"
MAX_BYTES = 60 * 1024
EXTRA_IDS = ["sinistcha", "milotic", "incineroar", "raichu", "mimikyu", "ninetalesalola"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def species_from_paste(paste):
    ids = set()
    for block in re.split(r"\n\s*\n", paste.strip()):
        block = block.strip()
        if not block:
            continue
        first = block.splitlines()[0].strip()
        name = first.split(" @ ")[0].strip()
        name = re.sub(r"\s\((M|F)\)$", "", name)
        m = re.search(r"\(([^()]+)\)$", name)
        if m and len(m.group(1).strip()) >= 3:
            name = m.group(1).strip()
        sid = norm(name)
        if sid:
            ids.add(sid)
    return ids


def paste_of(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and isinstance(x.get("paste"), str):
        return x["paste"]
    return None


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def collect_ids():
    ids = set(EXTRA_IDS)

    diverse = load("diversity_methods_all.json")
    for entry in diverse["gens"].values():
        for comp in entry.get("stream_comps", []) or []:
            for sp in comp:
                ids.add(norm(sp))
        for item in entry.get("battled", []) or []:
            p = paste_of(item)
            if p:
                ids |= species_from_paste(p)
        for item in entry.get("_isl", []) or []:
            p = paste_of(item.get("seed", {}))
            if p:
                ids |= species_from_paste(p)

    cem = load("cem_v3.json")
    g0 = cem["gens"]["0"]
    for item in list(g0.get("proposals", []) or []) + list(g0.get("battled", []) or []):
        p = paste_of(item)
        if p:
            ids |= species_from_paste(p)

    for item in load("confirm_top.json"):
        p = paste_of(item)
        if p:
            ids |= species_from_paste(p)

    ids.discard("")
    return ids


def build_index():
    index = {}
    seen = set()
    for root in (VISUALS,):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.png")):
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            if path.name.lower().startswith("item-"):
                continue
            if path.stat().st_size > MAX_BYTES:
                continue
            index.setdefault(norm(path.stem), []).append(path)
    return index


def find_png(sid, index):
    primary = PRIMARY / f"{sid}.png"
    if primary.is_file() and primary.stat().st_size <= MAX_BYTES and not primary.name.lower().startswith("item-"):
        return [primary]
    return index.get(sid, [])


def main():
    ids = collect_ids()
    index = build_index()

    sprites = {}
    missing = []
    ambiguous = []
    suspicious = []
    for sid in sorted(ids):
        matches = find_png(sid, index)
        if not matches:
            missing.append(sid)
            continue
        if len(matches) > 1:
            ambiguous.append((sid, [str(m) for m in matches]))
        path = matches[0]
        data = path.read_bytes()
        if len(data) == 0 or len(data) != path.stat().st_size:
            suspicious.append(str(path))
        if not data.startswith(b"\x89PNG"):
            suspicious.append(f"not PNG: {path}")
        sprites[sid] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    OUT.mkdir(parents=True, exist_ok=True)
    sprites_path = OUT / "sprites.json"
    sprites_path.write_text(json.dumps(sprites, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    (OUT / "sprites-missing.txt").write_text("".join(f"{m}\n" for m in missing), encoding="utf-8")

    size_kb = sprites_path.stat().st_size / 1024
    print(f"ids collected: {len(ids)}")
    print(f"ids found: {len(sprites)}")
    print(f"sprites.json: {size_kb:.1f} KB")
    print(f"missing ({len(missing)}): {', '.join(missing) if missing else '(none)'}")
    if ambiguous:
        print(f"ambiguous matches ({len(ambiguous)}):")
        for sid, paths in ambiguous:
            print(f"  {sid}: {paths}")
    if suspicious:
        print("warnings:")
        for s in suspicious:
            print(f"  {s}")


if __name__ == "__main__":
    main()
