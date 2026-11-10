#!/usr/bin/env python3
"""Re-download every species sprite in ../sprites (and every model-vocabulary species) from PokeAPI.

Each Showdown id is resolved through ../sprites/pokeapi_ids.json. The default 96x96 front sprite is
tried first; on HTTP 404 the PokeAPI HOME render is used and recorded as a fallback. Downloads land in
the empty directory given on the command line and are checked (PNG signature, IHDR size) before being
copied over <id>.png in the store, so file names and keys stay the same. A species with no PokeAPI
sprite keeps its old file and is listed as unresolved. Rewrites ../sprites/sources.json and
sprites.json (the model's species -> file map).

Usage: python3 fetch_sprites.py <download-dir>   (the directory must exist and be empty)
"""
import hashlib
import json
import shutil
import struct
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "sprites"
sys.dont_write_bytecode = True
sys.path.insert(0, str(STORE))
from pokeapi_sprites import entry, pokeapi_url  # noqa: E402

HOST = "raw.githubusercontent.com"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_BYTES = 2 * 1024 * 1024
RETRIEVED = date.today().isoformat()


def fetch(url):
    """Return (body, None) when every check passes, otherwise (None, reason)."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            final_host = urlsplit(resp.geturl()).hostname
            body = resp.read(MAX_BYTES + 1)
    except HTTPError as e:
        return None, f"HTTP {e.code}"
    except OSError as e:
        return None, f"network error: {e}"
    if final_host != HOST:
        return None, f"redirected to host {final_host}"
    if len(body) > MAX_BYTES:
        return None, f"larger than {MAX_BYTES} bytes"
    if len(body) < 33 or not body.startswith(PNG_SIGNATURE) or body[12:16] != b"IHDR":
        return None, "not a PNG"
    return body, None


def png_info(body):
    width, height, _depth, color_type = struct.unpack(">IIBB", body[16:26])
    alpha = color_type in (4, 6) or b"tRNS" in body
    return width, height, alpha


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 fetch_sprites.py <download-dir>")
    dl = Path(sys.argv[1])
    if not dl.is_dir() or any(dl.iterdir()):
        sys.exit(f"download directory must exist and be empty: {dl}")

    manifest = json.loads((ROOT / "model/manifest.json").read_text(encoding="utf-8"))
    vocab = manifest["vocab"]["species"][1:]
    ids = sorted({p.stem for p in STORE.glob("*.png")} | set(vocab))

    rows, unresolved = {}, {}
    for i, sid in enumerate(ids):
        if i:
            time.sleep(0.05)
        try:
            entry(sid)
        except KeyError:
            unresolved[sid] = "no entry in pokeapi_ids.json"
            continue
        tried = []
        for kind in ("default", "home"):
            url = pokeapi_url(sid, kind)
            body, reason = fetch(url)
            if body is not None:
                break
            tried.append(f"{url}: {reason}")
            if reason != "HTTP 404":
                break
        if body is None:
            unresolved[sid] = "; ".join(tried)
            continue
        with open(dl / f"{sid}.png", "xb") as out:
            out.write(body)
        width, height, alpha = png_info(body)
        rows[sid] = {"file": f"{sid}.png", "source": url, "pokeapi_name": entry(sid)["name"],
                     "sha256": hashlib.sha256(body).hexdigest(), "width": width, "height": height,
                     "alpha": alpha, "retrieved": RETRIEVED}
        if kind != "default":
            rows[sid]["fallback"] = kind
            rows[sid]["default_missing"] = tried[0]

    for sid in rows:
        shutil.copyfile(dl / f"{sid}.png", STORE / f"{sid}.png")

    sources_path = STORE / "sources.json"
    old_assets = {a["file"]: a for a in json.loads(sources_path.read_text(encoding="utf-8"))["assets"]}
    assets = [rows[sid] for sid in sorted(rows)]
    # A file PokeAPI could not replace keeps its old record (or gets one saying where it came from).
    for sid in sorted(unresolved):
        if (STORE / f"{sid}.png").exists():
            kept = dict(old_assets.get(f"{sid}.png", {"file": f"{sid}.png", "source": "unknown (pre-PokeAPI)"}))
            kept["unresolved"] = unresolved[sid]
            assets.append(kept)
    sources_path.write_text(json.dumps({
        "retrieved": RETRIEVED,
        "collection": "PokeAPI sprites repository (https://github.com/PokeAPI/sprites): default front "
                      "sprites (sprites/pokemon/<id>.png); 'fallback': 'home' marks a PokeAPI HOME render "
                      "used where no default sprite exists. Showdown id -> PokeAPI id: pokeapi_ids.json.",
        "assets": assets,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    model_rows = []
    for sid in vocab:
        if sid in rows:
            r = rows[sid]
            model_rows.append({"species": sid, "file": "./sprites/" + r["file"], "source": r["source"],
                               "available": True, "width": r["width"], "height": r["height"],
                               "sha256": r["sha256"], **({"fallback": r["fallback"]} if "fallback" in r else {})})
        else:
            model_rows.append({"species": sid, "source": pokeapi_url(sid) if sid not in unresolved or
                               "no entry" not in unresolved[sid] else None, "available": False,
                               "reason": unresolved.get(sid)})
    (ROOT / "model/sprites.json").write_text(json.dumps({
        "retrieved": RETRIEVED,
        "sprites": {x["species"]: x["file"] for x in model_rows if x["available"]},
        "sources": model_rows,
    }, indent=2) + "\n", encoding="utf-8")

    fallback = sorted(s for s, r in rows.items() if "fallback" in r)
    print(f"default: {len(rows) - len(fallback)}; fallback: {len(fallback)} {fallback}")
    print(f"unresolved: {len(unresolved)}")
    for sid, why in unresolved.items():
        print(f"  {sid}: {why}")


if __name__ == "__main__":
    main()
