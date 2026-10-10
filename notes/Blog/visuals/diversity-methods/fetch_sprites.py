#!/usr/bin/env python3
"""Fetch the Pokemon Showdown gen5 front sprites that data.json needs but sprites.json lacks.

Every download lands first in the download directory given on the command line. A file is
copied into the search-loop sprites folder only when no file of that name is already there, and
each copy gets an entry in that folder's sources.json. Downloaded bytes are never opened,
executed or imported.

Usage: python3 fetch_sprites.py <download-dir>   (the directory must exist and be empty)
"""
import hashlib
import json
import re
import shutil
import struct
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
SPR = HERE.parent / "search-loop" / "sprites"
URL = "https://play.pokemonshowdown.com/sprites/gen5/{name}.png"
HOST = "play.pokemonshowdown.com"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_BYTES = 61440
SIZE = 96
RETRIEVED = "2026-10-10"
SPECIAL_NAMES = {"raichumegay": "raichu-megay"}


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s)


def showdown_name(sid, display):
    if sid in SPECIAL_NAMES:
        return SPECIAL_NAMES[sid]
    lower = display.lower()
    if "-" in lower:
        base, forme = lower.split("-", 1)
        return f"{alnum(base)}-{alnum(forme)}"
    return alnum(lower)


def fetch(url):
    """Return (body, None) when every check passes, otherwise (None, reason)."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:
            status = resp.status
            final_host = urlsplit(resp.geturl()).hostname
            body = resp.read(MAX_BYTES + 1)
    except HTTPError as e:
        return None, f"HTTP {e.code}"
    except OSError as e:
        return None, f"network error: {e}"
    if status != 200:
        return None, f"HTTP {status}"
    if final_host != HOST:
        return None, f"redirected to host {final_host}"
    if not body.startswith(PNG_SIGNATURE):
        return None, "not a PNG (bad signature)"
    if len(body) > MAX_BYTES:
        return None, f"larger than {MAX_BYTES} bytes"
    if len(body) < 24:
        return None, "too short to hold an IHDR chunk"
    width, height = struct.unpack(">II", body[16:24])
    if (width, height) != (SIZE, SIZE):
        return None, f"IHDR is {width}x{height}, not {SIZE}x{SIZE}"
    return body, None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 fetch_sprites.py <download-dir>")
    dl = Path(sys.argv[1])
    if not dl.is_dir() or any(dl.iterdir()):
        sys.exit(f"download directory must exist and be empty: {dl}")

    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    sprites = json.loads((HERE / "sprites.json").read_text(encoding="utf-8"))
    missing = [sid for sid in data["species"] if sid not in sprites]
    print(f"missing: {len(missing)}")

    downloaded = []  # (id, url, bytes)
    failed = []      # (id, url, reason)
    for i, sid in enumerate(missing):
        if i:
            time.sleep(0.2)
        url = URL.format(name=showdown_name(sid, data["names"][sid]))
        body, reason = fetch(url)
        if reason is not None:
            failed.append((sid, url, reason))
            continue
        with open(dl / f"{sid}.png", "xb") as out:
            out.write(body)
        downloaded.append((sid, url, body))

    sources_path = SPR / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    copied, already = [], []
    for sid, url, body in downloaded:
        dst = SPR / f"{sid}.png"
        if dst.exists():
            already.append(sid)
            continue
        with open(dl / f"{sid}.png", "rb") as src, open(dst, "xb") as out:
            shutil.copyfileobj(src, out)
        sources["assets"].append({
            "file": f"{sid}.png",
            "source": url,
            "sha256": hashlib.sha256(body).hexdigest(),
            "width": SIZE,
            "height": SIZE,
            "retrieved": RETRIEVED,
        })
        copied.append(sid)
    if copied:
        sources_path.write_text(json.dumps(sources, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    total = sum(len(body) for _, _, body in downloaded)
    print(f"downloaded: {len(downloaded)}")
    print(f"total bytes downloaded: {total}")
    print(f"copied into {SPR}: {len(copied)}; already present, not copied: {len(already)}")
    print(f"failed: {len(failed)}")
    for sid, url, reason in failed:
        print(f"  {sid} ({url}): {reason}")


if __name__ == "__main__":
    main()
