#!/usr/bin/env python3
"""Byte-level check that every Pokémon sprite in the repo comes from PokeAPI.

Run from the repo root:
    python3 .claude/skills/verify-sprites/check_sprites.py --out verify-evidence/sprites/<run-id>
    python3 .claude/skills/verify-sprites/check_sprites.py --offline --out ...   (no network)

Prints one PASS/FAIL line per check, writes <out>/report.json, exits 1 on any FAIL.
Reads only; never changes a tracked file.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import pathlib
import re
import struct
import subprocess
import sys
import urllib.request

SKILL_DIR = pathlib.Path(__file__).resolve().parent
STORE = pathlib.Path("notes/Blog/visuals/search-loop/sprites")
POKEAPI_PREFIX = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/"
LOGO_PORYGON = pathlib.Path("brand/logo/build-video/sprites/porygon.png")
PNG_BASE64 = re.compile(rb"iVBORw0KGgo(?:[A-Za-z0-9+=]|\\?/)+")
SHOWDOWN_SPRITE_URL = "pokemonshowdown.com/sprites"
ALLOWED_SHOWDOWN_FOLDER = "/sprites/trainers/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], capture_output=True, check=True).stdout


def fetch(url: str) -> bytes | str:
    request = urllib.request.Request(url, headers={"User-Agent": "verify-sprites"})
    try:
        return urllib.request.urlopen(request, timeout=30).read()
    except Exception as error:  # noqa: BLE001 - every failure is reported, not raised
        return f"{type(error).__name__}: {error}"[:120]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="compare with recorded hashes, skip PokeAPI downloads")
    parser.add_argument("--out", type=pathlib.Path, help="folder that receives report.json")
    options = parser.parse_args()

    if not STORE.is_dir():
        print(f"FAIL run from the repo root: {STORE} not found")
        return 1

    checks: list[dict] = []

    def record(check_id: str, title: str, failures: list, note: str = "") -> None:
        passed = not failures
        checks.append({"id": check_id, "title": title, "passed": passed, "failures": failures[:50], "note": note})
        tail = f" | {note}" if note else ""
        print(f"{'PASS' if passed else 'FAIL'} {check_id}: {title}{tail}")
        for failure in failures[:8]:
            print(f"     {failure}")

    assets = json.loads((STORE / "sources.json").read_text())["assets"]
    store_files = {path.name: path.read_bytes() for path in sorted(STORE.glob("*.png"))}
    store_hashes = {sha256(data) for data in store_files.values()}
    recorded = {asset["file"]: asset for asset in assets}

    # store-records: every PNG has a record, every record names PokeAPI and matches the file
    failures = [f"{name}: no record in sources.json" for name in store_files if name not in recorded]
    for name, asset in recorded.items():
        if not asset["source"].startswith(POKEAPI_PREFIX):
            failures.append(f"{name}: source is {asset['source']}")
        elif name not in store_files:
            failures.append(f"{name}: recorded but the file is missing")
        elif sha256(store_files[name]) != asset["sha256"]:
            failures.append(f"{name}: file differs from its recorded sha256")
    record("store-records", f"each of {len(store_files)} store sprites has a PokeAPI record that matches it", failures)

    # store-live: the files equal what PokeAPI serves today
    if options.offline:
        record("store-live", "store sprites equal the live PokeAPI files", [], "SKIPPED (--offline)")
    else:
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            downloads = list(pool.map(lambda asset: fetch(asset["source"]), assets))
        failures = []
        for asset, got in zip(assets, downloads):
            if isinstance(got, str):
                failures.append(f"{asset['file']}: {got}")
            elif store_files.get(asset["file"]) != got:
                failures.append(f"{asset['file']}: differs from {asset['source']}")
        record("store-live", f"store sprites equal the live PokeAPI files ({len(assets)} downloads)", failures)

    # store-mapping: the lookup table covers every store sprite
    species = json.loads((STORE / "pokeapi_ids.json").read_text())["species"]
    failures = [f"{name}: no entry in pokeapi_ids.json" for name in store_files if name.removesuffix(".png") not in species]
    record("store-mapping", "pokeapi_ids.json has an entry for every store sprite", failures)

    # embedded-*: scan every tracked file for sprite bytes
    showdown = json.loads((SKILL_DIR / "showdown_sha256.json").read_text())["sha256"]
    showdown_only = {digest: name for digest, name in showdown.items() if digest not in store_hashes}
    allowed_other = {sha256(LOGO_PORYGON.read_bytes())} if LOGO_PORYGON.exists() else set()
    stale: list[str] = []
    unknown: list[str] = []
    blob_count = 0
    for raw_path in git("ls-files", "-z").split(b"\0"):
        path = pathlib.Path(raw_path.decode())
        if not raw_path or not path.is_file() or path.parent == STORE:
            continue
        data = path.read_bytes()
        if path.suffix == ".png":
            if sha256(data) in showdown_only:
                stale.append(f"{path}: copy of Showdown {showdown_only[sha256(data)]}")
            continue
        if b"iVBORw0KGgo" not in data:
            continue
        for blob in set(PNG_BASE64.findall(data)):
            try:
                image = base64.b64decode(blob.replace(b"\\/", b"/"), validate=True)
            except ValueError:
                continue
            blob_count += 1
            digest = sha256(image)
            if digest in showdown_only:
                stale.append(f"{path}: embeds Showdown {showdown_only[digest]}")
            elif struct.unpack(">II", image[16:24]) == (96, 96) and digest not in store_hashes | allowed_other:
                unknown.append(f"{path}: 96x96 image {digest[:12]} is not a store sprite")
    record("embedded-stale", f"no Showdown sprite bytes embedded or copied ({blob_count} embedded images scanned)", sorted(set(stale)))
    record("embedded-known", "every embedded 96x96 image is a store sprite", sorted(set(unknown)))

    # logo-porygon: the brand copy is PokeAPI's Porygon
    porygon_url = POKEAPI_PREFIX + f"{species['porygon']['sprite']}.png" if "porygon" in species else POKEAPI_PREFIX + "137.png"
    if options.offline or not LOGO_PORYGON.exists():
        record("logo-porygon", "logo Porygon equals PokeAPI's", [], "SKIPPED (--offline or file absent)")
    else:
        got = fetch(porygon_url)
        record("logo-porygon", "logo Porygon equals PokeAPI's", [] if got == LOGO_PORYGON.read_bytes() else [f"{LOGO_PORYGON} differs from {porygon_url}"])

    # hotlink-urls: no Showdown Pokémon sprite URL in tracked text
    grep = subprocess.run(["git", "grep", "-n", "-I", "-i", SHOWDOWN_SPRITE_URL], capture_output=True, text=True).stdout
    failures = [line[:160] for line in grep.splitlines() if ALLOWED_SHOWDOWN_FOLDER not in line and ".claude/skills/verify-sprites/" not in line]
    record("hotlink-urls", "no Showdown Pokémon sprite URL in tracked files (trainer sprites excepted)", failures)

    passed = all(check["passed"] for check in checks)
    skipped = [check["id"] for check in checks if check["note"].startswith("SKIPPED")]
    print(f"RESULT {'pass' if passed else 'FAIL'}" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    if options.out:
        options.out.mkdir(parents=True, exist_ok=True)
        head = git("rev-parse", "HEAD").decode().strip()
        report = {"commit": head, "offline": options.offline, "passed": passed, "skipped": skipped, "checks": checks}
        (options.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        print(f"report: {options.out / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
