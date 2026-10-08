#!/usr/bin/env python3
"""Mirror the local sources of truth into this repo, then commit and push.

    Obsidian vault project folder  ->  notes/
    code working directory         ->  code/

One-way. The laptop is the source of truth; edits made on github.com are
overwritten by the next sync. Which files travel is decided by each
source's own git ignore rules (`git ls-files --cached --others
--exclude-standard`), so ignored files (.obsidian, .venv, large binaries,
results) never leave the machine.

    tools/sync.py              mirror, commit, push
    tools/sync.py --dry-run    show what would change, touch nothing
    tools/sync.py --no-push    mirror and commit only
"""
import argparse, filecmp, os, re, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = {
    "notes": Path(os.environ.get("BF_VAULT", Path.home() / "Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team")),
    "code": Path(os.environ.get("BF_CODE", Path.home() / "Documents/code/vgc-team-generator-pilot")),
}
SECRET = re.compile(rb"(gh[opsu]_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def run(*cmd, cwd=REPO, **kw):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True, **kw).stdout


def source_files(src: Path) -> set[str]:
    out = run("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", cwd=src)
    return {p for p in out.split("\0") if p and (src / p).is_file()}


def mirror(name: str, src: Path, dry: bool) -> dict:
    dst = REPO / name
    want = source_files(src)
    have = {str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()} if dst.exists() else set()
    added, changed = [], []
    for rel in sorted(want):
        s, d = src / rel, dst / rel
        if not d.exists():
            added.append(rel)
        elif not filecmp.cmp(s, d, shallow=False):
            changed.append(rel)
        else:
            continue
        if SECRET.search(s.read_bytes()) if s.stat().st_size < 2_000_000 else False:
            sys.exit(f"refusing to publish {name}/{rel}: looks like it contains a secret")
        if not dry:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    removed = sorted(have - want)
    if not dry:
        for rel in removed:
            (dst / rel).unlink()
        for d in sorted((p for p in dst.rglob("*") if p.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    return {"added": added, "changed": changed, "removed": removed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    parts, total = [], 0
    for name, src in SOURCES.items():
        if not src.exists():
            sys.exit(f"{name}: source folder missing: {src}")
        r = mirror(name, src, a.dry_run)
        n = sum(map(len, r.values()))
        total += n
        if n:
            parts.append(f"{name}: +{len(r['added'])} ~{len(r['changed'])} -{len(r['removed'])}")
            for k, sign in (("added", "+"), ("changed", "~"), ("removed", "-")):
                for rel in r[k][:8]:
                    print(f"  {sign} {name}/{rel}")
                if len(r[k]) > 8:
                    print(f"  {sign} ... {len(r[k]) - 8} more {k}")
    if not total:
        return print("already in sync")
    if a.dry_run:
        return print("dry run:", "; ".join(parts))

    run("git", "add", "-A")
    if not run("git", "status", "--porcelain").strip():
        return print("already in sync")
    run("git", "commit", "-q", "-m", "sync from laptop (" + "; ".join(parts) + ")")
    print("committed:", "; ".join(parts))
    if not a.no_push:
        run("git", "pull", "--rebase", "--autostash", "-q")
        run("git", "push", "-q")
        print("pushed")


if __name__ == "__main__":
    main()
