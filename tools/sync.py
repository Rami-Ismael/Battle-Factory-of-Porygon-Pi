#!/usr/bin/env python3
"""Mirror the local sources of truth into this repo, then commit and push.

    Obsidian vault project folder  ->  notes/
    code working directory         ->  code/

One-way. The laptop is the source of truth. Only notes/ and code/ are staged, so folders
written in this repo (blog/) are never published by a sync. Which files travel is decided by each
source's own git ignore rules (`git ls-files --cached --others
--exclude-standard`), so ignored files (.obsidian, .venv, large binaries,
results) never leave the machine.

A sync does not silently undo a pull request merged on GitHub. Before copying
anything, it fetches origin/main, even with --dry-run. The newest commit there
whose subject starts with "sync from laptop" is the baseline. If a notes/ or
code/ file changed on GitHub after the baseline and the laptop copy differs,
mirroring would overwrite, delete, or re-add GitHub's version. The sync prints
those files instead and exits with status 1. Make the laptop match GitHub, or
keep the laptop's version with --allow FILE or --force. A GitHub change that
lands while the sync runs stops it before the push, whatever the flags, and the
next run pushes the commit it held back. Each
pushed sync commit becomes the next baseline.

    tools/sync.py                 mirror, commit, push
    tools/sync.py --dry-run       show what would change; write nothing but the fetch
    tools/sync.py --no-push       mirror and commit only
    tools/sync.py --allow FILE    also undo the GitHub edits to the paths listed in FILE
    tools/sync.py --force         also undo every GitHub edit
"""
import argparse, filecmp, hashlib, os, re, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = {
    "notes": Path(os.environ.get("BF_VAULT", Path.home() / "Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team")),
    "code": Path(os.environ.get("BF_CODE", Path.home() / "Documents/code/vgc-team-generator-pilot")),
}
SECRET = re.compile(rb"(gh[opsu]_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
UPSTREAM = "origin/main"
SYNC_SUBJECT = "sync from laptop"  # subject prefix of every sync commit; the newest on UPSTREAM is the baseline
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree: before the first sync, every GitHub file counts as an edit


def run(*cmd, cwd=REPO, **kw):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True, **kw).stdout


def source_files(src: Path) -> set[str]:
    out = run("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", cwd=src)
    return {p for p in out.split("\0") if p and (src / p).is_file()}


def blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def rows(undo: list[tuple[str, str]]) -> str:
    return "".join(f"  {act:9}  {path}\n" for act, path in undo)


def baseline() -> str | None:
    """Fetch UPSTREAM and return its newest sync commit, or None before the first sync."""
    try:
        run("git", "fetch", "-q", "origin", "main")
    except subprocess.CalledProcessError as e:
        sys.exit(f"cannot fetch {UPSTREAM} to check for GitHub edits: {e.stderr.strip()}")
    log = (line.partition(" ") for line in run("git", "log", "--format=%H %s", UPSTREAM).splitlines())
    return next((sha for sha, _, subject in log if subject.startswith(SYNC_SUBJECT)), None)


def github_edits(wants: dict[str, set[str]], since: str) -> list[tuple[str, str]]:
    """GitHub edits that mirroring would undo.

    One (action, path) pair for each notes/ or code/ file that changed on UPSTREAM
    after the commit `since` and differs on the laptop. The action is what
    mirroring would do to GitHub's copy: overwrite, delete, or re-add.
    """
    touched = run("git", "diff", "-z", "--name-only", "--no-renames", since, UPSTREAM, "--", *SOURCES)
    github = {}
    for entry in filter(None, run("git", "ls-tree", "-r", "-z", UPSTREAM, "--", *SOURCES).split("\0")):
        meta, path = entry.split("\t", 1)
        github[path] = meta.split()[2]
    undo = []
    for path in filter(None, touched.split("\0")):
        name, rel = path.split("/", 1)
        laptop = SOURCES[name] / rel if rel in wants[name] else None
        if laptop and path in github and blob_id(laptop) != github[path]:
            undo.append(("overwrite", path))
        elif not laptop and path in github:
            undo.append(("delete", path))
        elif laptop and path not in github:
            undo.append(("re-add", path))
    return undo


def mirror(name: str, src: Path, dry: bool, want: set[str] | None = None) -> dict:
    dst = REPO / name
    want = source_files(src) if want is None else want
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
    ap.add_argument("--allow", type=Path, metavar="FILE", help="file of repo paths (code/README.md), one per line, whose GitHub edits the sync may undo")
    ap.add_argument("--force", action="store_true", help="undo every GitHub edit made since the last sync")
    a = ap.parse_args()
    if a.allow and not a.allow.is_file():
        ap.error(f"--allow: no such file: {a.allow}")

    for name, src in SOURCES.items():
        if not src.exists():
            sys.exit(f"{name}: source folder missing: {src}")
    wants = {name: source_files(src) for name, src in SOURCES.items()}
    base = baseline()
    checked = run("git", "rev-parse", UPSTREAM).strip()
    undo = github_edits(wants, base or EMPTY_TREE)
    allowed = {line.strip() for line in a.allow.read_text().splitlines()} if a.allow else set()
    blocked = [(act, p) for act, p in undo if p not in allowed]
    since = f"since the last laptop sync ({base[:8]})" if base else "before any laptop sync"
    if blocked and not a.force:
        more = f" ({len(undo) - len(blocked)} more are allowed by {a.allow})" if len(blocked) < len(undo) else ""
        sys.exit(f"GitHub changed these files {since}, and the laptop's copy differs."
                 f" Syncing would do this to GitHub's copy:\n{rows(blocked)}"
                 f"refusing to sync: it would undo {len(blocked)} GitHub edits{more}. Make the laptop match"
                 " GitHub for each file, or keep the laptop's version with --force, or with --allow FILE where FILE"
                 " lists the paths.")
    if undo:
        print(f"{'would undo' if a.dry_run else 'undoing'} {len(undo)} GitHub edits made {since}:\n{rows(undo)}", end="")
    if not a.dry_run:
        # Build the sync commit on the GitHub commit just checked. On a stale checkout, the closing
        # pull --rebase would merge each allowed file with GitHub's edit instead of replacing it.
        run("git", "rebase", "--autostash", "-q", checked)

    parts, total = [], 0
    for name, src in SOURCES.items():
        r = mirror(name, src, a.dry_run, wants[name])
        n = sum(map(len, r.values()))
        total += n
        if n:
            parts.append(f"{name}: +{len(r['added'])} ~{len(r['changed'])} -{len(r['removed'])}")
            for k, sign in (("added", "+"), ("changed", "~"), ("removed", "-")):
                for rel in r[k][:8]:
                    print(f"  {sign} {name}/{rel}")
                if len(r[k]) > 8:
                    print(f"  {sign} ... {len(r[k]) - 8} more {k}")
    # A sync refused before its push leaves its commit here, so a rerun with nothing new still pushes it.
    unpushed = int(run("git", "rev-list", "--count", f"{checked}..HEAD"))
    held = [f"{unpushed} earlier commit{'s' * (unpushed > 1)} not yet on GitHub"] if unpushed else []
    if not total and not unpushed:
        return print("already in sync")
    if a.dry_run:
        return print("dry run:", "; ".join(parts + held))

    run("git", "add", "-A", "--", *SOURCES)
    if run("git", "diff", "--cached", "--name-only").strip():
        run("git", "commit", "-q", "-m", f"{SYNC_SUBJECT} (" + "; ".join(parts) + ")")
        print("committed:", "; ".join(parts))
    elif not unpushed:
        return print("already in sync")
    else:
        print("nothing new to commit;", held[0])
    if not a.no_push:
        run("git", "pull", "--rebase", "--autostash", "-q")
        # A pull request merged while this sync ran now sits under the sync commit,
        # which would become the baseline and hide it from every later check.
        late = github_edits(wants, checked)
        if late:
            sys.exit(f"GitHub changed these files while this sync ran, and the laptop's copy differs:\n{rows(late)}"
                     "refusing to push, so the next sync cannot undo them unchecked. The sync commit stays local."
                     " Rerun tools/sync.py to check these files; it pushes the commit once they pass.")
        run("git", "push", "-q")
        print("pushed")


if __name__ == "__main__":
    main()
