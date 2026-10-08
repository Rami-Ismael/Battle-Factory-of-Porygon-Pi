"""Build repo-sync.html from template.html (2026-10-08).

    python3 build.py   -> repo-sync.html (publish that file)
Inlines ../repo-sync-prototype/preact-htm-standalone.umd.js so the page is one self-contained file.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
lib = (HERE.parent / "repo-sync-prototype" / "preact-htm-standalone.umd.js").read_text()
out = (HERE / "template.html").read_text().replace("/*PREACT*/", lib)
(HERE / "repo-sync.html").write_text(out)
print("wrote repo-sync.html", len(out) // 1024, "KB")
