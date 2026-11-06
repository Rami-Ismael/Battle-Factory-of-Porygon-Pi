"""Build the remasking-failure page: data.json (from build_data.py) + template.html -> index.html.

    python3 build.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "data.json").read_text())
html = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_text(html)
print(f"index.html: {len(html):,} bytes")
