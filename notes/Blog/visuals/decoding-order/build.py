"""Build the decoding-order results page: data.json (from build_data.py), remask.json (from build_remask_data.py),
lookahead.json (from build_lookahead_data.py) + template.html -> index.html.

    python3 build.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "data.json").read_text())
remask = json.loads((HERE / "remask.json").read_text())
lookahead = json.loads((HERE / "lookahead.json").read_text())
data["sprites"].update(remask.pop("sprites"))
data["items"].update(remask.pop("items"))
for k, v in remask.pop("names").items(): data["names"].setdefault(k, v)
if any(remask.pop("missing").values()): raise SystemExit("remask.json lists missing sprites; rebuild it first")
data["remask"] = remask
for k in ("sprites", "items"): data[k].update(lookahead.pop(k))
for k, v in lookahead.pop("names").items(): data["names"].setdefault(k, v)
if any(lookahead.pop("missing").values()): raise SystemExit("lookahead.json lists missing sprites; rebuild it first")
data["lookahead"] = lookahead
html = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_text(html)
print(f"index.html: {len(html):,} bytes")
