"""Build index.html from template.html and the evidence ledger.

The ledger (evidence-ledger.json) and source list (sources.json) are the Codex brief's 21 checked claims and 18 sources, copied unchanged.
Every stone on the board is one of those claims, so the page carries the ledger inline.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ledger = json.loads((HERE / "evidence-ledger.json").read_text())
assert len(ledger["claims"]) == 21, "the board places exactly 21 stones"
html = (HERE / "template.html").read_text()
assert "/*__LEDGER__*/null" in html
html = html.replace("/*__LEDGER__*/null", json.dumps(ledger, ensure_ascii=False).replace("</", "<\\/"))
# sources.json: the Codex brief's 18 source records (title, date, type, summary, locator, limit), used for numbered citations
sources = json.loads((HERE / "sources.json").read_text().replace("\u2019", "'"))
assert "/*__SOURCES__*/null" in html
html = html.replace("/*__SOURCES__*/null", json.dumps(sources, ensure_ascii=False).replace("</", "<\\/"))
(HERE / "index.html").write_text(html)
print("wrote index.html", len(html), "bytes")
