"""PROTOTYPE data for bo-loop-prototype: reuse the numbers and sprites already inlined in
../bo-features/bo-features.html (built from the pilot repo's counter_matrix.json, ruggedness2.json,
multimodality.json). No battles are run here. Writes data.js."""
import json
from pathlib import Path

HERE = Path(__file__).parent
h = (HERE.parent / "bo-features" / "bo-features.html").read_text()
i = h.index("const D = ") + len("const D = ")
D, _ = json.JSONDecoder().raw_decode(h[i:])

teams = sorted(D["teams"].values(), key=lambda t: (t["kind"] != "meta", t["id"]))
edits = {op: {s: {k: v[k] for k in ("rho", "ci", "beyond", "repeat", "n")} for s, v in by.items()}
         for op, by in D["edits"].items()}
used = {s for t in teams for s in t["species"]}
sid = lambda n: "".join(c for c in n.lower() if c.isalnum())
sprites = {s: D["sprites"][sid(s)] for s in sorted(used)}

out = {"teams": [{"id": t["id"], "kind": t["kind"], "p": round(t["p"], 4), "n": t["n"], "species": t["species"]} for t in teams],
       "edits": edits, "sprites": sprites}
(HERE / "data.js").write_text("window.BO_LOOP = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("wrote data.js:", len(out["teams"]), "teams,", len(sprites), "sprites")
