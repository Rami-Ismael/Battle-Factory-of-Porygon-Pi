"""Data for the ask-frequencies figure (2026-10-30): where the win-rate condition's frequencies come from, and what
they do to asks between training-label steps. Writes data.json beside this file.

    /tmp/vgc-pilot/.venv/bin/python build_data.py     # run from anywhere; needs the code checkout and ~/vgc-data

Sources:
- the frequencies and ask embeddings: the deployed members' own Net.freqs and Net.wvec (code/src/lossdown.py line 206);
- training labels: each member's labelled training rows (DiscreteDiffusionModel._original_rows, as scripts/retrain_regmb.py chose them);
- win rates: ~/vgc-data/results/remask_ask_gap.json (plain masking at asks 0.3, 0.5, 0.531) and
  ~/vgc-data/results/remask_why.json (plain masking at the on-grid asks 7/24, 8/24, 13/24).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[3] / "code" / "src"
sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(CODE))
import discrete_diffusion_model

sys.path.insert(0, str(CODE))
import lossdown as L

RES = Path.home() / "vgc-data" / "results"
NAMES = {"F1_final_medium_noheldout_regmb_v3": "loop-teams model", "G2_family_matrix_noprtrain_noheldout_regmb_v3": "battle-records model"}


def main():
    w = discrete_diffusion_model.DiscreteDiffusionModel.load()
    out = {"freqs_over_pi": None, "members": {}, "winrate": {}, "sources": {}}
    grid = np.round(np.arange(0.20, 0.6001, 0.0025), 4)
    for name, net, cfg in w._nets:
        out["freqs_over_pi"] = [float(f / np.pi) for f in net.freqs.cpu()]
        with torch.no_grad():
            v = torch.nn.functional.normalize(net.wvec(torch.tensor(grid, dtype=torch.float32, device=L.DEV)), dim=1)
            ref = torch.nn.functional.normalize(net.wvec(torch.tensor([0.5], device=L.DEV)), dim=1)
        _xo, wo = w._original_rows(name, cfg)
        y = (wo[:, 0] if wo.dim() == 2 else wo).cpu().numpy(); y = y[~np.isnan(y)]
        hist = np.histogram(y, bins=240, range=(0, 1))[0]
        out["members"][NAMES[name]] = {
            "checkpoint": name, "labelled_rows": len(y), "label_hist_1_240": hist.tolist(),
            "share_on_24_grid": float(np.mean(np.isclose((y * 24) % 1, 0, atol=1e-3) | np.isclose((y * 24) % 1, 1, atol=1e-3))),
            "rows_near": {f"{a:.4f}": int(np.sum(np.abs(y - a) < 0.005)) for a in (7 / 24, 0.3, 8 / 24, 0.5, 0.531, 13 / 24)},
            "similarity_to_0_5": {"asks": grid.tolist(), "cos": [round(float(c), 4) for c in (v @ ref.T).squeeze(1).cpu()]}}
    gap = json.loads((RES / "remask_ask_gap.json").read_text())["cells"]; why = json.loads((RES / "remask_why.json").read_text())["cells"]
    for ask, src, name in [(0.3, gap, "remask_ask_gap.json"), (0.5, gap, "remask_ask_gap.json"), (0.531, gap, "remask_ask_gap.json"),
                           (7 / 24, why, "remask_why.json"), (8 / 24, why, "remask_why.json"), (13 / 24, why, "remask_why.json")]:
        cs = sorted((c for c in src.values() if c["arm"] == "plain" and abs(c["ask"] - ask) < 1e-9), key=lambda c: c["seed"])
        if [c["seed"] for c in cs] != [1, 2, 3] or any(len(c.get("y") or []) != 64 or None in c["y"] or "win" not in c for c in cs):
            raise SystemExit(f"ask {ask:.4f} in {name}: expected seeds 1-3, each with 64 scored teams")   # the page says 64 x 3
        out["winrate"][f"{ask:.4f}"] = {"ask": ask, "on_grid": bool(abs(ask * 24 - round(ask * 24)) < 1e-6),
                                            "seeds": [round(c["win"]["mean"], 4) for c in cs], "file": name}
    out["sources"] = {"code": "code/src/lossdown.py line 206: freqs = 2**arange(8) * pi; wvec line 215",
                          "winrate": "~/vgc-data/results/remask_ask_gap.json, ~/vgc-data/results/remask_why.json",
                          "battles": "49 per team, one against each distinct top-50 meta team, behaviour-cloning policy both sides, slots shuffled; 64 teams x 3 seeds per ask"}
    (HERE / "data.json").write_text(json.dumps(out))
    m = out["members"]["loop-teams model"]
    print("freqs/pi", out["freqs_over_pi"], "| on 1/24 grid", round(m["share_on_24_grid"], 3), "| near", m["rows_near"])
    print({k: v["seeds"] for k, v in out["winrate"].items()})


if __name__ == "__main__":
    main()
