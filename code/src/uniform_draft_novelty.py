"""Is Uniform's novelty lost to the validator? Novelty of every draft against the drafts the validator keeps.

The ask grids measure novelty only on validator-accepted teams, and Uniform's validator rejects 74-82% of its drafts.
If rejected drafts are far more novel than kept ones, the validator, not the model, removes Uniform's diversity.
Reads draft logs written by `AG84_DRAFTS_LOG=path ask_grid84.py gen` (one JSON line per draft).

    python uniform_draft_novelty.py LABEL=drafts.jsonl [LABEL=drafts.jsonl ...] [--grid-row ROW_PREFIX] [--out x.json]

Pools per log: all drafts; drafts with six distinct species (a repeated species is novel by construction, so this
pool is the fair one); rejected drafts with six distinct species; kept drafts. Distinct line-ups are compared at the
kept pool's size by subsampling the larger pool 200 times.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import grid_novelty as GN
import line_up_yield as LY

RES = Path.home() / "vgc-data/results"


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def pool_measures(lus, train, n_equal, rng):
    if not lus: return dict(teams=0)
    m = GN.measures(lus, train)
    if n_equal and len(lus) > n_equal:
        m["line_ups_at_kept_size"] = float(np.mean([len(set(lus[i] for i in rng.choice(len(lus), n_equal, replace=False))) for _ in range(200)]))
    else:
        m["line_ups_at_kept_size"] = float(len(set(lus)))
    return m


def analyse(rows, train, rng):
    for r in rows: r["lu"] = LY.line_up(r["paste"])
    distinct6 = [r for r in rows if len(set(r["lu"])) == 6]
    kept = [r for r in rows if r["status"] == "kept"]
    n = len(kept)
    pools = {"all drafts": rows, "six distinct species": distinct6,
             "rejected, six distinct species": [r for r in distinct6 if r["status"] == "rejected"], "kept": kept}
    out = {k: pool_measures([r["lu"] for r in v], train, n, rng) for k, v in pools.items()}
    out["legal_share"] = float(np.mean([r["status"] != "rejected" for r in rows]))
    out["repeated_species_share"] = 1 - len(distinct6) / max(len(rows), 1)
    novel6 = [r for r in distinct6 if r["lu"] not in train]
    copied6 = [r for r in distinct6 if r["lu"] in train]
    out["legal_share_novel_six_distinct"] = float(np.mean([r["status"] != "rejected" for r in novel6])) if novel6 else None
    out["legal_share_copied_six_distinct"] = float(np.mean([r["status"] != "rejected" for r in copied6])) if copied6 else None
    out["top_reasons_novel_six_distinct"] = Counter(r["reason"] for r in novel6 if r["reason"]).most_common(6)
    return out


def kept_match(rows, row_name):
    """Share of kept drafts identical to the stored grid's teams in the same cell (same seeds should redraw them)."""
    path = RES / f"stat_points_84_{row_name}.json"
    if not path.exists(): return None
    stored = {}
    for c in json.load(open(path))["cells"].values():
        stored.setdefault((round(c["numerator"] / 24, 6), c["seed"]), set()).update(c["pastes"])
    kept = [r for r in rows if r["status"] == "kept"]
    hits = sum(r["paste"] in stored.get((round(r["ask"], 6), r["seed"]), ()) for r in kept)
    return hits / max(len(kept), 1)


def line(label, m):
    def p(k): return m[k]
    a, d, rj, k = p("all drafts"), p("six distinct species"), p("rejected, six distinct species"), p("kept")
    return (f"{label}: {a['teams']:,} drafts, legal {m['legal_share']:.1%}, repeated species {m['repeated_species_share']:.1%} | "
            f"novel share: six-distinct drafts {d['novel_share']:.1%}, rejected six-distinct {rj.get('novel_share', float('nan')):.1%}, "
            f"kept {k['novel_share']:.1%} | line-ups at {k['teams']:,} teams: six-distinct {d['line_ups_at_kept_size']:.0f} vs kept {k['line_ups']:,} | "
            f"legal if novel {m['legal_share_novel_six_distinct'] or 0:.1%} vs if copied {m['legal_share_copied_six_distinct'] or 0:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="LABEL=path pairs, e.g. g4=drafts_g4.jsonl")
    ap.add_argument("--grid-row", help="stored grid row prefix for the redraw check, e.g. udlm_loglinear_gidd_lr_k84")
    ap.add_argument("--out")
    a = ap.parse_args()
    train, rng, result = GN.training_line_ups(), np.random.default_rng(0), {}
    for spec in a.logs:
        label, path = spec.split("=", 1)
        rows = load(path)
        m = analyse(rows, train, rng)
        if a.grid_row:
            g = label.lstrip("g")
            m["kept_identical_to_stored_grid"] = kept_match(rows, a.grid_row if g == "2" else f"{a.grid_row}_g{g}")
        result[label] = m
        print(line(label, m), flush=True)
        if m.get("kept_identical_to_stored_grid") is not None:
            print(f"  kept drafts identical to the stored grid's teams: {m['kept_identical_to_stored_grid']:.1%}")
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1, default=str)
    print("DRAFT_NOVELTY_DONE")


if __name__ == "__main__":
    main()
