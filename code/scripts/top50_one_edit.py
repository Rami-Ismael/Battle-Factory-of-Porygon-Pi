"""One legal edit of each kind, made to the top-50 meta teams themselves.  Pre-registration: docs/top50-one-edit.md.

The edit-kind study (scripts/ruggedness.py, ruggedness2.py; rerun 2026-11-12, issue 313) edited real teams from outside
the top 50 and search-found teams, and used the top 50 only as opponents. This script edits the top 50.

Stages, each cached under the output folder (default ~/vgc-data/top50_one_edit, or TOP50_EDIT_OUT):
  prep     write every starting team and edited team, validator only, no battles -> plan.json, preflight.json
  smoke    battle checks before the run: uniform-legal teams near 0, one team twice on one seed -> preflight.json
  battle   the battles, resumable -> battles.json
  analyse  statistics and the two pre-registered readings of each edit kind -> result.json
  figure   edit size against the starting team's rank -> figures/edit-size-by-rank.html

Run from code/:  /tmp/vgc-pilot/.venv/bin/python scripts/top50_one_edit.py prep|smoke|battle|analyse|figure
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("TOP50_EDIT_OUT", "~/vgc-data/top50_one_edit")).expanduser()
RERUN = Path(os.environ.get("EDIT_KIND_RERUN", "~/vgc-data/edit_kind_rerun/run/results")).expanduser()
UNIFORM_LEGAL = Path("~/vgc-data/teams/hps_reg_mb_100k.jsonl").expanduser()
EDIT_KINDS = {"A_move_swap": "One move replaced", "B_ability_swap": "Ability replaced", "C_item_swap": "Held item replaced",
              "D_alignment_swap": "Stat alignment replaced", "E_spread_copy": "Stat Points copied from another set",
              "F_candidate_copy": "Whole Pokemon copied from a real set"}
GROUPS = {"top50": "Top-50 meta teams", "outside_top50": "Tournament-ready teams", "search_found": "Search-found teams"}
EDITS_PER_KIND, REPLAYS, BATTLES, SEED, BOOTSTRAPS = 3, (202, 303, 404), 384, 1, 2000
ANCHOR_SEED, EDIT_SEED, EDIT_CHUNK = 101, 202, 126


def read_json(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".tmp"); scratch.write_text(json.dumps(value, indent=1)); scratch.replace(path)


# ---------------------------------------------------------------- teams and edits
def edit_operators():
    """The operators and tables of mutation_closure_test.py, without its run. It changes directory, so use absolute paths."""
    source = (HERE / "mutation_closure_test.py").read_text().split("# ---------------------------------------------------------------- run")[0]
    module = {"__name__": "mut", "__file__": str(HERE / "mutation_closure_test.py")}
    argv, sys.argv = sys.argv, [sys.argv[0], str(SEED)]
    exec(compile(source, "mutation_closure_test.py", "exec"), module)
    sys.argv = argv
    module["rng"] = np.random.default_rng(SEED)
    return module


def describe_change(before, after):
    """What one edit changed, in words, from the two teams' slot records."""
    for old, new in zip(before, after):
        if old == new: continue
        if old["species"] != new["species"]: return f"{old['species']} and its whole set -> {new['species']} and its whole set"
        if old["moves"] != new["moves"]:
            gone = [m for m in old["moves"] if m not in new["moves"]]; came = [m for m in new["moves"] if m not in old["moves"]]
            return f"{old['species']}: move {', '.join(gone)} -> {', '.join(came)}"
        if old["ability"] != new["ability"]: return f"{old['species']}: ability {old['ability']} -> {new['ability']}"
        if old["item"] != new["item"]: return f"{old['species']}: item {old['item']} -> {new['item']}"
        if old["nature"] != new["nature"]: return f"{old['species']}: stat alignment {old['nature']} -> {new['nature']}"
        if old["evs"] != new["evs"]:
            points = lambda evs: ", ".join(f"{k} {v}" for k, v in evs.items() if v)
            return f"{old['species']}: Stat Points {points(old['evs'])} -> {points(new['evs'])}"
    return "no change found"


def comparable(paste):
    """A paste without what the rewrite drops: gender marks and the shiny line. Used by the round-trip check."""
    import re
    lines = [re.sub(r"\s*\((M|F)\)", "", line.strip()) for line in paste.strip().splitlines()]
    return [line.lower() for line in lines if not line.lower().startswith("shiny")]


def prep():  # noqa: C901
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import heldout
    import top50
    module = edit_operators()
    from corpus import parse_team
    team_to_text, operators, validator = module["team_to_text"], module["COND"], module["Validator"]()
    folder = OUT / "teams"; folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*.txt"): stale.unlink()
    corpus_pastes = {team_to_text(team) for team in module["teams"]}
    teams, round_trip_same, short, tried = [], 0, [], 0
    for rank, team_id in enumerate(top50.ids(), start=1):
        if team_id in top50.DUPLICATES: continue                # MB494 is MB493 again; one starting team, not two
        source = Path(top50.resolve(team_id)); base = parse_team(str(source)); paste = team_to_text(base)
        round_trip_same += comparable(source.read_text()) == comparable(paste)
        if validator(paste) is not None: raise SystemExit(f"{team_id}: the rewritten team fails the validator")
        stem = f"t{rank:02d}_{team_id}"; file = folder / f"{stem}.txt"; file.write_text(paste)
        edits, seen = [], {paste}
        for kind, operator in operators.items():
            if kind not in EDIT_KINDS: continue
            made = attempts = 0
            while made < EDITS_PER_KIND and attempts < 200:
                attempts += 1; tried += 1
                child = copy.deepcopy(base)
                if operator(child): continue                    # the operator reports a draw that cannot change anything
                child_paste = team_to_text(child)
                if child_paste in seen or validator(child_paste) is not None: continue   # unchanged, repeated or illegal
                seen.add(child_paste)
                edit_file = folder / f"{stem}_{kind[0]}{made}.txt"; edit_file.write_text(child_paste)
                edits.append({"kind": kind, "file": str(edit_file), "paste": child_paste, "change": describe_change(base, child)})
                made += 1
            if made < EDITS_PER_KIND: short.append(f"{team_id} {kind}: {made}")
        teams.append({"id": team_id, "rank_in_list": rank, "file": str(file), "paste": paste, "edits": edits})
        print(f"{team_id}: {len(edits)} edits", flush=True)
    validator.close()
    files = [team["file"] for team in teams] + [edit["file"] for team in teams for edit in team["edits"]]
    heldout.assert_clear_files(files, "a scored candidate")
    starting = {team["paste"] for team in teams}
    edited = [edit["paste"] for team in teams for edit in team["edits"]]
    write_json(OUT / "plan.json", {"seed": SEED, "edits_per_kind": EDITS_PER_KIND, "teams": teams})
    preflight = read_json(OUT / "preflight.json") if (OUT / "preflight.json").exists() else {}
    preflight.update({
        "round_trip": {"teams_identical_ignoring_gender_and_shiny": round_trip_same, "of": len(teams), "pass": round_trip_same == len(teams)},
        "coverage": {"starting_teams": len(teams), "edited_teams": len(edited), "planned": len(teams) * len(EDIT_KINDS) * EDITS_PER_KIND,
                     "kinds_short_of_three": short, "draws_tried": tried, "pass": not short},
        "copies": {"edited_equal_to_a_starting_team": sum(p in starting for p in edited),
                   "edited_equal_to_a_corpus_team": sum(p in corpus_pastes for p in edited), "repeated_edited_teams": len(edited) - len(set(edited))},
        "separation": {"held_out_teams_among_scored": 0, "pass": True, "note": "heldout.assert_clear_files passed; no model is trained"}})
    write_json(OUT / "preflight.json", preflight)
    print(json.dumps({k: preflight[k] for k in ("round_trip", "coverage", "copies")}, indent=1))


# ---------------------------------------------------------------- battles
def scorer():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import pool
    import top50
    opponents = top50.files()            # all 50 files, 49 distinct teams; a starting team's own file stays in
    return lambda files, seed, battles=BATTLES: pool.score([(f, opponents) for f in files], battles=battles, conc=50,
                                                           opp_schedule=opponents[:], seed=seed)


def smoke():
    """Battle checks the run cannot start without. Uniform-legal teams must sit near 0, and one seed must give one design."""
    plan, score = read_json(OUT / "plan.json"), scorer()
    folder = OUT / "smoke"; folder.mkdir(exist_ok=True)
    uniform = []
    for i, line in enumerate(UNIFORM_LEGAL.read_text().splitlines()[:5]):
        path = folder / f"uniform_{i}.txt"; path.write_text(json.loads(line)["team"]); uniform.append(str(path))
    uniform_scores = [row["win_rate"] for row in score(uniform, 7, 100).values()]
    team = plan["teams"][0]["file"]
    first, second = score([team], ANCHOR_SEED)[team], score([team], ANCHOR_SEED)[team]
    noise = 1.96 * float(np.sqrt(2 * first["win_rate"] * (1 - first["win_rate"]) / first["battles"]))
    preflight = read_json(OUT / "preflight.json")
    preflight["controls_uniform_legal"] = {"win_rates": uniform_scores, "mean": float(np.mean(uniform_scores)), "pass": float(np.mean(uniform_scores)) < 0.10}
    preflight["repeat"] = {"team": team, "win_rates": [first["win_rate"], second["win_rate"]], "battles": [first["battles"], second["battles"]],
                           "allowed_gap": noise, "pass": abs(first["win_rate"] - second["win_rate"]) <= noise and first["battles"] == second["battles"] == BATTLES}
    write_json(OUT / "preflight.json", preflight)
    print(json.dumps({k: preflight[k] for k in ("controls_uniform_legal", "repeat")}, indent=1))
    if not (preflight["controls_uniform_legal"]["pass"] and preflight["repeat"]["pass"]): raise SystemExit("smoke failed")


def battle():
    plan, score = read_json(OUT / "plan.json"), scorer()
    done = read_json(OUT / "battles.json") if (OUT / "battles.json").exists() else {}
    starting = [team["file"] for team in plan["teams"]]
    edited = [edit["file"] for team in plan["teams"] for edit in team["edits"]]
    jobs = [("anchor", starting, ANCHOR_SEED)]
    jobs += [(f"edits_{i // EDIT_CHUNK:02d}", edited[i:i + EDIT_CHUNK], EDIT_SEED) for i in range(0, len(edited), EDIT_CHUNK)]
    jobs += [(f"replay_{seed}", starting, seed) for seed in REPLAYS]
    for key, files, seed in jobs:
        if key in done: continue
        start = time.time(); done[key] = score(files, seed); write_json(OUT / "battles.json", done)
        short = [f for f in files if done[key].get(f, {}).get("battles") != BATTLES]
        print(f"{key}: {len(files)} teams in {time.time() - start:.0f}s, {len(short)} short of {BATTLES} battles", flush=True)
        if short: raise SystemExit(f"{key}: scores short of {BATTLES} battles, stopping")
        if key == "anchor":                                     # the control: top-50 teams score near 0.50 against each other
            mean = float(np.mean([row["win_rate"] for row in done[key].values()]))
            preflight = read_json(OUT / "preflight.json")
            preflight["controls_top50"] = {"mean_win_rate": mean, "teams": len(files), "pass": 0.47 <= mean <= 0.53}
            write_json(OUT / "preflight.json", preflight)
            if not preflight["controls_top50"]["pass"]: raise SystemExit(f"top-50 teams average {mean:.3f}, outside 0.47 to 0.53")


# ---------------------------------------------------------------- statistics
def measured_groups():
    """Every starting team as {id, f, n, replays, edits: [{kind, f, n, change}]}, for the top 50 and the two rerun groups."""
    groups = {}
    if (OUT / "battles.json").exists():
        plan, scores = read_json(OUT / "plan.json"), read_json(OUT / "battles.json")
        edit_scores = {f: row for key, rows in scores.items() if key.startswith("edits_") for f, row in rows.items()}
        if "anchor" in scores and all(f"replay_{seed}" in scores for seed in REPLAYS) and all(e["file"] in edit_scores for t in plan["teams"] for e in t["edits"]):
            groups["top50"] = [{"id": t["id"], "f": scores["anchor"][t["file"]]["win_rate"], "n": scores["anchor"][t["file"]]["battles"],
                                "replays": [scores[f"replay_{seed}"][t["file"]]["win_rate"] for seed in REPLAYS],
                                "edits": [{"kind": e["kind"], "f": edit_scores[e["file"]]["win_rate"], "n": edit_scores[e["file"]]["battles"], "change": e["change"]}
                                          for e in t["edits"]]} for t in plan["teams"]]
    first, parts = RERUN / "ruggedness_shuffled50.json", RERUN / "ruggedness2_shuffled50_parts"
    if first.exists() and (parts / "vgc_battles.json").exists():
        run1, plan, scores = read_json(first)["vgc"]["anchors"], read_json(parts / "vgc_plan.json"), read_json(parts / "vgc_battles.json")
        edit_row = lambda e: {"kind": e["op"], "f": scores["edits"][e["file"]]["win_rate"], "n": scores["edits"][e["file"]]["battles"], "change": ""}
        groups["outside_top50"] = [{"id": a["name"].removesuffix(".txt"), "f": a["f"], "n": a["n"], "replays": a["reps"],
                                    "edits": [{"kind": e["op"], "f": e["f"], "n": e["n"], "change": ""} for e in a["edits"]] + [edit_row(e) for e in new["edits"]]}
                                   for a, new in zip(run1, plan["meta_new"])]
        groups["search_found"] = [{"id": f"search rank {a['rank']}", "f": scores["anchor"][a["file"]]["win_rate"], "n": scores["anchor"][a["file"]]["battles"],
                                   "replays": [scores[f"rep{seed}"][a["file"]]["win_rate"] for seed in REPLAYS], "edits": [edit_row(e) for e in a["edits"]]}
                                  for a in plan["search"]]
    return groups


def squared_moves(teams, kind=None):
    """Mean squared move of the edits of one kind, and of the replays, both against the team's first score."""
    edits = [(e["f"] - t["f"]) ** 2 for t in teams for e in t["edits"] if kind is None or e["kind"] == kind]
    replays = [(r - t["f"]) ** 2 for t in teams for r in t["replays"]]
    return float(np.mean(edits)), float(np.mean(replays))


def score_variance(groups):
    """Variance of the starting teams' true scores, pooled over every group so they share one scale, with replay noise removed."""
    scores = np.array([t["f"] for teams in groups.values() for t in teams])
    replay = np.mean([(r - t["f"]) ** 2 for teams in groups.values() for t in teams for r in t["replays"]])
    return float(scores.var(ddof=1) - replay / 2)


def edit_size(teams, kind, variance):
    """Noise-corrected size of one edit, three ways: in win-rate units, as a share of the spread of true scores between
    starting teams (size / standard deviation), and as the neighbour correlation that share implies, 1 - share^2 / 2."""
    edits, replays = squared_moves(teams, kind)
    signal = max(edits - replays, 0.0)
    return float(np.sqrt(signal)), float(np.sqrt(signal / variance)), float(1 - signal / (2 * variance))


def step_share(anchors):
    """One benchmark step as a share of the spread of true scores between starting points of the same kind.
    Random and hill-climbed starting points are never pooled here: their mean scores differ, and that gap would
    enter the spread and make one step look small (owner decision, 2026-11-13)."""
    return edit_size(anchors, None, score_variance({"own": anchors}))[1]


def readings(size_interval, share_gaps):
    """The two pre-registered readings of one edit kind. No chosen number enters either.
    Above zero: the 95% interval of the size excludes zero, so the edit does something battle noise does not.
    Against one Pest Control step: the edit as a share of the spread between teams, the step as a share of the spread
    between benchmark starting points of its own kind. 'larger' if every interval of the difference lies above zero,
    'smaller' if every one lies below, 'undecided' otherwise."""
    against = "larger" if all(low > 0 for low, _ in share_gaps) else "smaller" if all(high < 0 for _, high in share_gaps) else "undecided"
    return bool(size_interval[0] > 0), against


def share_beyond_noise(teams, kind):
    """Share of edits, and of replays, whose move is larger than 1.96 standard errors of a difference of two scores."""
    edits = [abs(e["f"] - t["f"]) > 1.96 * np.sqrt(t["f"] * (1 - t["f"]) / t["n"] + e["f"] * (1 - e["f"]) / e["n"])
             for t in teams for e in t["edits"] if e["kind"] == kind]
    replays = [abs(r - t["f"]) > 1.96 * np.sqrt(2 * t["f"] * (1 - t["f"]) / t["n"]) for t in teams for r in t["replays"]]
    return float(np.mean(edits)), float(np.mean(replays))


def per_team(groups):
    """One row for each starting team: its rank by replay mean (rank 1 wins most) and the size of its edits in points.
    The rank uses the three replays, not the first score the edits are measured against, so a lucky first score
    cannot both raise a team's rank and enlarge its edits."""
    rows = [{"group": name, "id": t["id"], "win_rate_for_rank": float(np.mean(t["replays"])), "first_score": t["f"],
             "edit_move_points": 100 * float(np.sqrt(np.mean([(e["f"] - t["f"]) ** 2 for e in t["edits"]]))),
             "replay_move_points": 100 * float(np.sqrt(np.mean([(r - t["f"]) ** 2 for r in t["replays"]]))), "edits": len(t["edits"])}
            for name, teams in groups.items() for t in teams]
    for rank, row in enumerate(sorted(rows, key=lambda r: -r["win_rate_for_rank"]), start=1): row["rank"] = rank
    return sorted(rows, key=lambda r: r["rank"])


def rank_correlation(rows):
    """Spearman correlation between a team's rank and the size of its edits. Negative means teams that win more change more."""
    if len(rows) < 3: return None
    order = lambda values: np.argsort(np.argsort(values))
    return float(np.corrcoef(order([r["rank"] for r in rows]), order([r["edit_move_points"] for r in rows]))[0, 1])


def interval(values): return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def analyse_groups(groups):
    """Sizes, intervals and the two readings for every group in `groups`. Scores are pooled over these groups for the spread."""
    rng = np.random.default_rng(7)
    resample = lambda teams: [teams[i] for i in rng.integers(0, len(teams), len(teams))]
    boots = [{name: resample(teams) for name, teams in groups.items()} for _ in range(BOOTSTRAPS)]
    variance, boot_variance = score_variance(groups), [score_variance(b) for b in boots]
    pest = read_json(RERUN / "ruggedness2_shuffled50_parts" / "benchmarks.json")["pest_control"]
    pest = {start: [{"f": a["f"], "replays": a["reps"], "edits": [{"kind": "step", "f": e["f"]} for e in a["edits"]]} for a in anchors] for start, anchors in pest.items()}
    pest_boot = {start: [step_share(resample(anchors)) for _ in range(BOOTSTRAPS)] for start, anchors in pest.items()}
    result = {"design": {"battles_per_score": BATTLES, "edits_per_kind": EDITS_PER_KIND, "replays": len(REPLAYS), "bootstraps": BOOTSTRAPS,
                         "starting_teams": {name: len(teams) for name, teams in groups.items()}},
              "score_variance_pooled": variance, "spread_between_teams_points": 100 * float(np.sqrt(variance)),
              "pest_control": {start: {"step_share_of_spread": step_share(anchors), "spread_between_starting_points": float(np.sqrt(score_variance({"own": anchors}))),
                                       "step_share_of_spread_ci95": interval(pest_boot[start])} for start, anchors in pest.items()},
              "groups": {}}
    for name, teams in groups.items():
        result["groups"][name] = {"label": GROUPS[name], "mean_first_score": float(np.mean([t["f"] for t in teams])), "kinds": {}}
        for kind, label in EDIT_KINDS.items():
            size, share, correlation = edit_size(teams, kind, variance)
            boot = [edit_size(b[name], kind, v) for b, v in zip(boots, boot_variance)]
            edits_beyond, replays_beyond = share_beyond_noise(teams, kind)
            share_gap = [np.subtract(*share_beyond_noise(b[name], kind)) for b in boots]
            shares, correlations = np.array([s for _, s, _ in boot]), [c for _, _, c in boot]
            signed = lambda sample: 100 * float(np.mean([e["f"] - t["f"] for t in sample for e in t["edits"] if e["kind"] == kind]))
            row = {"label": label, "edits": sum(e["kind"] == kind for t in teams for e in t["edits"]),
                   "mean_signed_change_points": signed(teams), "mean_signed_change_points_ci95": interval([signed(b[name]) for b in boots]),
                   "change_points": 100 * size, "change_points_ci95": [100 * x for x in interval([s for s, _, _ in boot])],
                   "share_of_spread": share, "share_of_spread_ci95": interval(shares),
                   "share_minus_pest_step_ci95": {start: interval(shares - np.array(pest_boot[start])) for start in pest},
                   "neighbour_correlation": correlation, "neighbour_correlation_ci95": interval(correlations),
                   "share_edits_beyond_noise": edits_beyond, "share_replays_beyond_noise": replays_beyond, "share_gap_ci95": interval(share_gap)}
            row["above_zero"], row["against_pest_control_step"] = readings(row["change_points_ci95"], list(row["share_minus_pest_step_ci95"].values()))
            result["groups"][name]["kinds"][kind] = row
            print(f"{GROUPS[name]:30s} {label:36s} {row['change_points']:5.2f} [{row['change_points_ci95'][0]:.2f}, {row['change_points_ci95'][1]:.2f}] points  "
                  f"above zero {row['above_zero']}  share of spread {share:.2f}, against a Pest Control step: {row['against_pest_control_step']}", flush=True)
    rows = per_team(groups)
    result["per_team"] = rows
    boot_rows = [per_team(b) for b in boots]
    result["rank_correlation"] = {"all_teams": rank_correlation(rows), "all_teams_ci95": interval([rank_correlation(r) for r in boot_rows])}
    if "top50" in groups:
        result["rank_correlation"].update({
            "top50_only": rank_correlation([r for r in rows if r["group"] == "top50"]),
            "top50_only_ci95": interval([rank_correlation([r for r in br if r["group"] == "top50"]) for br in boot_rows])})
    result["replay_move_points"] = {name: 100 * float(np.sqrt(squared_moves(teams)[1])) for name, teams in groups.items()}
    return result


def analyse():
    groups = measured_groups()
    if "top50" not in groups: raise SystemExit("the top-50 battles are not complete; run the battle stage first")
    result = analyse_groups(groups)
    if (OUT / "preflight.json").exists(): result["preflight"] = read_json(OUT / "preflight.json")
    write_json(OUT / "result.json", result)
    print("rank correlation", json.dumps(result["rank_correlation"]))


# ---------------------------------------------------------------- figure
def figure():
    """Edit size against rank: does one edit move a team more when the team wins more? Draws whichever groups are measured."""
    sys.path.insert(0, str(HERE.parent / "src"))
    from echarts_chart import write_chart
    groups = measured_groups()
    rows = per_team(groups)
    colors = {"top50": "#2a78d6", "outside_top50": "#14946a", "search_found": "#b07800"}
    noise = float(np.sqrt(np.mean([(r - t["f"]) ** 2 for teams in groups.values() for t in teams for r in t["replays"]]))) * 100
    series = [{"type": "scatter", "name": GROUPS[name], "color": colors[name], "symbolSize": 11, "emphasis": {"focus": "series"},
               "data": [[r["rank"], round(r["edit_move_points"], 2), r["id"], round(100 * r["win_rate_for_rank"], 1)] for r in rows if r["group"] == name]}
              for name in GROUPS if name in groups]
    series[0]["markLine"] = {"silent": True, "symbol": "none", "lineStyle": {"color": "#52514e", "type": "dashed"},
                             "label": {"formatter": f"replay alone: {noise:.1f} points", "position": "insideEndTop", "color": "#52514e"},
                             "data": [{"yAxis": round(noise, 2)}]}
    option = {"legend": {"top": 0}, "grid": {"left": 56, "right": 24, "top": 44, "bottom": 52},
              "tooltip": {"trigger": "item", "formatter": "js:p => `${p.data[2]}<br>rank ${p.data[0]}, wins ${p.data[3]}%<br>one edit moves it ${p.data[1]} points`"},
              "xAxis": {"type": "value", "name": "Rank of the starting team by win rate (1 wins most)", "nameLocation": "middle", "nameGap": 30, "min": 0, "max": 10 * (len(rows) // 10 + 1)},
              "yAxis": {"type": "value", "name": "How far one edit moves win rate, points", "nameLocation": "middle", "nameGap": 38, "min": 0},
              "series": series}
    counts = ", ".join(f"{len(teams)} {GROUPS[name].lower()}" for name, teams in groups.items())
    correlation = rank_correlation(rows)
    path = write_chart(option, OUT / "figures" / "edit-size-by-rank.html", "How far one edit moves a team, against how much the team wins",
                       subtitle=f"Each dot is one starting team: {counts}. Rank correlation {correlation:+.2f}; negative means teams that win more change more.",
                       note=("A dot's height is the typical move of that team's 18 edits, battle noise included; the dashed line is what replaying an unchanged team does. "
                             f"Each score is {BATTLES} shuffled-slot battles against all 50 meta files with the behaviour-cloning policy on both sides. "
                             "Rank uses the mean of a team's three replays, not the first score its edits are measured against."), height=460)
    print("wrote", path)


if __name__ == "__main__":
    {"prep": prep, "smoke": smoke, "battle": battle, "analyse": analyse, "figure": figure}[sys.argv[1]]()
