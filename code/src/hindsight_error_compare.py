"""Does the hindsight-relabeled fine-tune bring got closer to the ask than the same model without it? (GitHub issue #106;
docs/hindsight-relabel-fine-tune.md)

Reads three saved 23-ask grids, three sampling seeds each: no fine-tune at epoch 21 (the control, whose battled teams are the
fine-tune's training teams), the hindsight-relabeled fine-tune of it, and no fine-tune at epoch 80, the lowest error the
labelled-teams training reached. The ask-versus-got error (src/ask_line.py) is judged on asks 1/24 to 17/24 and reported apart
on 18/24 to 23/24. Differences carry a 95% interval from a bootstrap over teams within each ask and seed (src/ddom_error_compare.py).

Beside the error: response to the ask (mean got at asks 15/24 to 17/24 minus at 1/24 to 3/24; 14/24 on the line), the risk the
issue names, since Online Decision Transformer's Figure 4.2 warns that fine-tuning can flatten it; win rate at the top asks;
win-rate scatter across teams within an ask; line-ups, new ones against the fixed references; the share of Pokémon spending all
66 Stat Points; and how many fine-tuned teams repeat a relabelled team or its line-up.

    /tmp/vgc-pilot/.venv/bin/python src/hindsight_error_compare.py [output.json]   (from code/)
"""
import json
import sys
from pathlib import Path

import numpy as np

import ddom_error_compare as D
from ask_line import NEAR_LINE_MSE

CONTROL, FINE_TUNE, EPOCH_80 = "gen84_noise_byepoch_ep21", "ft_hindsight_relabel", "gen84_noise_byepoch_ep80"
NAMES = {CONTROL: "No fine-tune", FINE_TUNE: "Hindsight-relabeled fine-tune", EPOCH_80: "59 more training epochs"}   # what follows epoch 21
LOW, HIGH = [1, 2, 3], [15, 16, 17]                      # the response to the ask: top three scored asks minus bottom three
ON_LINE_RESPONSE = (np.mean(HIGH) - np.mean(LOW)) / 24
BOOT_SEED = 106
RES = Path.home() / "vgc-data/results"


def ask_response(win_by_ask):
    """Mean got at asks 15/24 to 17/24 minus mean got at 1/24 to 3/24. 14/24 on the line; smaller means a flatter response."""
    return float(np.mean([win_by_ask[k] for k in HIGH]) - np.mean([win_by_ask[k] for k in LOW]))


def scatter(by_ask, asks=D.SCORED):
    """Pooled standard deviation of team win rates within each ask, over these asks."""
    dev = [y - np.mean(ys) for k in asks for ys in [[v for c in by_ask[k] for v in c["y"]]] for y in ys]
    return float(np.sqrt(np.mean(np.square(dev))))


def team_key(paste):
    """A team's identity regardless of slot order: its six Pokémon blocks, sorted."""
    return tuple(sorted(b.strip() for b in paste.strip().split("\n\n")))


def repeats(by_ask, relabelled_pastes, species_of):
    """Fine-tuned teams identical to a relabelled team, and fine-tuned line-ups the relabelled teams already hold."""
    teams = [p for cs in by_ask.values() for c in cs for p in c["pastes"]]
    keys, lineups = {team_key(p) for p in relabelled_pastes}, {species_of(p) for p in relabelled_pastes}
    mine = {species_of(p) for p in teams}
    return dict(teams=len(teams), identical_to_relabelled=sum(team_key(p) in keys for p in teams),
                line_ups=len(mine), line_ups_in_relabelled=len(mine & lineups))


def response_rows(grids, rng, control=CONTROL):
    """Response to the ask per arm, with a 95% interval, and each other arm minus the control."""
    boot = {a: D.bootstrap(g, rng) for a, g in grids.items()}
    resp = {a: np.mean([boot[a][k] for k in HIGH], 0) - np.mean([boot[a][k] for k in LOW], 0) for a in grids}
    q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    rows = {}
    for a, g in grids.items():
        r = dict(response=ask_response(D.mean_got(g)), interval=q(resp[a]), on_line=ON_LINE_RESPONSE)
        if a != control:
            r["minus_control"] = dict(difference=r["response"] - ask_response(D.mean_got(grids[control])), interval=q(resp[a] - resp[control]))
        rows[a] = r
    return rows


def verdict(scored):
    """wins: at least 0.001 lower with the interval below zero; loses: interval above zero; else matches."""
    d = scored.get("minus_control")
    if d is None: return "control"
    if d["interval"][1] < 0 and d["difference"] <= -NEAR_LINE_MSE: return "wins"
    if d["interval"][0] > 0: return "loses"
    return "matches"


def main(path):
    from lineup_reference import references, species_of    # imports the 84-field model code; needs /tmp/vgc-pilot
    files = {a: json.load(open(RES / f"stat_points_84_{a}.json")) for a in NAMES}
    grids = {a: D.grid_cells(out) for a, out in files.items()}
    for a, g in grids.items():
        missing = [k for k in D.ASKS if len(g.get(k, [])) != 3 or any("y" not in c for c in g.get(k, []))]
        if missing: raise SystemExit(f"{a}: asks without three battled seeds: {missing}")
    arms = D.judge(grids, np.random.default_rng(BOOT_SEED), names=NAMES, control=CONTROL)
    response = response_rows(grids, np.random.default_rng(BOOT_SEED + 1))
    training, records = references()
    relabelled = [p for cs in grids[CONTROL].values() for c in cs for p in c["pastes"]]
    for a, g in grids.items():
        arms[a].update(verdict=verdict(arms[a]["scored"]), response=response[a], scatter=scatter(g),
                       share_spending_66=files[a]["summary"]["stat_points"]["share_spending_66"],
                       legal=files[a]["summary"]["legal"], condition=(files[a].get("preflight") or {}).get("condition", {}).get("share_teams_changed"),
                       line_ups=dict(all_asks=D.line_ups(g, species_of, training, records), top_asks=D.line_ups(g, species_of, training, records, D.TOP)))
    arms[FINE_TUNE]["repeats"] = repeats(grids[FINE_TUNE], relabelled, species_of)
    stop = json.load(open(RES / "stat84_labelled_noise_byepoch_ep21_ft_hindsight_relabel_es3_cap40.json"))
    arms[FINE_TUNE]["training"] = {k: stop[k] for k in ("best_pass", "stopped_pass", "cap", "patience", "lr", "batch", "label_noise",
                                                       "n_train", "relabelled_teams", "val_share", "split_seed", "relabels_per_ask", "curves")}
    out = dict(win_rate="against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, one battle per opponent",
               interval="95%, bootstrap over teams within each ask and seed, battles not resampled", draws=D.DRAWS,
               scored_asks=D.SCORED, top_asks=D.TOP, response_asks=dict(low=LOW, high=HIGH, on_line=ON_LINE_RESPONSE),
               reference_sizes=dict(training=len(training), loop_and_records=len(records)), arms=arms)
    json.dump(out, open(path, "w"), indent=1)

    def gap(d):
        return f"{d['difference']:+.4f} [{d['interval'][0]:+.4f}, {d['interval'][1]:+.4f}]" if d else ""
    for a, r in arms.items():
        lu = r["line_ups"]["all_asks"]
        print(f"{r['name']:<42} {r['verdict']:<8} error 1/24-17/24 {r['scored']['error']:.4f} {gap(r['scored'].get('minus_control'))} | "
              f"18/24-23/24 {r['top']['error']:.4f} {gap(r['top'].get('minus_control'))} | top-ask win {r['top_win']['win']:.3f} "
              f"{gap(r['top_win'].get('minus_control'))} | response {r['response']['response']:.3f} {gap(r['response'].get('minus_control'))} "
              f"| scatter {r['scatter']:.3f} | 66 share {r['share_spending_66']:.3f} | line-ups {lu['line_ups']} new {lu['new_vs_training']}")
    print("repeats of relabelled teams:", json.dumps(arms[FINE_TUNE]["repeats"]))
    print("HINDSIGHT_ERROR_DONE", path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "hindsight_error_compare.json")
