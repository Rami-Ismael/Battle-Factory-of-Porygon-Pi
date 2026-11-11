"""Replay the four validation draws of one wrong-Mega-Stone team and save every number the figure shows.

    /tmp/vgc-pilot/.venv/bin/python artifacts/wrong-mega-stone-validation-loss/probe_worst_check.py

Team: row 2262 of lossdata65.npz (Whimsicott @ Glimmoranite). Model: the subset-size control model, checkpoint
stat65_subset_39. It reuses the loaders and the draw order of wrong_mega_stone.py (issue #258's figure gist), so the
random numbers match the ones training logged. Laptop only: it needs the frozen code of the subset-size queue.
It replays the team twice: as it is, and with Whimsicott's Glimmoranite swapped for a Focus Sash (a what-if on the
same four draws; the other 5,405 validation teams are untouched). One team, eight forward passes, on one CPU thread.
Writes probe_worst_check.json beside this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
GIST = Path.home() / "vgc-data/validation_loss_accuracy/figures_gist"
sys.path.insert(0, str(GIST))
import wrong_mega_stone as W  # noqa: E402  (forces the CPU, then imports the frozen diffusion84)

D = W.D
STAT_NAMES = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
TOP = 8
WHAT_IF_ITEM = "Focus Sash"
WATCH = ["Whimsicott", "Garchomp", "Floette-Eternal", "Maushold-Four", "Basculegion", "Chandelure", "Gholdengo", "Charizard",
         "Glimmora", "Glimmora-Mega"]   # species the figure draws one by one; every other species is one bucket


@torch.no_grad()
def main():
    model, net, weights = W.load_model()
    teams, labels, _, is_validation = W.load_teams()
    rows = np.flatnonzero(is_validation)
    i = int(np.flatnonzero(rows == W.WORST_TEAM)[0])
    team = torch.as_tensor(teams[rows[i]]).unsqueeze(0)
    label_row = np.atleast_1d(np.asarray(labels[rows[i]], np.float32))
    label = torch.as_tensor(label_row).view(1, -1) if label_row.size > 1 else torch.as_tensor(label_row).view(1)
    wins_battles = np.load(W.train65.DATA)["C"][rows[i]].sum(0)          # [wins, battles] over the 39 training opponents
    species_names = model.V.itos["species"]
    show = lambda v: model.look.get(v, v)

    out = dict(row=int(W.WORST_TEAM), checkpoint=W.CHECKPOINT.name, data="lossdata65.npz", seed=W.SEED,
               n_species=len(species_names) - 1, win_rate_label=float(label_row[0]), label_wins=float(wins_battles[0]), label_battles=float(wins_battles[1]), validation_teams=int(len(rows)), fields=int(D.COLS),
               field_weights=dict(zip(W.FIELD_NAMES + ["stat point"], [float(weights[c]) for c in range(8)] + [float(weights[D.STAT0])])),
               what_if_item=WHAT_IF_ITEM)

    def replay(name, team):
        draws = []
        for draw, slot_order, t, hidden in W.validation_draws(len(rows), weights):
            t_i, hidden_i = t[i:i + 1], hidden[i:i + 1]
            shuffled = D.shuffle_slots(team, slot_order[i:i + 1])
            visible = shuffled.masked_fill(hidden_i, 0)
            raw = net.logits(net(visible, t_i, label))
            allowed = D.allowed(model.tables, visible)
            masked = {k: v.masked_fill(~allowed[k], -1e4) for k, v in raw.items()}
            ce_masked = D.per_column_ce(masked, shuffled) * hidden_i
            ce_raw = D.per_column_ce(raw, shuffled) * hidden_i
            scale = weights / t_i / D.COLS
            cost = (ce_masked * scale)[0]

            slots = model.to_paste(shuffled[0])[1]
            pokemon = []
            for s, p in enumerate(slots):
                base = s * 8
                names = [p["species"], p["ability"], p["item"]] + list(p["moves"]) + [""] * (4 - len(p["moves"])) + [p["nature"]]
                fields = []
                for f in range(8):
                    c = base + f
                    fields.append(dict(name=W.FIELD_NAMES[f], value=names[f], hidden=bool(hidden_i[0, c]),
                                       ce_model_alone=float(ce_raw[0, c]), ce_with_mask=float(ce_masked[0, c]),
                                       weight=float(weights[c]), cost=float(cost[c])))
                evs = p["evs"]
                ev_values = [evs.get(k, evs.get(k.lower(), 0)) for k in ("hp", "atk", "def", "spa", "spd", "spe")] if isinstance(evs, dict) else list(evs)
                for k in range(6):
                    c = D.STAT0 + s * 6 + k
                    fields.append(dict(name=STAT_NAMES[k], value=int(shuffled[0, c]) - 1, hidden=bool(hidden_i[0, c]),
                                       ce_model_alone=float(ce_raw[0, c]), ce_with_mask=float(ce_masked[0, c]),
                                       weight=float(weights[c]), cost=float(cost[c])))
                pokemon.append(dict(species=p["species"], item=p["item"], ability=p["ability"], nature=p["nature"],
                                    moves=list(p["moves"]), evs_from_paste=ev_values, fields=fields))

            entry = dict(draw=draw + 1, t=float(t_i), hidden=int(hidden_i.sum()), cost=float(cost.sum()), pokemon=pokemon, species_checks=[])
            for s in range(6):
                c = s * 8
                if not bool(hidden_i[0, c]):
                    continue
                true = int(shuffled[0, c])
                lr, lm, al = raw["species"][0, s].float(), masked["species"][0, s].float(), allowed["species"][0, s]
                pr, pm = lr.softmax(-1), lm.softmax(-1)
                top = torch.topk(pr, TOP).indices.tolist()
                allowed_ids = torch.nonzero(al).flatten().tolist()
                entry["species_checks"].append(dict(
                    slot=s, true=show(species_names[true]), true_allowed=bool(al[true]),
                    true_logit=float(lr[true]), true_prob_model_alone=float(pr[true]), true_prob_with_mask=float(pm[true]),
                    top_model_alone=[dict(species=show(species_names[j]), prob=float(pr[j]), logit=float(lr[j]), allowed=bool(al[j])) for j in top],
                    n_allowed=len(allowed_ids),
                    allowed=[dict(species=show(species_names[j]), prob_model_alone=float(pr[j]), prob_with_mask=float(pm[j]), logit=float(lr[j]))
                             for j in sorted(allowed_ids, key=lambda j: -float(pm[j]))[:TOP]],
                    logsumexp_allowed=float(torch.logsumexp(lr[al], 0)),
                    allowed_share_model_alone=float(pr[al].sum()), mask_token_prob=float(pr[0]),
                    watch=[dict(species=show(species_names[j]), allowed=bool(al[j]), prob_model_alone=float(pr[j]),
                                prob_with_filter=float(pm[j]), logit=float(lr[j]))
                           for j in (species_names.index(k) if k in species_names else next(n for n, v in enumerate(species_names) if show(v) == k) for k in WATCH)],
                    ce_model_alone=float(ce_raw[0, c]), ce_with_mask=float(ce_masked[0, c]), cost=float(cost[c])))
            draws.append(entry)
            print(f"{name} draw {draw + 1}: t {float(t_i):.2f}, {entry['hidden']} hidden, cost {entry['cost']:,.1f}")
        return draws

    out["draws"] = replay("as battled", team)
    items = model.V.itos["item"]
    sash = next(j for j, v in enumerate(items) if show(v) == WHAT_IF_ITEM)
    stone = next(j for j, v in enumerate(items) if show(v) == "Glimmoranite")
    col = next(c for c in range(2, D.STAT0, 8) if int(team[0, c]) == stone)
    swapped = team.clone(); swapped[0, col] = sash
    out["what_if_flagged_by_tables"] = bool(model.tables.check(swapped[:, :D.STAT0].numpy())[0])
    out["what_if_draws"] = replay("what-if", swapped)
    out["what_if_team_loss"] = float(np.mean([d["cost"] for d in out["what_if_draws"]]))
    # The eight wrong-stone validation teams, decoded: no forward pass, only the vocabulary.
    eight = json.loads(Path(__file__).with_name("per_team_validation_loss.json").read_text())["eight"]
    out["eight_teams"] = [dict(row=e["team"], pokemon=[dict(species=p["species"], item=p["item"], wrong_stone=not W.stone_fits(p["species"], p["item"]))
                                                       for p in W.describe_team(model, teams[e["team"]])]) for e in eight]
    out["team_loss"] = float(np.mean([d["cost"] for d in out["draws"]]))
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
    print("team loss", round(out["team_loss"], 2), "· what-if", round(out["what_if_team_loss"], 3), "· saved", Path(__file__).with_suffix(".json").name)


if __name__ == "__main__":
    main()
