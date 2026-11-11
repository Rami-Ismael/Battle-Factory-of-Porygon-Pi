"""Build the figure "How one wrong Mega Stone becomes a loss of 2,292" (issue #258): index.html, the animated team
sheet the owner picked on 2026-11-11, and storyboard.html, its still frames for the GitHub issue.

    python3 artifacts/wrong-mega-stone-validation-loss/build.py

Reads probe_worst_check.json (the replayed draws of validation team 2262, written by probe_worst_check.py) and
per_team_validation_loss.json (each validation team's loss, copied from the figure gist's fig1_github.html, the
output of `wrong_mega_stone.py teams`). Embeds the Pokémon sprites and item icons as data URIs and fills
template.html and storyboard.template.html. render_storyboard.py then turns storyboard.html into the two PNGs. Every number on the page comes from those two files or is arithmetic on them, done here.
"""
import base64
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SPRITES = ROOT / "notes/Blog/visuals/search-loop/sprites"
ITEMS = ROOT / "notes/Blog/visuals/item-sprites"
DISPLAY = {"Maushold-Four": "Maushold", "Floette-Eternal": "Eternal Floette", "Glimmora-Mega": "Mega Glimmora", "Blaziken-Mega": "Mega Blaziken",
           "Ninetales-Alola": "Alolan Ninetales", "Meowstic-M-Mega": "Mega Meowstic"}
FALLBACK_SPRITE = {"meowsticmmega": "meowstic"}        # the sprite store has no male Mega Meowstic, so the base form stands in
ORDER = ["species", "ability", "item", "move 1", "move 2", "move 3", "move 4", "nature", "HP", "Atk", "Def", "SpA", "SpD", "Spe"]
FIELD_LABEL = {"nature": "Stat Alignment", "move 1": "move", "move 2": "move", "move 3": "move", "move 4": "move"}


def sid(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def whimsicott_check(draw):
    return next(c for c in draw["species_checks"] if c["true"] == "Whimsicott")


def beads(shares, total=100):
    """Whole beads per share by largest remainder, so the beads always add up to `total`."""
    scale = total / sum(shares)
    raw = [s * scale for s in shares]
    out = [math.floor(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


def state(probe, draws_key, loss_key, item):
    draws = probe[draws_key]
    check = whimsicott_check(draws[3])
    watch = [dict(w, name=DISPLAY.get(w["species"], w["species"]), sprite=sid(w["species"])) for w in check["watch"]]
    other_alone = 1 - sum(w["prob_model_alone"] for w in watch)
    other_filter = 1 - sum(w["prob_with_filter"] for w in watch)
    rows = watch[:8] + [dict(name="Every other species", sprite=None, allowed=other_filter > 1e-4, prob_model_alone=other_alone,
                             prob_with_filter=max(other_filter, 0.0), logit=None, other=True)] + watch[8:]
    above = beads([r["prob_model_alone"] for r in rows])
    below = beads([r["prob_with_filter"] for r in rows])
    for r, a, b in zip(rows, above, below):
        r["beads_alone"], r["beads_filter"] = a, b
    order = [p["species"] for p in draws[3]["pokemon"]]            # the fourth draw's slot order, used for every draw
    out_draws = []
    for d in draws:
        by = {p["species"]: p for p in d["pokemon"]}
        mons = []
        for sp in order:
            p = by[sp]
            mons.append(dict(species=DISPLAY.get(sp, sp), sprite=sid(sp), item=p["item"] if sp != "Whimsicott" else item,
                             item_sprite=sid(p["item"] if sp != "Whimsicott" else item),
                             fields=[dict(name=FIELD_LABEL.get(f["name"], f["name"]), value=(item if (sp == "Whimsicott" and f["name"] == "item") else DISPLAY.get(f["value"], f["value"])),
                                          hidden=f["hidden"], cost=f["cost"], ce=f["ce_with_mask"], ce_alone=f["ce_model_alone"],
                                          weight=f["weight"]) for f in p["fields"]]))
        out_draws.append(dict(draw=d["draw"], t=d["t"], hidden=d["hidden"], cost=d["cost"], pokemon=mons))
    d4 = draws[3]
    return dict(item=item, item_sprite=sid(item), draws=out_draws, team_loss=probe[loss_key],
                check=dict(rows=rows, n_allowed=check["n_allowed"], true_allowed=check["true_allowed"],
                           p_alone=check["true_prob_model_alone"], p_filter=check["true_prob_with_mask"],
                           ce_alone=check["ce_model_alone"], ce_filter=check["ce_with_mask"], cost=check["cost"],
                           true_logit=check["true_logit"], logsumexp_allowed=check["logsumexp_allowed"],
                           allowed_share=check["allowed_share_model_alone"],
                           scale=16.0 / d4["t"] / 84, others_cost=d4["cost"] - check["cost"], others_n=d4["hidden"] - 1))


def main():
    probe = json.loads((HERE / "probe_worst_check.json").read_text())
    per_team = json.loads((HERE / "per_team_validation_loss.json").read_text())
    clean, eight = per_team["clean"], per_team["eight"]
    n = len(clean) + len(eight)
    assert n == probe["validation_teams"]
    stone = state(probe, "draws", "team_loss", "Glimmoranite")
    sash = state(probe, "what_if_draws", "what_if_team_loss", probe["what_if_item"])
    total = sum(clean) + sum(e["loss"] for e in eight)
    this = next(e for e in eight if e["team"] == probe["row"])
    other_seven = sum(e["loss"] for e in eight) - this["loss"]
    for st in (stone, sash):
        st["validation_loss"] = (sum(clean) + other_seven + st["team_loss"]) / n
        st["team_share"] = st["team_loss"] / n
    assert abs(stone["team_loss"] - this["loss"]) < 0.01
    teams = {t["row"]: t["pokemon"] for t in probe["eight_teams"]}
    eight_rows = [dict(row=e["team"], loss=e["loss"], draws=e["draws"], bad=e["bad"], this=e["team"] == probe["row"],
                       pokemon=[dict(species=DISPLAY.get(p["species"], p["species"]), sprite=FALLBACK_SPRITE.get(sid(p["species"]), sid(p["species"])),
                                     item=p["item"], item_sprite=sid(p["item"]), wrong=p["wrong_stone"]) for p in teams[e["team"]]])
                  for e in sorted(eight, key=lambda e: -e["loss"])]
    spiked = [e for e in eight if e["loss"] > 50]            # a draw split the pair: the team's loss is in the hundreds
    data = dict(
        row=probe["row"], checkpoint=probe["checkpoint"].replace(".pt", ""), n_species=probe["n_species"], fields=probe["fields"],
        label=probe["win_rate_label"], label_wins=int(probe["label_wins"]), label_battles=int(probe["label_battles"]),
        validation_teams=n, others_share=sum(clean) / n, others_mean=sum(clean) / len(clean), other_seven_share=other_seven / n,
        eight_share=sum(e["loss"] for e in eight) / n, validation_loss=total / n, clean_min=min(clean), clean_max=max(clean),
        states=dict(stone=stone, sash=sash), eight=eight_rows, weights=probe["field_weights"],
        spiked_n=len(spiked), spiked_share=sum(e["loss"] for e in spiked) / n, spiked_min=min(e["loss"] for e in spiked), spiked_max=max(e["loss"] for e in spiked),
        logged_training_loss=per_team["logged_training_loss"], logged_epoch=per_team["logged_epoch"],
    )
    need = {m["sprite"] for st in (stone, sash) for m in st["draws"][0]["pokemon"]} | {r["sprite"] for r in stone["check"]["rows"] if r["sprite"]}
    need |= {p["sprite"] for e in eight_rows for p in e["pokemon"]}
    data["sprites"] = {k: uri(SPRITES / f"{k}.png") for k in sorted(need)}
    items = {m["item_sprite"] for st in (stone, sash) for m in st["draws"][0]["pokemon"]} | {p["item_sprite"] for e in eight_rows for p in e["pokemon"]}
    data["items"] = {k: uri(ITEMS / f"item-{k}.png") for k in sorted(items)}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    for template, out in (("template.html", "index.html"), ("storyboard.template.html", "storyboard.html")):
        page = (HERE / template).read_text().replace("/*DATA*/null", blob)
        (HERE / out).write_text(page)
    print(f"wrote index.html and storyboard.html ({len(page) / 1024:.0f} KB): validation loss {data['validation_loss']:.4f}, what-if {sash['validation_loss']:.4f}, "
          f"{len(data['sprites'])} sprites, {len(data['items'])} item icons")


if __name__ == "__main__":
    main()
