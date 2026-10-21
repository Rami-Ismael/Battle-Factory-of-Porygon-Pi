"""Per-Pokémon and per-team features from a canonical paste."""
import json, re
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
DEX = json.load(open(DATA / "pokedex.json"))
MOVES = json.load(open(DATA / "moves.json"))
nid = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())

PROTECT = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap", "burningbulwark", "obstruct", "maxguard"}
SPEED_CTRL = {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "scaryface", "bulldoze", "rocktomb", "stringshot"}
SUPPORT = {"fakeout", "followme", "ragepowder", "helpinghand", "wideguard", "quickguard", "allyswitch", "spore", "encore",
           "taunt", "reflect", "lightscreen", "auroraveil", "partingshot", "willowisp", "snarl", "faketears"}
SPREAD = {"allAdjacentFoes", "allAdjacent"}
TYPES = None  # filled lazily


def parse(paste):
    mons = []
    for block in paste.strip().split("\n\n"):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        head = lines[0]
        name, _, item = head.partition(" @ ")
        m = re.search(r"\(([^)]+)\)", name)
        species = m.group(1) if m and len(m.group(1)) > 1 else name.strip()
        mon = dict(species=species.strip(), item=item.strip(), ability="", nature="", moves=[], evs={})
        for l in lines[1:]:
            if l.startswith("Ability:"): mon["ability"] = l[8:].strip()
            elif l.endswith("Nature"): mon["nature"] = l[:-6].strip()
            elif l.startswith("- "): mon["moves"].append(l[2:].strip())
            elif l.startswith("EVs:"):
                for part in l[4:].split("/"):
                    v, s = part.split(); mon["evs"][s] = int(v)
        mons.append(mon)
    return mons


def effective_species(mon):
    """Mega forme when the item is its stone, else the listed forme."""
    sid = nid(mon["species"])
    it = mon["item"]
    if it.endswith("ite") or re.search(r"ite [XY]$", it):
        base = DEX.get(sid, {}).get("baseSpecies", mon["species"])
        suffix = nid(it.split()[-1]) if re.search(r" [XY]$", it) else ""
        for cand in (nid(base) + "mega" + suffix, sid + "mega" + suffix):
            if cand in DEX: return cand, True
    return sid, False


def mon_features(mon):
    sid, mega = effective_species(mon)
    d = DEX.get(sid) or DEX.get(nid(mon["species"].split("-")[0]), {})
    bs = d.get("baseStats", {})
    types = set(d.get("types", []))
    mv = [MOVES.get(nid(x), {}) for x in mon["moves"]]
    dmg = [m for m in mv if m.get("category") in ("Physical", "Special")]
    stab = [m for m in dmg if m.get("type") in types and m.get("basePower", 0) >= 60]
    phys = sum(m.get("category") == "Physical" for m in dmg)
    spec = len(dmg) - phys
    # attacks that use the Pokémon's better attacking stat
    better = "Physical" if bs.get("atk", 0) >= bs.get("spa", 0) else "Special"
    ids = {nid(x) for x in mon["moves"]}
    return dict(
        species=sid, mega=int(mega), known=int(bool(bs)),
        bst=sum(bs.values()) if bs else None, spe=bs.get("spe"), hp=bs.get("hp"),
        off=max(bs.get("atk", 0), bs.get("spa", 0)) if bs else None,
        bulk=(bs["hp"] * (bs["def"] + bs["spd"]) / 2) ** 0.5 if bs else None,
        n_dmg=len(dmg), n_stab=len(stab), n_status=len(mv) - len(dmg),
        protect=int(bool(ids & PROTECT)), speed_ctrl=int(bool(ids & SPEED_CTRL)), support=int(bool(ids & SUPPORT)),
        spread=sum(m.get("target") in SPREAD for m in dmg),
        matched_stat=sum(m.get("category") == better for m in dmg),
        split_attacker=int(phys > 0 and spec > 0),
        bp=max([m.get("basePower", 0) for m in dmg], default=0),
        neutral_nature=int(mon["nature"] in ("Hardy", "Docile", "Serious", "Bashful", "Quirky")),
        ev_max=max(mon["evs"].values()) if mon["evs"] else 0,
    )


def team_features(paste):
    ms = [mon_features(m) for m in parse(paste)]
    keys = [k for k in ms[0] if k not in ("species",)]
    out = {}
    for k in keys:
        vals = [m[k] for m in ms if m[k] is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    out["n_protect"] = sum(m["protect"] for m in ms)
    return out, ms
