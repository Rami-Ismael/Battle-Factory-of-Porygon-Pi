"""Map a Pokemon Showdown species id (e.g. "charizardmegay", the store's file stem) to a PokeAPI sprite URL.

    from pokeapi_sprites import pokeapi_url          # add this folder to sys.path first
    pokeapi_url("charizardmegay")                     # -> ".../PokeAPI/sprites/master/sprites/pokemon/10035.png"
    pokeapi_url("charizardmegay", kind="home")        # -> ".../sprites/pokemon/other/home/10035.png" (512x512 render)

The table lives in pokeapi_ids.json beside this file; non-Python readers can use it directly
(species[<showdown id>]["sprite"] goes into url_templates[kind]). An unknown id raises KeyError.
Rebuild the table, keeping existing entries and adding any extra ids given:
    python3 pokeapi_sprites.py --rebuild [extra showdown ids...]
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "pokeapi_ids.json"
BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/"
URL_TEMPLATES = {"default": BASE + "{sprite}.png", "home": BASE + "other/home/{sprite}.png"}
API = "https://pokeapi.co/api/v2/"

# Showdown ids whose PokeAPI name is not the Showdown id with punctuation restored. Each target
# name was read off the PokeAPI pokemon list (or a species' "forms" list) on 2026-11-09.
# Showdown's bare base name means the base forme; PokeAPI often spells that forme out.
OVERRIDES = {
    "aegislash": "aegislash-shield",
    "basculegion": "basculegion-male",
    "basculegionf": "basculegion-female",
    "gourgeist": "gourgeist-average",
    "lycanroc": "lycanroc-midday",
    # Showdown's base Maushold is the family of three; PokeAPI's default (925) is the family of four.
    "maushold": "maushold-family-of-three",
    "mausholdfour": "maushold-family-of-four",
    "meowstic": "meowstic-male",
    "meowsticf": "meowstic-female",
    "meowsticfmega": "meowstic-female-mega",
    "mimikyu": "mimikyu-disguised",
    "morpeko": "morpeko-full-belly",
    "palafin": "palafin-zero",
    "pyroar": "pyroar-male",
    "taurospaldeaaqua": "tauros-paldea-aqua-breed",
    "taurospaldeacombat": "tauros-paldea-combat-breed",
    # Cosmetic formes exist only as PokeAPI pokemon-forms; Showdown ignores Alcremie's sweet,
    # so the strawberry sweet (the default) stands in.
    "alcremiecaramelswirl": "alcremie-caramel-swirl-strawberry-sweet",
    "alcremielemoncream": "alcremie-lemon-cream-strawberry-sweet",
    "alcremiematchacream": "alcremie-matcha-cream-strawberry-sweet",
    "alcremiemintcream": "alcremie-mint-cream-strawberry-sweet",
    "alcremierainbowswirl": "alcremie-rainbow-swirl-strawberry-sweet",
    "sinistchamasterpiece": "sinistcha-masterpiece",
    "vivilloncontinental": "vivillon-continental",
    "vivillonelegant": "vivillon-elegant",
    "vivillonfancy": "vivillon-fancy",
    "vivillonhighplains": "vivillon-high-plains",
    "vivillonjungle": "vivillon-jungle",
    "vivillonmodern": "vivillon-modern",
    "vivillonmonsoon": "vivillon-monsoon",
    "vivillonpolar": "vivillon-polar",
    "vivillonriver": "vivillon-river",
    "vivillonsandstorm": "vivillon-sandstorm",
    "vivillonsavanna": "vivillon-savanna",
}

_table = None


def table():
    """Return the parsed pokeapi_ids.json (cached)."""
    global _table
    if _table is None:
        _table = json.loads(TABLE.read_text(encoding="utf-8"))
    return _table


def entry(showdown_id):
    """Return {"name", "id", "sprite", ...} for a Showdown id; KeyError if unmapped."""
    return table()["species"][showdown_id]


def pokeapi_url(showdown_id, kind="default"):
    """PokeAPI sprite URL for a Showdown id. kind: "default" (96x96 front) or "home" (512x512)."""
    return URL_TEMPLATES[kind].format(sprite=entry(showdown_id)["sprite"])


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _store_ids():
    ids = {p.stem for p in HERE.glob("*.png")}
    model = HERE.parent / "model" / "sprites.json"
    if model.exists():
        ids |= {row["species"] for row in json.loads(model.read_text(encoding="utf-8"))["sources"]}
    return ids


def rebuild(extra_ids=()):
    """Resolve every store id (plus extra_ids and existing entries) against the live PokeAPI."""
    old = json.loads(TABLE.read_text(encoding="utf-8"))["species"] if TABLE.exists() else {}
    ids = sorted(_store_ids() | set(extra_ids) | set(old))
    pokemon = {r["name"]: int(r["url"].rstrip("/").split("/")[-1])
               for r in _get(API + "pokemon?limit=3000")["results"]}
    by_alnum = {}
    for name in pokemon:
        by_alnum.setdefault(_alnum(name), []).append(name)
    species, unresolved = {}, {}
    for sid in ids:
        name = OVERRIDES.get(sid)
        if name is None:
            hits = by_alnum.get(sid, [])
            if len(hits) != 1:
                unresolved[sid] = f"{len(hits)} PokeAPI names match after stripping punctuation"
                continue
            name = hits[0]
        if name in pokemon:
            species[sid] = {"name": name, "id": pokemon[name], "sprite": str(pokemon[name])}
            continue
        # Not a pokemon entry: look it up as a pokemon-form and take the sprite stem PokeAPI reports.
        try:
            form = _get(API + "pokemon-form/" + name)
        except OSError as e:
            unresolved[sid] = f"{name}: not a PokeAPI pokemon or pokemon-form ({e})"
            continue
        url = form["sprites"]["front_default"] or form["sprites"]["other"]["home"]["front_default"]
        pid = int(form["pokemon"]["url"].rstrip("/").split("/")[-1])
        species[sid] = {"name": name, "id": pid, "form": True, "sprite": url.rsplit("/", 1)[1][:-4]}
    TABLE.write_text(json.dumps({
        "about": "Showdown species id -> PokeAPI sprite. URL = url_templates[kind] with {sprite} "
                 "replaced by species[id].sprite; id is the PokeAPI pokemon id; form=true marks a "
                 "cosmetic pokemon-form. Built by pokeapi_sprites.py --rebuild.",
        "retrieved_from": API + "pokemon?limit=3000",
        "url_templates": URL_TEMPLATES,
        "species": species,
        "unresolved": unresolved,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return species, unresolved


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "--rebuild":
        sys.exit(__doc__)
    done, missing = rebuild(sys.argv[2:])
    print(f"mapped {len(done)}; unresolved {len(missing)}")
    for sid, why in missing.items():
        print(f"  {sid}: {why}")
