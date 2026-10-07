"""Embed pool.json and the needed sprites into selection-temperature.html.

pool.json comes from ~/vgc-data/matchup_regmb.sqlite: the 16 teams of origin
'mdts:direct:1101' (direct samples from the masked diffusion model), with wins
and battles summed over the top-50 meta columns under policy 1.
"""
import base64, json, pathlib

here = pathlib.Path(__file__).parent
sprites_dir = here.parent / "search-loop" / "sprites"
pool = json.loads((here / "pool.json").read_text())

teams = []
species = set()
for t in pool:
    members = t["species_key"].split(",")
    species.update(members)
    teams.append({"id": t["team_id"], "members": members, "wins": t["w"], "battles": t["b"]})

sprites = {
    s: "data:image/png;base64," + base64.b64encode((sprites_dir / f"{s}.png").read_bytes()).decode()
    for s in sorted(species)
}
data = json.dumps({"teams": teams, "sprites": sprites}, separators=(",", ":"))
html = (here / "template.html").read_text().replace("/*__DATA__*/null", data)
(here / "selection-temperature.html").write_text(html)
print(f"{len(teams)} teams, {len(sprites)} sprites, {len(html):,} bytes")
