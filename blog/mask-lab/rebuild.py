"""Rebuild the checked-in snapshot from template.html and data.json.

Sprites are kept from the checked-in page. Opponent species that the page
lacks are added from the vault's Showdown sprite folder so the threat
constellation can draw every meta team.
"""
import base64
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SPRITE_DIR = REPO / 'notes' / 'Blog' / 'visuals' / 'search-loop' / 'sprites'
MAX_SPRITE_BYTES = 4000  # skip oversized sprites; the page falls back to the base form

page = (HERE / 'index.html').read_text()
sprite_start = page.index('const SPRITES = ') + len('const SPRITES = ')
sprites, _ = json.JSONDecoder().raw_decode(page[sprite_start:])
data = json.loads((HERE / 'data.json').read_text())

norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
added = []
for species in sorted({s for o in data['opponents'] for s in o['species']}):
    key = norm(species)
    if key in sprites:
        continue
    png = SPRITE_DIR / f'{key}.png'
    if png.exists() and png.stat().st_size <= MAX_SPRITE_BYTES:
        sprites[key] = 'data:image/png;base64,' + base64.b64encode(png.read_bytes()).decode()
        added.append(key)

packed = base64.b64encode(gzip.compress(json.dumps(data, separators=(',', ':')).encode(), 9, mtime=0)).decode()
template = (HERE / 'template.html').read_text()
page = template.replace('/*DATAGZ*/', packed).replace('/*SPRITES*/{}', json.dumps(sprites, separators=(',', ':')))
(HERE / 'index.html').write_text(page)
print(f'Rebuilt Mask Lab: {len(sprites)} sprites embedded ({len(added)} added: {", ".join(added) or "none"}).')
