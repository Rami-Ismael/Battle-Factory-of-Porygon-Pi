"""Rebuild the checked-in snapshot, retaining its embedded sprite assets."""
import base64
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
page = (HERE / 'index.html').read_text()
sprite_start = page.index('const SPRITES = ') + len('const SPRITES = ')
sprites, _ = json.JSONDecoder().raw_decode(page[sprite_start:])
data = json.loads((HERE / 'data.json').read_text())
packed = base64.b64encode(gzip.compress(json.dumps(data, separators=(',', ':')).encode(), 9, mtime=0)).decode()
template = (HERE / 'template.html').read_text()
page = template.replace('/*DATAGZ*/', packed).replace('/*SPRITES*/{}', json.dumps(sprites, separators=(',', ':')))
(HERE / 'index.html').write_text(page)
print('Rebuilt Mask Lab from data.json and template.html; existing embedded sprites preserved.')
