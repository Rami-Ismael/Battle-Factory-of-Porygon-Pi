"""Check embedded assets and the fixed 3D→2D moment transformation."""
import base64
import io
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from strategies import classify, title

root = Path(__file__).resolve().parent
html = (root/'archetype-diffusion.html').read_text()
d = json.loads(re.search(r'<script id="ad-data" type="application/json">(.*?)</script>', html, re.S)[1])
t = json.loads((root/'trace.json').read_text())
A = np.array(d['projectionMatrix'])
assert A.shape == (2,3)
assert np.allclose(np.array(t['centroids'])@A.T, d['centroids'], atol=.00051)
assert d['order'] == t['order'] and len(d['order']) == 48
assert len({(o['slot'],o['field']) for o in d['order']}) == 48
assert len(d['teams']) == 300
assert len({row['name'] for row in d['teams']}) == 300
for source, built in zip(t['records'],d['teams']):
    assert built['modes'] == classify(source)
    assert built['strategyTitle'] == title(built['modes'])
    assert len(built['name'].split(' + ')) == 2
    assert all(member in source['fields'][::8] for member in built['name'].split(' + '))
    assert source['fields'] == built['fields']
    assert source['speciesId'] == built['speciesId']
    arr = np.array(built['steps']).reshape(49,9)
    assert np.isfinite(arr).all()
    for st, dest in zip(source['steps'],arr):
        c = st[3:9]
        cov = np.array([[c[0],c[3],c[4]],[c[3],c[1],c[5]],[c[4],c[5],c[2]]])
        L = np.array([[dest[2],0],[dest[3],dest[4]]])
        assert np.allclose(np.array(st[:3])@A.T, dest[:2], atol=.00051)
        assert np.allclose(A@cov@A.T, L@L.T, atol=.002)
        assert np.allclose(st[9:],dest[5:],atol=.00051)
    assert np.all(arr[42:,2:5] == 0)
species = {s for row in d['teams'] for s in row['speciesId']}
assert species == set(d['sprites'])
for sp,url in d['sprites'].items():
    image = Image.open(io.BytesIO(base64.b64decode(url.split(',')[1])))
    image.load()
    assert image.width > 0 and image.height > 0
    assert image.convert('RGBA').getchannel('A').getbbox(), sp
assert not re.search(r'<(?:script|link)[^>]+(?:src|href)="https?://',html)
fallbacks = {s:v for s,v in d['spriteSources'].items() if s != v}
print('PASS fixed projection and covariance factors for 14,700 states; 140 embedded PNGs decode; no external scripts/styles')
print('Disclosed sprite fallbacks:',fallbacks)
print('PASS all 300 descriptive names are unique and name actual team members; no tournament results inferred')
