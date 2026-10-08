"""Vendor species sprites for the existing model vocabulary with source hashes."""
import json,re,subprocess,hashlib,struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parent.parent
m=json.loads((root/'model/manifest.json').read_text());out=root/'sprites'
def fetch(norm):
 name=m['displayNames'].get(norm,norm)
 slug=re.sub(r'[^a-z0-9-]','',name.lower())
 slug=slug.replace('-mega-x','-megax').replace('-mega-y','-megay').replace('kommo-o','kommoo').replace('tauros-paldea-aqua','tauros-paldeaaqua')
 url='https://play.pokemonshowdown.com/sprites/gen5/'+slug+'.png';p=out/(norm+'.png')
 if not p.exists():
  res=subprocess.run(['curl','-fsSL','--max-time','20',url],capture_output=True)
  if res.returncode or not res.stdout.startswith(b'\x89PNG\r\n\x1a\n'):return dict(species=norm,source=url,available=False)
  p.write_bytes(res.stdout)
 data=p.read_bytes();w,h=struct.unpack('>II',data[16:24])
 return dict(species=norm,file='./sprites/'+p.name,source=url,available=True,width=w,height=h,sha256=hashlib.sha256(data).hexdigest())
with ThreadPoolExecutor(max_workers=10) as pool:rows=list(pool.map(fetch,m['vocab']['species'][1:]))
(root/'model/sprites.json').write_text(json.dumps({'retrieved':'2026-09-13','sprites':{x['species']:x['file'] for x in rows if x['available']},'sources':rows},indent=2)+'\n')
print('available',sum(x['available'] for x in rows),'of',len(rows));print('missing',[x['species'] for x in rows if not x['available']])
