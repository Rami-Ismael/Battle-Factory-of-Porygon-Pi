import urllib.request,json,pathlib,struct,math,time
ROOT=pathlib.Path('research_notes/Masked model checkpoint candidates/qwen')
def get(url,label,limit=20000000,headers=None):
 d=ROOT/label;d.mkdir()
 req=urllib.request.Request(url,headers=headers or {})
 with urllib.request.urlopen(req,timeout=45) as r:
  data=r.read(limit+1); meta={'url':url,'status':r.status,'headers':dict(r.headers)}
 if len(data)>limit: raise RuntimeError('limit exceeded')
 (d/'data').write_bytes(data);(d/'provenance.json').write_text(json.dumps(meta,indent=2));return data
models=json.loads(get('https://huggingface.co/api/models?author=dllm-hub&limit=100','model_inventory'))
print('\n'.join(m['id'] for m in models))
for family in ['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B']:
 for style in ['mdlm','bd3lm']:
  id=f'dllm-hub/{family}-diffusion-{style}-v0.1'; tag=family+'-'+style
  try: meta=json.loads(get('https://huggingface.co/api/models/'+id+'?blobs=true',tag+'-api'))
  except Exception as e:print(id,e);continue
  sha=meta['sha'];print(id,sha,[(x['rfilename'],x.get('size')) for x in meta['siblings']])
  (ROOT/(tag+'-summary.json')).write_text(json.dumps({'id':id,'sha':sha},indent=2))
  for file in ['config.json','tokenizer_config.json','special_tokens_map.json','LICENSE','README.md']:
   try: data=get(f'https://huggingface.co/{id}/resolve/{sha}/{file}',tag+'-'+file.replace('.','_'));print(file,data[:2000].decode(errors='replace') if file=='config.json' else len(data))
   except Exception as e:print(file,e)
  if style=='mdlm':
   for s in meta['siblings']:
    file=s['rfilename']
    if file.endswith('.py') or file=='tokenizer.json':
     get(f'https://huggingface.co/{id}/resolve/{sha}/{file}',tag+'-'+file.replace('/','_').replace('.','_'))
    if file.endswith('.safetensors'):
     url=f'https://huggingface.co/{id}/resolve/{sha}/{file}?header=1';prefix=get(url,tag+'-'+file+'-prefix',limit=8,headers={'Range':'bytes=0-7'})
     n=struct.unpack('<Q',prefix)[0];hdr=json.loads(get(url+'&part=2',tag+'-'+file+'-header',limit=n,headers={'Range':f'bytes=8-{7+n}'}))
     tensors={k:v for k,v in hdr.items() if k!='__metadata__'};total=sum(math.prod(v['shape']) for v in tensors.values()); emb={k:math.prod(v['shape']) for k,v in tensors.items() if 'embed' in k or 'lm_head' in k};print('COUNTS',total,emb,'NONEMB',total-sum(emb.values()),'DTYPE',set(v['dtype'] for v in tensors.values()))
