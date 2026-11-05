import urllib.request,json,pathlib,tempfile,struct,math
root=pathlib.Path(__file__).parent
manifest=[]
def get(url,name,limit=20000000,headers=None):
 d=pathlib.Path(tempfile.mkdtemp(prefix=name+'-',dir=root))
 req=urllib.request.Request(url,headers=headers or {})
 with urllib.request.urlopen(req,timeout=50) as r:
  if headers and r.status!=206: raise RuntimeError('range ignored')
  b=r.read(limit+1)
  if len(b)>limit:raise RuntimeError('too big')
 (d/'data').write_bytes(b);manifest.append({'url':url,'file':str((d/'data').relative_to(root)),'bytes':len(b)})
 return b
for model in ['diffusionfamily/diffugpt-s','diffusionfamily/diffugpt-m','diffusionfamily/diffullama']:
 info=json.loads(get('https://huggingface.co/api/models/'+model+'?blobs=true',model.split('/')[-1]+'-api'))
 rev=info['sha'];print(model,rev)
 print([(x['rfilename'],x.get('size')) for x in info['siblings']])
 for f in ['config.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','LICENSE']:
  if f not in [x['rfilename'] for x in info['siblings']]:continue
  data=get(f'https://huggingface.co/{model}/resolve/{rev}/{f}',f.replace('.','-')); print(f,data[:10000].decode())
 for x in info['siblings']:
  f=x['rfilename']
  if not f.endswith('.safetensors'):continue
  url=f'https://huggingface.co/{model}/resolve/{rev}/{f}'
  n=struct.unpack('<Q',get(url,'prefix',8,{'Range':'bytes=0-7'}))[0]
  hdr=json.loads(get(url,'header',n,{'Range':f'bytes=8-{7+n}'}))
  tensors={k:v for k,v in hdr.items() if k!='__metadata__'}
  total=sum(math.prod(v['shape']) for v in tensors.values()); emb={k:v for k,v in tensors.items() if any(t in k for t in ['wte','wpe','embed_tokens','lm_head'])}
  print('WEIGHTS',f,'filebytes',x.get('size'),'params',total,'dtype',set(v['dtype'] for v in tensors.values()),'embeddings',emb)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
