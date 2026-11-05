import urllib.request,json,pathlib,tempfile,struct,math,concurrent.futures
root=pathlib.Path(__file__).parent
repos={'Dream-org/Dream-v0-Base-7B':'6572adb5535263e4d1a337b56942ba48b6dee2a9','GSAI-ML/LLaDA-8B-Base':'0f2787f2d87eac5eed8a087d5ecd24277e6255b2','GSAI-ML/LLaDA-8B-Instruct':'08b83a6feb34df1a6011b80c3c00c7563e963b07'}
def fetch(repo,rev,f,header=False):
 u=f'https://huggingface.co/{repo}/resolve/{rev}/{f}';d=pathlib.Path(tempfile.mkdtemp(prefix=repo.split('/')[-1]+'-',dir=root));
 if header:
  req=urllib.request.Request(u,headers={'Range':'bytes=0-7'});r=urllib.request.urlopen(req);b=r.read(8);assert r.status==206;n=struct.unpack('<Q',b)[0];assert n<2000000
  req=urllib.request.Request(u+'?header=true',headers={'Range':f'bytes=8-{n+7}'});r=urllib.request.urlopen(req);assert r.status==206;b=r.read(n);j=json.loads(b);(d/(f+'.header.json')).write_bytes(b)
 else:
  b=urllib.request.urlopen(u).read();(d/f).write_bytes(b)
 (d/'source.txt').write_text(u+'\n');return f,b
for repo,rev in repos.items():
 api=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true')); files=[x['rfilename'] for x in api['siblings'] if x['rfilename'].endswith(('.json','.py')) and x['rfilename']!='README.md'];weights=[x for x in api['siblings'] if x['rfilename'].endswith('.safetensors')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  res=list(ex.map(lambda f:fetch(repo,rev,f),files));hdr=list(ex.map(lambda f:fetch(repo,rev,f['rfilename'],True),weights))
 tensors={k:v for _,b in hdr for k,v in json.loads(b).items() if k!='__metadata__'};total=sum(math.prod(v['shape']) for v in tensors.values());emb={k:v for k,v in tensors.items() if k in ['model.embed_tokens.weight','lm_head.weight','model.transformer.wte.weight','model.transformer.ff_out.weight']};print(repo,'total',total,'embeddings',[(k,v['shape']) for k,v in emb.items()],'dtypes',set(v['dtype'] for v in tensors.values()),'bytes',sum(x['size'] for x in weights));print('CONFIG',dict(res)['config.json'].decode())
 (root/(repo.split('/')[-1]+'-summary.json')).write_text(json.dumps({'repo':repo,'revision':rev,'parameters':total,'embedding_tensors':emb,'dtype':list(set(v['dtype'] for v in tensors.values())),'weight_file_bytes':sum(x['size'] for x in weights)},indent=2))
