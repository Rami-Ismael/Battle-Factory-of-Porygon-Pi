import urllib.request,tempfile,pathlib,json
root=pathlib.Path(__file__).parent
rev='c17e897f6476c174b4623da594e4c65554f1613d'
for f in ['inf_diffugpt.py','inf_diffullama.py','model.py','attention_patch.py','DiffuLLaMA-training/model_llama.py','LLaMA-Factory/src/llamafactory/train/ddm/model.py','LLaMA-Factory/src/llamafactory/attention_patch.py']:
 d=pathlib.Path(tempfile.mkdtemp(prefix=f.rsplit('/',1)[-1]+'-',dir=root))
 url=f'https://raw.githubusercontent.com/HKUNLP/DiffuLLaMA/{rev}/{f}'
 b=urllib.request.urlopen(url).read();(d/'data').write_bytes(b);(d/'source.json').write_text(json.dumps({'url':url}));print(f,d)
