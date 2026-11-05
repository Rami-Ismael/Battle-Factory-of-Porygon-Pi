import urllib.request,tempfile,pathlib,json
root=pathlib.Path(__file__).parent
jobs=[('https://huggingface.co/diffusionfamily/diffugpt-s/resolve/391d9850148c0beda355a8874bea958a40164c74/tokenizer.json','gpt-small-tok'),('https://huggingface.co/diffusionfamily/diffullama/resolve/31b22850c7b397d6948ea04076099b9ff7110adc/tokenizer.json','llama-tok'),('https://api.github.com/repos/meta-llama/llama/commits/llama_v2','llama-license-rev'),('https://api.github.com/repos/openai/gpt-2/commits/master','gpt-license-rev')]
for url,name in jobs:
 d=pathlib.Path(tempfile.mkdtemp(prefix=name+'-',dir=root));b=urllib.request.urlopen(url).read();(d/'data').write_bytes(b);(d/'source.json').write_text(json.dumps({'url':url}));j=json.loads(b)
 if name=='gpt-small-tok':print('MASK small',j['model']['vocab'].get('------------'))
 elif name=='llama-tok':print('MASK llama',[(i,v) for i,v in enumerate(j['model']['vocab']) if v[0]=='ти'])
 else:print(name,j['sha'])
