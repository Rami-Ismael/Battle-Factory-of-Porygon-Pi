exec(open('/tmp/qwen_research.py').read().split('models=json.loads')[0])
a=json.loads(get('https://api.github.com/repos/ZHZisZZ/dllm/commits/main','dllm-revision'));sha=a['sha'];print('DLLM',sha)
tree=json.loads(get(f'https://api.github.com/repos/ZHZisZZ/dllm/git/trees/{sha}?recursive=1','dllm-tree'))
for item in tree['tree']:
 p=item['path']
 if ('sampl' in p or 'trainer' in p) and 'mdlm' in p and p.endswith('.py') or p=='LICENSE':
  data=get(f'https://raw.githubusercontent.com/ZHZisZZ/dllm/{sha}/{p}','dllm-'+p.replace('/','_'));print('FILE',p,len(data))
for family in ['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B']:
 id='Qwen/'+family;meta=json.loads(get('https://huggingface.co/api/models/'+id+'?blobs=true',family+'-upstream-api'));sha=meta['sha'];print('UPSTREAM',id,sha)
 for file in ['LICENSE','NOTICE','config.json']:
  try: get(f'https://huggingface.co/{id}/resolve/{sha}/{file}',family+'-upstream-'+file)
  except Exception as e:print(file,e)
for file in ['src/transformers/models/qwen2/modeling_qwen2.py','src/transformers/models/qwen3/modeling_qwen3.py','src/transformers/integrations/sdpa_attention.py','src/transformers/modeling_utils.py']:
 try:get('https://raw.githubusercontent.com/huggingface/transformers/v4.57.0/'+file,'transformers-'+file.replace('/','_'))
 except Exception as e:print(e)
for family in ['Qwen2.5-Coder-0.5B-Instruct','Qwen3-0.6B']:
 t=json.loads((ROOT/(family+'-mdlm-tokenizer_json')/'data').read_text());c=json.loads((ROOT/(family+'-mdlm-tokenizer_config_json')/'data').read_text());print('TOKENIZER',family,c['tokenizer_class'],c.get('model_max_length'),t['model']['type'],t['pre_tokenizer'],[(x['id'],x['content']) for x in t['added_tokens'] if 'mask' in x['content']])
