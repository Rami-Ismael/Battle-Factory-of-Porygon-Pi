/* Real local masked-diffusion inference. No generation API is used. */
'use strict';
let manifest, session, loading;
const send=(type,payload={})=>postMessage({type,...payload});
async function prepare(){
  if(session)return;
  if(loading)return loading;
  loading=(async()=>{
    send('status',{message:'Loading the model and browser runtime…'});
    importScripts('./runtime/ort.wasm.min.js');
    ort.env.wasm.numThreads=1;
    ort.env.wasm.proxy=false;
    ort.env.wasm.wasmPaths=new URL('./runtime/',self.location.href).href;
    const response=await fetch('./model/manifest.json');
    if(!response.ok)throw new Error('Model manifest could not load. Serve the complete visual directory over HTTP.');
    manifest=await response.json();
    session=await ort.InferenceSession.create(new URL('./model/model.onnx',self.location.href).href,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    send('status',{message:'Loading the Pokémon legality checker…'});
    importScripts('./runtime/showdown-validator.js');
    send('ready',{manifest});
  })();
  try{await loading}catch(error){loading=null;session=null;throw error}
}
function randomSource(seed){let state=seed>>>0;return()=>{state+=0x6D2B79F5;let z=state;z=Math.imul(z^(z>>>15),z|1);z^=z+Math.imul(z^(z>>>7),z|61);return((z^(z>>>14))>>>0)/4294967296}}
function allowedIndices(col,row){
  const key=manifest.columns[col],size=manifest.sizes[key],slot=Math.floor(col/8),field=col%8;
  let ok=Array.from({length:size},(_,i)=>i>0);
  const species=row[slot*8];
  if(field===0){const used=new Set();for(let s=0;s<6;s++)if(s!==slot&&row[s*8])used.add(manifest.baseOf[row[s*8]]);for(let i=1;i<size;i++)if(used.has(manifest.baseOf[i]))ok[i]=false}
  else if(field===2){for(let s=0;s<6;s++)if(s!==slot&&row[s*8+2])ok[row[s*8+2]]=false;if(species){const base=manifest.baseOf[species],mega=manifest.vocab.species[species].includes('mega');for(let i=1;i<size;i++){const stone=manifest.vocab.item[i];if(Object.hasOwn(manifest.megaStoneOf,stone)&&(mega||manifest.megaStoneOf[stone]!==base))ok[i]=false}}}
  else if(field===1&&species&&manifest.abilityAllowed[species]){const set=new Set(manifest.abilityAllowed[species]);ok=ok.map((v,i)=>v&&set.has(i))}
  else if(field>=3&&field<=6){if(species&&manifest.moveAllowed[species]){const set=new Set(manifest.moveAllowed[species]);ok=ok.map((v,i)=>v&&set.has(i))}for(let j=3;j<7;j++)if(j!==field&&row[slot*8+j])ok[row[slot*8+j]]=false}
  if(!ok.some(Boolean)){if(field>=3&&field<=6&&species&&manifest.moveAllowed[species]){const set=new Set(manifest.moveAllowed[species]);ok=ok.map((_,i)=>set.has(i))}if(!ok.some(Boolean))ok[0]=true}
  return ok.flatMap((v,i)=>v?[i]:[]);
}
function sample(logits,indices,random){const max=Math.max(...indices.map(i=>logits[i]));const weights=indices.map(i=>Math.exp(logits[i]-max)),total=weights.reduce((a,b)=>a+b,0);if(!Number.isFinite(total)||total<=0)throw new Error('The model produced unusable sampling probabilities.');let threshold=random()*total;for(let i=0;i<indices.length;i++){threshold-=weights[i];if(threshold<0)return indices[i]}return indices.at(-1)}
function decode(row,spreads=null){const value=(slot,offset)=>{const col=slot*8+offset,index=row[col];if(!index)return null;const token=manifest.vocab[manifest.columns[col]][index];return manifest.displayNames[token]??token};return Array.from({length:6},(_,i)=>({slot:i+1,species:value(i,0),ability:value(i,1),item:value(i,2),moves:[3,4,5,6].map(c=>value(i,c)),nature:value(i,7),statPoints:spreads?.[i]??null,level:spreads?50:null}))}
function paste(team){return team.map(s=>`${s.species}${s.item?' @ '+s.item:''}\nAbility: ${s.ability}\nLevel: 50\nEVs: ${Object.entries(s.statPoints).filter(([,v])=>v).map(([k,v])=>`${v} ${k}`).join(' / ')}\n${s.nature} Nature\n${s.moves.map(m=>'- '+m).join('\n')}`).join('\n\n')+'\n'}
const fieldNames=['species','ability','item','move 1','move 2','move 3','move 4','nature'];
async function generateOne(seed){await prepare();const rng=randomSource(seed),row=Array(48).fill(0),started=performance.now();const frame=(label,kind,team,extra={})=>send('frame',{frame:{label,kind,team,row:row.slice(),...extra}});frame('All 48 fields masked','sample',decode(row));
  for(let step=0;step<manifest.order.length;step++){
    const col=manifest.order[step];
    const result=await session.run({x:new ort.Tensor('int64',BigInt64Array.from(row,BigInt),[1,48]),t:new ort.Tensor('float32',Float32Array.of(1-step/48),[1]),w:new ort.Tensor('int64',BigInt64Array.of(BigInt(manifest.condition)),[1])});
    const logits=result.logits.data.subarray(col*manifest.outputWidth,(col+1)*manifest.outputWidth),allowed=allowedIndices(col,row);
    row[col]=sample(logits,allowed,rng);
    for(const tensor of Object.values(result))tensor.dispose?.();
    frame(`Field ${step+1} / 48 · slot ${Math.floor(col/8)+1} ${fieldNames[col%8]}`,'sample',decode(row),{column:col,token:row[col],allowedCount:allowed.length});
  }
  const spreads=Array.from({length:6},(_,i)=>{const species=manifest.vocab.species[row[i*8]],donors=manifest.spreads[species];if(!donors?.length)throw new Error('No same-species stat-point donor is available for '+species);return {...donors[Math.floor(rng()*donors.length)]}});
  const team=decode(row,spreads),showdown=paste(team);frame('Copy same-species Stat Points','spreads',team);
  send('status',{message:'Checking the generated team against the format rules…'});
  const errors=PokemonTeamValidator.validateTeam(showdown);
  frame(errors.length?'Generated team failed validation':'Generated team passed validation','validation',team,{valid:errors.length===0,validationErrors:errors});
  return {team,showdown,valid:errors.length===0,validationErrors:errors,seed,seconds:(performance.now()-started)/1000,metadata:{condition:manifest.condition,temperature:1,format:PokemonTeamValidator.format,revision:PokemonTeamValidator.revision,modelSha256:manifest.modelSha256,checkpointSha256:manifest.checkpointSha256,parameters:manifest.parameters,training:manifest.training}};
}
async function generate(seed){const attempts=[];let result;for(let attempt=1;attempt<=5;attempt++){send('attempt',{attempt,maxAttempts:5});result=await generateOne((seed+attempt-1)>>>0);attempts.push({attempt,seed:result.seed,valid:result.valid,validationErrors:result.validationErrors});if(result.valid)break;send('status',{message:`Attempt ${attempt} failed validation. ${attempt<5?'Sampling another candidate…':'Stopping after five attempts.'}`})}send('complete',{...result,attempts,initialSeed:seed})}
self.onmessage=async({data})=>{try{if(data.type==='generate')await generate(data.seed);else if(data.type==='fixture'){await prepare();const row=data.x;const outputs=await session.run({x:new ort.Tensor('int64',BigInt64Array.from(row,BigInt),[1,48]),t:new ort.Tensor('float32',Float32Array.of(data.t),[1]),w:new ort.Tensor('int64',BigInt64Array.of(BigInt(data.w??manifest.condition)),[1])});send('fixture',{logits:Array.from(outputs.logits.data),allowed:manifest.order.map(col=>({col,indices:allowedIndices(col,row)}))})}}catch(error){send('error',{message:error.message||String(error)})}};
