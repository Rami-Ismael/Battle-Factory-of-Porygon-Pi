// Experimental actor colocated with Showdown. Python still provides reference
// observations, masks, and action sampling: this does NOT eliminate round trips.
'use strict';
const path=require('path'),crypto=require('crypto'),readline=require('readline');
const {BattleStream,getPlayerStreams,Teams}=require(path.join(process.argv[2],'dist/sim'));
const {TeamValidator}=require(path.join(process.argv[2],'dist/sim/team-validator'));
const ort=require(path.join(process.argv[4],'node_modules/onnxruntime-node'));
const battles=new Map(),send=x=>process.stdout.write(JSON.stringify(x)+'\n');
function seed(n){const b=crypto.createHash('sha256').update(String(n)).digest();return [0,2,4,6].map(i=>b.readUInt16BE(i));}
async function start(job){
  const parsed=[Teams.import(job.a),Teams.import(job.b)];
  for(const team of parsed){const errors=new TeamValidator(job.format).validateTeam(team);if(errors)throw new Error(errors.join('; '));}
  const stream=new BattleStream(),streams=getPlayerStreams(stream);battles.set(job.id,{stream,streams});
  const outputs=['p1','p2'].map(async side=>{
    let first=true;
    for await(const chunk of streams[side]){send({kind:'chunk',id:job.id,side,chunk:(first?'|init|battle\n':'')+chunk});first=false;}
    send({kind:'end',id:job.id,side});
  });
  const hash=crypto.createHash('sha256');
  outputs.push((async()=>{
    for await(const chunk of streams.omniscient)for(const line of chunk.split('\n'))if(!line.startsWith('|t:|'))hash.update(line+'\n');
    send({kind:'trace',id:job.id,trace:hash.digest('hex')});
  })());
  Promise.all(outputs).then(async()=>{await stream.writeEnd();battles.delete(job.id);})
    .catch(e=>send({kind:'error',id:job.id,error:e.stack}));
  await streams.omniscient.write(`>start ${JSON.stringify({formatid:job.format,seed:seed(job.seed)})}\n`+
    `>player p1 ${JSON.stringify({name:'Candidate',team:Teams.pack(parsed[0])})}\n`+
    `>player p2 ${JSON.stringify({name:'Opponent',team:Teams.pack(parsed[1])})}`);
  await streams.omniscient.write('>show-openteamsheets');
}
(async()=>{
  const before=performance.now();
  const actor=await ort.InferenceSession.create(process.argv[3],{
    executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
  send({kind:'ready',node:process.version,actor_load_seconds:(performance.now()-before)/1000});
  for await(const line of readline.createInterface({input:process.stdin})){
    const message=JSON.parse(line);
    try{
      if(message.kind==='infer'){
        const bytes=Buffer.from(message.data,'base64');
        const [batch,width]=message.shape;
        if(!Number.isInteger(batch)||batch<1||!Number.isInteger(width)||width<1||bytes.length!==batch*width*4)
          throw new Error('Invalid actor input shape');
        const input=new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
        const output=await actor.run({observations:new ort.Tensor('float32',input,[batch,width])});
        const logits=output.logits;
        send({kind:'logits',id:message.id,shape:logits.dims,
          data:Buffer.from(logits.data.buffer,logits.data.byteOffset,logits.data.byteLength).toString('base64')});
      }else if(message.kind==='start')await start(message);
      else if(message.kind==='choice'){
        if(message.choice.startsWith('/acceptopenteamsheets')||message.choice.startsWith('/leave '))continue;
        const battle=battles.get(message.id);if(!battle)throw new Error('Choice for missing battle');
        let choice=message.choice;
        if(choice.startsWith('/choose '))choice=choice.slice(8);
        else if(choice.startsWith('/team '))choice='team '+choice.slice(6);
        else throw new Error('Unsupported transport command '+choice);
        await battle.streams[message.side].write(choice);
      }
    }catch(e){send({kind:'error',id:message.id,error:e.stack});}
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
