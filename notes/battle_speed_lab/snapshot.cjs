// Exact Showdown snapshots for opening-library / prefix-reuse experiments.
// RandomPlayerAI has no history-dependent policy state; its PRNG is checkpointed.
'use strict';
const path=require('path'),crypto=require('crypto'),readline=require('readline');
const root=process.argv[2];
const {Battle,extractChannelMessages}=require(path.join(root,'dist/sim/battle'));
const {Teams}=require(path.join(root,'dist/sim/teams'));
const {TeamValidator}=require(path.join(root,'dist/sim/team-validator'));
const {RandomPlayerAI}=require(path.join(root,'dist/sim/tools/random-player-ai'));
const prepared=new Map(),snapshots=new Map();
function seedFor(seed,label){const b=crypto.createHash('sha256').update(`${seed}:${label}`).digest();return [0,2,4,6].map(i=>b.readUInt16BE(i));}
function pack(paste,format){
  const key=format+'\0'+paste;if(prepared.has(key))return prepared.get(key);
  const team=Teams.import(paste),errors=new TeamValidator(format).validateTeam(team);
  if(errors)throw new Error(errors.join('; '));const result=Teams.pack(team);prepared.set(key,result);return result;
}
function play(job,options,saved=null,capture=false){
  let inbox=saved?structuredClone(saved.pending):[],battle;
  const choices=[];
  const send=(kind,data)=>{
    if(kind==='sideupdate'){
      const [side,...lines]=(Array.isArray(data)?data.join('\n'):String(data)).split('\n');
      for(const line of lines)inbox.push({side,line});
    }
  };
  const ai=['p1','p2'].map((side,i)=>new RandomPlayerAI({write:choice=>{choices.push({side,choice});}},
    {seed:saved?saved.policySeeds[i]:seedFor(job.seed,side),mega:1}));
  if(saved){battle=Battle.fromJSON(saved.battle);battle.restart(send);}
  else{
    battle=new Battle({formatid:options.format,seed:seedFor(job.seed,'battle'),send});
    battle.setPlayer('p1',{name:'Candidate',team:pack(job.a,options.format)});
    battle.setPlayer('p2',{name:'Opponent',team:pack(job.b,options.format)});
    battle.sendUpdates();
  }
  let snapshot=null,stateHash=null,boardHash=null,steps=0;
  const started=performance.now();
  try{
    while(!battle.ended){
      if(++steps>100000)throw new Error('Driver step limit exceeded');
      if(capture&&!snapshot&&battle.turn>=(options.prefixTurn||3)&&choices.length===0){
        snapshot=JSON.stringify({battle:battle.toJSON(),pending:inbox,
          policySeeds:ai.map(p=>p.prng.getSeed())});
        const state=JSON.parse(snapshot);
        state.battle.log=state.battle.log.filter(line=>!line.startsWith('|t:|'));
        stateHash=crypto.createHash('sha256').update(JSON.stringify(state)).digest('hex');
        const board=battle.sides.map(side=>side.active.map(p=>p?{
          species:p.species.id,hp:p.hp,maxhp:p.maxhp,status:p.status,boosts:p.boosts}:null));
        boardHash=crypto.createHash('sha256').update(JSON.stringify(board)).digest('hex');
      }
      if(choices.length){
        const {side,choice}=choices.shift();
        battle.choose(side,choice);battle.sendUpdates();
      }else if(inbox.length){
        const {side,line}=inbox.shift();
        ai[Number(side.slice(1))-1].receiveLine(line);
      }else throw new Error('No pending request in nonterminal battle');
    }
    const normalized=extractChannelMessages(battle.log.join('\n'),[-1])[-1]
      .filter(line=>!line.startsWith('|t:|')).join('\n')+'\n';
    return {id:job.id,turns:battle.turn,outcome:battle.winner==='Candidate'?'win':battle.winner==='Opponent'?'loss':'draw',
      trace:crypto.createHash('sha256').update(normalized).digest('hex'),
      snapshot,stateHash,boardHash,...(options.returnLogs?{log:normalized}:{}),simulationMilliseconds:performance.now()-started};
  }finally{battle.destroy();}
}
function key(job,options){return JSON.stringify([options.format,options.prefixTurn||3,job.seed,job.a,job.b]);}
function run(request){
  const start=performance.now(),before=process.cpuUsage();let bytes=0,hits=0,misses=0;
  const results=[];
  for(const job of request.jobs){
    const k=key(job,request);let result;
    if(request.mode==='build'){
      result=play(job,request,null,true);
      if(result.snapshot){snapshots.set(k,result.snapshot);bytes+=Buffer.byteLength(result.snapshot);}
    }else if(request.mode==='resume'&&snapshots.has(k)){
      hits++;result=play(job,request,JSON.parse(snapshots.get(k)),false);
    }else{
      if(request.mode==='resume')misses++;
      result=play(job,request);
    }
    delete result.snapshot;results.push(result);
  }
  const cpu=process.cpuUsage(before);
  return {seconds:(performance.now()-start)/1000,completed:results.length,turns:results.reduce((s,r)=>s+r.turns,0),
    results,hits,misses,snapshotBytes:bytes,storedSnapshots:snapshots.size,cpuSeconds:(cpu.user+cpu.system)/1e6,
    limitation:'Exact repeated-seed suffix queries. Reused prefixes do not create independent full-battle observations.'};
}
process.stdout.write(JSON.stringify({ready:true,node:process.version})+'\n');
(async()=>{for await(const line of readline.createInterface({input:process.stdin})){
  try{process.stdout.write(JSON.stringify(run(JSON.parse(line)))+'\n');}
  catch(e){process.stderr.write(e.stack+'\n');process.stdout.write(JSON.stringify({error:e.message})+'\n');}
}})().catch(e=>{console.error(e);process.exitCode=1;});
