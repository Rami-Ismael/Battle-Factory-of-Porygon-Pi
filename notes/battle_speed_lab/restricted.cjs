// Showdown reference for a deliberately narrow fixed-damage doubles kernel.
// Never route arbitrary VGC teams into this engine subset.
'use strict';
const path=require('path'),readline=require('readline');
const {Battle,extractChannelMessages}=require(path.join(process.argv[2],'dist/sim/battle'));
const {Teams}=require(path.join(process.argv[2],'dist/sim/teams'));
function team(evs,speeds){return evs.map((hp,i)=>({name:`Unit${i}`,species:'Chansey',ability:'Natural Cure',
  item:'',moves:['Seismic Toss'],nature:'Hardy',level:50,evs:{hp,atk:0,def:0,spa:0,spd:0,spe:speeds[i]},
  ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31}}));}
function play(job){
  if(job.hpEVs.length!==4||job.speedEVs.length!==4)throw new Error('Four EV entries required');
  const pending=[];
  const battle=new Battle({formatid:'gen9doublescustomgame',seed:[1,2,3,4],
    send:(kind,data)=>{if(kind==='sideupdate'){
      const [side,...lines]=String(data).split('\n');
      for(const line of lines)if(line.startsWith('|request|'))pending.push([side,JSON.parse(line.slice(9))]);
    }}});
  try{
    battle.setPlayer('p1',{name:'Candidate',team:Teams.pack(team(job.hpEVs.slice(0,2),job.speedEVs.slice(0,2)))});
    battle.setPlayer('p2',{name:'Opponent',team:Teams.pack(team(job.hpEVs.slice(2,4),job.speedEVs.slice(2,4)))});
    const initial=battle.sides.flatMap(s=>s.pokemon.map(p=>({hp:p.maxhp,speed:p.storedStats.spe})));
    battle.sendUpdates();
    let steps=0;
    while(!battle.ended){
      if(++steps>500||!pending.length)throw new Error(`Restricted driver stalled: turn=${battle.turn} request=${battle.requestState} log=${battle.log.slice(-4).join(';')}`);
      const [sideid,request]=pending.shift();if(request.wait)continue;
      const index=sideid==='p1'?0:1,side=battle.sides[index],opponent=battle.sides[1-index];
      let choice;
      if(request.teamPreview)choice='team 1, 2';
      else if(request.forceSwitch)throw new Error('Unsupported switch request');
      else{
        const target=opponent.active.findIndex(p=>p&&!p.fainted)+1;
        choice=side.active.map(p=>!p||p.fainted?'pass':`move 1 ${target}`).join(', ');
      }
      if(!battle.choose(sideid,choice))throw new Error(`Rejected restricted choice ${choice}`);
      battle.sendUpdates();
    }
    const lines=extractChannelMessages(battle.log.join('\n'),[-1])[-1];
    const events=[];let turn=0;
    for(const line of lines){
      if(line.startsWith('|turn|'))turn=Number(line.split('|')[2]);
      if(line.startsWith('|-damage|')){
        const parts=line.split('|'),match=parts[2].match(/^p([12])([ab]):/);
        if(!match)throw new Error('Unknown damage target');
        events.push([turn,(Number(match[1])-1)*2+(match[2]==='b'?1:0),parseInt(parts[3])]);
      }
    }
    return {id:job.id,initial,events,turns:battle.turn,
      hp:battle.sides.flatMap(s=>s.pokemon.map(p=>p.hp)),
      winner:battle.winner==='Candidate'?0:battle.winner==='Opponent'?1:-1};
  }finally{battle.destroy();}
}
process.stdout.write(JSON.stringify({ready:true})+'\n');
(async()=>{for await(const line of readline.createInterface({input:process.stdin})){
  try{
    const request=JSON.parse(line),start=performance.now();
    const results=request.jobs.map(play);
    process.stdout.write(JSON.stringify({seconds:(performance.now()-start)/1000,results})+'\n');
  }catch(e){console.error(e.stack);process.stdout.write(JSON.stringify({error:e.message})+'\n');}
}})();
