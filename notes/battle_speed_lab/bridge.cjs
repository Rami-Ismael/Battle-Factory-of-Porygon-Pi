// Multiplexed direct-engine transport. Uses Showdown's own player-view streams
// and open-team-sheet generation rather than reconstructing hidden battle state.
'use strict';
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');
const {BattleStream,getPlayerStreams,Teams} = require(path.join(process.argv[2],'dist/sim'));
const {TeamValidator} = require(path.join(process.argv[2],'dist/sim/team-validator'));
const battles = new Map();
const send = x => process.stdout.write(JSON.stringify(x)+'\n');
function seed(n) {
  const bytes = crypto.createHash('sha256').update(String(n)).digest();
  return [0,2,4,6].map(i=>bytes.readUInt16BE(i));
}
async function start(job) {
  const parsed = [Teams.import(job.a),Teams.import(job.b)];
  for (const team of parsed) {
    const errors = new TeamValidator(job.format).validateTeam(team);
    if (errors) throw new Error(errors.join('; '));
  }
  const stream = new BattleStream(), streams = getPlayerStreams(stream);
  battles.set(job.id,{stream,streams});
  const outputs = ['p1','p2'].map(async side => {
    let first = true;
    for await (const chunk of streams[side]) {
      send({kind:'chunk',id:job.id,side,chunk:(first?'|init|battle\n':'')+chunk});
      first=false;
    }
    send({kind:'end',id:job.id,side});
  });
  const hash = crypto.createHash('sha256');
  outputs.push((async()=>{
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split('\n')) if (!line.startsWith('|t:|')) hash.update(line+'\n');
    }
    send({kind:'trace',id:job.id,trace:hash.digest('hex')});
  })());
  Promise.all(outputs).then(async()=>{await stream.writeEnd();battles.delete(job.id);})
    .catch(e=>send({kind:'error',id:job.id,error:e.stack}));
  await streams.omniscient.write(`>start ${JSON.stringify({formatid:job.format,seed:seed(job.seed)})}\n`+
    `>player p1 ${JSON.stringify({name:'Candidate',team:Teams.pack(parsed[0])})}\n`+
    `>player p2 ${JSON.stringify({name:'Opponent',team:Teams.pack(parsed[1])})}`);
  // Flush initial state and team-preview requests before OTS triggers decisions.
  await streams.omniscient.write('>show-openteamsheets');
}
send({kind:'ready',node:process.version});
const input = readline.createInterface({input:process.stdin});
(async()=>{
  for await (const line of input) {
    const message = JSON.parse(line);
    try {
      if (message.kind==='start') await start(message);
      else if (message.kind==='choice') {
        if (message.choice.startsWith('/acceptopenteamsheets') || message.choice.startsWith('/leave ')) continue;
        const battle = battles.get(message.id);
        if (!battle) throw new Error('Choice for missing battle');
        let choice = message.choice;
        if (choice.startsWith('/choose ')) choice=choice.slice(8);
        else if (choice.startsWith('/team ')) choice='team '+choice.slice(6);
        else throw new Error('Unsupported transport command '+choice);
        await battle.streams[message.side].write(choice);
      }
    } catch(e) {send({kind:'error',id:message.id,error:e.stack});}
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
