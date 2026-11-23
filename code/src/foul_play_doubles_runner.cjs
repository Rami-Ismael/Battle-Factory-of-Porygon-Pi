// Battles between battle policies on the pinned Showdown runtime, one JSON message per line.
// A side whose policy is 'bc' is relayed to Python (foul_play_doubles_battles.py), which plays it
// with the behaviour-cloning policy through poke-env, as battle_speed_lab/bridge.cjs does; every
// other policy (foulplay, montecarlo, random) is played here by foul_play_doubles.cjs, which sees
// only its own side's stream, as a player would.
//   in : {kind:'start', id, a, b, seed, p1:{policy,...}, p2:{policy,...}}
//        {kind:'choice', id, side, choice}            (a 'bc' side's poke-env message)
//   out: {kind:'ready'} {kind:'chunk', id, side, chunk} {kind:'end', id, side}
//        {kind:'result', id, winner, turns, ms, stats:{p1, p2}} {kind:'error', id, error}
'use strict';
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');
const {Agent, RT, FORMAT} = require('./foul_play_doubles.cjs');
const {closeRustClient} = require('./rust_engine_client.cjs');
const {BattleStream, getPlayerStreams, Teams} = require(path.join(RT, 'dist/sim'));
const {TeamValidator} = require(path.join(RT, 'dist/sim/team-validator'));

const MAX_TURNS = 150;
const battles = new Map();
const send = x => process.stdout.write(JSON.stringify(x) + '\n');
const spreads = process.env.FOUL_PLAY_SPREADS ? require(process.env.FOUL_PLAY_SPREADS) : {};
function seed(n) {
  const bytes = crypto.createHash('sha256').update(String(n)).digest();
  return [0, 2, 4, 6].map(i => bytes.readUInt16BE(i));
}

async function start(job) {
  const parsed = [Teams.import(job.a), Teams.import(job.b)];
  for (const team of parsed) {
    const errors = new TeamValidator(FORMAT).validateTeam(team);
    if (errors) throw new Error(errors.join('; '));
  }
  const t0 = performance.now();
  const stream = new BattleStream(), streams = getPlayerStreams(stream);
  const state = {streams, agents: {}, turns: 0, winner: undefined};
  battles.set(job.id, state);
  const pastes = {p1: job.a, p2: job.b};
  const outputs = ['p1', 'p2'].map(async (side, k) => {
    const spec = job[side];
    if (spec.policy === 'bc' || spec.policy === 'bc_det') {
      let first = true;
      for await (const chunk of streams[side]) {
        send({kind: 'chunk', id: job.id, side, chunk: (first ? '|init|battle\n' : '') + chunk});
        first = false;
      }
      send({kind: 'end', id: job.id, side});
      return;
    }
    const agent = new Agent({...spec, side, paste: pastes[side], spreads, seed: (job.seed * 2 + k) >>> 0});
    state.agents[side] = agent;
    for await (const chunk of streams[side]) {
      // An unavailable choice (a move the agent could not know was disabled) is followed by a fresh
      // request, which the agent answers again; an invalid one gets Showdown's default.
      if (chunk.includes('|error|[')) {
        agent.stats.rejected = (agent.stats.rejected || []).concat(chunk.slice(0, 160));
        if (chunk.includes('|error|[Invalid choice]')) { agent.stats.fallbacks++; await streams[side].write('default'); }
        continue;
      }
      agent.receive(chunk);
      // A decision blocks this process while it searches; other battles here wait their turn.
      await new Promise(r => setImmediate(r));
      const choice = await agent.decideAsync();
      if (choice) await streams[side].write(choice);
    }
  });
  outputs.push((async () => {
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split('\n')) {
        if (line.startsWith('|turn|')) {
          state.turns = Number(line.split('|')[2]);
          if (state.turns > MAX_TURNS) await streams.omniscient.write('>forcetie');
        }
        if (line.startsWith('|win|')) state.winner = line.slice(5) === 'Candidate' ? 'p1' : 'p2';
        if (line === '|tie' || line === '|tie|') state.winner = null;
      }
    }
  })());
  Promise.all(outputs).then(async () => {
    await stream.writeEnd();
    battles.delete(job.id);
    const stats = Object.fromEntries(Object.entries(state.agents).map(([s, a]) => [s, a.stats]));
    send({kind: 'result', id: job.id, winner: state.winner, turns: state.turns, ms: performance.now() - t0, stats});
  }).catch(e => send({kind: 'error', id: job.id, error: e.stack}));
  await streams.omniscient.write(`>start ${JSON.stringify({formatid: FORMAT, seed: seed(job.seed)})}\n` +
    `>player p1 ${JSON.stringify({name: 'Candidate', team: Teams.pack(parsed[0])})}\n` +
    `>player p2 ${JSON.stringify({name: 'Opponent', team: Teams.pack(parsed[1])})}`);
  await streams.omniscient.write('>show-openteamsheets');
}

send({kind: 'ready', node: process.version});
const input = readline.createInterface({input: process.stdin});
(async () => {
  for await (const line of input) {
    const message = JSON.parse(line);
    try {
      if (message.kind === 'start') await start(message);
      else if (message.kind === 'choice') {
        if (message.choice.startsWith('/acceptopenteamsheets') || message.choice.startsWith('/leave ')) continue;
        const battle = battles.get(message.id);
        if (!battle) throw new Error('Choice for missing battle');
        let choice = message.choice;
        if (choice.startsWith('/choose ')) choice = choice.slice(8);
        else if (choice.startsWith('/team ')) choice = 'team ' + choice.slice(6);
        else throw new Error('Unsupported transport command ' + choice);
        await battle.streams[message.side].write(choice);
      }
    } catch (e) { send({kind: 'error', id: message.id, error: e.stack}); }
  }
  closeRustClient();
})().catch(e => { closeRustClient(); console.error(e); process.exitCode = 1; });
