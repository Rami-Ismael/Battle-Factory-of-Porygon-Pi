// Audit of the battle Foul Play for doubles searches from: at every decision of every battle, the
// sampled battle built from one side's view is compared with the real battle (BattleStream.battle),
// field by field. Hidden information (the foe's Stat Points, which unrevealed Pokémon they brought)
// is left out; everything a player can see must match.
//   node foul_play_doubles_audit.cjs teams.json battles [seed]   -> one JSON summary line
// teams.json is a list of Showdown pastes; battles pair them round robin with random play.
'use strict';
const path = require('path');
const fs = require('fs');
const {Agent, determinize, Spreads, mulberry, RT, FORMAT} = require('./foul_play_doubles.cjs');
const {BattleStream, getPlayerStreams, Teams} = require(path.join(RT, 'dist/sim'));

const VOLATILES = ['confusion', 'substitute', 'taunt', 'encore', 'yawn', 'leechseed', 'choicelock', 'stall'];

function compare(view, real, built, out) {
  const check = (key, ok, detail) => {
    out.checks[key] = (out.checks[key] || 0) + 1;
    if (!ok) {
      out.mismatches[key] = (out.mismatches[key] || 0) + 1;
      if (out.examples.filter(e => e.key === key).length < 4) {
        const name = detail.name;
        const lines = name ? view.lines.filter(l => l.includes(': ' + name)).slice(-4) : [];
        out.examples.push({key, turn: view.turn, clock: view.clock(), ...detail, lines});
      }
    }
  };
  const realSides = {[view.me]: real.sides[view.me === 'p1' ? 0 : 1], [view.foe]: real.sides[view.foe === 'p1' ? 0 : 1]};
  const builtSides = {[view.me]: built.sides[0], [view.foe]: built.sides[1]};
  for (const key of [view.me, view.foe]) {
    const mine = key === view.me;
    const rs = realSides[key], bs = builtSides[key];
    rs.active.forEach((rp, slot) => {
      const bp = bs.active[slot];
      const rName = rp && !rp.fainted ? rp.name : '-', bName = bp && !bp.fainted ? bp.name : '-';
      check(`${mine ? 'own' : 'foe'} active`, rName === bName, {slot, real: rName, built: bName});
      if (rName === '-' || rName !== bName) return;
      const frac = x => x.hp / x.maxhp;
      check(`${mine ? 'own' : 'foe'} hp`, mine ? rp.hp === bp.hp : Math.abs(frac(rp) - frac(bp)) <= 0.012,
        {name: rName, real: frac(rp).toFixed(3), built: frac(bp).toFixed(3)});
      check(`${mine ? 'own' : 'foe'} status`, (rp.status || '') === (bp.status || ''), {name: rName, real: rp.status, built: bp.status});
      check(`${mine ? 'own' : 'foe'} boosts`, ['atk', 'def', 'spa', 'spd', 'spe'].every(s => rp.boosts[s] === bp.boosts[s]),
        {name: rName, real: rp.boosts, built: bp.boosts});
      check(`${mine ? 'own' : 'foe'} species`, rp.species.name === bp.species.name, {real: rp.species.name, built: bp.species.name});
      check(`${mine ? 'own' : 'foe'} item`, !!rp.item === !!bp.item, {name: rName, real: rp.item, built: bp.item});
      for (const v of VOLATILES) {
        check(`volatile ${v}`, !!rp.volatiles[v] === !!bp.volatiles[v], {name: rName, real: !!rp.volatiles[v], built: !!bp.volatiles[v]});
      }
      check('fake out window', (rp.activeMoveActions > 0) === (bp.activeMoveActions > 0),
        {name: rName, real: rp.activeMoveActions, built: bp.activeMoveActions});
    });
    for (const id of new Set([...Object.keys(rs.sideConditions), ...Object.keys(bs.sideConditions)])) {
      const r = rs.sideConditions[id], b = bs.sideConditions[id];
      check(`side ${id}`, !!r === !!b && (!r || r.duration === b.duration), {real: r && r.duration, built: b && b.duration});
    }
    check(`${mine ? 'own' : 'foe'} Pokémon left`, rs.pokemonLeft === bs.pokemonLeft || !mine,
      {real: rs.pokemonLeft, built: bs.pokemonLeft});
  }
  const rf = real.field, bf = built.field;
  check('weather', rf.weather === bf.weather && (!rf.weather || rf.weatherState.duration === bf.weatherState.duration),
    {real: [rf.weather, rf.weatherState.duration], built: [bf.weather, bf.weatherState.duration]});
  check('terrain', rf.terrain === bf.terrain && (!rf.terrain || rf.terrainState.duration === bf.terrainState.duration),
    {real: [rf.terrain, rf.terrainState.duration], built: [bf.terrain, bf.terrainState.duration]});
  for (const id of new Set([...Object.keys(rf.pseudoWeather), ...Object.keys(bf.pseudoWeather)])) {
    const r = rf.pseudoWeather[id], b = bf.pseudoWeather[id];
    check(`field ${id}`, !!r === !!b && (!r || r.duration === b.duration), {real: r && r.duration, built: b && b.duration});
  }
  // The searched battle must offer the same decision: same request kind, same forced slots.
  const rr = realSides[view.me].activeRequest, br = built.sides[0].activeRequest;
  const kind = r => (!r ? 'none' : r.wait ? 'wait' : r.forceSwitch ? 'switch ' + r.forceSwitch.map(Number).join('') : 'move');
  check('request', kind(rr) === kind(br), {real: kind(rr), built: kind(br)});
}

async function battle(a, b, seed, spreads, out) {
  const stream = new BattleStream(), streams = getPlayerStreams(stream);
  const agents = {p1: new Agent({side: 'p1', paste: a, policy: 'random', seed: seed * 2, spreads}),
    p2: new Agent({side: 'p2', paste: b, policy: 'random', seed: seed * 2 + 1, spreads})};
  const rng = mulberry(seed);
  const sp = new Spreads(spreads);
  for (const a of Object.values(agents)) { a.view.keepLines = true; a.view.lines = []; }
  const play = Object.entries(agents).map(async ([side, agent]) => {
    for await (const chunk of streams[side]) {
      if (chunk.includes('|error|[')) {
        out.rejected.push(chunk.slice(0, 200));
        if (chunk.includes('|error|[Invalid choice]')) await streams[side].write('default');
        continue;
      }
      agent.receive(chunk);
      await new Promise(r => setImmediate(r));
      const req = agent.view.request;
      if (req && !req.wait && !req.teamPreview && !agent.view.ended && stream.battle) {
        out.decisions++;
        let built = null;
        try { built = determinize(agent.view, sp, null, rng); } catch (e) { out.errors.push(String(e.stack).slice(0, 400)); }
        if (built) compare(agent.view, stream.battle, built, out);
        else out.unbuilt++;
      }
      const choice = agent.decide();
      out.stall = {side, turn: agent.view.turn, request: req && (req.wait ? 'wait' : req.forceSwitch ? 'switch' : req.teamPreview ? 'preview' : 'move'), choice, chunk: chunk.slice(-300)};
      if (choice) await streams[side].write(choice);
    }
  });
  const pack = p => Teams.pack(Teams.import(p));
  await streams.omniscient.write(`>start ${JSON.stringify({formatid: FORMAT, seed: [seed, 1, 2, 3]})}\n` +
    `>player p1 ${JSON.stringify({name: 'A', team: pack(a)})}\n>player p2 ${JSON.stringify({name: 'B', team: pack(b)})}`);
  await streams.omniscient.write('>show-openteamsheets');
  (async () => { for await (const chunk of streams.omniscient) { if (chunk.includes('|turn|200')) await streams.omniscient.write('>forcetie'); } })();
  await Promise.all(play);
  await stream.writeEnd();
}

async function main() {
  const teams = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const n = Number(process.argv[3] || 20), seed = Number(process.argv[4] || 1);
  const spreads = process.env.FOUL_PLAY_SPREADS ? require(process.env.FOUL_PLAY_SPREADS) : {};
  const out = {battles: n, decisions: 0, unbuilt: 0, checks: {}, mismatches: {}, examples: [], errors: [], rejected: []};
  process.on('beforeExit', () => { if (!out.done) process.stderr.write(`stalled in battle ${out.current}: ${JSON.stringify(out.stall || {})}\n`); });
  for (let i = 0; i < n; i++) {
    out.current = i;
    await battle(teams[i % teams.length], teams[(i + 1 + Math.floor(i / teams.length)) % teams.length], seed + i, spreads, out);
  }
  out.done = true;
  process.stdout.write(JSON.stringify(out) + '\n');
}
main().catch(e => { console.error(e); process.exitCode = 1; });
