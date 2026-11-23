// Differential audit: compare Rust poke-engine-doubles single-turn damage & faint outcomes
// against Pokémon Showdown on real Regulation M-B top-50 doubles turns.
'use strict';
const path = require('path');
const {spawnSync} = require('child_process');
const {Battle, Teams} = require(path.join(process.env.HOME, 'vgc-data/battle-evaluations/runtime/dist/sim'));
const {slotOptions, compatible, mulberry, FORMAT} = require('../src/foul_play_doubles.cjs');
const {RustEngineClient, serializeBattle} = require('../src/rust_engine_client.cjs');

function pickRandomChoice(battle, sideIndex, rng) {
  const slots = slotOptions(battle, sideIndex);
  if (!slots) return null;
  const a = slots[0][Math.floor(rng() * slots[0].length)];
  if (slots.length === 1) return a.choice;
  const ok = slots[1].filter(b => compatible(a, b));
  const b = ok.length ? ok[Math.floor(rng() * ok.length)] : slots[1][0];
  return `${a.choice}, ${b.choice}`;
}

async function main() {
  const teamsJson = spawnSync('/tmp/vgc-pilot/.venv/bin/python', ['-c',
    'import sqlite3,json,os;c=sqlite3.connect(os.path.expanduser("~/vgc-data/matchup_regmb.sqlite"));' +
    'print(json.dumps([r[0] for r in c.execute("SELECT t.paste FROM team t JOIN team_set s ON s.team_id=t.team_id WHERE s.set_name=\'top50\' ORDER BY t.team_id LIMIT 20")]))'
  ], {encoding: 'utf8'}).stdout;
  const teams = JSON.parse(teamsJson);
  const rng = mulberry(20261123);
  const client = new RustEngineClient();
  let turns = 0, hpChecks = 0, hpAbsErrSum = 0, faintChecks = 0, faintMatch = 0, koExact = 0;

  for (let g = 0; g < 20; g++) {
    const b = new Battle({formatid: FORMAT, seed: [1 + g, 2 + g, 3 + g, 4 + g]});
    b.setPlayer('p1', {name: 'me', team: Teams.pack(Teams.import(teams[g % teams.length]))});
    b.setPlayer('p2', {name: 'foe', team: Teams.pack(Teams.import(teams[(g + 7) % teams.length]))});
    b.choose('p1', 'team 1234');
    b.choose('p2', 'team 1234');
    for (let t = 0; t < 8 && !b.ended; t++) {
      if (b.sides[0].activeRequest.forceSwitch || b.sides[1].activeRequest.forceSwitch) {
        for (let s = 0; s < 2; s++) {
          const ch = pickRandomChoice(b, s, rng);
          if (ch) b.choose(s === 0 ? 'p1' : 'p2', ch);
        }
        continue;
      }
      const wire = serializeBattle(b);
      const c1 = pickRandomChoice(b, 0, rng);
      const c2 = pickRandomChoice(b, 1, rng);
      if (!c1 || !c2) break;
      // Average 5 Rust random rolls to compare expected post-turn HP against Showdown
      const rustRuns = [];
      for (let r = 0; r < 5; r++) {
        rustRuns.push(await client.request({cmd: 'step', state: wire, p1: c1, p2: c2, seed: 100 + g * 100 + t * 10 + r}));
      }
      b.choose('p1', c1);
      b.choose('p2', c2);
      turns++;
      for (let s = 0; s < 2; s++) {
        const key = s === 0 ? 'p1_hp' : 'p2_hp';
        b.sides[s].pokemon.forEach((p, i) => {
          const sdFrac = p.hp / p.maxhp;
          const rustMean = rustRuns.reduce((acc, rr) => acc + rr[key][i], 0) / rustRuns.length / p.maxhp;
          hpChecks++;
          hpAbsErrSum += Math.abs(sdFrac - rustMean);
          const sdFaint = p.hp <= 0;
          const rustFaint = rustRuns[0][key][i] <= 0;
          faintChecks++;
          if (sdFaint === rustFaint) faintMatch++;
          if (sdFaint && rustFaint) koExact++;
        });
      }
    }
  }
  client.close();
  console.log(JSON.stringify({
    turns,
    hp_checks: hpChecks,
    mean_abs_hp_frac_error: +(hpAbsErrSum / hpChecks).toFixed(4),
    faint_agreement: +(faintMatch / faintChecks).toFixed(4),
    ko_matches: koExact,
  }, null, 2));
}

main().catch(e => { console.error(e); process.exit(1); });
