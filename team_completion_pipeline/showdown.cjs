// JSON-in / JSON-out bridge to the existing pinned Pokémon Showdown runtime.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function fingerprint(root) {
  const hash = crypto.createHash('sha256');
  function visit(folder) {
    for (const entry of fs.readdirSync(folder, {withFileTypes: true}).sort((a,b) => a.name.localeCompare(b.name))) {
      const file = path.join(folder, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile()) {hash.update(path.relative(root, file)); hash.update(fs.readFileSync(file));}
    }
  }
  visit(path.join(root, 'dist'));
  for (const name of ['package.json', 'package-lock.json']) {
    const file = path.join(root, name);
    if (fs.existsSync(file)) hash.update(fs.readFileSync(file));
  }
  hash.update(fs.readFileSync(__filename));
  hash.update(process.version);
  return hash.digest('hex');
}

function packTeam(team, Teams) {
  return Teams.pack(team.map((m) => {
    const stats = {HP:'hp', Atk:'atk', Def:'def', SpA:'spa', SpD:'spd', Spe:'spe'};
    const convert = value => Object.fromEntries(Object.entries(value).map(([k,v]) => [stats[k] || k, v]));
    return {name:m.species, species:m.species, item:m.item, ability:m.ability,
      moves:m.moves, nature:m.nature, evs:convert(m.statPoints),
      ...(m.ivs ? {ivs:convert(m.ivs)} : {}),
      level:m.level ?? 50, ...(m.gender ? {gender:m.gender} : {}),
      ...(m.shiny !== undefined ? {shiny:m.shiny} : {}),
      ...(m.happiness !== undefined ? {happiness:m.happiness} : {})};
  }));
}

async function main(request) {
  const root = path.resolve(request.runtime);
  if (request.operation === 'fingerprint') return {fingerprint:fingerprint(root)};
  const {Teams, BattleStream, getPlayerStreams} = require(path.join(root, 'dist/sim'));
  const {TeamValidator} = require(path.join(root, 'dist/sim/team-validator'));
  const validator = new TeamValidator(request.format);
  if (request.operation === 'import') {
    const source = Teams.import(request.paste);
    if (!source) throw new Error('Could not parse paste');
    if (source.some(m => m.teraType)) throw new Error('Tera Types are not supported by this pipeline regulation');
    const keys = {hp:'HP', atk:'Atk', def:'Def', spa:'SpA', spd:'SpD', spe:'Spe'};
    const convert = values => Object.fromEntries(Object.entries(values || {}).map(([k,v]) => [keys[k] || k,v]));
    return {team:source.map((m,i) => ({slot:i+1, species:m.species, item:m.item || '',
      ability:m.ability, moves:m.moves, nature:m.nature, statPoints:convert(m.evs),
      ivs:m.ivs ? convert(m.ivs) : null,
      level:m.level || 50, ...(m.gender ? {gender:m.gender} : {}),
      ...(m.shiny ? {shiny:true} : {}),
      ...(m.happiness !== undefined ? {happiness:m.happiness} : {})}))};
  }
  const team = packTeam(request.team, Teams);
  const errors = validator.validateTeam(Teams.unpack(team)) || [];
  if (request.operation === 'validate') return {errors};
  if (errors.length) throw new Error(errors.join('; '));
  if (request.operation !== 'battle') throw new Error('Unknown operation');
  const opponent = packTeam(request.opponent, Teams);
  const opponentErrors = validator.validateTeam(Teams.unpack(opponent)) || [];
  if (opponentErrors.length) throw new Error(opponentErrors.join('; '));
  const {RandomPlayerAI} = require(path.join(root, 'dist/sim/tools/random-player-ai'));
  const seedFor = label => {
    const bytes = crypto.createHash('sha256').update(`${request.seed}:${label}`).digest();
    return [0,2,4,6].map(i => bytes.readUInt16BE(i));
  };
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const p1 = new RandomPlayerAI(streams.p1, {seed:seedFor('p1'), mega:1});
  const p2 = new RandomPlayerAI(streams.p2, {seed:seedFor('p2'), mega:1});
  // Surface policy errors immediately; the Python parent also enforces a wall-time timeout.
  p1.start().catch(error => {console.error(error.message); process.exit(1);});
  p2.start().catch(error => {console.error(error.message); process.exit(1);});
  const logs = [];
  let outcome = null;
  const reading = (async () => {
    for await (const chunk of streams.omniscient) {
      logs.push(chunk);
      for (const line of chunk.split('\n')) {
        if (line === '|win|Candidate') outcome = 'win';
        if (line === '|win|Opponent') outcome = 'loss';
        if (line === '|tie|') outcome = 'draw';
        if (line.startsWith('|turn|') && Number(line.split('|')[2]) > request.max_turns) {
          throw new Error('Turn limit exceeded; this is a simulation error, not a draw');
        }
      }
    }
  })();
  await streams.omniscient.write(`>start ${JSON.stringify({formatid:request.format, seed:seedFor('battle')})}\n` +
    `>player p1 ${JSON.stringify({name:'Candidate', team})}\n` +
    `>player p2 ${JSON.stringify({name:'Opponent', team:opponent})}`);
  await reading;
  if (!outcome) throw new Error('Battle ended without an outcome');
  return {outcome, log:logs.join('\n'), policy:'showdown-random-v1'};
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => main(JSON.parse(input)).then(result => {
  process.stdout.write(JSON.stringify(result));
}).catch(error => {console.error(error.message); process.exit(1);}));
