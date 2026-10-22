// A persistent direct Showdown experiment. Random policy is engine-only evidence,
// never a replacement for the BC policy in competitive team evaluation.
'use strict';
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');
const root = process.argv[2];
const {BattleStream, getPlayerStreams, Teams} = require(path.join(root, 'dist/sim'));
const {TeamValidator} = require(path.join(root, 'dist/sim/team-validator'));
const {RandomPlayerAI} = require(path.join(root, 'dist/sim/tools/random-player-ai'));
const prepared = new Map();

function seedFor(seed, label) {
  const bytes = crypto.createHash('sha256').update(`${seed}:${label}`).digest();
  return [0, 2, 4, 6].map(i => bytes.readUInt16BE(i));
}
function team(paste, format, cache) {
  const key = format + '\0' + paste;
  if (cache && prepared.has(key)) return prepared.get(key);
  const parsed = Teams.import(paste);
  const errors = new TeamValidator(format).validateTeam(parsed);
  if (errors) throw new Error(errors.join('; '));
  const packed = Teams.pack(parsed);
  if (cache) prepared.set(key, packed);
  return packed;
}
async function battle(job, options) {
  const start = performance.now();
  const a = team(job.a, options.format, options.cacheTeams);
  const b = team(job.b, options.format, options.cacheTeams);
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const players = [new RandomPlayerAI(streams.p1, {seed: seedFor(job.seed, 'p1'), mega: 1}),
    new RandomPlayerAI(streams.p2, {seed: seedFor(job.seed, 'p2'), mega: 1})];
  const playerTasks = players.map(p => p.start());
  // Promise handlers are installed immediately; Promise.all below also observes errors.
  for (const p of playerTasks) p.catch(() => {});
  let outcome = null, turns = 0;
  const hash = crypto.createHash('sha256');
  const logs = [];
  const reader = (async () => {
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('|t:|')) hash.update(line+'\n');
      }
      if (options.returnLogs) logs.push(chunk);
      for (const line of chunk.split('\n')) {
        if (line.startsWith('|turn|')) turns = Number(line.split('|')[2]);
        if (line === '|win|Candidate') outcome = 'win';
        if (line === '|win|Opponent') outcome = 'loss';
        if (line === '|tie|') outcome = 'draw';
        if (turns > 1000) throw new Error('Exceeded 1000 turns; not counted as draw');
      }
    }
  })();
  reader.catch(() => {});
  try {
    await streams.omniscient.write(`>start ${JSON.stringify({formatid: options.format, seed: seedFor(job.seed, 'battle')})}\n`+
      `>player p1 ${JSON.stringify({name: 'Candidate', team: a})}\n`+
      `>player p2 ${JSON.stringify({name: 'Opponent', team: b})}`);
    await Promise.all([reader, ...playerTasks]);
    if (!outcome) throw new Error('No terminal outcome');
    return {id: job.id, outcome, turns, trace: hash.digest('hex'),
      milliseconds: performance.now()-start, ...(options.returnLogs ? {log: logs.join('\n')} : {})};
  } finally {
    await stream.writeEnd();
  }
}
async function run(request) {
  const before = process.cpuUsage(), start = performance.now();
  let index = 0;
  const results = new Array(request.jobs.length);
  async function worker() {
    while (index < request.jobs.length) {
      const i = index++;
      results[i] = await battle(request.jobs[i], request);
    }
  }
  await Promise.all(Array.from({length: Math.min(request.concurrency || 1, request.jobs.length)}, worker));
  const cpu = process.cpuUsage(before);
  return {seconds: (performance.now()-start)/1000, completed: results.length,
    turns: results.reduce((s, r) => s+r.turns, 0), cpuSeconds: (cpu.user+cpu.system)/1e6,
    rssBytes: process.memoryUsage().rss, results, policy: 'showdown-random-v1'};
}
process.stdout.write(JSON.stringify({ready: true, node: process.version})+'\n');
const lines = readline.createInterface({input: process.stdin});
(async () => {
  for await (const line of lines) {
    try { process.stdout.write(JSON.stringify(await run(JSON.parse(line)))+'\n'); }
    catch (error) { process.stderr.write(error.stack+'\n'); process.stdout.write(JSON.stringify({error: error.message})+'\n'); }
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
