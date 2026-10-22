// Random-play battles on the pinned Showdown runtime, read straight from the Battle object.
//
// "Random play": Showdown's RandomPlayerAI on both sides (a random move every turn, Mega Evolution
// whenever possible, switches only when forced), plus a random team preview (random four of six,
// random leads); the stock AI always sends "default", which brings the first four in paste order.
//
// One JSON request on stdin, one JSON line per battle (or per spine) on stdout:
//   {"mode": "walk", "teams": {id: paste}, "pairs": [[row, col, n], ...], "seed": s}
//     n independent random battles per pair; per-turn trajectory of Pokemon left and HP.
//   {"mode": "tree", "teams": {...}, "pairs": [[row, col, spines], ...], "k": K, "seed": s}
//     spine battles; at the root and at every turn boundary of a spine, K random rollouts from a
//     saved copy of that state estimate the random-play win probability there (a comb of the tree).
// Side p1 is always `row`. Win is from p1's side: 1, 0, or 0.5 for a tie.
'use strict';
const path = require('path');
const crypto = require('crypto');
const RT = process.env.SHOWDOWN_RUNTIME || path.join(process.env.HOME, 'vgc-data/battle-evaluations/runtime');
const {Battle, Teams, PRNG} = require(path.join(RT, 'dist/sim'));
const {RandomPlayerAI} = require(path.join(RT, 'dist/sim/tools/random-player-ai'));
const FORMAT = 'gen9championsvgc2026regmb';
const MAX_TURNS = 100;
const stats = {fallback: 0};

function seedFor(...parts) {
  const b = crypto.createHash('sha256').update(parts.join(':')).digest();
  return [0, 2, 4, 6].map(i => b.readUInt16BE(i));
}

function chooser(seed) {
  const ai = new RandomPlayerAI({write() {}}, {seed, mega: 1});
  ai.picked = null;
  ai.choose = function (c) { this.picked = c; };
  ai.chooseTeamPreview = function (team) {
    const idx = team.map((_, i) => i + 1);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = this.prng.random(i + 1);
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return 'team ' + idx.join(',');
  };
  return ai;
}

// Pokemon left and summed HP fraction per side; after team preview each side holds its four.
function board(battle) {
  const out = [];
  for (const side of battle.sides) {
    let alive = 0, hp = 0;
    for (const p of side.pokemon) {
      if (!p.fainted && p.hp > 0) alive++;
      hp += p.hp / p.maxhp;
    }
    out.push(alive, Math.round(hp * 1e4) / 1e4);
  }
  return out;
}

// Pokemon that fainted since the last call; side.pokemon reorders on every switch, so track objects.
function newFaints(battle, seen) {
  const out = [];
  for (const side of battle.sides) {
    for (const p of side.pokemon) {
      if ((p.fainted || p.hp <= 0) && !seen.has(p)) { seen.add(p); out.push([side.n, p.set.species]); }
    }
  }
  return out;
}

function winOf(battle) {
  if (!battle.winner) return 0.5;
  return battle.winner === battle.sides[0].name ? 1 : 0;
}

// Play until the battle ends. onTurn(battle) runs once at the start of every turn (move request);
// it may return false to stop early.
function advance(battle, ais, onTurn) {
  let lastTurn = -1, guard = 0;
  while (!battle.ended) {
    if (++guard > 5000) throw new Error('decision guard');
    if (battle.requestState === 'move' && battle.turn !== lastTurn) {
      lastTurn = battle.turn;
      if (battle.turn > MAX_TURNS) throw new Error('turn limit');
      if (onTurn && onTurn(battle) === false) return false;
    }
    let chose = false;
    for (const side of battle.sides) {
      const req = side.activeRequest;
      if (!req || req.wait || side.isChoiceDone()) continue;
      const ai = ais[side.n];
      ai.picked = null;
      ai.receiveRequest(req);
      if (ai.picked == null || !battle.choose(side.id, ai.picked)) {
        stats.fallback++;
        if (!battle.choose(side.id, 'default')) throw new Error(`no valid choice (${ai.picked})`);
      }
      chose = true;
      if (battle.ended) break;
    }
    if (!chose && !battle.ended) throw new Error(`stuck in ${battle.requestState}`);
  }
  return true;
}

function newBattle(teams, row, col, seed) {
  const battle = new Battle({formatid: FORMAT, seed});
  battle.setPlayer('p1', {name: 'Row', team: teams[row]});
  battle.setPlayer('p2', {name: 'Col', team: teams[col]});
  return battle;
}

function walkOne(teams, row, col, seed) {
  const battle = newBattle(teams, row, col, seedFor(seed, 'battle'));
  const ais = [chooser(seedFor(seed, 'p1')), chooser(seedFor(seed, 'p2'))];
  const traj = [], faints = [], seen = new Set();
  let preview = null;
  const note = (b, turn) => { for (const [s, sp] of newFaints(b, seen)) faints.push([turn, s, sp]); };
  advance(battle, ais, b => {
    if (!preview) preview = b.sides.map(side => side.pokemon.map(p => p.set.species));
    note(b, b.turn - 1);
    traj.push([b.turn, ...board(b)]);
  });
  note(battle, battle.turn);
  traj.push([battle.turn + 1, ...board(battle)]);
  return {row, col, seed, win: winOf(battle), turns: battle.turn, traj, faints, preview};
}

function rollouts(state, k, tag) {
  let wins = 0, turns = 0;
  for (let i = 0; i < k; i++) {
    const b = Battle.fromJSON(state);
    b.prng = new PRNG(seedFor(tag, i, 'battle'));
    advance(b, [chooser(seedFor(tag, i, 'p1')), chooser(seedFor(tag, i, 'p2'))], null);
    wins += winOf(b);
    turns += b.turn;
  }
  return [wins / k, turns / k];
}

function spine(teams, row, col, seed, k) {
  const battle = newBattle(teams, row, col, seedFor(seed, 'battle'));
  const ais = [chooser(seedFor(seed, 'p1')), chooser(seedFor(seed, 'p2'))];
  const root = rollouts(JSON.stringify(battle.toJSON()), k, `${seed}:root`);
  const nodes = [];
  advance(battle, ais, b => {
    // the spine's log is not needed; keep the saved state small
    b.log = []; b.sentLogPos = 0; b.lastMoveLine = -1;
    const [v, t] = rollouts(JSON.stringify(b.toJSON()), k, `${seed}:t${b.turn}`);
    nodes.push([b.turn, ...board(b), v, t]);
  });
  return {row, col, seed, k, root, nodes, win: winOf(battle), turns: battle.turn, end: board(battle)};
}

// The four brought Pokemon per side, for drawing a node: current forme, HP fraction, on the field.
function detail(battle) {
  return battle.sides.map(side => side.pokemon.map(p => ({
    species: p.species.name, hp: Math.round(p.hp / p.maxhp * 1000) / 1000, active: !!p.isActive,
  })));
}

const quiet = b => { b.log = []; b.sentLogPos = 0; b.lastMoveLine = -1; return b; };

// From a saved state, play random turns until the next turn boundary (or the end).
function nextTurn(state, tag) {
  const b = Battle.fromJSON(state);
  b.prng = new PRNG(seedFor(tag, 'battle'));
  const atMove = b.requestState === 'move', t0 = b.turn;
  advance(b, [chooser(seedFor(tag, 'p1')), chooser(seedFor(tag, 'p2'))], bb => atMove && bb.turn === t0);
  return quiet(b);
}

function node(b, k, tag) {
  const s = JSON.stringify(b.toJSON());
  const out = {turn: b.turn, board: board(b), detail: detail(b), ended: !!b.ended};
  out.v = b.ended ? winOf(b) : rollouts(s, k, `${tag}:v`)[0];
  return [out, s];
}

// A real tree: `branch` random continuations per node for `depth` turns; each leaf then plays one
// random path to the end, with the win probability (k_path rollouts) at every turn on the way.
function grow(state, tag, depth, req) {
  const kids = [];
  for (let i = 0; i < req.branch; i++) {
    const b = nextTurn(state, `${tag}.${i}`);
    const [n, s] = node(b, req.k, `${tag}.${i}`);
    if (!b.ended && depth > 1) n.children = grow(s, `${tag}.${i}`, depth - 1, req);
    else if (!b.ended) n.path = playOut(s, `${tag}.${i}:path`, req.k_path);
    kids.push(n);
  }
  return kids;
}

function playOut(state, tag, k) {
  const b = Battle.fromJSON(state);
  b.prng = new PRNG(seedFor(tag, 'battle'));
  const atMove = b.requestState === 'move', t0 = b.turn, pts = [];
  advance(b, [chooser(seedFor(tag, 'p1')), chooser(seedFor(tag, 'p2'))], bb => {
    if (atMove && bb.turn === t0) return true;
    quiet(bb);
    pts.push({turn: bb.turn, board: board(bb), v: rollouts(JSON.stringify(bb.toJSON()), k, `${tag}:t${bb.turn}`)[0]});
  });
  return {pts, win: winOf(b), turns: b.turn, end: board(b), detail: detail(b)};
}

function branchTree(teams, row, col, seed, req) {
  const root = quiet(newBattle(teams, row, col, seedFor(seed, 'battle')));
  const s = JSON.stringify(root.toJSON());
  const [v] = rollouts(s, req.k, `${seed}:root`);
  return {row, col, seed, k: req.k, k_path: req.k_path, root: {turn: 0, v, children: grow(s, seed, req.depth, req)}};
}

function main(req) {
  const teams = {};
  for (const [id, paste] of Object.entries(req.teams)) teams[id] = Teams.pack(Teams.import(paste));
  for (const [row, col, n] of req.pairs) {
    for (let r = 0; r < n; r++) {
      const seed = `${req.seed}:${row}:${col}:${r}`;
      let out;
      try {
        out = req.mode === 'tree' ? spine(teams, row, col, seed, req.k)
          : req.mode === 'branch' ? branchTree(teams, row, col, seed, req)
          : walkOne(teams, row, col, seed);
      } catch (e) {
        out = {row, col, seed, error: String(e && e.message || e)};
      }
      process.stdout.write(JSON.stringify(out) + '\n');
    }
  }
  process.stderr.write(JSON.stringify({done: true, fallback: stats.fallback}) + '\n');
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', c => input += c);
process.stdin.on('end', () => main(JSON.parse(input)));
