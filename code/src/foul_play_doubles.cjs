// Foul Play for doubles: pmariglia/foul-play's decision loop rebuilt on Pokémon Showdown's own
// simulator, so it plays VGC (Reg M-B, open team sheets, Mega Evolution) instead of singles only.
// See code/docs/foul-play-doubles.md. Source read: foul-play 8f541f8 (fp/search/main.py,
// standard_battles.py, bss.py) and poke-engine f7b147f (src/mcts.rs, src/genx/evaluate.rs).
//
// One decision, as in Foul Play's find_best_move:
//   1. track what this side has seen (its request and its player-view protocol, never the
//      omniscient log), including speed bounds from move order (fp/battle/inference.py's idea);
//   2. sample N full battles consistent with it: the opponent's hidden Stat Points from a team
//      dataset filtered by those bounds, and which unrevealed Pokémon they brought weighted by the
//      team-preview search's affinities (bss.py's sample_pkmn_to_remove);
//   3. run decoupled-UCT MCTS on each sample (poke-engine's mcts.rs: UCB1 per side, one new node
//      per iteration, leaf scored by sigmoid(0.0125 * (evaluate - root evaluate)), no rollouts);
//   4. add the root visit shares across samples, keep options within 75% of the best, and draw
//      one in proportion (select_move_from_mcts_results).
// Doubles changes: each active slot keeps its own UCB statistics (the joint action of two slots
// has ~100 options per side, too many for one UCB table at JavaScript speed); a slot's choice
// excludes what its partner already took (the same switch-in, a second Mega); the tree is
// open-loop, so a child is keyed by the joint choice plus which Pokémon are in and who must
// switch, and each iteration replays the simulator from the root with fresh randomness.
'use strict';
const path = require('path');
const RT = process.env.SHOWDOWN_RUNTIME || path.join(process.env.HOME, 'vgc-data/battle-evaluations/runtime');
const {Battle, Teams, PRNG, Dex} = require(path.join(RT, 'dist/sim'));
const FORMAT = 'gen9championsvgc2026regmb';
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
// The name Showdown shows for a set: its nickname, else its base species (Floette-Eternal is 'Floette').
// Open team sheets carry the species in the name field; a real nickname differs from it.
const setName = s => (!s.name || s.name === s.species ? Dex.species.get(s.species).baseSpecies : s.name);
const matchSet = (sets, name) => sets.find(s => setName(s) === name);

// ---------------------------------------------------------------- random numbers
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const seed4 = rng => [0, 1, 2, 3].map(() => Math.floor(rng() * 65536));
function weightedPick(items, weights, rng) {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = rng() * total;
  for (let i = 0; i < items.length; i++) { r -= weights[i]; if (r <= 0) return items[i]; }
  return items[items.length - 1];
}

// ---------------------------------------------------------------- evaluation (poke-engine genx/evaluate.rs)
const W = {alive: 30, hp: 100, atk: 30, def: 15, spa: 30, spd: 15, spe: 30, frz: -40, slp: -25, par: -25,
  tox: -30, psn: -10, brn: -25, leechseed: -30, substitute: 75, confusion: -20, reflect: 20, lightscreen: 20,
  auroraveil: 40, safeguard: 5, tailwind: 7, healingwish: 30, stealthrock: -10, spikes: -7,
  toxicspikes: -7, stickyweb: -25};
const BOOST = {6: 3.3, 5: 3.15, 4: 3, 3: 2.5, 2: 2, 1: 1, 0: 0, '-1': -1, '-2': -2, '-3': -2.5, '-4': -3,
  '-5': -3.15, '-6': -3.3};

function evalPokemon(p) {
  let s = W.hp * p.hp / p.maxhp;
  const ability = toID(p.ability);
  if (p.status === 'brn') {
    if (['guts', 'marvelscale', 'quickfeet'].includes(ability)) s -= 2 * W.brn;
    else {
      let m = p.moveSlots.filter(m => p.battle.dex.moves.get(m.id).category === 'Physical').length;
      if (p.storedStats.spa > p.storedStats.atk) m /= 2;
      s += m * W.brn;
    }
  } else if (p.status === 'psn' || p.status === 'tox') {
    if (ability === 'poisonheal') s += 15;
    else if (['guts', 'marvelscale', 'quickfeet', 'toxicboost', 'magicguard'].includes(ability)) s += 10;
    else s += W[p.status];
  } else if (p.status && W[p.status]) s += W[p.status];
  if (p.item) s += 10;
  return Math.max(s, 0) + W.alive;
}

function evalSide(side) {
  let s = 0;
  for (const p of side.pokemon) {
    if (p.hp <= 0 || p.fainted) continue;
    s += evalPokemon(p);
    if (!p.isGrounded || toID(p.item) === 'heavydutyboots' || toID(p.ability) === 'magicguard') {
      // no hazard terms
    } else {
      for (const h of ['spikes', 'toxicspikes', 'stickyweb']) {
        if (side.sideConditions[h]) s += W[h] * (side.sideConditions[h].layers || 1);
      }
    }
    if (side.sideConditions.stealthrock && toID(p.item) !== 'heavydutyboots') s += W.stealthrock;
    if (p.isActive) {
      for (const v of ['leechseed', 'substitute', 'confusion']) if (p.volatiles[v]) s += W[v];
      for (const stat of ['atk', 'def', 'spa', 'spd', 'spe']) s += BOOST[p.boosts[stat]] * W[stat];
    }
  }
  for (const c of ['reflect', 'lightscreen', 'auroraveil', 'safeguard', 'tailwind', 'healingwish']) {
    const cond = side.sideConditions[c];
    if (cond) s += W[c] * (cond.duration || 1);
  }
  return s;
}
// Score from p1's side; in every searched battle p1 is the side doing the search.
const evaluate = battle => evalSide(battle.sides[0]) - evalSide(battle.sides[1]);
const sigmoid = x => 1 / (1 + Math.exp(-0.0125 * x));

// VGC 4v4 doubles static evaluator: scales KO value for 4-Pokémon teams, scores Trick Room and
// Tailwind against actual active Speed tiers, rewards Fake Out pressure and doubles volatiles,
// and down-weights singles-centric entry hazards.
const WD = {alive: 65, hp: 100, atk: 28, def: 14, spa: 28, spd: 14, spe: 28, frz: -45, slp: -35, par: -28,
  tox: -25, psn: -10, brn: -28, leechseed: -25, substitute: 65, confusion: -18, encore: -35, taunt: -18,
  yawn: -22, perishsong: -25, disable: -15, reflect: 18, lightscreen: 18, auroraveil: 36, safeguard: 6,
  tailwind: 24, healingwish: 25, stealthrock: -3, spikes: -2, toxicspikes: -2, stickyweb: -12};
const WEATHER_SPEED_AB = {swiftswim: 'raindance', chlorophyll: 'sunnyday', sandrush: 'sandstorm', slushrush: 'snowscape'};
function effectiveActiveSpeed(battle, side, p) {
  let spe = (p.storedStats && p.storedStats.spe) || 100;
  const b = (p.boosts && p.boosts.spe) || 0;
  spe *= b >= 0 ? (2 + b) / 2 : 2 / (2 - b);
  if (side.sideConditions && side.sideConditions.tailwind) spe *= 2;
  if (p.status === 'par') spe *= 0.5;
  const item = toID(p.item);
  if (item === 'choicescarf') spe *= 1.5;
  if (item === 'ironball') spe *= 0.5;
  const ab = toID(p.ability);
  const w = battle.field && battle.field.weather;
  if (WEATHER_SPEED_AB[ab] && w && (w === WEATHER_SPEED_AB[ab] || (ab === 'slushrush' && w === 'snow'))) spe *= 2;
  if (ab === 'surgesurfer' && battle.field && battle.field.terrain === 'electricterrain') spe *= 2;
  return spe;
}
function evalSideDoubles(battle, side, foe) {
  let s = 0;
  const psychicTerrain = battle.field && battle.field.terrain === 'psychicterrain';
  for (const p of side.pokemon) {
    if (p.hp <= 0 || p.fainted) continue;
    let ps = WD.hp * p.hp / p.maxhp;
    const ability = toID(p.ability);
    if (p.status === 'brn') {
      if (['guts', 'marvelscale', 'quickfeet'].includes(ability)) ps -= 2 * WD.brn;
      else {
        let m = p.moveSlots.filter(mv => p.battle.dex.moves.get(mv.id).category === 'Physical').length;
        if (p.storedStats.spa > p.storedStats.atk) m /= 2;
        ps += m * WD.brn;
      }
    } else if (p.status === 'psn' || p.status === 'tox') {
      if (ability === 'poisonheal') ps += 15;
      else if (['guts', 'marvelscale', 'quickfeet', 'toxicboost', 'magicguard'].includes(ability)) ps += 10;
      else ps += WD[p.status];
    } else if (p.status && WD[p.status]) ps += WD[p.status];
    if (p.item) ps += 8;
    s += Math.max(ps, 0) + WD.alive;
    if (p.isGrounded && toID(p.item) !== 'heavydutyboots' && ability !== 'magicguard') {
      for (const h of ['spikes', 'toxicspikes', 'stickyweb']) {
        if (side.sideConditions[h]) s += WD[h] * (side.sideConditions[h].layers || 1);
      }
    }
    if (side.sideConditions.stealthrock && toID(p.item) !== 'heavydutyboots') s += WD.stealthrock;
    if (p.isActive) {
      for (const v of ['leechseed', 'substitute', 'confusion', 'encore', 'taunt', 'yawn', 'perishsong', 'disable']) {
        if (p.volatiles[v]) s += WD[v];
      }
      for (const stat of ['atk', 'def', 'spa', 'spd', 'spe']) s += BOOST[p.boosts[stat]] * WD[stat];
      if (!psychicTerrain && p.activeMoveActions === 0 && p.moveSlots.some(mv => mv.id === 'fakeout')) s += 16;
    }
  }
  for (const c of ['reflect', 'lightscreen', 'auroraveil', 'safeguard', 'tailwind', 'healingwish']) {
    const cond = side.sideConditions[c];
    if (cond) s += WD[c] * (cond.duration || 1);
  }
  // Speed-tier advantage between active Pokémon (flips sign under Trick Room).
  const tr = !!(battle.field && battle.field.pseudoWeather && battle.field.pseudoWeather.trickroom);
  const myActives = side.active.filter(p => p && !p.fainted && p.hp > 0);
  const foeActives = foe.active.filter(p => p && !p.fainted && p.hp > 0);
  for (const a of myActives) {
    const sa = effectiveActiveSpeed(battle, side, a);
    for (const f of foeActives) {
      const sf = effectiveActiveSpeed(battle, foe, f);
      if (sa === sf) continue;
      const faster = (sa > sf) !== tr;
      s += faster ? 10 : -10;
    }
  }
  return s;
}
const evaluateDoubles = battle => evalSideDoubles(battle, battle.sides[0], battle.sides[1]) -
  evalSideDoubles(battle, battle.sides[1], battle.sides[0]);
const sigmoidDoubles = x => 1 / (1 + Math.exp(-0.009 * x));

// ---------------------------------------------------------------- choices from a request
const ALLY_TARGET_MOVES = new Set(['pollenpuff', 'healpulse', 'floralhealing', 'skillswap', 'decorate',
  'afteryou', 'instruct', 'beatup', 'acupressure', 'coaching', 'quash']);
const PREVIEW = (() => {
  const out = [];
  for (let a = 1; a <= 6; a++) for (let b = a + 1; b <= 6; b++) {
    const rest = [1, 2, 3, 4, 5, 6].filter(x => x !== a && x !== b);
    for (let c = 0; c < 4; c++) for (let d = c + 1; d < 4; d++) {
      out.push({choice: `team ${a}${b}${rest[c]}${rest[d]}`, leads: [a, b], brought: [a, b, rest[c], rest[d]]});
    }
  }
  return out;
})();

// Each slot's options for one request: {choice, switchTo, mega}. Team preview is one slot of 90.
function slotOptions(battle, sideIndex, real = null) {
  const side = battle.sides[sideIndex];
  const req = real || side.activeRequest;
  if (!req || req.wait) return null;
  if (req.teamPreview) return [PREVIEW.map(p => ({choice: p.choice, preview: p}))];
  const bench = [];
  side.pokemon.forEach((p, i) => {
    if (i >= side.active.length && !p.fainted && p.hp > 0) bench.push(i + 1);
  });
  if (req.forceSwitch) {
    // With fewer Pokémon left than slots to fill, Showdown wants a pass for the extra slot.
    const short = bench.length < req.forceSwitch.filter(Boolean).length;
    return req.forceSwitch.map(must => {
      if (!must) return [{choice: 'pass'}];
      const opts = bench.map(n => ({choice: `switch ${n}`, switchTo: n}));
      if (!opts.length) opts.push({choice: 'pass'});
      else if (short) opts.push({choice: 'pass', optionalPass: true});
      return opts;
    });
  }
  return req.active.map((data, slot) => {
    const me = side.active[slot];
    if (!me || me.fainted || me.hp <= 0 || me.volatiles.commanding) return [{choice: 'pass'}];
    const opts = [];
    const foes = side.foe.active.map((p, i) => (p && !p.fainted && p.hp > 0 ? i + 1 : 0)).filter(Boolean);
    const ally = side.active[1 - slot];
    const allyAlive = ally && !ally.fainted && ally.hp > 0;
    const allyLoc = -(2 - slot);   // slot 0's ally is position -2, slot 1's is -1
    const selfLoc = -(slot + 1);
    data.moves.forEach((m0, i) => {
      if (m0.disabled) return;
      const m = m0.target ? m0 : {...m0, target: Dex.moves.get(m0.id).target};   // some requests omit it
      let targets = [0];
      if (['normal', 'any', 'adjacentFoe'].includes(m.target)) {
        targets = foes.length ? [...foes] : [1];
        if (allyAlive && ALLY_TARGET_MOVES.has(m.id)) targets.push(allyLoc);
      } else if (m.target === 'adjacentAlly') {
        if (!allyAlive) return;
        targets = [allyLoc];
      } else if (m.target === 'adjacentAllyOrSelf') {
        targets = allyAlive ? [selfLoc, allyLoc] : [selfLoc];
      }
      for (const t of targets) {
        const base = `move ${i + 1}${t ? ' ' + t : ''}`;
        opts.push({choice: base});
        if (data.canMegaEvo) opts.push({choice: base + ' mega', mega: true});
      }
    });
    if (!data.trapped) for (const n of bench) opts.push({choice: `switch ${n}`, switchTo: n});
    return opts.length ? opts : [{choice: 'move 1'}];
  });
}
const compatible = (a, b) => !(a.switchTo && a.switchTo === b.switchTo) && !(a.mega && b.mega) &&
  !(a.optionalPass && b.optionalPass);

// ---------------------------------------------------------------- what one side has seen
// 'cur/max status'. Champions shows a foe's HP as floor(100 * hp / maxhp) out of 100, with a colour
// letter at 20 and 50 telling which side of the line it is; the true fraction is in [p, p + 1) / 100.
function parseHP(text) {
  const [hp, status] = String(text).split(' ');
  if (hp === '0' || status === 'fnt') return {frac: 0, cur: 0, max: 0, status: 'fnt'};
  const [curText, maxText] = hp.split('/');
  const cur = parseInt(curText, 10), max = parseInt(maxText, 10), colour = maxText.replace(/[0-9]/g, '');
  let frac = cur / max;
  if (max === 100 && cur < 100) frac = (cur + 0.5) / 100;
  if (max === 100 && colour) frac = cur === 50 ? (colour === 'g' ? 0.505 : 0.495) : (colour === 'y' ? 0.205 : 0.195);
  return {frac, cur, max, status: status || ''};
}
const BASE_DURATION = {tailwind: 4, trickroom: 5, reflect: 5, lightscreen: 5, auroraveil: 5, safeguard: 5,
  mist: 5, luckychant: 5, gravity: 5, wonderroom: 5, magicroom: 5};
const PROTECT = new Set(['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'obstruct',
  'silktrap', 'burningbulwark', 'endure', 'wideguard', 'quickguard']);

class View {
  constructor(sideId, myPaste) {
    this.me = sideId;
    this.foe = sideId === 'p1' ? 'p2' : 'p1';
    this.mySets = Teams.import(myPaste);
    this.foeSheet = null;            // open team sheet: species, item, ability, moves, nature
    this.foePreview = [];
    this.mons = new Map();           // 'p2: Name' -> tracked state
    this.active = {p1: [null, null], p2: [null, null]};
    this.turn = 0;
    this.field = {weather: null, terrain: null, pseudo: {}};
    this.sides = {p1: {}, p2: {}};
    this.megaUsed = {p1: false, p2: false};
    this.request = null;
    this.ended = false;
    this.speed = new Map();          // foe name -> {min, max} raw Speed stat
    this.myStats = {};               // our name -> Speed stat, from the latest request
    this.moveOrder = [];             // this turn's moves, with each mover's Speed modifier then
    this.orderSpoiled = false;
  }
  // Turns of end-of-turn countdown seen so far: the turn number, plus one once this turn's upkeep ran.
  clock() { return this.turn + (this.upkeep ? 1 : 0); }
  mon(ident) {
    // 'p2a: Name' or 'p2: Name' -> state record
    const m = /^(p\d)([ab]?): (.*)$/.exec(ident);
    if (!m) return null;
    const key = `${m[1]}: ${m[3]}`;
    if (!this.mons.has(key)) {
      this.mons.set(key, {side: m[1], name: m[3], species: m[3], frac: 1, status: '', boosts: {}, volatiles: {},
        fainted: false, revealed: false, item: undefined, itemGone: false, megaed: false,
        switchedTurn: 0, lastMove: null, lastMoveTurn: -1, protectChain: 0, movesSinceSwitch: 0});
    }
    return this.mons.get(key);
  }
  receive(chunk) {
    for (const line of chunk.split('\n')) this.line(line);
  }
  line(line) {  // eslint-disable-line complexity
    if (!line.startsWith('|')) return;
    if (this.keepLines) this.lines.push(line);
    const a = line.slice(1).split('|');
    const kind = a[0];
    const from = a.find(x => x.startsWith('[from]')) || '';
    if (kind === 'request') {
      if (a[1]) {
        this.request = JSON.parse(a.slice(1).join('|'));
        for (const p of this.request.side.pokemon) this.myStats[p.ident.slice(4)] = p.stats.spe;
      }
      return;
    }
    if (kind === 'poke') { if (a[1] === this.foe) this.foePreview.push(a[2].split(',')[0]); return; }
    if (kind === 'showteam') { if (a[1] === this.foe) this.foeSheet = Teams.unpack(a.slice(2).join('|')); return; }
    if (kind === 'turn') {
      this.boundSpeeds();
      this.turn = Number(a[1]); this.moveOrder = []; this.epoch = 0; this.spoiled = false; this.upkeep = false;
      return;
    }
    if (kind === 'upkeep') { this.upkeep = true; return; }
    if (kind === 'win' || kind === 'tie') { this.ended = true; return; }
    // Moves are compared only within one epoch: anything that can change Speed or order starts a new one.
    if ((kind === '-boost' || kind === '-unboost' || kind === '-setboost') && a[2] === 'spe') this.epoch = (this.epoch || 0) + 1;
    if ((kind === '-status' && a[2] === 'par') || ['-clearallboost', '-clearboost', '-fieldstart', '-fieldend',
      '-sidestart', '-sideend', '-weather', '-enditem', '-item', 'detailschange', '-mega', 'switch', 'drag'].includes(kind)) {
      this.epoch = (this.epoch || 0) + 1;
    }
    // Quick Claw, After You, Quash and the like reorder the turn: no bound from it at all.
    if (kind === '-activate' && /Quick Claw|After You|Quash|Custap|Quick Draw|Instruct/.test(line)) this.spoiled = true;
    if (kind === 'switch' || kind === 'drag' || kind === 'replace') {
      const slot = a[1][2] === 'a' ? 0 : 1, side = a[1].slice(0, 2);
      const old = this.active[side][slot];
      if (old && kind !== 'replace') { old.boosts = {}; old.volatiles = {}; old.protectChain = 0; }
      const m = this.mon(a[1]);
      m.species = a[2].split(',')[0];
      const hp = parseHP(a[3]);
      if (Number.isFinite(hp.frac)) { m.frac = hp.frac; m.status = hp.status === 'fnt' ? '' : hp.status; }
      m.revealed = true;
      m.switchedTurn = this.turn; m.movesSinceSwitch = 0; m.lastMove = null;
      this.active[side][slot] = m;
    } else if (kind === 'detailschange' || kind === '-formechange') {
      this.mon(a[1]).species = a[2].split(',')[0];
    } else if (kind === '-mega') {
      const m = this.mon(a[1]);
      m.megaed = true; this.megaUsed[m.side] = true;
    } else if (['-damage', '-heal', '-sethp'].includes(kind)) {
      const m = this.mon(a[1]), hp = parseHP(a[2]);
      if (!Number.isFinite(hp.frac)) return;
      m.frac = hp.frac; if (hp.status !== 'fnt') m.status = hp.status;
    } else if (kind === 'faint') {
      const m = this.mon(a[1]); m.fainted = true; m.frac = 0;
    } else if (kind === '-status') {
      this.mon(a[1]).status = a[2];
    } else if (kind === '-curestatus') {
      this.mon(a[1]).status = '';
    } else if (kind === '-cureteam') {
      for (const m of this.mons.values()) if (m.side === a[1].slice(0, 2)) m.status = '';
    } else if (kind === '-boost' || kind === '-unboost') {
      const m = this.mon(a[1]), n = Number(a[3]) * (kind === '-boost' ? 1 : -1);
      m.boosts[a[2]] = Math.max(-6, Math.min(6, (m.boosts[a[2]] || 0) + n));
    } else if (kind === '-setboost') {
      this.mon(a[1]).boosts[a[2]] = Number(a[3]);
    } else if (kind === '-clearboost' || kind === '-clearallboost') {
      if (kind === '-clearallboost') for (const s of ['p1', 'p2']) for (const m of this.active[s]) { if (m) m.boosts = {}; }
      else this.mon(a[1]).boosts = {};
    } else if (kind === '-clearnegativeboost') {
      const m = this.mon(a[1]);
      for (const k of Object.keys(m.boosts)) if (m.boosts[k] < 0) m.boosts[k] = 0;
    } else if (kind === '-copyboost') {
      this.mon(a[1]).boosts = {...this.mon(a[2]).boosts};
    } else if (kind === '-invertboost') {
      const m = this.mon(a[1]);
      for (const k of Object.keys(m.boosts)) m.boosts[k] = -m.boosts[k];
    } else if (kind === '-weather') {
      if (a[2] === '[upkeep]') return;
      if (a[1] === 'none') { this.field.weather = null; return; }
      const of = a.find(x => x.startsWith('[of] '));
      const source = of ? this.mon(of.slice(5)) : this.lastMover;
      this.field.weather = {id: toID(a[1]), start: this.clock(), source};
    } else if (kind === '-fieldstart') {
      const id = toID(a[1].replace(/^move: /, ''));
      const of = a.find(x => x.startsWith('[of] '));
      const source = of ? this.mon(of.slice(5)) : this.lastMover;
      if (id.endsWith('terrain')) this.field.terrain = {id, start: this.clock(), source};
      else this.field.pseudo[id] = {id, start: this.clock(), source};
    } else if (kind === '-fieldend') {
      const id = toID(a[1].replace(/^move: /, ''));
      if (id.endsWith('terrain')) this.field.terrain = null; else delete this.field.pseudo[id];
    } else if (kind === '-sidestart') {
      const side = a[1].slice(0, 2), id = toID(a[2].replace(/^move: /, ''));
      const cur = this.sides[side][id];
      this.sides[side][id] = {id, start: this.clock(), layers: (cur ? cur.layers || 1 : 0) + 1, source: this.lastMover};
    } else if (kind === '-sideend') {
      delete this.sides[a[1].slice(0, 2)][toID(a[2].replace(/^move: /, ''))];
    } else if (kind === '-start') {
      const m = this.mon(a[1]); const id = toID(a[2].replace(/^move: /, ''));
      m.volatiles[id] = {turn: this.turn};
    } else if (kind === '-end') {
      const m = this.mon(a[1]); delete m.volatiles[toID(a[2].replace(/^move: /, ''))];
    } else if (kind === '-enditem') {
      const m = this.mon(a[1]);
      if (!/\[from\] (ability: Frisk|ability: Harvest)/.test(line)) { m.item = ''; m.itemGone = true; }
    } else if (kind === '-item') {
      const m = this.mon(a[1]); m.item = toID(a[2]); m.itemGone = false;
    } else if (kind === 'cant') {
      // A turn it could not move still counts as an action; '[of]' marks a blocker (Armor Tail), not the mover.
      if (!a.some(x => x.startsWith('[of]'))) this.mon(a[1]).movesSinceSwitch++;
    } else if (kind === '-fail' && this.lastMover && a[1].replace(/^(p\d)[ab]:/, '$1:') === `${this.lastMover.side}: ${this.lastMover.name}` &&
               PROTECT.has(this.lastMover.lastMove)) {
      this.lastMover.protectChain = 0;          // a failed Protect leaves no counter
    } else if (kind === 'move') {
      const m = this.mon(a[1]);
      this.lastMover = m;
      if (!from) {
        const id = toID(a[2]);
        m.protectChain = PROTECT.has(id) && m.lastMoveTurn === this.turn - 1 && PROTECT.has(m.lastMove)
          ? m.protectChain + 1 : (PROTECT.has(id) ? 1 : 0);
        m.lastMove = id; m.lastMoveTurn = this.turn; m.movesSinceSwitch++;
        const set = m.side === this.me ? this.mySet(m.name) : this.foeSet(m.name);
        this.moveOrder.push({m, set, epoch: this.epoch || 0, priority: priorityOf(id, set),
          mod: speedModifier(this, m, set), trickRoom: !!this.field.pseudo.trickroom});
      }
    }
  }
  // Foul Play narrows a foe's Speed from who moved first (fp/battle/inference.py). Two moves in one
  // epoch with the same priority bracket: our effective Speed is known exactly, so the foe's raw
  // Speed stat is bounded once its own visible modifiers are divided out.
  boundSpeeds() {
    if (this.spoiled) return;
    const order = this.moveOrder;
    for (let i = 0; i < order.length; i++) for (let j = i + 1; j < order.length; j++) {
      const [x, y] = [order[i], order[j]];
      if (x.epoch !== y.epoch || x.priority === null || x.priority !== y.priority || x.m.side === y.m.side) continue;
      if (x.mod === null || y.mod === null || x.trickRoom !== y.trickRoom) continue;
      const mine = x.m.side === this.me ? x : y, foe = mine === x ? y : x;
      const stat = this.myStats[mine.m.name];
      if (!stat) continue;
      const mineSpeed = Math.floor(stat * mine.mod);
      const foeFaster = (foe === x) !== x.trickRoom;
      const bound = this.speed.get(foe.m.name) || {min: 0, max: 9999};
      if (foeFaster) bound.min = Math.max(bound.min, Math.floor(mineSpeed / foe.mod) - 1);
      else bound.max = Math.min(bound.max, Math.ceil(mineSpeed / foe.mod) + 1);
      this.speed.set(foe.m.name, bound);
    }
  }
  foeSet(name) { return this.foeSheet ? matchSet(this.foeSheet, name) : null; }
  mySet(name) { return matchSet(this.mySets, name); }
}

// ---------------------------------------------------------------- Speed modifiers a player can see
const SPEED_ABILITY = {swiftswim: 'raindance', chlorophyll: 'sunnyday', sandrush: 'sandstorm', slushrush: 'snowscape'};
function speedModifier(view, m, set) {
  let mod = 1;
  const b = m.boosts.spe || 0;
  mod *= b >= 0 ? (2 + b) / 2 : 2 / (2 - b);
  if (view.sides[m.side].tailwind) mod *= 2;
  if (m.status === 'par') mod *= 0.5;
  const item = m.itemGone ? '' : toID(m.item !== undefined ? m.item : set && set.item);
  if (item === 'choicescarf') mod *= 1.5;
  if (item === 'ironball') mod *= 0.5;
  const ability = toID(set && set.ability);
  if (SPEED_ABILITY[ability]) {
    const w = view.field.weather && view.field.weather.id;
    if (w && (w === SPEED_ABILITY[ability] || (ability === 'slushrush' && w === 'snow'))) mod *= 2;
  }
  if (['unburden', 'quickfeet', 'surgesurfer', 'protosynthesis', 'quarkdrive', 'slowstart'].includes(ability)) return null;
  return mod;
}
function priorityOf(moveId, set) {
  const move = Dex.moves.get(moveId);
  let p = move.priority;
  const ability = toID(set && set.ability);
  if (ability === 'prankster' && move.category === 'Status') p += 1;
  if (ability === 'triage' && move.flags && move.flags.heal) p += 3;
  if (ability === 'galewings' || ability === 'stall' || ability === 'myceliummight') return null;
  return p;
}
// ---------------------------------------------------------------- the opponent's hidden Stat Points
// A team dataset in Foul Play's sense: Stat Points seen on public teams, keyed by species, nature
// and item. Built by foul_play_doubles_battles.py from the VGCPastes Reg M-B pool.
class Spreads {
  constructor(table) { this.table = table || {}; this.scratch = null; }
  candidates(set) {
    const sp = toID(set.species), t = this.table[sp] || [];
    const tiers = [t.filter(r => r.nature === set.nature && toID(r.item) === toID(set.item)),
      t.filter(r => r.nature === set.nature), t];
    return tiers.find(x => x.length) || [];
  }
  speedStat(set, evs) {
    if (!this.scratch) this.scratch = new Battle({formatid: FORMAT, seed: [1, 2, 3, 4]});
    const sp = this.scratch.dex.species.get(set.species);
    return this.scratch.spreadModify(sp.baseStats, {...set, evs}).spe;
  }
  sample(set, bound, rng) {
    const all = this.candidates(set);
    let ok = bound ? all.filter(r => { const s = this.speedStat(set, r.evs); return s >= bound.min && s <= bound.max; }) : all;
    if (!ok.length) ok = all;
    if (!ok.length) return defaultSpread(set);
    return {...weightedPick(ok, ok.map(r => r.count), rng).evs};
  }
}
function defaultSpread(set) {
  // No dataset row: 32 points in the two stats the nature favours most plausibly, 2 in HP.
  const nature = Dex.natures.get(set.nature || 'Serious');
  const physical = set.moves.some(m => Dex.moves.get(m).category === 'Physical');
  const atk = nature.plus === 'spa' || (!physical && nature.plus !== 'atk') ? 'spa' : 'atk';
  return {hp: 2, atk: atk === 'atk' ? 32 : 0, def: 0, spa: atk === 'spa' ? 32 : 0, spd: 0, spe: 32};
}

// ---------------------------------------------------------------- one sampled battle
function orderMine(view) {
  const req = view.request;
  const mons = req.side.pokemon;
  const names = mons.map(p => p.ident.slice(4));
  return {mons, names};
}

function pickFoeTeam(view, affinities, rng) {
  const actives = view.active[view.foe].map(m => m && m.name);
  const revealed = [...view.mons.values()].filter(m => m.side === view.foe && m.revealed).map(m => m.name);
  const chosen = [];
  for (const n of actives) if (n && !chosen.includes(n)) chosen.push(n);
  for (const n of revealed) if (!chosen.includes(n)) chosen.push(n);
  let pool = view.foeSheet.map(setName).filter(n => !chosen.includes(n));
  while (chosen.length < 4 && pool.length) {
    const pick = weightedPick(pool, pool.map(n => Math.max(1e-3, (affinities && affinities[n]) || 1)), rng);
    chosen.push(pick);
    pool = pool.filter(n => n !== pick);
  }
  return chosen;
}

// Turns left on a field or side effect: its full length by Showdown's own rule for its source
// (Heat Rock, Light Clay, Terrain Extender), minus the end-of-turn countdowns since it started.
function remaining(view, b, rec) {
  const cond = b.dex.conditions.get(rec.id);
  let total = cond.duration || BASE_DURATION[rec.id] || 5;
  if (cond.durationCallback) {
    const src = rec.source && builtMon(b, view, rec.source);
    // hasItem ignores the items of benched Pokémon; the source was active when it set the effect.
    const wasActive = src && src.isActive;
    if (src) src.isActive = true;
    // Weather and terrain take (source, effect); side conditions and rooms take (target, source, effect).
    const args = cond.effectType === 'Weather' || rec.id.endsWith('terrain') ? [src, cond] : [src, src, cond];
    try { total = cond.durationCallback.call(b, ...args) || total; } catch (e) { /* keep the default */ }
    if (src) src.isActive = wasActive;
  }
  return Math.max(1, total - (view.clock() - Math.max(rec.start, 1)));
}
function builtMon(b, view, m) {
  const side = b.sides[m.side === view.me ? 0 : 1];
  return side.pokemon.find(p => p.name === m.name) || null;
}

// Build a Showdown battle that matches the view, with our side as p1 and sampled hidden information.
function determinize(view, spreads, affinities, rng) {  // eslint-disable-line complexity
  const preview = view.request.teamPreview;
  let mine, theirs;
  if (preview) {
    mine = view.mySets.map(s => ({...s}));
    theirs = view.foeSheet.map(s => ({...s, ivs: s.ivs || IVS, evs: spreads.sample(s, null, rng)}));
  } else {
    const {names} = orderMine(view);
    mine = names.map(n => { const s = view.mySet(n); if (!s) throw new Error(`own set missing: ${n} in ${names}`); return {...s}; });
    const foeNames = pickFoeTeam(view, affinities, rng);
    theirs = foeNames.map(n => {
      const s = view.foeSet(n);
      if (!s) throw new Error(`foe set missing: ${n} in ${view.foeSheet.map(setName)}`);
      return {...s, ivs: s.ivs || IVS, evs: spreads.sample(s, view.speed.get(n), rng)};
    });
  }
  const b = new Battle({formatid: FORMAT, seed: seed4(rng)});
  b.setPlayer('p1', {name: 'me', team: Teams.pack(mine)});
  b.setPlayer('p2', {name: 'foe', team: Teams.pack(theirs)});
  if (preview) return b;
  b.choose('p1', 'team 1234');
  b.choose('p2', 'team 1234');
  const sideFor = {[view.me]: b.sides[0], [view.foe]: b.sides[1]};
  // Put the observed active Pokémon in their slots (the 4 were packed actives-first already for p2).
  const req = view.request;
  for (const [key, side] of Object.entries(sideFor)) {
    const wanted = key === view.me
      ? req.side.pokemon.filter(p => p.active).map(p => p.ident.slice(4))
      : view.active[key].map(m => m && m.name);
    wanted.forEach((name, slot) => {
      if (!name) return;
      const idx = side.pokemon.findIndex(p => p.name === name);
      if (idx >= 0 && idx !== slot) {
        const [x, y] = [side.pokemon[slot], side.pokemon[idx]];
        side.pokemon[slot] = y; side.pokemon[idx] = x;
        y.position = slot; x.position = idx;
        if (slot < side.active.length) { side.active[slot] = y; y.isActive = true; x.isActive = idx < side.active.length; }
      }
    });
  }
  // Our side: exact HP, status, item and PP from the request; foe: what the log showed.
  const myReq = Object.fromEntries(req.side.pokemon.map(p => [p.ident.slice(4), p]));
  for (const side of b.sides) {
    for (const p of side.pokemon) {
      const seen = view.mons.get(`${side === b.sides[0] ? view.me : view.foe}: ${p.name}`);
      if (seen && seen.megaed && p.canMegaEvo) b.actions.runMegaEvo(p);
    }
  }
  // Protect's counter and the Fake Out window count moves up to this decision; a switch request
  // mid-turn or at its end comes after this turn's moves.
  const lastTurn = req.forceSwitch ? view.turn : view.turn - 1;
  for (const side of b.sides) {
    for (const p of side.pokemon) {
      const mine = side === b.sides[0];
      const seen = view.mons.get(`${mine ? view.me : view.foe}: ${p.name}`);
      p.boosts = {atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0};
      Object.keys(p.volatiles).forEach(v => p.removeVolatile(v));
      if (mine) {
        const r = myReq[p.name];
        const hp = parseHP(r.condition);
        p.item = toID(r.item);
        if (hp.status === 'fnt') { faint(p); continue; }
        p.hp = hp.cur;
        if (hp.status) p.setStatus(hp.status, p, null, true);
        else if (p.status) p.clearStatus();
        const data = req.active && p.isActive ? req.active[p.position] : null;
        if (data) for (const m of data.moves) {
          const slot = p.moveSlots.find(s => s.id === m.id);
          if (slot && m.pp !== undefined) slot.pp = m.pp;
        }
      } else if (seen) {
        if (seen.fainted || seen.frac === 0) { faint(p); continue; }
        p.hp = Math.max(1, Math.round(seen.frac * p.maxhp));
        if (seen.status) p.setStatus(seen.status, p, null, true);
        else if (p.status) p.clearStatus();
        // The rebuilt battle's own switch-ins may have used an item (White Herb after Intimidate).
        p.item = seen.itemGone ? '' : toID(seen.item !== undefined ? seen.item : p.set.item);
        p.itemState = b.initEffectState({id: p.item, target: p});
      }
      if (seen && p.isActive && !seen.megaed && seen.species !== p.species.name && b.dex.species.get(seen.species).exists) {
        try { p.formeChange(seen.species, null, false); } catch (e) { /* keep the base forme */ }
      }
      if (seen && p.isActive) {
        for (const [k, v] of Object.entries(seen.boosts)) if (k in p.boosts) p.boosts[k] = v;
        if (seen.lastMove) p.lastMove = b.dex.getActiveMove(seen.lastMove);
        for (const v of ['confusion', 'substitute', 'taunt', 'encore', 'yawn', 'leechseed', 'perish3', 'perish2',
          'perish1', 'disable', 'torment', 'healblock']) {
          if (!seen.volatiles[v]) continue;
          if ((v === 'encore' || v === 'disable') && !seen.lastMove) continue;
          withActiveMove(b, seen.lastMove, () => p.addVolatile(v.startsWith('perish') ? 'perishsong' : v, p));
        }
        p.activeTurns = Math.max(0, view.turn - Math.max(seen.switchedTurn, 1));
        p.activeMoveActions = seen.movesSinceSwitch;
        if (seen.protectChain && seen.lastMoveTurn === lastTurn) {
          p.addVolatile('stall');
          if (p.volatiles.stall) { p.volatiles.stall.counter = Math.pow(3, seen.protectChain); p.volatiles.stall.duration = 1; }
        }
        if (seen.lastMove && seen.movesSinceSwitch > 0 && b.dex.items.get(p.item).isChoice) {
          withActiveMove(b, seen.lastMove, () => p.addVolatile('choicelock'));
        }
      }
    }
    if (view.megaUsed[side === b.sides[0] ? view.me : view.foe]) for (const p of side.pokemon) p.canMegaEvo = null;
  }
  // Field and side conditions.
  b.field.weather = ''; b.field.weatherState = b.initEffectState({id: ''});
  b.field.terrain = ''; b.field.terrainState = b.initEffectState({id: ''});
  for (const id of Object.keys(b.field.pseudoWeather)) delete b.field.pseudoWeather[id];
  const w = view.field.weather;
  if (w) {
    b.field.weather = w.id;
    b.field.weatherState = b.initEffectState({id: w.id, duration: remaining(view, b, w)});
  }
  const t = view.field.terrain;
  if (t) {
    b.field.terrain = t.id;
    b.field.terrainState = b.initEffectState({id: t.id, duration: remaining(view, b, t)});
  }
  for (const pw of Object.values(view.field.pseudo)) {
    if (!b.field.addPseudoWeather(pw.id, 'debug')) continue;
    if (b.field.pseudoWeather[pw.id].duration) b.field.pseudoWeather[pw.id].duration = remaining(view, b, pw);
  }
  for (const [key, side] of Object.entries(sideFor)) {
    for (const id of Object.keys(side.sideConditions)) delete side.sideConditions[id];
    for (const sc of Object.values(view.sides[key])) {
      if (!side.addSideCondition(sc.id, 'debug')) continue;
      if (side.sideConditions[sc.id].duration) side.sideConditions[sc.id].duration = remaining(view, b, sc);
      if (sc.layers > 1) side.sideConditions[sc.id].layers = sc.layers;
    }
  }
  b.turn = view.turn;
  b.queue.clear();
  b.log = []; b.sentLogPos = 0;
  if (b.ended) return null;
  if (req.forceSwitch) {
    let any = false;
    for (const side of b.sides) side.active.forEach(p => {
      const benchAlive = side.pokemon.some((q, i) => i >= side.active.length && !q.fainted && q.hp > 0);
      p.switchFlag = !!(p && (p.fainted || p.hp <= 0) && benchAlive);
      if (side === b.sides[0]) p.switchFlag = !!req.forceSwitch[p.position];
      any = any || p.switchFlag;
    });
    if (!any) return null;
    b.makeRequest('switch');
  } else {
    b.makeRequest('move');
  }
  return b;
}
// Conditions whose start handler reads the move being used (choice lock, Encore, Disable).
function withActiveMove(b, moveId, fn) {
  const saved = b.activeMove;
  try {
    if (moveId) b.activeMove = b.dex.getActiveMove(moveId);
    fn();
  } catch (e) {
    // A volatile that cannot be rebuilt is left off this sample.
  } finally {
    b.activeMove = saved;
  }
}
function faint(p) {
  if (p.fainted) return;
  p.hp = 0; p.fainted = true; p.status = 'fnt'; p.switchFlag = false; p.isActive = false;
  p.side.pokemonLeft--;
}

// ---------------------------------------------------------------- decoupled UCT, UCB statistics per slot
class Node {
  constructor() { this.visits = 0; this.slots = null; this.children = new Map(); }
}
function initNode(node, battle, real = null) {
  node.slots = [0, 1].map(i => {
    const opts = slotOptions(battle, i, i === 0 ? real : null);
    return opts ? opts.map(list => list.map(o => ({...o, n: 0, w: 0}))) : null;
  });
}
function ucbPick(list, parentVisits, taken, rng) {
  let best = -1, bestScore = -Infinity, unvisited = [];
  for (let i = 0; i < list.length; i++) {
    const o = list[i];
    if (o.dead || (taken && !compatible(o, taken))) continue;
    if (o.n === 0) { unvisited.push(i); continue; }
    const s = o.w / o.n + Math.sqrt(2 * Math.log(Math.max(1, parentVisits)) / o.n);
    if (s > bestScore) { bestScore = s; best = i; }
  }
  if (unvisited.length) return unvisited[Math.floor(rng() * unvisited.length)];
  return best;
}
function pickSide(node, i, rng, policy) {
  const slots = node.slots[i];
  if (!slots) return null;
  const picks = [];
  let taken = null;
  for (const list of slots) {
    let k = policy === 'random' ? randomIndex(list, taken, rng) : ucbPick(list, node.visits, taken, rng);
    if (k < 0) k = list.findIndex(o => !o.dead);
    if (k < 0) k = 0;
    picks.push(k);
    taken = list[k];
  }
  return picks;
}
function randomIndex(list, taken, rng) {
  const ok = list.map((o, i) => i).filter(i => !list[i].dead && (!taken || compatible(list[i], taken)));
  return ok.length ? ok[Math.floor(rng() * ok.length)] : -1;
}
const choiceString = (node, i, picks) => node.slots[i].map((list, s) => list[picks[s]].choice).join(', ');

// Apply both sides' picks; an option Showdown rejects is marked dead and re-picked.
function applyPicks(battle, node, rng, policy) {
  const picks = [];
  for (let i = 0; i < 2; i++) {
    let p = pickSide(node, i, rng, policy);
    for (let tries = 0; p && tries < 6; tries++) {
      const ok = battle.sides[i].choose(choiceString(node, i, p)) && battle.sides[i].isChoiceDone();
      if (ok) break;
      battle.sides[i].clearChoice();
      // Mark the first slot option that fails on its own.
      let marked = false;
      for (let s = 0; s < p.length && !marked; s++) {
        const alone = node.slots[i].map((list, t) => (t === s ? list[p[s]].choice : 'default'));
        if (!(battle.sides[i].choose(alone.join(', ')))) { node.slots[i][s][p[s]].dead = true; marked = true; }
        battle.sides[i].clearChoice();
      }
      if (!marked) node.slots[i][p.length - 1][p[p.length - 1]].dead = true;
      p = pickSide(node, i, rng, policy);
      if (tries === 5) { battle.sides[i].autoChoose(); p = null; }
    }
    picks.push(p);
  }
  if (battle.allChoicesDone()) battle.commitChoices();
  // Nothing reads a searched battle's log; Showdown stops a battle whose unsent log passes 1,000 lines.
  battle.log.length = 0; battle.sentLogPos = 0;
  return picks;
}
function signature(battle) {
  return battle.sides.map(s => s.active.map(p => (p && !p.fainted ? p.name : '-')).join('/') +
    ':' + (s.activeRequest ? (s.activeRequest.wait ? 'w' : s.activeRequest.forceSwitch ? 'f' + s.activeRequest.forceSwitch.map(Number).join('') : 'm') : 'n')).join('|');
}
function terminal(battle) {
  if (!battle.ended) return null;
  if (battle.winner === 'me') return 1;
  if (!battle.winner) return 0.5;
  return 0;
}

function mcts(rootBattle, budget, rng, real = null, maxDepth = 12, opts = {}) {
  const leafMode = opts.leafMode || 'singles'; // 'singles' | 'doubles' | 'rollout' | 'hybrid'
  const evalFn = leafMode === 'singles' ? evaluate : evaluateDoubles;
  const sigFn = leafMode === 'singles' ? sigmoid : sigmoidDoubles;
  const rootJSON = rootBattle.toJSON();
  const rootEval = evalFn(rootBattle);
  const root = new Node();
  initNode(root, rootBattle, real);
  const deadline = performance.now() + budget.ms;
  let iterations = 0;
  while (iterations < (budget.iterations || Infinity) && (budget.iterations || performance.now() < deadline)) {
    const b = Battle.fromJSON(rootJSON);
    b.prng = new PRNG(seed4(rng));
    let node = root, score = null;
    const path = [];
    for (let depth = 0; depth < maxDepth; depth++) {
      const picks = applyPicks(b, node, rng, 'ucb');
      path.push([node, picks]);
      score = terminal(b);
      if (score !== null) break;
      const key = picks.map(p => (p ? p.join(',') : '-')).join('/') + '#' + (node === root && rootBattle.sides[0].activeRequest.teamPreview ? leadKey(b) : signature(b));
      let child = node.children.get(key);
      if (!child) {
        child = new Node(); node.children.set(key, child); initNode(child, b);
        if (!(node === root && rootBattle.sides[0].activeRequest.teamPreview)) {
          if (leafMode === 'rollout') {
            randomPlayout(b, rng, 60);
            score = terminal(b) ?? sigFn(evalFn(b) - rootEval);
          } else if (leafMode === 'hybrid') {
            randomPlayout(b, rng, 2);
            score = terminal(b) ?? sigFn(evalFn(b) - rootEval);
          } else {
            score = sigFn(evalFn(b) - rootEval);
          }
          break;
        }
      }
      node = child;
      if (!node.slots) initNode(node, b);
    }
    if (score === null) {
      if (leafMode === 'rollout') {
        randomPlayout(b, rng, 60);
        score = terminal(b) ?? sigFn(evalFn(b) - rootEval);
      } else {
        score = sigFn(evalFn(b) - rootEval);
      }
    }
    for (const [n, picks] of path) {
      n.visits++;
      picks.forEach((p, i) => {
        if (!p) return;
        p.forEach((k, s) => { const o = n.slots[i][s][k]; o.n++; o.w += i === 0 ? score : 1 - score; });
      });
    }
    iterations++;
  }
  return {root, iterations};
}
const leadKey = b => b.sides.map(s => s.active.map(p => p.name).sort().join('+')).join('|');

// ---------------------------------------------------------------- joint-action progressive-widening UCT
const TACTICAL_MOVES = new Set(['fakeout', 'protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker',
  'burningbulwark', 'wideguard', 'quickguard', 'followme', 'ragepowder', 'tailwind', 'trickroom',
  'helpinghand', 'icywind', 'electroweb', 'snarl', 'partingshot', 'spore', 'willowisp', 'thunderwave',
  'taunt', 'encore', 'coaching', 'allyswitch', 'substitute', 'swordsdance', 'nastyplot', 'calmmind',
  'dragondance', 'quiverdance', 'bulkup']);

function singleOptPrior(battle, sideIndex, slot, opt) {
  if (!opt || opt.choice === 'pass') return 0.1;
  if (opt.preview) return 1.0;
  if (opt.switchTo) return 0.6;
  const side = battle.sides[sideIndex];
  const me = side.active[slot];
  if (!me || me.fainted) return 0.1;
  const parts = opt.choice.split(' ');
  const mIdx = Number(parts[1]) - 1;
  const ms = me.moveSlots && me.moveSlots[mIdx];
  const mv = ms ? battle.dex.moves.get(ms.id) : null;
  if (!mv) return 0.5;
  let score = 1.0;
  if (opt.mega) score += 0.65;
  if (mv.id === 'fakeout') score += me.activeMoveActions === 0 ? 1.4 : -0.9;
  else if (PROTECT.has(mv.id)) score += me.volatiles && me.volatiles.stall ? -0.8 : 0.7;
  else if (TACTICAL_MOVES.has(mv.id)) score += 0.75;
  if (mv.category !== 'Status') {
    const bp = mv.basePower || 60;
    score += Math.min(1.2, bp / 100);
    if (me.hasType && me.hasType(mv.type)) score += 0.25;
    // Check target type effectiveness when targeting a specific foe.
    const tgt = Number(parts[2]);
    if (tgt > 0) {
      const foe = side.foe.active[tgt - 1];
      if (foe && !foe.fainted) {
        if (!battle.dex.getImmunity(mv.type, foe)) score -= 1.6;
        else {
          const eff = battle.dex.getEffectiveness(mv.type, foe);
          if (eff > 0) score += 0.5 * eff;
          else if (eff < 0) score += 0.25 * eff;
        }
      }
    }
  }
  return Math.max(0.05, score);
}

function jointOptionsForSide(battle, sideIndex, real = null) {
  const slots = slotOptions(battle, sideIndex, real);
  if (!slots) return null;
  if (slots.length === 1) {
    return slots[0].map(o => ({choice: o.choice, preview: o.preview, prior: 1 / slots[0].length, n: 0, w: 0}));
  }
  const [s0, s1] = slots;
  const raw = [];
  const p0s = s0.map(o => singleOptPrior(battle, sideIndex, 0, o));
  const p1s = s1.map(o => singleOptPrior(battle, sideIndex, 1, o));
  for (let i = 0; i < s0.length; i++) {
    for (let j = 0; j < s1.length; j++) {
      const a = s0[i], b = s1[j];
      if (!compatible(a, b)) continue;
      let sc = p0s[i] + p1s[j];
      // Slight penalty for double switching or double protecting unless needed.
      if (a.switchTo && b.switchTo) sc -= 0.5;
      raw.push({choice: `${a.choice}, ${b.choice}`, score: sc, n: 0, w: 0});
    }
  }
  if (!raw.length) return [{choice: 'default', prior: 1, n: 0, w: 0}];
  raw.sort((x, y) => y.score - x.score);
  // Softmax prior over tactical scores for PUCT ordering + progressive widening.
  const maxS = raw[0].score;
  let sum = 0;
  for (const r of raw) { r.exp = Math.exp((r.score - maxS) * 0.7); sum += r.exp; }
  return raw.map(r => ({choice: r.choice, prior: r.exp / sum, n: 0, w: 0}));
}

function initJointNode(node, battle, real = null) {
  node.joint = [0, 1].map(i => jointOptionsForSide(battle, i, i === 0 ? real : null));
}

function pickJointSide(node, i, rng) {
  const list = node.joint[i];
  if (!list || !list.length) return -1;
  // Progressive widening: unlock top K joint pairs as visits grow so synergy pairs get real depth.
  const limit = Math.min(list.length, Math.max(10, Math.floor(4 * Math.sqrt(Math.max(1, node.visits)))));
  const cPuct = 1.6;
  const sqrtN = Math.sqrt(Math.max(1, node.visits));
  let best = -1, bestScore = -Infinity;
  const unvisited = [];
  for (let k = 0; k < limit; k++) {
    const o = list[k];
    if (o.dead) continue;
    if (o.n === 0) { unvisited.push(k); continue; }
    const s = o.w / o.n + cPuct * o.prior * sqrtN / (1 + o.n) + 0.5 * Math.sqrt(Math.log(Math.max(2, node.visits)) / o.n);
    if (s > bestScore) { bestScore = s; best = k; }
  }
  if (unvisited.length) return unvisited[0]; // visit highest-prior unlocked pair first
  if (best >= 0) return best;
  for (let k = 0; k < list.length; k++) if (!list[k].dead) return k;
  return 0;
}

function applyJointPicks(battle, node, rng) {
  const picks = [];
  for (let i = 0; i < 2; i++) {
    if (!node.joint[i]) { picks.push(-1); continue; }
    let k = pickJointSide(node, i, rng);
    for (let tries = 0; k >= 0 && tries < 6; tries++) {
      const ok = battle.sides[i].choose(node.joint[i][k].choice) && battle.sides[i].isChoiceDone();
      if (ok) break;
      battle.sides[i].clearChoice();
      node.joint[i][k].dead = true;
      k = pickJointSide(node, i, rng);
      if (tries === 5) { battle.sides[i].autoChoose(); k = -1; }
    }
    picks.push(k);
  }
  if (battle.allChoicesDone()) battle.commitChoices();
  battle.log.length = 0; battle.sentLogPos = 0;
  return picks;
}

function mctsJoint(rootBattle, budget, rng, real = null, maxDepth = 12) {
  const rootJSON = rootBattle.toJSON();
  const rootEval = evaluateDoubles(rootBattle);
  const root = new Node();
  initJointNode(root, rootBattle, real);
  const deadline = performance.now() + budget.ms;
  let iterations = 0;
  while (iterations < (budget.iterations || Infinity) && (budget.iterations || performance.now() < deadline)) {
    const b = Battle.fromJSON(rootJSON);
    b.prng = new PRNG(seed4(rng));
    let node = root, score = null;
    const path = [];
    for (let depth = 0; depth < maxDepth; depth++) {
      const picks = applyJointPicks(b, node, rng);
      path.push([node, picks]);
      score = terminal(b);
      if (score !== null) break;
      const key = picks.join('/') + '#' + (node === root && rootBattle.sides[0].activeRequest.teamPreview ? leadKey(b) : signature(b));
      let child = node.children.get(key);
      if (!child) {
        child = new Node(); node.children.set(key, child); initJointNode(child, b);
        if (!(node === root && rootBattle.sides[0].activeRequest.teamPreview)) {
          score = sigmoidDoubles(evaluateDoubles(b) - rootEval);
          break;
        }
      }
      node = child;
      if (!node.joint) initJointNode(node, b);
    }
    if (score === null) score = sigmoidDoubles(evaluateDoubles(b) - rootEval);
    for (const [n, picks] of path) {
      n.visits++;
      picks.forEach((k, i) => {
        if (k < 0 || !n.joint[i]) return;
        const o = n.joint[i][k];
        o.n++; o.w += i === 0 ? score : 1 - score;
      });
    }
    iterations++;
  }
  return {root, iterations};
}

function combineJoint(results, rng, rule) {
  const map = new Map();
  for (const {root} of results) {
    const list = root.joint && root.joint[0];
    if (!list) continue;
    const total = list.reduce((a, o) => a + o.n, 0) || 1;
    for (const o of list) {
      const cur = map.get(o.choice) || {o, share: 0, n: 0, w: 0};
      cur.share += (o.n / total) / results.length; cur.n += o.n; cur.w += o.w;
      map.set(o.choice, cur);
    }
  }
  let entries = [...map.values()].filter(e => !e.o.dead);
  if (!entries.length) entries = [...map.values()];
  entries.sort((x, y) => (y.share - x.share) || ((y.n ? y.w / y.n : 0) - (x.n ? x.w / x.n : 0)));
  if (rule === 'foulplay') {
    const near = entries.filter(e => e.share >= entries[0].share * 0.75);
    return weightedPick(near, near.map(e => e.share), rng).o.choice;
  }
  return entries[0].o.choice;
}

function previewAffinitiesJoint(results, foeNames) {
  const score = {};
  for (const {root} of results) {
    const list = root.joint && root.joint[1];
    if (!list) continue;
    const total = list.reduce((a, o) => a + o.n, 0) || 1;
    for (const o of list) if (o.preview) for (const k of o.preview.brought) score[foeNames[k - 1]] = (score[foeNames[k - 1]] || 0) + o.n / total;
  }
  return score;
}

// ---------------------------------------------------------------- flat Monte Carlo and random play
function randomPlayout(b, rng, maxTurns = 60) {
  const start = b.turn;
  while (!b.ended && b.turn - start < maxTurns) {
    const node = new Node();
    initNode(node, b);
    if (!node.slots[0] && !node.slots[1]) break;
    applyPicks(b, node, rng, 'random');
  }
}
// The "Monte Carlo battle policy": each iteration draws both sides' first choice at random, plays
// both sides at random to the end, and credits the win to every slot option drawn; pick the best mean.
function flatMonteCarlo(rootBattle, budget, rng, real = null) {
  const rootJSON = rootBattle.toJSON();
  const root = new Node();
  initNode(root, rootBattle, real);
  const deadline = performance.now() + budget.ms;
  let iterations = 0;
  while (iterations < (budget.iterations || Infinity) && (budget.iterations || performance.now() < deadline)) {
    const b = Battle.fromJSON(rootJSON);
    b.prng = new PRNG(seed4(rng));
    const picks = applyPicks(b, root, rng, 'random');
    randomPlayout(b, rng);
    const score = terminal(b) ?? sigmoid(evaluate(b));
    root.visits++;
    picks.forEach((p, i) => p && p.forEach((k, s) => { const o = root.slots[i][s][k]; o.n++; o.w += i === 0 ? score : 1 - score; }));
    iterations++;
  }
  return {root, iterations};
}

// ---------------------------------------------------------------- the agent
// Root visit shares summed over samples (each weighted 1/N), options within 75% of the best kept,
// one drawn in proportion: select_move_from_mcts_results, applied slot by slot.
function combine(results, rng, rule) {
  const slots = [];
  for (const {root} of results) {
    const mine = root.slots[0];
    if (!mine) continue;
    mine.forEach((list, s) => {
      slots[s] = slots[s] || new Map();
      const total = list.reduce((a, o) => a + o.n, 0) || 1;
      for (const o of list) {
        const cur = slots[s].get(o.choice) || {o, share: 0, n: 0, w: 0};
        cur.share += (o.n / total) / results.length; cur.n += o.n; cur.w += o.w;
        slots[s].set(o.choice, cur);
      }
    });
  }
  const chosen = [];
  let taken = null;
  for (const m of slots) {
    let entries = [...m.values()].filter(e => !e.o.dead && (!taken || compatible(e.o, taken)));
    if (!entries.length) entries = [...m.values()];
    let pick;
    if (rule === 'mean') {
      pick = entries.reduce((a, e) => ((e.n ? e.w / e.n : -1) > (a.n ? a.w / a.n : -1) ? e : a), entries[0]);
    } else if (rule === 'argmax') {
      entries.sort((x, y) => (y.share - x.share) || ((y.n ? y.w / y.n : 0) - (x.n ? x.w / x.n : 0)));
      pick = entries[0];
    } else {
      entries.sort((x, y) => y.share - x.share);
      const near = entries.filter(e => e.share >= entries[0].share * 0.75);
      pick = weightedPick(near, near.map(e => e.share), rng);
    }
    chosen.push(pick);
    taken = pick.o;
  }
  return chosen;
}

// BSS-style opponent preferences: the share of side-two preview visits that brought each Pokémon.
function previewAffinities(results, foeNames) {
  const score = {};
  for (const {root} of results) {
    const list = root.slots[1] && root.slots[1][0];
    if (!list) continue;
    const total = list.reduce((a, o) => a + o.n, 0) || 1;
    for (const o of list) for (const k of o.preview.brought) score[foeNames[k - 1]] = (score[foeNames[k - 1]] || 0) + o.n / total;
  }
  return score;
}

const {getRustClient, serializeBattle} = require('./rust_engine_client.cjs');

class Agent {
  // policy: 'foulplay' | 'foulplay_det' | 'foulplay_doubles' | 'foulplay_joint' | 'foulplay_hybrid' |
  //         'uct_rollout' | 'montecarlo' | 'random' |
  //         'rust_foulplay' | 'rust_foulplay_det' | 'rust_foulplay_doubles' | 'rust_foulplay_joint' |
  //         'rust_foulplay_hybrid' | 'rust_uct_rollout' | 'rust_montecarlo'
  constructor({side, paste, policy = 'foulplay', samples = 4, ms = 250, previewMs = 1000, iterations = 0,
    seed = 1, spreads = null}) {
    this.view = new View(side, paste);
    this.policy = policy;
    this.samples = samples; this.ms = ms; this.previewMs = previewMs; this.iterations = iterations;
    this.rng = mulberry(seed);
    this.spreads = new Spreads(spreads);
    this.affinities = null;
    this.stats = {decisions: 0, iterations: 0, fallbacks: 0, failedSamples: 0, ms: 0};
  }
  receive(chunk) { this.view.receive(chunk); }
  // Returns a choice string for the pending request, or null when there is nothing to decide.
  decide() {
    const req = this.view.request;
    if (!req || req.wait || this.view.ended) return null;
    if (req.teamPreview && !this.view.foeSheet) return null;
    const t0 = performance.now();
    this.stats.decisions++;
    let choice;
    if (this.policy === 'random') choice = this.randomChoice();
    else choice = this.searchChoice(req);
    this.stats.ms += performance.now() - t0;
    this.view.request = null;
    return choice;
  }
  async decideAsync() {
    if (!this.policy.startsWith('rust_')) return this.decide();
    const req = this.view.request;
    if (!req || req.wait || this.view.ended) return null;
    if (req.teamPreview && !this.view.foeSheet) return null;
    const t0 = performance.now();
    this.stats.decisions++;
    const choice = req.teamPreview ? await this.rustPreviewChoice() : await this.rustSearchChoice(req);
    this.stats.ms += performance.now() - t0;
    this.view.request = null;
    return choice;
  }
  async rustPreviewChoice() {
    const subPolicy = this.policy.slice(5);
    const client = getRustClient();
    const results = [];
    for (let i = 0; i < this.samples; i++) {
      let b;
      try { b = determinize(this.view, this.spreads, null, this.rng); } catch (e) { b = null; }
      if (!b) { this.stats.failedSamples++; continue; }
      const wire = serializeBattle(b);
      const seed = Math.floor(this.rng() * 0xffffffff) + 1;
      const res = await client.request({
        cmd: 'preview',
        policy: subPolicy,
        ms: this.previewMs,
        iterations: this.iterations,
        seed,
        teams: [wire.sides[0].mons, wire.sides[1].mons],
      });
      this.stats.iterations += res.iterations || 0;
      results.push(res);
    }
    if (!results.length) { this.stats.fallbacks++; return 'team 1234'; }
    const foeNames = this.view.foeSheet.map(setName);
    const aff = {};
    for (const r of results) {
      (r.foe_affinities || []).forEach((v, k) => {
        if (foeNames[k]) aff[foeNames[k]] = (aff[foeNames[k]] || 0) + v / results.length;
      });
    }
    this.affinities = aff;
    const agg = new Map();
    for (const r of results) {
      for (const o of (r.options || [])) {
        const cur = agg.get(o.choice) || {choice: o.choice, share: 0, n: 0, w: 0};
        cur.share += o.share / results.length; cur.n += o.n; cur.w += o.w;
        agg.set(o.choice, cur);
      }
    }
    const entries = [...agg.values()];
    if (!entries.length) return results[0].choice;
    if (subPolicy === 'montecarlo') {
      return entries.reduce((a, e) => ((e.n ? e.w / e.n : -1) > (a.n ? a.w / a.n : -1) ? e : a), entries[0]).choice;
    }
    entries.sort((a, b) => (b.share - a.share) || ((b.n ? b.w / b.n : 0) - (a.n ? a.w / a.n : 0)));
    if (subPolicy === 'foulplay') {
      const near = entries.filter(e => e.share >= entries[0].share * 0.75);
      return weightedPick(near, near.map(e => e.share), this.rng).choice;
    }
    return entries[0].choice;
  }
  async rustSearchChoice(req) {
    const subPolicy = this.policy.slice(5);
    const isJoint = subPolicy === 'foulplay_joint';
    const client = getRustClient();
    const ms = this.ms;
    const results = [];
    for (let i = 0; i < this.samples; i++) {
      let b;
      try { b = determinize(this.view, this.spreads, this.affinities, this.rng); } catch (e) { b = null; }
      if (!b || !b.sides[0].activeRequest || b.sides[0].activeRequest.wait) { this.stats.failedSamples++; continue; }
      const myOpts = slotOptions(b, 0, req);
      if (!myOpts) { this.stats.failedSamples++; continue; }
      if (i === 0 && myOpts.every(list => list.length === 1)) {
        return myOpts.map(list => list[0].choice).join(', ');
      }
      const myOptStrings = [
        myOpts[0].map(o => o.choice),
        (myOpts[1] || [{choice: 'pass'}]).map(o => o.choice),
      ];
      const seed = Math.floor(this.rng() * 0xffffffff) + 1;
      const res = await client.request({
        cmd: 'search',
        policy: subPolicy,
        ms,
        iterations: this.iterations,
        seed,
        state: serializeBattle(b),
        myOptions: myOptStrings,
      });
      this.stats.iterations += res.iterations || 0;
      results.push({res, myOpts});
    }
    if (!results.length) { this.stats.fallbacks++; return 'default'; }
    if (isJoint) {
      const agg = new Map();
      for (const {res} of results) {
        for (const o of (res.joint || [])) {
          const cur = agg.get(o.choice) || {choice: o.choice, share: 0, n: 0, w: 0};
          cur.share += o.share / results.length; cur.n += o.n; cur.w += o.w;
          agg.set(o.choice, cur);
        }
      }
      const entries = [...agg.values()].sort((a, b) => (b.share - a.share) || ((b.n ? b.w / b.n : 0) - (a.n ? a.w / a.n : 0)));
      return entries.length ? entries[0].choice : results[0].res.choice;
    }
    // Combine slot shares across samples (same rule as combine())
    const slots = [new Map(), new Map()];
    for (const {res, myOpts} of results) {
      for (let s = 0; s < 2; s++) {
        const meta = new Map((myOpts[s] || []).map(o => [o.choice, o]));
        for (const o of (res.slots[s] || [])) {
          const cur = slots[s].get(o.choice) || {o: meta.get(o.choice) || {choice: o.choice}, share: 0, n: 0, w: 0};
          cur.share += o.share / results.length; cur.n += o.n; cur.w += o.w;
          slots[s].set(o.choice, cur);
        }
      }
    }
    const chosen = [];
    let taken = null;
    for (let s = 0; s < 2; s++) {
      let entries = [...slots[s].values()].filter(e => !taken || compatible(e.o, taken));
      if (!entries.length) entries = [...slots[s].values()];
      let pick;
      if (subPolicy === 'montecarlo') {
        pick = entries.reduce((a, e) => ((e.n ? e.w / e.n : -1) > (a.n ? a.w / a.n : -1) ? e : a), entries[0]);
      } else if (subPolicy === 'foulplay') {
        entries.sort((x, y) => y.share - x.share);
        const near = entries.filter(e => e.share >= entries[0].share * 0.75);
        pick = weightedPick(near, near.map(e => e.share), this.rng);
      } else {
        entries.sort((x, y) => (y.share - x.share) || ((y.n ? y.w / y.n : 0) - (x.n ? x.w / x.n : 0)));
        pick = entries[0];
      }
      chosen.push(pick.o.choice);
      taken = pick.o;
    }
    return chosen.join(', ');
  }
  randomChoice() {
    const b = determinize(this.view, this.spreads, null, this.rng);
    if (!b) return 'default';
    const node = new Node();
    initNode(node, b, this.view.request);
    const p = pickSide(node, 0, this.rng, 'random');
    return p ? choiceString(node, 0, p) : 'default';
  }
  searchChoice(req) {
    const results = [];
    const ms = req.teamPreview ? this.previewMs : this.ms;
    const isJoint = this.policy === 'foulplay_joint';
    for (let i = 0; i < this.samples; i++) {
      let b;
      try { b = determinize(this.view, this.spreads, this.affinities, this.rng); } catch (e) { b = null; }
      if (!b || !b.sides[0].activeRequest || b.sides[0].activeRequest.wait) { this.stats.failedSamples++; continue; }
      if (i === 0) {   // a forced choice (one switch-in left, a locked move) needs no search
        const only = new Node();
        initNode(only, b, req);
        if (only.slots[0] && only.slots[0].every(list => list.length === 1)) return choiceString(only, 0, only.slots[0].map(() => 0));
      }
      const budget = {ms, iterations: this.iterations};
      let r;
      if (this.policy === 'montecarlo') r = flatMonteCarlo(b, budget, this.rng, req);
      else if (this.policy === 'uct_rollout') r = mcts(b, budget, this.rng, req, 12, {leafMode: 'rollout'});
      else if (this.policy === 'foulplay_doubles') r = mcts(b, budget, this.rng, req, 12, {leafMode: 'doubles'});
      else if (this.policy === 'foulplay_hybrid') r = mcts(b, budget, this.rng, req, 12, {leafMode: 'hybrid'});
      else if (isJoint) r = mctsJoint(b, budget, this.rng, req);
      else r = mcts(b, budget, this.rng, req, 12, {leafMode: 'singles'});
      this.stats.iterations += r.iterations;
      results.push(r);
    }
    if (!results.length) { this.stats.fallbacks++; return 'default'; }
    if (req.teamPreview) {
      const names = this.view.foeSheet.map(setName);
      this.affinities = isJoint ? previewAffinitiesJoint(results, names) : previewAffinities(results, names);
    }
    if (isJoint) return combineJoint(results, this.rng, 'argmax');
    const rule = this.policy === 'montecarlo' ? 'mean'
      : (this.policy === 'foulplay' ? 'foulplay' : 'argmax');
    const chosen = combine(results, this.rng, rule);
    return chosen.map(e => e.o.choice).join(', ');
  }
}

module.exports = {Agent, View, Spreads, determinize, mcts, mctsJoint, flatMonteCarlo, evaluate, evaluateDoubles,
  slotOptions, compatible, PREVIEW, combine, mulberry, FORMAT, RT};
