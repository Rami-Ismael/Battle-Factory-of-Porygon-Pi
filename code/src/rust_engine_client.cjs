// Persistent JSON-lines client for the Rust VGC Regulation M-B doubles search engine
// (code/rust/poke-engine-doubles). Keeps regmb-dex.json resident in RAM (<0.2 ms IPC per search).
'use strict';
const path = require('path');
const {spawn} = require('child_process');
const readline = require('readline');

const CODE_DIR = path.resolve(__dirname, '..');
const BIN = process.env.RUST_ENGINE_BIN ||
  path.join(CODE_DIR, 'rust/poke-engine-doubles/target/release/poke-engine-doubles');
const DEX = process.env.REGMB_DEX || path.join(CODE_DIR, 'data/regmb-dex.json');

function serializeBattle(b) {
  return {
    sides: b.sides.map(sd => ({
      mons: sd.pokemon.map(p => ({
        species: p.species.id,
        maxhp: p.maxhp,
        hp: p.hp,
        stats: [
          p.maxhp,
          p.storedStats.atk,
          p.storedStats.def,
          p.storedStats.spa,
          p.storedStats.spd,
          p.storedStats.spe,
        ],
        boosts: [
          p.boosts.atk || 0,
          p.boosts.def || 0,
          p.boosts.spa || 0,
          p.boosts.spd || 0,
          p.boosts.spe || 0,
          p.boosts.accuracy || 0,
        ],
        ability: p.ability || '',
        item: p.item || '',
        status: p.status || '',
        moves: p.moveSlots.map(m => m.id),
        pp: p.moveSlots.map(m => m.pp),
        disabled: p.moveSlots.map(m => !!m.disabled),
        volatiles: Object.keys(p.volatiles || {}),
        stall_counter: p.volatiles && p.volatiles.stall
          ? Math.max(0, Math.round(Math.log(p.volatiles.stall.counter || 1) / Math.log(3)))
          : 0,
        active_move_actions: p.activeMoveActions || 0,
        last_move_idx: p.lastMove ? p.moveSlots.findIndex(m => m.id === p.lastMove.id) : -1,
        can_mega: !!p.canMegaEvo,
        fainted: !!p.fainted || p.hp <= 0,
        trapped: !!p.trapped,
      })),
      active: sd.active.map(p => (p && !p.fainted && p.hp > 0 ? sd.pokemon.indexOf(p) : -1)),
      mega_used: !sd.pokemon.some(p => p.canMegaEvo) &&
        sd.pokemon.some(p => p.species.forme && p.species.forme.startsWith('Mega')),
      force_switch: sd.activeRequest && sd.activeRequest.forceSwitch
        ? sd.activeRequest.forceSwitch.map(Boolean)
        : [false, false],
      reflect: sd.sideConditions.reflect ? (sd.sideConditions.reflect.duration || 1) : 0,
      light_screen: sd.sideConditions.lightscreen ? (sd.sideConditions.lightscreen.duration || 1) : 0,
      aurora_veil: sd.sideConditions.auroraveil ? (sd.sideConditions.auroraveil.duration || 1) : 0,
      tailwind: sd.sideConditions.tailwind ? (sd.sideConditions.tailwind.duration || 1) : 0,
      safeguard: sd.sideConditions.safeguard ? (sd.sideConditions.safeguard.duration || 1) : 0,
      mist: sd.sideConditions.mist ? (sd.sideConditions.mist.duration || 1) : 0,
      stealth_rock: !!sd.sideConditions.stealthrock,
      spikes: sd.sideConditions.spikes ? (sd.sideConditions.spikes.layers || 1) : 0,
      toxic_spikes: sd.sideConditions.toxicspikes ? (sd.sideConditions.toxicspikes.layers || 1) : 0,
      sticky_web: !!sd.sideConditions.stickyweb,
    })),
    weather: b.field.weather || '',
    weather_turns: (b.field.weatherState && b.field.weatherState.duration) || 0,
    terrain: b.field.terrain || '',
    terrain_turns: (b.field.terrainState && b.field.terrainState.duration) || 0,
    trick_room: (b.field.pseudoWeather.trickroom && b.field.pseudoWeather.trickroom.duration) || 0,
    gravity: (b.field.pseudoWeather.gravity && b.field.pseudoWeather.gravity.duration) || 0,
    turn: b.turn,
  };
}

class RustEngineClient {
  constructor(bin = BIN, dex = DEX) {
    this.proc = spawn(bin, [dex], {stdio: ['pipe', 'pipe', 'pipe']});
    this.rl = readline.createInterface({input: this.proc.stdout});
    this.queue = [];
    this.ready = new Promise((resolve, reject) => {
      this._resolveReady = resolve;
      this._rejectReady = reject;
    });
    this._gotReady = false;
    this.rl.on('line', line => {
      if (!line.trim()) return;
      let msg;
      try { msg = JSON.parse(line); } catch (e) { return; }
      if (!this._gotReady && msg.kind === 'ready') {
        this._gotReady = true;
        this._resolveReady(msg);
        return;
      }
      const next = this.queue.shift();
      if (next) {
        if (msg.error) next.reject(new Error(msg.error));
        else next.resolve(msg);
      }
    });
    this.proc.on('exit', code => {
      const err = new Error(`poke-engine-doubles exited with code ${code}`);
      if (!this._gotReady) this._rejectReady(err);
      while (this.queue.length) this.queue.shift().reject(err);
    });
  }

  async request(payload) {
    await this.ready;
    return new Promise((resolve, reject) => {
      this.queue.push({resolve, reject});
      this.proc.stdin.write(JSON.stringify(payload) + '\n');
    });
  }

  close() {
    try { this.proc.stdin.end(); } catch (e) { /* ignore */ }
    try { this.proc.kill(); } catch (e) { /* ignore */ }
  }
}

let shared = null;
function getRustClient() {
  if (!shared) shared = new RustEngineClient();
  return shared;
}
function closeRustClient() {
  if (shared) {
    shared.close();
    shared = null;
  }
}

module.exports = {RustEngineClient, getRustClient, closeRustClient, serializeBattle, BIN, DEX};
