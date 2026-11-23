mod dex;
mod search;
mod sim;
mod state;

use dex::*;
use search::*;
use serde::{Deserialize, Serialize};
use sim::*;
use state::*;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Deserialize)]
struct WireMon {
    species: String,
    #[serde(rename = "maxhp")]
    max_hp: u16,
    hp: u16,
    stats: [u16; 6], // hp, atk, def, spa, spd, spe
    #[serde(default)]
    boosts: [i8; 6],
    ability: String,
    item: String,
    #[serde(default)]
    status: String,
    moves: Vec<String>,
    #[serde(default)]
    pp: Vec<u8>,
    #[serde(default)]
    disabled: Vec<bool>,
    #[serde(default)]
    volatiles: Vec<String>,
    #[serde(default)]
    stall_counter: u8,
    #[serde(default)]
    active_move_actions: u8,
    #[serde(default)]
    last_move_idx: i8,
    #[serde(default)]
    can_mega: bool,
    #[serde(default)]
    fainted: bool,
    #[serde(default)]
    trapped: bool,
}

#[derive(Deserialize)]
struct WireSide {
    mons: Vec<WireMon>,
    active: [i8; 2],
    #[serde(default)]
    mega_used: bool,
    #[serde(default)]
    force_switch: [bool; 2],
    #[serde(default)]
    reflect: u8,
    #[serde(default)]
    light_screen: u8,
    #[serde(default)]
    aurora_veil: u8,
    #[serde(default)]
    tailwind: u8,
    #[serde(default)]
    safeguard: u8,
    #[serde(default)]
    mist: u8,
    #[serde(default)]
    stealth_rock: bool,
    #[serde(default)]
    spikes: u8,
    #[serde(default)]
    toxic_spikes: u8,
    #[serde(default)]
    sticky_web: bool,
}

#[derive(Deserialize)]
struct WireState {
    sides: [WireSide; 2],
    #[serde(default)]
    weather: String,
    #[serde(default)]
    weather_turns: u8,
    #[serde(default)]
    terrain: String,
    #[serde(default)]
    terrain_turns: u8,
    #[serde(default)]
    trick_room: u8,
    #[serde(default)]
    gravity: u8,
    #[serde(default)]
    turn: u16,
}

#[derive(Deserialize)]
struct Request {
    cmd: String,
    #[serde(default)]
    policy: String,
    #[serde(default)]
    ms: u64,
    #[serde(default)]
    iterations: u32,
    #[serde(default = "default_seed")]
    seed: u64,
    state: Option<WireState>,
    teams: Option<[Vec<WireMon>; 2]>,
    #[serde(rename = "myOptions")]
    my_options: Option<[Vec<String>; 2]>,
    p1: Option<String>,
    p2: Option<String>,
}

fn default_seed() -> u64 {
    1
}

#[derive(Serialize)]
struct OptOut {
    choice: String,
    share: f32,
    n: u32,
    w: f32,
}

#[derive(Serialize)]
struct SearchOut {
    choice: String,
    iterations: u32,
    slots: [Vec<OptOut>; 2],
    joint: Vec<OptOut>,
}

#[derive(Serialize)]
struct StepOut {
    p1_hp: Vec<u16>,
    p2_hp: Vec<u16>,
    p1_status: Vec<u8>,
    p2_status: Vec<u8>,
    turn: u16,
    terminal: Option<f32>,
}

#[derive(Serialize)]
struct PreviewOut {
    choice: String,
    iterations: u32,
    options: Vec<OptOut>,
    foe_affinities: [f32; 6],
}

fn mon_from_wire(wm: &WireMon, dex: &Dex) -> Mon {
    let sp_id = dex.species_id(&wm.species);
    let sp = &dex.species[sp_id as usize];
    let item_id = to_id(&wm.item);
    let is_mega_stone = dex.mega_stones.contains_key(&item_id);
    let mega_target = *dex.mega_map.get(&(sp_id, item_id.clone())).unwrap_or(&0);
    let mut m = Mon {
        species_id: sp_id,
        mega_target,
        max_hp: wm.max_hp.max(1),
        hp: wm.hp,
        stats: wm.stats,
        boosts: wm.boosts,
        types: sp.types,
        ability: Ability::from_id(&to_id(&wm.ability)),
        item: Item::from_id(&item_id, is_mega_stone),
        status: parse_status(&wm.status),
        sleep_turns: if parse_status(&wm.status) == STATUS_SLP { 2 } else { 0 },
        toxic_counter: 0,
        moves: [0; 4],
        pp: [16; 4],
        disabled: [false; 4],
        move_count: wm.moves.len().min(4) as u8,
        volatiles: 0,
        sub_hp: 0,
        stall_counter: wm.stall_counter,
        active_move_actions: wm.active_move_actions,
        last_move_idx: wm.last_move_idx,
        locked_move_idx: -1,
        taunt_turns: 0,
        encore_turns: 0,
        confuse_turns: 0,
        perish_turns: 0,
        yawn_turns: 0,
        fainted: wm.fainted || wm.hp == 0,
        can_mega: wm.can_mega && mega_target > 0,
        is_trapped: wm.trapped,
    };
    for (k, mv) in wm.moves.iter().take(4).enumerate() {
        m.moves[k] = dex.move_id(mv);
        if let Some(&pp) = wm.pp.get(k) {
            m.pp[k] = pp;
        }
        if let Some(&dis) = wm.disabled.get(k) {
            m.disabled[k] = dis;
        }
    }
    for v in &wm.volatiles {
        match v.as_str() {
            "substitute" => {
                m.volatiles |= V_SUBSTITUTE;
                m.sub_hp = (m.max_hp / 4).max(1);
            }
            "confusion" => { m.volatiles |= V_CONFUSION; m.confuse_turns = 2; }
            "taunt" => { m.volatiles |= V_TAUNT; m.taunt_turns = 2; }
            "encore" => { m.volatiles |= V_ENCORE; m.encore_turns = 2; m.locked_move_idx = m.last_move_idx; }
            "yawn" => { m.volatiles |= V_YAWN; m.yawn_turns = 1; }
            "leechseed" => m.volatiles |= V_LEECHSEED,
            "perishsong" => { m.volatiles |= V_PERISH; m.perish_turns = 2; }
            "choicelock" => { m.volatiles |= V_CHOICELOCK; m.locked_move_idx = m.last_move_idx; }
            "focusenergy" => m.volatiles |= V_FOCUS_ENERGY,
            _ => {}
        }
    }
    m
}

fn from_wire(w: &WireState, dex: &Dex) -> BattleState {
    let mut st = BattleState {
        sides: [Side::default(), Side::default()],
        weather: parse_weather(&w.weather),
        weather_turns: w.weather_turns,
        terrain: parse_terrain(&w.terrain),
        terrain_turns: w.terrain_turns,
        trick_room: w.trick_room,
        gravity: w.gravity,
        turn: w.turn,
    };
    for s in 0..2 {
        let ws = &w.sides[s];
        let sd = &mut st.sides[s];
        sd.mon_count = ws.mons.len().min(4) as u8;
        sd.active = ws.active;
        sd.mega_used = ws.mega_used;
        sd.force_switch = ws.force_switch;
        sd.reflect = ws.reflect;
        sd.light_screen = ws.light_screen;
        sd.aurora_veil = ws.aurora_veil;
        sd.tailwind = ws.tailwind;
        sd.safeguard = ws.safeguard;
        sd.mist = ws.mist;
        sd.stealth_rock = ws.stealth_rock;
        sd.spikes = ws.spikes;
        sd.toxic_spikes = ws.toxic_spikes;
        sd.sticky_web = ws.sticky_web;
        for (i, wm) in ws.mons.iter().take(4).enumerate() {
            sd.mons[i] = mon_from_wire(wm, dex);
        }
    }
    st
}

fn parse_side_choice(s: &str) -> [SlotAction; 2] {
    let parts: Vec<&str> = s.split(',').map(|x| x.trim()).collect();
    let a0 = parts.first().map(|x| parse_slot_action(x)).unwrap_or(SlotAction::Pass);
    let a1 = parts.get(1).map(|x| parse_slot_action(x)).unwrap_or(SlotAction::Pass);
    [a0, a1]
}

fn main() {
    let dex_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("code/data/regmb-dex.json"));
    let dex = Dex::load(&dex_path).expect("failed to load regmb-dex.json");

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    writeln!(stdout, "{{\"kind\":\"ready\",\"species\":{},\"moves\":{}}}", dex.species.len(), dex.moves.len()).unwrap();
    stdout.flush().unwrap();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break; };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                writeln!(stdout, "{{\"error\":{:?}}}", format!("bad json: {e}")).unwrap();
                stdout.flush().unwrap();
                continue;
            }
        };
        if req.cmd == "preview" {
            let Some(raw_teams) = req.teams else {
                writeln!(stdout, "{{\"error\":\"missing teams\"}}").unwrap();
                stdout.flush().unwrap();
                continue;
            };
            let mut teams = [[Mon::default(); 6]; 2];
            for s in 0..2 {
                for (i, wm) in raw_teams[s].iter().take(6).enumerate() {
                    teams[s][i] = mon_from_wire(wm, &dex);
                }
            }
            let mode = SearchMode::from_str(&req.policy);
            let res = run_preview_search(&teams, &dex, mode, req.ms.max(1), req.iterations, req.seed);
            let out = PreviewOut {
                choice: res.choice,
                iterations: res.iterations,
                options: res
                    .options
                    .into_iter()
                    .take(20)
                    .map(|(choice, share, n, w)| OptOut { choice, share, n, w })
                    .collect(),
                foe_affinities: res.foe_affinities,
            };
            writeln!(stdout, "{}", serde_json::to_string(&out).unwrap()).unwrap();
            stdout.flush().unwrap();
            continue;
        }
        let Some(ws) = req.state else {
            writeln!(stdout, "{{\"error\":\"missing state\"}}").unwrap();
            stdout.flush().unwrap();
            continue;
        };
        let mut st = from_wire(&ws, &dex);
        match req.cmd.as_str() {
            "search" => {
                let mode = SearchMode::from_str(&req.policy);
                let ov = req.my_options.map(|[v0, v1]| {
                    [
                        v0.iter().map(|s| parse_slot_action(s)).collect(),
                        v1.iter().map(|s| parse_slot_action(s)).collect(),
                    ]
                });
                let res = run_search(&st, &dex, mode, req.ms.max(1), req.iterations, req.seed, ov);
                let map_v = |v: Vec<(String, f32, u32, f32)>| {
                    v.into_iter()
                        .map(|(choice, share, n, w)| OptOut { choice, share, n, w })
                        .collect()
                };
                let out = SearchOut {
                    choice: res.choice,
                    iterations: res.iterations,
                    slots: [map_v(res.slot_shares[0].clone()), map_v(res.slot_shares[1].clone())],
                    joint: map_v(res.joint_shares),
                };
                writeln!(stdout, "{}", serde_json::to_string(&out).unwrap()).unwrap();
                stdout.flush().unwrap();
            }
            "step" => {
                let a0 = parse_side_choice(req.p1.as_deref().unwrap_or("pass, pass"));
                let a1 = parse_side_choice(req.p2.as_deref().unwrap_or("pass, pass"));
                let mut rng = Rng::new(req.seed);
                step(&mut st, &dex, [a0, a1], &mut rng);
                let out = StepOut {
                    p1_hp: (0..st.sides[0].mon_count as usize).map(|i| st.sides[0].mons[i].hp).collect(),
                    p2_hp: (0..st.sides[1].mon_count as usize).map(|i| st.sides[1].mons[i].hp).collect(),
                    p1_status: (0..st.sides[0].mon_count as usize).map(|i| st.sides[0].mons[i].status).collect(),
                    p2_status: (0..st.sides[1].mon_count as usize).map(|i| st.sides[1].mons[i].status).collect(),
                    turn: st.turn,
                    terminal: st.terminal(),
                };
                writeln!(stdout, "{}", serde_json::to_string(&out).unwrap()).unwrap();
                stdout.flush().unwrap();
            }
            "bench" => {
                let ms = req.ms.max(200);
                let r_mcts = run_search(&st, &dex, SearchMode::FoulPlayDoubles, ms, 0, 1, None);
                let r_joint = run_search(&st, &dex, SearchMode::FoulPlayJoint, ms, 0, 2, None);
                let r_uct = run_search(&st, &dex, SearchMode::UctRollout, ms, 0, 3, None);
                let r_mc = run_search(&st, &dex, SearchMode::FlatMonteCarlo, ms, 0, 4, None);
                let scale = 1000.0 / (ms as f64);
                writeln!(
                    stdout,
                    "{{\"mcts_per_sec\":{:.0},\"joint_per_sec\":{:.0},\"uct_rollout_per_sec\":{:.0},\"flat_mc_per_sec\":{:.0}}}",
                    r_mcts.iterations as f64 * scale,
                    r_joint.iterations as f64 * scale,
                    r_uct.iterations as f64 * scale,
                    r_mc.iterations as f64 * scale,
                ).unwrap();
                stdout.flush().unwrap();
            }
            _ => {
                writeln!(stdout, "{{\"error\":\"unknown cmd\"}}").unwrap();
                stdout.flush().unwrap();
            }
        }
    }
}
