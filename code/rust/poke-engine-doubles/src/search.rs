use crate::dex::*;
use crate::sim::*;
use crate::state::*;
use std::collections::HashMap;
use std::time::Instant;

const BOOST_TABLE: [f32; 13] = [
    -3.3, -3.15, -3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.15, 3.3,
];

#[inline]
fn boost_w(b: i8) -> f32 {
    BOOST_TABLE[(b.clamp(-6, 6) + 6) as usize]
}

/// Exact poke-engine genx/evaluate.rs static evaluation extended to both active slots.
pub fn evaluate_singles(state: &BattleState, dex: &Dex) -> f32 {
    eval_side_singles(state, dex, 0) - eval_side_singles(state, dex, 1)
}

fn eval_side_singles(state: &BattleState, dex: &Dex, s: usize) -> f32 {
    let side = &state.sides[s];
    let gravity = state.gravity > 0;
    let mut score = 0.0f32;
    for i in 0..side.mon_count as usize {
        let p = &side.mons[i];
        if !p.alive() {
            continue;
        }
        let mut ps = 100.0 * (p.hp as f32) / (p.max_hp.max(1) as f32);
        match p.status {
            STATUS_BRN => {
                if matches!(p.ability, Ability::Guts | Ability::MarvelScale | Ability::QuickFeet) {
                    ps += 50.0;
                } else {
                    let mut phys = 0.0f32;
                    for k in 0..p.move_count as usize {
                        if dex.moves[p.moves[k] as usize].category == Category::Physical {
                            phys += 1.0;
                        }
                    }
                    if p.stats[3] > p.stats[1] {
                        phys *= 0.5;
                    }
                    ps -= 25.0 * phys;
                }
            }
            STATUS_PSN | STATUS_TOX => {
                if p.ability == Ability::PoisonHeal {
                    ps += 15.0;
                } else if matches!(
                    p.ability,
                    Ability::Guts
                        | Ability::MarvelScale
                        | Ability::QuickFeet
                        | Ability::ToxicBoost
                        | Ability::MagicGuard
                ) {
                    ps += 10.0;
                } else {
                    ps -= if p.status == STATUS_TOX { 30.0 } else { 10.0 };
                }
            }
            STATUS_FRZ => ps -= 40.0,
            STATUS_SLP | STATUS_PAR => ps -= 25.0,
            _ => {}
        }
        if p.item != Item::None {
            ps += 10.0;
        }
        score += ps.max(0.0) + 30.0;
        if p.is_grounded(gravity) && p.item != Item::HeavyDutyBoots && p.ability != Ability::MagicGuard {
            score -= 7.0 * (side.spikes as f32);
            score -= 7.0 * (side.toxic_spikes as f32);
            if side.sticky_web {
                score -= 25.0;
            }
        }
        if side.stealth_rock && p.item != Item::HeavyDutyBoots {
            score -= 10.0;
        }
        if side.is_active_idx(i) {
            if (p.volatiles & V_LEECHSEED) != 0 { score -= 30.0; }
            if (p.volatiles & V_SUBSTITUTE) != 0 { score += 75.0; }
            if (p.volatiles & V_CONFUSION) != 0 { score -= 20.0; }
            score += boost_w(p.boosts[0]) * 30.0;
            score += boost_w(p.boosts[1]) * 15.0;
            score += boost_w(p.boosts[2]) * 30.0;
            score += boost_w(p.boosts[3]) * 15.0;
            score += boost_w(p.boosts[4]) * 30.0;
        }
    }
    score += 20.0 * (side.reflect as f32);
    score += 20.0 * (side.light_screen as f32);
    score += 40.0 * (side.aurora_veil as f32);
    score += 5.0 * (side.safeguard as f32);
    score += 7.0 * (side.tailwind as f32);
    score
}

/// VGC 4v4 doubles static evaluator: higher KO value, Trick Room / Tailwind speed-tier awareness,
/// Fake Out pressure, doubles volatiles, down-weighted entry hazards.
pub fn evaluate_doubles(state: &BattleState, dex: &Dex) -> f32 {
    eval_side_doubles(state, dex, 0) - eval_side_doubles(state, dex, 1)
}

fn eval_side_doubles(state: &BattleState, dex: &Dex, s: usize) -> f32 {
    let side = &state.sides[s];
    let foe = &state.sides[1 - s];
    let gravity = state.gravity > 0;
    let psy_t = state.terrain == TERRAIN_PSYCHIC;
    let mut score = 0.0f32;
    for i in 0..side.mon_count as usize {
        let p = &side.mons[i];
        if !p.alive() {
            continue;
        }
        let mut ps = 100.0 * (p.hp as f32) / (p.max_hp.max(1) as f32);
        match p.status {
            STATUS_BRN => {
                if matches!(p.ability, Ability::Guts | Ability::MarvelScale | Ability::QuickFeet) {
                    ps += 56.0;
                } else {
                    let mut phys = 0.0f32;
                    for k in 0..p.move_count as usize {
                        if dex.moves[p.moves[k] as usize].category == Category::Physical {
                            phys += 1.0;
                        }
                    }
                    if p.stats[3] > p.stats[1] {
                        phys *= 0.5;
                    }
                    ps -= 28.0 * phys;
                }
            }
            STATUS_PSN | STATUS_TOX => {
                if p.ability == Ability::PoisonHeal {
                    ps += 15.0;
                } else if matches!(
                    p.ability,
                    Ability::Guts
                        | Ability::MarvelScale
                        | Ability::QuickFeet
                        | Ability::ToxicBoost
                        | Ability::MagicGuard
                ) {
                    ps += 10.0;
                } else {
                    ps -= if p.status == STATUS_TOX { 25.0 } else { 10.0 };
                }
            }
            STATUS_FRZ => ps -= 45.0,
            STATUS_SLP => ps -= 35.0,
            STATUS_PAR => ps -= 28.0,
            _ => {}
        }
        if p.item != Item::None {
            ps += 8.0;
        }
        score += ps.max(0.0) + 65.0;
        if p.is_grounded(gravity) && p.item != Item::HeavyDutyBoots && p.ability != Ability::MagicGuard {
            score -= 2.0 * (side.spikes as f32 + side.toxic_spikes as f32);
            if side.sticky_web {
                score -= 12.0;
            }
        }
        if side.stealth_rock && p.item != Item::HeavyDutyBoots {
            score -= 3.0;
        }
        if side.is_active_idx(i) {
            if (p.volatiles & V_LEECHSEED) != 0 { score -= 25.0; }
            if (p.volatiles & V_SUBSTITUTE) != 0 { score += 65.0; }
            if (p.volatiles & V_CONFUSION) != 0 { score -= 18.0; }
            if (p.volatiles & V_ENCORE) != 0 { score -= 35.0; }
            if (p.volatiles & V_TAUNT) != 0 { score -= 18.0; }
            if (p.volatiles & V_YAWN) != 0 { score -= 22.0; }
            if (p.volatiles & V_PERISH) != 0 { score -= 25.0; }
            if (p.volatiles & V_DISABLE) != 0 { score -= 15.0; }
            score += boost_w(p.boosts[0]) * 28.0;
            score += boost_w(p.boosts[1]) * 14.0;
            score += boost_w(p.boosts[2]) * 28.0;
            score += boost_w(p.boosts[3]) * 14.0;
            score += boost_w(p.boosts[4]) * 28.0;
            if !psy_t && p.active_move_actions == 0 {
                for k in 0..p.move_count as usize {
                    if dex.moves[p.moves[k] as usize].is_fake_out {
                        score += 16.0;
                        break;
                    }
                }
            }
        }
    }
    score += 18.0 * (side.reflect as f32);
    score += 18.0 * (side.light_screen as f32);
    score += 36.0 * (side.aurora_veil as f32);
    score += 6.0 * (side.safeguard as f32);
    score += 24.0 * (side.tailwind as f32);

    // Speed-tier advantage between active slots (flips under Trick Room)
    let tr = state.trick_room > 0;
    for sl in 0..2 {
        let ai = side.active[sl];
        if ai < 0 || !side.mons[ai as usize].alive() { continue; }
        let sa = effective_speed(state, s, &side.mons[ai as usize]);
        for fsl in 0..2 {
            let fi = foe.active[fsl];
            if fi < 0 || !foe.mons[fi as usize].alive() { continue; }
            let sf = effective_speed(state, 1 - s, &foe.mons[fi as usize]);
            if sa != sf {
                let faster = (sa > sf) != tr;
                score += if faster { 10.0 } else { -10.0 };
            }
        }
    }
    score
}

#[inline]
fn sigmoid(x: f32, k: f32) -> f32 {
    1.0 / (1.0 + (-k * x).exp())
}

// ---------------------------------------------------------------- decoupled slot-UCB MCTS
#[derive(Clone)]
pub struct OptStat {
    pub action: SlotAction,
    pub n: u32,
    pub w: f32,
}

struct SlotNode {
    visits: u32,
    slots: [[Vec<OptStat>; 2]; 2],
    children: HashMap<(u64, u64), usize>,
}

impl SlotNode {
    fn new(state: &BattleState, dex: &Dex, root_p1_override: Option<[Vec<SlotAction>; 2]>) -> Self {
        let s0 = root_p1_override.unwrap_or_else(|| slot_options(state, dex, 0));
        let s1 = slot_options(state, dex, 1);
        let to_stats = |v: Vec<SlotAction>| {
            v.into_iter()
                .map(|a| OptStat { action: a, n: 0, w: 0.0 })
                .collect()
        };
        Self {
            visits: 0,
            slots: [
                [to_stats(s0[0].clone()), to_stats(s0[1].clone())],
                [to_stats(s1[0].clone()), to_stats(s1[1].clone())],
            ],
            children: HashMap::new(),
        }
    }
}

fn ucb_pick_slot(list: &[OptStat], parent_visits: u32, taken: Option<SlotAction>, rng: &mut Rng) -> usize {
    let mut best = 0usize;
    let mut best_s = f32::NEG_INFINITY;
    let mut unvisited = [0usize; 32];
    let mut u_len = 0usize;
    let ln_n = (parent_visits.max(1) as f32).ln();
    for (i, o) in list.iter().enumerate() {
        if let Some(t) = taken {
            if !SlotAction::compatible(o.action, t) {
                continue;
            }
        }
        if o.n == 0 {
            if u_len < 32 {
                unvisited[u_len] = i;
                u_len += 1;
            }
            continue;
        }
        let s = (o.w / o.n as f32) + (2.0 * ln_n / o.n as f32).sqrt();
        if s > best_s {
            best_s = s;
            best = i;
        }
    }
    if u_len > 0 {
        unvisited[rng.gen_range(u_len)]
    } else {
        best
    }
}

fn random_pick_slot(list: &[OptStat], taken: Option<SlotAction>, rng: &mut Rng) -> usize {
    let mut ok = [0usize; 32];
    let mut n = 0usize;
    for (i, o) in list.iter().enumerate() {
        if let Some(t) = taken {
            if !SlotAction::compatible(o.action, t) {
                continue;
            }
        }
        if n < 32 {
            ok[n] = i;
            n += 1;
        }
    }
    if n > 0 { ok[rng.gen_range(n)] } else { 0 }
}

fn random_playout(state: &mut BattleState, dex: &Dex, rng: &mut Rng, max_turns: u16) {
    let start_turn = state.turn;
    while state.terminal().is_none() && state.turn.saturating_sub(start_turn) < max_turns {
        let mut acts = [[SlotAction::Pass; 2]; 2];
        for s in 0..2 {
            let opts = slot_options(state, dex, s);
            let a0 = opts[0][rng.gen_range(opts[0].len())];
            let mut cands = [SlotAction::Pass; 32];
            let mut cn = 0;
            for &o in &opts[1] {
                if SlotAction::compatible(a0, o) && cn < 32 {
                    cands[cn] = o;
                    cn += 1;
                }
            }
            let a1 = if cn > 0 { cands[rng.gen_range(cn)] } else { opts[1][0] };
            acts[s] = [a0, a1];
        }
        step(state, dex, acts, rng);
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SearchMode {
    FoulPlay,
    FoulPlayDet,
    FoulPlayDoubles,
    FoulPlayJoint,
    FoulPlayHybrid,
    UctRollout,
    FlatMonteCarlo,
}

impl SearchMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "foulplay" => Self::FoulPlay,
            "foulplay_det" => Self::FoulPlayDet,
            "foulplay_doubles" => Self::FoulPlayDoubles,
            "foulplay_joint" => Self::FoulPlayJoint,
            "foulplay_hybrid" => Self::FoulPlayHybrid,
            "uct_rollout" => Self::UctRollout,
            "montecarlo" => Self::FlatMonteCarlo,
            _ => Self::FoulPlayDoubles,
        }
    }
}

pub struct SearchResult {
    pub choice: String,
    pub iterations: u32,
    pub slot_shares: [Vec<(String, f32, u32, f32)>; 2],
    pub joint_shares: Vec<(String, f32, u32, f32)>,
}

pub fn run_search(
    root_state: &BattleState,
    dex: &Dex,
    mode: SearchMode,
    ms: u64,
    max_iters: u32,
    seed: u64,
    p1_override: Option<[Vec<SlotAction>; 2]>,
) -> SearchResult {
    let mut rng = Rng::new(seed);
    match mode {
        SearchMode::FlatMonteCarlo => run_flat_mc(root_state, dex, ms, max_iters, &mut rng, p1_override),
        SearchMode::FoulPlayJoint => run_joint_mcts(root_state, dex, ms, max_iters, &mut rng, p1_override),
        _ => run_slot_mcts(root_state, dex, mode, ms, max_iters, &mut rng, p1_override),
    }
}

fn run_slot_mcts(
    root_state: &BattleState,
    dex: &Dex,
    mode: SearchMode,
    ms: u64,
    max_iters: u32,
    rng: &mut Rng,
    p1_override: Option<[Vec<SlotAction>; 2]>,
) -> SearchResult {
    let use_singles = matches!(mode, SearchMode::FoulPlay | SearchMode::FoulPlayDet);
    let root_eval = if use_singles {
        evaluate_singles(root_state, dex)
    } else {
        evaluate_doubles(root_state, dex)
    };
    let mut nodes = Vec::with_capacity(4096);
    nodes.push(SlotNode::new(root_state, dex, p1_override));

    let t0 = Instant::now();
    let mut iterations = 0u32;
    let mut path: Vec<(usize, [[usize; 2]; 2])> = Vec::with_capacity(16);

    loop {
        if max_iters > 0 {
            if iterations >= max_iters {
                break;
            }
        } else if iterations & 63 == 0 && t0.elapsed().as_millis() as u64 >= ms {
            break;
        }
        let mut st = *root_state;
        let mut node_idx = 0usize;
        let mut score = None;
        path.clear();

        for _depth in 0..12 {
            let mut picks = [[0usize; 2]; 2];
            let mut acts = [[SlotAction::Pass; 2]; 2];
            {
                let nd = &nodes[node_idx];
                for s in 0..2 {
                    let p0 = ucb_pick_slot(&nd.slots[s][0], nd.visits, None, rng);
                    let a0 = nd.slots[s][0][p0].action;
                    let p1 = ucb_pick_slot(&nd.slots[s][1], nd.visits, Some(a0), rng);
                    let a1 = nd.slots[s][1][p1].action;
                    picks[s] = [p0, p1];
                    acts[s] = [a0, a1];
                }
            }
            path.push((node_idx, picks));
            step(&mut st, dex, acts, rng);
            if let Some(term) = st.terminal() {
                score = Some(term);
                break;
            }
            let pk_key = ((picks[0][0] as u64) << 24)
                | ((picks[0][1] as u64) << 16)
                | ((picks[1][0] as u64) << 8)
                | (picks[1][1] as u64);
            let child_key = (pk_key, st.signature());
            if let Some(&ch) = nodes[node_idx].children.get(&child_key) {
                node_idx = ch;
            } else {
                let new_idx = nodes.len();
                nodes.push(SlotNode::new(&st, dex, None));
                nodes[node_idx].children.insert(child_key, new_idx);
                score = Some(match mode {
                    SearchMode::UctRollout => {
                        random_playout(&mut st, dex, rng, 50);
                        st.terminal().unwrap_or_else(|| sigmoid(evaluate_doubles(&st, dex) - root_eval, 0.009))
                    }
                    SearchMode::FoulPlayHybrid => {
                        random_playout(&mut st, dex, rng, 2);
                        st.terminal().unwrap_or_else(|| sigmoid(evaluate_doubles(&st, dex) - root_eval, 0.009))
                    }
                    SearchMode::FoulPlay | SearchMode::FoulPlayDet => {
                        sigmoid(evaluate_singles(&st, dex) - root_eval, 0.0125)
                    }
                    _ => sigmoid(evaluate_doubles(&st, dex) - root_eval, 0.009),
                });
                break;
            }
        }
        let sc = score.unwrap_or_else(|| match mode {
            SearchMode::UctRollout => {
                random_playout(&mut st, dex, rng, 50);
                st.terminal().unwrap_or_else(|| sigmoid(evaluate_doubles(&st, dex) - root_eval, 0.009))
            }
            SearchMode::FoulPlay | SearchMode::FoulPlayDet => {
                sigmoid(evaluate_singles(&st, dex) - root_eval, 0.0125)
            }
            _ => sigmoid(evaluate_doubles(&st, dex) - root_eval, 0.009),
        });

        for &(ni, picks) in &path {
            let nd = &mut nodes[ni];
            nd.visits += 1;
            for s in 0..2 {
                let val = if s == 0 { sc } else { 1.0 - sc };
                for sl in 0..2 {
                    let o = &mut nd.slots[s][sl][picks[s][sl]];
                    o.n += 1;
                    o.w += val;
                }
            }
        }
        iterations += 1;
    }

    let root = &nodes[0];
    let mut slot_shares: [Vec<(String, f32, u32, f32)>; 2] = [Vec::new(), Vec::new()];
    let mut chosen = [SlotAction::Pass; 2];
    let mut taken = None;
    for sl in 0..2 {
        let total = root.slots[0][sl].iter().map(|o| o.n).sum::<u32>().max(1) as f32;
        let entries: Vec<&OptStat> = root.slots[0][sl].iter().collect();
        for o in &entries {
            slot_shares[sl].push((o.action.to_showdown_string(), o.n as f32 / total, o.n, o.w));
        }
        let mut compat: Vec<&OptStat> = entries
            .iter()
            .copied()
            .filter(|o| taken.map_or(true, |t| SlotAction::compatible(o.action, t)))
            .collect();
        if compat.is_empty() {
            compat = entries;
        }
        compat.sort_by(|a, b| {
            b.n.cmp(&a.n).then_with(|| {
                let ma = if a.n > 0 { a.w / a.n as f32 } else { 0.0 };
                let mb = if b.n > 0 { b.w / b.n as f32 } else { 0.0 };
                mb.total_cmp(&ma)
            })
        });
        let pick = if matches!(mode, SearchMode::FoulPlay) {
            let top = compat[0].n as f32;
            let near: Vec<&OptStat> = compat
                .iter()
                .copied()
                .filter(|o| (o.n as f32) >= top * 0.75)
                .collect();
            let sum: f32 = near.iter().map(|o| o.n as f32).sum();
            let mut r = rng.next_f32() * sum;
            let mut sel = near[0];
            for &o in &near {
                r -= o.n as f32;
                if r <= 0.0 {
                    sel = o;
                    break;
                }
            }
            sel
        } else {
            compat[0]
        };
        chosen[sl] = pick.action;
        taken = Some(pick.action);
    }

    SearchResult {
        choice: format!("{}, {}", chosen[0].to_showdown_string(), chosen[1].to_showdown_string()),
        iterations,
        slot_shares,
        joint_shares: Vec::new(),
    }
}

fn run_flat_mc(
    root_state: &BattleState,
    dex: &Dex,
    ms: u64,
    max_iters: u32,
    rng: &mut Rng,
    p1_override: Option<[Vec<SlotAction>; 2]>,
) -> SearchResult {
    let mut root = SlotNode::new(root_state, dex, p1_override);
    let t0 = Instant::now();
    let mut iterations = 0u32;
    loop {
        if max_iters > 0 {
            if iterations >= max_iters {
                break;
            }
        } else if iterations & 15 == 0 && t0.elapsed().as_millis() as u64 >= ms {
            break;
        }
        let mut st = *root_state;
        let mut picks = [[0usize; 2]; 2];
        let mut acts = [[SlotAction::Pass; 2]; 2];
        for s in 0..2 {
            let p0 = random_pick_slot(&root.slots[s][0], None, rng);
            let a0 = root.slots[s][0][p0].action;
            let p1 = random_pick_slot(&root.slots[s][1], Some(a0), rng);
            let a1 = root.slots[s][1][p1].action;
            picks[s] = [p0, p1];
            acts[s] = [a0, a1];
        }
        step(&mut st, dex, acts, rng);
        random_playout(&mut st, dex, rng, 50);
        let sc = st.terminal().unwrap_or_else(|| sigmoid(evaluate_doubles(&st, dex), 0.009));
        root.visits += 1;
        for s in 0..2 {
            let val = if s == 0 { sc } else { 1.0 - sc };
            for sl in 0..2 {
                let o = &mut root.slots[s][sl][picks[s][sl]];
                o.n += 1;
                o.w += val;
            }
        }
        iterations += 1;
    }

    let mut slot_shares: [Vec<(String, f32, u32, f32)>; 2] = [Vec::new(), Vec::new()];
    let mut chosen = [SlotAction::Pass; 2];
    let mut taken = None;
    for sl in 0..2 {
        let total = root.slots[0][sl].iter().map(|o| o.n).sum::<u32>().max(1) as f32;
        for o in &root.slots[0][sl] {
            slot_shares[sl].push((o.action.to_showdown_string(), o.n as f32 / total, o.n, o.w));
        }
        let mut compat: Vec<&OptStat> = root.slots[0][sl]
            .iter()
            .filter(|o| taken.map_or(true, |t| SlotAction::compatible(o.action, t)))
            .collect();
        if compat.is_empty() {
            compat = root.slots[0][sl].iter().collect();
        }
        compat.sort_by(|a, b| {
            let ma = if a.n > 0 { a.w / a.n as f32 } else { -1.0 };
            let mb = if b.n > 0 { b.w / b.n as f32 } else { -1.0 };
            mb.total_cmp(&ma)
        });
        chosen[sl] = compat[0].action;
        taken = Some(compat[0].action);
    }
    SearchResult {
        choice: format!("{}, {}", chosen[0].to_showdown_string(), chosen[1].to_showdown_string()),
        iterations,
        slot_shares,
        joint_shares: Vec::new(),
    }
}

// ---------------------------------------------------------------- joint-action progressive-widening PUCT
struct JointOpt {
    pair: [SlotAction; 2],
    prior: f32,
    n: u32,
    w: f32,
}

struct JointNode {
    visits: u32,
    sides: [Vec<JointOpt>; 2],
    children: HashMap<(u32, u64), usize>,
}

fn single_prior(state: &BattleState, dex: &Dex, s: usize, slot: usize, act: SlotAction) -> f32 {
    match act {
        SlotAction::Pass => 0.1,
        SlotAction::Switch(_) => 0.6,
        SlotAction::Move { move_idx, target, mega } => {
            let ai = state.sides[s].active[slot];
            if ai < 0 { return 0.1; }
            let me = &state.sides[s].mons[ai as usize];
            let mv = &dex.moves[me.moves[(move_idx as usize).min(3)] as usize];
            let mut sc = 1.0f32;
            if mega { sc += 0.65; }
            if mv.is_fake_out { sc += if me.active_move_actions == 0 { 1.4 } else { -0.9 }; }
            else if mv.is_protect { sc += if me.stall_counter > 0 { -0.8 } else { 0.7 }; }
            else if mv.is_tactical { sc += 0.75; }
            if mv.category != Category::Status {
                sc += ((mv.base_power.max(60) as f32) / 100.0).min(1.2);
                if me.has_type(mv.move_type) { sc += 0.25; }
                if target > 0 {
                    let fsl = (target as usize - 1).min(1);
                    let fi = state.sides[1 - s].active[fsl];
                    if fi >= 0 && state.sides[1 - s].mons[fi as usize].alive() {
                        let foe = &state.sides[1 - s].mons[fi as usize];
                        let eff = TYPE_CHART[mv.move_type as usize][foe.types[0] as usize]
                            * TYPE_CHART[mv.move_type as usize][foe.types[1] as usize];
                        if eff == 0.0 { sc -= 1.6; }
                        else if eff > 1.0 { sc += 0.5 * eff.log2(); }
                        else if eff < 1.0 { sc += 0.25 * eff.log2(); }
                    }
                }
            }
            sc.max(0.05)
        }
    }
}

fn build_joint_opts(
    state: &BattleState,
    dex: &Dex,
    s: usize,
    override_slots: Option<[Vec<SlotAction>; 2]>,
) -> Vec<JointOpt> {
    let slots = override_slots.unwrap_or_else(|| slot_options(state, dex, s));
    let p0s: Vec<f32> = slots[0].iter().map(|&a| single_prior(state, dex, s, 0, a)).collect();
    let p1s: Vec<f32> = slots[1].iter().map(|&a| single_prior(state, dex, s, 1, a)).collect();
    let mut raw: Vec<([SlotAction; 2], f32)> = Vec::with_capacity(slots[0].len() * slots[1].len());
    for (i, &a) in slots[0].iter().enumerate() {
        for (j, &b) in slots[1].iter().enumerate() {
            if !SlotAction::compatible(a, b) { continue; }
            let mut sc = p0s[i] + p1s[j];
            if a.switch_target().is_some() && b.switch_target().is_some() { sc -= 0.5; }
            raw.push(([a, b], sc));
        }
    }
    if raw.is_empty() {
        raw.push(([SlotAction::Pass, SlotAction::Pass], 1.0));
    }
    raw.sort_by(|x, y| y.1.total_cmp(&x.1));
    let max_s = raw[0].1;
    let exps: Vec<f32> = raw.iter().map(|r| ((r.1 - max_s) * 0.7).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-6);
    raw.into_iter()
        .zip(exps)
        .map(|((pair, _), e)| JointOpt { pair, prior: e / sum, n: 0, w: 0.0 })
        .collect()
}

fn pick_joint(list: &[JointOpt], visits: u32) -> usize {
    let limit = list.len().min((4.0 * (visits.max(1) as f32).sqrt()) as usize).max(10).min(list.len());
    let sqrt_n = (visits.max(1) as f32).sqrt();
    let ln_n = (visits.max(2) as f32).ln();
    let mut best = 0usize;
    let mut best_s = f32::NEG_INFINITY;
    for k in 0..limit {
        let o = &list[k];
        if o.n == 0 {
            return k;
        }
        let s = (o.w / o.n as f32) + 1.6 * o.prior * sqrt_n / (1.0 + o.n as f32) + 0.5 * (ln_n / o.n as f32).sqrt();
        if s > best_s {
            best_s = s;
            best = k;
        }
    }
    best
}

fn run_joint_mcts(
    root_state: &BattleState,
    dex: &Dex,
    ms: u64,
    max_iters: u32,
    rng: &mut Rng,
    p1_override: Option<[Vec<SlotAction>; 2]>,
) -> SearchResult {
    let root_eval = evaluate_doubles(root_state, dex);
    let mut nodes = Vec::with_capacity(4096);
    nodes.push(JointNode {
        visits: 0,
        sides: [
            build_joint_opts(root_state, dex, 0, p1_override),
            build_joint_opts(root_state, dex, 1, None),
        ],
        children: HashMap::new(),
    });
    let t0 = Instant::now();
    let mut iterations = 0u32;
    let mut path: Vec<(usize, [usize; 2])> = Vec::with_capacity(16);

    loop {
        if max_iters > 0 {
            if iterations >= max_iters {
                break;
            }
        } else if iterations & 63 == 0 && t0.elapsed().as_millis() as u64 >= ms {
            break;
        }
        let mut st = *root_state;
        let mut node_idx = 0usize;
        let mut score = None;
        path.clear();

        for _depth in 0..12 {
            let (p0, p1, acts) = {
                let nd = &nodes[node_idx];
                let p0 = pick_joint(&nd.sides[0], nd.visits);
                let p1 = pick_joint(&nd.sides[1], nd.visits);
                (p0, p1, [nd.sides[0][p0].pair, nd.sides[1][p1].pair])
            };
            path.push((node_idx, [p0, p1]));
            step(&mut st, dex, acts, rng);
            if let Some(term) = st.terminal() {
                score = Some(term);
                break;
            }
            let pk = ((p0 as u32) << 16) | (p1 as u32);
            let child_key = (pk, st.signature());
            if let Some(&ch) = nodes[node_idx].children.get(&child_key) {
                node_idx = ch;
            } else {
                let new_idx = nodes.len();
                nodes.push(JointNode {
                    visits: 0,
                    sides: [
                        build_joint_opts(&st, dex, 0, None),
                        build_joint_opts(&st, dex, 1, None),
                    ],
                    children: HashMap::new(),
                });
                nodes[node_idx].children.insert(child_key, new_idx);
                score = Some(sigmoid(evaluate_doubles(&st, dex) - root_eval, 0.009));
                break;
            }
        }
        let sc = score.unwrap_or_else(|| sigmoid(evaluate_doubles(&st, dex) - root_eval, 0.009));
        for &(ni, [p0, p1]) in &path {
            let nd = &mut nodes[ni];
            nd.visits += 1;
            nd.sides[0][p0].n += 1;
            nd.sides[0][p0].w += sc;
            nd.sides[1][p1].n += 1;
            nd.sides[1][p1].w += 1.0 - sc;
        }
        iterations += 1;
    }

    let root = &nodes[0];
    let total = root.sides[0].iter().map(|o| o.n).sum::<u32>().max(1) as f32;
    let mut joint_shares: Vec<(String, f32, u32, f32)> = root.sides[0]
        .iter()
        .filter(|o| o.n > 0)
        .map(|o| {
            (
                format!("{}, {}", o.pair[0].to_showdown_string(), o.pair[1].to_showdown_string()),
                o.n as f32 / total,
                o.n,
                o.w,
            )
        })
        .collect();
    joint_shares.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| (b.3 / b.2.max(1) as f32).total_cmp(&(a.3 / a.2.max(1) as f32))));
    let best = joint_shares
        .first()
        .map(|x| x.0.clone())
        .unwrap_or_else(|| "pass, pass".to_string());
    SearchResult {
        choice: best,
        iterations,
        slot_shares: [Vec::new(), Vec::new()],
        joint_shares,
    }
}

// ---------------------------------------------------------------- native Rust Team Preview (90 bring-4 options)
pub struct PreviewResult {
    pub choice: String,
    pub iterations: u32,
    pub options: Vec<(String, f32, u32, f32)>,
    pub foe_affinities: [f32; 6],
}

fn all_preview_90() -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(90);
    for a in 0u8..6 {
        for b in (a + 1)..6 {
            let mut rest = [0u8; 4];
            let mut rn = 0;
            for x in 0u8..6 {
                if x != a && x != b {
                    rest[rn] = x;
                    rn += 1;
                }
            }
            for c in 0..4 {
                for d in (c + 1)..4 {
                    out.push([a, b, rest[c], rest[d]]);
                }
            }
        }
    }
    out
}

fn build_preview_battle(teams: &[[Mon; 6]; 2], p1: [u8; 4], p2: [u8; 4]) -> BattleState {
    let mut st = BattleState::default();
    let picks = [p1, p2];
    for s in 0..2 {
        st.sides[s].mon_count = 4;
        st.sides[s].active = [0, 1];
        for i in 0..4 {
            st.sides[s].mons[i] = teams[s][picks[s][i] as usize];
        }
    }
    // Trigger lead switch-in abilities (Intimidate, weather, terrain, Protosynthesis/Quark Drive) in Speed order
    let mut leads = [(0usize, 0usize, 0u16); 4];
    let mut ln = 0;
    for s in 0..2 {
        for sl in 0..2 {
            leads[ln] = (s, sl, st.sides[s].mons[sl].stats[5]);
            ln += 1;
        }
    }
    leads.sort_by(|a, b| b.2.cmp(&a.2));
    for &(s, sl, _) in &leads {
        run_switch_in(&mut st, s, sl);
    }
    st
}

fn greedy_turn(state: &mut BattleState, dex: &Dex, rng: &mut Rng) {
    let mut acts = [[SlotAction::Pass; 2]; 2];
    for s in 0..2 {
        let slots = slot_options(state, dex, s);
        let mut best_a = slots[0][0];
        let mut best_sa = f32::NEG_INFINITY;
        for &a in &slots[0] {
            let sc = single_prior(state, dex, s, 0, a) + rng.next_f32() * 0.4;
            if sc > best_sa { best_sa = sc; best_a = a; }
        }
        let mut best_b = slots[1][0];
        let mut best_sb = f32::NEG_INFINITY;
        for &b in &slots[1] {
            if !SlotAction::compatible(best_a, b) { continue; }
            let sc = single_prior(state, dex, s, 1, b) + rng.next_f32() * 0.4;
            if sc > best_sb { best_sb = sc; best_b = b; }
        }
        acts[s] = [best_a, best_b];
    }
    step(state, dex, acts, rng);
}

pub fn run_preview_search(
    teams: &[[Mon; 6]; 2],
    dex: &Dex,
    mode: SearchMode,
    ms: u64,
    max_iters: u32,
    seed: u64,
) -> PreviewResult {
    let opts = all_preview_90();
    let n_opts = opts.len();
    let mut n = [[0u32; 90]; 2];
    let mut w = [[0.0f32; 90]; 2];
    let mut visits = 0u32;
    let mut rng = Rng::new(seed);
    let t0 = Instant::now();
    let is_flat = matches!(mode, SearchMode::FlatMonteCarlo);
    let is_rollout = matches!(mode, SearchMode::FlatMonteCarlo | SearchMode::UctRollout);
    let use_singles = matches!(mode, SearchMode::FoulPlay | SearchMode::FoulPlayDet);

    loop {
        if max_iters > 0 {
            if visits >= max_iters {
                break;
            }
        } else if visits & 31 == 0 && t0.elapsed().as_millis() as u64 >= ms {
            break;
        }
        let mut p = [0usize; 2];
        for s in 0..2 {
            if is_flat {
                p[s] = rng.gen_range(n_opts);
            } else {
                let ln_n = (visits.max(1) as f32).ln();
                let mut best = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                let mut unv = None;
                for k in 0..n_opts {
                    if n[s][k] == 0 {
                        unv = Some(k);
                        break;
                    }
                    let sc = (w[s][k] / n[s][k] as f32) + (1.414 * (ln_n / n[s][k] as f32).sqrt());
                    if sc > best_s { best_s = sc; best = k; }
                }
                p[s] = unv.unwrap_or(best);
            }
        }
        let mut st = build_preview_battle(teams, opts[p[0]], opts[p[1]]);
        let sc = if is_rollout {
            random_playout(&mut st, dex, &mut rng, 45);
            st.terminal().unwrap_or_else(|| sigmoid(evaluate_doubles(&st, dex), 0.009))
        } else {
            // 3 tactical turns to capture lead interactions (Fake Out, Tailwind, Trick Room, KOs) + static eval
            for _ in 0..3 {
                if st.terminal().is_some() { break; }
                greedy_turn(&mut st, dex, &mut rng);
            }
            st.terminal().unwrap_or_else(|| {
                if use_singles {
                    sigmoid(evaluate_singles(&st, dex), 0.0125)
                } else {
                    sigmoid(evaluate_doubles(&st, dex), 0.009)
                }
            })
        };
        visits += 1;
        n[0][p[0]] += 1;
        w[0][p[0]] += sc;
        n[1][p[1]] += 1;
        w[1][p[1]] += 1.0 - sc;
    }

    let total0 = visits.max(1) as f32;
    let mut foe_aff = [0.0f32; 6];
    for k in 0..n_opts {
        let sh = n[1][k] as f32 / total0;
        for &idx in &opts[k] {
            foe_aff[idx as usize] += sh;
        }
    }
    let mut options: Vec<(String, f32, u32, f32)> = (0..n_opts)
        .map(|k| {
            let o = opts[k];
            let ch = format!("team {}{}{}{}", o[0] + 1, o[1] + 1, o[2] + 1, o[3] + 1);
            (ch, n[0][k] as f32 / total0, n[0][k], w[0][k])
        })
        .collect();
    if is_flat {
        options.sort_by(|a, b| {
            let ma = if a.2 > 0 { a.3 / a.2 as f32 } else { -1.0 };
            let mb = if b.2 > 0 { b.3 / b.2 as f32 } else { -1.0 };
            mb.total_cmp(&ma)
        });
    } else {
        options.sort_by(|a, b| {
            b.2.cmp(&a.2).then_with(|| {
                let ma = if a.2 > 0 { a.3 / a.2 as f32 } else { 0.0 };
                let mb = if b.2 > 0 { b.3 / b.2 as f32 } else { 0.0 };
                mb.total_cmp(&ma)
            })
        });
    }
    let choice = if matches!(mode, SearchMode::FoulPlay) {
        let top = options[0].2 as f32;
        let near: Vec<&(String, f32, u32, f32)> = options.iter().filter(|o| o.2 as f32 >= top * 0.75).collect();
        let sum: f32 = near.iter().map(|o| o.2 as f32).sum();
        let mut r = rng.next_f32() * sum;
        let mut sel = &near[0].0;
        for o in &near {
            r -= o.2 as f32;
            if r <= 0.0 { sel = &o.0; break; }
        }
        sel.clone()
    } else {
        options[0].0.clone()
    };

    PreviewResult {
        choice,
        iterations: visits,
        options,
        foe_affinities: foe_aff,
    }
}
