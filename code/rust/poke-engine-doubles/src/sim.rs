use crate::dex::*;
use crate::state::*;

/// One slot's action in a doubles turn or force-switch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SlotAction {
    Pass,
    /// move_idx (0..3), target (+1/+2 = foe slot 0/1, -1/-2 = my slot 0/1, 0 = untargeted), mega
    Move {
        move_idx: u8,
        target: i8,
        mega: bool,
    },
    /// bench mon index in side.mons (0..3)
    Switch(u8),
}

impl SlotAction {
    pub fn to_showdown_string(self) -> String {
        match self {
            SlotAction::Pass => "pass".to_string(),
            SlotAction::Switch(idx) => format!("switch {}", idx + 1),
            SlotAction::Move {
                move_idx,
                target,
                mega,
            } => {
                let mut s = if target != 0 {
                    format!("move {} {}", move_idx + 1, target)
                } else {
                    format!("move {}", move_idx + 1)
                };
                if mega {
                    s.push_str(" mega");
                }
                s
            }
        }
    }

    pub fn switch_target(self) -> Option<u8> {
        match self {
            SlotAction::Switch(i) => Some(i),
            _ => None,
        }
    }

    pub fn is_mega(self) -> bool {
        matches!(self, SlotAction::Move { mega: true, .. })
    }

    pub fn compatible(a: Self, b: Self) -> bool {
        if let (Some(sa), Some(sb)) = (a.switch_target(), b.switch_target()) {
            if sa == sb {
                return false;
            }
        }
        if a.is_mega() && b.is_mega() {
            return false;
        }
        true
    }
}

pub fn parse_slot_action(s: &str) -> SlotAction {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() || parts[0] == "pass" {
        return SlotAction::Pass;
    }
    if parts[0] == "switch" {
        let idx = parts.get(1).and_then(|x| x.parse::<u8>().ok()).unwrap_or(1);
        return SlotAction::Switch(idx.saturating_sub(1));
    }
    if parts[0] == "move" {
        let m_idx = parts
            .get(1)
            .and_then(|x| x.parse::<u8>().ok())
            .unwrap_or(1)
            .saturating_sub(1);
        let mut target = 0i8;
        let mut mega = false;
        for &p in parts.iter().skip(2) {
            if p == "mega" {
                mega = true;
            } else if let Ok(t) = p.parse::<i8>() {
                target = t;
            }
        }
        return SlotAction::Move {
            move_idx: m_idx,
            target,
            mega,
        };
    }
    SlotAction::Pass
}

#[inline]
pub fn boost_mult(b: i8) -> f32 {
    if b >= 0 {
        (2 + b as i32) as f32 / 2.0
    } else {
        2.0 / (2 - b as i32) as f32
    }
}

#[inline]
pub fn effective_speed(state: &BattleState, side_idx: usize, mon: &Mon) -> u32 {
    let mut spe = mon.stats[5] as f32;
    spe *= boost_mult(mon.boosts[4]);
    if state.sides[side_idx].tailwind > 0 {
        spe *= 2.0;
    }
    if mon.status == STATUS_PAR {
        spe *= 0.5;
    }
    if mon.item == Item::ChoiceScarf {
        spe *= 1.5;
    } else if mon.item == Item::IronBall {
        spe *= 0.5;
    }
    if (mon.volatiles & V_PROTO_SPE) != 0 {
        spe *= 1.5;
    }
    match mon.ability {
        Ability::SwiftSwim if state.weather == WEATHER_RAIN => spe *= 2.0,
        Ability::Chlorophyll if state.weather == WEATHER_SUN => spe *= 2.0,
        Ability::SandRush if state.weather == WEATHER_SAND => spe *= 2.0,
        Ability::SlushRush if state.weather == WEATHER_SNOW => spe *= 2.0,
        Ability::SurgeSurfer if state.terrain == TERRAIN_ELECTRIC => spe *= 2.0,
        Ability::QuickFeet if mon.status != STATUS_NONE => spe *= 1.5,
        _ => {}
    }
    spe.max(1.0) as u32
}

pub fn slot_options(state: &BattleState, dex: &Dex, side_idx: usize) -> [Vec<SlotAction>; 2] {
    let side = &state.sides[side_idx];
    let foe = &state.sides[1 - side_idx];
    let (bench, bench_len) = side.bench_alive();

    if state.needs_force_switch(side_idx) {
        let mut out = [Vec::new(), Vec::new()];
        let need_count = (side.force_switch[0] as usize) + (side.force_switch[1] as usize);
        let short = bench_len < need_count;
        for slot in 0..2 {
            if !side.force_switch[slot] {
                out[slot].push(SlotAction::Pass);
            } else if bench_len == 0 {
                out[slot].push(SlotAction::Pass);
            } else {
                for b in 0..bench_len {
                    out[slot].push(SlotAction::Switch(bench[b]));
                }
                if short {
                    out[slot].push(SlotAction::Pass);
                }
            }
        }
        return out;
    }

    let mut out = [Vec::with_capacity(14), Vec::with_capacity(14)];
    let mut foes = [0i8; 2];
    let mut foe_n = 0usize;
    for fslot in 0..2 {
        let fi = foe.active[fslot];
        if fi >= 0 && foe.mons[fi as usize].alive() {
            foes[foe_n] = (fslot + 1) as i8;
            foe_n += 1;
        }
    }

    for slot in 0..2 {
        let mi = side.active[slot];
        if mi < 0 || !side.mons[mi as usize].alive() {
            out[slot].push(SlotAction::Pass);
            continue;
        }
        let me = &side.mons[mi as usize];
        let ally_idx = side.active[1 - slot];
        let ally_alive = ally_idx >= 0 && side.mons[ally_idx as usize].alive();
        let ally_loc = -((2 - slot) as i8);
        let self_loc = -((slot + 1) as i8);
        let can_mega = me.can_mega && !side.mega_used && me.mega_target > 0;
        let has_locked = (me.volatiles & (V_CHOICELOCK | V_ENCORE)) != 0 && me.locked_move_idx >= 0;

        for i in 0..me.move_count as usize {
            if me.disabled[i] || me.pp[i] == 0 {
                continue;
            }
            if has_locked && me.locked_move_idx as usize != i {
                continue;
            }
            let mv = &dex.moves[me.moves[i] as usize];
            if (me.volatiles & V_TAUNT) != 0 && mv.category == Category::Status {
                continue;
            }
            if me.item == Item::AssaultVest && mv.category == Category::Status {
                continue;
            }
            let mut targets = [0i8; 3];
            let mut tn = 0usize;
            match mv.target {
                TargetKind::Normal | TargetKind::AdjacentFoe | TargetKind::Any => {
                    if foe_n > 0 {
                        for k in 0..foe_n {
                            targets[tn] = foes[k];
                            tn += 1;
                        }
                    } else {
                        targets[0] = 1;
                        tn = 1;
                    }
                    if ally_alive && mv.is_ally_target_allowed {
                        targets[tn] = ally_loc;
                        tn += 1;
                    }
                }
                TargetKind::AdjacentAlly => {
                    if !ally_alive {
                        continue;
                    }
                    targets[0] = ally_loc;
                    tn = 1;
                }
                TargetKind::AdjacentAllyOrSelf => {
                    targets[0] = self_loc;
                    tn = 1;
                    if ally_alive {
                        targets[1] = ally_loc;
                        tn = 2;
                    }
                }
                _ => {
                    targets[0] = 0;
                    tn = 1;
                }
            }
            for k in 0..tn {
                out[slot].push(SlotAction::Move {
                    move_idx: i as u8,
                    target: targets[k],
                    mega: false,
                });
                if can_mega {
                    out[slot].push(SlotAction::Move {
                        move_idx: i as u8,
                        target: targets[k],
                        mega: true,
                    });
                }
            }
        }
        if !me.is_trapped {
            for b in 0..bench_len {
                out[slot].push(SlotAction::Switch(bench[b]));
            }
        }
        if out[slot].is_empty() {
            out[slot].push(SlotAction::Move {
                move_idx: 0,
                target: if foe_n > 0 { foes[0] } else { 1 },
                mega: false,
            });
        }
    }
    out
}

fn apply_boost(mon: &mut Mon, stat: usize, mut delta: i8, is_self: bool) {
    if delta == 0 || !mon.alive() {
        return;
    }
    if mon.ability == Ability::Contrary {
        delta = -delta;
    }
    if delta < 0 && !is_self {
        if mon.item == Item::ClearAmulet
            || matches!(
                mon.ability,
                Ability::ClearBody | Ability::WhiteSmoke | Ability::FullMetalBody
            )
        {
            return;
        }
    }
    let old = mon.boosts[stat];
    mon.boosts[stat] = (old + delta).clamp(-6, 6);
    if !is_self && mon.boosts[stat] < old {
        if mon.ability == Ability::Defiant {
            mon.boosts[0] = (mon.boosts[0] + 2).min(6);
        } else if mon.ability == Ability::Competitive {
            mon.boosts[2] = (mon.boosts[2] + 2).min(6);
        }
        if mon.item == Item::WhiteHerb {
            for s in 0..6 {
                if mon.boosts[s] < 0 {
                    mon.boosts[s] = 0;
                }
            }
            mon.item = Item::None;
        }
    }
}

fn check_proto_quark(state: &mut BattleState, side_idx: usize, mon_idx: usize) {
    let mon = &mut state.sides[side_idx].mons[mon_idx];
    if !mon.alive() {
        return;
    }
    let active = match mon.ability {
        Ability::Protosynthesis => state.weather == WEATHER_SUN || mon.item == Item::BoosterEnergy,
        Ability::QuarkDrive => {
            state.terrain == TERRAIN_ELECTRIC || mon.item == Item::BoosterEnergy
        }
        _ => return,
    };
    if !active {
        return;
    }
    if (mon.volatiles & (V_PROTO_ATK | V_PROTO_DEF | V_PROTO_SPA | V_PROTO_SPD | V_PROTO_SPE)) != 0
    {
        return;
    }
    let weather_or_terrain = (mon.ability == Ability::Protosynthesis
        && state.weather == WEATHER_SUN)
        || (mon.ability == Ability::QuarkDrive && state.terrain == TERRAIN_ELECTRIC);
    if !weather_or_terrain && mon.item == Item::BoosterEnergy {
        mon.item = Item::None;
    }
    // Highest stat among atk(1), def(2), spa(3), spd(4), spe(5)
    let mut best_i = 1usize;
    let mut best_v = mon.stats[1];
    for i in 2..=5 {
        if mon.stats[i] > best_v {
            best_v = mon.stats[i];
            best_i = i;
        }
    }
    mon.volatiles |= match best_i {
        1 => V_PROTO_ATK,
        2 => V_PROTO_DEF,
        3 => V_PROTO_SPA,
        4 => V_PROTO_SPD,
        _ => V_PROTO_SPE,
    };
}

pub fn run_switch_in(state: &mut BattleState, side_idx: usize, mon_idx: usize) {
    // Entry hazards
    let gravity = state.gravity > 0;
    let sr = state.sides[side_idx].stealth_rock;
    let spikes = state.sides[side_idx].spikes;
    let tspikes = state.sides[side_idx].toxic_spikes;
    let web = state.sides[side_idx].sticky_web;
    {
        let mon = &mut state.sides[side_idx].mons[mon_idx];
        if !mon.alive() {
            return;
        }
        if mon.item != Item::HeavyDutyBoots && mon.ability != Ability::MagicGuard {
            if sr {
                let eff = TYPE_CHART[TYPE_ROCK as usize][mon.types[0] as usize]
                    * TYPE_CHART[TYPE_ROCK as usize][mon.types[1] as usize];
                let dmg = ((mon.max_hp as f32) * eff / 8.0).floor().max(1.0) as u16;
                mon.hp = mon.hp.saturating_sub(dmg);
                if mon.hp == 0 {
                    mon.fainted = true;
                    return;
                }
            }
            if mon.is_grounded(gravity) {
                if spikes > 0 {
                    let frac = match spikes {
                        1 => 1.0 / 8.0,
                        2 => 1.0 / 6.0,
                        _ => 1.0 / 4.0,
                    };
                    let dmg = ((mon.max_hp as f32) * frac).floor().max(1.0) as u16;
                    mon.hp = mon.hp.saturating_sub(dmg);
                    if mon.hp == 0 {
                        mon.fainted = true;
                        return;
                    }
                }
                if tspikes > 0 && mon.status == STATUS_NONE && !mon.has_type(TYPE_STEEL) {
                    if mon.has_type(TYPE_POISON) {
                        state.sides[side_idx].toxic_spikes = 0;
                    } else {
                        mon.status = if tspikes >= 2 { STATUS_TOX } else { STATUS_PSN };
                    }
                }
                if web {
                    apply_boost(mon, 4, -1, false);
                }
            }
        }
    }
    let ab = state.sides[side_idx].mons[mon_idx].ability;
    let item = state.sides[side_idx].mons[mon_idx].item;
    match ab {
        Ability::Intimidate => {
            let foe = 1 - side_idx;
            for fslot in 0..2 {
                let fi = state.sides[foe].active[fslot];
                if fi >= 0 {
                    let fm = &mut state.sides[foe].mons[fi as usize];
                    if fm.alive()
                        && !matches!(
                            fm.ability,
                            Ability::InnerFocus | Ability::Oblivious | Ability::OwnTempo | Ability::Scrappy
                        )
                    {
                        apply_boost(fm, 0, -1, false);
                    }
                }
            }
        }
        Ability::Drought | Ability::OrichalcumPulse => {
            state.weather = WEATHER_SUN;
            state.weather_turns = if item == Item::HeatRock { 8 } else { 5 };
        }
        Ability::Drizzle => {
            state.weather = WEATHER_RAIN;
            state.weather_turns = if item == Item::DampRock { 8 } else { 5 };
        }
        Ability::SandStream => {
            state.weather = WEATHER_SAND;
            state.weather_turns = if item == Item::SmoothRock { 8 } else { 5 };
        }
        Ability::SnowWarning => {
            state.weather = WEATHER_SNOW;
            state.weather_turns = if item == Item::IcyRock { 8 } else { 5 };
        }
        Ability::ElectricSurge | Ability::HadronEngine => {
            state.terrain = TERRAIN_ELECTRIC;
            state.terrain_turns = if item == Item::TerrainExtender { 8 } else { 5 };
        }
        Ability::GrassySurge => {
            state.terrain = TERRAIN_GRASSY;
            state.terrain_turns = if item == Item::TerrainExtender { 8 } else { 5 };
        }
        Ability::PsychicSurge => {
            state.terrain = TERRAIN_PSYCHIC;
            state.terrain_turns = if item == Item::TerrainExtender { 8 } else { 5 };
        }
        Ability::MistySurge => {
            state.terrain = TERRAIN_MISTY;
            state.terrain_turns = if item == Item::TerrainExtender { 8 } else { 5 };
        }
        _ => {}
    }
    check_proto_quark(state, side_idx, mon_idx);
}

fn run_mega(state: &mut BattleState, dex: &Dex, side_idx: usize, slot: usize) {
    let mi = state.sides[side_idx].active[slot];
    if mi < 0 || state.sides[side_idx].mega_used {
        return;
    }
    let mon = &mut state.sides[side_idx].mons[mi as usize];
    if !mon.alive() || !mon.can_mega || mon.mega_target == 0 {
        return;
    }
    let old_sp = &dex.species[mon.species_id as usize];
    let new_sp = &dex.species[mon.mega_target as usize];
    // Rescale non-HP stats proportionally to base stat change (accurate to ~1 point without storing EVs/nature).
    for i in 1..6 {
        let old_b = old_sp.base_stats[i].max(1) as f32;
        let new_b = new_sp.base_stats[i].max(1) as f32;
        let diff = (new_b - old_b) as i32;
        mon.stats[i] = (mon.stats[i] as i32 + diff).max(5) as u16;
    }
    mon.species_id = mon.mega_target;
    mon.types = new_sp.types;
    mon.ability = new_sp.primary_ability;
    mon.can_mega = false;
    state.sides[side_idx].mega_used = true;
    run_switch_in(state, side_idx, mi as usize);
}

fn do_switch(state: &mut BattleState, side_idx: usize, slot: usize, target_idx: u8) {
    let t = target_idx as usize;
    if t >= state.sides[side_idx].mon_count as usize
        || !state.sides[side_idx].mons[t].alive()
        || state.sides[side_idx].is_active_idx(t)
    {
        return;
    }
    let old = state.sides[side_idx].active[slot];
    if old >= 0 {
        state.sides[side_idx].mons[old as usize].clear_on_switch_out();
    }
    // Mirror Showdown's in-place pokemon array swap so active Pokémon stay at indices 0 and 1,
    // and bench Pokémon are always at indices 2 and 3 (matching Showdown's 'switch 3' / 'switch 4').
    if t != slot && slot < state.sides[side_idx].mon_count as usize {
        state.sides[side_idx].mons.swap(slot, t);
        if state.sides[side_idx].active[1 - slot] == slot as i8 {
            state.sides[side_idx].active[1 - slot] = t as i8;
        }
        state.sides[side_idx].active[slot] = slot as i8;
        state.sides[side_idx].force_switch[slot] = false;
        run_switch_in(state, side_idx, slot);
    } else {
        state.sides[side_idx].active[slot] = t as i8;
        state.sides[side_idx].force_switch[slot] = false;
        run_switch_in(state, side_idx, t);
    }
}

fn has_ruin(state: &BattleState, ab: Ability) -> bool {
    for s in 0..2 {
        for slot in 0..2 {
            let idx = state.sides[s].active[slot];
            if idx >= 0 && state.sides[s].mons[idx as usize].alive() && state.sides[s].mons[idx as usize].ability == ab {
                return true;
            }
        }
    }
    false
}

fn ally_has_ability(state: &BattleState, side_idx: usize, slot: usize, ab: Ability) -> bool {
    let idx = state.sides[side_idx].active[1 - slot];
    idx >= 0 && state.sides[side_idx].mons[idx as usize].alive() && state.sides[side_idx].mons[idx as usize].ability == ab
}

fn check_berries(mon: &mut Mon) {
    if !mon.alive() {
        return;
    }
    match mon.item {
        Item::SitrusBerry if mon.hp * 2 <= mon.max_hp => {
            mon.hp = (mon.hp + (mon.max_hp / 4).max(1)).min(mon.max_hp);
            mon.item = Item::None;
        }
        Item::PinchBerry if mon.hp * 4 <= mon.max_hp => {
            mon.hp = (mon.hp + (mon.max_hp / 3).max(1)).min(mon.max_hp);
            mon.item = Item::None;
        }
        Item::LumBerry if mon.status != STATUS_NONE || (mon.volatiles & V_CONFUSION) != 0 => {
            mon.status = STATUS_NONE;
            mon.volatiles &= !V_CONFUSION;
            mon.item = Item::None;
        }
        _ => {}
    }
}

fn try_inflict_status(terrain: u8, gravity: bool, target: &mut Mon, status: u8) -> bool {
    if !target.alive() || status == STATUS_NONE || target.status != STATUS_NONE {
        return false;
    }
    if terrain == TERRAIN_MISTY && target.is_grounded(gravity) {
        return false;
    }
    if target.ability == Ability::PurifyingSalt || target.ability == Ability::GoodAsGold {
        return false;
    }
    match status {
        STATUS_BRN if target.has_type(TYPE_FIRE) || target.ability == Ability::ThermalExchange => return false,
        STATUS_PAR if target.has_type(TYPE_ELECTRIC) => return false,
        STATUS_PSN | STATUS_TOX if target.has_type(TYPE_POISON) || target.has_type(TYPE_STEEL) => return false,
        STATUS_FRZ if target.has_type(TYPE_ICE) => return false,
        STATUS_SLP => {
            if terrain == TERRAIN_ELECTRIC && target.is_grounded(gravity) {
                return false;
            }
            if matches!(target.ability, Ability::VitalSpirit | Ability::Insomnia | Ability::SweetVeil) {
                return false;
            }
            target.sleep_turns = 2; // Champions max sleep = 2 turns
        }
        _ => {}
    }
    target.status = status;
    check_berries(target);
    true
}

fn compute_damage(
    state: &BattleState,
    dex: &Dex,
    atk_side: usize,
    atk_slot: usize,
    def_side: usize,
    def_slot: usize,
    mv: &MoveData,
    spread: bool,
    rng: &mut Rng,
) -> (u16, f32, bool) {
    let ai = state.sides[atk_side].active[atk_slot] as usize;
    let di = state.sides[def_side].active[def_slot] as usize;
    let atk = &state.sides[atk_side].mons[ai];
    let def = &state.sides[def_side].mons[di];
    let ignore_ab = matches!(atk.ability, Ability::MoldBreaker | Ability::Teravolt | Ability::Turboblaze);

    let mut mtype = mv.move_type;
    let mut bp = mv.base_power as f32;
    match mv.id.as_str() {
        "weatherball" if state.weather != WEATHER_NONE => {
            bp = 100.0;
            mtype = match state.weather {
                WEATHER_SUN => TYPE_FIRE,
                WEATHER_RAIN => TYPE_WATER,
                WEATHER_SAND => TYPE_ROCK,
                WEATHER_SNOW => TYPE_ICE,
                _ => TYPE_NORMAL,
            };
        }
        "lastrespects" => {
            let fainted = (0..state.sides[atk_side].mon_count as usize)
                .filter(|&i| !state.sides[atk_side].mons[i].alive())
                .count();
            bp = 50.0 + 50.0 * (fainted as f32);
        }
        "facade" if atk.status != STATUS_NONE => bp *= 2.0,
        "acrobatics" if atk.item == Item::None => bp *= 2.0,
        "knockoff" if def.item != Item::None && def.item != Item::MegaStone => bp *= 1.5,
        "expandingforce" if state.terrain == TERRAIN_PSYCHIC && atk.is_grounded(state.gravity > 0) => bp *= 1.5,
        "lowkick" | "grassknot" => {
            let w = dex.species[def.species_id as usize].weight_hg;
            bp = if w < 100 { 20.0 }
                else if w < 250 { 40.0 }
                else if w < 500 { 60.0 }
                else if w < 1000 { 80.0 }
                else if w < 2000 { 100.0 }
                else { 120.0 };
        }
        _ => {}
    }
    if mtype == TYPE_NORMAL {
        match atk.ability {
            Ability::Pixilate => { mtype = TYPE_FAIRY; bp *= 1.2; }
            Ability::Aerilate => { mtype = TYPE_FLYING; bp *= 1.2; }
            Ability::Refrigerate => { mtype = TYPE_ICE; bp *= 1.2; }
            Ability::Galvanize => { mtype = TYPE_ELECTRIC; bp *= 1.2; }
            _ => {}
        }
    }
    if bp <= 0.0 {
        bp = 80.0;
    }

    // Type immunity and abilities
    let mut eff = TYPE_CHART[mtype as usize][def.types[0] as usize]
        * TYPE_CHART[mtype as usize][def.types[1] as usize];
    if eff == 0.0 && matches!(atk.ability, Ability::Scrappy | Ability::MindsEye)
        && (mtype == TYPE_NORMAL || mtype == TYPE_FIGHTING)
        && def.has_type(TYPE_GHOST)
    {
        eff = 1.0;
    }
    if !ignore_ab {
        match def.ability {
            Ability::Levitate if mtype == TYPE_GROUND && state.gravity == 0 && def.item != Item::IronBall => return (0, 0.0, false),
            Ability::FlashFire | Ability::WellBakedBody if mtype == TYPE_FIRE => return (0, 0.0, false),
            Ability::WaterAbsorb | Ability::StormDrain | Ability::DrySkin if mtype == TYPE_WATER => return (0, 0.0, false),
            Ability::VoltAbsorb | Ability::LightningRod | Ability::MotorDrive if mtype == TYPE_ELECTRIC => return (0, 0.0, false),
            Ability::SapSipper if mtype == TYPE_GRASS => return (0, 0.0, false),
            Ability::EarthEater if mtype == TYPE_GROUND => return (0, 0.0, false),
            _ => {}
        }
    }
    if eff == 0.0 {
        return (0, 0.0, false);
    }

    // Crit check (Gen 9 / Champions: 1/24 base, 1/8 +1, 1/2 +2, 100% +3)
    let crit_stage = mv.crit_ratio + if (atk.volatiles & V_FOCUS_ENERGY) != 0 { 2 } else { 0 };
    let crit = mv.will_crit
        || match crit_stage {
            0 | 1 => rng.gen_range(24) == 0,
            2 => rng.gen_range(8) == 0,
            3 => rng.gen_range(2) == 0,
            _ => true,
        };

    // Base power modifiers
    if atk.ability == Ability::Technician && bp <= 60.0 { bp *= 1.5; }
    if atk.ability == Ability::ToughClaws && (mv.flags & FLAG_CONTACT) != 0 { bp *= 1.3; }
    if atk.ability == Ability::StrongJaw && (mv.flags & FLAG_BITE) != 0 { bp *= 1.5; }
    if atk.ability == Ability::MegaLauncher && (mv.flags & FLAG_PULSE) != 0 { bp *= 1.5; }
    if atk.ability == Ability::Sharpness && (mv.flags & FLAG_SLICING) != 0 { bp *= 1.5; }
    if atk.ability == Ability::SheerForce && mv.secondary.is_some() { bp *= 1.3; }
    if atk.ability == Ability::Overgrow && mtype == TYPE_GRASS && atk.hp * 3 <= atk.max_hp { bp *= 1.5; }
    if atk.ability == Ability::Blaze && mtype == TYPE_FIRE && atk.hp * 3 <= atk.max_hp { bp *= 1.5; }
    if atk.ability == Ability::Torrent && mtype == TYPE_WATER && atk.hp * 3 <= atk.max_hp { bp *= 1.5; }
    if atk.ability == Ability::Swarm && mtype == TYPE_BUG && atk.hp * 3 <= atk.max_hp { bp *= 1.5; }
    if (atk.volatiles & V_HELPING_HAND) != 0 { bp *= 1.5; }
    if let Item::TypeBoost(t) = atk.item {
        if t == mtype { bp *= 1.2; }
    }
    if atk.item == Item::MuscleBand && mv.category == Category::Physical { bp *= 1.1; }
    if atk.item == Item::WiseGlasses && mv.category == Category::Special { bp *= 1.1; }

    // Terrain BP boost
    if atk.is_grounded(state.gravity > 0) {
        match state.terrain {
            TERRAIN_ELECTRIC if mtype == TYPE_ELECTRIC => bp *= 1.3,
            TERRAIN_GRASSY if mtype == TYPE_GRASS => bp *= 1.3,
            TERRAIN_PSYCHIC if mtype == TYPE_PSYCHIC => bp *= 1.3,
            _ => {}
        }
    }
    if def.is_grounded(state.gravity > 0) {
        if state.terrain == TERRAIN_MISTY && mtype == TYPE_DRAGON { bp *= 0.5; }
        if state.terrain == TERRAIN_GRASSY && matches!(mv.id.as_str(), "earthquake" | "bulldoze") { bp *= 0.5; }
    }

    // Attack & Defense stats
    let is_phys = mv.category == Category::Physical;
    let off_idx = if mv.override_off_stat == 1 { 2 } else if is_phys { 1 } else { 3 };
    let def_idx = if mv.override_def_stat == 1 { 2 } else if is_phys { 2 } else { 4 };
    let mut a_stat = atk.stats[off_idx] as f32;
    let mut d_stat = def.stats[def_idx] as f32;

    let a_boost = atk.boosts[off_idx - 1];
    let d_boost = def.boosts[def_idx - 1];
    if def.ability != Ability::Unaware || ignore_ab {
        let ab = if crit && a_boost < 0 { 0 } else { a_boost };
        a_stat *= boost_mult(ab);
    }
    if atk.ability != Ability::Unaware && !mv.ignore_defensive {
        let db = if crit && d_boost > 0 { 0 } else { d_boost };
        d_stat *= boost_mult(db);
    }

    if is_phys {
        if matches!(atk.ability, Ability::HugePower | Ability::PurePower) { a_stat *= 2.0; }
        if atk.ability == Ability::Hustle { a_stat *= 1.5; }
        if atk.ability == Ability::Guts && atk.status != STATUS_NONE { a_stat *= 1.5; }
        if atk.ability == Ability::ToxicBoost && matches!(atk.status, STATUS_PSN | STATUS_TOX) { a_stat *= 1.5; }
        if atk.ability == Ability::OrichalcumPulse && state.weather == WEATHER_SUN { a_stat *= 1.3333; }
        if atk.item == Item::ChoiceBand { a_stat *= 1.5; }
        if (atk.volatiles & V_PROTO_ATK) != 0 { a_stat *= 1.3; }
        if has_ruin(state, Ability::TabletsOfRuin) && atk.ability != Ability::TabletsOfRuin { a_stat *= 0.75; }
        if has_ruin(state, Ability::SwordOfRuin) && def.ability != Ability::SwordOfRuin { d_stat *= 0.75; }
        if def.ability == Ability::MarvelScale && def.status != STATUS_NONE && !ignore_ab { d_stat *= 1.5; }
        if state.weather == WEATHER_SNOW && def.has_type(TYPE_ICE) { d_stat *= 1.5; }
        if (def.volatiles & V_PROTO_DEF) != 0 { d_stat *= 1.3; }
    } else {
        if atk.ability == Ability::HadronEngine && state.terrain == TERRAIN_ELECTRIC { a_stat *= 1.3333; }
        if atk.item == Item::ChoiceSpecs { a_stat *= 1.5; }
        if (atk.volatiles & V_PROTO_SPA) != 0 { a_stat *= 1.3; }
        if has_ruin(state, Ability::VesselOfRuin) && atk.ability != Ability::VesselOfRuin { a_stat *= 0.75; }
        if has_ruin(state, Ability::BeadsOfRuin) && def.ability != Ability::BeadsOfRuin { d_stat *= 0.75; }
        if def.item == Item::AssaultVest { d_stat *= 1.5; }
        if state.weather == WEATHER_SAND && def.has_type(TYPE_ROCK) { d_stat *= 1.5; }
        if (def.volatiles & V_PROTO_SPD) != 0 { d_stat *= 1.3; }
    }
    if def.item == Item::Eviolite { d_stat *= 1.5; }
    if !ignore_ab && def.ability == Ability::ThickFat && (mtype == TYPE_FIRE || mtype == TYPE_ICE) { a_stat *= 0.5; }
    if !ignore_ab && def.ability == Ability::PurifyingSalt && mtype == TYPE_GHOST { a_stat *= 0.5; }

    let mut dmg = ((22.0 * bp * a_stat / d_stat.max(1.0)) / 50.0).floor() + 2.0;
    if spread { dmg *= 0.75; }
    // Weather
    match state.weather {
        WEATHER_SUN => {
            if mtype == TYPE_FIRE { dmg *= 1.5; }
            else if mtype == TYPE_WATER { dmg *= 0.5; }
        }
        WEATHER_RAIN => {
            if mtype == TYPE_WATER { dmg *= 1.5; }
            else if mtype == TYPE_FIRE { dmg *= 0.5; }
        }
        _ => {}
    }
    if crit { dmg *= 1.5; }
    let roll = (85 + rng.gen_range(16)) as f32 / 100.0;
    dmg *= roll;
    if atk.has_type(mtype) {
        dmg *= if atk.ability == Ability::Adaptability { 2.0 } else { 1.5 };
    }
    dmg *= eff;
    if is_phys && atk.status == STATUS_BRN && atk.ability != Ability::Guts && mv.id != "facade" {
        dmg *= 0.5;
    }
    // Doubles screens (2/3)
    if !crit {
        let ds = &state.sides[def_side];
        if (is_phys && (ds.reflect > 0 || ds.aurora_veil > 0))
            || (!is_phys && (ds.light_screen > 0 || ds.aurora_veil > 0))
        {
            dmg *= 2.0 / 3.0;
        }
    }
    if atk.item == Item::LifeOrb { dmg *= 1.3; }
    if atk.item == Item::ExpertBelt && eff > 1.0 { dmg *= 1.2; }
    if !ignore_ab && matches!(def.ability, Ability::Multiscale | Ability::ShadowShield) && def.hp == def.max_hp {
        dmg *= 0.5;
    }
    if !ignore_ab && matches!(def.ability, Ability::Filter | Ability::SolidRock) && eff > 1.0 {
        dmg *= 0.75;
    }
    if ally_has_ability(state, def_side, def_slot, Ability::FriendGuard) {
        dmg *= 0.75;
    }
    let _ = dex;
    (dmg.floor().max(1.0) as u16, eff, crit)
}

fn resolve_targets(
    state: &BattleState,
    atk_side: usize,
    atk_slot: usize,
    mv: &MoveData,
    raw_target: i8,
    rng: &mut Rng,
) -> ([(usize, usize); 4], usize) {
    let foe = 1 - atk_side;
    let mut out = [(0usize, 0usize); 4];
    let mut n = 0usize;
    let push_alive = |s: usize, sl: usize, out: &mut [(usize, usize); 4], n: &mut usize| {
        let idx = state.sides[s].active[sl];
        if idx >= 0 && state.sides[s].mons[idx as usize].alive() {
            out[*n] = (s, sl);
            *n += 1;
        }
    };
    match mv.target {
        TargetKind::AllAdjacentFoes => {
            push_alive(foe, 0, &mut out, &mut n);
            push_alive(foe, 1, &mut out, &mut n);
        }
        TargetKind::AllAdjacent => {
            push_alive(atk_side, 1 - atk_slot, &mut out, &mut n);
            push_alive(foe, 0, &mut out, &mut n);
            push_alive(foe, 1, &mut out, &mut n);
        }
        TargetKind::SelfTarget | TargetKind::AllySide | TargetKind::All => {
            out[0] = (atk_side, atk_slot);
            n = 1;
        }
        TargetKind::FoeSide => {
            out[0] = (foe, 0);
            n = 1;
        }
        TargetKind::AdjacentAlly => {
            push_alive(atk_side, 1 - atk_slot, &mut out, &mut n);
        }
        TargetKind::AdjacentAllyOrSelf => {
            let sl = if raw_target < 0 {
                ((-raw_target) as usize).saturating_sub(1).min(1)
            } else {
                atk_slot
            };
            push_alive(atk_side, sl, &mut out, &mut n);
        }
        TargetKind::RandomNormal => {
            let mut cands = [(0usize, 0usize); 2];
            let mut cn = 0;
            for sl in 0..2 {
                let idx = state.sides[foe].active[sl];
                if idx >= 0 && state.sides[foe].mons[idx as usize].alive() {
                    cands[cn] = (foe, sl);
                    cn += 1;
                }
            }
            if cn > 0 {
                out[0] = cands[rng.gen_range(cn)];
                n = 1;
            }
        }
        _ => {
            // Single target (foe or ally)
            if raw_target < 0 {
                let sl = ((-raw_target) as usize).saturating_sub(1).min(1);
                push_alive(atk_side, sl, &mut out, &mut n);
            } else {
                // Check Follow Me / Rage Powder / Lightning Rod / Storm Drain redirection on foe side
                let mut redirected = None;
                let atk_mon = &state.sides[atk_side].mons[state.sides[atk_side].active[atk_slot] as usize];
                let immune_powder = atk_mon.has_type(TYPE_GRASS) || atk_mon.item == Item::SafetyGoggles;
                for fsl in 0..2 {
                    let fi = state.sides[foe].active[fsl];
                    if fi < 0 || !state.sides[foe].mons[fi as usize].alive() {
                        continue;
                    }
                    let fm = &state.sides[foe].mons[fi as usize];
                    if (fm.volatiles & V_FOLLOW_ME) != 0
                        || ((fm.volatiles & V_RAGE_POWDER) != 0 && !immune_powder)
                        || (mv.move_type == TYPE_ELECTRIC && fm.ability == Ability::LightningRod)
                        || (mv.move_type == TYPE_WATER && fm.ability == Ability::StormDrain)
                    {
                        redirected = Some(fsl);
                        break;
                    }
                }
                let want_slot = redirected.unwrap_or_else(|| {
                    if raw_target >= 1 {
                        (raw_target as usize - 1).min(1)
                    } else {
                        0
                    }
                });
                push_alive(foe, want_slot, &mut out, &mut n);
                if n == 0 {
                    // Retarget to the other living foe if chosen target fainted earlier this turn
                    push_alive(foe, 1 - want_slot, &mut out, &mut n);
                }
            }
        }
    }
    (out, n)
}

fn execute_move(
    state: &mut BattleState,
    dex: &Dex,
    side_idx: usize,
    slot: usize,
    move_idx: u8,
    raw_target: i8,
    rng: &mut Rng,
) {
    let ai = state.sides[side_idx].active[slot];
    if ai < 0 {
        return;
    }
    let a_idx = ai as usize;
    if !state.sides[side_idx].mons[a_idx].alive() {
        return;
    }

    // Status / flinch / confusion checks before moving
    {
        let m = &mut state.sides[side_idx].mons[a_idx];
        if (m.volatiles & V_FLINCH) != 0 {
            m.volatiles &= !V_FLINCH;
            m.active_move_actions = m.active_move_actions.saturating_add(1);
            return;
        }
        if m.status == STATUS_SLP {
            if m.sleep_turns > 0 {
                m.sleep_turns -= 1;
            }
            if m.sleep_turns > 0 {
                m.active_move_actions = m.active_move_actions.saturating_add(1);
                return;
            } else {
                m.status = STATUS_NONE;
            }
        } else if m.status == STATUS_FRZ {
            // Champions thaw chance = 25%
            if rng.roll_100(25) {
                m.status = STATUS_NONE;
            } else {
                m.active_move_actions = m.active_move_actions.saturating_add(1);
                return;
            }
        } else if m.status == STATUS_PAR {
            // Champions full paralysis = 12.5%
            if rng.gen_range(8) == 0 {
                m.active_move_actions = m.active_move_actions.saturating_add(1);
                return;
            }
        }
        if (m.volatiles & V_CONFUSION) != 0 {
            if m.confuse_turns <= 1 {
                m.volatiles &= !V_CONFUSION;
                m.confuse_turns = 0;
            } else {
                m.confuse_turns -= 1;
                if rng.gen_range(3) == 0 {
                    // Self-hit
                    let dmg = ((22.0 * 40.0 * (m.stats[1] as f32) / (m.stats[2].max(1) as f32)) / 50.0 + 2.0) as u16;
                    m.hp = m.hp.saturating_sub(dmg);
                    if m.hp == 0 {
                        m.fainted = true;
                    }
                    m.active_move_actions = m.active_move_actions.saturating_add(1);
                    return;
                }
            }
        }
    }

    let m_idx = (move_idx as usize).min(3);
    let mv_id = state.sides[side_idx].mons[a_idx].moves[m_idx];
    let mv = &dex.moves[mv_id as usize];

    // Update PP, lastMove, choice lock, active_move_actions
    {
        let m = &mut state.sides[side_idx].mons[a_idx];
        if m.pp[m_idx] > 0 {
            m.pp[m_idx] -= 1;
        }
        let first_turn = m.active_move_actions == 0;
        m.active_move_actions = m.active_move_actions.saturating_add(1);
        m.last_move_idx = m_idx as i8;
        if m.item.is_choice() {
            m.volatiles |= V_CHOICELOCK;
            m.locked_move_idx = m_idx as i8;
        }
        // Fake Out only works on first turn active
        if mv.is_fake_out && !first_turn {
            m.stall_counter = 0;
            return;
        }
        // Protect / Endure / Wide Guard / Quick Guard stall check
        if mv.is_protect || mv.volatile == VOL_ENDURE || mv.is_wide_guard || mv.is_quick_guard {
            let denom = 3u32.pow(m.stall_counter.min(4) as u32) as usize;
            if m.stall_counter > 0 && rng.gen_range(denom) != 0 {
                m.stall_counter = 0;
                return;
            }
            m.stall_counter = m.stall_counter.saturating_add(1).min(4);
            if mv.is_protect {
                m.volatiles |= V_PROTECT;
            } else if mv.volatile == VOL_ENDURE {
                m.volatiles |= V_ENDURE;
            } else if mv.is_wide_guard {
                state.sides[side_idx].wide_guard = true;
            } else if mv.is_quick_guard {
                state.sides[side_idx].quick_guard = true;
            }
            return;
        } else {
            m.stall_counter = 0;
        }
    }

    // Special tactical doubles status moves
    if mv.is_follow_me {
        state.sides[side_idx].mons[a_idx].volatiles |= V_FOLLOW_ME;
        return;
    }
    if mv.is_rage_powder {
        state.sides[side_idx].mons[a_idx].volatiles |= V_RAGE_POWDER;
        return;
    }
    if mv.is_helping_hand {
        let ally = state.sides[side_idx].active[1 - slot];
        if ally >= 0 && state.sides[side_idx].mons[ally as usize].alive() {
            state.sides[side_idx].mons[ally as usize].volatiles |= V_HELPING_HAND;
        }
        return;
    }

    // Field / side condition setters
    if mv.pseudo_weather == PW_TRICKROOM {
        state.trick_room = if state.trick_room > 0 { 0 } else { 5 };
        return;
    }
    if mv.pseudo_weather == PW_GRAVITY {
        state.gravity = 5;
        return;
    }
    if mv.weather != WEATHER_NONE {
        let item = state.sides[side_idx].mons[a_idx].item;
        state.weather = mv.weather;
        state.weather_turns = match (mv.weather, item) {
            (WEATHER_SUN, Item::HeatRock)
            | (WEATHER_RAIN, Item::DampRock)
            | (WEATHER_SAND, Item::SmoothRock)
            | (WEATHER_SNOW, Item::IcyRock) => 8,
            _ => 5,
        };
        return;
    }
    if mv.terrain != TERRAIN_NONE {
        let item = state.sides[side_idx].mons[a_idx].item;
        state.terrain = mv.terrain;
        state.terrain_turns = if item == Item::TerrainExtender { 8 } else { 5 };
        return;
    }
    if mv.side_condition != SC_NONE {
        let item = state.sides[side_idx].mons[a_idx].item;
        let foe = 1 - side_idx;
        let clay = if item == Item::LightClay { 8 } else { 5 };
        match mv.side_condition {
            SC_REFLECT => state.sides[side_idx].reflect = clay,
            SC_LIGHTSCREEN => state.sides[side_idx].light_screen = clay,
            SC_AURORAVEIL if state.weather == WEATHER_SNOW => state.sides[side_idx].aurora_veil = clay,
            SC_TAILWIND => state.sides[side_idx].tailwind = 4,
            SC_SAFEGUARD => state.sides[side_idx].safeguard = 5,
            SC_MIST => state.sides[side_idx].mist = 5,
            SC_STEALTHROCK => state.sides[foe].stealth_rock = true,
            SC_SPIKES => state.sides[foe].spikes = (state.sides[foe].spikes + 1).min(3),
            SC_TOXICSPIKES => state.sides[foe].toxic_spikes = (state.sides[foe].toxic_spikes + 1).min(2),
            SC_STICKYWEB => state.sides[foe].sticky_web = true,
            _ => {}
        }
        return;
    }

    // Substitute
    if mv.volatile == VOL_SUBSTITUTE && mv.category == Category::Status {
        let m = &mut state.sides[side_idx].mons[a_idx];
        let cost = (m.max_hp / 4).max(1);
        if (m.volatiles & V_SUBSTITUTE) == 0 && m.hp > cost {
            m.hp -= cost;
            m.sub_hp = cost;
            m.volatiles |= V_SUBSTITUTE;
        }
        return;
    }

    // Self-heal moves (Recover, Roost, Synthesis, Moonlight, Protect-like heals)
    if mv.heal_num > 0 && mv.target == TargetKind::SelfTarget {
        let m = &mut state.sides[side_idx].mons[a_idx];
        let heal = ((m.max_hp as u32) * (mv.heal_num as u32) / (mv.heal_den as u32)).max(1) as u16;
        m.hp = (m.hp + heal).min(m.max_hp);
        return;
    }

    // Special charge moves that skip if weather doesn't make them instant (approximate 2-turn delay)
    if (mv.id == "solarbeam" || mv.id == "solarblade") && state.weather != WEATHER_SUN && state.sides[side_idx].mons[a_idx].ability != Ability::MegaSol {
        return;
    }
    if mv.id == "electroshot" && state.weather != WEATHER_RAIN {
        apply_boost(&mut state.sides[side_idx].mons[a_idx], 2, 1, true);
        return;
    }

    let (targets, t_count) = resolve_targets(state, side_idx, slot, mv, raw_target, rng);
    if t_count == 0 {
        return;
    }
    let spread = matches!(mv.target, TargetKind::AllAdjacentFoes | TargetKind::AllAdjacent) && t_count > 1;
    let atk_ab = state.sides[side_idx].mons[a_idx].ability;
    let eff_priority = mv.priority
        + if atk_ab == Ability::Prankster && mv.category == Category::Status { 1 } else { 0 }
        + if atk_ab == Ability::Triage && (mv.flags & FLAG_HEAL) != 0 { 3 } else { 0 };

    let mut did_hit_any = false;
    let mut total_dealt = 0u32;

    for ti in 0..t_count {
        let (ts, tsl) = targets[ti];
        let t_idx_i8 = state.sides[ts].active[tsl];
        if t_idx_i8 < 0 {
            continue;
        }
        let t_idx = t_idx_i8 as usize;
        if !state.sides[ts].mons[t_idx].alive() {
            continue;
        }
        let targeting_foe = ts != side_idx;

        // Priority blockers (Psychic Terrain, Quick Guard, Armor Tail / Dazzling / Queenly Majesty)
        if targeting_foe && eff_priority > 0 {
            let tm = &state.sides[ts].mons[t_idx];
            if state.sides[ts].quick_guard
                || (state.terrain == TERRAIN_PSYCHIC && tm.is_grounded(state.gravity > 0))
                || matches!(tm.ability, Ability::ArmorTail | Ability::Dazzling | Ability::QueenlyMajesty)
                || ally_has_ability(state, ts, tsl, Ability::ArmorTail)
                || ally_has_ability(state, ts, tsl, Ability::Dazzling)
                || ally_has_ability(state, ts, tsl, Ability::QueenlyMajesty)
            {
                continue;
            }
        }
        // Wide Guard blocks spread moves
        if targeting_foe
            && matches!(mv.target, TargetKind::AllAdjacentFoes | TargetKind::AllAdjacent)
            && state.sides[ts].wide_guard
        {
            continue;
        }
        // Protect blocks protectable moves unless Unseen Fist + contact or breaks_protect
        let bypass_protect = mv.breaks_protect
            || (atk_ab == Ability::UnseenFist && (mv.flags & FLAG_CONTACT) != 0);
        if (mv.flags & FLAG_PROTECT) != 0
            && (state.sides[ts].mons[t_idx].volatiles & V_PROTECT) != 0
            && !bypass_protect
        {
            continue;
        }
        if mv.breaks_protect {
            state.sides[ts].mons[t_idx].volatiles &= !V_PROTECT;
        }
        // Prankster vs Dark-type immunity
        if targeting_foe
            && atk_ab == Ability::Prankster
            && mv.category == Category::Status
            && state.sides[ts].mons[t_idx].has_type(TYPE_DARK)
        {
            continue;
        }
        // Powder immunity (Grass / Safety Goggles)
        if (mv.flags & FLAG_POWDER) != 0 {
            let tm = &state.sides[ts].mons[t_idx];
            if tm.has_type(TYPE_GRASS) || tm.item == Item::SafetyGoggles {
                continue;
            }
        }
        // Good as Gold immunity to status moves
        if targeting_foe && mv.category == Category::Status && state.sides[ts].mons[t_idx].ability == Ability::GoodAsGold {
            continue;
        }
        // Accuracy check
        if mv.accuracy > 0 && mv.accuracy < 100 {
            let mut acc = mv.accuracy as f32;
            if state.gravity > 0 { acc *= 5.0 / 3.0; }
            if atk_ab == Ability::Hustle && mv.category == Category::Physical { acc *= 0.8; }
            if !rng.roll_100(acc.min(100.0) as u8) {
                continue;
            }
        }

        did_hit_any = true;

        if mv.category == Category::Status {
            // Status move effects
            if mv.status != STATUS_NONE {
                let (ter, grav) = (state.terrain, state.gravity > 0);
                try_inflict_status(ter, grav, &mut state.sides[ts].mons[t_idx], mv.status);
            }
            if mv.has_boosts {
                let tm = &mut state.sides[ts].mons[t_idx];
                let is_self_t = ts == side_idx && t_idx == a_idx;
                for s in 0..6 {
                    if mv.boosts.vals[s] != 0 {
                        apply_boost(tm, s, mv.boosts.vals[s], is_self_t);
                    }
                }
            }
            match mv.volatile {
                VOL_TAUNT => {
                    let tm = &mut state.sides[ts].mons[t_idx];
                    if !matches!(tm.ability, Ability::Oblivious | Ability::AromaVeil) {
                        tm.volatiles |= V_TAUNT;
                        tm.taunt_turns = 3;
                    }
                }
                VOL_ENCORE => {
                    let tm = &mut state.sides[ts].mons[t_idx];
                    if tm.last_move_idx >= 0 && tm.ability != Ability::AromaVeil {
                        tm.volatiles |= V_ENCORE;
                        tm.locked_move_idx = tm.last_move_idx;
                        tm.encore_turns = 3;
                    }
                }
                VOL_YAWN => {
                    let tm = &mut state.sides[ts].mons[t_idx];
                    if tm.status == STATUS_NONE && (tm.volatiles & V_YAWN) == 0 {
                        tm.volatiles |= V_YAWN;
                        tm.yawn_turns = 2;
                    }
                }
                VOL_LEECHSEED => {
                    let tm = &mut state.sides[ts].mons[t_idx];
                    if !tm.has_type(TYPE_GRASS) {
                        tm.volatiles |= V_LEECHSEED;
                    }
                }
                VOL_CONFUSION => {
                    let tm = &mut state.sides[ts].mons[t_idx];
                    if tm.ability != Ability::OwnTempo {
                        tm.volatiles |= V_CONFUSION;
                        tm.confuse_turns = 2 + rng.gen_range(3) as u8;
                    }
                }
                _ => {}
            }
        } else {
            // Damaging move (single or multi-hit)
            let hits = if mv.max_hits > mv.min_hits {
                if atk_ab == Ability::SkillLink {
                    mv.max_hits
                } else if state.sides[side_idx].mons[a_idx].item == Item::LoadedDice {
                    4 + rng.gen_range(2) as u8
                } else {
                    match rng.gen_range(20) {
                        0..=6 => 2,
                        7..=13 => 3,
                        14..=16 => 4,
                        _ => 5,
                    }
                }
            } else if atk_ab == Ability::ParentalBond && !spread && mv.max_hits == 1 {
                2
            } else {
                mv.min_hits.max(1)
            };

            let mut eff_last = 1.0f32;
            for h in 0..hits {
                if !state.sides[ts].mons[t_idx].alive() {
                    break;
                }
                let (mut dmg, eff, _crit) = compute_damage(state, dex, side_idx, slot, ts, tsl, mv, spread, rng);
                eff_last = eff;
                if dmg == 0 {
                    break;
                }
                if atk_ab == Ability::ParentalBond && h == 1 {
                    dmg = (dmg / 4).max(1);
                }
                let tm = &mut state.sides[ts].mons[t_idx];
                if (tm.volatiles & V_SUBSTITUTE) != 0 && (mv.flags & (FLAG_SOUND | FLAG_BYPASS_SUB)) == 0 {
                    if dmg >= tm.sub_hp {
                        tm.sub_hp = 0;
                        tm.volatiles &= !V_SUBSTITUTE;
                    } else {
                        tm.sub_hp -= dmg;
                    }
                } else {
                    let was_full = tm.hp == tm.max_hp;
                    if dmg >= tm.hp {
                        if (tm.volatiles & V_ENDURE) != 0 {
                            dmg = tm.hp - 1;
                        } else if was_full && (tm.item == Item::FocusSash || tm.ability == Ability::Sturdy) {
                            dmg = tm.hp - 1;
                            if tm.item == Item::FocusSash {
                                tm.item = Item::None;
                            }
                        } else {
                            dmg = tm.hp;
                        }
                    }
                    tm.hp -= dmg;
                    total_dealt += dmg as u32;
                    if tm.hp == 0 {
                        tm.fainted = true;
                        tm.volatiles = 0;
                    } else {
                        check_berries(tm);
                        if eff > 1.0 && tm.item == Item::WeaknessPolicy {
                            tm.item = Item::None;
                            apply_boost(tm, 0, 2, true);
                            apply_boost(tm, 2, 2, true);
                        }
                    }
                }
            }

            // Contact effects (Rocky Helmet, Rough Skin / Iron Barbs, Flame Body, Static)
            if eff_last > 0.0 && (mv.flags & FLAG_CONTACT) != 0 {
                let (t_item, t_ab) = (state.sides[ts].mons[t_idx].item, state.sides[ts].mons[t_idx].ability);
                let am = &mut state.sides[side_idx].mons[a_idx];
                if am.alive() && am.ability != Ability::MagicGuard {
                    if t_item == Item::RockyHelmet {
                        let d = (am.max_hp / 6).max(1);
                        am.hp = am.hp.saturating_sub(d);
                    }
                    if matches!(t_ab, Ability::RoughSkin | Ability::IronBarbs) {
                        let d = (am.max_hp / 8).max(1);
                        am.hp = am.hp.saturating_sub(d);
                    }
                    if am.hp == 0 { am.fainted = true; }
                }
                let (ter, grav) = (state.terrain, state.gravity > 0);
                if state.sides[side_idx].mons[a_idx].alive() && t_ab == Ability::FlameBody && rng.roll_100(30) {
                    try_inflict_status(ter, grav, &mut state.sides[side_idx].mons[a_idx], STATUS_BRN);
                } else if state.sides[side_idx].mons[a_idx].alive() && t_ab == Ability::Static && rng.roll_100(30) {
                    try_inflict_status(ter, grav, &mut state.sides[side_idx].mons[a_idx], STATUS_PAR);
                }
            }

            // Moxie on KO
            if !state.sides[ts].mons[t_idx].alive() && atk_ab == Ability::Moxie {
                apply_boost(&mut state.sides[side_idx].mons[a_idx], 0, 1, true);
            }

            // Secondary effects (suppressed by Sheer Force on user or Covert Cloak / Shield Dust on target)
            if eff_last > 0.0 && atk_ab != Ability::SheerForce {
                if let Some(sec) = mv.secondary {
                    let chance = if atk_ab == Ability::SereneGrace {
                        (sec.chance.saturating_mul(2)).min(100)
                    } else {
                        sec.chance
                    };
                    if rng.roll_100(chance) {
                        let blocked = {
                            let tm = &state.sides[ts].mons[t_idx];
                            !tm.alive()
                                || (tm.volatiles & V_SUBSTITUTE) != 0
                                || tm.item == Item::CovertCloak
                                || tm.ability == Ability::ShieldDust
                        };
                        if !blocked {
                            if sec.status != STATUS_NONE {
                                let (ter, grav) = (state.terrain, state.gravity > 0);
                                try_inflict_status(ter, grav, &mut state.sides[ts].mons[t_idx], sec.status);
                            }
                            if sec.volatile == VOL_FLINCH
                                && state.sides[ts].mons[t_idx].ability != Ability::InnerFocus
                            {
                                state.sides[ts].mons[t_idx].volatiles |= V_FLINCH;
                            } else if sec.volatile == VOL_CONFUSION
                                && state.sides[ts].mons[t_idx].ability != Ability::OwnTempo
                            {
                                state.sides[ts].mons[t_idx].volatiles |= V_CONFUSION;
                                state.sides[ts].mons[t_idx].confuse_turns = 2 + rng.gen_range(3) as u8;
                            }
                            if sec.has_boosts {
                                let tm = &mut state.sides[ts].mons[t_idx];
                                for s in 0..6 {
                                    if sec.boosts.vals[s] != 0 {
                                        apply_boost(tm, s, sec.boosts.vals[s], false);
                                    }
                                }
                            }
                        }
                        if sec.has_self_boost {
                            let am = &mut state.sides[side_idx].mons[a_idx];
                            for s in 0..6 {
                                if sec.self_boost.vals[s] != 0 {
                                    apply_boost(am, s, sec.self_boost.vals[s], true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Drain, recoil, Life Orb, self boosts, self switch
    if did_hit_any && state.sides[side_idx].mons[a_idx].alive() {
        let am = &mut state.sides[side_idx].mons[a_idx];
        if mv.drain_num > 0 && total_dealt > 0 {
            let heal = ((total_dealt * mv.drain_num as u32) / mv.drain_den as u32).max(1) as u16;
            am.hp = (am.hp + heal).min(am.max_hp);
        }
        if mv.recoil_num > 0 && total_dealt > 0 && am.ability != Ability::MagicGuard {
            let rec = ((total_dealt * mv.recoil_num as u32) / mv.recoil_den as u32).max(1) as u16;
            am.hp = am.hp.saturating_sub(rec);
            if am.hp == 0 { am.fainted = true; }
        }
        if am.alive() && mv.category != Category::Status && total_dealt > 0
            && am.item == Item::LifeOrb && am.ability != Ability::MagicGuard
            && !(am.ability == Ability::SheerForce && mv.secondary.is_some())
        {
            let lo = (am.max_hp / 10).max(1);
            am.hp = am.hp.saturating_sub(lo);
            if am.hp == 0 { am.fainted = true; }
        }
        if am.alive() && mv.has_self_boost {
            for s in 0..6 {
                if mv.self_boost.vals[s] != 0 {
                    apply_boost(am, s, mv.self_boost.vals[s], true);
                }
            }
        }
        // Self-switch (U-turn, Volt Switch, Flip Turn, Parting Shot): pivot to first healthy bench mon
        if am.alive() && mv.self_switch {
            let (bench, bn) = state.sides[side_idx].bench_alive();
            if bn > 0 {
                do_switch(state, side_idx, slot, bench[0]);
            }
        }
    }
}

fn end_of_turn(state: &mut BattleState) {
    // Clear single-turn volatiles
    for s in 0..2 {
        state.sides[s].wide_guard = false;
        state.sides[s].quick_guard = false;
        for slot in 0..2 {
            let idx = state.sides[s].active[slot];
            if idx >= 0 {
                let m = &mut state.sides[s].mons[idx as usize];
                m.volatiles &= !(V_PROTECT | V_ENDURE | V_HELPING_HAND | V_FOLLOW_ME | V_RAGE_POWDER | V_FLINCH);
            }
        }
    }

    // Weather countdown & Sand damage
    let gravity = state.gravity > 0;
    if state.weather == WEATHER_SAND {
        for s in 0..2 {
            for slot in 0..2 {
                let idx = state.sides[s].active[slot];
                if idx >= 0 {
                    let m = &mut state.sides[s].mons[idx as usize];
                    if m.alive()
                        && !m.has_type(TYPE_ROCK)
                        && !m.has_type(TYPE_GROUND)
                        && !m.has_type(TYPE_STEEL)
                        && m.ability != Ability::MagicGuard
                        && m.ability != Ability::SandRush
                        && m.item != Item::SafetyGoggles
                    {
                        let d = (m.max_hp / 16).max(1);
                        m.hp = m.hp.saturating_sub(d);
                        if m.hp == 0 { m.fainted = true; }
                    }
                }
            }
        }
    }
    if state.weather_turns > 0 {
        state.weather_turns -= 1;
        if state.weather_turns == 0 {
            state.weather = WEATHER_NONE;
        }
    }
    // Grassy terrain heal & terrain countdown
    if state.terrain == TERRAIN_GRASSY {
        for s in 0..2 {
            for slot in 0..2 {
                let idx = state.sides[s].active[slot];
                if idx >= 0 {
                    let m = &mut state.sides[s].mons[idx as usize];
                    if m.alive() && m.is_grounded(gravity) {
                        m.hp = (m.hp + (m.max_hp / 16).max(1)).min(m.max_hp);
                    }
                }
            }
        }
    }
    if state.terrain_turns > 0 {
        state.terrain_turns -= 1;
        if state.terrain_turns == 0 {
            state.terrain = TERRAIN_NONE;
        }
    }
    if state.trick_room > 0 { state.trick_room -= 1; }
    if state.gravity > 0 { state.gravity -= 1; }

    // Side conditions countdown
    for s in 0..2 {
        let sd = &mut state.sides[s];
        if sd.reflect > 0 { sd.reflect -= 1; }
        if sd.light_screen > 0 { sd.light_screen -= 1; }
        if sd.aurora_veil > 0 { sd.aurora_veil -= 1; }
        if sd.tailwind > 0 { sd.tailwind -= 1; }
        if sd.safeguard > 0 { sd.safeguard -= 1; }
        if sd.mist > 0 { sd.mist -= 1; }
    }

    // Per-active residual: Leftovers, status damage, Leech Seed, Yawn, Perish, Taunt/Encore, Speed Boost
    for s in 0..2 {
        for slot in 0..2 {
            let idx = state.sides[s].active[slot];
            if idx < 0 { continue; }
            let i = idx as usize;
            if !state.sides[s].mons[i].alive() { continue; }
            let m = &mut state.sides[s].mons[i];
            if m.item == Item::Leftovers || (m.item == Item::BlackSludge && m.has_type(TYPE_POISON)) {
                m.hp = (m.hp + (m.max_hp / 16).max(1)).min(m.max_hp);
            }
            if m.ability != Ability::MagicGuard {
                match m.status {
                    STATUS_BRN => {
                        let d = (m.max_hp / 16).max(1);
                        m.hp = m.hp.saturating_sub(d);
                    }
                    STATUS_PSN => {
                        if m.ability == Ability::PoisonHeal {
                            m.hp = (m.hp + (m.max_hp / 8).max(1)).min(m.max_hp);
                        } else {
                            let d = (m.max_hp / 8).max(1);
                            m.hp = m.hp.saturating_sub(d);
                        }
                    }
                    STATUS_TOX => {
                        if m.ability == Ability::PoisonHeal {
                            m.hp = (m.hp + (m.max_hp / 8).max(1)).min(m.max_hp);
                        } else {
                            m.toxic_counter = m.toxic_counter.saturating_add(1).min(15);
                            let d = ((m.max_hp as u32 * m.toxic_counter as u32) / 16).max(1) as u16;
                            m.hp = m.hp.saturating_sub(d);
                        }
                    }
                    _ => {}
                }
            }
            if m.hp == 0 {
                m.fainted = true;
                continue;
            }
            if (m.volatiles & V_YAWN) != 0 {
                if m.yawn_turns <= 1 {
                    m.volatiles &= !V_YAWN;
                    m.yawn_turns = 0;
                    if m.status == STATUS_NONE {
                        m.status = STATUS_SLP;
                        m.sleep_turns = 2;
                    }
                } else {
                    m.yawn_turns -= 1;
                }
            }
            if (m.volatiles & V_TAUNT) != 0 {
                if m.taunt_turns <= 1 { m.volatiles &= !V_TAUNT; m.taunt_turns = 0; }
                else { m.taunt_turns -= 1; }
            }
            if (m.volatiles & V_ENCORE) != 0 {
                if m.encore_turns <= 1 { m.volatiles &= !V_ENCORE; m.encore_turns = 0; }
                else { m.encore_turns -= 1; }
            }
            if (m.volatiles & V_PERISH) != 0 {
                if m.perish_turns <= 1 {
                    m.hp = 0;
                    m.fainted = true;
                    continue;
                } else {
                    m.perish_turns -= 1;
                }
            }
            if m.ability == Ability::SpeedBoost {
                apply_boost(m, 4, 1, true);
            }
            check_berries(m);
        }
    }

    // Check which active slots need force switches
    for s in 0..2 {
        let (_, bn) = state.sides[s].bench_alive();
        for slot in 0..2 {
            let idx = state.sides[s].active[slot];
            let dead = idx < 0 || !state.sides[s].mons[idx as usize].alive();
            if dead {
                state.sides[s].active[slot] = -1;
                state.sides[s].force_switch[slot] = bn > 0;
            } else {
                state.sides[s].force_switch[slot] = false;
            }
        }
    }
    state.turn += 1;
}

/// Advance the battle by one decision step (either force-switch resolution or a full doubles turn).
pub fn step(state: &mut BattleState, dex: &Dex, actions: [[SlotAction; 2]; 2], rng: &mut Rng) {
    if state.terminal().is_some() {
        return;
    }

    // Force-switch phase: execute pending switches and return cleanly to the next move turn.
    if state.needs_force_switch(0) || state.needs_force_switch(1) {
        for s in 0..2 {
            if !state.needs_force_switch(s) {
                continue;
            }
            for slot in 0..2 {
                if !state.sides[s].force_switch[slot] {
                    continue;
                }
                if let SlotAction::Switch(t) = actions[s][slot] {
                    do_switch(state, s, slot, t);
                } else {
                    // Fallback: auto-fill if bench available
                    let (bench, bn) = state.sides[s].bench_alive();
                    if bn > 0 {
                        do_switch(state, s, slot, bench[0]);
                    } else {
                        state.sides[s].force_switch[slot] = false;
                    }
                }
            }
        }
        return;
    }

    // 1. Switches execute before moves (ordered by speed, irrelevant except for weather/terrain ties)
    for s in 0..2 {
        for slot in 0..2 {
            if let SlotAction::Switch(t) = actions[s][slot] {
                do_switch(state, s, slot, t);
            }
        }
    }

    // 2. Mega Evolutions execute before moves
    for s in 0..2 {
        for slot in 0..2 {
            if actions[s][slot].is_mega() {
                run_mega(state, dex, s, slot);
            }
        }
    }

    // 3. Collect all move actions and resolve in dynamic (priority, speed) order
    let mut pending = [(0usize, 0usize, 0u8, 0i8); 4];
    let mut p_len = 0usize;
    for s in 0..2 {
        for slot in 0..2 {
            if let SlotAction::Move { move_idx, target, .. } = actions[s][slot] {
                pending[p_len] = (s, slot, move_idx, target);
                p_len += 1;
            }
        }
    }

    let mut used = [false; 4];
    for _ in 0..p_len {
        let mut best_k = None;
        let mut best_key = (i8::MIN, 0i64, 0u64);
        let tr = state.trick_room > 0;
        for k in 0..p_len {
            if used[k] {
                continue;
            }
            let (s, slot, m_idx, _) = pending[k];
            let ai = state.sides[s].active[slot];
            if ai < 0 || !state.sides[s].mons[ai as usize].alive() {
                used[k] = true;
                continue;
            }
            let mon = &state.sides[s].mons[ai as usize];
            let mv = &dex.moves[mon.moves[(m_idx as usize).min(3)] as usize];
            let prio = mv.priority
                + if mon.ability == Ability::Prankster && mv.category == Category::Status { 1 } else { 0 }
                + if mon.ability == Ability::Triage && (mv.flags & FLAG_HEAL) != 0 { 3 } else { 0 }
                + if mon.ability == Ability::GaleWings && mv.move_type == TYPE_FLYING && mon.hp == mon.max_hp { 1 } else { 0 };
            let spe = effective_speed(state, s, mon) as i64;
            let spe_ord = if tr { -spe } else { spe };
            let tie = rng.next_u64();
            let key = (prio, spe_ord, tie);
            if best_k.is_none() || key > best_key {
                best_k = Some(k);
                best_key = key;
            }
        }
        let Some(k) = best_k else { break; };
        used[k] = true;
        let (s, slot, m_idx, target) = pending[k];
        execute_move(state, dex, s, slot, m_idx, target, rng);
        if state.terminal().is_some() {
            return;
        }
    }

    // 4. End of turn residuals and force-switch flags
    end_of_turn(state);
}
