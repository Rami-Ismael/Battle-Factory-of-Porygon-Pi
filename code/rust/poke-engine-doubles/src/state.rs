use crate::dex::{Ability, Item, STATUS_NONE, TYPE_FLYING, TYPE_NONE};

pub const V_PROTECT: u32 = 1 << 0;
pub const V_ENDURE: u32 = 1 << 1;
pub const V_SUBSTITUTE: u32 = 1 << 2;
pub const V_CONFUSION: u32 = 1 << 3;
pub const V_TAUNT: u32 = 1 << 4;
pub const V_ENCORE: u32 = 1 << 5;
pub const V_YAWN: u32 = 1 << 6;
pub const V_LEECHSEED: u32 = 1 << 7;
pub const V_PERISH: u32 = 1 << 8;
pub const V_DISABLE: u32 = 1 << 9;
pub const V_CHOICELOCK: u32 = 1 << 10;
pub const V_HELPING_HAND: u32 = 1 << 11;
pub const V_FOLLOW_ME: u32 = 1 << 12;
pub const V_RAGE_POWDER: u32 = 1 << 13;
pub const V_FLINCH: u32 = 1 << 14;
pub const V_FOCUS_ENERGY: u32 = 1 << 15;
pub const V_FLASH_FIRE: u32 = 1 << 16;
pub const V_PROTO_ATK: u32 = 1 << 17;
pub const V_PROTO_DEF: u32 = 1 << 18;
pub const V_PROTO_SPA: u32 = 1 << 19;
pub const V_PROTO_SPD: u32 = 1 << 20;
pub const V_PROTO_SPE: u32 = 1 << 21;

#[derive(Copy, Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    #[inline]
    pub fn gen_range(&mut self, n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }

    #[inline]
    pub fn roll_100(&mut self, pct: u8) -> bool {
        if pct >= 100 {
            true
        } else if pct == 0 {
            false
        } else {
            ((self.next_u64() % 100) as u8) < pct
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Mon {
    pub species_id: u16,
    pub mega_target: u16,
    pub max_hp: u16,
    pub hp: u16,
    pub stats: [u16; 6],  // 0:hp, 1:atk, 2:def, 3:spa, 4:spd, 5:spe
    pub boosts: [i8; 6],  // 0:atk, 1:def, 2:spa, 3:spd, 4:spe, 5:acc
    pub types: [u8; 2],
    pub ability: Ability,
    pub item: Item,
    pub status: u8,
    pub sleep_turns: u8,
    pub toxic_counter: u8,
    pub moves: [u16; 4],
    pub pp: [u8; 4],
    pub disabled: [bool; 4],
    pub move_count: u8,
    pub volatiles: u32,
    pub sub_hp: u16,
    pub stall_counter: u8,
    pub active_move_actions: u8,
    pub last_move_idx: i8, // 0..3 or -1
    pub locked_move_idx: i8,
    pub taunt_turns: u8,
    pub encore_turns: u8,
    pub confuse_turns: u8,
    pub perish_turns: u8,
    pub yawn_turns: u8,
    pub fainted: bool,
    pub can_mega: bool,
    pub is_trapped: bool,
}

impl Default for Mon {
    fn default() -> Self {
        Self {
            species_id: 0,
            mega_target: 0,
            max_hp: 100,
            hp: 100,
            stats: [100; 6],
            boosts: [0; 6],
            types: [0, TYPE_NONE],
            ability: Ability::Other,
            item: Item::None,
            status: STATUS_NONE,
            sleep_turns: 0,
            toxic_counter: 0,
            moves: [0; 4],
            pp: [10; 4],
            disabled: [false; 4],
            move_count: 0,
            volatiles: 0,
            sub_hp: 0,
            stall_counter: 0,
            active_move_actions: 0,
            last_move_idx: -1,
            locked_move_idx: -1,
            taunt_turns: 0,
            encore_turns: 0,
            confuse_turns: 0,
            perish_turns: 0,
            yawn_turns: 0,
            fainted: true,
            can_mega: false,
            is_trapped: false,
        }
    }
}

impl Mon {
    #[inline]
    pub fn alive(&self) -> bool {
        !self.fainted && self.hp > 0
    }

    #[inline]
    pub fn has_type(&self, t: u8) -> bool {
        self.types[0] == t || self.types[1] == t
    }

    #[inline]
    pub fn is_grounded(&self, gravity: bool) -> bool {
        if gravity || self.item == Item::IronBall {
            return true;
        }
        !self.has_type(TYPE_FLYING) && self.ability != Ability::Levitate
    }

    pub fn clear_on_switch_out(&mut self) {
        self.boosts = [0; 6];
        self.volatiles = 0;
        self.sub_hp = 0;
        self.stall_counter = 0;
        self.active_move_actions = 0;
        self.last_move_idx = -1;
        self.locked_move_idx = -1;
        self.taunt_turns = 0;
        self.encore_turns = 0;
        self.confuse_turns = 0;
        self.perish_turns = 0;
        self.yawn_turns = 0;
        self.toxic_counter = 0;
        self.is_trapped = false;
        if self.ability == Ability::Regenerator && self.alive() {
            let heal = (self.max_hp / 3).max(1);
            self.hp = (self.hp + heal).min(self.max_hp);
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Side {
    pub mons: [Mon; 4],
    pub mon_count: u8,
    pub active: [i8; 2], // index in 0..4, or -1 if fainted/empty
    pub mega_used: bool,
    pub force_switch: [bool; 2],
    pub reflect: u8,
    pub light_screen: u8,
    pub aurora_veil: u8,
    pub tailwind: u8,
    pub safeguard: u8,
    pub mist: u8,
    pub stealth_rock: bool,
    pub spikes: u8,
    pub toxic_spikes: u8,
    pub sticky_web: bool,
    pub wide_guard: bool,
    pub quick_guard: bool,
}

impl Side {
    #[inline]
    pub fn alive_count(&self) -> u8 {
        let mut c = 0;
        for i in 0..self.mon_count as usize {
            if self.mons[i].alive() {
                c += 1;
            }
        }
        c
    }

    #[inline]
    pub fn is_active_idx(&self, idx: usize) -> bool {
        self.active[0] == idx as i8 || self.active[1] == idx as i8
    }

    pub fn bench_alive(&self) -> ([u8; 4], usize) {
        let mut out = [0u8; 4];
        let mut n = 0;
        for i in 0..self.mon_count as usize {
            if !self.is_active_idx(i) && self.mons[i].alive() {
                out[n] = i as u8;
                n += 1;
            }
        }
        (out, n)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BattleState {
    pub sides: [Side; 2],
    pub weather: u8,
    pub weather_turns: u8,
    pub terrain: u8,
    pub terrain_turns: u8,
    pub trick_room: u8,
    pub gravity: u8,
    pub turn: u16,
}

impl BattleState {
    /// Returns None if ongoing, Some(1.0) if p1 won, Some(0.0) if p2 won, Some(0.5) if tie.
    #[inline]
    pub fn terminal(&self) -> Option<f32> {
        let a0 = self.sides[0].alive_count();
        let a1 = self.sides[1].alive_count();
        if a0 == 0 && a1 == 0 {
            Some(0.5)
        } else if a1 == 0 {
            Some(1.0)
        } else if a0 == 0 {
            Some(0.0)
        } else {
            None
        }
    }

    #[inline]
    pub fn needs_force_switch(&self, side: usize) -> bool {
        self.sides[side].force_switch[0] || self.sides[side].force_switch[1]
    }

    /// Compact signature for open-loop MCTS child keying.
    #[inline]
    pub fn signature(&self) -> u64 {
        let mut h = 0u64;
        for s in 0..2 {
            let sd = &self.sides[s];
            for slot in 0..2 {
                let idx = sd.active[slot];
                let alive_code = if idx >= 0 && sd.mons[idx as usize].alive() {
                    (idx as u64) + 1
                } else {
                    0
                };
                h = (h << 4) | alive_code;
                h = (h << 1) | (sd.force_switch[slot] as u64);
            }
        }
        h = (h << 3) | (self.weather as u64);
        h = (h << 3) | (self.terrain as u64);
        h = (h << 1) | ((self.trick_room > 0) as u64);
        h
    }
}
