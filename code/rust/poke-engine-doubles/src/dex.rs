use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const TYPE_NONE: u8 = 18;
pub const TYPE_NORMAL: u8 = 0;
pub const TYPE_FIRE: u8 = 1;
pub const TYPE_WATER: u8 = 2;
pub const TYPE_ELECTRIC: u8 = 3;
pub const TYPE_GRASS: u8 = 4;
pub const TYPE_ICE: u8 = 5;
pub const TYPE_FIGHTING: u8 = 6;
pub const TYPE_POISON: u8 = 7;
pub const TYPE_GROUND: u8 = 8;
pub const TYPE_FLYING: u8 = 9;
pub const TYPE_PSYCHIC: u8 = 10;
pub const TYPE_BUG: u8 = 11;
pub const TYPE_ROCK: u8 = 12;
pub const TYPE_GHOST: u8 = 13;
pub const TYPE_DRAGON: u8 = 14;
pub const TYPE_DARK: u8 = 15;
pub const TYPE_STEEL: u8 = 16;
pub const TYPE_FAIRY: u8 = 17;

#[rustfmt::skip]
pub const TYPE_CHART: [[f32; 19]; 19] = [
/*        NOR  FIR  WAT  ELE  GRA  ICE  FIG  POI  GRO  FLY  PSY  BUG  ROC  GHO  DRA  DAR  STE  FAI  NON */
/* NOR */ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 1.0, 1.0],
/* FIR */ [1.0, 0.5, 0.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0, 1.0],
/* WAT */ [1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
/* ELE */ [1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
/* GRA */ [1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0],
/* ICE */ [1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0],
/* FIG */ [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 2.0, 0.0, 1.0, 2.0, 2.0, 0.5, 1.0],
/* POI */ [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 2.0, 1.0],
/* GRO */ [1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
/* FLY */ [1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0],
/* PSY */ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 1.0, 1.0],
/* BUG */ [1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.5, 0.5, 1.0],
/* ROC */ [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0],
/* GHO */ [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 1.0],
/* DRA */ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 0.0, 1.0],
/* DAR */ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5, 1.0],
/* STE */ [1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 2.0, 1.0],
/* FAI */ [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.0, 1.0],
/* NON */ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
];

pub fn parse_type(s: &str) -> u8 {
    match s.to_ascii_lowercase().as_str() {
        "normal" => TYPE_NORMAL,
        "fire" => TYPE_FIRE,
        "water" => TYPE_WATER,
        "electric" => TYPE_ELECTRIC,
        "grass" => TYPE_GRASS,
        "ice" => TYPE_ICE,
        "fighting" => TYPE_FIGHTING,
        "poison" => TYPE_POISON,
        "ground" => TYPE_GROUND,
        "flying" => TYPE_FLYING,
        "psychic" => TYPE_PSYCHIC,
        "bug" => TYPE_BUG,
        "rock" => TYPE_ROCK,
        "ghost" => TYPE_GHOST,
        "dragon" => TYPE_DRAGON,
        "dark" => TYPE_DARK,
        "steel" => TYPE_STEEL,
        "fairy" => TYPE_FAIRY,
        _ => TYPE_NONE,
    }
}

pub fn to_id(s: &str) -> String {
    s.bytes()
        .filter(|b| b.is_ascii_alphanumeric())
        .map(|b| b.to_ascii_lowercase() as char)
        .collect()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Category {
    Physical,
    Special,
    Status,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TargetKind {
    Normal,
    AdjacentFoe,
    Any,
    AllAdjacentFoes,
    AllAdjacent,
    AdjacentAlly,
    AdjacentAllyOrSelf,
    SelfTarget,
    AllySide,
    FoeSide,
    All,
    RandomNormal,
    Scripted,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Ability {
    #[default]
    Other,
    Intimidate,
    Drought,
    Drizzle,
    SandStream,
    SnowWarning,
    ElectricSurge,
    GrassySurge,
    PsychicSurge,
    MistySurge,
    SwiftSwim,
    Chlorophyll,
    SandRush,
    SlushRush,
    SurgeSurfer,
    Prankster,
    Triage,
    GaleWings,
    HugePower,
    PurePower,
    Technician,
    Adaptability,
    ToughClaws,
    SheerForce,
    StrongJaw,
    MegaLauncher,
    Sharpness,
    Hustle,
    Guts,
    MarvelScale,
    QuickFeet,
    PoisonHeal,
    ToxicBoost,
    MagicGuard,
    Multiscale,
    ShadowShield,
    ThickFat,
    Filter,
    SolidRock,
    Levitate,
    FlashFire,
    WaterAbsorb,
    StormDrain,
    DrySkin,
    VoltAbsorb,
    LightningRod,
    MotorDrive,
    SapSipper,
    EarthEater,
    WellBakedBody,
    GoodAsGold,
    ArmorTail,
    Dazzling,
    QueenlyMajesty,
    InnerFocus,
    ShieldDust,
    ClearBody,
    WhiteSmoke,
    FullMetalBody,
    Defiant,
    Competitive,
    Contrary,
    Unaware,
    MoldBreaker,
    Teravolt,
    Turboblaze,
    Sturdy,
    Regenerator,
    SpeedBoost,
    Moxie,
    UnseenFist,
    SwordOfRuin,
    BeadsOfRuin,
    TabletsOfRuin,
    VesselOfRuin,
    Protosynthesis,
    QuarkDrive,
    OrichalcumPulse,
    HadronEngine,
    Overgrow,
    Blaze,
    Torrent,
    Swarm,
    SereneGrace,
    SkillLink,
    MegaSol,
    ShadowTag,
    ArenaTrap,
    RoughSkin,
    IronBarbs,
    FlameBody,
    Static,
    ParentalBond,
    Pixilate,
    Aerilate,
    Refrigerate,
    Galvanize,
    Scrappy,
    MindsEye,
    FriendGuard,
    ThermalExchange,
    PurifyingSalt,
    VitalSpirit,
    Insomnia,
    SweetVeil,
    Oblivious,
    OwnTempo,
    AromaVeil,
}

impl Ability {
    pub fn from_id(id: &str) -> Self {
        match id {
            "intimidate" => Self::Intimidate,
            "drought" => Self::Drought,
            "drizzle" => Self::Drizzle,
            "sandstream" => Self::SandStream,
            "snowwarning" => Self::SnowWarning,
            "electricsurge" => Self::ElectricSurge,
            "grassysurge" => Self::GrassySurge,
            "psychicsurge" => Self::PsychicSurge,
            "mistysurge" => Self::MistySurge,
            "swiftswim" => Self::SwiftSwim,
            "chlorophyll" => Self::Chlorophyll,
            "sandrush" => Self::SandRush,
            "slushrush" => Self::SlushRush,
            "surgesurfer" => Self::SurgeSurfer,
            "prankster" => Self::Prankster,
            "triage" => Self::Triage,
            "galewings" => Self::GaleWings,
            "hugepower" => Self::HugePower,
            "purepower" => Self::PurePower,
            "technician" => Self::Technician,
            "adaptability" => Self::Adaptability,
            "toughclaws" => Self::ToughClaws,
            "sheerforce" => Self::SheerForce,
            "strongjaw" => Self::StrongJaw,
            "megalauncher" => Self::MegaLauncher,
            "sharpness" => Self::Sharpness,
            "hustle" => Self::Hustle,
            "guts" => Self::Guts,
            "marvelscale" => Self::MarvelScale,
            "quickfeet" => Self::QuickFeet,
            "poisonheal" => Self::PoisonHeal,
            "toxicboost" => Self::ToxicBoost,
            "magicguard" => Self::MagicGuard,
            "multiscale" => Self::Multiscale,
            "shadowshield" => Self::ShadowShield,
            "thickfat" => Self::ThickFat,
            "filter" => Self::Filter,
            "solidrock" => Self::SolidRock,
            "levitate" => Self::Levitate,
            "flashfire" => Self::FlashFire,
            "waterabsorb" => Self::WaterAbsorb,
            "stormdrain" => Self::StormDrain,
            "dryskin" => Self::DrySkin,
            "voltabsorb" => Self::VoltAbsorb,
            "lightningrod" => Self::LightningRod,
            "motordrive" => Self::MotorDrive,
            "sapsipper" => Self::SapSipper,
            "eartheater" => Self::EarthEater,
            "wellbakedbody" => Self::WellBakedBody,
            "goodasgold" => Self::GoodAsGold,
            "armortail" => Self::ArmorTail,
            "dazzling" => Self::Dazzling,
            "queenlymajesty" => Self::QueenlyMajesty,
            "innerfocus" => Self::InnerFocus,
            "shielddust" => Self::ShieldDust,
            "clearbody" => Self::ClearBody,
            "whitesmoke" => Self::WhiteSmoke,
            "fullmetalbody" => Self::FullMetalBody,
            "defiant" => Self::Defiant,
            "competitive" => Self::Competitive,
            "contrary" => Self::Contrary,
            "unaware" => Self::Unaware,
            "moldbreaker" => Self::MoldBreaker,
            "teravolt" => Self::Teravolt,
            "turboblaze" => Self::Turboblaze,
            "sturdy" => Self::Sturdy,
            "regenerator" => Self::Regenerator,
            "speedboost" => Self::SpeedBoost,
            "moxie" => Self::Moxie,
            "unseenfist" => Self::UnseenFist,
            "swordofruin" => Self::SwordOfRuin,
            "beadsofruin" => Self::BeadsOfRuin,
            "tabletsofruin" => Self::TabletsOfRuin,
            "vesselofruin" => Self::VesselOfRuin,
            "protosynthesis" => Self::Protosynthesis,
            "quarkdrive" => Self::QuarkDrive,
            "orichalcumpulse" => Self::OrichalcumPulse,
            "hadronengine" => Self::HadronEngine,
            "overgrow" => Self::Overgrow,
            "blaze" => Self::Blaze,
            "torrent" => Self::Torrent,
            "swarm" => Self::Swarm,
            "serenegrace" => Self::SereneGrace,
            "skilllink" => Self::SkillLink,
            "megasol" => Self::MegaSol,
            "shadowtag" => Self::ShadowTag,
            "arenatrap" => Self::ArenaTrap,
            "roughskin" => Self::RoughSkin,
            "ironbarbs" => Self::IronBarbs,
            "flamebody" => Self::FlameBody,
            "static" => Self::Static,
            "parentalbond" => Self::ParentalBond,
            "pixilate" => Self::Pixilate,
            "aerilate" => Self::Aerilate,
            "refrigerate" => Self::Refrigerate,
            "galvanize" => Self::Galvanize,
            "scrappy" => Self::Scrappy,
            "mindseye" => Self::MindsEye,
            "friendguard" => Self::FriendGuard,
            "thermalexchange" => Self::ThermalExchange,
            "purifyingsalt" => Self::PurifyingSalt,
            "vitalspirit" => Self::VitalSpirit,
            "insomnia" => Self::Insomnia,
            "sweetveil" => Self::SweetVeil,
            "oblivious" => Self::Oblivious,
            "owntempo" => Self::OwnTempo,
            "aromaveil" => Self::AromaVeil,
            _ => Self::Other,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Item {
    #[default]
    None,
    Other,
    LifeOrb,
    ChoiceBand,
    ChoiceSpecs,
    ChoiceScarf,
    AssaultVest,
    FocusSash,
    Leftovers,
    BlackSludge,
    SitrusBerry,
    PinchBerry,
    LumBerry,
    SafetyGoggles,
    ClearAmulet,
    CovertCloak,
    RockyHelmet,
    HeavyDutyBoots,
    Eviolite,
    WeaknessPolicy,
    WhiteHerb,
    MentalHerb,
    LoadedDice,
    BoosterEnergy,
    MuscleBand,
    WiseGlasses,
    ExpertBelt,
    IronBall,
    HeatRock,
    DampRock,
    SmoothRock,
    IcyRock,
    LightClay,
    TerrainExtender,
    EjectButton,
    EjectPack,
    TypeBoost(u8),
    MegaStone,
}

impl Item {
    pub fn from_id(id: &str, is_mega_stone: bool) -> Self {
        if id.is_empty() {
            return Self::None;
        }
        if is_mega_stone {
            return Self::MegaStone;
        }
        match id {
            "lifeorb" => Self::LifeOrb,
            "choiceband" => Self::ChoiceBand,
            "choicespecs" => Self::ChoiceSpecs,
            "choicescarf" => Self::ChoiceScarf,
            "assaultvest" => Self::AssaultVest,
            "focussash" => Self::FocusSash,
            "leftovers" => Self::Leftovers,
            "blacksludge" => Self::BlackSludge,
            "sitrusberry" => Self::SitrusBerry,
            "figyberry" | "wikiberry" | "magoberry" | "aguavberry" | "iapapaberry" => Self::PinchBerry,
            "lumberry" => Self::LumBerry,
            "safetygoggles" => Self::SafetyGoggles,
            "clearamulet" => Self::ClearAmulet,
            "covertcloak" => Self::CovertCloak,
            "rockyhelmet" => Self::RockyHelmet,
            "heavydutyboots" => Self::HeavyDutyBoots,
            "eviolite" => Self::Eviolite,
            "weaknesspolicy" => Self::WeaknessPolicy,
            "whiteherb" => Self::WhiteHerb,
            "mentalherb" => Self::MentalHerb,
            "loadeddice" => Self::LoadedDice,
            "boosterenergy" => Self::BoosterEnergy,
            "muscleband" => Self::MuscleBand,
            "wiseglasses" => Self::WiseGlasses,
            "expertbelt" => Self::ExpertBelt,
            "ironball" => Self::IronBall,
            "heatrock" => Self::HeatRock,
            "damprock" => Self::DampRock,
            "smoothrock" => Self::SmoothRock,
            "icyrock" => Self::IcyRock,
            "lightclay" => Self::LightClay,
            "terrainextender" => Self::TerrainExtender,
            "ejectbutton" => Self::EjectButton,
            "ejectpack" => Self::EjectPack,
            "silkscarf" => Self::TypeBoost(TYPE_NORMAL),
            "charcoal" | "flameplate" => Self::TypeBoost(TYPE_FIRE),
            "mysticwater" | "splashplate" | "waveincense" | "seaincense" => Self::TypeBoost(TYPE_WATER),
            "magnet" | "zapplate" => Self::TypeBoost(TYPE_ELECTRIC),
            "miracleseed" | "meadowplate" | "roseincense" => Self::TypeBoost(TYPE_GRASS),
            "nevermeltice" | "icicleplate" => Self::TypeBoost(TYPE_ICE),
            "blackbelt" | "fistplate" => Self::TypeBoost(TYPE_FIGHTING),
            "poisonbarb" | "toxicplate" => Self::TypeBoost(TYPE_POISON),
            "softsand" | "earthplate" => Self::TypeBoost(TYPE_GROUND),
            "sharpbeak" | "skyplate" => Self::TypeBoost(TYPE_FLYING),
            "twistedspoon" | "mindplate" | "oddincense" => Self::TypeBoost(TYPE_PSYCHIC),
            "silverpowder" | "insectplate" => Self::TypeBoost(TYPE_BUG),
            "hardstone" | "stoneplate" | "rockincense" => Self::TypeBoost(TYPE_ROCK),
            "spelltag" | "spookyplate" => Self::TypeBoost(TYPE_GHOST),
            "dragonfang" | "dracoplate" => Self::TypeBoost(TYPE_DRAGON),
            "blackglasses" | "dreadplate" => Self::TypeBoost(TYPE_DARK),
            "metalcoat" | "ironplate" => Self::TypeBoost(TYPE_STEEL),
            "fairyfeather" | "pixieplate" => Self::TypeBoost(TYPE_FAIRY),
            _ => Self::Other,
        }
    }

    pub fn is_choice(&self) -> bool {
        matches!(self, Self::ChoiceBand | Self::ChoiceSpecs | Self::ChoiceScarf)
    }
}

pub const FLAG_CONTACT: u16 = 1 << 0;
pub const FLAG_PROTECT: u16 = 1 << 1;
pub const FLAG_SOUND: u16 = 1 << 2;
pub const FLAG_PUNCH: u16 = 1 << 3;
pub const FLAG_BITE: u16 = 1 << 4;
pub const FLAG_PULSE: u16 = 1 << 5;
pub const FLAG_SLICING: u16 = 1 << 6;
pub const FLAG_POWDER: u16 = 1 << 7;
pub const FLAG_HEAL: u16 = 1 << 8;
pub const FLAG_BYPASS_SUB: u16 = 1 << 9;

#[derive(Clone, Debug)]
pub struct SpeciesData {
    pub id: String,
    pub types: [u8; 2],
    pub base_stats: [u16; 6], // hp, atk, def, spa, spd, spe
    pub weight_hg: u32,
    pub primary_ability: Ability,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Boosts6 {
    pub vals: [i8; 6], // atk, def, spa, spd, spe, acc
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SecondaryData {
    pub chance: u8,
    pub status: u8,
    pub volatile: u8,
    pub boosts: Boosts6,
    pub self_boost: Boosts6,
    pub has_boosts: bool,
    pub has_self_boost: bool,
}

#[derive(Clone, Debug)]
pub struct MoveData {
    pub id: String,
    pub move_type: u8,
    pub category: Category,
    pub base_power: u16,
    pub accuracy: u8, // 0 = always hits
    pub priority: i8,
    pub target: TargetKind,
    pub flags: u16,
    pub status: u8,
    pub volatile: u8,
    pub side_condition: u8,
    pub pseudo_weather: u8,
    pub weather: u8,
    pub terrain: u8,
    pub boosts: Boosts6,
    pub has_boosts: bool,
    pub self_boost: Boosts6,
    pub has_self_boost: bool,
    pub self_switch: bool,
    pub secondary: Option<SecondaryData>,
    pub drain_num: u8,
    pub drain_den: u8,
    pub recoil_num: u8,
    pub recoil_den: u8,
    pub heal_num: u8,
    pub heal_den: u8,
    pub min_hits: u8,
    pub max_hits: u8,
    pub crit_ratio: u8,
    pub will_crit: bool,
    pub ignore_defensive: bool,
    pub override_off_stat: u8, // 0=none, 1=def (Body Press)
    pub override_def_stat: u8, // 0=none, 1=def (Psyshock)
    pub breaks_protect: bool,
    pub is_protect: bool,
    pub is_fake_out: bool,
    pub is_helping_hand: bool,
    pub is_follow_me: bool,
    pub is_rage_powder: bool,
    pub is_wide_guard: bool,
    pub is_quick_guard: bool,
    pub is_ally_target_allowed: bool,
    pub is_tactical: bool,
}

pub const STATUS_NONE: u8 = 0;
pub const STATUS_BRN: u8 = 1;
pub const STATUS_PAR: u8 = 2;
pub const STATUS_SLP: u8 = 3;
pub const STATUS_FRZ: u8 = 4;
pub const STATUS_PSN: u8 = 5;
pub const STATUS_TOX: u8 = 6;

pub fn parse_status(s: &str) -> u8 {
    match s {
        "brn" => STATUS_BRN,
        "par" => STATUS_PAR,
        "slp" => STATUS_SLP,
        "frz" => STATUS_FRZ,
        "psn" => STATUS_PSN,
        "tox" => STATUS_TOX,
        _ => STATUS_NONE,
    }
}

pub const WEATHER_NONE: u8 = 0;
pub const WEATHER_SUN: u8 = 1;
pub const WEATHER_RAIN: u8 = 2;
pub const WEATHER_SAND: u8 = 3;
pub const WEATHER_SNOW: u8 = 4;

pub fn parse_weather(s: &str) -> u8 {
    match s.to_ascii_lowercase().as_str() {
        "sunnyday" | "sun" | "desolateland" => WEATHER_SUN,
        "raindance" | "rain" | "primordialsea" => WEATHER_RAIN,
        "sandstorm" | "sand" => WEATHER_SAND,
        "snowscape" | "snow" | "hail" => WEATHER_SNOW,
        _ => WEATHER_NONE,
    }
}

pub const TERRAIN_NONE: u8 = 0;
pub const TERRAIN_ELECTRIC: u8 = 1;
pub const TERRAIN_GRASSY: u8 = 2;
pub const TERRAIN_PSYCHIC: u8 = 3;
pub const TERRAIN_MISTY: u8 = 4;

pub fn parse_terrain(s: &str) -> u8 {
    match s.to_ascii_lowercase().as_str() {
        "electricterrain" => TERRAIN_ELECTRIC,
        "grassyterrain" => TERRAIN_GRASSY,
        "psychicterrain" => TERRAIN_PSYCHIC,
        "mistyterrain" => TERRAIN_MISTY,
        _ => TERRAIN_NONE,
    }
}

pub const SC_NONE: u8 = 0;
pub const SC_REFLECT: u8 = 1;
pub const SC_LIGHTSCREEN: u8 = 2;
pub const SC_AURORAVEIL: u8 = 3;
pub const SC_TAILWIND: u8 = 4;
pub const SC_SAFEGUARD: u8 = 5;
pub const SC_MIST: u8 = 6;
pub const SC_STEALTHROCK: u8 = 7;
pub const SC_SPIKES: u8 = 8;
pub const SC_TOXICSPIKES: u8 = 9;
pub const SC_STICKYWEB: u8 = 10;
pub const SC_WIDEGUARD: u8 = 11;
pub const SC_QUICKGUARD: u8 = 12;

pub const PW_NONE: u8 = 0;
pub const PW_TRICKROOM: u8 = 1;
pub const PW_GRAVITY: u8 = 2;

pub const VOL_NONE: u8 = 0;
pub const VOL_CONFUSION: u8 = 1;
pub const VOL_FLINCH: u8 = 2;
pub const VOL_SUBSTITUTE: u8 = 3;
pub const VOL_TAUNT: u8 = 4;
pub const VOL_ENCORE: u8 = 5;
pub const VOL_YAWN: u8 = 6;
pub const VOL_LEECHSEED: u8 = 7;
pub const VOL_PERISH: u8 = 8;
pub const VOL_DISABLE: u8 = 9;
pub const VOL_PROTECT: u8 = 10;
pub const VOL_ENDURE: u8 = 11;

pub struct Dex {
    pub species: Vec<SpeciesData>,
    pub species_by_id: HashMap<String, u16>,
    pub moves: Vec<MoveData>,
    pub move_by_id: HashMap<String, u16>,
    pub mega_map: HashMap<(u16, String), u16>, // (base_species_id, item_id) -> mega_species_id
    pub mega_stones: HashMap<String, bool>,
}

#[derive(Deserialize)]
struct RawDex {
    species: HashMap<String, RawSpecies>,
    moves: HashMap<String, RawMove>,
    items: HashMap<String, RawItem>,
    #[serde(rename = "megaMap")]
    mega_map: HashMap<String, String>,
}

#[derive(Deserialize)]
struct RawSpecies {
    types: Vec<String>,
    #[serde(rename = "baseStats")]
    base_stats: RawStats,
    weightkg: Option<f32>,
    abilities: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct RawStats {
    hp: u16,
    atk: u16,
    def: u16,
    spa: u16,
    spd: u16,
    spe: u16,
}

#[derive(Deserialize)]
struct RawBoosts {
    atk: Option<i8>,
    def: Option<i8>,
    spa: Option<i8>,
    spd: Option<i8>,
    spe: Option<i8>,
    accuracy: Option<i8>,
}

#[derive(Deserialize)]
struct RawSecondary {
    chance: Option<u8>,
    status: Option<String>,
    #[serde(rename = "volatileStatus")]
    volatile_status: Option<String>,
    boosts: Option<RawBoosts>,
    #[serde(rename = "selfBoost")]
    self_boost: Option<RawBoosts>,
}

#[derive(Deserialize)]
struct RawMove {
    #[serde(rename = "type")]
    move_type: String,
    category: String,
    #[serde(rename = "basePower")]
    base_power: u16,
    accuracy: f32,
    priority: i8,
    target: String,
    flags: Vec<String>,
    status: Option<String>,
    #[serde(rename = "volatileStatus")]
    volatile_status: Option<String>,
    #[serde(rename = "sideCondition")]
    side_condition: Option<String>,
    #[serde(rename = "pseudoWeather")]
    pseudo_weather: Option<String>,
    weather: Option<String>,
    terrain: Option<String>,
    boosts: Option<RawBoosts>,
    #[serde(rename = "selfBoost")]
    self_boost: Option<RawBoosts>,
    #[serde(rename = "selfSwitch")]
    self_switch: Option<bool>,
    secondary: Option<RawSecondary>,
    drain: Option<Vec<u8>>,
    recoil: Option<Vec<u8>>,
    heal: Option<Vec<u8>>,
    multihit: Option<serde_json::Value>,
    #[serde(rename = "critRatio")]
    crit_ratio: Option<u8>,
    #[serde(rename = "willCrit")]
    will_crit: Option<bool>,
    #[serde(rename = "ignoreDefensive")]
    ignore_defensive: Option<bool>,
    #[serde(rename = "overrideOffensiveStat")]
    override_off_stat: Option<String>,
    #[serde(rename = "overrideDefensiveStat")]
    override_def_stat: Option<String>,
    #[serde(rename = "breaksProtect")]
    breaks_protect: Option<bool>,
}

#[derive(Deserialize)]
struct RawItem {
    #[serde(rename = "megaStone")]
    mega_stone: Option<serde_json::Value>,
}

fn parse_boosts(b: Option<&RawBoosts>) -> (Boosts6, bool) {
    let Some(b) = b else {
        return (Boosts6::default(), false);
    };
    let vals = [
        b.atk.unwrap_or(0),
        b.def.unwrap_or(0),
        b.spa.unwrap_or(0),
        b.spd.unwrap_or(0),
        b.spe.unwrap_or(0),
        b.accuracy.unwrap_or(0),
    ];
    let any = vals.iter().any(|&v| v != 0);
    (Boosts6 { vals }, any)
}

fn parse_vol(s: &str) -> u8 {
    match s {
        "confusion" => VOL_CONFUSION,
        "flinch" => VOL_FLINCH,
        "substitute" => VOL_SUBSTITUTE,
        "taunt" => VOL_TAUNT,
        "encore" => VOL_ENCORE,
        "yawn" => VOL_YAWN,
        "leechseed" => VOL_LEECHSEED,
        "perishsong" => VOL_PERISH,
        "disable" => VOL_DISABLE,
        "protect" | "spikyshield" | "kingsshield" | "banefulbunker" | "obstruct"
        | "silktrap" | "burningbulwark" => VOL_PROTECT,
        "endure" => VOL_ENDURE,
        _ => VOL_NONE,
    }
}

impl Dex {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("read dex: {e}"))?;
        let raw: RawDex = serde_json::from_str(&text).map_err(|e| format!("parse dex: {e}"))?;

        let mut species = Vec::with_capacity(raw.species.len() + 1);
        let mut species_by_id = HashMap::with_capacity(raw.species.len());
        // index 0 = placeholder
        species.push(SpeciesData {
            id: String::new(),
            types: [TYPE_NORMAL, TYPE_NONE],
            base_stats: [80; 6],
            weight_hg: 500,
            primary_ability: Ability::Other,
        });
        let mut sp_keys: Vec<String> = raw.species.keys().cloned().collect();
        sp_keys.sort();
        for k in sp_keys {
            let rs = &raw.species[&k];
            let t0 = rs.types.first().map(|s| parse_type(s)).unwrap_or(TYPE_NORMAL);
            let t1 = rs.types.get(1).map(|s| parse_type(s)).unwrap_or(TYPE_NONE);
            let ab = rs
                .abilities
                .as_ref()
                .and_then(|v| v.first())
                .map(|a| Ability::from_id(a))
                .unwrap_or(Ability::Other);
            let idx = species.len() as u16;
            species.push(SpeciesData {
                id: k.clone(),
                types: [t0, t1],
                base_stats: [
                    rs.base_stats.hp,
                    rs.base_stats.atk,
                    rs.base_stats.def,
                    rs.base_stats.spa,
                    rs.base_stats.spd,
                    rs.base_stats.spe,
                ],
                weight_hg: (rs.weightkg.unwrap_or(50.0) * 10.0).round() as u32,
                primary_ability: ab,
            });
            species_by_id.insert(k, idx);
        }

        let mut moves = Vec::with_capacity(raw.moves.len() + 1);
        let mut move_by_id = HashMap::with_capacity(raw.moves.len());
        moves.push(MoveData {
            id: "struggle".into(),
            move_type: TYPE_NONE,
            category: Category::Physical,
            base_power: 50,
            accuracy: 0,
            priority: 0,
            target: TargetKind::RandomNormal,
            flags: FLAG_CONTACT,
            status: STATUS_NONE,
            volatile: VOL_NONE,
            side_condition: SC_NONE,
            pseudo_weather: PW_NONE,
            weather: WEATHER_NONE,
            terrain: TERRAIN_NONE,
            boosts: Boosts6::default(),
            has_boosts: false,
            self_boost: Boosts6::default(),
            has_self_boost: false,
            self_switch: false,
            secondary: None,
            drain_num: 0,
            drain_den: 1,
            recoil_num: 0,
            recoil_den: 1,
            heal_num: 0,
            heal_den: 1,
            min_hits: 1,
            max_hits: 1,
            crit_ratio: 1,
            will_crit: false,
            ignore_defensive: false,
            override_off_stat: 0,
            override_def_stat: 0,
            breaks_protect: false,
            is_protect: false,
            is_fake_out: false,
            is_helping_hand: false,
            is_follow_me: false,
            is_rage_powder: false,
            is_wide_guard: false,
            is_quick_guard: false,
            is_ally_target_allowed: false,
            is_tactical: false,
        });

        let mut mv_keys: Vec<String> = raw.moves.keys().cloned().collect();
        mv_keys.sort();
        for k in mv_keys {
            let rm = &raw.moves[&k];
            let category = match rm.category.as_str() {
                "Physical" => Category::Physical,
                "Special" => Category::Special,
                _ => Category::Status,
            };
            let target = match rm.target.as_str() {
                "normal" => TargetKind::Normal,
                "adjacentFoe" => TargetKind::AdjacentFoe,
                "any" => TargetKind::Any,
                "allAdjacentFoes" => TargetKind::AllAdjacentFoes,
                "allAdjacent" => TargetKind::AllAdjacent,
                "adjacentAlly" => TargetKind::AdjacentAlly,
                "adjacentAllyOrSelf" => TargetKind::AdjacentAllyOrSelf,
                "self" => TargetKind::SelfTarget,
                "allySide" | "allyTeam" => TargetKind::AllySide,
                "foeSide" => TargetKind::FoeSide,
                "all" => TargetKind::All,
                "randomNormal" => TargetKind::RandomNormal,
                _ => TargetKind::Scripted,
            };
            let mut flags = 0u16;
            for f in &rm.flags {
                match f.as_str() {
                    "contact" => flags |= FLAG_CONTACT,
                    "protect" => flags |= FLAG_PROTECT,
                    "sound" => flags |= FLAG_SOUND,
                    "punch" => flags |= FLAG_PUNCH,
                    "bite" => flags |= FLAG_BITE,
                    "pulse" => flags |= FLAG_PULSE,
                    "slicing" => flags |= FLAG_SLICING,
                    "powder" => flags |= FLAG_POWDER,
                    "heal" => flags |= FLAG_HEAL,
                    "bypasssub" => flags |= FLAG_BYPASS_SUB,
                    _ => {}
                }
            }
            let sc = match rm.side_condition.as_deref().unwrap_or("") {
                "reflect" => SC_REFLECT,
                "lightscreen" => SC_LIGHTSCREEN,
                "auroraveil" => SC_AURORAVEIL,
                "tailwind" => SC_TAILWIND,
                "safeguard" => SC_SAFEGUARD,
                "mist" => SC_MIST,
                "stealthrock" => SC_STEALTHROCK,
                "spikes" => SC_SPIKES,
                "toxicspikes" => SC_TOXICSPIKES,
                "stickyweb" => SC_STICKYWEB,
                "wideguard" => SC_WIDEGUARD,
                "quickguard" => SC_QUICKGUARD,
                _ => SC_NONE,
            };
            let pw = match rm.pseudo_weather.as_deref().unwrap_or("") {
                "trickroom" => PW_TRICKROOM,
                "gravity" => PW_GRAVITY,
                _ => PW_NONE,
            };
            let (boosts, has_boosts) = parse_boosts(rm.boosts.as_ref());
            let (self_boost, has_self_boost) = parse_boosts(rm.self_boost.as_ref());
            let secondary = rm.secondary.as_ref().map(|sec| {
                let (sb, hsb) = parse_boosts(sec.boosts.as_ref());
                let (ssb, hssb) = parse_boosts(sec.self_boost.as_ref());
                SecondaryData {
                    chance: sec.chance.unwrap_or(100),
                    status: parse_status(sec.status.as_deref().unwrap_or("")),
                    volatile: parse_vol(sec.volatile_status.as_deref().unwrap_or("")),
                    boosts: sb,
                    self_boost: ssb,
                    has_boosts: hsb,
                    has_self_boost: hssb,
                }
            });
            let (min_hits, max_hits) = match &rm.multihit {
                Some(serde_json::Value::Number(n)) => {
                    let h = n.as_u64().unwrap_or(1) as u8;
                    (h, h)
                }
                Some(serde_json::Value::Array(arr)) if arr.len() == 2 => (
                    arr[0].as_u64().unwrap_or(2) as u8,
                    arr[1].as_u64().unwrap_or(5) as u8,
                ),
                _ => (1, 1),
            };
            let (drain_num, drain_den) = rm
                .drain
                .as_ref()
                .filter(|v| v.len() == 2)
                .map(|v| (v[0], v[1].max(1)))
                .unwrap_or((0, 1));
            let (recoil_num, recoil_den) = rm
                .recoil
                .as_ref()
                .filter(|v| v.len() == 2)
                .map(|v| (v[0], v[1].max(1)))
                .unwrap_or((0, 1));
            let (heal_num, heal_den) = rm
                .heal
                .as_ref()
                .filter(|v| v.len() == 2)
                .map(|v| (v[0], v[1].max(1)))
                .unwrap_or((0, 1));

            let is_protect = matches!(
                k.as_str(),
                "protect"
                    | "detect"
                    | "spikyshield"
                    | "kingsshield"
                    | "banefulbunker"
                    | "obstruct"
                    | "silktrap"
                    | "burningbulwark"
            );
            let is_ally_target_allowed = matches!(
                k.as_str(),
                "pollenpuff"
                    | "healpulse"
                    | "floralhealing"
                    | "skillswap"
                    | "decorate"
                    | "afteryou"
                    | "instruct"
                    | "beatup"
                    | "acupressure"
                    | "coaching"
                    | "quash"
            );
            let is_tactical = is_protect
                || matches!(
                    k.as_str(),
                    "fakeout"
                        | "wideguard"
                        | "quickguard"
                        | "followme"
                        | "ragepowder"
                        | "tailwind"
                        | "trickroom"
                        | "helpinghand"
                        | "icywind"
                        | "electroweb"
                        | "snarl"
                        | "partingshot"
                        | "spore"
                        | "willowisp"
                        | "thunderwave"
                        | "taunt"
                        | "encore"
                        | "coaching"
                        | "substitute"
                        | "swordsdance"
                        | "nastyplot"
                        | "calmmind"
                        | "dragondance"
                        | "quiverdance"
                        | "bulkup"
                );

            let idx = moves.len() as u16;
            moves.push(MoveData {
                id: k.clone(),
                move_type: parse_type(&rm.move_type),
                category,
                base_power: rm.base_power,
                accuracy: if rm.accuracy <= 0.0 { 0 } else { rm.accuracy.round() as u8 },
                priority: rm.priority,
                target,
                flags,
                status: parse_status(rm.status.as_deref().unwrap_or("")),
                volatile: parse_vol(rm.volatile_status.as_deref().unwrap_or("")),
                side_condition: sc,
                pseudo_weather: pw,
                weather: parse_weather(rm.weather.as_deref().unwrap_or("")),
                terrain: parse_terrain(rm.terrain.as_deref().unwrap_or("")),
                boosts,
                has_boosts,
                self_boost,
                has_self_boost,
                self_switch: rm.self_switch.unwrap_or(false),
                secondary,
                drain_num,
                drain_den,
                recoil_num,
                recoil_den,
                heal_num,
                heal_den,
                min_hits,
                max_hits,
                crit_ratio: rm.crit_ratio.unwrap_or(1),
                will_crit: rm.will_crit.unwrap_or(false),
                ignore_defensive: rm.ignore_defensive.unwrap_or(false),
                override_off_stat: if rm.override_off_stat.as_deref() == Some("def") { 1 } else { 0 },
                override_def_stat: if rm.override_def_stat.as_deref() == Some("def") { 1 } else { 0 },
                breaks_protect: rm.breaks_protect.unwrap_or(false),
                is_protect,
                is_fake_out: k == "fakeout",
                is_helping_hand: k == "helpinghand",
                is_follow_me: k == "followme",
                is_rage_powder: k == "ragepowder",
                is_wide_guard: k == "wideguard",
                is_quick_guard: k == "quickguard",
                is_ally_target_allowed,
                is_tactical,
            });
            move_by_id.insert(k, idx);
        }

        let mut mega_stones = HashMap::new();
        for (id, it) in &raw.items {
            if it.mega_stone.is_some() {
                mega_stones.insert(id.clone(), true);
            }
        }
        let mut mega_map = HashMap::new();
        for (pair, target_sp) in &raw.mega_map {
            if let Some((base_sp, item_id)) = pair.split_once(':') {
                if let (Some(&b_idx), Some(&t_idx)) =
                    (species_by_id.get(base_sp), species_by_id.get(target_sp))
                {
                    mega_map.insert((b_idx, item_id.to_string()), t_idx);
                    mega_stones.insert(item_id.to_string(), true);
                }
            }
        }

        Ok(Self {
            species,
            species_by_id,
            moves,
            move_by_id,
            mega_map,
            mega_stones,
        })
    }

    pub fn species_id(&self, name: &str) -> u16 {
        let id = to_id(name);
        *self.species_by_id.get(&id).unwrap_or(&0)
    }

    pub fn move_id(&self, name: &str) -> u16 {
        let id = to_id(name);
        *self.move_by_id.get(&id).unwrap_or(&0)
    }
}
