"""Exact observation optimization; retains upstream feature order and values.

Sparse status/effect assembly, constant-time vocabulary lookup, and bounded caching
of move features keyed by every property used by the reference encoder.
"""
from functools import lru_cache
import numpy as np
from vgc_bench.src import policy_player as ref


def first_indices(values):
    result = {}
    for i, value in enumerate(values):
        result.setdefault(value,i)
    return result


ABILITY = first_indices(ref.abilities)
ITEM = first_indices(ref.items)
MOVE = first_indices(ref.moves)
EFFECT = {v:i for i,v in enumerate(ref.Effect)}


def hot_table(enum):
    values = list(enum)
    return {**{v:np.eye(len(values),dtype=np.float32)[i] for i,v in enumerate(values)},
            None:np.zeros(len(values),dtype=np.float32)}


TYPE = hot_table(ref.PokemonType)
GENDER = hot_table(ref.PokemonGender)
STATUS = hot_table(ref.Status)
POSITION = np.eye(6,dtype=np.float32)


@lru_cache(maxsize=8192)
def move_from_key(key):
    power,acc,category,target,priority,crit,drain,force,recoil,destruct,switch,maxpp,currentpp,last,typ = key
    result = np.array([power/250,acc/100,
        *[float(c==category) for c in ref.MoveCategory],
        *[float(t==target) for t in ref.Target],
        (priority+7)/12,crit,drain,float(force),recoil,float(destruct),float(switch),maxpp/64,
        currentpp/maxpp,float(last),*[float(t==typ) for t in ref.PokemonType]])
    result.flags.writeable = False
    return result


def embed_move(move):
    return move_from_key((move.base_power,move.accuracy,move.category,move.target,move.priority,
        move.crit_ratio,move.drain,move.force_switch,move.recoil,move.self_destruct is not None,
        move.self_switch is not False,move.max_pp,move.current_pp,move.is_last_used,move.type))


def embed_pokemon(pokemon,pos,from_opponent,active_a,active_b):
    assert from_opponent or not pokemon.revealed or pokemon.selected_in_teampreview
    move_list = list(pokemon.moves.values())[-4:]
    ids = [MOVE[m.id] for m in move_list]+[0]*(4-len(move_list))
    move_arrays = [embed_move(m) for m in move_list]
    move_arrays += [np.zeros(ref.move_obs_len,dtype=np.float32)]*(4-len(move_list))
    types = np.zeros(len(TYPE[None]),dtype=np.float32)
    for typ in pokemon.base_types:
        if typ in TYPE:
            types = np.maximum(types,TYPE[typ])
    effects = np.zeros(len(EFFECT),dtype=np.float64)
    for effect,value in pokemon.effects.items():
        if effect in EFFECT:
            effects[EFFECT[effect]] = min(value,8)/8
    stats = [-1]*6 if from_opponent else [s/255 for s in pokemon.stats.values() if s is not None]
    assert len(stats)==6
    return np.concatenate([
        [ABILITY['null' if pokemon.ability is None else pokemon.ability],
         ITEM['null' if pokemon.item is None else pokemon.item],*ids],
        *move_arrays, types, TYPE.get(pokemon.tera_type,TYPE[None]),
        [s/255 for s in pokemon.base_stats.values()],stats,GENDER.get(pokemon.gender,GENDER[None]),
        [pokemon.weight/1000,pokemon.current_hp_fraction,float(pokemon.revealed),float(pokemon.selected_in_teampreview)],
        STATUS.get(pokemon.status,STATUS[None]),[pokemon.status_counter/16],
        [b/6 for b in pokemon.boosts.values()],effects,
        [float(pokemon.first_turn),pokemon.protect_counter/5,float(pokemon.must_recharge),float(pokemon.preparing),
         float(pokemon.is_dynamaxed),float(pokemon.is_terastallized),float(active_a),float(active_b)],
        POSITION[pos] if 0<=pos<6 else np.zeros(6),[float(from_opponent)]
    ],dtype=np.float32)


def embed_battle(battle,fake_rating=None):
    glob = ref.PolicyPlayer.embed_global(battle)
    sides = [ref.PolicyPlayer.embed_side(battle,fake_rating),
             ref.PolicyPlayer.embed_side(battle,None if fake_rating is None else 0,opp=True)]
    teams = [battle.team,battle.opponent_team]
    actives = [battle.active_pokemon,battle.opponent_active_pokemon]
    width = len(glob)+len(sides[0])+ref.pokemon_obs_len
    result = np.zeros((12,width),dtype=np.float32)
    for opp in (0,1):
        result[opp*6:(opp+1)*6,:len(glob)] = glob
        result[opp*6:(opp+1)*6,len(glob):len(glob)+len(sides[opp])] = sides[opp]
        a,b = actives[opp]
        for i,p in enumerate(teams[opp].values()):
            result[opp*6+i,len(glob)+len(sides[opp]):] = embed_pokemon(p,i,bool(opp),
                a is not None and p.name==a.name,b is not None and p.name==b.name)
    return result.ravel()
