"""Reference equality over varied volatile and static Pokemon features."""
import os
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BENCH = Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(BENCH))
_cwd = os.getcwd()
os.chdir(BENCH)
import encoder
from vgc_bench.src import policy_player as ref
os.chdir(_cwd)
import numpy as np


class EncoderTests(unittest.TestCase):
    def test_move_cache_includes_pp_and_last_used(self):
        move = ref.Move('protect',gen=9)
        expected = ref.PolicyPlayer.embed_move(move)
        np.testing.assert_array_equal(encoder.embed_move(move),expected)
        old = encoder.embed_move(move).copy()
        move._current_pp = max(0,move.current_pp-1)
        np.testing.assert_array_equal(encoder.embed_move(move),ref.PolicyPlayer.embed_move(move))
        self.assertFalse(np.array_equal(old,encoder.embed_move(move)))

    def test_pokemon_reference_values_across_varied_states(self):
        rng = random.Random(22)
        for i in range(100):
            moves = {m:ref.Move(m,gen=9) for m in rng.sample(['protect','tackle','gigadrain','uturn','explosion','swordsdance'],rng.randrange(5))}
            p = SimpleNamespace(ability=rng.choice(['intimidate','chlorophyll',None]),item=rng.choice(['leftovers','lifeorb',None]),
                moves=moves,base_types=tuple(rng.sample(list(ref.PokemonType),2)),tera_type=rng.choice([None,*ref.PokemonType]),
                base_stats=dict(zip(['hp','atk','def','spa','spd','spe'],[rng.randrange(1,256) for _ in range(6)])),
                stats=dict(zip(['hp','atk','def','spa','spd','spe'],[rng.randrange(1,500) for _ in range(6)])),
                gender=rng.choice([None,*ref.PokemonGender]),weight=rng.random()*1000,current_hp_fraction=rng.random(),
                revealed=True,selected_in_teampreview=True,status=rng.choice([None,*ref.Status]),status_counter=rng.randrange(16),
                boosts={'atk':rng.randrange(-6,7),'def':rng.randrange(-6,7),'spa':0,'spd':0,'spe':0,'accuracy':0,'evasion':0},
                effects={e:rng.randrange(12) for e in rng.sample(list(ref.Effect),4)},first_turn=bool(i%2),protect_counter=rng.randrange(5),
                must_recharge=bool(i%3),preparing=bool(i%4),is_dynamaxed=bool(i%5),is_terastallized=bool(i%6))
            args = (p,i%6,bool(i%2),bool(i%3),bool(i%4))
            np.testing.assert_array_equal(encoder.embed_pokemon(*args),ref.PolicyPlayer.embed_pokemon(*args))


if __name__ == '__main__':
    unittest.main()
