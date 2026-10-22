import unittest
import numpy as np
from array_kernel import simulate,scalar,eligible


class KernelTests(unittest.TestCase):
    def test_vectorization_matches_scalar_damage_sequence(self):
        rng=np.random.default_rng(871)
        hp=rng.integers(1,401,size=(100,4))
        speeds=np.stack([rng.permutation([70,80,90,100]) for _ in range(100)])
        a=simulate(hp,speeds,True);b=scalar(hp,speeds,True)
        self.assertEqual(a['events'],b['events'])
        for key in ['hp','turns','winner']:np.testing.assert_array_equal(a[key],b[key])
    def test_speed_ties_fail_closed(self):
        with self.assertRaises(ValueError):simulate([[100]*4],[[70]*4])
    def test_unsupported_teams_fail_closed(self):
        pokemon={'species':'Chansey','ability':'Natural Cure','item':'','level':50,'moves':['Seismic Toss']}
        self.assertTrue(eligible([pokemon,pokemon]))
        self.assertFalse(eligible([pokemon]*6))
        self.assertFalse(eligible([pokemon,{**pokemon,'item':'Leftovers'}]))
        self.assertFalse(eligible([pokemon,{**pokemon,'moves':['Metronome']}]))


if __name__=='__main__':unittest.main()
