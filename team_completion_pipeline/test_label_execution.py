import copy
import unittest
from label_remaining import same_battle_semantics


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(configuration={'concurrency':24,'threads':1,'encoder':'fast'},
                         policy_sha256='policy',simulator_sha256='sim',formats_sha256='format',
                         bench_source_sha256='source',bench_data_sha256='data',packages={'torch':'version'},
                         workers=6,implementation={'fast_battle_runner.py':'old','worker.py':'worker','encoder.py':'encoder'})

    def test_only_scheduling_change_is_compatible(self):
        changed=copy.deepcopy(self.base)
        changed.update(workers=8,request_chunk_size=96)
        changed['implementation']['fast_battle_runner.py']='new'
        self.assertTrue(same_battle_semantics(self.base,changed))

    def test_policy_encoder_runtime_and_numerics_cannot_silently_change(self):
        for key in ['configuration','policy_sha256','simulator_sha256','formats_sha256',
                    'bench_source_sha256','bench_data_sha256','packages']:
            changed=copy.deepcopy(self.base);changed[key]='changed'
            with self.subTest(key=key):self.assertFalse(same_battle_semantics(self.base,changed))
        changed=copy.deepcopy(self.base);changed['implementation']['encoder.py']='changed'
        self.assertFalse(same_battle_semantics(self.base,changed))
        del changed['packages']
        self.assertFalse(same_battle_semantics(self.base,changed))


if __name__=='__main__':unittest.main()
