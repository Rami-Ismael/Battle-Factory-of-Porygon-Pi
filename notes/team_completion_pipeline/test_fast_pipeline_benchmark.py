import unittest
from benchmark_fast_pipeline import analyze


def block(index, baseline_seconds):
    return [{'block':index,'position':i,'variant':variant,'seconds':baseline_seconds/(2 if variant=='fast' else 1),
             'completed':100,'clock':{'pause_or_clock_jump':False}}
            for i,variant in enumerate(['baseline','fast','fast','baseline'])]


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_paired_ratio_survives_large_between_block_load_changes(self):
        trials=[row for i,seconds in enumerate([1,10,2,20,5]) for row in block(i,seconds)]
        result=analyze(trials)
        self.assertAlmostEqual(result['geometric_mean_speedup'],2)
        self.assertEqual(result['bootstrap95'],[2,2])
        self.assertEqual(result['accepted_blocks'],list(range(5)))

    def test_clock_gap_rejects_entire_block_and_partial_block_is_not_used(self):
        trials=block(0,2)+block(1,20)+block(2,1)[:3]
        trials[4]['clock']['pause_or_clock_jump']=True
        result=analyze(trials)
        self.assertEqual(result['accepted_blocks'],[0])
        self.assertEqual(result['invalid_clock_blocks'],[1])
        self.assertIsNone(result['bootstrap95'])


if __name__=='__main__':unittest.main()
