import asyncio
import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import analyze, workload


class AnalysisTests(unittest.TestCase):
    def test_does_not_count_incomplete_blocks(self):
        rows = [{'phase':'measure','treatment':'x','block':0,'variant':'x','completed':10,'turns':50,'seconds':1}]
        self.assertEqual(analyze(rows), {})

    def test_uses_block_level_throughput_and_excludes_warmups(self):
        rows = []
        for block in range(5):
            for name, secs in [('baseline',2),('x',1),('x',1),('baseline',2)]:
                rows.append({'phase':'measure','treatment':'x','block':block,'variant':name,
                             'completed':10,'turns':50,'seconds':secs})
        rows.append({**rows[0], 'phase':'warmup','seconds':1000})
        result = analyze(rows)['x']
        self.assertEqual(result['blocks'],5)
        self.assertAlmostEqual(result['geometric_mean_speedup'],2)
        self.assertEqual(result['bootstrap95'],[2,2])

    def test_workload_pairs_have_fixed_candidate_and_expected_size(self):
        work = workload(57)
        self.assertEqual(len(work['pairs']),57)
        self.assertEqual(len({p['a_id'] for p in work['pairs']}),1)
        self.assertGreater(len({p['b_id'] for p in work['pairs']}),1)

    def test_clock_pause_invalidates_entire_paired_block(self):
        rows=[]
        for block in range(2):
            for name,secs in [('baseline',2),('x',1),('x',1),('baseline',2)]:
                rows.append({'phase':'measure','treatment':'x','block':block,'variant':name,
                             'completed':10,'turns':50,'seconds':secs})
        rows[1]['clock']={'pause_or_clock_jump':True}
        result=analyze(rows)['x']
        self.assertEqual(result['blocks'],1)
        self.assertEqual(result['invalid_clock_blocks'],[0])
        self.assertAlmostEqual(result['geometric_mean_speedup'],2)

    def test_incomplete_block_is_excluded_from_accepted_ids_and_trace_status(self):
        rows=[{'phase':'measure','treatment':'x','block':0,'variant':name,
               'completed':10,'turns':50,'seconds':secs,'trace_equivalent':True}
              for name,secs in [('baseline',2),('x',1),('x',1),('baseline',2)]]
        rows.append({**rows[0],'block':1,'trace_equivalent':False})
        result=analyze(rows)['x']
        self.assertEqual(result['accepted_block_ids'],[0])
        self.assertTrue(result['trace_equivalent'])


if __name__ == '__main__':
    unittest.main()
