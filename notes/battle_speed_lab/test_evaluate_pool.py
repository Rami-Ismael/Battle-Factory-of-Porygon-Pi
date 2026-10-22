import unittest
from evaluate_pool import candidate_rows,make_jobs


class PoolJobTests(unittest.TestCase):
    def test_chunking_preserves_exact_candidate_workload(self):
        ids=[i for i,_ in candidate_rows()[:2]]
        jobs=make_jobs(ids,49,24,120)
        self.assertEqual([j['job_id'] for j in jobs],list(range(6)))
        self.assertEqual(len({j['seed'] for j in jobs}),6)
        for identity in ids:
            selected=[p for j in jobs if j['candidate_id']==identity for p in j['pairs']]
            self.assertEqual(len(selected),49)
            self.assertEqual([p['id'] for p in selected],list(range(49)))
            self.assertTrue(all(p['a_id']==identity and p['b_id']!=identity for p in selected))
    def test_unknown_and_duplicate_candidates_rejected(self):
        identity=candidate_rows()[0][0]
        with self.assertRaises(ValueError):make_jobs([identity,identity],10,4,12)
        with self.assertRaises(ValueError):make_jobs([-1],10,4,12)


if __name__=='__main__':unittest.main()
