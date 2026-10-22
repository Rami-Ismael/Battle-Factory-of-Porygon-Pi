import unittest
import numpy as np
from allocation import replay,paired_variance


class AllocationTests(unittest.TestCase):
    def test_budget_and_minimum_coverage(self):
        data=np.random.default_rng(9).integers(0,2,size=(4,3,32))
        for method in ['uniform','race','stratified']:
            chosen,used,counts=replay(data,120,method,8)
            self.assertIn(chosen,range(4))
            self.assertEqual(used,int(counts.sum()))
            self.assertLessEqual(used,120)
            self.assertTrue((counts>=2).all())
            self.assertTrue((counts<=32).all())

    def test_racing_keeps_a_dominant_candidate(self):
        data=np.zeros((4,3,32));data[2]=1
        chosen,used,counts=replay(data,120,'race',4)
        self.assertEqual(chosen,2)
        self.assertGreater(counts[2].sum(),counts[0].sum())

    def test_uniform_full_budget_recovers_equal_weight_means(self):
        data=np.zeros((3,2,8));data[1,0]=1;data[2,:,:6]=1
        chosen,used,counts=replay(data,data.size,'uniform',12)
        self.assertEqual(chosen,2)
        self.assertEqual(used,data.size)
        self.assertTrue((counts==8).all())

    def test_common_randomness_can_help_or_hurt(self):
        data=np.array([[[0,1,0,1]],[[0,1,0,1]],[[1,0,1,0]]],dtype=float)
        result=paired_variance(data)
        self.assertEqual(result[0]['paired_variance_of_mean_difference'],0)
        self.assertLess(result[1]['estimated_independent_over_paired'],1)

    def test_surrogate_counts_pilot_and_audits_in_budget(self):
        data=np.random.default_rng(16).integers(0,2,size=(6,3,32))
        features=np.eye(6)
        chosen,used,counts=replay(data,240,'surrogate',13,features)
        self.assertEqual(used,240)
        self.assertEqual(used,int(counts.sum()))
        self.assertTrue((counts>=2).all())
        self.assertIn(chosen,range(6))


if __name__=='__main__':unittest.main()
