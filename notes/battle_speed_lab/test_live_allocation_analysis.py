import json
import tempfile
import unittest
from pathlib import Path
from analyze_live_allocation import analyze


class AccountingTests(unittest.TestCase):
    def fixture(self,folder,clock=False,wrong=False):
        rows=[];entries=[]
        (folder/'metadata.json').write_text(json.dumps({'candidate_ids':[1],'opponent_ids':[2]}))
        for i,method in enumerate(['uniform','race','race','uniform']):
            rows.append({'trial':i,'block':0,'position':i,'treatment':'race','method':method,
                'clock':{'pause_or_clock_jump':clock and i==0},'counts':[[2]],
                'eligible_battles':2,'physical_battles':3+(wrong and i==0),'late_battles':1,
                'pilot_complete':True,'selected_candidate':i,'heldout_score':.7 if method=='race' else .5})
            entries.extend({'trial':i,'candidate':1,'opponent':2,'eligible_before_deadline':j<2} for j in range(3))
        for name,data in [('trials',rows),('exploration',entries)]:
            (folder/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in data))

    def test_late_results_are_accounted_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p);result=analyze(p)
            self.assertEqual(result['accounting_failures'],[])
            self.assertEqual(result['physical_battles'],12)
            self.assertEqual(result['eligible_battles'],8)
            self.assertAlmostEqual(result['comparisons']['race']['selected_holdout_score_difference']['mean'],.2)

    def test_clock_gap_excludes_whole_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p,clock=True);result=analyze(p)
            self.assertEqual(result['comparisons']['race']['accepted_blocks'],[])
            self.assertIsNone(result['comparisons']['race']['selected_holdout_score_difference'])

    def test_missing_physical_battle_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p,wrong=True)
            self.assertTrue(analyze(p)['accounting_failures'])


if __name__=='__main__':unittest.main()
