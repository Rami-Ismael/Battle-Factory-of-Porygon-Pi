"""Opt-in real six-worker tests: VGC_LIVE_TESTS=1 python -m unittest ..."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import mask_lab_battle as pipeline
from fast_battle_runner import FastBattleRunner
from battle_integration_fixture import create_fixture


@unittest.skipUnless(os.environ.get('VGC_LIVE_TESTS')=='1','Set VGC_LIVE_TESTS=1 for real BC battles')
class LiveBattleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='vgc-integration-')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.rows,self.cols=create_fixture(self.folder)

    def test_six_persistent_workers_topup_cache_and_cleanup(self):
        con=pipeline.MDB.connect(self.folder/'matrix.sqlite');self.addCleanup(con.close)
        with FastBattleRunner(self.folder/'runner') as runner:
            policy=runner.policy_id(con,pipeline.MDB)
            self.assertEqual(pipeline.MDB.ensure(con,policy,self.rows,self.cols,1,seed=101,runner=runner),24)
            pids=[w.proc.pid for w in runner.workers]
            self.assertEqual(len(pids),6)
            self.assertEqual(runner.config['concurrency'],24)
            self.assertEqual(pipeline.MDB.ensure(con,policy,self.rows,self.cols,1,seed=101,runner=runner),0)
            self.assertEqual(pipeline.MDB.ensure(con,policy,self.rows,self.cols,2,seed=102,runner=runner),24)
            self.assertEqual([w.proc.pid for w in runner.workers],pids)
            self.assertEqual(con.execute('SELECT SUM(battles) FROM cell WHERE policy_id=?',(policy,)).fetchone()[0],48)
            self.assertEqual(con.execute('SELECT SUM(unattributed) FROM batch WHERE policy_id=?',(policy,)).fetchone()[0],0)
            processes=[w.proc for w in runner.workers]+[s.proc for s in runner.servers]
            used={json.loads(line)['worker'] for line in (runner.folder/'results.jsonl').read_text().splitlines()}
            self.assertEqual(used,set(range(6)))
        self.assertTrue(all(p.poll() is not None for p in processes))
        self.assertTrue(all(w.proc.stdin.closed and w.proc.stdout.closed for w in runner.workers))

    def test_main_cli_writes_panels_and_reuses_versioned_cache(self):
        script=Path(pipeline.__file__)
        command=[sys.executable,str(script),'--database',str(self.folder/'matrix.sqlite'),
                 '--run-dir',str(self.folder)]
        first=subprocess.run(command+['--evidence-dir',str(self.folder/'first')],capture_output=True,text=True,timeout=240)
        self.assertEqual(first.returncode,0,first.stderr[-3000:])
        panels=json.loads((self.folder/'panels.json').read_text())
        self.assertEqual((panels['evaluator']['workers'],panels['evaluator']['concurrency']),(6,24))
        self.assertEqual(len(panels['panels']),6)
        self.assertTrue(all(p['battles']==4 and p['coverage']==4 for p in panels['panels'].values()))
        second=subprocess.run(command+['--evidence-dir',str(self.folder/'second')],capture_output=True,text=True,timeout=240)
        self.assertEqual(second.returncode,0,second.stderr[-3000:])
        self.assertEqual(json.loads((self.folder/'panels.json').read_text())['policy_id'],panels['policy_id'])
        self.assertFalse((self.folder/'second/startup.jsonl').exists())
        with sqlite3.connect(self.folder/'matrix.sqlite') as con:
            self.assertEqual(con.execute('SELECT SUM(battles) FROM cell').fetchone()[0],24)
            self.assertEqual(con.execute('SELECT COUNT(*) FROM batch').fetchone()[0],1)


if __name__=='__main__':unittest.main()
