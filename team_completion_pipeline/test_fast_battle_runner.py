import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mask_lab_battle as pipeline
import fast_battle_runner as fast

MDB = pipeline.MDB
SPECIES = {i:[f'mon{i}{j}' for j in range(6)] for i in range(1, 5)}


def key(names):
    return ','.join(sorted(names))


def database():
    con = sqlite3.connect(':memory:')
    con.executescript(MDB.SCHEMA)
    for i in SPECIES:
        con.execute("INSERT INTO team(team_id,format_id,canon_hash,paste,species_key,base_key,legal,origin) VALUES (?,?,?,?,?,?,1,'test')",
                    (i, fast.FORMAT, str(i), str(i), key(SPECIES[i]), key(SPECIES[i])))
        con.execute("INSERT INTO team_set VALUES ('top50',?,1)", (i,))
    MDB.get_policy(con, row_policy='test', simulator='test')
    con.commit()
    return con


def outcome(request):
    return {'completed':len(request['pairs']), 'battles':[
        {'tag':str(i), 'finished':True, 'team':SPECIES[p['a_id']],
         'opponent':SPECIES[p['b_id']], 'won':[True, False, None][i % 3]}
        for i,p in reversed(list(enumerate(request['pairs'])))]}


class JobTests(unittest.TestCase):
    def setUp(self):
        self.con = database()
        self.addCleanup(self.con.close)

    def test_defaults_and_compatibility_alias(self):
        args = pipeline.parse_args([])
        self.assertEqual((args.backend,args.workers,args.concurrency), ('fast',6,24))
        self.assertEqual(pipeline.parse_args(['--servers','2']).workers, 2)

    def test_exact_chunk_schedule_and_fixed_candidate(self):
        requests = fast.build_requests(self.con, [{'row':1,'schedule':[2,3,2,3,2]}], 2)
        self.assertEqual([len(r['pairs']) for r in requests], [2,2,1])
        self.assertEqual([p['b_id'] for r in requests for p in r['pairs']], [2,3,2,3,2])
        self.assertTrue(all(p['a_id']==1 for r in requests for p in r['pairs']))

    def test_49_opponent_remainders_are_balanced_across_all_six_workers(self):
        requests=[{'job_id':i, 'pairs':[None]*size} for i,size in enumerate([24,24,1]*12)]
        shards=fast.assign_requests(requests,6)
        self.assertEqual([sum(len(r['pairs']) for r in s) for s in shards],[98]*6)
        self.assertEqual(sorted(r['job_id'] for s in shards for r in s),list(range(36)))

    def test_self_illegal_unknown_and_ambiguous_teams_rejected(self):
        for opponent in [1,99]:
            with self.assertRaises(ValueError):fast.build_requests(self.con,[{'row':1,'schedule':[opponent]}])
        self.con.execute('UPDATE team SET legal=0 WHERE team_id=2')
        with self.assertRaises(ValueError):fast.build_requests(self.con,[{'row':1,'schedule':[2]}])
        self.con.execute('UPDATE team SET base_key=? WHERE team_id=4',(key(SPECIES[3]),))
        with self.assertRaises(ValueError):fast.build_requests(self.con,[{'row':1,'schedule':[3,4]}])

    def test_reordered_results_preserve_wins_losses_and_ties(self):
        request = fast.build_requests(self.con,[{'row':1,'schedule':[2,3,2]}])[0]
        with patch.object(fast,'base_key',side_effect=key):
            self.assertEqual(fast.attribute(request,outcome(request)),[(1,2,1,0,1),(1,3,0,1,0)])

    def test_missing_wrong_duplicate_and_invalid_results_rejected(self):
        request = fast.build_requests(self.con,[{'row':1,'schedule':[2,3]}])[0]
        with patch.object(fast,'base_key',side_effect=key):
            for change in ['missing','candidate','opponent','duplicate','outcome']:
                result = outcome(request)
                if change=='missing':result['battles'].pop()
                elif change=='candidate':result['battles'][0]['team']=SPECIES[4]
                elif change=='opponent':result['battles'][0]['opponent']=SPECIES[4]
                elif change=='duplicate':result['battles'][0]['tag']=result['battles'][1]['tag']
                else:result['battles'][0]['won']='draw'
                with self.subTest(change=change),self.assertRaises(RuntimeError):fast.attribute(request,result)

    def test_python_source_identity_ignores_bytecode_but_tracks_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'module.py').write_text('x=1\n')
            first=fast.python_source_digest(root)
            (root/'module.pyc').write_bytes(b'compiled')
            self.assertEqual(fast.python_source_digest(root),first)
            (root/'module.py').write_text('x=2\n')
            self.assertNotEqual(fast.python_source_digest(root),first)

    def test_cache_identity_reuses_configuration_and_separates_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with fast.FastBattleRunner(root/'first') as first,fast.FastBattleRunner(root/'second') as second,fast.FastBattleRunner(root/'changed',concurrency=8) as changed:
                identity=first.policy_id(self.con,MDB)
                self.assertEqual(second.policy_id(self.con,MDB),identity)
                self.assertNotEqual(changed.policy_id(self.con,MDB),identity)
                self.assertEqual(first.workers,[])
            with fast.FastBattleRunner(os.path.relpath(root/'relative')) as relative:
                self.assertEqual(relative.folder,(root/'relative').resolve())
            with fast.FastBattleRunner(root/'longer',request_chunk_size=96) as longer:
                self.assertNotEqual(longer.policy_id(self.con,MDB),identity)
                self.assertEqual(longer.config['concurrency'],24)

    def test_checkpoint_snapshot_survives_shared_file_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root/'shared.zip'
            shared.write_bytes(b'verified weights')
            expected = fast.lab.digest(shared)
            with patch.object(fast, 'CHECKPOINT', shared), patch.object(fast, 'CHECKPOINT_SHA256', expected):
                with fast.FastBattleRunner(root/'run') as runner:
                    shared.write_bytes(b'replacement weights')
                    self.assertEqual(runner.checkpoint.read_bytes(), b'verified weights')
                    self.assertEqual(fast.lab.digest(runner.checkpoint), runner.provenance['policy_sha256'])
                    with patch.object(fast.lab, 'prepare_runtime'), patch.object(fast.lab, 'tree_digest', return_value=runner.provenance['simulator_sha256']), patch.object(fast.lab, 'Server', FakeServer), patch.object(fast.lab, 'Worker', FakeWorker) as worker:
                        runner._start()
                    self.assertEqual(FakeWorker.instances[-1].config['checkpoint_path'], str(runner.checkpoint))
                    self.assertEqual(FakeWorker.instances[-1].config['checkpoint_sha256'], expected)


class FakeServer:
    instances=[]
    def __init__(self,*args):
        self.port=10000+len(self.instances);self.closed=False;self.proc=SimpleNamespace(pid=self.port);self.instances.append(self)
    def close(self):self.closed=True


class FakeWorker:
    instances=[]
    fail=False
    def __init__(self,*args):
        self.config=args[0];self.closed=False;self.requests=[];self.proc=SimpleNamespace(pid=20000+len(self.instances));self.ready={};self.instances.append(self)
    def trial(self,request):
        self.requests.append(request)
        if self.fail:raise RuntimeError('worker failed')
        return outcome(request)
    def close(self):self.closed=True


class RunnerDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.con=database();self.addCleanup(self.con.close)
        self.runner=fast.FastBattleRunner.__new__(fast.FastBattleRunner)
        self.runner.folder=Path(self.temp.name);self.runner.runtime=self.runner.folder/'runtime'
        self.runner.worker_count=6;self.runner.config={'concurrency':24}
        self.runner.closed=False;self.runner.calls=0;self.runner.servers=[];self.runner.workers=[]
        self.runner.checkpoint=self.runner.folder/'policy.zip'
        self.runner.provenance={'simulator_sha256':'test','policy_sha256':'test-policy'}
        FakeServer.instances=[];FakeWorker.instances=[];FakeWorker.fail=False
        self.addCleanup(self.runner.close)
        for target,name,value in [(fast.lab,'Server',FakeServer),(fast.lab,'Worker',FakeWorker),
                                  (fast.lab,'prepare_runtime',lambda _:None),(fast,'base_key',key),
                                  (fast.lab,'tree_digest',lambda _:'test'),
                                  (MDB.H,'team_ids',lambda _:set())]:
            p=patch.object(target,name,value);p.start();self.addCleanup(p.stop)

    def test_cache_topup_and_worker_reuse_with_real_sqlite(self):
        self.assertEqual(MDB.ensure(self.con,1,[1],[2,3],1,seed=5,runner=self.runner),2)
        self.assertEqual(MDB.ensure(self.con,1,[1],[2,3],1,seed=5,runner=self.runner),0)
        self.assertEqual(MDB.ensure(self.con,1,[1],[2,3],3,seed=5,runner=self.runner),4)
        self.assertEqual(len(FakeWorker.instances),6)
        self.assertEqual(len(FakeServer.instances),6)
        self.assertEqual(self.runner.calls,2)
        self.assertEqual(self.con.execute('SELECT SUM(battles) FROM cell').fetchone()[0],6)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM batch').fetchone()[0],2)
        self.runner.close();self.runner.close()
        self.assertTrue(all(p.closed for p in FakeWorker.instances+FakeServer.instances))

    def test_long_request_keeps_concurrency_and_every_scheduled_battle(self):
        self.runner.request_chunk_size = 96
        schedule = [2, 3] * 25
        results, unattributed, _ = self.runner(self.con, [{'row':1,'schedule':schedule}], seed=5)
        requests = [r for w in FakeWorker.instances for r in w.requests]
        self.assertEqual([len(r['pairs']) for r in requests], [50])
        self.assertEqual([p['b_id'] for r in requests for p in r['pairs']], schedule)
        self.assertEqual(requests[0]['config']['concurrency'],24)
        self.assertEqual(sum(sum(r[2:]) for r in results),50)
        self.assertEqual(unattributed,0)

    def test_worker_failure_cannot_create_cached_draws_or_partial_batch(self):
        FakeWorker.fail=True
        with self.assertRaises(RuntimeError):MDB.ensure(self.con,1,[1],[2,3],1,runner=self.runner)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM cell').fetchone()[0],0)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM batch').fetchone()[0],0)
        self.assertTrue(self.runner.closed)

    def test_invalid_seed_does_not_start_workers(self):
        with self.assertRaises(ValueError):self.runner(self.con,[{'row':1,'schedule':[2]}],seed=-1)
        self.assertEqual(FakeWorker.instances,[])

    def test_heldout_guard_still_precedes_execution(self):
        with patch.object(MDB.H,'team_ids',return_value={2}),self.assertRaises(MDB.H.HeldoutLeak):
            MDB.ensure(self.con,1,[1],[2],1,runner=self.runner)
        self.assertEqual(FakeWorker.instances,[])

    def test_startup_failure_closes_every_created_process(self):
        def fail(*args):raise RuntimeError('model load failed')
        with patch.object(fast.lab,'Worker',side_effect=fail),self.assertRaises(RuntimeError):
            self.runner(self.con,[{'row':1,'schedule':[2]}])
        self.assertTrue(FakeServer.instances[0].closed)
        self.assertTrue(self.runner.closed)


if __name__=='__main__':unittest.main()
