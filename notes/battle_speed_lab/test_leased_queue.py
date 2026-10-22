import tempfile
import threading
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from leased_queue import LeaseQueue,Client,http_server


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.now=100.
        self.path=Path(self.temp.name)/'jobs.sqlite'
        self.queue=LeaseQueue(self.path,'engine-policy-preprocessor-v1',lambda:self.now)
    def tearDown(self):self.temp.cleanup()

    def test_retry_fences_crashed_worker_and_completion_is_idempotent(self):
        identity=self.queue.submit({'battle':1,'seed':45})['id']
        old=self.queue.claim('old',self.queue.runtime,1)
        self.now+=2
        new=self.queue.claim('new',self.queue.runtime,10)
        self.assertEqual(old['id'],new['id']);self.assertNotEqual(old['token'],new['token'])
        with self.assertRaises(ValueError):self.queue.complete(identity,old['token'],{'win':True})
        self.assertFalse(self.queue.complete(identity,new['token'],{'win':True})['duplicate'])
        self.assertTrue(self.queue.complete(identity,new['token'],{'win':True})['duplicate'])
        with self.assertRaises(ValueError):self.queue.complete(identity,new['token'],{'win':False})
        self.assertEqual(len(self.queue.results()),1)

    def test_payload_identity_and_runtime_isolation(self):
        self.assertEqual(self.queue.submit({'seed':1}),self.queue.submit({'seed':1}))
        self.assertNotEqual(self.queue.submit({'seed':1}),self.queue.submit({'seed':2}))
        with self.assertRaises(ValueError):self.queue.claim('worker','wrong-runtime')
        with self.assertRaises(ValueError):LeaseQueue(self.path,'wrong-runtime')

    def test_parallel_claims_are_unique_and_persist_after_reopen(self):
        for i in range(20):self.queue.submit({'battle':i})
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs=list(pool.map(lambda i:self.queue.claim(str(i),self.queue.runtime),range(20)))
        self.assertEqual(len({j['id'] for j in jobs}),20)
        restored=LeaseQueue(self.path,self.queue.runtime,lambda:self.now)
        self.assertEqual(restored.status()['counts'],{'leased':20})

    def test_heartbeat_and_http_roundtrip(self):
        server=http_server(self.queue,'test-secret')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            client=Client(f'http://127.0.0.1:{server.server_port}','test-secret')
            client.call('submit',payload={'battle':3})
            job=client.call('claim',worker='http-worker',runtime=self.queue.runtime,seconds=1)
            client.call('heartbeat',identity=job['id'],token=job['token'],seconds=10)
            self.now+=2
            client.call('complete',identity=job['id'],token=job['token'],result={'win':False})
            self.assertEqual(client.call('status')['counts'],{'complete':1})
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
