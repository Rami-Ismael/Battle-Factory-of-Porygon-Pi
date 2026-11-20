import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import label_remaining as labels
import label_vgcpastes as real_labels


class RecoveryTests(unittest.TestCase):
    def test_vgcpastes_excludes_overlapping_runs_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(real_labels, 'DATABASE', Path(tmp) / 'labels.sqlite'):
            with real_labels.exclusive_run():
                with self.assertRaises(BlockingIOError):
                    with real_labels.exclusive_run():
                        self.fail('Overlapping writer acquired the lock')
            with real_labels.exclusive_run():
                pass

    def test_stop_discards_uncommitted_labels_and_rebuilds_from_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.npz'
            np.savez(source, W=np.array([np.nan, np.nan]), NB=np.zeros(2),
                     M=np.zeros((2, 1, 2)), fast_label=np.zeros(2, dtype=bool))
            job = labels.Job.__new__(labels.Job)
            job.con = sqlite3.connect(':memory:')
            self.addCleanup(job.con.close)
            job.con.executescript('CREATE TABLE label_value(row_id,team_id,score,battles,policy_id);'
                                 'CREATE TABLE cell(battles);')
            job.con.execute('INSERT INTO label_value VALUES (0,10,0.5,2,1)')
            job.con.execute('INSERT INTO cell VALUES (2)')
            job.con.commit()
            job.con.execute('INSERT INTO label_value VALUES (1,11,1.0,2,1)')
            job.d = {'W': np.array([0.5, 1.0])}
            job.baseline = np.array([np.nan, np.nan])
            job.cols = [20]
            job.done = 2
            with patch.object(labels, 'SOURCE', source), patch.object(labels.B.MDB, 'row_cells', return_value={20: (1, 2)}):
                job.recover_committed()
            self.assertEqual(job.done, 1)
            self.assertEqual(job.battles, 2)
            np.testing.assert_equal(job.d['W'], [0.5, np.nan])
            np.testing.assert_equal(job.d['fast_label'], [True, False])

    def test_background_preserves_batch_limit(self):
        command = labels.background_command(SimpleNamespace(workers=6, chunk_size=96, max_batches=1))
        self.assertEqual(command[-2:], ['--max-batches', '1'])
        command = labels.background_command(SimpleNamespace(workers=6, chunk_size=96, max_batches=None))
        self.assertNotIn('--max-batches', command)


if __name__ == '__main__':
    unittest.main()
