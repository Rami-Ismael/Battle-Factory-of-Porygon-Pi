import sys
import tempfile
import unittest
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
import numpy as np
import pickle
import torch
from checkpoints import epoch_names, load_checkpoint


class LoadCheckpointTests(unittest.TestCase):
    """load_checkpoint refuses pickled objects beyond tensors unless the caller trusts the file."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.dir=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tensor_checkpoint_loads_by_default(self):
        path=self.dir/'tensors.pt'
        torch.save({'sd':{'w':torch.arange(3.)},'cfg':{'d':8},'itos':['a','b']},path)
        ck=load_checkpoint(path,map_location='cpu')
        self.assertTrue(torch.equal(ck['sd']['w'],torch.arange(3.)))
        self.assertEqual(ck['cfg'],{'d':8})

    def test_numpy_checkpoint_refused_by_default(self):
        path=self.dir/'numpy.pt'
        torch.save({'counts':np.arange(3)},path)
        with self.assertRaises(pickle.UnpicklingError):load_checkpoint(path,map_location='cpu')

    def test_numpy_checkpoint_loads_when_trusted(self):
        path=self.dir/'numpy.pt'
        torch.save({'counts':np.arange(3)},path)
        ck=load_checkpoint(path,map_location='cpu',trusted_objects=True)
        np.testing.assert_array_equal(ck['counts'],np.arange(3))


OLD_INFO = dict(best_pass=2, stopped_pass=5, curves=[dict(pass_=0, val_loss=1.0), dict(pass_=1, train_loss=2.0)],
                base=dict(best_pass=21, stopped_pass=21), members=[dict(stopped_pass=80), dict(stopped_pass=60)])


class EpochNamesTests(unittest.TestCase):
    """Run info saved before 2026-11-08 says pass for an epoch; loading gives the epoch names."""

    def test_old_names_become_epoch_names_at_every_level(self):
        info = epoch_names(OLD_INFO)
        self.assertEqual((info['best_epoch'], info['stopped_epoch']), (2, 5))
        self.assertEqual([e['epoch'] for e in info['curves']], [0, 1])
        self.assertEqual(info['base']['best_epoch'], 21)
        self.assertEqual([m['stopped_epoch'] for m in info['members']], [80, 60])
        self.assertNotIn('best_pass', repr(info))
        self.assertNotIn('pass_', repr(info))

    def test_a_loaded_checkpoint_has_epoch_names(self):
        path = self.tmp / 'old.pt'
        torch.save(dict(members=[], info=OLD_INFO), path)
        self.assertEqual(load_checkpoint(path, map_location='cpu')['info']['best_epoch'], 2)

    def test_new_info_is_unchanged(self):
        new = dict(best_epoch=3, stopped_epoch=6, curves=[dict(epoch=0)])
        self.assertEqual(epoch_names(new), new)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


if __name__=='__main__':
    unittest.main()
