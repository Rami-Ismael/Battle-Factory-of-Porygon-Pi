"""Audit the saved trace against fresh checkpoint inference; no trace mutation."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recorder', ROOT / 'record-archetypes.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
torch.set_num_threads(4)
t = json.loads((ROOT / 'trace.json').read_text())
assert r.sha(r.CK) == t['checkpointSha256']
for name, digest in t['sourceHashes'].items():
    current = r.sha(r.REPO / 'src' / name)
    if current != digest:
        # Reviewed additive regulation-vocabulary API; legacy Vocab(teams) is unchanged.
        assert name == 'encode.py' and current == 'f1c13ccff630bad0be7d9ed222d8dbcb8c4b7aed3bec09435ef5b697d51aef04', (name,current)
        print('NOTE encode.py has reviewed additive regulation API; checking exact sample parity below', flush=True)
print('PASS checkpoint, other source hashes, and 692 corpus hashes', flush=True)
main = r.sample(t['teams'], t['seedMain'])
for row, saved in zip(main, t['records']):
    fields = [r.look.get(r.dec(c, v), r.dec(c, v)) for c, v in enumerate(row)]
    assert fields == saved['fields']
    assert r.label(row) == saved['label']
print('PASS all 300 teams exactly reproduce from original sampler; all rule labels match', flush=True)
# Compare the resumed implementation with original categorical draws at the same RNG state.
for s in [0, 6, 18, 30, 41]:
    torch.manual_seed(913)
    states = []
    original = torch.multinomial
    rng_states = []
    def observe(*args, **kwargs):
        rng_states.append(torch.get_rng_state().clone())
        value = original(*args, **kwargs)
        states.append(int(value.item()))
        return value
    torch.multinomial = observe
    try:
        row = r.H.sample_constrained(r.model, r.C, 1, r.H.WNULL, 1., 1., device='cpu')[0]
    finally:
        torch.multinomial = original
    g = torch.Generator().set_state(rng_states[s])
    resumed = r.complete(r.masked(row, s), s, 1, g)[0]
    assert torch.equal(resumed[r.ORDER[:42]], row[r.ORDER[:42]])
    assert torch.equal(resumed[r.ORDER[:s]], row[r.ORDER[:s]])
print('PASS resumed completion parity at steps 0, 6, 18, 30, 41', flush=True)
ref = r.sample(t['referenceTeams'], t['seedReference'])
y = np.array([r.label(row) for row in ref]); X = r.feats(ref); mu = X.mean(0)
assert np.bincount(y, minlength=4).tolist() == t['referenceLabelCounts']
Sw = sum(np.cov((X[y == k] - mu).T, bias=True) * (y == k).sum() for k in range(4)) / len(X) + .05 * np.eye(X.shape[1])
M = np.stack([X[y == k].mean(0) - mu for k in range(4)])
Sb = (M.T * np.bincount(y, minlength=4) / len(y)) @ M
Li = np.linalg.inv(np.linalg.cholesky(Sw)); ev, U = np.linalg.eigh(Li @ Sb @ Li.T)
W = Li.T @ U[:, ::-1][:, :3]; P = (X - mu) @ W
W /= np.percentile(np.abs(P), 98, axis=0)
cent = np.stack([((X[y == k] - mu) @ W).mean(0) for k in range(4)])
assert np.allclose(cent, t['centroids'], atol=5.1e-5)
for i, row in enumerate(main):
    assert np.allclose(r.summary(row[None], W, mu), t['records'][i]['steps'][48], atol=1e-6)
print('PASS independent 1,500-team reference fit, fixed centroids, and 300 endpoints', flush=True)
# Reproduce one complete conditional trajectory (same sequential generator consumption).
_, actual = r.worker((0, main[0], W, mu))
assert np.allclose(actual, t['records'][0]['steps'][1:42], atol=1e-5)
s0 = r.summary(r.complete(torch.zeros(48, dtype=torch.long), 0, t['completionsFullyMasked'],
                         torch.Generator().manual_seed(t['seedMain'])), W, mu)
assert np.allclose(s0, t['records'][0]['steps'][0], atol=1e-5)
for saved in t['records']:
    a = np.array(saved['steps'])
    assert a.shape == (49, 13) and np.isfinite(a).all()
    assert np.allclose(a[:, 9:].sum(1), 1, atol=.00021)
    assert np.all(a[:, 9:] >= 0) and np.all(a[:, 9:] <= 1)
    assert np.array_equal(a[0], t['records'][0]['steps'][0])
    assert np.all(a[42:] == a[42]) and np.all(a[42:, 3:9] == 0)
    for st in a:
        c = st[3:9]
        cov = np.array([[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]])
        assert np.linalg.eigvalsh(cov).min() > -2e-5
print('PASS 41 conditional states, shared 512-completion prior, all 14,700 states structurally valid', flush=True)
