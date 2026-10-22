#!/usr/bin/env python3
"""Fill missing real-team labels using the verified fast BC evaluator.

Original training data and shared matrix are read-only inputs. Completed batches
are resumable in a dedicated database; exports are a versioned training dataset.
"""
import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import time

import numpy as np
import mask_lab_battle as B
import asked_vs_got as X
import heldout as H
from corpus import parse_team_text

ROOT = B.HERE / 'runs/vgcpastes-labels-20261013'
SOURCE = B.REPO / 'results/lossdata.npz'
DATABASE = Path.home() / 'vgc-data/vgcpastes_fast_labels_20261013.sqlite'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'manifest.json').exists():
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        assert manifest['source_sha256'] == digest(SOURCE), 'Training source changed'
        return manifest
    d = np.load(SOURCE)
    _, vocab, *_ = X.setup()
    missing = np.flatnonzero((d['T'] == 1) & ~np.isfinite(d['W']))
    keep = H.row_mask(d['X'], vocab)
    wanted = {d['X'][i].tobytes(): int(i) for i in missing if keep[i]}
    matches = defaultdict(list)
    # Preserve the original Stat Points and paste when the raw source exists.
    for path in sorted(B.MDB.POOL_DIR.rglob('*.txt')):
        paste = path.read_text()
        team = parse_team_text(paste)
        if len(team) != 6:
            continue
        row = wanted.get(vocab.encode(team).tobytes())
        if row is not None:
            matches[row].append({'paste': paste, 'source': str(path)})
    # A few encoded records came from the canonical database representation.
    with sqlite3.connect('file:' + str(B.MDB.DEFAULT_DB) + '?mode=ro', uri=True) as con:
        for tid, paste in con.execute("SELECT DISTINCT t.team_id,t.paste FROM team t JOIN team_source s USING(team_id) WHERE s.source='vgcpastes' ORDER BY t.team_id"):
            team = parse_team_text(paste)
            if len(team) == 6:
                row = wanted.get(vocab.encode(team).tobytes())
                if row is not None and row not in matches:
                    matches[row].append({'paste': paste, 'source': f'matrix-team:{tid}'})
    B.private_copy(DATABASE)
    con = B.MDB.connect(DATABASE)
    validator = B.hps_generate.Validator()
    records, excluded = [], []
    try:
        for row in sorted(wanted.values()):
            if row not in matches:
                excluded.append({'row': row, 'reason': 'No exact source paste'})
                continue
            original = matches[row][0]
            assert not H.is_heldout_text(original['paste'])
            tid, _ = B.MDB.add_team(con, original['paste'], 'vgcpastes:fast-labels', validator)
            legal, error, canon = con.execute('SELECT legal,legal_error,paste FROM team WHERE team_id=?', (tid,)).fetchone()
            expected = d['X'][row].copy()
            defaulted = []
            for field in range(7, 48, 8):
                if vocab.itos['nature'][expected[field]] == '':
                    expected[field] = vocab.stoi['nature']['serious']
                    defaulted.append(field // 8)
            assert H.grid_key_text(canon) == H.row_key(expected, vocab), f'Identity mismatch at row {row}'
            assert not H.is_heldout_text(canon)
            if legal != 1:
                excluded.append({'row': row, 'team_id': tid, 'reason': error or 'Not validated'})
                continue
            records.append({'row': row, 'team_id': tid, 'source': original['source'],
                            'paste_sha256': hashlib.sha256(canon.encode()).hexdigest(),
                            'matching_sources': len(matches[row]),
                            'neutral_nature_default_slots': defaulted})
        opponents = B.MDB.set_members(con, 'top50')
        assert opponents == list(d['top']), 'Opponent columns changed'
        held = H.team_ids(con)
        assert not (held & (set(opponents) | {r['team_id'] for r in records}))
    finally:
        validator.close()
        con.close()
    manifest = {'source': str(SOURCE), 'source_sha256': digest(SOURCE), 'database': str(DATABASE),
                'missing_real': len(missing), 'protected_missing_real': int((~keep[missing]).sum()),
                'protected_rows_total': int((~keep).sum()), 'records': records, 'excluded': excluded,
                'opponents': opponents, 'per_cell': 1,
                'selection': 'First lexicographic exact source paste per missing encoded real-team record; canonical matrix fallback.',
                'scoring': 'Top50 membership-weighted mean opponent win rate; draws are non-wins; self-match skipped.'}
    save(ROOT / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('records', 'opponents')}) + f' candidates={len(records)}', flush=True)
    return manifest


def export(manifest, con, policy, evidence):
    d = {k: v.copy() for k, v in np.load(SOURCE).items()}
    assert digest(SOURCE) == manifest['source_sha256']
    cols = manifest['opponents']
    colmap = {tid: j for j, tid in enumerate(cols)}
    labels = []
    for rec in manifest['records']:
        tid, i = rec['team_id'], rec['row']
        score = B.MDB.score(con, policy, tid)
        cells = B.MDB.row_cells(con, policy, tid, cols)
        assert score['coverage'] == score['columns'] > 0
        assert all(n >= manifest['per_cell'] for _, n in cells.values())
        assert np.isnan(d['W'][i]) and d['T'][i] == 1
        d['W'][i] = score['score']
        d['NB'][i] = score['battles']
        d['M'][i] = 0  # Replace, never mix old and fast-policy matrix cells.
        for op, (wins, n) in cells.items():
            d['M'][i, colmap[op]] = [wins, n]
        labels.append({**rec, **score, 'policy_id': policy,
                       'by_opponent': {str(op): {'wins': w, 'battles': n} for op, (w, n) in cells.items()}})
    _, vocab, *_ = X.setup()
    keep = H.row_mask(d['X'], vocab)
    original_rows = np.flatnonzero(keep)
    for k in ('X', 'W', 'NB', 'T', 'S', 'M'):
        d[k] = d[k][keep]
    d['original_row'] = original_rows
    new_rows = {r['row'] for r in labels}
    d['fast_label'] = np.array([int(i) in new_rows for i in original_rows])
    assert H.row_mask(d['X'], vocab).all()
    path = ROOT / 'lossdata-vgcpastes-fast.npz'
    np.savez_compressed(path, **d)
    hist = {}
    for name, mask in [('all', np.ones(len(d['W']), bool))] + [(name, d['T'] == i) for i, name in enumerate(('model', 'real', 'fill', 'hps'))]:
        w = d['W'][mask]
        finite = w[np.isfinite(w)]
        counts = [0] * 101
        for x in finite:
            counts[int((Decimal(str(x)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))] += 1
        assert sum(counts) == len(finite)
        hist[name] = dict(rows=len(w), labeled=len(finite), unlabeled=int((~np.isfinite(w)).sum()),
                          mean=float(finite.mean()) if len(finite) else None, counts=counts)
    save(ROOT / 'histogram.json', hist)
    save(ROOT / 'labels.json', {'source_sha256': manifest['source_sha256'], 'policy_id': policy,
                              'evidence': evidence, 'records': labels})
    save(ROOT / 'summary.json', {'added_labels': len(labels), 'dataset': str(path),
          'dataset_sha256': digest(path), 'source_sha256': manifest['source_sha256'],
          'source_unchanged': digest(SOURCE) == manifest['source_sha256'],
          'protected_rows_removed': int((~keep).sum()), 'protected_missing_real': manifest['protected_missing_real'],
          'excluded': manifest['excluded'], 'real_records': hist['real']['rows'],
          'real_labeled': hist['real']['labeled'], 'real_unlabeled': hist['real']['unlabeled'],
          'policy_id': policy, 'evidence': evidence,
          'new_label_battle_observations': sum(r['battles'] for r in labels),
          'note': 'Preserved existing non-heldout labels. Added fast-policy labels only to missing real rows. Model-specific train/validation selection still applies; no model was retrained.'})
    print((ROOT / 'summary.json').read_text(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    manifest = prepare()
    if args.prepare_only:
        return
    if B.MDB.sha256_file(B.MDB.POLICY_FILE) != B.BC_HASH:
        raise RuntimeError('BC checkpoint changed')
    a = argparse.Namespace(backend='fast', workers=6, concurrency=24, batch=24,
                           per_cell=manifest['per_cell'], evidence_dir=ROOT / ('evaluator-' + str(time.time_ns())))
    con = B.MDB.connect(DATABASE)
    old = signal.getsignal(signal.SIGTERM)
    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        rows = sorted({r['team_id'] for r in manifest['records']})
        print(f'Evaluating {len(rows)} teams against {len(manifest["opponents"])} meta opponents', flush=True)
        policy, evidence = B.evaluate(con, rows, manifest['opponents'], a)
        export(manifest, con, policy, evidence)
    finally:
        signal.signal(signal.SIGTERM, old)
        con.close()


if __name__ == '__main__':
    main()
