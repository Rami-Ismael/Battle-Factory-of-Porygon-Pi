#!/usr/bin/env python3
"""Durable, resumable fast-BC label generation with a live chart snapshot."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import numpy as np
import mask_lab_battle as B
import asked_vs_got as X
import heldout as H
import lossdata as LD
from corpus import parse_team_text

ROOT = B.HERE / 'runs/all-labels-20261013'
SOURCE = B.HERE / 'runs/vgcpastes-labels-20261013/lossdata-vgcpastes-fast.npz'
DB = ROOT / 'battles.sqlite'
LIVE = B.HERE.parent / 'artifacts/win-rate-frequency-live.json'
NAMES = ('model', 'real', 'fill', 'hps')


def same_battle_semantics(before, after):
    """Scheduling may change; policy, numerical settings, and simulator may not."""
    keys = ('configuration','policy_sha256','simulator_sha256','formats_sha256',
            'bench_source_sha256','bench_data_sha256','packages')
    if any(k not in before or k not in after for k in (*keys,'implementation')):
        return False
    if any(before.get(k) != after.get(k) for k in keys):
        return False
    return {k:v for k,v in before['implementation'].items() if k!='fast_battle_runner.py'} == {
        k:v for k,v in after['implementation'].items() if k!='fast_battle_runner.py'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False, separators=(',', ':')) + '\n')
    tmp.replace(path)


def histogram(d):
    result = {}
    # Match the chart's decimal half-up convention, including float32 decimals.
    rounded = {float(x): int((Decimal(str(x)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
               for x in np.unique(d['W'][np.isfinite(d['W'])])}
    for name, mask in [('all', np.ones(len(d['W']), bool))] + [(n, d['T'] == i) for i, n in enumerate(NAMES)]:
        w = d['W'][mask]
        good = w[np.isfinite(w)]
        counts = np.bincount([rounded[float(x)] for x in good], minlength=101).tolist()
        result[name] = dict(rows=len(w), labeled=len(good), unlabeled=int((~np.isfinite(w)).sum()),
                            mean=float(good.mean()) if len(good) else None, counts=counts)
    return result


class Job:
    def __init__(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.d = {k: v.copy() for k, v in np.load(SOURCE).items()}
        self.baseline = self.d['W'].copy()
        self.source_hash = sha(SOURCE)
        self.con = B.MDB.connect(DB)
        self.con.executescript('''
          CREATE TABLE IF NOT EXISTS label_job(row_id INTEGER PRIMARY KEY,team_id INTEGER,source TEXT,status TEXT,error TEXT);
          CREATE TABLE IF NOT EXISTS label_value(row_id INTEGER PRIMARY KEY,team_id INTEGER,score REAL,battles INTEGER,policy_id INTEGER);
          CREATE TABLE IF NOT EXISTS label_scan(path TEXT PRIMARY KEY,sha256 TEXT);
          CREATE TABLE IF NOT EXISTS label_meta(key TEXT PRIMARY KEY,value TEXT);
        ''')
        existing = self.con.execute("SELECT value FROM label_meta WHERE key='source_sha256'").fetchone()
        if existing and existing[0] != self.source_hash:
            raise RuntimeError('Source dataset changed; use a new job folder')
        self.con.execute("INSERT OR IGNORE INTO label_meta VALUES ('source_sha256',?)", (self.source_hash,))
        self.con.commit()
        self.cols = list(map(int, self.d['top']))
        invalid_opponents = any(self.con.execute('SELECT legal FROM team WHERE team_id=?',(tid,)).fetchone() != (1,) for tid in self.cols)
        if not B.MDB.set_members(self.con, 'top50') or invalid_opponents:
            assert self.con.execute('SELECT COUNT(*) FROM cell').fetchone()[0] == 0, 'Cannot change opponent snapshot after battles'
            src = sqlite3.connect('file:' + str(B.MDB.DEFAULT_DB) + '?mode=ro', uri=True)
            columns = [r[1] for r in src.execute('PRAGMA table_info(team)')]
            names = ','.join(columns)
            updates = ','.join(k+'=excluded.'+k for k in columns if k!='team_id')
            for tid in self.cols:
                row = src.execute('SELECT * FROM team WHERE team_id=?', (tid,)).fetchone()
                self.con.execute('INSERT INTO team ('+names+') VALUES (' + ','.join('?' * len(row)) + ') ON CONFLICT(team_id) DO UPDATE SET '+updates, row)
                weight = src.execute("SELECT weight FROM team_set WHERE set_name='top50' AND team_id=?", (tid,)).fetchone()[0]
                self.con.execute("INSERT OR REPLACE INTO team_set VALUES ('top50',?,?)", (tid, weight))
            self.con.commit()
            src.close()
        assert B.MDB.set_members(self.con, 'top50') == self.cols
        self.policy = None
        self.evidence = None
        self.execution = None
        self.total = int((~np.isfinite(self.baseline)).sum())
        self.done = 0
        self.battles = 0
        self.restore()

    def restore(self):
        for i, tid, score, n, pid in self.con.execute('SELECT * FROM label_value'):
            self.apply(i, tid, score, n, pid)
        self.done = self.con.execute('SELECT COUNT(*) FROM label_value').fetchone()[0]
        self.battles = self.con.execute('SELECT COALESCE(SUM(battles),0) FROM cell').fetchone()[0]

    def apply(self, i, tid, score, n, pid):
        assert not np.isfinite(self.baseline[i])
        self.d['W'][i], self.d['NB'][i] = score, n
        self.d['M'][i] = 0
        for op, (wins, battles) in B.MDB.row_cells(self.con, pid, tid, self.cols).items():
            self.d['M'][i, self.cols.index(op)] = [wins, battles]
        self.d['fast_label'][i] = True

    def publish(self, status, detail='', checkpoint=False):
        counts = dict(self.con.execute('SELECT status,COUNT(*) FROM label_job GROUP BY status'))
        payload = dict(status=status,detail=detail,updated_at=datetime.now(timezone.utc).isoformat(),
                       pid=os.getpid(),source=str(SOURCE),source_sha256=self.source_hash,
                       target_labels=self.total,added_labels=self.done,unique_battles=self.battles,
                       mapping=counts,policy_id=self.policy,evidence=self.evidence,execution=self.execution,
                       dataset=str(ROOT/'lossdata-all-fast.npz'),data=histogram(self.d))
        save(ROOT/'status.json', payload)
        save(LIVE, payload)
        if checkpoint:
            path = ROOT/'lossdata-all-fast.npz'
            tmp = ROOT/'lossdata-all-fast.tmp.npz'
            np.savez_compressed(tmp, **self.d)
            tmp.replace(path)
            save(ROOT/'dataset-manifest.json', dict(source=str(SOURCE),source_sha256=self.source_hash,
                 dataset_sha256=sha(path),added_labels=self.done,status=status,
                 policy_id=self.policy,evidence=self.evidence,
                 note='Original row mapping and all existing labels preserved. New labels require full top50 opponent coverage. No retraining performed.'))

    def prepare(self):
        if self.con.execute("SELECT 1 FROM label_meta WHERE key='prepared'").fetchone():
            return
        self.publish('preparing', 'Matching saved team pastes and checking legality.')
        _, vocab, *_ = X.setup()
        missing = np.flatnonzero(~np.isfinite(self.baseline))
        assert H.row_mask(self.d['X'][missing], vocab).all()
        wanted = {self.d['X'][i].tobytes(): int(i) for i in missing}
        ready = {i for i, in self.con.execute("SELECT row_id FROM label_job WHERE status='ready'")}
        seen = set()
        validator = B.hps_generate.Validator()
        def consider(paste, source):
            digest = hashlib.sha256(paste.encode()).digest()
            if digest in seen:
                return
            seen.add(digest)
            team = parse_team_text(paste)
            if len(team) != 6:
                return
            i = wanted.get(vocab.encode(team).tobytes())
            if i is None or i in ready:
                return
            canon = B.MDB.canonical_paste(paste)
            expected = self.d['X'][i].copy()
            for field in range(7, 48, 8):
                if vocab.itos['nature'][expected[field]] == '':
                    expected[field] = vocab.stoi['nature']['serious']
            if H.grid_key_text(canon) != H.row_key(expected, vocab) or H.is_heldout_text(canon):
                status, error, tid = 'unusable', 'Canonical identity mismatch or protected team', None
            else:
                tid, _ = B.MDB.add_team(self.con, paste, 'training-labels:' + NAMES[int(self.d['T'][i])], validator)
                legal, error = self.con.execute('SELECT legal,legal_error FROM team WHERE team_id=?',(tid,)).fetchone()
                status = 'ready' if legal == 1 else 'illegal'
            self.con.execute('INSERT OR REPLACE INTO label_job VALUES (?,?,?,?,?)',(i,tid,source,status,error))
            if status == 'ready':
                ready.add(i)
        files = [p for p in sorted(X.RES.glob('*.json')) if p.stat().st_size >= 2000 and not any(s in p.stem for s in LD.SKIP)]
        try:
            for n, path in enumerate(files):
                if self.con.execute('SELECT 1 FROM label_scan WHERE path=?',(str(path),)).fetchone():
                    continue
                try:
                    data = json.loads(path.read_text())
                except (ValueError, OSError):
                    continue
                acc = []
                LD.walk_pastes(data, acc)
                for paste, _, _ in acc:
                    consider(paste, str(path))
                self.con.execute('INSERT INTO label_scan VALUES (?,?)',(str(path),sha(path)))
                self.con.commit()
                self.publish('preparing', f'Checked result file {n+1}/{len(files)}; {len(ready):,} legal records matched.')
                print(f'Sources {n+1}/{len(files)}: {len(ready)} ready', flush=True)
                del data, acc
            source = sqlite3.connect('file:' + str(B.MDB.DEFAULT_DB) + '?mode=ro', uri=True)
            for tid, paste in source.execute('SELECT team_id,paste FROM team WHERE legal=1 ORDER BY team_id'):
                consider(paste, f'matrix-team:{tid}')
            source.close()
            self.con.commit()
            self.publish('preparing', f'Matched {len(ready):,} legal records; checking HPS sources.')
            with open(LD.HPS) as f:
                for n, line in enumerate(f):
                    consider(json.loads(line)['team'], f'{LD.HPS}:line:{n+1}')
                    if (n+1) % 10000 == 0:
                        self.con.commit()
                        self.publish('preparing', f'Checked {n+1:,} HPS records; {len(ready):,} legal records matched.')
            for i in missing:
                self.con.execute("INSERT OR IGNORE INTO label_job VALUES (?,NULL,NULL,'unmatched','No original paste found')",(int(i),))
            self.con.execute("INSERT INTO label_meta VALUES ('prepared','1')")
            self.con.commit()
        finally:
            validator.close()
        self.publish('prepared', 'Source matching complete; starting battles.', checkpoint=True)

    def run(self, max_batches=None, workers=6, chunk_size=96):
        self.prepare()
        held = H.team_ids(self.con)
        assert not (held & set(self.cols))
        mapping = defaultdict(list)
        for i, tid in self.con.execute("SELECT row_id,team_id FROM label_job WHERE status='ready' AND row_id NOT IN (SELECT row_id FROM label_value)"):
            assert tid not in held
            mapping[tid].append(i)
        # Finish the smaller datasets first. Original labels are never rerun.
        priority = {2: 0, 3: 1, 0: 2, 1: 3}
        rows = sorted(mapping, key=lambda tid:(min(priority[int(self.d['T'][i])] for i in mapping[tid]),tid))
        completed_batches = 0
        with B.FastBattleRunner(ROOT/('evaluator-'+str(time.time_ns())),workers,24,request_chunk_size=chunk_size) as runner:
            self.policy = runner.policy_id(self.con, B.MDB)
            self.evidence = str(runner.folder)
            self.execution = dict(workers=workers,concurrency=24,request_chunk_size=chunk_size)
            known = {p for p, in self.con.execute('SELECT DISTINCT policy_id FROM label_value')}
            for previous in known - {self.policy}:
                identity = self.con.execute('SELECT row_policy FROM policy WHERE policy_id=?',(previous,)).fetchone()[0]
                previous_metadata = [json.loads(p.read_text()) for p in ROOT.glob('evaluator-*/metadata.json')]
                matches = [m for m in previous_metadata if 'bc-fast:'+m['identity']==identity]
                assert matches and all(same_battle_semantics(m,runner.provenance) for m in matches), 'Battle semantics changed'
            if known - {self.policy}:
                self.con.execute('INSERT OR REPLACE INTO label_meta VALUES (?,?)',
                    ('execution-transition:'+str(self.policy),json.dumps(dict(previous_policy_ids=sorted(known),
                       new_policy_id=self.policy,execution=self.execution,evidence=self.evidence))))
                self.con.commit()
            self.publish('running', f'{len(rows):,} distinct teams queued; 49 meta opponents.')
            for offset in range(0,len(rows),96):
                batch = rows[offset:offset+96]
                count = B.MDB.ensure(self.con,self.policy,batch,self.cols,1,seed=20261013,
                                      runner=runner,note='missing-training-labels')
                for tid in batch:
                    score = B.MDB.score(self.con,self.policy,tid,held=held)
                    assert score['coverage'] == score['columns'] > 0
                    for i in mapping[tid]:
                        self.con.execute('INSERT INTO label_value VALUES (?,?,?,?,?)',
                                         (i,tid,score['score'],score['battles'],self.policy))
                        self.apply(i,tid,score['score'],score['battles'],self.policy)
                        self.done += 1
                self.con.commit()
                self.battles += count
                completed_batches += 1
                self.publish('running',f'{self.done:,} of {self.total:,} missing labels filled.',checkpoint=completed_batches%10==0)
                print(f'Batch {completed_batches}: {count} battles; {self.done}/{self.total} labels',flush=True)
                if max_batches and completed_batches >= max_batches:
                    self.publish('paused','Requested batch limit reached.',checkpoint=True)
                    return
        remaining = int((~np.isfinite(self.d['W'])).sum())
        status = 'complete' if remaining == 0 else 'complete_with_exclusions'
        self.publish(status,f'{self.done:,} labels generated; {remaining:,} records remain unlabelable. See exclusions.',checkpoint=True)
        excluded = [dict(row=i,status=s,reason=e,source=p) for i,s,e,p in self.con.execute("SELECT row_id,status,error,source FROM label_job WHERE status!='ready'")]
        save(ROOT/'exclusions.json',excluded)
        assert sha(SOURCE) == self.source_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--background',action='store_true')
    ap.add_argument('--max-batches',type=int)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--chunk-size',type=int,default=96)
    args = ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.background:
        with open(ROOT/'job.log','a') as log:
            proc = subprocess.Popen(['caffeinate','-i',sys.executable,str(Path(__file__).resolve()),
                                     '--workers',str(args.workers),'--chunk-size',str(args.chunk_size)],
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':proc.pid,'status':str(ROOT/'status.json'),'log':str(ROOT/'job.log')}))
        return
    with open(ROOT/'job.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job = Job()
        def interrupt(*_):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupt)
        try:
            # Completed DB batches survive worker failures. Retry with fresh owned workers.
            for attempt in range(3):
                try:
                    job.run(args.max_batches,args.workers,args.chunk_size)
                    break
                except Exception as exc:
                    job.con.rollback()
                    job.d = {k: v.copy() for k, v in np.load(SOURCE).items()}
                    job.restore()
                    job.publish('retrying' if attempt<2 else 'failed',f'{type(exc).__name__}: {exc}',checkpoint=True)
                    print(f'Attempt {attempt+1} failed: {exc}',flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(5)
        except KeyboardInterrupt:
            job.publish('interrupted','Restart the same command to resume saved battles.',checkpoint=True)
            raise
        finally:
            job.con.close()


if __name__ == '__main__':
    main()
