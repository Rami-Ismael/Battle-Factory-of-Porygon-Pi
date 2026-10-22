"""Durable, fenced battle jobs over an authenticated loopback HTTP transport.

Prototype for multiple machines: only local transport is exercised here. Job
identities include the runtime fingerprint and complete immutable payload.
Completion is idempotent; expired workers cannot overwrite a later lease.
"""
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import Request,urlopen


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


class LeaseQueue:
    def __init__(self,path,runtime,clock=time.time):
        self.path=str(path);self.runtime=runtime;self.clock=clock
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            old=db.execute("SELECT value FROM metadata WHERE key='runtime'").fetchone()
            if old and old[0]!=runtime:raise ValueError('Queue runtime fingerprint differs')
            db.execute("INSERT OR IGNORE INTO metadata VALUES('runtime',?)",(runtime,))
            db.execute('''CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY,payload TEXT NOT NULL,state TEXT NOT NULL,
                worker TEXT,token TEXT,expires REAL,attempts INTEGER NOT NULL DEFAULT 0,
                result TEXT,result_hash TEXT)''')

    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.path,timeout=30)
        try:
            with db:yield db
        finally:db.close()

    def submit(self,payload):
        identity=sha({'runtime':self.runtime,'payload':payload})
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO jobs(id,payload,state) VALUES(?,?,?)',
                       (identity,canonical(payload),'queued'))
        return {'id':identity}

    def claim(self,worker,runtime,seconds=60):
        if runtime!=self.runtime:raise ValueError('Worker runtime fingerprint differs')
        if not worker or not 0<seconds<=3600:raise ValueError('Invalid worker or lease duration')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            now=self.clock()
            row=db.execute("SELECT id,payload,attempts FROM jobs WHERE state='queued' OR (state='leased' AND expires<=?) ORDER BY id LIMIT 1",(now,)).fetchone()
            if row is None:return None
            token=secrets.token_hex(24)
            db.execute("UPDATE jobs SET state='leased',worker=?,token=?,expires=?,attempts=attempts+1 WHERE id=?",
                       (worker,token,now+seconds,row[0]))
            return {'id':row[0],'payload':json.loads(row[1]),'token':token,
                    'expires':now+seconds,'attempt':row[2]+1}

    def heartbeat(self,identity,token,seconds=60):
        if not 0<seconds<=3600:raise ValueError('Invalid lease duration')
        with self.connection() as db:
            now=self.clock()
            changed=db.execute("UPDATE jobs SET expires=? WHERE id=? AND state='leased' AND token=? AND expires>?",
                               (now+seconds,identity,token,now)).rowcount
            if changed!=1:raise ValueError('Lease expired or fenced out')
        return {'expires':now+seconds}

    def complete(self,identity,token,result):
        fingerprint=sha(result)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT state,token,expires,result_hash FROM jobs WHERE id=?',(identity,)).fetchone()
            if row is None:raise ValueError('Unknown job')
            if row[0]=='complete':
                if row[1]==token and row[3]==fingerprint:return {'accepted':True,'duplicate':True}
                raise ValueError('Conflicting completion')
            if row[0]!='leased' or row[1]!=token or row[2]<=self.clock():
                raise ValueError('Lease expired or fenced out')
            db.execute("UPDATE jobs SET state='complete',result=?,result_hash=? WHERE id=?",
                       (canonical(result),fingerprint,identity))
        return {'accepted':True,'duplicate':False}

    def status(self):
        with self.connection() as db:
            counts=dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
            attempts=db.execute('SELECT coalesce(sum(attempts),0) FROM jobs').fetchone()[0]
        return {'counts':counts,'attempts':attempts,'runtime':self.runtime}

    def results(self):
        with self.connection() as db:
            return [{'id':identity,'payload':json.loads(payload),'result':json.loads(result)}
                    for identity,payload,result in db.execute("SELECT id,payload,result FROM jobs WHERE state='complete' ORDER BY id")]


def http_server(queue,token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16*1024*1024:raise ValueError('Invalid request size')
                args=json.loads(self.rfile.read(size))
                methods={'/submit':queue.submit,'/claim':queue.claim,'/heartbeat':queue.heartbeat,
                         '/complete':queue.complete,'/status':queue.status}
                if self.path not in methods:raise ValueError('Unknown operation')
                value=methods[self.path](**args);status=200
            except (ValueError,TypeError,json.JSONDecodeError) as exc:
                value={'error':str(exc)};status=409
            except Exception:
                value={'error':'Queue operation failed'};status=500
            payload=canonical(value).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    return ThreadingHTTPServer(('127.0.0.1',0),Handler)


class Client:
    def __init__(self,url,token):self.url=url;self.token=token
    def call(self,operation,**args):
        request=Request(self.url+'/'+operation,data=canonical(args).encode(),
            headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        with urlopen(request,timeout=30) as response:return json.load(response)
