"""Small isolated databases drawn read-only from real legal Mask Lab teams."""
import json
import sqlite3
from pathlib import Path
import mask_lab_battle as pipeline


def create_fixture(folder, candidates=6, opponents=4):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    mdb=pipeline.MDB
    src=sqlite3.connect('file:'+str(pipeline.PRIVATE_DB)+'?mode=ro',uri=True)
    try:
        held=mdb.H.team_ids(src)
        cols=[]
        for identity,base in src.execute("SELECT t.team_id,t.base_key FROM team t JOIN team_set s USING(team_id) WHERE s.set_name='top50' AND t.legal=1 ORDER BY t.team_id"):
            if identity not in held:
                cols.append(identity)
                if len(cols)==opponents:break
        rows=[identity for identity, in src.execute("SELECT team_id FROM team WHERE origin='llm-completion:maskLab' AND legal=1 ORDER BY team_id")
              if identity not in held and identity not in cols][:candidates]
        if len(rows)!=candidates or len(cols)!=opponents:
            raise ValueError('Not enough legal non-held-out teams for the fixture')
        con=mdb.connect(folder/'matrix.sqlite')
        fields=[r[1] for r in con.execute('PRAGMA table_info(team)')]
        query='SELECT '+','.join(fields)+' FROM team WHERE team_id=?'
        with con:
            for identity in rows+cols:
                con.execute('INSERT INTO team('+','.join(fields)+') VALUES ('+','.join('?' for _ in fields)+')',src.execute(query,(identity,)).fetchone())
            for identity in cols:con.execute("INSERT INTO team_set VALUES ('top50',?,1)",(identity,))
        mdb.get_policy(con,row_policy='fixture-original',simulator='fixture')
        records=[{'label':'integration-'+str(identity),'paste':con.execute('SELECT paste FROM team WHERE team_id=?',(identity,)).fetchone()[0]} for identity in rows]
        (folder/'candidates.json').write_text(json.dumps({'candidates':records},indent=2)+'\n')
        con.close()
        return rows,cols
    finally:
        src.close()
