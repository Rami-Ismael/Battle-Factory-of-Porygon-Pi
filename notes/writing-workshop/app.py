"""Margin: a local, single-user writing workshop."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

ROOT = Path(__file__).parent


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def paragraph(text, annotation=None):
    node = {"type": "text", "text": text}
    if annotation:
        node["marks"] = [{"type": "annotation", "attrs": {"id": annotation}}]
    return {"type": "paragraph", "content": [node]}


def create_app(database=None):
    app = Flask(__name__)
    app.config["DATABASE"] = str(database or ROOT / "instance" / "workshop.sqlite3")
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect():
        db = sqlite3.connect(app.config["DATABASE"])
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                comments TEXT NOT NULL DEFAULT '[]', version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS revisions (
                id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
                title TEXT NOT NULL, content TEXT NOT NULL, comments TEXT NOT NULL,
                label TEXT NOT NULL, major INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)
        if not db.execute("SELECT id FROM documents LIMIT 1").fetchone():
            content = {"type": "doc", "content": [
                paragraph("The house at the end of the road had been empty for eleven years. I knew this because my mother counted them, each autumn, as though the house were someone who had forgotten to come home."),
                paragraph("On the morning I returned, the windows were open.", "seed-1"),
                paragraph("I stopped the car where the tarmac gave way to gravel. Beyond the gate, the garden had grown into a small country of its own: nettles at the border, foxgloves in the shade, a pear tree leaning over the path. Nothing had been cut back. Nothing had been invited to stay."),
                paragraph("There was a blue cup on the kitchen sill. Steam lifted from it in a thin, almost invisible thread.", "seed-2"),
                paragraph("My mother had always said that a place could remember you. I had thought she meant the marks on the doorframe, the dent in the stair, the particular way a room holds the afternoon. Standing at that gate, I began to think she had meant something else."),
                paragraph("I left the engine running.")
            ]}
            comments = [
                {"id": "seed-1", "quote": "On the morning I returned, the windows were open.", "body": "This is where the story opens its eyes. Give this turn a little room to breathe; the short paragraph is doing good work.", "replacement": "", "pass": "Structure", "status": "open"},
                {"id": "seed-2", "quote": "There was a blue cup on the kitchen sill. Steam lifted from it in a thin, almost invisible thread.", "body": "Try a more direct opening. The cup and the steam already tell us someone is here.", "replacement": "A blue cup sat on the kitchen sill. A thin thread of steam rose from it.", "pass": "Line edit", "status": "open"}
            ]
            for title, body, notes in [("The house remembers", content, comments), ("Notes on paying attention", {"type": "doc", "content": [paragraph("Attention is a kind of hospitality. To notice something is to make room for it."), paragraph("What do we miss when we hurry? Begin here.")]}, [])]:
                cur = db.execute("INSERT INTO documents(title,content,comments,updated_at) VALUES(?,?,?,?)", (title, json.dumps(body), json.dumps(notes), now()))
                db.execute("INSERT INTO revisions(document_id,title,content,comments,label,major,created_at) SELECT id,title,content,comments,'First draft',1,updated_at FROM documents WHERE id=?", (cur.lastrowid,))

    def get_doc(db, doc_id):
        doc = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not doc:
            abort(404)
        return doc

    def listing(db):
        return db.execute("SELECT id,title,updated_at FROM documents ORDER BY updated_at DESC,id ASC").fetchall()

    def snapshot(db, doc_id, label, major=False):
        db.execute("INSERT INTO revisions(document_id,title,content,comments,label,major,created_at) SELECT id,title,content,comments,?,?,? FROM documents WHERE id=?", (label, int(major), now(), doc_id))

    @app.get("/")
    def home():
        with connect() as db:
            return redirect(url_for("document", doc_id=listing(db)[0]["id"]))

    @app.get("/documents/<int:doc_id>")
    def document(doc_id):
        with connect() as db:
            doc = get_doc(db, doc_id)
            state = dict(doc)
            state["content"] = json.loads(doc["content"])
            state["comments"] = json.loads(doc["comments"])
            count = db.execute("SELECT count(*) FROM revisions WHERE document_id=?", (doc_id,)).fetchone()[0]
            return render_template("index.html", doc=doc, documents=listing(db), state=state, revision_count=count)

    @app.post("/documents")
    def create_document():
        title = request.form.get("title", "Untitled document").strip()[:200] or "Untitled document"
        with connect() as db:
            cur = db.execute("INSERT INTO documents(title,content,updated_at) VALUES(?,?,?)", (title, json.dumps({"type": "doc", "content": [{"type": "paragraph"}]}), now()))
            snapshot(db, cur.lastrowid, "First draft", True)
            location = url_for("document", doc_id=cur.lastrowid)
        return "", 200, {"HX-Redirect": location}

    @app.put("/api/documents/<int:doc_id>")
    def save(doc_id):
        data = request.get_json()
        if not isinstance(data, dict) or not isinstance(data.get("content"), dict) or data["content"].get("type") != "doc" or not isinstance(data.get("comments"), list):
            return jsonify(error="Invalid document"), 400
        title = str(data.get("title", "")).strip()[:200] or "Untitled document"
        content, comments = json.dumps(data["content"]), json.dumps(data["comments"])
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            doc = get_doc(db, doc_id)
            if data.get("version") != doc["version"]:
                return jsonify(error="This document changed in another tab. Copy your latest text before reloading."), 409
            changed = (title, data["content"], data["comments"]) != (doc["title"], json.loads(doc["content"]), json.loads(doc["comments"]))
            if changed:
                db.execute("UPDATE documents SET title=?,content=?,comments=?,version=version+1,updated_at=? WHERE id=?", (title, content, comments, now(), doc_id))
            if changed or data.get("checkpoint"):
                snapshot(db, doc_id, str(data.get("label") or "Autosave")[:200], bool(data.get("major")))
            return jsonify(version=doc["version"] + int(changed), saved_at=now())

    @app.get("/documents/<int:doc_id>/history")
    def history(doc_id):
        with connect() as db:
            get_doc(db, doc_id)
            revisions = db.execute("SELECT id,label,major,created_at,title FROM revisions WHERE document_id=? ORDER BY id DESC", (doc_id,)).fetchall()
            return render_template("history.html", revisions=revisions, doc_id=doc_id)

    @app.get("/api/documents/<int:doc_id>/revisions/<int:revision_id>")
    def revision(doc_id, revision_id):
        with connect() as db:
            row = db.execute("SELECT * FROM revisions WHERE id=? AND document_id=?", (revision_id, doc_id)).fetchone()
            if not row:
                abort(404)
            data = dict(row)
            for field in ("content", "comments"):
                data[field] = json.loads(data[field])
            return jsonify(data)

    @app.post("/api/documents/<int:doc_id>/revisions/<int:revision_id>/flag")
    def flag(doc_id, revision_id):
        with connect() as db:
            row = db.execute("UPDATE revisions SET major=1-major WHERE id=? AND document_id=?", (revision_id, doc_id))
            if not row.rowcount:
                abort(404)
        return "", 204

    @app.post("/api/documents/<int:doc_id>/revisions/<int:revision_id>/restore")
    def restore(doc_id, revision_id):
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            doc = get_doc(db, doc_id)
            if (request.get_json() or {}).get("version") != doc["version"]:
                return jsonify(error="This document changed in another tab. Reload before restoring."), 409
            row = db.execute("SELECT * FROM revisions WHERE id=? AND document_id=?", (revision_id, doc_id)).fetchone()
            if not row:
                abort(404)
            snapshot(db, doc_id, "Before restore")
            db.execute("UPDATE documents SET title=?,content=?,comments=?,version=version+1,updated_at=? WHERE id=?", (row["title"], row["content"], row["comments"], now(), doc_id))
            snapshot(db, doc_id, "Restored: " + row["label"], True)
        return jsonify(ok=True)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5057)), debug=False)
