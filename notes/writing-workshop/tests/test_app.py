import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app import create_app


class WorkshopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'test.sqlite3'
        self.app = create_app(self.path)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def row(self, sql, params=()):
        db = sqlite3.connect(self.path)
        try:
            db.row_factory = sqlite3.Row
            return dict(db.execute(sql, params).fetchone())
        finally:
            db.close()

    def payload(self, doc_id=1):
        row = self.row('SELECT * FROM documents WHERE id=?', (doc_id,))
        return {key: json.loads(row[key]) if key in ('content', 'comments') else row[key] for key in ('title', 'content', 'comments', 'version')}

    def test_save_atomic_document_and_annotations_and_reject_stale_tab(self):
        payload = self.payload()
        payload['title'] = 'A revised opening'
        payload['comments'][0]['status'] = 'resolved'
        payload['content']['content'].append({'type':'paragraph', 'content':[{'type':'text','text':'New ending.'}]})
        response = self.client.put('/api/documents/1', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['version'], 2)
        saved = self.payload()
        self.assertEqual(saved['comments'][0]['status'], 'resolved')
        self.assertEqual(saved['content'], payload['content'])
        payload['title'] = 'Stale overwrite'
        self.assertEqual(self.client.put('/api/documents/1', json=payload).status_code, 409)
        self.assertEqual(self.payload()['title'], 'A revised opening')

    def test_restore_keeps_current_draft_and_restores_comments(self):
        original = self.payload()
        payload = self.payload()
        payload.update(title='Second draft', comments=[])
        self.client.put('/api/documents/1', json=payload)
        response = self.client.post('/api/documents/1/revisions/1/restore', json={'version':2})
        self.assertEqual(response.status_code, 200)
        restored = self.payload()
        self.assertEqual(restored['content'], original['content'])
        self.assertEqual(restored['comments'], original['comments'])
        self.assertEqual(restored['title'], original['title'])
        prior = self.row("SELECT * FROM revisions WHERE label='Before restore'")
        self.assertEqual(prior['title'], 'Second draft')
        self.assertEqual(json.loads(prior['comments']), [])
        self.assertEqual(self.client.post('/api/documents/1/revisions/1/restore', json={'version':2}).status_code, 409)

    def test_checkpoint_and_major_flag_even_without_text_changes(self):
        payload = self.payload()
        self.client.put('/api/documents/1', json=payload)
        self.assertEqual(self.row('SELECT count(*) AS count FROM revisions WHERE document_id=1')['count'], 1)
        payload.update(checkpoint=True, label='Structural rewrite', major=True)
        self.client.put('/api/documents/1', json=payload)
        revision = self.row('SELECT * FROM revisions WHERE document_id=1 ORDER BY id DESC')
        self.assertEqual(revision['major'], 1)
        self.client.post(f"/api/documents/1/revisions/{revision['id']}/flag")
        self.assertEqual(self.row('SELECT major FROM revisions WHERE id=?', (revision['id'],))['major'], 0)

    def test_documents_isolate_history_and_escape_titles(self):
        response = self.client.post('/documents', data={'title':'<script>alert(1)</script>'})
        self.assertEqual(response.headers['HX-Redirect'], '/documents/3')
        html = self.client.get('/documents/3').text
        self.assertIn('&lt;script&gt;', html)
        self.assertEqual(self.client.get('/api/documents/3/revisions/1').status_code, 404)
        self.assertEqual(self.client.post('/api/documents/3/revisions/1/restore', json={'version':1}).status_code, 404)
        self.assertEqual(self.client.get('/documents/999').status_code, 404)

    def test_invalid_document_is_not_saved(self):
        self.assertEqual(self.client.put('/api/documents/1', json={'content':{},'comments':[]}).status_code, 400)
        self.assertEqual(self.payload()['version'], 1)


if __name__ == '__main__':
    unittest.main()
