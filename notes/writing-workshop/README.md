# Margin

A local writing workshop built with Python/Flask, SQLite, HTMX, locally compiled Tailwind, and Tiptap/ProseMirror. Browser assets are bundled locally; no CDN or remote fonts are required.

## Run

```sh
cd writing-workshop
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npm run build
.venv/bin/python app.py
```

Open http://127.0.0.1:5057. Set `PORT` to change the port. The first launch creates two example documents; create your own with **New document**.

## Workshop

- Write with headings, bold, italic, lists, quotes, Markdown shortcuts, and undo/redo. Type `/` at the start of a block to pick a paragraph, heading, list, quote, or divider; use the arrow keys and Enter to choose.
- Select text, then **Highlight & note**. Pick an editing pass, add commentary, and optionally suggest replacement wording.
- Click a highlight or margin note to move between them. The arrows cycle through open suggestions in the selected pass.
- Accept suggested wording, or resolve/reopen a note. If you delete its passage, the note remains in the margin with a detached label.
- Drafts autosave after one second of inactivity. Save manually with Ctrl/⌘ S. Save status reports failures; unsaved navigation is blocked when saving fails.
- **Save revision** adds a named checkpoint, optionally flagged as major. **History** lets you preview, flag/unflag, and restore any saved version. Restoring preserves the current draft first, then restores text and commentary together.

SQLite stores documents and immutable revision snapshots in `instance/workshop.sqlite3`. Every changed autosave creates a revision. Back up that file to preserve your writing. Version checks reject concurrent overwrites from another tab. This first version is intended for one person on localhost; it has no accounts, sharing, or automated editorial suggestions.

## Development and verification

```sh
npm run watch:css
npm run watch:js
# in separate terminals
.venv/bin/python -m unittest discover -s tests
npx playwright install chromium
npx playwright test
```

The editor uses a custom mark carrying an annotation UUID; ProseMirror keeps marks attached when surrounding text moves. Document creation and revision history use HTMX. Editor state uses JSON requests because prose and its annotations must save atomically.

Upstream integration references: [Tiptap vanilla JavaScript](https://tiptap.dev/docs/editor/getting-started/install/vanilla-javascript), [Tailwind CLI](https://tailwindcss.com/docs/installation/tailwind-cli), [HTMX](https://htmx.org/docs/).
