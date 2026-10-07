import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import create_app

with tempfile.TemporaryDirectory() as directory:
    create_app(Path(directory) / 'browser-tests.sqlite3').run(host='127.0.0.1', port=5058)
