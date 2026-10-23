"""Every PyTorch checkpoint load goes through src/checkpoints.py.

Ruff's TID251 ban sees `torch.load` only when `torch` is imported by that name;
a call through a module alias such as `E.torch.load(...)` slips past it. This
test reads the code itself, so any `torch.load(` call outside the loader fails,
whatever object it is reached through. Comments and strings do not count.
"""
import io
import tokenize
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ALLOWED = {CODE/'src'/'checkpoints.py'}
SKIP = {'.venv', 'venv', 'node_modules', '__pycache__'}


def direct_loads(source):
    """Line numbers of `torch.load(` calls, read as code tokens."""
    toks = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]
    return [a.start[0] for a, b, c, d in zip(toks, toks[1:], toks[2:], toks[3:])
            if (a.string, b.string, c.string, d.string) == ('torch', '.', 'load', '(')]


def python_files():
    for path in sorted(CODE.rglob('*.py')):
        if not SKIP & set(path.relative_to(CODE).parts):
            yield path


class TorchLoadBanTests(unittest.TestCase):
    def test_detector_sees_direct_and_aliased_calls(self):
        src = 'torch.load(p)\nE.torch.load(p)["sd"]\nself.torch . load (p)\n# torch.load(p)\ns = "torch.load(p)"\n'
        self.assertEqual(direct_loads(src), [1, 2, 3])

    def test_no_torch_load_outside_the_loader(self):
        found = [f'{path.relative_to(CODE)}:{line}' for path in python_files() if path not in ALLOWED
                 for line in direct_loads(path.read_text())]
        self.assertEqual(found, [], 'use load_checkpoint from src/checkpoints.py instead of torch.load')


if __name__ == '__main__':
    unittest.main()
