# Battle Factory of Porygon-Pi

Searching the space of competitive Pokémon VGC teams (Champions Reg Set M-B) for teams that beat a meta, using a diffusion proposer, a battle simulator and a surrogate.

Built on [poke-env](https://github.com/hsahovic/poke-env) and benchmarked against [VGC-Bench](https://github.com/cameronangliss/vgc-bench):

[![poke-env on PyPI](https://img.shields.io/pypi/v/poke-env)](https://pypi.org/project/poke-env/)
[![poke-env Python versions](https://img.shields.io/pypi/pyversions/poke-env)](https://pypi.org/project/poke-env/)
[![poke-env licence: MIT](https://img.shields.io/badge/poke--env-MIT-yellow)](https://opensource.org/licenses/MIT)
[![poke-env CI](https://github.com/hsahovic/poke-env/actions/workflows/tests.yml/badge.svg)](https://github.com/hsahovic/poke-env/actions/workflows/tests.yml)
[![VGC-Bench CI](https://github.com/cameronangliss/vgc-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/cameronangliss/vgc-bench/actions/workflows/tests.yml)
[![VGC-Bench Python 3.10–3.14](https://img.shields.io/badge/vgc--bench-python%203.10%E2%80%933.14-blue)](https://github.com/cameronangliss/vgc-bench)
[![VGC-Bench licence: MIT](https://img.shields.io/badge/vgc--bench-MIT-green)](https://github.com/cameronangliss/vgc-bench/blob/main/LICENSE)
[![VGC-Bench on arXiv](https://img.shields.io/badge/arXiv-2506.10326-b31b1b)](https://arxiv.org/abs/2506.10326)

| Folder | What it is |
| --- | --- |
| [`notes/`](notes) | The research notes: Markdown, Obsidian-flavoured (`[[links]]`), plus the blog and its interactive figures. |
| [`code/`](code) | The pipeline: team generator, battle runner, search loops, GUI. Start at [`code/README.md`](code/README.md). |
| [`tools/sync.py`](tools/sync.py) | Copies both folders from the author's laptop and pushes. |

## Where the truth lives

The source of truth is the Obsidian vault and the code folder **on the author's laptop**. This repository is a published mirror of them, one-way:

```
Obsidian vault  ──sync──▶  notes/
code folder     ──sync──▶  code/
```

An edit in Obsidian shows up here as the same edit in the next sync. An edit made on github.com is overwritten by that sync, so open a note locally instead.

Ignored files (`.obsidian`, virtual environments, battle results, files over 5 MB) never leave the laptop, so the large datasets under `code/` are not included.

```bash
tools/sync.py --dry-run   # show what would change
tools/sync.py             # mirror, commit, push
```
