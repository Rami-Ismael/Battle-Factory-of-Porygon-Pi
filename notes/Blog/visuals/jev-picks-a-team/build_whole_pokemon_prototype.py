"""Assemble whole-pokemon.prototype.html (PROTOTYPE, throwaway): the template plus inline data.

    python3 build_whole_pokemon_prototype.py

Inlines whole-pokemon-trace.json and the sprites the figure needs: species from
../diversity-methods/sprites.json and items from ../item-sprites/ (Showdown sprites).
"""
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def main():
    trace = json.loads((HERE / 'whole-pokemon-trace.json').read_text())
    sprites = json.loads((HERE.parent / 'diversity-methods/sprites.json').read_text())
    species = {m['species'] for m in trace['team']} | {q['pick'] for q in trace['questions'] if q['field'] == ['species']}
    items = {m['item'] for m in trace['team'] if m['item']}
    data = {'trace': trace,
            'species': {s: sprites[norm(s)] for s in species if norm(s) in sprites},
            'items': {i: 'data:image/png;base64,' + base64.b64encode(
                (HERE.parent / f'item-sprites/item-{norm(i)}.png').read_bytes()).decode() for i in items}}
    page = (HERE / 'whole-pokemon.prototype.template.html').read_text()
    out = page.replace('/*DATA*/null', json.dumps(data, separators=(',', ':'), ensure_ascii=False))
    (HERE / 'whole-pokemon.prototype.html').write_text(out)
    print(f"{len(data['species'])} species sprites, {len(data['items'])} item sprites, {len(out) // 1024} KB")


if __name__ == '__main__':
    main()
