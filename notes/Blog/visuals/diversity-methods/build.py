#!/usr/bin/env python3
"""Build diversity-methods.html ("Keeping Teams Diverse") from template.html.

Inlines data.json (made by build_data.py) and sprites.json (made by build_sprites.py) into the two
<script type="application/json"> blocks of the template. Writes one self-contained file next to this
script. No network, no new battles.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SLOTS = {"/*__DATA__*/null": "data.json", "/*__SPRITES__*/null": "sprites.json"}


def blob(name):
    """Minified JSON, safe inside <script>: '</' cannot close the element."""
    obj = json.loads((HERE / name).read_text(encoding="utf-8"))
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def main():
    html = (HERE / "template.html").read_text(encoding="utf-8")
    for marker, name in SLOTS.items():
        assert html.count(marker) == 1, f"template must hold {marker} exactly once"
        html = html.replace(marker, blob(name))
    out = HERE / "diversity-methods.html"
    out.write_text(html, encoding="utf-8")
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    sprites = json.loads((HERE / "sprites.json").read_text(encoding="utf-8"))
    print(f"wrote {out.name}: {out.stat().st_size:,} bytes; {len(data['species'])} species, {len(sprites)} sprites")


if __name__ == "__main__":
    main()
