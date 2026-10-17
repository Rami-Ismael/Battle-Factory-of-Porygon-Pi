"""Build noise-to-archetypes.html, one offline file: python3 build.py

Reads parts/ (page shell, CSS, markup, scripts) and data/figdata.js + data/sprites.js
(written by prep.py) and inlines them all."""
from pathlib import Path
H = Path(__file__).resolve().parent
P, D = H / 'parts', H / 'data'
rd = lambda p: p.read_text()
esc = lambda s: s.replace('</script', '<\\/script')
out = rd(P / 'page.html')
for k, v in [('__TITLE__', rd(P / 'title.txt').strip()), ('__BASE_CSS__', rd(P / 'base.css')),
             ('__FIG_CSS__', rd(P / 'style.css')), ('__MOTION_CSS__', rd(P / 'motion.css')),
             ('__FIG_BODY__', rd(P / 'body.html')),
             ('__FIGDATA__', esc(rd(D / 'figdata.js'))), ('__SPRITES__', esc(rd(D / 'sprites.js'))),
             ('__MOTION_HELPER__', esc(rd(P / 'motion-helper.js'))), ('__APP_JS__', esc(rd(P / 'app.js'))),
             ('__MOTION_JS__', esc(rd(P / 'motion.js'))), ('__PROTO_JS__', esc(rd(P / 'runtime.js')))]:
    assert k in out, k
    out = out.replace(k, v)
(H / 'noise-to-archetypes.html').write_text(out)
print('built noise-to-archetypes.html', len(out) // 1024, 'KB')
