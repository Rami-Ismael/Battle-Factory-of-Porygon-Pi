// Shared SVG emitters for the logo directions. Geometry is in mark units; ops are
// ['M',[x,y]] ['L',[x,y]] ['A',r,large,sweep,[x,y]] ['Z'].
import fs from 'node:fs';

const f = n => +n.toFixed(2);
const d = (ops, T, k) => ops.map(o => {
  if (o[0] === 'Z') return 'Z';
  if (o[0] === 'A') { const p = T(o[4]); return `A${f(o[1] * k)} ${f(o[1] * k)} 0 ${o[2]} ${o[3]} ${f(p[0])} ${f(p[1])}`; }
  const p = T(o[1]); return `${o[0]}${f(p[0])} ${f(p[1])}`;
}).join(' ');

export function writeMark(file, shapes, bbox, pad = 0) {
  const [x0, y0, x1, y1] = bbox;
  const T = p => [p[0] - x0 + pad, p[1] - y0 + pad];
  const w = x1 - x0 + 2 * pad, h = y1 - y0 + 2 * pad, k = 1;
  const body = shapes.map(s => `<path d="${d(s, T, k)}"/>`).join('');
  fs.writeFileSync(file, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${f(w)} ${f(h)}" fill="#111">${body}</svg>\n`);
}

// Clip the infinite line through a,b to rect r (Liang–Barsky).
function clip(a, b, r) {
  const [x0, y0, x1, y1] = r, dx = b[0] - a[0], dy = b[1] - a[1];
  let t0 = -1e9, t1 = 1e9;
  for (const [p, q] of [[-dx, a[0] - x0], [dx, x1 - a[0]], [-dy, a[1] - y0], [dy, y1 - a[1]]]) {
    if (p === 0) { if (q < 0) return null; continue; }
    const t = q / p; if (p < 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
  }
  return t0 > t1 ? null : [[a[0] + t0 * dx, a[1] + t0 * dy], [a[0] + t1 * dx, a[1] + t1 * dy]];
}

const STYLE = {
  primary: 'fill="none" stroke="#333" stroke-width="2.4"',
  secondary: 'fill="none" stroke="#666" stroke-width="1.7" stroke-dasharray="8 6"',
  auxiliary: 'fill="none" stroke="#999" stroke-width="1.3" stroke-dasharray="2 5"',
};

// guides: {auxiliary:[], secondary:[], primary:[]}; each guide is
// {line:[a,b]} (extended to the frame), {seg:[a,b]}, {circle:[c,r]} or {poly:[pts]}.
export function writeBoard(file, shapes, bbox, guides, anchors) {
  const W = 2560, H = 1440, [x0, y0, x1, y1] = bbox, mh = y1 - y0;
  const k = (0.43 * H) / mh, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const T = p => [W / 2 + (p[0] - cx) * k, H / 2 + (p[1] - cy) * k];
  const e = 0.15 * mh, frame = [x0 - e, y0 - e, x1 + e, y1 + e];
  const el = g => {
    if (g.circle) { const c = T(g.circle[0]); return `<circle cx="${f(c[0])}" cy="${f(c[1])}" r="${f(g.circle[1] * k)}"/>`; }
    if (g.poly) return `<polygon points="${g.poly.map(p => T(p).map(f).join(',')).join(' ')}"/>`;
    const s = g.seg ?? clip(g.line[0], g.line[1], frame); if (!s) return '';
    const [a, b] = s.map(T); return `<line x1="${f(a[0])}" y1="${f(a[1])}" x2="${f(b[0])}" y2="${f(b[1])}"/>`;
  };
  const tier = n => `<g id="${n}" ${STYLE[n]}>${(guides[n] ?? []).map(el).join('')}</g>`;
  const mark = `<g id="mark" fill="rgba(0,0,0,0.10)" stroke="#222" stroke-width="2.6" stroke-linejoin="round">${shapes.map(s => `<path d="${d(s, T, k)}"/>`).join('')}</g>`;
  const corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(T)
    .map(([x, y]) => `<rect x="${f(x - 5)}" y="${f(y - 5)}" width="10" height="10"/>`).join('');
  const dots = anchors.points.map(T).map(([x, y]) => `<circle cx="${f(x)}" cy="${f(y)}" r="7"/>`).join('');
  const c = T(anchors.centre);
  const centre = `<circle cx="${f(c[0])}" cy="${f(c[1])}" r="9"/><line x1="${f(c[0] - 16)}" y1="${f(c[1])}" x2="${f(c[0] + 16)}" y2="${f(c[1])}"/><line x1="${f(c[0])}" y1="${f(c[1] - 16)}" x2="${f(c[0])}" y2="${f(c[1] + 16)}"/>`;
  const anc = `<g id="anchors" fill="#FAFAFA" stroke="#333" stroke-width="1.6">${corners}${dots}${centre}</g>`;
  fs.writeFileSync(file, `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="#FAFAFA"/>`
    + tier('auxiliary') + tier('secondary') + tier('primary') + mark + anc + '</svg>\n');
}

export const bboxOf = pts => [Math.min(...pts.map(p => p[0])), Math.min(...pts.map(p => p[1])),
  Math.max(...pts.map(p => p[0])), Math.max(...pts.map(p => p[1]))];
