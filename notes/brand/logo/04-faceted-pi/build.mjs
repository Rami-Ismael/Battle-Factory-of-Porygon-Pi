// Direction 04 — faceted π. Porygon is a Pokémon built from polygons; this π is built from
// lattice triangles and leans like an italic math symbol, the way π is set for a policy.
// System: triangular lattice, U = lattice edge. Bar = one row of 11 triangles; legs run along
// the 120° lattice direction; the right foot kicks out by one triangle.
import { writeMark, writeBoard, bboxOf } from '../lib.mjs';
import { fileURLToPath } from 'node:url';
const here = fileURLToPath(new URL('.', import.meta.url));

const U = 100, S = Math.sqrt(3) / 2 * U;
const GAP = U / 8;               // facet gap; visible from 64 px up, fuses into one silhouette at 16 px
// Optical corrections
const POINT_OVERSHOOT = 0.02 * 4 * S;    // the bar's 60° end points read short; push them out 2% of mark height
const P = (a, b) => [a * U + b * U / 2, b * S];
const add = (p, q) => [p[0] + q[0], p[1] + q[1]], mul = (p, s) => [p[0] * s, p[1] * s];

const D = (a, b) => [[a, b], [a + 1, b], [a, b + 1]];      // down-pointing
const Up = (a, b) => [[a, b], [a, b + 1], [a - 1, b + 1]];  // up-pointing
const cells = [];
for (let a = 0; a < 6; a++) cells.push(D(a, 0));
for (let a = 1; a < 6; a++) cells.push(Up(a, 0));
for (const a0 of [1, 3]) for (let k = 0; k < 3; k++) cells.push(D(a0 - k, 1 + k), Up(a0 - k, 1 + k));
cells.push(Up(2, 3));                                         // right foot kick

const rho = U / (2 * Math.sqrt(3));
const ends = { '0,0': [-POINT_OVERSHOOT, 0], '6,0': [POINT_OVERSHOOT, 0] };
const tris = cells.map(c => {
  const pts = c.map(([a, b]) => add(P(a, b), ends[`${a},${b}`] ?? [0, 0]));
  const raw = c.map(([a, b]) => P(a, b));
  const I = mul(add(add(raw[0], raw[1]), raw[2]), 1 / 3);
  return pts.map(p => add(I, mul(add(p, mul(I, -1)), (rho - GAP / 2) / rho)));
});
const shapes = tris.map(t => [['M', t[0]], ['L', t[1]], ['L', t[2]], ['Z']]);
const bbox = bboxOf(tris.flat());
writeMark(here + 'mark.svg', shapes, bbox, 2);

const lineB = b => ({ line: [P(0, b), P(1, b)] });                 // 0° lattice line
const lineA = a => ({ line: [P(a, 0), P(a, 1)] });                 // 60° lattice line
const lineS = s => ({ line: [P(s, 0), P(s - 1, 1)] });             // 120° lattice line, a + b = s
const inc = c => mul(c.map(([a, b]) => P(a, b)).reduce(add), 1 / 3);
writeBoard(here + 'board.svg', shapes, bbox, {
  auxiliary: [
    ...[0, 1, 2, 3, 4, 5, 6].map(lineA),
    ...[2, 3].map(lineB),
    ...[1, 6, 7].map(lineS),
  ],
  secondary: [
    lineB(1),                                                        // bar bottom
    { circle: [inc(D(0, 0)), rho] }, { circle: [inc(D(5, 0)), rho] }, // facets are their incircle less GAP/2
    { circle: [inc(D(1, 1)), rho] }, { circle: [inc(Up(2, 3)), rho] },
  ],
  primary: [
    lineB(0), lineB(4),                                              // cap line and baseline
    ...[2, 3, 4, 5].map(lineS),                                      // leg edges along 120°
  ],
}, { centre: [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], points: [P(0, 0), P(6, 0), P(1, 1), P(2, 1), P(3, 1), P(4, 1), P(-2, 4), P(0, 4), P(2, 4)] });
console.log('wrote mark.svg, board.svg');
