// Direction 02 — the counter in the bite. The meta is a disc; the counter is the piece
// cut to fit the gap it leaves. System: tangent circles of related radii, U module.
// Meta disc R = 3U; bite circle r = 2U centred on the meta's rim; the counter sits
// in the bite, one gap smaller.
import { writeMark, writeBoard, bboxOf } from '../lib.mjs';
import { fileURLToPath } from 'node:url';
const here = fileURLToPath(new URL('.', import.meta.url));

const U = 100, R = 3 * U, r = 2 * U;
const GAP = U / 2;               // gap between meta and counter; the 16 px icon is ~8U wide, so this keeps 1 px open
const O = [0, 0], C = [R, 0];    // bite centre on the rim
// Optical corrections
const COUNTER_OVERSHOOT = 0.02 * 2 * R; // the counter's round edge overshoots the meta's vertical extent a little less than it would read; enlarge by 2% of mark height
const TIP_BLUNT = 0.012 * 2 * R;        // crescent tips fill in at 16 px; pull each tip back along the meta rim

// Intersection of the meta circle and the bite circle.
const px = (R * R - r * r + R * R) / (2 * R), py = Math.sqrt(R * R - px * px);
const P1 = [px, -py], P2 = [px, py];
const rot = (p, a) => [p[0] * Math.cos(a) - p[1] * Math.sin(a), p[0] * Math.sin(a) + p[1] * Math.cos(a)];
const da = TIP_BLUNT / R;
const T1 = rot(P1, -da), T2 = rot(P2, da);
// Where the bite circle meets the blunted tips: same angle on the meta rim, project to the bite circle.
const onBite = p => { const v = [p[0] - C[0], p[1] - C[1]], n = Math.hypot(...v); return [C[0] + v[0] / n * r, C[1] + v[1] / n * r]; };
const B1 = onBite(T1), B2 = onBite(T2);
const meta = [['M', T1], ['A', R, 1, 0, T2], ['L', B2], ['A', r, 0, 1, B1], ['Z']];
const rc = r - GAP + COUNTER_OVERSHOOT / 2;
const counter = [['M', [C[0] - rc, 0]], ['A', rc, 0, 1, [C[0] + rc, 0]], ['A', rc, 0, 1, [C[0] - rc, 0]], ['Z']];
const shapes = [meta, counter];
const bbox = [-R, -R, C[0] + rc, R];
writeMark(here + 'mark.svg', shapes, bbox, 2);

const grid = [];
for (let x = -3; x <= 5; x++) grid.push({ line: [[x * U, -R], [x * U, R]] });
for (let y = -3; y <= 3; y++) grid.push({ line: [[-R, y * U], [5 * U, y * U]] });
writeBoard(here + 'board.svg', shapes, bbox, {
  auxiliary: [...grid, { circle: [O, r] }],                          // U module grid; 2U circle shows bite radius = 2U
  secondary: [{ circle: [C, r] }, { seg: [P1, P2] }], // bite circle, common chord
  primary: [{ line: [O, C] }, { seg: [O, P1] }, { seg: [C, P1] }, { seg: [O, P2] }, { seg: [C, P2] }], // centre line and the 3U–2U radii triangle
}, { centre: O, points: [C, P1, P2, [C[0] + rc, 0]] });
console.log('wrote mark.svg, board.svg');
