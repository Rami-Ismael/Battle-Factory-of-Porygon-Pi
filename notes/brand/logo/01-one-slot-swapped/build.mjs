// Direction 01 — one slot swapped. A team is six slots; the search edits one at a time.
// System: triangular lattice. U = lattice edge = hexagon side. Six equilateral triangles
// meet at the centre; one is pushed out along its bisector, the move the search makes.
import { writeMark, writeBoard, bboxOf } from '../lib.mjs';
import { fileURLToPath } from 'node:url';
const here = fileURLToPath(new URL('.', import.meta.url));

const U = 100;
const GAP = U / 8;              // gap between slots; keeps about 1 px open at 16 px
const PUSH = U / 4;             // how far the edited slot moves out along its bisector
const SWAP = 5;                 // index of the edited slot (upper right, bisector at −30°)
// Optical corrections
const POINT_OVERSHOOT = 0.02 * Math.sqrt(3) * U; // side points overshoot the flat top and bottom edges
const LIFT = -0.01 * Math.sqrt(3) * U;           // centre by visual mass: the pushed slot pulls the eye up, so the board centres on the bbox and the mark sits 1% higher

const O = [0, 0];
const V = Array.from({ length: 6 }, (_, i) => [U * Math.cos(i * Math.PI / 3), U * Math.sin(i * Math.PI / 3)]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1]], mul = (a, s) => [a[0] * s, a[1] * s];
const rho = U / (2 * Math.sqrt(3));                 // inradius of each lattice triangle
const inc = i => mul(add(V[i], V[(i + 1) % 6]), 1 / 3); // incentre (= centroid) of triangle i
const inset = (tri, I) => tri.map(p => add(I, mul(add(p, mul(I, -1)), (rho - GAP / 2) / rho)));
const bis = i => [Math.cos((i + 0.5) * Math.PI / 3), Math.sin((i + 0.5) * Math.PI / 3)];

const tris = [], raw = [];
for (let i = 0; i < 6; i++) {
  const Vc = V.map((p, j) => (j === 0 || j === 3) ? [p[0] + Math.sign(p[0]) * POINT_OVERSHOOT, p[1]] : p);
  let t = inset([O, Vc[i], Vc[(i + 1) % 6]], inc(i));
  if (i === SWAP) t = t.map(p => add(p, mul(bis(i), PUSH)));
  t = t.map(p => [p[0], p[1] + LIFT]);
  tris.push(t);
  raw.push([O, V[i], V[(i + 1) % 6]]);
}
const shapes = tris.map(t => [['M', t[0]], ['L', t[1]], ['L', t[2]], ['Z']]);
const bbox = bboxOf(tris.flat());
writeMark(here + 'mark.svg', shapes, bbox, 2);

const ghost = raw[SWAP];
const moved = ghost.map(p => add(p, mul(bis(SWAP), PUSH)));
writeBoard(here + 'board.svg', shapes, bbox, {
  auxiliary: [
    ...V.map((p, i) => ({ line: [p, V[(i + 1) % 6]] })),            // lattice: extended hexagon edges
    ...[0, 1, 2, 3, 4, 5].map(i => ({ circle: [inc(i), rho] })),     // incircles: each slot is inset from them by GAP/2
  ],
  secondary: [
    { circle: [O, U] },                                               // circumcircle through the six lattice nodes
    { circle: [O, U * Math.sqrt(3) / 2] },                            // incircle of the hexagon
    { poly: ghost },                                                  // the edited slot before the move
  ],
  primary: [
    { line: [V[0], V[3]] }, { line: [V[1], V[4]] }, { line: [V[2], V[5]] }, // the three lattice axes
    { line: [O, bis(SWAP)] },                                         // the push direction
  ],
}, { centre: O, points: [...V, ...tris[SWAP]] });
console.log('wrote mark.svg, board.svg');
