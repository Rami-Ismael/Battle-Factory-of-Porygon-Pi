// Direction 06 — polygon bird π. A nod to Porygon, the man-made Pokémon built from polygons,
// without drawing Porygon itself: the π stands up as a faceted bird. Bar = body, legs = legs,
// a tail fin on the left and a beak on the right, cut apart by thin gaps like separate polygons.
// System: square module grid, U = 10 modules; every slanted edge runs at 2:1, the pixel-art line.
import { writeMark, writeBoard, bboxOf } from '../lib.mjs';
import { fileURLToPath } from 'node:url';
const here = fileURLToPath(new URL('.', import.meta.url));

const U = 100, m = U / 10;       // module
const G = 1 * m;                 // facet gap; fuses at 16 px so the icon reads as one π
const BAR = 12 * m;              // body depth
const LEG = 9 * m;               // leg width
const BASE = 30 * m;             // baseline; short legs keep the bird chunky, the way Porygon stands
// Optical corrections
const BEAK_OVERSHOOT = 0.02 * (BASE + 6 * m); // the beak point reads short against the flat body; extend 2% of mark height
const EYE_UP = 0.5 * m;                        // the eye sits above the bar's centre line, where an eye reads as an eye

const tail = [[-3 * m, -6 * m], [9 * m, 0], [9 * m, BAR], [3 * m, BAR]];
const body = [[10 * m, 0], [49 * m, 0], [49 * m, BAR], [10 * m, BAR]];
const tip = [62 * m + BEAK_OVERSHOOT, BAR / 2];
const beak = [[50 * m, 0], tip, [50 * m, BAR]];
const ey = BAR / 2 - EYE_UP, eye = [[40 * m, ey - 2 * m], [40 * m, ey + 2 * m], [44 * m, ey + 2 * m], [44 * m, ey - 2 * m]]; // reversed winding: a hole
const leg = x0 => { const x1 = x0 + LEG; return [[x0, BAR + G], [x1, BAR + G], [x1, BASE - 3 * m], [x1 + 6 * m, BASE], [x0, BASE]]; };
const poly = (p, rest = []) => [['M', p[0]], ...p.slice(1).map(q => ['L', q]), ['Z'], ...rest];
const holeOps = [['M', eye[0]], ...eye.slice(1).map(q => ['L', q]), ['Z']];
const legs = [leg(14 * m), leg(32 * m)];
const shapes = [poly(tail), poly(body, holeOps), poly(beak), ...legs.map(l => poly(l))];
const bbox = bboxOf([...tail, ...body, ...beak, ...legs.flat()]);
writeMark(here + 'mark.svg', shapes, bbox, 2);

const two = (p, dir) => ({ line: [p, [p[0] + dir[0], p[1] + dir[1]]] }); // 2:1 line through p
const vl = x => ({ line: [[x, 0], [x, 1]] }), hl = y => ({ line: [[0, y], [1, y]] });
writeBoard(here + 'board.svg', shapes, bbox, {
  auxiliary: [
    ...[0, 1, 2, 3, 4, 5, 6].map(i => vl(i * U)),            // U grid
    ...[1, 2, 3].map(j => hl(j * U)).filter(g => Math.abs(g.line[0][1] - BAR) > 2 * m && Math.abs(g.line[0][1] - BASE) > 2 * m),
    { circle: [[42 * m, ey], 2 * Math.SQRT2 * m] },          // eye: a 4-module square
  ],
  secondary: [
    vl(14 * m), vl(23 * m), vl(32 * m), vl(41 * m),          // leg edges, one leg width apart
    hl(BAR / 2), hl(BAR + G), hl(-6 * m),                    // beak axis, gap under the body, tail tip
    vl(9 * m), vl(50 * m),                                   // facet cuts
  ],
  primary: [
    hl(0), hl(BAR), hl(BASE),                                // back, belly, baseline
    two([9 * m, 0], [2, 1]),                 // tail fin top, 2:1
    two([50 * m, 0], [2, 1]), two([50 * m, BAR], [2, -1]), // beak, 2:1
    two([23 * m, BASE - 3 * m], [2, 1]), two([41 * m, BASE - 3 * m], [2, 1]), // toes, 2:1
  ],
}, { centre: [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], points: [...tail, tip, [50 * m, 0], [50 * m, BAR], [29 * m, BASE], [47 * m, BASE]] });
console.log('wrote mark.svg, board.svg');
