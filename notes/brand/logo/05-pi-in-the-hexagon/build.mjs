// Direction 05 — π in the hexagon. Six sides for the six Pokémon of a team; the policy π is
// the counter cut through it. System: flat-top hexagon, U = wall. Every edge of the counter
// is a hexagon edge offset by U, or a vertical at a whole multiple of U/2.
import { writeMark, writeBoard, bboxOf } from '../lib.mjs';
import { fileURLToPath } from 'node:url';
const here = fileURLToPath(new URL('.', import.meta.url));

const U = 100, R = 5 * U, h = R * Math.sqrt(3) / 2;
const K = 1.1 * U;               // counter stroke; at 16 px the hexagon is 10U wide, so the counter stays ~1.8 px open
// Optical corrections
const HORIZ_THIN = 0.9;          // the white bar is horizontal; thinner than the white legs to read the same weight
const OFF = 2 * U / Math.sqrt(3);                       // horizontal width of a U-thick wall along a 60° side
const sideX = y => R - Math.abs(y) / Math.sqrt(3) - OFF; // inner hexagon (inset U) half-width at height y

const hex = [[-R, 0], [-R / 2, -h], [R / 2, -h], [R, 0], [R / 2, h], [-R / 2, h]];
const yt = -h + U, yb = yt + K * HORIZ_THIN, yl = h - U;
const lx0 = 0.5 * K, lx1 = 1.5 * K;
const counterCW = [[-sideX(yt), yt], [sideX(yt), yt], [sideX(yb), yb], [lx1, yb], [lx1, yl], [lx0, yl], [lx0, yb],
  [-lx0, yb], [-lx0, yl], [-lx1, yl], [-lx1, yb], [-sideX(yb), yb]];
const hole = [...counterCW].reverse();
const shape = [['M', hex[0]], ...hex.slice(1).map(p => ['L', p]), ['Z'], ['M', hole[0]], ...hole.slice(1).map(p => ['L', p]), ['Z']];
const bbox = bboxOf(hex);
writeMark(here + 'mark.svg', [shape], bbox, 2);

const inner = hex.map(([x, y]) => [x === 0 ? 0 : Math.sign(x) * (Math.abs(x) - (Math.abs(x) === R ? OFF : OFF / 2)), y === 0 ? 0 : Math.sign(y) * (h - U)]);
const edge = i => ({ line: [hex[i], hex[(i + 1) % 6]] });
const vl = x => ({ line: [[x, 0], [x, 1]] }), hl = y => ({ line: [[0, y], [1, y]] });
writeBoard(here + 'board.svg', [shape], bbox, {
  auxiliary: [
    { seg: [hex[0], hex[3]] }, { seg: [hex[1], hex[4]] }, { seg: [hex[2], hex[5]] }, // the six team slots
    vl(-lx1), vl(-lx0), vl(lx0), vl(lx1),                                            // leg edges
    vl(-R / 2), vl(R / 2),                                                           // top vertices
    hl(yb),                                                                          // bar bottom (thinned)
  ],
  secondary: [
    { poly: inner },                                   // hexagon inset by U: clips the bar ends and leg feet
    { circle: [[0, 0], R] }, { circle: [[0, 0], h] },  // circumcircle and incircle
    hl(yt), hl(yl),                                    // counter top and leg feet, U in from the flat edges
  ],
  primary: [
    ...[0, 1, 2, 3, 4, 5].map(edge),                   // hexagon edges, extended
    vl(0),                                             // mirror axis
  ],
}, { centre: [0, 0], points: [...hex, ...counterCW.slice(0, 3), counterCW[11]] });
console.log('wrote mark.svg, board.svg');
