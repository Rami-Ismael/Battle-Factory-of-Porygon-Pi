// Direction 03 — six tiles, two leads. π is the policy (the RL symbol). Its bar is the team:
// six square tiles. Its legs drop from two of them: the two Pokémon the policy leads with in doubles.
// System: square module grid with quarter-circle arcs. U = tile pitch; every radius is U or 2U − G.
import { writeMark, writeBoard, bboxOf } from '../lib.mjs';
import { fileURLToPath } from 'node:url';
const here = fileURLToPath(new URL('.', import.meta.url));

const U = 100;
const G = U / 10;                // gap between tiles; wide enough to see the six tiles at 64 px, fuses into one bar at 16 px
const W = U - G;                 // tile and leg width
const CY = 2 * U;                // foot arc centres sit on grid row 2, keeping the π wider than tall
const RI = U, RO = 2 * U - G;    // inner and outer foot radii: the foot ends under tile 0's left edge
// Optical corrections
const HORIZ_THIN = 0.94;                 // the bar is a horizontal stroke; 6% thinner than the legs to read the same weight
const ROUND_OVERSHOOT = 0.015 * 4 * U;   // the round feet overshoot the grid baseline by 1.5% of mark height
const HB = W * HORIZ_THIN;

const mirror = ops => ops.map(o => o[0] === 'Z' ? o
  : o[0] === 'A' ? ['A', o[1], o[2], 1 - o[3], [6 * U - o[4][0], o[4][1]]]
  : [o[0], [6 * U - o[1][0], o[1][1]]]);

const tile = i => [['M', [i * U + G / 2, 0]], ['L', [(i + 1) * U - G / 2, 0]],
  ['L', [(i + 1) * U - G / 2, HB]], ['L', [i * U + G / 2, HB]], ['Z']];
const x0 = U + G / 2, x1 = 2 * U - G / 2, cy = CY + ROUND_OVERSHOOT;
const hook = [['M', [x0, 0]], ['L', [x1, 0]], ['L', [x1, cy]], ['A', RO, 0, 1, [G / 2, cy + RO]],
  ['L', [G / 2, cy + RI]], ['A', RI, 0, 0, [x0, cy]], ['Z']];
const shapes = [tile(0), hook, tile(2), tile(3), mirror(hook), tile(5)];
const bbox = bboxOf([[G / 2, 0], [6 * U - G / 2, cy + RO]]);
writeMark(here + 'mark.svg', shapes, bbox, 2);

const cL = [G / 2, CY], cR = [6 * U - G / 2, CY], base = CY + RO;
const s2 = Math.SQRT1_2;
writeBoard(here + 'board.svg', shapes, bbox, {
  auxiliary: [
    ...[0, 1, 2, 3, 4, 5, 6].map(i => ({ line: [[i * U, 0], [i * U, 1]] })),   // tile pitch
    { seg: [cL, [cL[0] + RO * s2, CY + RO * s2]] }, { seg: [cR, [cR[0] - RO * s2, CY + RO * s2]] }, // arc bisectors
  ],
  secondary: [
    { circle: [cL, RO] }, { circle: [cL, RI] }, { circle: [cR, RO] }, { circle: [cR, RI] }, // quarter-circle feet
    { line: [[G / 2, 0], [G / 2, 1]] }, { line: [[6 * U - G / 2, 0], [6 * U - G / 2, 1]] },  // foot ends align under tiles 0 and 5
    { line: [[0, HB], [1, HB]] },                                                           // bar bottom (thinned)
  ],
  primary: [
    { line: [[3 * U, 0], [3 * U, 1]] },        // mirror axis
    { line: [[0, 0], [1, 0]] },                // cap line
    { line: [[0, base], [1, base]] },          // baseline (before overshoot)
    { line: [[0, CY], [1, CY]] },              // arc-centre row
  ],
}, { centre: [3 * U, (bbox[1] + bbox[3]) / 2], points: [cL, cR, [x0, 0], [x1, 0], [6 * U - x0, 0], [6 * U - x1, 0], [G / 2, cy + RO], [6 * U - G / 2, cy + RO]] });
console.log('wrote mark.svg, board.svg');
