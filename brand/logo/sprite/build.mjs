// Sprite versions of the 04b faceted π: the same 25 lattice facets rasterised into a 96 px frame, the size of a
// Gen 5 front sprite. No anti-aliasing, a black outline, and only colours from porygon.png (PokeAPI sprites repository).
import fs from 'node:fs';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
const here = fileURLToPath(new URL('.', import.meta.url));

// Geometry, as in ../04b-faceted-pi-two-feet/build.mjs (that script writes its files when imported).
const U = 100, S = Math.sqrt(3) / 2 * U;
const GAP = U / 8;
const POINT_OVERSHOOT = 0.02 * 4 * S;
const P = (a, b) => [a * U + b * U / 2, b * S];
const add = (p, q) => [p[0] + q[0], p[1] + q[1]], mul = (p, s) => [p[0] * s, p[1] * s];
const D = (a, b) => [[a, b], [a + 1, b], [a, b + 1]];
const Up = (a, b) => [[a, b], [a, b + 1], [a - 1, b + 1]];
const cells = [];
for (let a = 0; a < 6; a++) cells.push(D(a, 0));
for (let a = 1; a < 6; a++) cells.push(Up(a, 0));
for (const a0 of [1, 3]) for (let k = 0; k < 3; k++) cells.push(D(a0 - k, 1 + k), Up(a0 - k, 1 + k));
cells.push(Up(2, 3), D(-2, 3));
const BAR_FACETS = 11;
const rho = U / (2 * Math.sqrt(3));
const ends = { '0,0': [-POINT_OVERSHOOT, 0], '6,0': [POINT_OVERSHOOT, 0] };
const facets = gap => cells.map(c => {
  const pts = c.map(([a, b]) => add(P(a, b), ends[`${a},${b}`] ?? [0, 0]));
  const raw = c.map(([a, b]) => P(a, b));
  const I = mul(add(add(raw[0], raw[1]), raw[2]), 1 / 3);
  return pts.map(p => add(I, mul(add(p, mul(I, -1)), (rho - gap / 2) / rho)));
});
const isDown = i => cells[i][1][1] === cells[i][0][1];       // D cells have their top edge first

// Colours taken from porygon.png.
const BLACK = '#000000';
const PINK = { light: '#eeac9c', base: '#ff6262', shade: '#c5395a', deep: '#7b2941' };
const BLUE = { light: '#8bd5ee', base: '#08acd5', shade: '#0883a4', deep: '#085a73' };
const ramp = i => (i < BAR_FACETS ? PINK : BLUE);

// Rasterise: which facet does each pixel centre fall in?
const FRAME = 96, MARK_W = 88;
const all = facets(GAP).flat();
const bx0 = Math.min(...all.map(p => p[0])), bx1 = Math.max(...all.map(p => p[0]));
const by0 = Math.min(...all.map(p => p[1])), by1 = Math.max(...all.map(p => p[1]));
const k = MARK_W / (bx1 - bx0);
const originX = Math.round((FRAME - MARK_W) / 2), originY = Math.round((FRAME - (by1 - by0) * k) / 2);
const inside = (t, x, y) => {
  const side = (a, b) => (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
  const s0 = side(t[0], t[1]), s1 = side(t[1], t[2]), s2 = side(t[2], t[0]);
  return (s0 >= 0 && s1 >= 0 && s2 >= 0) || (s0 <= 0 && s1 <= 0 && s2 <= 0);
};
function rasterise(tris) {
  const grid = Array.from({ length: FRAME }, () => new Array(FRAME).fill(-1));
  for (let y = 0; y < FRAME; y++) for (let x = 0; x < FRAME; x++) {
    const wx = bx0 + (x + 0.5 - originX) / k, wy = by0 + (y + 0.5 - originY) / k;
    grid[y][x] = tris.findIndex(t => inside(t, wx, wy));
  }
  return grid;
}
const at = (g, x, y) => (y >= 0 && y < FRAME && x >= 0 && x < FRAME ? g[y][x] : -1);
const blank = () => Array.from({ length: FRAME }, () => new Array(FRAME).fill(null));
function outline(px) {
  const out = px.map(r => r.slice()), H = px.length, W = px[0].length;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (px[y][x]) continue;
    if ((y > 0 && px[y - 1][x]) || (y < H - 1 && px[y + 1][x]) || (x > 0 && px[y][x - 1]) || (x < W - 1 && px[y][x + 1])) out[y][x] = BLACK;
  }
  return out;
}

// Facets: 25 separate triangles, each with its own outline, lit from the top left.
function facetsVersion() {
  const g = rasterise(facets(GAP)), px = blank();
  for (let y = 0; y < FRAME; y++) for (let x = 0; x < FRAME; x++) {
    const f = g[y][x]; if (f < 0) continue;
    const r = ramp(f);
    const lit = at(g, x, y - 1) !== f || at(g, x - 1, y) !== f;
    const dark = at(g, x, y + 1) !== f || at(g, x + 1, y) !== f;
    px[y][x] = lit ? r.light : dark ? r.shade : r.base;
  }
  return outline(px);
}

// Gem: one silhouette, with the facets cut into it as alternating tones.
function gemPixels() {
  const g = rasterise(facets(0)), px = blank();
  for (let y = 0; y < FRAME; y++) for (let x = 0; x < FRAME; x++) {
    const f = g[y][x]; if (f < 0) continue;
    const r = ramp(f);
    const topEdge = at(g, x, y - 1) < 0;
    px[y][x] = topEdge ? r.light : isDown(f) ? r.base : r.shade;
  }
  return { g, px };
}
const gemVersion = () => outline(gemPixels().px);

// Block: the gem with a dark side face, like the solid body of Porygon.
function blockVersion() {
  const { g, px } = gemPixels(), solid = blank();
  const DEPTH_X = 2, DEPTH_Y = 3;
  for (let step = Math.max(DEPTH_X, DEPTH_Y); step >= 1; step--) {
    const dx = Math.round(DEPTH_X * step / Math.max(DEPTH_X, DEPTH_Y)), dy = Math.round(DEPTH_Y * step / Math.max(DEPTH_X, DEPTH_Y));
    for (let y = 0; y < FRAME; y++) for (let x = 0; x < FRAME; x++) {
      const f = g[y][x]; if (f < 0) continue;
      if (y + dy < FRAME && x + dx < FRAME) solid[y + dy][x + dx] = ramp(f).deep;
    }
  }
  for (let y = 0; y < FRAME; y++) for (let x = 0; x < FRAME; x++) if (px[y][x]) solid[y][x] = px[y][x];
  return outline(solid);
}

// A minimal RGBA PNG writer, so this script stays free of dependencies.
const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let i = 0; i < 8; i++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = b => { let c = 0xFFFFFFFF; for (const v of b) c = crcTable[(c ^ v) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
const chunk = (type, data) => {
  const body = Buffer.concat([Buffer.from(type, 'latin1'), data]), len = Buffer.alloc(4), sum = Buffer.alloc(4);
  len.writeUInt32BE(data.length); sum.writeUInt32BE(crc(body));
  return Buffer.concat([len, body, sum]);
};
function writePng(file, px) {
  const H = px.length, W = px[0].length;
  const raw = Buffer.alloc((W * 4 + 1) * H);
  px.forEach((row, y) => row.forEach((c, x) => {
    if (!c) return;
    const o = y * (W * 4 + 1) + 1 + x * 4;
    raw.writeUInt32BE((parseInt(c.slice(1), 16) << 8 | 255) >>> 0, o);
  }));
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 6;
  fs.writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
}
// The Battle Factory: a new drawing with the overall shape of the Emerald building (a centre tower between
// two lower wings, white roofs, yellow and brown bands, a small door), carrying the block logo on the tower.
function factoryVersion() {
  const W = 220, H = 146, px = Array.from({ length: H }, () => new Array(W).fill(null));
  const WHITE = '#ffffff', ROOF_MID = '#e4e4ee', ROOF_LIP = '#a8a8bc';
  const YELLOW_LIGHT = '#fbe88a', YELLOW = '#f2c83a', YELLOW_DARK = '#c8922a', BROWN = '#a86c3c', BROWN_DARK = '#6e4628';
  const STONE = '#a4a4b4', STONE_DARK = '#74748a', CREAM = '#fff8dc', DOORWAY = '#2b2a36';
  const fill = (x, y, w, h, c) => { for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) px[j][i] = c; };
  // A slab with its corners cut off, the way the Emerald roofs read from above.
  const roof = (x, y, w, h) => {
    const CUT = 4, LIP = 3;
    for (let j = 0; j < h; j++) {
      const inset = Math.max(0, CUT - j, CUT - (h - 1 - j));
      const c = j >= h - LIP ? ROOF_LIP : j < 2 ? WHITE : j >= h - LIP - 2 ? ROOF_MID : WHITE;
      fill(x + inset, y + j, w - 2 * inset, 1, c);
    }
  };
  const BAND = 24, BROWN_H = 6;
  const bandedWall = (x, y, w, h) => {
    for (let j = 0; j < h; j++) {
      const r = j % BAND;
      const c = r >= BAND - BROWN_H ? (r === BAND - 1 ? BROWN_DARK : BROWN) : r < 3 ? YELLOW_LIGHT : r >= BAND - BROWN_H - 2 ? YELLOW_DARK : YELLOW;
      fill(x, y + j, w, 1, c);
    }
  };
  const BASE = 4;
  // Wings.
  for (const x of [4, 158]) {
    bandedWall(x + 3, 40, 52, H - 41 - BASE);
    fill(x + 3, H - 1 - BASE, 52, BASE, STONE);
    fill(x + 27, 40, 2, H - 41 - BASE, STONE_DARK);
    roof(x, 28, 58, 13);
  }
  // Tower.
  const TX = 58, TW = 104;
  fill(TX + 3, 16, TW - 6, H - 17 - BASE, YELLOW);
  fill(TX + 3, H - 1 - BASE, TW - 6, BASE, STONE);
  fill(TX + 3, 16, 2, H - 17 - BASE, STONE_DARK);
  fill(TX + TW - 5, 16, 2, H - 17 - BASE, STONE_DARK);
  fill(TX + 5, 18, TW - 10, 54, CREAM);
  fill(TX + 5, 72, TW - 10, 2, YELLOW_DARK);
  fill(TX + 5, 74, TW - 10, BROWN_H, BROWN);
  fill(TX + 5, 74 + BROWN_H - 1, TW - 10, 1, BROWN_DARK);
  bandedWall(TX + 5, 80, TW - 10, H - 81 - BASE);
  roof(TX, 1, TW, 16);
  // Door, framed in yellow.
  const DW = 32, DH = 48, DX = Math.round((W - DW) / 2), DY = H - 1 - DH;
  fill(DX - 3, DY - 3, DW + 6, DH + 3, YELLOW_LIGHT);
  fill(DX - 3, DY - 3, DW + 6, 1, YELLOW_DARK);
  fill(DX, DY, DW, DH, DOORWAY);
  // The block logo, on the tower's cream panel.
  const logo = blockVersion(), LX = Math.round((W - FRAME) / 2) - 1, LY = 45 - Math.round(FRAME / 2);
  for (let y = 0; y < FRAME; y++) for (let x = 0; x < FRAME; x++) if (logo[y][x] && LY + y >= 0 && LY + y < H) px[LY + y][LX + x] = logo[y][x];
  return outline(px);
}
// Small drawing helpers for the Generation IV sprites below.
const canvasOf = (W, H) => Array.from({ length: H }, () => new Array(W).fill(null));
const fillRect = (px, x, y, w, h, c) => { for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) px[j][i] = c; };
function fillPolygon(px, pts, c) {
  for (let y = 0; y < px.length; y++) for (let x = 0; x < px[0].length; x++) {
    let odd = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y + 0.5) !== (yj > y + 0.5) && x + 0.5 < (xj - xi) * (y + 0.5 - yi) / (yj - yi) + xi) odd = !odd;
    }
    if (odd) px[y][x] = c;
  }
}
function drawLine(px, x0, y0, x1, y1, c) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
  for (let i = 0; i <= steps; i++) px[Math.round(y0 + (y1 - y0) * i / steps)][Math.round(x0 + (x1 - x0) * i / steps)] = c;
}
function fillDisc(px, cx, cy, r, c) {
  for (let y = 0; y < px.length; y++) for (let x = 0; x < px[0].length; x++) if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r) px[y][x] = c;
}

// The Generation IV Battle Factory, as in Platinum, HeartGold and SoulSilver: a new drawing with the overall shape of
// the building (a faceted blue hull with red light strips under a domed roof), plus the exhaust fans and pipes
// Bulbapedia describes, and the block logo.
function blueFactoryVersion() {
  const W = 220, H = 170, px = canvasOf(W, H);
  const DARK = '#2c3340', ARCH = '#35688e', ARCH_LIGHT = '#7ab0d4', HULL = '#4f86bd', HULL_LOW = '#3f74a8', RIDGE = '#7fb3df', RIDGE_LOW = '#6a9cce';
  const TRIM = '#f5e27a', STRIP = '#d23a3a', SEAM = '#2b4f76', CREAM = '#fff8dc', DOORWAY = '#2b2a36';
  // Domed roof, lighter towards the top.
  for (let y = 2; y < 62; y++) {
    const half = y < 44 ? 82 * Math.sqrt(1 - ((y - 44) / 42) ** 2) : 82;
    fillRect(px, Math.round(110 - half), y, Math.round(2 * half), 1, y < 18 ? '#c4d2e2' : y < 38 ? '#a6bacf' : '#8aa1ba');
  }
  fillRect(px, 109, 2, 2, 50, DARK);
  for (let x = 44; x <= 176; x++) px[Math.round(26 + ((x - 110) / 66) ** 2 * 11)][x] = DARK;
  fillDisc(px, 110, 26, 4, DARK);
  // Pipes at both ends, behind the arch.
  for (const x of [6, 208]) { fillRect(px, x, 52, 6, 22, DARK); fillRect(px, x - 2, 50, 10, 3, DARK); }
  // The arch across the front.
  fillPolygon(px, [[22, 50], [198, 50], [210, 62], [206, 70], [14, 70], [10, 62]], ARCH);
  drawLine(px, 22, 51, 198, 51, ARCH_LIGHT);
  // Exhaust fans on the arch.
  for (const cx of [32, 188]) {
    fillDisc(px, cx, 60, 7, DARK);
    fillDisc(px, cx, 60, 5, '#8aa1ba');
    drawLine(px, cx - 4, 60, cx + 4, 60, DARK); drawLine(px, cx, 56, cx, 64, DARK);
  }
  // The hull.
  fillPolygon(px, [[34, 72], [186, 72], [214, 118], [196, 169], [24, 169], [6, 118]], HULL);
  fillPolygon(px, [[6, 118], [214, 118], [196, 169], [24, 169]], HULL_LOW);
  fillRect(px, 34, 72, 152, 2, TRIM);
  drawLine(px, 56, 74, 19, 116, RIDGE); drawLine(px, 164, 74, 201, 116, RIDGE);
  drawLine(px, 64, 120, 58, 168, RIDGE_LOW); drawLine(px, 156, 120, 162, 168, RIDGE_LOW);
  fillRect(px, 8, 117, 204, 2, SEAM);
  fillRect(px, 18, 113, 34, 3, STRIP); fillRect(px, 168, 113, 34, 3, STRIP);
  // Logo panel and door.
  fillRect(px, 60, 76, 100, 50, CREAM);
  const logo = blockVersion(), LX = 61, LY = 50;
  for (let y = 0; y < FRAME; y++) for (let x = 0; x < FRAME; x++) if (logo[y][x]) px[LY + y][LX + x] = logo[y][x];
  fillRect(px, 91, 128, 38, 41, TRIM);
  fillRect(px, 94, 131, 32, 38, DOORWAY);
  return outline(px);
}

// Inside the same building: glass panels that hold the rental Poké Balls, and a long glass panel along the floor.
function factoryInsideVersion() {
  const W = 220, H = 146, px = canvasOf(W, H);
  const WALL = '#dfe7ee', GRID = '#aebccc', FRAME_BLUE = '#7f9ab4', GLASS = '#bfe0ee', SHINE = '#dff1f8', FLOOR = '#cfdde2', TILE = '#bccdd4', FLOOR_GLASS = '#a4d0e4';
  fillRect(px, 1, 1, W - 2, 83, WALL);
  fillRect(px, 1, 1, W - 2, 2, '#7888a8');
  for (let x = 1; x < W - 1; x += 12) fillRect(px, x, 3, 1, 19, GRID);
  fillRect(px, 1, 12, W - 2, 1, GRID); fillRect(px, 1, 22, W - 2, 1, GRID);
  fillRect(px, 1, 84, W - 2, H - 85, FLOOR);
  for (let y = 92; y < H - 1; y += 14) fillRect(px, 1, y, W - 2, 1, TILE);
  for (let x = 10; x < W - 1; x += 24) fillRect(px, x, 84, 1, H - 85, TILE);
  fillRect(px, 1, 104, W - 2, 16, FLOOR_GLASS);
  fillRect(px, 1, 104, W - 2, 1, SHINE);
  fillRect(px, 1, 119, W - 2, 1, FRAME_BLUE);
  // Six compartments, one Poké Ball each; the page draws the balls.
  fillRect(px, 30, 28, 158, 56, FRAME_BLUE);
  for (let i = 0; i < 6; i++) {
    const x = 34 + 25 * i;
    fillRect(px, x, 32, 25 - 2, 48, GLASS);
    drawLine(px, x + 2, 44, x + 12, 34, SHINE); drawLine(px, x + 2, 50, x + 18, 34, SHINE);
    fillRect(px, x, 64, 23, 2, FRAME_BLUE);
  }
  // A lift door on the right.
  fillRect(px, 196, 40, 22, 44, '#5a78a0');
  fillRect(px, 206, 40, 2, 44, '#3f5678');
  return outline(px);
}

// A Poké Ball, 14 px, in colour for a picked rental and in grey for the others.
function ballVersion(top, bottom) {
  const S = 14, px = canvasOf(S, S);
  fillDisc(px, 7, 7, 5.6, bottom);
  for (let y = 0; y < 7; y++) for (let x = 0; x < S; x++) if (px[y][x]) px[y][x] = top;
  fillRect(px, 2, 6, 10, 2, BLACK);
  fillDisc(px, 7, 7, 2.4, BLACK);
  fillDisc(px, 7, 7, 1.3, '#ffffff');
  return outline(px);
}
writePng(here + 'sprite-factory-blue.png', blueFactoryVersion());
writePng(here + 'sprite-factory-inside.png', factoryInsideVersion());
writePng(here + 'sprite-ball.png', ballVersion('#e8454a', '#f4f4f4'));
writePng(here + 'sprite-ball-dim.png', ballVersion('#a9aab0', '#d4d4da'));
writePng(here + 'sprite-factory.png', factoryVersion());
writePng(here + 'sprite-facets.png', facetsVersion());
writePng(here + 'sprite-gem.png', gemVersion());
writePng(here + 'sprite-block.png', blockVersion());
console.log('wrote all sprites');
