// Usage: node export.mjs — square PNG icons from the final SVGs, via headless Chrome.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'icon-'));
const shot = (name, svg, size, bg, inset = 0.1) => {
  const html = path.join(tmp, name + '.html');
  fs.writeFileSync(html, `<!doctype html><style>*{margin:0}body{width:${size}px;height:${size}px;background:${bg};display:flex;align-items:center;justify-content:center}img{width:${size * (1 - 2 * inset)}px}</style><img src="file://${path.join(here, svg)}">`);
  execFileSync(chrome, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--allow-file-access-from-files',
    `--screenshot=${path.join(here, name + '.png')}`, `--window-size=${size},${size}`, `file://${html}`], { stdio: 'pipe' });
};
// Colour is the primary logo (owner, 2026-10-30); black and white stay for one-colour uses.
shot('icon-512', 'mark-color.svg', 512, '#fff');
shot('icon-512-on-dark', 'mark-color.svg', 512, '#111');
shot('icon-512-black', 'mark.svg', 512, '#fff');
shot('icon-512-white', 'mark-white.svg', 512, '#111');
shot('icon-192', 'mark-color.svg', 192, '#fff');
shot('apple-touch-icon-180', 'mark-color.svg', 180, '#fff');
shot('favicon-32', 'mark-solid-color.svg', 32, '#fff', 0.03);
shot('favicon-16', 'mark-solid-color.svg', 16, '#fff', 0.02);
fs.rmSync(tmp, { recursive: true });
console.log('wrote icons');
