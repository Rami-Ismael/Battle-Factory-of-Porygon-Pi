// Drives headless Chrome over the DevTools protocol to screenshot logo-build.html.
// LOGO_PAGE picks the page and its ?script=, LOGO_PORT the first debugging port.
// node render.mjs check                      seek every 0.25 s and compare geometry with mark-color.svg
// node render.mjs times OUT_DIR t1 t2 ...    one PNG per time
// node render.mjs frames OUT_DIR [workers]   480 fps frames, f%05d.png
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const W = 1080, H = 1350;

async function openPage(port) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'logo-video-'));
  const proc = spawn(chrome, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=1',
    `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, `--window-size=${W},${H}`, 'about:blank'], { stdio: 'ignore' });
  let target;
  for (let i = 0; i < 100 && !target; i++) {
    await new Promise(r => setTimeout(r, 100));
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(x => x.type === 'page'); } catch {}
  }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(r => { ws.onopen = r; });
  let id = 0; const waiting = new Map(); const onLoad = [];
  ws.onmessage = m => {
    const msg = JSON.parse(m.data);
    if (msg.id) { waiting.get(msg.id)(msg); waiting.delete(msg.id); }
    if (msg.method === 'Page.loadEventFired') onLoad.splice(0).forEach(f => f());
  };
  const send = (method, params = {}) => new Promise(r => { const n = ++id; waiting.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  const loaded = new Promise(r => onLoad.push(r));
  await send('Page.navigate', { url: 'file://' + path.join(here, process.env.LOGO_PAGE ?? 'logo-build.html') });
  await loaded;
  const evaluate = async expr => {
    const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.result.exceptionDetails) throw new Error(expr + ': ' + JSON.stringify(r.result.exceptionDetails));
    return r.result.result.value;
  };
  await evaluate('document.fonts.ready.then(() => 1)');
  const shot = async (t, file) => {
    await evaluate(`seek(${t})`);
    const r = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
    fs.writeFileSync(file, Buffer.from(r.result.data, 'base64'));
  };
  const close = () => { ws.close(); proc.kill(); };
  return { evaluate, shot, close };
}

const [mode, out, ...rest] = process.argv.slice(2);
if (mode === 'check') {
  const page = await openPage(9331);
  for (let i = 0; i <= 96; i++) await page.evaluate(`seek(${i * 0.25})`);
  console.log('seek ran for t = 0 to 24 in 0.25 s steps');
  const mine = await page.evaluate('finalPathCoords()');
  const svg = fs.readFileSync(out ?? path.join(here, '../final/mark-color.svg'), 'utf8');
  const real = [...svg.matchAll(/d="M([^"]+) Z"/g)].map(m => m[1].split(' L').map(s => s.split(' ').map(Number)));
  let worst = 0;
  mine.forEach((t, i) => t.forEach((p, j) => { worst = Math.max(worst, Math.abs(p[0] - real[i][j][0]), Math.abs(p[1] - real[i][j][1])); }));
  console.log(`facets: page ${mine.length}, mark-color.svg ${real.length}; largest coordinate difference ${worst}`);
  page.close();
} else if (mode === 'eval') {
  const page = await openPage(9333);
  console.log(JSON.stringify(await page.evaluate(out)));
  page.close();
} else if (mode === 'times') {
  fs.mkdirSync(out, { recursive: true });
  const page = await openPage(9332);
  for (const t of rest) await page.shot(+t, path.join(out, `t${(+t).toFixed(3).padStart(7, '0')}.png`));
  page.close();
} else if (mode === 'frames') {
  fs.mkdirSync(out, { recursive: true });
  const workers = +(rest[0] ?? 6), fps = 480, total = 24 * fps;
  await Promise.all(Array.from({ length: workers }, async (_, w) => {
    const page = await openPage(+(process.env.LOGO_PORT ?? 9340) + w);
    for (let n = w; n < total; n += workers) {
      await page.shot(n / fps, path.join(out, `f${String(n).padStart(5, '0')}.png`));
      if (n % fps === 0) console.log(`${n}/${total}`);
    }
    page.close();
  }));
  console.log('frames done');
}
