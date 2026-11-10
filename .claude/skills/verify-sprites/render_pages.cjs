#!/usr/bin/env node
// Render sprite pages in headless Chrome, light and dark, and count broken images.
//
// Run from the repo root:
//   node .claude/skills/verify-sprites/render_pages.cjs <out-dir> [page.html ...]
//   node .claude/skills/verify-sprites/render_pages.cjs --doctor
//
// With no pages, renders DEFAULT_PAGES. Serves the repo from its own throwaway
// local server (random port, closed on exit). Writes <out-dir>/render.json and one
// screenshot per page and colour scheme. Exits 1 if any page has a broken image,
// a failed request, a page error, a request to a Showdown sprite URL, or shows no
// sprite at all (a page that shows none proves nothing).
// Playwright is loaded from $PLAYWRIGHT_DIR, else the Codex runtime copy, else `playwright`.

const fs = require('fs');
const http = require('http');
const path = require('path');

const DEFAULT_PAGES = [
  'artifacts/jev-request-anatomy.html',
  'notes/Blog/visuals/looped-diffusion/index.html',
  'notes/Blog/visuals/team-lineage/team-lineage.html',
  'notes/Blog/visuals/search-loop/index.html',
  'notes/Blog/visuals/teams-differ-prototype/index.html',
  'blog/mask-lab/index.html',
  'blog/kingambit-finder/index.html',
  'notes/Blog/visuals/team-map-path/team-map-path.html',
];
const PLAYWRIGHT_CANDIDATES = [
  process.env.PLAYWRIGHT_DIR,
  path.join(process.env.HOME || '', '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'),
  'playwright',
].filter(Boolean);
// Pages that show sprites only after a user action: the action to take, by accessible name.
const PAGE_DRIVES = {
  'notes/Blog/visuals/search-loop/index.html': async page => { await page.getByLabel('Team reveal progress').fill('6'); },
};
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.png': 'image/png', '.svg': 'image/svg+xml', '.gif': 'image/gif' };

function loadPlaywright() {
  for (const candidate of PLAYWRIGHT_CANDIDATES) {
    try { return { playwright: require(candidate), from: candidate }; } catch (error) { /* try the next one */ }
  }
  return null;
}

function serveRepo(root) {
  const server = http.createServer((request, response) => {
    const file = path.join(root, decodeURIComponent(request.url.split('?')[0]));
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { response.writeHead(404); response.end(); return; }
    response.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(response);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

(async () => {
  const root = process.cwd();
  const loaded = loadPlaywright();
  if (process.argv[2] === '--doctor') {
    const storeOk = fs.existsSync(path.join(root, 'notes/Blog/visuals/search-loop/sprites/sources.json'));
    console.log(`${storeOk ? 'PASS' : 'FAIL'} repo root: ${root}`);
    console.log(`${loaded ? 'PASS' : 'FAIL'} playwright: ${loaded ? loaded.from : 'not found; set PLAYWRIGHT_DIR'}`);
    process.exit(storeOk && loaded ? 0 : 1);
  }
  const out = process.argv[2];
  if (!out) { console.error('usage: render_pages.cjs <out-dir> [page.html ...]'); process.exit(2); }
  if (!loaded) { console.error('FAIL playwright not found; set PLAYWRIGHT_DIR'); process.exit(2); }
  const pages = process.argv.length > 3 ? process.argv.slice(3) : DEFAULT_PAGES;
  fs.mkdirSync(out, { recursive: true });

  const server = await serveRepo(root);
  const base = `http://127.0.0.1:${server.address().port}/`;
  const browser = await loaded.playwright.chromium.launch({ headless: true, channel: 'chrome' });
  const results = [];
  try {
    for (const page_path of pages) {
      for (const scheme of ['light', 'dark']) {
        const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: scheme });
        const page = await context.newPage();
        const pageErrors = [], failedRequests = [], showdownRequests = [];
        let pokeapiRequests = 0, spriteFilesLoaded = 0;
        page.on('pageerror', error => pageErrors.push(error.message.slice(0, 200)));
        page.on('request', request => {
          if (request.url().includes('pokemonshowdown.com/sprites')) showdownRequests.push(request.url());
          if (request.url().includes('/PokeAPI/sprites/')) pokeapiRequests += 1;
        });
        page.on('response', response => {
          if (response.status() === 200 && /\/(PokeAPI\/sprites|search-loop\/sprites)\/.*\.png$/.test(response.url())) spriteFilesLoaded += 1;
          if (response.status() >= 400 && !response.url().endsWith('favicon.ico')) failedRequests.push(`${response.status()} ${response.url().slice(0, 160)}`);
        });
        page.on('requestfailed', request => failedRequests.push(`failed ${request.url().slice(0, 160)}`));
        let missing = !fs.existsSync(path.join(root, page_path));
        if (!missing) {
          await page.goto(base + encodeURI(page_path), { waitUntil: 'load', timeout: 30000 });
          await page.waitForTimeout(1500);
          if (PAGE_DRIVES[page_path]) { await PAGE_DRIVES[page_path](page); await page.waitForTimeout(1200); }
          const height = await page.evaluate(() => document.documentElement.scrollHeight);
          for (let y = 0; y < Math.min(height, 40000); y += 700) { await page.evaluate(top => window.scrollTo(0, top), y); await page.waitForTimeout(50); }
          await page.evaluate(() => window.scrollTo(0, 0));
          await page.waitForTimeout(800);
        }
        const counts = missing ? { images: 0, sprites: 0, broken: [] } : await page.evaluate(() => {
          const images = [...document.images].filter(image => image.getAttribute('src'));
          return {
            images: images.length,
            sprites: images.filter(image => image.naturalWidth === 96 || image.naturalWidth === 512).length,
            broken: images.filter(image => image.complete && image.naturalWidth === 0).map(image => image.getAttribute('src').slice(0, 100)),
          };
        });
        const shot = path.join(out, page_path.replace(/[\/ ]/g, '_').replace(/\.html$/, '') + `-${scheme}.png`);
        if (!missing) await page.screenshot({ path: shot });
        const spritesSeen = counts.sprites + spriteFilesLoaded;
        const passed = !missing && spritesSeen > 0 && counts.broken.length === 0 && failedRequests.length === 0 && pageErrors.length === 0 && showdownRequests.length === 0;
        results.push({ page: page_path, scheme, passed, missing, ...counts, spriteFilesLoaded, spritesSeen, pokeapiRequests, showdownRequests, failedRequests: [...new Set(failedRequests)].slice(0, 10), pageErrors, screenshot: shot });
        console.log(`${passed ? 'PASS' : 'FAIL'} ${scheme.padEnd(5)} ${page_path} | sprites seen ${spritesSeen} (embedded ${counts.sprites}, loaded as files ${spriteFilesLoaded}), broken ${counts.broken.length}, PokeAPI requests ${pokeapiRequests}, failed requests ${failedRequests.length}, page errors ${pageErrors.length}${missing ? ', FILE MISSING' : ''}`);
        await context.close();
      }
    }
  } finally {
    await browser.close();
    server.close();
  }
  const passed = results.every(result => result.passed);
  fs.writeFileSync(path.join(out, 'render.json'), JSON.stringify({ passed, results }, null, 1) + '\n');
  console.log(`RESULT ${passed ? 'pass' : 'FAIL'} | ${path.join(out, 'render.json')}`);
  process.exit(passed ? 0 : 1);
})();
