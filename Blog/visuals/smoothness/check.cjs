// Run with the bundled Node runtime and its node_modules on NODE_PATH.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1050 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()}: ${r.url()}`); });
  await page.goto(process.env.SMOOTHNESS_URL || 'http://127.0.0.1:8769/smoothness/');
  const data = await page.evaluate(() => window.SMOOTHNESS_DATA);
  assert.equal(data.teams.length, 8);
  assert.equal(data.fullStudyRun, false);
  assert.equal(data.teams.reduce((n,t) => n + t.baselinePanels.reduce((m,p) => m+p.battles,0) + t.edges.reduce((m,e) => m+e.panels.reduce((k,p) => k+p.battles,0),0),0), 9900);
  assert.equal(data.primaryBattles,4800);
  assert.equal(data.measuredMembers,48);
  assert.equal(data.storage.kind,'matchup-matrix-sqlite');
  let inspected = 0;
  for (const team of data.teams) {
    await page.locator('#cohort').selectOption(team.group);
    await page.locator('#team').selectOption(team.id);
    assert.equal(await page.locator('.pokemon').count(), 6);
    assert.equal(await page.locator('.held-item-icon').count(), team.roster.filter(p=>p.heldItem && p.itemSprite).length);
    for(let slot=0;slot<6;slot++) {
      const card=page.locator(`[data-slot="${slot}"]`);
      assert.equal(await card.locator('.pokemon-item').textContent(),team.roster[slot].heldItem || 'No held item');
      if(team.roster[slot].heldItem && team.roster[slot].itemSprite) {
        const icon=card.locator('.held-item-icon');
        await icon.evaluate(img=>img.decode());
        assert(await icon.evaluate(img=>img.naturalWidth>0 && img.getBoundingClientRect().width===24 && img.getBoundingClientRect().height===24));
        assert((await card.getAttribute('aria-label')).includes(team.roster[slot].heldItem));
      }
      await page.locator(`[data-slot="${slot}"]`).click();
      assert(await page.locator('#effect-content').isVisible(),`${team.id} member ${slot} must have a measured result`);
      assert((await page.locator('#replacement option').count())>=2);
    }
    for (const edge of team.edges.filter(e => e.edit)) {
      await page.locator(`[data-slot="${edge.edit.slot}"]`).click();
      await page.locator('#replacement').selectOption(edge.candidate);
      const displayed = await page.locator('#delta-value').innerText();
      const parsed = Number(displayed.replace('−','-').replace('pp','').trim());
      assert(Math.abs(parsed - edge.delta * 100) <= .051, `${team.id}: effect readout must match recorded data`);
      assert.equal(await page.locator('#old-move').textContent(), edge.edit.old_move);
      assert((await page.locator('#interpretation').textContent()).includes(edge.edit.new_name));
      const meanOriginal = team.baselinePanels.reduce((n,p)=>n+p.wins,0)/150;
      const meanEdited = edge.panels.reduce((n,p)=>n+p.wins,0)/150;
      assert(Math.abs((meanEdited - meanOriginal)-edge.delta)<1e-12);
      const corrected = edge.replicate_deltas.reduce((n,a,i,arr)=>n+arr.reduce((m,b,j)=>m+(i===j?0:a*b),0),0)/6;
      assert(Math.abs(corrected-edge.corrected_mse)<1e-12);
      const limit=Number(await page.locator('#delta-chart').getAttribute('data-axis-limit'));
      assert(edge.ci95.every(v=>Math.abs(v*100)<=limit), 'Fixed axis must contain every interval');
      inspected++;
    }
  }
  assert.equal(inspected,50);
  await page.locator('#cohort').selectOption('tournament');
  await page.locator('#team').selectOption('tournament-000');
  await page.locator('#noise-toggle').click();
  assert.equal(await page.locator('#noise-toggle').getAttribute('aria-pressed'),'true');
  assert((await page.locator('#interpretation').textContent()).includes('Same team, new battle panels'));
  const nullDelta = data.teams.find(t=>t.id==='tournament-000').edges.find(e=>e.kind==='null').delta;
  assert(Math.abs(Number((await page.locator('#delta-value').innerText()).replace('−','-').replace('pp','').trim())-nullDelta*100)<=.051);
  await page.locator('#undo').click();
  assert.equal(await page.locator('#replacement').inputValue(),'');
  assert.equal(await page.locator('#ci-low').evaluate(el=>el.style.opacity),'0');
  assert((await page.locator('#interval-label').textContent()).includes('no new comparison'));
  await page.locator('[data-slot="1"]').click();
  assert(await page.locator('#effect-content').isVisible());
  assert.equal(await page.locator('#effect-mode').textContent(),'Additional member edit');
  await page.locator('[data-slot="0"]').click();
  assert(await page.locator('#effect-content').isVisible());

  // Keyboard actions are immediate; tab arrow navigation follows the ARIA pattern.
  await page.locator('#tab-edit').focus();
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#tab-compare').getAttribute('aria-selected'),'true');
  assert.equal(await page.locator('.team-dot').count(),8);
  await page.locator('#metric').selectOption('absolute');
  assert((await page.locator('#metric-explanation').textContent()).includes('contains battle noise'));
  await page.locator('[data-team="random-002"]').click();
  assert.equal(await page.locator('#team').inputValue(),'random-002');
  assert.equal(await page.locator('#tab-edit').getAttribute('aria-selected'),'true');
  await page.keyboard.press('Tab');
  assert.equal(await page.locator('#delta-dot').evaluate(el=>getComputedStyle(el).transitionDuration),'0s');

  // Benchmark stepping must leave the Pokémon comparison fixed.
  await page.locator('#tab-benchmarks').click();
  for (const benchmark of ['pest_control','hartmann6']) {
    await page.locator('#benchmark').selectOption(benchmark);
    const ref = await page.locator('.benchmark-value').first().innerText();
    for (const condition of data.benchmarks.filter(b=>b.domain===benchmark)) {
      await page.locator('#benchmark-step').selectOption(String(condition.step));
      assert.equal(await page.locator('.benchmark-value').first().innerText(),ref);
      assert((await page.locator('#benchmark-chart svg').textContent()).includes('Symmetric log'));
      await page.waitForTimeout(280);
      const plotted = await page.locator('#bench-domain-point').evaluate(el=>({x:new DOMMatrix(getComputedStyle(el).transform).m41,width:Number(document.getElementById('benchmark-chart').dataset.width)}));
      const log = v=>Math.sign(v)*Math.log10(1+Math.abs(v)/1e-9);
      const start = plotted.width<620?22:130;
      const expected = start+(log(condition.summary.mean)-log(-.001))/(log(.01)-log(-.001))*(plotted.width-26-start);
      assert(Math.abs(plotted.x-expected)<.1,'Benchmark marker must settle at the recorded normalized mean');
    }
  }

  const shots = path.join(__dirname,'previews');
  fs.mkdirSync(shots,{recursive:true});
  for (const width of [320,390,768,1280]) {
    await page.setViewportSize({width,height:1050});
    for (const name of ['edit','compare','benchmarks']) {
      await page.locator(`#tab-${name}`).click();
      const overflow = await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      assert.equal(overflow,false,`${name} overflows at ${width}px`);
      if(name==='compare') {
        const reachable = await page.locator('.team-dot').evaluateAll(buttons=>buttons.every(button=>{const r=button.getBoundingClientRect();return document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.closest('button')===button}));
        assert(reachable,`Team dots must have unobstructed hit targets at ${width}px`);
      }
    }
  }
  await page.setViewportSize({width:1280,height:1050});
  await page.locator('#tab-edit').click();
  await page.locator('#cohort').selectOption('tournament');
  await page.locator('#team').selectOption('tournament-000');
  await page.screenshot({path:path.join(shots,'edit-desktop.png'),fullPage:true});
  await page.locator('#tab-compare').click();
  await page.locator('#metric').selectOption('corrected');
  await page.screenshot({path:path.join(shots,'compare-desktop.png'),fullPage:true});
  await page.locator('#tab-benchmarks').click();
  await page.locator('#benchmark').selectOption('hartmann6');
  await page.waitForTimeout(280);
  await page.screenshot({path:path.join(shots,'benchmarks-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:1000});
  await page.locator('#tab-edit').click();
  await page.screenshot({path:path.join(shots,'edit-mobile.png'),fullPage:true});

  // Rapid pointer updates settle at the final selection, without a stale interval.
  await page.locator('#noise-toggle').click();
  await page.locator('#noise-toggle').click();
  await page.waitForTimeout(300);
  const position = await page.locator('#delta-dot').evaluate(el=>({m:new DOMMatrix(getComputedStyle(el).transform).m41,w:el.getBoundingClientRect().width}));
  const axisLimit=Number(await page.locator('#delta-chart').getAttribute('data-axis-limit'));
  const expected = (data.teams.find(t=>t.id==='tournament-000').edges.find(e=>e.edit).delta*100+axisLimit)/(2*axisLimit);
  assert(Math.abs(position.m/position.w-expected)<.001);
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.locator('#noise-toggle').click();
  assert.equal(await page.locator('#delta-dot').evaluate(el=>getComputedStyle(el).transitionDuration),'0s');
  // Missing-data protection remains valid if a future import has incomplete coverage.
  const missing=JSON.parse(JSON.stringify(data));
  missing.teams.find(t=>t.id==='tournament-000').roster[1].heldItem=null;
  missing.teams.find(t=>t.id==='tournament-000').edges=missing.teams.find(t=>t.id==='tournament-000').edges.filter(e=>e.edit?.slot!==1);
  await page.route('**/smoothness/data.js',route=>route.fulfill({contentType:'text/javascript',body:'window.SMOOTHNESS_DATA = '+JSON.stringify(missing)+';'}));
  await page.reload();
  await page.locator('[data-slot="1"]').click();
  assert.equal(await page.locator('[data-slot="1"] .held-item-icon').count(),0);
  assert.equal(await page.locator('[data-slot="1"] .pokemon-item').textContent(),'No held item');
  assert(await page.locator('#effect-empty').isVisible());
  assert(!(await page.locator('#effect-content').isVisible()));
  assert.deepEqual(errors,[]);
  await browser.close();
  console.log(JSON.stringify({passed:true,recordedEditsChecked:inspected,measuredMembers:48,teams:8,battles:9900,viewportWidths:[320,390,768,1280],verified:['all six members selectable','saved counts and effects','noise repeat','undo','missing data','linked cohort selection','keyboard tabs','fixed benchmark reference','fixed chart scale','reduced motion','rapid retargeting','no page errors'],screenshots:shots},null,2));
})().catch(error=>{console.error(error);process.exit(1)});
