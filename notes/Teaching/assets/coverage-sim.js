/* Coverage picker — an interactive toy instance shared by the coverage lessons.
   Usage:
     <div id="sim"></div>
     <script src="../assets/coverage-sim.js"></script>
     <script>CoverageSim.mount(document.getElementById('sim'), {
        elements: [{name:'meta team 1', w:5}, …],   // universe U with element weights
        sets:     [{name:'A', covers:[0,1]}, …],    // one set per candidate: indices into elements
        k: 2,                                      // budget for the max-coverage reading
        optMaxCov: 17,                             // optional: known optimum covered weight at k (for feedback)
        optSetCover: 3                             // optional: known minimum number of sets covering U
     });</script>
   Shows a matrix (sets × elements), lets the learner toggle sets, and reports the
   instance under BOTH readings: max coverage (budget k, maximise covered weight)
   and set cover (cover everything, minimise the count). "Greedy step" adds the set
   with the largest still-uncovered weight — the rule both unweighted problems share. */
window.CoverageSim = (function () {
  const css = `
  .csim { border: 1px solid var(--rule,#ccc); border-radius: 6px; padding: 1rem 1.2rem; margin: 1.5rem 0; font-size: .92em; }
  .csim table { border-collapse: collapse; margin: .6rem 0 1rem; }
  .csim th, .csim td { padding: .35rem .6rem; text-align: center; border-bottom: 1px solid var(--rule,#ccc); }
  .csim th { font-weight: 600; font-size: .8em; color: var(--ink-soft,#555); }
  .csim td.name { text-align: left; font-weight: 600; cursor: pointer; user-select: none; }
  .csim tr.on td.name { color: var(--cover,#2c6e49); }
  .csim tr.on td.name::before { content: '☑ '; } .csim tr:not(.on) td.name::before { content: '☐ '; }
  .csim td.dot { font-family: var(--mono, monospace); color: var(--ink-soft,#777); }
  .csim tr.on td.dot.hit { color: var(--cover,#2c6e49); font-weight: 700; }
  .csim th.covered { color: var(--cover,#2c6e49); }
  .csim .panel { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .csim .panel > div { border-top: 3px solid; padding-top: .5rem; }
  .csim .panel .mc { border-color: var(--cover,#2c6e49); } .csim .panel .sc { border-color: var(--setc,#3d5a80); }
  .csim .panel h4 { margin: 0 0 .3rem; font-size: .85em; text-transform: uppercase; letter-spacing: .05em; }
  .csim .panel .mc h4 { color: var(--cover,#2c6e49); } .csim .panel .sc h4 { color: var(--setc,#3d5a80); }
  .csim .panel p { margin: .2rem 0; font-size: .95em; }
  .csim .ctl { display: flex; gap: .6rem; flex-wrap: wrap; align-items: center; margin-top: .8rem; }
  .csim button { font: inherit; font-size: .9em; padding: .35rem .8rem; border: 1px solid var(--rule,#ccc); border-radius: 4px; background: transparent; color: inherit; cursor: pointer; }
  .csim button:hover { border-color: var(--accent,#8a3b12); }
  .csim .bad { color: var(--warn,#9c4a1a); } .csim .good { color: var(--cover,#2c6e49); }
  @media (max-width: 700px) { .csim .panel { grid-template-columns: 1fr; } }`;
  let injected = false;

  function mount(root, cfg) {
    if (!injected) { const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s); injected = true; }
    const E = cfg.elements, S = cfg.sets, k = cfg.k;
    const total = E.reduce((a, e) => a + e.w, 0);
    const chosen = new Set();
    root.classList.add('csim');
    root.innerHTML = '';

    const tbl = document.createElement('table');
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th style="text-align:left">candidate → beats</th>' +
      E.map((e, j) => `<th data-el="${j}">${e.name}<br><span class="m">×${e.w}</span></th>`).join('') +
      '<th>weight of its slice</th></tr>';
    tbl.appendChild(thead);
    const tbody = document.createElement('tbody');
    S.forEach((s, i) => {
      const tr = document.createElement('tr'); tr.dataset.set = i;
      const sw = s.covers.reduce((a, j) => a + E[j].w, 0);
      tr.innerHTML = `<td class="name">${s.name}</td>` +
        E.map((e, j) => `<td class="dot ${s.covers.includes(j) ? 'hit' : ''}">${s.covers.includes(j) ? '●' : '·'}</td>`).join('') +
        `<td class="m">${sw}</td>`;
      tr.querySelector('.name').addEventListener('click', () => { chosen.has(i) ? chosen.delete(i) : chosen.add(i); render(); });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    root.appendChild(tbl);

    const panel = document.createElement('div'); panel.className = 'panel';
    panel.innerHTML = '<div class="mc"><h4>Read as maximum coverage (k = ' + k + ')</h4><div class="mcout"></div></div>' +
                      '<div class="sc"><h4>Read as set cover</h4><div class="scout"></div></div>';
    root.appendChild(panel);

    const ctl = document.createElement('div'); ctl.className = 'ctl';
    const bG = document.createElement('button'); bG.textContent = 'Greedy step: add largest uncovered weight';
    const bR = document.createElement('button'); bR.textContent = 'Reset';
    const hint = document.createElement('span'); hint.className = 'note';
    hint.textContent = 'Click a candidate to add / remove it.';
    ctl.append(bG, bR, hint); root.appendChild(ctl);

    function covered() { const c = new Set(); chosen.forEach(i => S[i].covers.forEach(j => c.add(j))); return c; }
    function coveredWeight(c) { let w = 0; c.forEach(j => w += E[j].w); return w; }
    bG.addEventListener('click', () => {
      const c = covered(); let best = -1, bestGain = 0;
      S.forEach((s, i) => { if (chosen.has(i)) return; const g = s.covers.filter(j => !c.has(j)).reduce((a, j) => a + E[j].w, 0); if (g > bestGain) { bestGain = g; best = i; } });
      if (best >= 0) chosen.add(best);
      render();
    });
    bR.addEventListener('click', () => { chosen.clear(); render(); });

    function render() {
      const c = covered(), w = coveredWeight(c), n = chosen.size, all = c.size === E.length;
      tbody.querySelectorAll('tr').forEach(tr => tr.classList.toggle('on', chosen.has(+tr.dataset.set)));
      thead.querySelectorAll('th[data-el]').forEach(th => th.classList.toggle('covered', c.has(+th.dataset.el)));
      const pct = total ? Math.round(100 * w / total) : 0;
      let mc = `<p>Chosen: <b>${n}</b> of budget <b>${k}</b>${n > k ? ' <span class="bad">— over budget: not a feasible answer</span>' : ''}</p>` +
               `<p>Covered weight: <b>${w}</b> / ${total} (${pct}% of the meta)</p>`;
      if (cfg.optMaxCov != null && n <= k) mc += `<p>Best possible at k=${k}: <b>${cfg.optMaxCov}</b>${w === cfg.optMaxCov ? ' <span class="good">— optimal</span>' : ''}</p>`;
      let sc = `<p>Everything covered? <b class="${all ? 'good' : 'bad'}">${all ? 'yes' : 'no'}</b>${all ? '' : ' — not a feasible answer'}</p>` +
               `<p>Sets used: <b>${n}</b>${all && cfg.optSetCover != null ? ` (minimum possible: <b>${cfg.optSetCover}</b>${n === cfg.optSetCover ? ' <span class="good">— optimal</span>' : ''})` : ''}</p>`;
      panel.querySelector('.mcout').innerHTML = mc;
      panel.querySelector('.scout').innerHTML = sc;
    }
    render();
    return { chosen, render };
  }
  return { mount };
})();
