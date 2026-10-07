/* Coupling demo: marginals fixed, joint distribution is a design choice. Reusable.
   Usage: <div id="coupling-sim"></div><script src="../assets/coupling-sim.js"></script>
   then  CouplingSim.mount('#coupling-sim', {p1:0.55, p2:0.50})
   Model: two Bernoulli outcomes built from uniform draws — X = 1{U < p1} (team T₁ wins),
   Y = 1{V < p2} (team T₂ wins). Independent mode draws U and V separately; shared mode
   sets V = U. Either way each outcome ALONE keeps its marginal (P(X=1)=p1, P(Y=1)=p2);
   only the 2×2 joint table changes — which is the entire point being taught.
   Part A: drag one shared U and watch both outcomes move together.
   Part B: sample many pairs both ways; compare joint tables, marginals, cov, var(X−Y). */
(function (global) {
  const css = `
  .cpl { border:1px solid var(--rule,#ccc); border-radius:6px; padding:1rem 1.2rem; margin:1.4rem 0; max-width:46rem; font-size:.92em; }
  .cpl h5 { margin:1rem 0 .4rem; font-size:.75em; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-soft,#555); }
  .cpl .line { position:relative; height:26px; border-radius:4px; overflow:hidden; margin:.5rem 0 .2rem; }
  .cpl .zone { position:absolute; top:0; bottom:0; }
  .cpl .zone.bothwin { background:var(--ok-bg,#e5f1e9); }
  .cpl .zone.split { background:var(--bad-bg,#f5e3dc); }
  .cpl .zone.bothlose { background:transparent; border-left:1px solid var(--rule,#ccc); }
  .cpl .marker { position:absolute; top:0; bottom:0; width:2px; background:var(--accent,#8a3b12); }
  .cpl input[type=range] { width:100%; }
  .cpl .chips { display:flex; gap:.6rem; margin:.4rem 0; flex-wrap:wrap; font-family:var(--mono,monospace); font-size:.82em; }
  .cpl .chip { padding:.15em .6em; border:1px solid var(--rule,#ccc); border-radius:3px; }
  .cpl .chip.win { background:var(--ok-bg,#e5f1e9); }
  .cpl .chip.loss { background:var(--bad-bg,#f5e3dc); }
  .cpl .cols { display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:.6rem; }
  .cpl .col { flex:1 1 15rem; min-width:14rem; }
  .cpl table { border-collapse:collapse; font-family:var(--mono,monospace); font-size:.8em; margin:.4rem 0; }
  .cpl td, .cpl th { border:1px solid var(--rule,#ccc); padding:.25rem .55rem; text-align:center; }
  .cpl th { font-weight:600; }
  .cpl td.marg, .cpl th.marg { color:var(--ink-soft,#555); border-style:dashed; }
  .cpl td.hot { outline:2px solid var(--accent,#8a3b12); }
  .cpl .stats { font-family:var(--mono,monospace); font-size:.82em; color:var(--ink-soft,#555); }
  .cpl .stats b { color:var(--ink,#1d1d1b); }
  .cpl button { font:inherit; padding:.4rem .8rem; border:1px solid var(--rule,#ccc); border-radius:4px; background:transparent; color:inherit; cursor:pointer; }
  .cpl button:hover { border-color:var(--accent,#8a3b12); }
  .cpl .note { font-size:.85em; color:var(--ink-soft,#555); margin:.5rem 0 0; }`;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  function pct(x, n) { return (100 * x / n).toFixed(1) + '%'; }

  function jointTable(c, n, hotCell) {
    // c = {w11,w10,w01,w00}: first index X (T₁), second Y (T₂); 1 = win
    const r1 = c.w11 + c.w10, r0 = c.w01 + c.w00;   // X marginals
    const k1 = c.w11 + c.w01, k0 = c.w10 + c.w00;   // Y marginals
    const hot = (k) => hotCell === k ? ' class="hot"' : '';
    return `<table>
      <tr><th></th><th>T₂ win</th><th>T₂ loss</th><th class="marg">T₁ marginal</th></tr>
      <tr><th>T₁ win</th><td${hot('w11')}>${pct(c.w11, n)}</td><td${hot('w10')}>${pct(c.w10, n)}</td><td class="marg">${pct(r1, n)}</td></tr>
      <tr><th>T₁ loss</th><td${hot('w01')}>${pct(c.w01, n)}</td><td${hot('w00')}>${pct(c.w00, n)}</td><td class="marg">${pct(r0, n)}</td></tr>
      <tr><th class="marg">T₂ marginal</th><td class="marg">${pct(k1, n)}</td><td class="marg">${pct(k0, n)}</td><td class="marg"></td></tr>
    </table>`;
  }

  function sample(shared, n, p1, p2) {
    const c = { w11: 0, w10: 0, w01: 0, w00: 0 };
    let sx = 0, sy = 0, sxy = 0, sd2 = 0, sd = 0;
    for (let i = 0; i < n; i++) {
      const u = Math.random(), v = shared ? u : Math.random();
      const x = u < p1 ? 1 : 0, y = v < p2 ? 1 : 0;
      c['w' + x + y]++;
      sx += x; sy += y; sxy += x * y; const d = x - y; sd += d; sd2 += d * d;
    }
    const mx = sx / n, my = sy / n, md = sd / n;
    return { c, cov: sxy / n - mx * my, varD: sd2 / n - md * md };
  }

  function mount(sel, opts) {
    const o = Object.assign({ p1: 0.55, p2: 0.50, n: 2000 }, opts || {});
    const root = document.querySelector(sel);
    if (!root) return;
    root.className = 'cpl';
    root.innerHTML = `
      <div><strong>Both outcomes from one uniform draw</strong> — T₁ wins iff U &lt; ${o.p1}, T₂ wins iff U &lt; ${o.p2}.</div>
      <h5>Part A — drag U</h5>
      <div class="line">
        <div class="zone bothwin" style="left:0;width:${o.p2 * 100}%"></div>
        <div class="zone split" style="left:${o.p2 * 100}%;width:${(o.p1 - o.p2) * 100}%"></div>
        <div class="zone bothlose" style="left:${o.p1 * 100}%;width:${(1 - o.p1) * 100}%"></div>
        <div class="marker"></div>
      </div>
      <input type="range" min="0" max="1" step="0.001" value="0.3">
      <div class="chips"></div>
      <p class="note">Green zone: both win. Orange sliver [${o.p2}, ${o.p1}): the ONLY place the shared draw separates them. Right of ${o.p1}: both lose. Notice what can never happen: T₂ winning while T₁ loses.</p>
      <h5>Part B — same marginals, two different joints</h5>
      <button>draw ${o.n} pairs, both ways</button>
      <div class="cols"></div>
      <p class="note">Compare the dashed marginals across the two tables — near-identical, as unbiasedness demands. Everything that changed is inside the 2×2: sharing moved mass out of the disagreement corners (outlined) into the diagonal. That reallocation IS the coupling, and cov &gt; 0 is its receipt.</p>`;

    const slider = root.querySelector('input');
    const marker = root.querySelector('.marker');
    const chips = root.querySelector('.chips');
    function partA() {
      const u = Number(slider.value);
      marker.style.left = (u * 100) + '%';
      const x = u < o.p1, y = u < o.p2;
      chips.innerHTML =
        `<span class="chip">U = ${u.toFixed(3)}</span>` +
        `<span class="chip ${x ? 'win' : 'loss'}">T₁ ${x ? 'win' : 'loss'}</span>` +
        `<span class="chip ${y ? 'win' : 'loss'}">T₂ ${y ? 'win' : 'loss'}</span>` +
        `<span class="chip">difference ${x - y >= 0 ? '+' : ''}${(x ? 1 : 0) - (y ? 1 : 0)}</span>`;
    }
    slider.addEventListener('input', partA); partA();

    const cols = root.querySelector('.cols');
    root.querySelector('button').addEventListener('click', () => {
      const ind = sample(false, o.n, o.p1, o.p2);
      const shr = sample(true, o.n, o.p1, o.p2);
      cols.innerHTML =
        `<div class="col"><h5>independent draws</h5>${jointTable(ind.c, o.n, 'w01')}
          <div class="stats">cov = <b>${ind.cov.toFixed(3)}</b> · var(difference) = <b>${ind.varD.toFixed(3)}</b></div></div>` +
        `<div class="col"><h5>one shared draw</h5>${jointTable(shr.c, o.n, 'w01')}
          <div class="stats">cov = <b>${shr.cov.toFixed(3)}</b> · var(difference) = <b>${shr.varD.toFixed(3)}</b>
          <br><span style="color:var(--cover,#2c6e49);font-weight:600">×${(ind.varD / Math.max(shr.varD, 1e-9)).toFixed(1)} smaller</span></div></div>`;
    });
  }

  global.CouplingSim = { mount };
})(window);
