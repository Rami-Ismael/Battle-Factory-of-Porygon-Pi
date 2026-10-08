/* Common-random-numbers paired-comparison demo. Reusable component.
   Usage: <div id="crn-sim"></div><script src="../assets/crn-sim.js"></script>
   then  CRNSim.mount('#crn-sim', opts?)
   Model: a meta of opponents with draw weights; two teams A and B with a true
   win probability against each opponent. One estimate of Δ = P(A wins) − P(B wins)
   uses n battles per team; the widget repeats that R times under three seeding
   modes and histograms the R estimates of Δ̂:
     independent      — each battle draws its own opponent and its own luck  (X_I)
     paired opponent  — battle i of A and B faces the SAME opponent draw
     fully paired     — same opponent AND same luck u per pair               (X_D)
   Battle model: draw opponent j ~ weights, luck u ~ U(0,1); win iff u < p_team[j].
   The variance ratio vs independent is reported as "×k fewer battles for the
   same precision" — the whole point of CRN. */
(function (global) {
  const css = `
  .crnsim { border:1px solid var(--rule,#ccc); border-radius:6px; padding:1rem 1.2rem; margin:1.4rem 0; max-width:46rem; font-size:.92em; }
  .crnsim .cols { display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:.8rem; }
  .crnsim .col { flex:1 1 12rem; min-width:11rem; }
  .crnsim .col h5 { margin:.2rem 0 .4rem; font-size:.75em; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-soft,#555); }
  .crnsim .hist { display:flex; align-items:flex-end; gap:1px; height:70px; border-bottom:1px solid var(--rule,#ccc); }
  .crnsim .hist div { flex:1; background:var(--setc,#3d5a80); min-height:1px; }
  .crnsim .hist div.truth { background:var(--accent,#8a3b12); }
  .crnsim .stats { font-family:var(--mono,monospace); font-size:.82em; margin-top:.4rem; color:var(--ink-soft,#555); }
  .crnsim .stats b { color:var(--ink,#1d1d1b); }
  .crnsim .gain { color:var(--cover,#2c6e49); font-weight:600; }
  .crnsim button { font:inherit; padding:.4rem .8rem; border:1px solid var(--rule,#ccc); border-radius:4px; background:transparent; color:inherit; cursor:pointer; margin-right:.4rem; }
  .crnsim button:hover { border-color:var(--accent,#8a3b12); }
  .crnsim button.on { background:var(--ok-bg,#e5f1e9); border-color:var(--cover,#2c6e49); }
  .crnsim .note { font-size:.85em; color:var(--ink-soft,#555); margin:.6rem 0 0; }`;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  const DEFAULTS = {
    weights: [0.5, 0.3, 0.2],            // meta: opponent draw probabilities
    pA: [0.80, 0.55, 0.30],              // P(team A beats opponent j)
    pB: [0.75, 0.50, 0.28],              // P(team B beats opponent j)
    R: 400,                              // repeated estimates per mode
    nChoices: [25, 100, 400],            // battles per team per estimate
  };

  function drawOpp(w) {
    let u = Math.random(), acc = 0;
    for (let j = 0; j < w.length; j++) { acc += w[j]; if (u < acc) return j; }
    return w.length - 1;
  }

  function estimate(mode, n, o) {
    let wa = 0, wb = 0;
    for (let i = 0; i < n; i++) {
      if (mode === 'independent') {
        wa += Math.random() < o.pA[drawOpp(o.weights)] ? 1 : 0;
        wb += Math.random() < o.pB[drawOpp(o.weights)] ? 1 : 0;
      } else {
        const j = drawOpp(o.weights);                     // shared opponent draw
        if (mode === 'paired-opponent') {
          wa += Math.random() < o.pA[j] ? 1 : 0;
          wb += Math.random() < o.pB[j] ? 1 : 0;
        } else {                                          // fully paired: shared luck too
          const u = Math.random();
          wa += u < o.pA[j] ? 1 : 0;
          wb += u < o.pB[j] ? 1 : 0;
        }
      }
    }
    return (wa - wb) / n;
  }

  function run(mode, n, o) {
    const xs = [];
    for (let r = 0; r < o.R; r++) xs.push(estimate(mode, n, o));
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    const v = xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1);
    return { xs, mean: m, sd: Math.sqrt(v), varc: v };
  }

  function histogram(xs, lo, hi, bins, truth) {
    const counts = new Array(bins).fill(0);
    xs.forEach(x => {
      let b = Math.floor(((x - lo) / (hi - lo)) * bins);
      b = Math.max(0, Math.min(bins - 1, b)); counts[b]++;
    });
    const peak = Math.max(...counts, 1);
    const tb = Math.max(0, Math.min(bins - 1, Math.floor(((truth - lo) / (hi - lo)) * bins)));
    return counts.map((c, i) =>
      `<div style="height:${Math.round((c / peak) * 68)}px"${i === tb ? ' class="truth"' : ''}></div>`).join('');
  }

  const MODES = [
    { key: 'independent',     label: 'independent (X_I)' },
    { key: 'paired-opponent', label: 'paired opponent' },
    { key: 'fully-paired',    label: 'fully paired (X_D)' },
  ];

  function mount(sel, opts) {
    const o = Object.assign({}, DEFAULTS, opts || {});
    const root = document.querySelector(sel);
    if (!root) return;
    root.className = 'crnsim';
    const truth = o.weights.reduce((a, w, j) => a + w * (o.pA[j] - o.pB[j]), 0);
    let n = o.nChoices[1];

    function render() {
      const results = MODES.map(m => ({ m, r: run(m.key, n, o) }));
      const lo = Math.min(...results.map(x => x.r.mean - 3.5 * x.r.sd), truth - 0.02);
      const hi = Math.max(...results.map(x => x.r.mean + 3.5 * x.r.sd), truth + 0.02);
      const base = results[0].r.varc;
      root.innerHTML =
        `<div><strong>Estimating Δ = P(T₁ wins) − P(T₂ wins)</strong> — true Δ = ${truth.toFixed(3)} (orange bin). ` +
        `${o.R} repeated estimates, each from <span class="m">n</span> battles per team.</div>` +
        `<div style="margin:.6rem 0">` + o.nChoices.map(c =>
          `<button data-n="${c}" class="${c === n ? 'on' : ''}">n = ${c}</button>`).join('') +
        `<button data-n="rerun">resample</button></div>` +
        `<div class="cols">` + results.map(({ m, r }) => {
          const k = base / r.varc;
          return `<div class="col"><h5>${m.label}</h5>` +
            `<div class="hist">${histogram(r.xs, lo, hi, 36, truth)}</div>` +
            `<div class="stats">sd(Δ̂) = <b>${r.sd.toFixed(3)}</b><br>` +
            (m.key === 'independent' ? 'baseline'
              : `<span class="gain">×${k.toFixed(1)} fewer battles</span> for equal precision`) +
            `</div></div>`;
        }).join('') + `</div>` +
        `<p class="note">Same battle budget in every column — the spread shrinks because pairing makes ` +
        `cov(T̂₁, T̂₂) &gt; 0, and that covariance is subtracted from var(Δ̂). "Fully paired" shares the luck ` +
        `draw <span class="m">u</span> inside the battle model; a real Showdown seed couples less than this ` +
        `once two battles' histories diverge, so expect real gains between the middle and right columns.</p>`;
      root.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
        if (b.dataset.n !== 'rerun') n = Number(b.dataset.n);
        render();
      }));
    }
    render();
  }

  global.CRNSim = { mount };
})(window);
