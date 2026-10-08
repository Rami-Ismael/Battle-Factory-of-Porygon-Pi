/* Filtered-BC weight explorer. Reusable across lessons on offline RL.
   Usage: <div id="filter-sim"></div>  then  FilterSim.mount('#filter-sim')
   Shows, for one timestep: critic Q-values for the legal actions, the action the
   human took, the advantage A = Q(h,a) − E_{a'~π}[Q(h,a')], and the BC weight w
   under IL / Exp (AWAC) / Binary (CRR) — Eq. (2) and Table 1 of Metamon.
   No randomness at import; the learner drives it. */
(function () {
  const css = `
  .fsim { border:1px solid var(--rule,#ccc); border-radius:6px; padding:1rem 1.2rem; margin:1.4rem 0; max-width:46rem; font-size:.92em; }
  .fsim table { border-collapse:collapse; width:100%; margin:.6rem 0; }
  .fsim th, .fsim td { text-align:left; padding:.3rem .5rem; border-top:1px solid var(--rule,#ccc); }
  .fsim th { font-size:.75em; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-soft,#555); border-top:0; }
  .fsim input[type=range] { width:9rem; vertical-align:middle; }
  .fsim input[type=radio] { margin-right:.3rem; }
  .fsim .w { font-family:var(--mono,monospace); }
  .fsim .bar { display:inline-block; height:.7em; background:var(--cover,#2c6e49); vertical-align:middle; margin-left:.4rem; border-radius:2px; }
  .fsim .ctl { margin:.6rem 0; color:var(--ink-soft,#555); }
  .fsim .read { margin-top:.8rem; padding:.6rem .8rem; background:var(--ok-bg,#e5f1e9); border-radius:4px; }
  .fsim .zero { color:var(--warn,#9c4a1a); }`;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  function mount(sel, opts) {
    const root = document.querySelector(sel); if (!root) return;
    const names = (opts && opts.actions) || ['Move 1', 'Move 2', 'Move 3', 'Switch 1'];
    const q = (opts && opts.q) ? opts.q.slice() : [0.3, -0.2, 0.1, 0.0];
    const p = (opts && opts.pi) ? opts.pi.slice() : [0.4, 0.2, 0.3, 0.1];
    let taken = (opts && opts.taken) || 0;
    let beta = 1.0;
    root.className = 'fsim';
    root.innerHTML = `
      <div class="ctl">Critic Q-values for each legal action (drag), the policy's current probabilities π (fixed), and the action the <em>human</em> actually clicked (radio).</div>
      <table><thead><tr><th>Action</th><th>π(a|h)</th><th>Q(h,a)</th><th>human took</th></tr></thead><tbody></tbody></table>
      <div class="ctl">β for the Exp weight: <input type="range" id="fs-beta" min="0.25" max="8" step="0.25" value="1"> <span id="fs-beta-v">1.0</span>
      &nbsp;·&nbsp; clip at <span class="w">w ≤ 20</span> (Metamon clips; AWAC convention)</div>
      <div class="read" id="fs-read"></div>`;
    const tb = root.querySelector('tbody');
    names.forEach((n, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${n}</td><td class="w">${p[i].toFixed(2)}</td>
        <td><input type="range" data-i="${i}" min="-1" max="1" step="0.05" value="${q[i]}"> <span class="w" id="fs-q${i}">${q[i].toFixed(2)}</span></td>
        <td><input type="radio" name="fs-taken" data-i="${i}" ${i === taken ? 'checked' : ''}></td>`;
      tb.appendChild(tr);
    });
    root.querySelectorAll('input[type=range][data-i]').forEach(r => r.addEventListener('input', e => {
      const i = +e.target.dataset.i; q[i] = +e.target.value; root.querySelector('#fs-q' + i).textContent = q[i].toFixed(2); render();
    }));
    root.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', e => { taken = +e.target.dataset.i; render(); }));
    const bs = root.querySelector('#fs-beta');
    bs.addEventListener('input', () => { beta = +bs.value; root.querySelector('#fs-beta-v').textContent = beta.toFixed(2); render(); });

    function render() {
      const v = q.reduce((s, x, i) => s + p[i] * x, 0);           // E_{a'~π} Q(h,a')
      const A = q[taken] - v;
      const wExp = Math.min(20, Math.exp(beta * A));
      const wBin = A > 0 ? 1 : 0;
      const bar = w => `<span class="bar" style="width:${Math.min(200, w * 10)}px"></span>`;
      root.querySelector('#fs-read').innerHTML =
        `V(h) = Σ π·Q = <span class="w">${v.toFixed(3)}</span> &nbsp;·&nbsp; A(h, ${names[taken]}) = Q − V = <span class="w">${A.toFixed(3)}</span><br>
         w<sub>IL</sub> = <span class="w">1.00</span>${bar(1)}<br>
         w<sub>Exp</sub> = exp(β·A) = <span class="w">${wExp.toFixed(2)}</span>${bar(wExp)}<br>
         w<sub>Binary</sub> = 𝟙[A&gt;0] = <span class="w ${wBin ? '' : 'zero'}">${wBin.toFixed(2)}</span>${bar(wBin)}
         ${wBin === 0 ? ' <span class="zero">— this human decision is dropped from the BC loss entirely</span>' : ''}`;
    }
    render();
  }
  window.FilterSim = { mount };
})();
