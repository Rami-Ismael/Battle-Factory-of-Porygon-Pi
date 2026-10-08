/* Double Oracle stepper for small zero-sum matrix games. Reusable component.
   Usage: <div id="do-sim"></div><script src="../assets/do-sim.js"></script>
   then  DOSim.mount('#do-sim', {names:[...], M:[[...]], start:0})
   M[i][j] = payoff to the ROW player when row i meets column j (antisymmetric => symmetric game).
   Each step = one DO iteration (McMahan, Gordon & Blum 2003):
     1. solve the restricted game (rows R̄, cols C̄) for an equilibrium (p, q)   — the MSS
     2. row oracle: best pure response r to q over ALL rows; col oracle: c to p  — the ORACLE (exact BR)
     3. bounds: v_l = V(r,q) ≤ value ≤ v_u = V(p,c); add r to R̄, c to C̄
     4. stop when r ∈ R̄ and c ∈ C̄ (equivalently v_l = v_u)
   Restricted-game equilibrium is computed by fictitious play (20k iterations) — exact enough for ≤ 5×5. */
(function (global) {
  const css = `
  .dosim { border:1px solid var(--rule,#ccc); border-radius:6px; padding:1rem 1.2rem; margin:1.4rem 0; max-width:46rem; font-size:.92em; }
  .dosim table { border-collapse:collapse; margin:.6rem 0; }
  .dosim td, .dosim th { padding:.25rem .6rem; text-align:center; border:1px solid var(--rule,#ccc); font-family:var(--mono,monospace); font-size:.85em; }
  .dosim th { font-weight:600; }
  .dosim td.in, .dosim th.in { background:var(--ok-bg,#e5f1e9); }
  .dosim td.new, .dosim th.new { outline:2px solid var(--accent,#8a3b12); }
  .dosim .row { display:flex; gap:1.5rem; flex-wrap:wrap; align-items:flex-start; }
  .dosim .panel { min-width:14rem; }
  .dosim .panel h5 { margin:.4rem 0 .2rem; font-size:.75em; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-soft,#555); }
  .dosim button { font:inherit; padding:.4rem .8rem; border:1px solid var(--rule,#ccc); border-radius:4px; background:transparent; color:inherit; cursor:pointer; margin-right:.4rem; }
  .dosim button:hover { border-color:var(--accent,#8a3b12); }
  .dosim .log { margin-top:.8rem; font-size:.9em; color:var(--ink-soft,#555); }
  .dosim .log li { max-width:none; }
  .dosim .done { color:var(--cover,#2c6e49); font-weight:600; }
  .dosim .bound { font-family:var(--mono,monospace); font-size:.85em; }`;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  function fmt(x) { return (Math.abs(x) < 1e-9 ? 0 : x).toFixed(2); }

  // Equilibrium of restricted zero-sum game via fictitious play. Returns {p,q,value} over restricted indices.
  function solve(M, R, C) {
    const n = R.length, m = C.length;
    if (n === 1 && m === 1) return { p: [1], q: [1], value: M[R[0]][C[0]] };
    const cntR = new Array(n).fill(0), cntC = new Array(m).fill(0);
    let r = 0, c = 0, iters = 20000;
    const sumR = new Array(n).fill(0), sumC = new Array(m).fill(0); // cumulative payoffs
    for (let t = 0; t < iters; t++) {
      cntR[r]++; cntC[c]++;
      for (let i = 0; i < n; i++) sumR[i] += M[R[i]][C[c]];
      for (let j = 0; j < m; j++) sumC[j] += M[R[r]][C[j]];
      r = argmax(sumR); c = argmin(sumC);
    }
    const p = cntR.map(x => x / iters), q = cntC.map(x => x / iters);
    let v = 0; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) v += p[i] * q[j] * M[R[i]][C[j]];
    return { p, q, value: v };
  }
  function argmax(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k] + 1e-12) k = i; return k; }
  function argmin(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[k] - 1e-12) k = i; return k; }

  function mount(sel, cfg) {
    const root = document.querySelector(sel); if (!root) return;
    const { names, M } = cfg; const N = names.length;
    let R, C, log, finished, lastNew;
    const el = document.createElement('div'); el.className = 'dosim'; root.appendChild(el);

    function reset() { R = [cfg.start || 0]; C = [cfg.start || 0]; log = []; finished = false; lastNew = { r: null, c: null }; render(); }

    function step() {
      if (finished) return;
      const { p, q, value } = solve(M, R, C);
      // oracles: exact best responses over the FULL strategy set
      const rowPay = names.map((_, i) => C.reduce((s, cj, jj) => s + q[jj] * M[i][cj], 0));
      const colPay = names.map((_, j) => R.reduce((s, ri, ii) => s + p[ii] * M[ri][j], 0));
      const r = argmax(rowPay), c = argmin(colPay);
      const vl = rowPay[r], vu = colPay[c];
      const rNew = !R.includes(r), cNew = !C.includes(c);
      const it = log.length + 1;
      const pStr = R.map((i, k) => `${names[i]} ${fmt(p[k])}`).join(', ');
      log.push({ it, pStr, value, r, c, vl, vu, rNew, cNew });
      if (rNew) R.push(r); if (cNew) C.push(c);
      lastNew = { r: rNew ? r : null, c: cNew ? c : null };
      if (!rNew && !cNew) finished = true;
      render();
    }

    function render() {
      let h = `<div class="row"><div class="panel"><h5>Full game — payoff to row player</h5><table><tr><th></th>`;
      names.forEach((nm, j) => h += `<th class="${C.includes(j) ? 'in' : ''} ${lastNew.c === j ? 'new' : ''}">${nm}</th>`);
      h += `</tr>`;
      names.forEach((nm, i) => {
        h += `<tr><th class="${R.includes(i) ? 'in' : ''} ${lastNew.r === i ? 'new' : ''}">${nm}</th>`;
        names.forEach((_, j) => h += `<td class="${R.includes(i) && C.includes(j) ? 'in' : ''}">${fmt(M[i][j])}</td>`);
        h += `</tr>`;
      });
      h += `</table><p class="note">Shaded = the <b>restricted game</b> (rows R̄ × cols C̄). Outlined = added this step.</p></div>`;
      h += `<div class="panel"><h5>Controls</h5><button data-a="step">${finished ? 'Converged' : 'DO step'}</button><button data-a="reset">Reset</button>`;
      h += `<h5>Population Π (the survey's view)</h5><p>${R.map(i => names[i]).join(' → ')}</p></div></div>`;
      if (log.length) {
        h += `<ol class="log">`;
        log.forEach(L => {
          h += `<li><b>Iter ${L.it}.</b> MSS — Nash of restricted game: (${L.pStr}), value ${fmt(L.value)}. `;
          h += `ORACLE — row best response <b>${names[L.r]}</b>${L.rNew ? ' (new, added to Π)' : ' (already in Π)'}; column best response <b>${names[L.c]}</b>${L.cNew ? ' (new)' : ' (already in)'}. `;
          h += `<span class="bound">bounds: ${fmt(L.vl)} ≤ value ≤ ${fmt(L.vu)}</span></li>`;
        });
        h += `</ol>`;
        if (finished) h += `<p class="done">Both oracles returned strategies already in the restricted sets → v_l = v_u → Theorem 1: the restricted-game equilibrium is an equilibrium of the full game. ${R.length < N ? `Note: ${N - R.length} of ${N} strategies never entered Π.` : 'Every strategy entered Π — the worst case McMahan et al. warn about.'}</p>`;
      }
      el.innerHTML = h;
      el.querySelector('[data-a=step]').onclick = step;
      el.querySelector('[data-a=reset]').onclick = reset;
    }
    reset();
  }
  global.DOSim = { mount };
})(window);
