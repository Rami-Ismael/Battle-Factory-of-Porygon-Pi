/* ArchiveSim — the two-gates MAP-Elites insert widget shared by the QD lessons.
   Usage:
     <div id="sim"></div>
     <script src="../assets/archive-sim.js"></script>
     <script>ArchiveSim.mount(document.getElementById('sim'));</script>
   Shows a 2-D behaviour archive (x: mean battle length, y: KO spread). Each click
   breeds a child from a random elite and animates BOTH gates: Gate 1 — its
   measures decide WHERE it lands; Gate 2 — its objective decides WHETHER it stays
   (empty niche admits anything; occupied niche keeps the higher win rate).
   Counters track coverage and the sum of elite objectives (the QD-score reading). */
window.ArchiveSim = (function () {
  const css = `
  .asim { border: 1px solid var(--rule,#ccc); border-radius: 6px; padding: 1rem 1.2rem; margin: 1.5rem 0; font-size: .92em; }
  .asim .board { display: grid; grid-template-columns: 1.6rem repeat(6, 1fr); gap: 4px; align-items: stretch; }
  .asim .ylab { grid-column: 1; grid-row: 1 / span 4; writing-mode: vertical-rl; transform: rotate(180deg); font-size: .72em; color: var(--ink-soft,#555);
                text-align: center; font-style: italic; align-self: center; }
  .asim .cell { min-height: 72px; border: 1px solid var(--rule,#ccc); border-radius: 4px; display: flex;
                flex-direction: column; align-items: center; justify-content: center; gap: 2px;
                background: color-mix(in srgb, var(--paper,#fff) 92%, var(--rule)); transition: box-shadow .25s, background .35s; }
  .asim .cell .obj { font-family: var(--mono, monospace); font-weight: 700; }
  .asim .cell.empty::after { content: '·'; color: var(--ink-soft,#777); }
  .asim .cell.parent { box-shadow: 0 0 0 3px var(--accent,#8a3b12) inset; }
  .asim .cell.target { box-shadow: 0 0 0 3px var(--setc,#3d5a80); }
  .asim .cell.win { background: var(--ok-bg,#e5f1e9); }
  .asim .cell.lose { background: var(--bad-bg,#f5e3dc); }
  .asim .cell .tag { font-size: .62em; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-soft,#555); }
  .asim .xlab { display: grid; grid-template-columns: 1.6rem repeat(6, 1fr); gap: 4px; margin-top: .3rem; }
  .asim .xlab span { text-align: center; font-size: .72em; color: var(--ink-soft,#555); }
  .asim .xlab .cap { grid-column: 2 / span 6; font-style: italic; }
  .asim .story { min-height: 4.2em; margin-top: .8rem; padding: .6rem .9rem; border-left: 3px solid var(--rule,#ccc);
                 font-size: .95em; max-width: 40rem; }
  .asim .story b.g1 { color: var(--setc,#3d5a80); } .asim .story b.g2 { color: var(--accent,#8a3b12); }
  .asim .counters { display: flex; gap: 1.4rem; flex-wrap: wrap; margin-top: .7rem; font-family: var(--mono, monospace); font-size: .85em; }
  .asim .ctl { display: flex; gap: .6rem; flex-wrap: wrap; align-items: center; margin-top: .8rem; }
  .asim button { font: inherit; font-size: .9em; padding: .35rem .8rem; border: 1px solid var(--rule,#ccc);
                 border-radius: 4px; background: transparent; color: inherit; cursor: pointer; }
  .asim button:hover:not(:disabled) { border-color: var(--accent,#8a3b12); }
  .asim button:disabled { opacity: .45; cursor: wait; }`;
  let injected = false;

  const COLS = [3, 5, 7, 9, 11, 13];
  const ROWS = [1, 2, 3, 4];
  const START = [
    { t: 3, s: 4, obj: 51 }, { t: 5, s: 3, obj: 63 }, { t: 7, s: 2, obj: 66 },
    { t: 9, s: 2, obj: 59 }, { t: 13, s: 1, obj: 57 }, { t: 11, s: 3, obj: 48 },
  ];

  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  function mount(root) {
    if (!injected) { const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s); injected = true; }
    root.classList.add('asim');
    let archive, stats;

    function reset() {
      archive = {}; stats = { bred: 0, admitted: 0, replaced: 0 };
      START.forEach(e => { archive[cellKey(e.t, e.s)] = { t: e.t, s: e.s, obj: e.obj }; });
    }
    const cellKey = (t, s) => COLS.indexOf(t) + ',' + ROWS.indexOf(s);

    root.innerHTML = '';
    const board = document.createElement('div'); board.className = 'board';
    const ylab = document.createElement('div'); ylab.className = 'ylab';
    ylab.textContent = 'KO spread — how many of your four battlers scored ≥1 knockout';
    board.appendChild(ylab);
    ROWS.forEach(s => {
      COLS.forEach(t => {
        const c = document.createElement('div'); c.className = 'cell';
        c.dataset.key = cellKey(t, s); c.dataset.t = t; c.dataset.s = s;
        board.appendChild(c);
      });
    });
    root.appendChild(board);
    const xlab = document.createElement('div'); xlab.className = 'xlab';
    xlab.innerHTML = '<span></span>' + COLS.map(t => `<span>${t}</span>`).join('') +
      '<span class="cap">mean battle length (turns) →</span>';
    root.appendChild(xlab);

    const story = document.createElement('div'); story.className = 'story';
    story.innerHTML = 'Press <b>Breed a child</b>: an elite is picked, mutated, and walked through both gates.';
    root.appendChild(story);
    const counters = document.createElement('div'); counters.className = 'counters'; root.appendChild(counters);

    const ctl = document.createElement('div'); ctl.className = 'ctl';
    const bOne = document.createElement('button'); bOne.textContent = 'Breed a child';
    const bMany = document.createElement('button'); bMany.textContent = 'Run ×25';
    const bRst = document.createElement('button'); bRst.textContent = 'Reset archive';
    ctl.append(bOne, bMany, bRst); root.appendChild(ctl);

    function render(cls, msg) {
      board.querySelectorAll('.cell').forEach(c => {
        const e = archive[c.dataset.key];
        c.className = 'cell' + (e ? '' : ' empty') + (cls && c.dataset.key === cls.key ? ' ' + cls.name : '');
        c.innerHTML = e ? `<span class="obj">${e.obj}%</span>` +
          (cls && cls.tag && c.dataset.key === cls.key ? `<span class="tag">${cls.tag}</span>` : '') : '';
      });
      const keys = Object.keys(archive);
      counters.innerHTML =
        `<span>cells filled: <b>${keys.length}/24</b></span>` +
        `<span>Σ elite objectives (QD-score reading): <b>${keys.reduce((a, k) => a + archive[k].obj, 0)}</b></span>` +
        `<span>children bred: <b>${stats.bred}</b></span>` +
        `<span>admitted: <b>${stats.admitted}</b> (${stats.replaced} by replacement)</span>`;
      if (msg) story.innerHTML = msg;
    }

    async function step() {
      const keys = Object.keys(archive);
      const parent = archive[keys[Math.floor(Math.random() * keys.length)]];
      const child = {
        t: COLS[clamp(COLS.indexOf(parent.t) + Math.round(gauss()), 0, COLS.length - 1)],
        s: ROWS[clamp(ROWS.indexOf(parent.s) + Math.round(gauss()), 0, ROWS.length - 1)],
        obj: Math.round(clamp(parent.obj + gauss() * 8, 5, 95)),
      };
      const key = cellKey(child.t, child.s);
      const inc = archive[key];
      stats.bred++;
      render({ key, name: 'parent' }, `Parent picked: <b>${parent.obj}%</b> at ${parent.t} turns, KO spread ${parent.s}. Mutating…`);
      await wait(650);
      render({ key, name: 'target', tag: 'gate 1 · WHERE' },
        `<b class="g1">Gate 1 — WHERE.</b> The child measured <b>${child.t} turns, KO spread ${child.s}</b>. ` +
        `It did not choose these; the simulator produced them. They send it to the marked cell.`);
      await wait(900);
      if (!inc) {
        archive[key] = child; stats.admitted++;
        render({ key, name: 'win' }, `<b class="g2">Gate 2 — WHETHER.</b> The cell was <b>empty</b>, so there is nobody to beat: the child is admitted at <b>${child.obj}%</b>. Even a weak child earns a niche this way — this is what keeps diversity alive.`);
      } else if (child.obj > inc.obj) {
        archive[key] = child; stats.admitted++; stats.replaced++;
        render({ key, name: 'win' }, `<b class="g2">Gate 2 — WHETHER.</b> Child <b>${child.obj}%</b> vs incumbent <b>${inc.obj}%</b> — the child wins <em>this cell only</em> and takes it. No other elite was ever compared.`);
      } else {
        render({ key, name: 'lose' }, `<b class="g2">Gate 2 — WHETHER.</b> Child <b>${child.obj}%</b> vs incumbent <b>${inc.obj}%</b> — the child loses and is <b>discarded</b>. It never got to fight any cell except this one.`);
      }
    }
    const wait = ms => new Promise(r => setTimeout(r, ms));

    async function run(n) {
      for (let i = 0; i < n; i++) { await step(); await wait(n === 1 ? 500 : 140); }
    }

    async function guarded(fn, btn) {
      bOne.disabled = bMany.disabled = true;
      try { await fn(); } finally { bOne.disabled = bMany.disabled = false; btn.blur(); }
    }

    bOne.addEventListener('click', () => guarded(() => run(1)));
    bMany.addEventListener('click', () => guarded(() => run(25)));
    bRst.addEventListener('click', () => { reset(); render(null, 'Archive reset. Press <b>Breed a child</b>.'); });

    reset();
    render(null);
    return { get archive() { return archive; }, step: () => run(1) };
  }
  return { mount };
})();
