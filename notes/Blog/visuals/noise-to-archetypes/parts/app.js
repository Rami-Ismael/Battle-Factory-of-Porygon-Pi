/* Four chapters, two views: scroll takes the turns. Prefix sc-. */
window.figInit = function (ctx) {
  const root = document.getElementById('fig-root');
  const TOTAL = 48;
  const FUT_STEP = 41;
  const ARCH = ctx.NAMES;
  const SLOT_LABELS = ['Species', 'Ability', 'Item', 'Move', 'Move', 'Move', 'Move', 'Nature'];
  const state = { chapter: 0, progress: 0, step: 0, openFuture: null, predictions: [null, null, null, null] };
  let suppress = false;

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const r1 = (v) => Math.round(v * 10) / 10;

  const chapters = [...root.querySelectorAll('.sc-chapter')].map((el) => ({
    el, n: +el.dataset.chapter, start: +el.dataset.start, end: +el.dataset.end,
  }));
  const chapterN = (n) => chapters[n - 1];
  const chapterOf = (s) => {
    if (s === 0) return 0;
    const c = chapters.find((x) => s >= x.start && s <= x.end);
    return c ? c.n : 4;
  };

  // MB607 rows: use the steps array when the data has it, else rebuild the row from its futures.
  const futuresAt = (s) => (s === 0 ? ctx.D.mb607.futures0 || [] : ctx.futures(Math.min(s, FUT_STEP)) || []);
  const deriveRow = (fs) => {
    const n = fs.length, cnt = [0, 0, 0, 0];
    let mx = 0, my = 0;
    fs.forEach((f) => { mx += f.x; my += f.y; cnt[f.lab] += 1; });
    mx /= n; my /= n;
    let vx = 0, vy = 0, cv = 0;
    fs.forEach((f) => { const dx = f.x - mx, dy = f.y - my; vx += dx * dx; vy += dy * dy; cv += dx * dy; });
    return [mx, my, vx / n, vy / n, cv / n, ...cnt];
  };
  const mbRow = (s) => {
    const sm = Math.min(s, FUT_STEP);
    const steps = ctx.D.mb607.steps;
    if (steps && steps[sm]) return steps[sm];
    return deriveRow(futuresAt(sm));
  };

  // The 24 buttons show the step-41 futures (the model redraws its 24 futures at every step).
  const btnFutures = ctx.futures(FUT_STEP);
  const futBox = $('sc-futures');
  futBox.innerHTML = btnFutures.map((f, i) =>
    `<button type="button" class="sc-fut-btn" data-future="${i}" aria-pressed="false" aria-label="Future ${i + 1} of ${btnFutures.length}, ends as ${ARCH[f.lab]}">${ctx.six(f.sp, 32)}<span class="sc-fut-n">${i + 1}</span></button>`
  ).join('');

  // Caption: the archetype at step 41, and the first step from which Trick Room stays at 0.
  const trStop = (() => {
    let first = null;
    for (let s = FUT_STEP; s >= 1; s--) {
      if (ctx.counts(mbRow(s))[0] !== 0) break;
      first = s;
    }
    return first;
  })();
  $('sc-cap-end').textContent = ARCH[ctx.majority(mbRow(FUT_STEP))];
  $('sc-cap-step').textContent = trStop === null ? 'never' : String(trStop);

  const grid = () => {
    let g = '<g class="sc-grid">';
    for (let i = 1; i < 5; i++) {
      const v = i * 104;
      g += `<line x1="${v}" y1="0" x2="${v}" y2="520"/><line x1="0" y1="${v}" x2="520" y2="${v}"/>`;
    }
    return g + '</g>';
  };

  const noiseSvg = (s) => {
    let out = grid();
    ctx.D.teams.forEach((t, i) => {
      const [x, y] = ctx.noise(i, t.steps[s]);
      out += `<circle class="sc-dot sc-a${ctx.majority(t.steps[s])}" cx="${r1(ctx.X(x))}" cy="${r1(ctx.Y(y))}" r="4"/>`;
    });
    ctx.D.centroids.forEach((c, k) => {
      out += `<text class="sc-cent-label" x="${r1(ctx.X(c[0]))}" y="${r1(ctx.Y(c[1]) - 10)}">${esc(ARCH[k])}</text>`;
    });
    return out;
  };

  const mbSvg = (s, row, fs) => {
    const e = ctx.ellipse(row[0], row[1], row[2], row[3], row[4]);
    let out = grid();
    out += `<ellipse class="sc-ell-mb" cx="${r1(e.cx)}" cy="${r1(e.cy)}" rx="${r1(e.rx)}" ry="${r1(e.ry)}" transform="rotate(${r1(e.rotate)} ${r1(e.cx)} ${r1(e.cy)})"/>`;
    const rad = s === 0 ? 3 : 6;
    fs.forEach((f, i) => {
      const open = s > 0 && i === state.openFuture ? ' sc-fut-open' : '';
      out += `<circle class="sc-fut sc-a${f.lab}${open}" cx="${r1(ctx.X(f.x))}" cy="${r1(ctx.Y(f.y))}" r="${rad}"/>`;
    });
    const mx = r1(ctx.X(row[0])), my = r1(ctx.Y(row[1]));
    out += `<circle class="sc-mb-ring" cx="${mx}" cy="${my}" r="10"/><circle class="sc-mb-dot" cx="${mx}" cy="${my}" r="4"/>`;
    return out;
  };

  const countsText = (s, row) => {
    const c = ctx.counts(row), total = s === 0 ? 512 : 24;
    const tally = ARCH.map((a, i) => `${a} ${c[i]} of ${total}`).join(' · ');
    return s > FUT_STEP ? `Settled at step ${FUT_STEP}: ${tally}.` : `${tally}.`;
  };

  const renderSheet = () => {
    const sheet = $('sc-sheet');
    if (state.openFuture === null) { sheet.hidden = true; return; }
    const f = btnFutures[state.openFuture];
    sheet.hidden = false;
    $('sc-sheet-title').textContent = `Future ${state.openFuture + 1} of ${btnFutures.length}: ends as ${ARCH[f.lab]}`;
    $('sc-sheet-six').innerHTML = ctx.six(f.sp, 40);
    const fields = f.f || [];
    $('sc-sheet-fields').innerHTML = fields.length
      ? fields.map((v, j) => `<li>${SLOT_LABELS[j % 8]}: ${esc(v)}</li>`).join('')
      : '<li>This future has no field list in the data.</li>';
  };

  const renderPredictions = () => {
    const s = state.step;
    root.querySelectorAll('.sc-pbtn').forEach((b) => {
      const k = +b.dataset.chapter, a = +b.dataset.arch;
      b.setAttribute('aria-pressed', String(state.predictions[k - 1] === a));
      b.setAttribute('aria-disabled', String(state.chapter !== k));
    });
    let right = 0;
    chapters.forEach((c) => {
      const el = $('sc-reveal-' + c.n);
      const p = state.predictions[c.n - 1];
      const rrow = mbRow(Math.min(c.end, FUT_STEP));
      const maj = ctx.majority(rrow);
      const real = ctx.counts(rrow)[maj];
      el.classList.remove('is-right', 'is-wrong');
      if (s < c.end) {
        el.textContent = p === null ? '' : `You predicted ${ARCH[p]}. The verdict comes at step ${c.end}.`;
      } else if (p === null) {
        el.textContent = `No prediction. ${ARCH[maj]} held ${real} of 24 futures.`;
      } else if (p === maj) {
        right += 1;
        el.classList.add('is-right');
        el.textContent = `Right. You said ${ARCH[p]}, and ${ARCH[maj]} held ${real} of 24 futures.`;
      } else {
        el.classList.add('is-wrong');
        el.textContent = `Wrong. You said ${ARCH[p]}; ${ARCH[maj]} held ${real} of 24 futures.`;
      }
    });
    $('sc-tally').textContent = `Right on ${right} of 4.`;
  };

  const render = () => {
    const s = state.step;
    const sm = Math.min(s, FUT_STEP);
    state.chapter = chapterOf(s);
    const cur = state.chapter ? chapterN(state.chapter) : null;
    state.progress = cur ? clamp((s - cur.start) / (cur.end - cur.start), 0, 1) : 0;
    const row = mbRow(sm);
    $('sc-step').value = s;
    $('sc-step-out').textContent = `Step ${s} of ${TOTAL}`;
    root.querySelectorAll('.sc-chap').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.chapter === state.chapter)));
    $('sc-noise').innerHTML = noiseSvg(s);
    $('sc-mb').innerHTML = mbSvg(s, row, futuresAt(sm));
    $('sc-counts').textContent = countsText(s, row);
    root.querySelectorAll('.sc-fut-btn').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.future === state.openFuture)));
    renderSheet();
    renderPredictions();
    ctx.show(state);
  };

  const setStep = (s) => { state.step = s; render(); };

  // Programmatic page scrolls set `suppress` so the scroll handler does not echo them back as new steps.
  const scrollToStep = (s) => {
    const mid = window.innerHeight * 0.5;
    const c = s === 0 ? chapters[0] : chapters.find((x) => s >= x.start && s <= x.end);
    const r = c.el.getBoundingClientRect();
    const targetTop = s === 0 ? mid + 2 : mid - ((s - c.start) / (c.end - c.start)) * r.height;
    const delta = r.top - targetTop;
    if (Math.abs(delta) > 1) {
      suppress = true;
      window.scrollTo(0, window.scrollY + delta);
    }
  };

  const onScroll = () => {
    if (suppress) { suppress = false; return; }
    const mid = window.innerHeight * 0.5;
    if (chapters[0].el.getBoundingClientRect().top > mid) {
      if (state.step !== 0) setStep(0);
      return;
    }
    let c = chapters[0];
    chapters.forEach((x) => { if (x.el.getBoundingClientRect().top <= mid) c = x; });
    const r = c.el.getBoundingClientRect();
    const p = clamp((mid - r.top) / r.height, 0, 1);
    const s = c.start + Math.round(p * (c.end - c.start));
    if (s !== state.step) setStep(s);
  };

  root.querySelectorAll('.sc-chap').forEach((b) => b.addEventListener('click', () => {
    const c = chapterN(+b.dataset.chapter);
    scrollToStep(c.start);
    setStep(c.start);
  }));

  root.querySelectorAll('.sc-pbtn').forEach((b) => b.addEventListener('click', () => {
    const k = +b.dataset.chapter;
    if (state.chapter !== k) return;
    state.predictions[k - 1] = +b.dataset.arch;
    render();
  }));

  $('sc-step').addEventListener('input', (e) => {
    const s = +e.target.value;
    scrollToStep(s);
    setStep(s);
  });

  futBox.addEventListener('click', (e) => {
    const b = e.target.closest('.sc-fut-btn');
    if (!b) return;
    const i = +b.dataset.future;
    state.openFuture = state.openFuture === i ? null : i;
    render();
  });

  $('sc-sheet-close').addEventListener('click', () => { state.openFuture = null; render(); });

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);

  render();
  onScroll();
};
