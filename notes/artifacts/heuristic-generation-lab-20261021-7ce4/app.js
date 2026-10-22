(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const escape = (text) => String(text).replace(/[&<>"']/g, (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
  const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  let motionEnabled = !motionQuery.matches;
  let loopTimer = null;
  let loopStep = 0;
  let isPlaying = false;
  const fade = (element) => {
    if (element.animate) element.animate([{opacity: .55}, {opacity: 1}], {duration: motionEnabled ? 180 : 100, easing: 'ease'});
  };

  function applyMotion() {
    document.body.classList.toggle('motion-off', !motionEnabled);
    document.documentElement.style.scrollBehavior = motionEnabled ? '' : 'auto';
    $('#motion-toggle').setAttribute('aria-pressed', String(motionEnabled));
    $('#motion-toggle').textContent = motionEnabled ? 'Motion on' : 'Motion off';
    $('#loop-play').disabled = !motionEnabled;
    $('#loop-play').title = motionEnabled ? '' : 'Motion is off. Use Next step to inspect the cycle.';
    if (!motionEnabled) pauseLoop();
  }
  $('#motion-toggle').addEventListener('click', () => { motionEnabled = !motionEnabled; applyMotion(); });
  motionQuery.addEventListener('change', () => { motionEnabled = !motionQuery.matches; applyMotion(); });

  const loopSteps = [
    {title: 'Propose one candidate heuristic', text: 'Generate a small decision rule with a precise input and output. For example, choose which team slots to regenerate.', node: 'propose'},
    {title: 'Check the contract before using it', text: 'Does the function return a valid slot mask? Reject invalid outputs. Use isolated execution and time and resource limits before connecting generated code to a search.', node: 'propose'},
    {title: 'Run the inner search with this rule', text: 'Let this heuristic choose edits while the existing procedure refills slots, checks legality, and evaluates teams. Keep the starting inputs and search budget comparable.', node: 'search'},
    {title: 'Measure the team the search produced', text: 'Use fresh development battles to estimate that team’s win rate. A useful-looking rule or a good surrogate prediction is not the outcome you ultimately care about.', node: 'measure'},
    {title: 'Compare across search runs', text: 'Assess repeated searches under the same protocol. A single lucky team does not establish that this heuristic is better. These development results may guide the generator.', node: 'feedback'},
    {title: 'Revise, then test a new candidate', text: 'Use measured feedback to propose another rule. Later, choose on validation runs, freeze the winner, and test it without feeding final results back into selection.', node: 'feedback'}
  ];
  function pauseLoop() {
    clearTimeout(loopTimer);
    loopTimer = null;
    isPlaying = false;
    $('#loop-play').textContent = loopStep === 5 ? 'Replay cycle' : (loopStep === 0 ? '▶ Play one cycle' : '▶ Continue cycle');
    $('#loop-play').setAttribute('aria-pressed', 'false');
  }
  function showStep(index) {
    loopStep = index;
    const step = loopSteps[index];
    $('#step-title').textContent = step.title;
    $('#step-text').textContent = step.text;
    $('#step-counter').textContent = `${index + 1} / 6`;
    $$('.step-track button').forEach((button) => {
      if (Number(button.dataset.step) === index) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    $('.step-progress').style.transform = `translateX(${index * 100}%)`;
    $$('[data-node]').forEach((node) => node.classList.toggle('is-active', node.dataset.node === step.node));
    $('#loop-next').textContent = index === 5 ? 'Start again' : 'Next step';
    fade($('#loop-caption'));
  }
  function scheduleStep() {
    loopTimer = setTimeout(() => {
      if (!isPlaying) return;
      if (loopStep < 5) showStep(loopStep + 1);
      if (loopStep === 5) pauseLoop();
      else scheduleStep();
    }, 2600);
  }
  $('#loop-play').addEventListener('click', () => {
    if (isPlaying) { pauseLoop(); return; }
    if (!motionEnabled) return;
    if (loopStep === 5) showStep(0);
    isPlaying = true;
    $('#loop-play').textContent = 'Ⅱ Pause cycle';
    $('#loop-play').setAttribute('aria-pressed', 'true');
    scheduleStep();
  });
  $('#loop-next').addEventListener('click', () => { pauseLoop(); showStep((loopStep + 1) % 6); pauseLoop(); });
  $('#loop-reset').addEventListener('click', () => { pauseLoop(); showStep(0); pauseLoop(); });
  $$('.step-track button').forEach((button) => button.addEventListener('click', () => { pauseLoop(); showStep(Number(button.dataset.step)); pauseLoop(); }));
  document.addEventListener('visibilitychange', () => { if (document.hidden) pauseLoop(); });

  const methodSymbols = {gp: 'ƒ(x)', grammar: '{ }', llm: 'h++', neural: 'π(a|s)', config: 'θ'};
  function showMethod(id) {
    const method = window.METHODS.find((item) => item.id === id);
    if (!method) return;
    $$('#method-tabs button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.method === id)));
    const link = (label, url, className = '') => `<a class="${className}" href="${escape(url)}" target="_blank" rel="noopener">${escape(label)}</a>`;
    $('#method-panel').innerHTML = `
      <div class="method-tagline">${escape(method.tagline)}</div>
      <div class="method-top"><div><h3>${escape(method.label)}</h3><p>${escape(method.intro)}</p></div><div class="method-symbol" aria-hidden="true">${escape(methodSymbols[id])}</div></div>
      <div class="method-body"><div><h4>What gets generated?</h4><p class="method-representation">${escape(method.representation)}</p><h4>What you provide</h4><ul>${method.provide.map((item) => `<li>${escape(item)}</li>`).join('')}</ul></div><div class="method-example"><h4>Illustrative code shape</h4><pre><code>${escape(method.example.before)}</code></pre><div class="example-divider" aria-hidden="true">↓</div><pre><code>${escape(method.example.after)}</code></pre><p class="small">${escape(method.example.caption)}</p></div></div>
      <div class="method-steps">${method.steps.map((step, index) => `<div class="method-step"><h4>${index + 1}. ${escape(step.title)}</h4><p>${escape(step.text)}</p></div>`).join('')}</div>
      <div class="method-fit"><div><h4>A good fit for</h4><p>${escape(method.bestFor)}</p></div><div><h4>What to watch</h4><p>${escape(method.limitation)}</p></div></div>
      ${method.variants ? `<div class="llm-variants"><h4>Three research approaches you can look at</h4><div class="variant-grid">${method.variants.map((variant) => `<article class="variant"><h4>${escape(variant.name)}</h4><p>${escape(variant.idea)}</p><p class="variant-implementation">${escape(variant.implementation)}</p>${link('Paper', variant.source)}${link('Official code', variant.repo)}</article>`).join('')}</div></div>` : ''}
      <div class="method-links">${link(method.startHere.label, method.startHere.url, 'start-link')}${method.sources.map((source) => link(source.label, source.url)).join('')}</div>`;
    $('#method-status').textContent = `${method.label} details shown.`;
    fade($('#method-panel'));
  }
  $$('#method-tabs button').forEach((button) => button.addEventListener('click', () => showMethod(button.dataset.method)));

  const edits = [
    {id: 'A', name: 'Replace the support slot', c: 6, s: 6, d: 6},
    {id: 'B', name: 'Change a coverage move', c: 10, s: 4, d: 5},
    {id: 'C', name: 'Change a speed-control move', c: 5, s: 10, d: 5}
  ];
  const rules = {
    base: {root: '+', leaves: ['c', 's', 'd'], score: (e) => e.c + e.s + e.d, arithmetic: (e) => `${e.c} + ${e.s} + ${e.d}`, explanation: 'The starting rule adds all three features. The edit with the greatest sum ranks first.'},
    tuned: {root: '+', leaves: ['2 × c', 's', 'd'], score: (e) => 2 * e.c + e.s + e.d, arithmetic: (e) => `2 × ${e.c} + ${e.s} + ${e.d}`, explanation: 'Only a weight changes. Coverage counts twice, so the coverage edit rises to first place. The formula is still a weighted sum.'},
    generated: {root: 'min', leaves: ['c', 's', 'd'], score: (e) => Math.min(e.c, e.s, e.d), arithmetic: (e) => `min(${e.c}, ${e.s}, ${e.d})`, explanation: 'The structure changes to a minimum. The balanced support edit ranks first because its weakest feature is strongest. This is a structural example; other weighted sums could still favor that edit.'}
  };
  $('#ranking-list').innerHTML = edits.map((edit) => `<article class="edit-card" id="edit-${edit.id}"><span class="rank-number"></span><div><h4 class="edit-name">${edit.id}. ${edit.name}</h4><div class="edit-features">Coverage ${edit.c} · Speed ${edit.s} · Defense ${edit.d}</div><div class="edit-calculation"></div></div><div class="edit-score"><strong></strong><span>rule score</span></div></article>`).join('');
  function showRule(id) {
    const rule = rules[id];
    if (!rule) return;
    $$('[data-rule]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.rule === id)));
    $('#rule-explanation').textContent = rule.explanation;
    $('#formula-tree').innerHTML = `<div class="tree-root">${escape(rule.root)}</div><div class="tree-trunk"></div><div class="tree-leaves">${rule.leaves.map((leaf) => `<span class="tree-leaf">${escape(leaf)}</span>`).join('')}</div>`;
    $('#formula-tree').setAttribute('aria-label', `Rule tree: ${rule.root} of ${rule.leaves.join(', ')}`);
    const ranked = [...edits].sort((a, b) => rule.score(b) - rule.score(a));
    ranked.forEach((edit, index) => {
      const card = $(`#edit-${edit.id}`);
      card.style.transform = `translateY(${index * 108}px)`;
      card.classList.toggle('is-first', index === 0);
      card.setAttribute('aria-label', `Rank ${index + 1}: edit ${edit.id}, ${edit.name}, rule score ${rule.score(edit)}`);
      card.querySelector('.rank-number').textContent = index + 1;
      card.querySelector('.edit-score strong').textContent = rule.score(edit);
      card.querySelector('.edit-calculation').textContent = `${rule.arithmetic(edit)} = ${rule.score(edit)}`;
    });
    $('#rank-result').textContent = `First choice: ${ranked[0].id}, ${ranked[0].name.toLowerCase()}. This is a ranking decision, not a measured win-rate improvement.`;
  }
  $$('[data-rule]').forEach((button) => button.addEventListener('click', () => showRule(button.dataset.rule)));

  const referenceScores = [7, 8, 9, 8, 10];
  const samples = [
    {id: 'R1', score: 8, win: 60}, {id: 'R2', score: 7, win: 70},
    {id: 'U', score: 3, win: 90, unusual: true}, {id: 'R4', score: 5, win: 45},
    {id: 'R5', score: 6, win: 50}, {id: 'R6', score: 4, win: 40},
    {id: 'R7', score: 2, win: 55}, {id: 'R8', score: 5, win: 35},
    {id: 'R9', score: 1, win: 30}, {id: 'R10', score: 4, win: 20}
  ];
  let outcomesRevealed = false;
  function showFilter() {
    const threshold = Number($('#threshold').value);
    $('#threshold-output').value = threshold;
    const keptReference = referenceScores.filter((value) => value >= threshold).length;
    const keptSample = samples.filter((sample) => sample.score >= threshold);
    $('#meta-dots').innerHTML = referenceScores.map((score, index) => `<span class="meta-dot ${score < threshold ? 'is-rejected' : ''}" title="K${index + 1}: score ${score}, ${score >= threshold ? 'kept' : 'rejected'}" aria-label="Reference K${index + 1}: score ${score}, ${score >= threshold ? 'kept' : 'rejected'}">K${index + 1}</span>`).join('');
    $('#meta-result').textContent = `${keptReference} / 5 kept`;
    $('#r-value').textContent = `${keptReference * 20}%`;
    $('#q-value').textContent = `${keptSample.length * 10}%`;
    $('#random-sample').innerHTML = samples.map((sample) => {
      const kept = sample.score >= threshold;
      return `<article class="toy-team ${kept ? 'is-kept' : 'is-rejected'} ${sample.unusual ? 'unusual' : ''}"><h4 class="toy-name"><span class="team-glyph" aria-hidden="true"></span>${sample.unusual ? 'U · Unusual' : sample.id}</h4><p class="toy-detail">Conventional score ${sample.score}</p><div class="toy-status">${kept ? '✓ Kept' : '× Rejected'}</div><div class="toy-outcome">${outcomesRevealed ? `Invented win chance<strong>${sample.win}%</strong>` : '<span class="unrevealed">Matchup outcome<br>not revealed</span>'}</div></article>`;
    }).join('');
    $('#reveal-outcomes').setAttribute('aria-pressed', String(outcomesRevealed));
    $('#reveal-outcomes').textContent = outcomesRevealed ? 'Hide invented outcomes' : 'Reveal invented matchup outcomes';
    let verdict;
    if (!outcomesRevealed) {
      verdict = `At threshold ${threshold}, the rule keeps ${keptReference} of 5 reference teams and ${keptSample.length} of 10 random teams. Is that enough to know it keeps the best counter?`;
    } else if (threshold > 3) {
      verdict = `U has the highest invented win chance, 90%, but the filter rejects it. ${keptSample.length ? `The strongest surviving sample has ${Math.max(...keptSample.map((sample) => sample.win))}%.` : 'No random-sample candidates survive.'} Good retention scores alone cannot establish that a filter preserves the best counter.`;
    } else {
      verdict = `U now survives with its invented 90% win chance. The rule also retains ${keptSample.length} of the 10 random teams. This toy sample shows a pruning tradeoff; only a real downstream search comparison can establish battle benefit.`;
    }
    $('#proxy-verdict').textContent = verdict;
  }
  $('#threshold').addEventListener('input', showFilter);
  $('#reveal-outcomes').addEventListener('click', () => { outcomesRevealed = !outcomesRevealed; showFilter(); fade($('#random-sample')); });
  $('#proxy-reset').addEventListener('click', () => { $('#threshold').value = '7'; outcomesRevealed = false; showFilter(); });

  $$('[data-answer]').forEach((button) => button.addEventListener('click', () => {
    $$('[data-answer]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    $('#check-feedback').textContent = button.dataset.answer === 'rule'
      ? 'Yes. The outer search produced a reusable decision rule. The teams are outputs of the inner searches. Its benefit still needs independent battle evaluation.'
      : 'The inner search produces a team. Here the outer search evolves the function that chooses an edit, so its output is a reusable decision rule.';
  }));

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          $$('.chapter-nav a').forEach((anchor) => {
            if (anchor.hash === `#${entry.target.id}`) anchor.setAttribute('aria-current', 'location');
            else anchor.removeAttribute('aria-current');
          });
        }
      }
    }, {rootMargin: '-10% 0px -65% 0px', threshold: 0});
    $$('section.chapter[id]').forEach((section) => observer.observe(section));
    const loopVisibility = new IntersectionObserver((entries) => {
      if (!entries[0].isIntersecting && isPlaying) pauseLoop();
    }, {threshold: 0});
    loopVisibility.observe($('#loops'));
  }
  showStep(0);
  showMethod('gp');
  showRule('base');
  showFilter();
  applyMotion();
})();
