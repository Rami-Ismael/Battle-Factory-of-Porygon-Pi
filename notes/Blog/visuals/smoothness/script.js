/* Recorded-data explorer. Controls select saved results; no live battle model. */
(() => {
  'use strict';
  const data = window.SMOOTHNESS_DATA;
  const $ = id => document.getElementById(id);
  const axisLimit = Math.max(30, Math.ceil(Math.max(...data.teams.flatMap(t => t.edges.flatMap(e => e.ci95.map(v => Math.abs(v * 100))))) / 15) * 15);
  document.querySelectorAll('.delta-axis .grid-line > span').forEach((label,i) => {
    const value = [-axisLimit,-axisLimit/2,0,axisLimit/2,axisLimit][i];
    label.textContent = (value > 0 ? '+' : value < 0 ? '−' : '') + Math.abs(value);
  });
  $('delta-chart').dataset.axisLimit = axisLimit;
  $('data-note').textContent = `Saved in matchup DB · ${data.battles.toLocaleString()} battles`;
  $('storage-note').textContent = `${data.storage?.liveDatabase ? 'Loaded from matchup database' : 'Database export'} · no battles run by these controls`;
  const labels = { random: 'Random legal teams', tournament: 'Frozen top placements' };
  const colors = { random: '#a76328', tournament: '#2362a4' };
  const state = { tab: 'edit', teamId: 'tournament-000', slot: 0, edgeId: null, applied: true, noise: false, metric: 'corrected', benchmark: 'pest_control', step: 1, reference: 'tournament' };
  const rememberedTeams = { random: 'random-000', tournament: 'tournament-000' };
  const signed = (n, digits = 1) => (n > 0.000001 ? '+' : n < -0.000001 ? '−' : '') + Math.abs(n).toFixed(digits);
  const number = (n, digits = 1) => (n < -0.000001 ? '−' : '') + Math.abs(n).toFixed(digits);
  const percentage = n => `${number(n * 100)}%`;
  const scientific = n => {
    if (n === 0) return '0';
    const [mantissa, exponent] = Math.abs(n).toExponential(2).split('e');
    const superscripts = { '-': '⁻', '+': '', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
    return `${n < 0 ? '−' : ''}${mantissa} × 10${[...exponent].map(c => superscripts[c]).join('')}`;
  };
  const team = () => data.teams.find(t => t.id === state.teamId);
  const edits = () => team().edges.filter(e => e.edit && e.edit.slot === state.slot);
  const selectedEdge = () => team().edges.find(e => e.candidate === state.edgeId);
  const escapeHTML = text => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const option = (value, text) => `<option value="${escapeHTML(value)}">${escapeHTML(text)}</option>`;

  document.addEventListener('keydown', () => document.documentElement.dataset.input = 'keyboard', true);
  document.addEventListener('pointerdown', () => document.documentElement.dataset.input = 'pointer', true);
  function announce(text) { $('announcement').textContent = text; }
  function instantly(fn) {
    document.documentElement.classList.add('no-motion');
    fn();
    // Commit the immediate position before allowing the next pointer transition.
    void $('delta-chart').offsetWidth;
    document.documentElement.classList.remove('no-motion');
  }

  function selectTeam(id, announceChange = true) {
    const next = data.teams.find(t => t.id === id);
    if (!next) return;
    state.teamId = id;
    rememberedTeams[next.group] = id;
    const first = next.edges.find(e => e.edit);
    state.slot = first.edit.slot;
    state.edgeId = first.candidate;
    state.applied = true;
    state.noise = false;
    $('cohort').value = next.group;
    const groupTeams = data.teams.filter(t => t.group === next.group);
    $('team').innerHTML = groupTeams.map(t => option(t.id, `${t.label} · ${t.source}`)).join('');
    $('team').value = id;
    $('team-count').textContent = `${groupTeams.length} measured originals · 50 planned`;
    renderRoster();
    renderEditControls();
    instantly(renderEffect);
    if (announceChange) announce(`${next.label}, ${labels[next.group]}. ${next.edges.filter(e => e.edit).length} recorded edits across ${new Set(next.edges.filter(e=>e.edit).map(e=>e.edit.slot)).size} Pokémon.`);
  }

  function renderRoster() {
    $('roster').innerHTML = team().roster.map((p, slot) => {
      const count = team().edges.filter(e => e.edit && e.edit.slot === slot).length;
      const heldItem = p.heldItem ? `holding ${p.heldItem} at battle start` : 'no held item';
      const portrait = p.sprite ? `<img src="${escapeHTML(p.sprite)}" alt="" width="84" height="72">` : `<span class="sprite-fallback" aria-hidden="true">${escapeHTML(p.name.slice(0, 1))}</span>`;
      const itemIcon = p.heldItem && p.itemSprite ? `<img class="held-item-icon" src="${escapeHTML(p.itemSprite)}" alt="" width="24" height="24">` : '';
      return `<button type="button" class="pokemon" data-slot="${slot}" aria-pressed="${slot === state.slot}" aria-label="${escapeHTML(p.name)}, ${escapeHTML(heldItem)}, ${count ? count + ' recorded edit' + (count === 1 ? '' : 's') : 'no recorded edits'}">${count ? '<span class="recorded-dot" aria-hidden="true"></span>' : ''}<span class="pokemon-portrait">${portrait}${itemIcon}</span><span>${escapeHTML(p.name.replaceAll('-', ' '))}</span><span class="pokemon-item">${escapeHTML(p.heldItem || 'No held item')}</span></button>`;
    }).join('');
  }

  $('roster').addEventListener('click', event => {
    const button = event.target.closest('[data-slot]');
    if (!button) return;
    state.slot = Number(button.dataset.slot);
    state.edgeId = edits()[0]?.candidate ?? null;
    state.applied = true;
    state.noise = false;
    // Keep the clicked button and its focus rather than replacing the roster.
    $('roster').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b === button));
    renderEditControls();
    renderEffect();
    announce(edits().length ? `${team().roster[state.slot].name}. ${$('interpretation').textContent}` : 'No recorded edit for this Pokémon. Choose a teammate marked with a blue dot.');
  });

  function renderEditControls() {
    const available = edits();
    $('pokemon-name').textContent = team().roster[state.slot].name.replaceAll('-', ' ');
    $('edit-count').textContent = available.length ? `${available.length} recorded replacement${available.length === 1 ? '' : 's'}` : 'No measured replacement';
    $('edit-controls').hidden = !available.length;
    $('unmeasured').hidden = !!available.length;
    if (!available.length) return;
    const current = selectedEdge() || available[0];
    $('old-move').textContent = current.edit.old_move;
    $('replacement').innerHTML = option('', 'Keep original move') + available.map(e => option(e.candidate, available.some(x => x.edit.old_move !== e.edit.old_move) ? `${e.edit.old_move} → ${e.edit.new_name}` : e.edit.new_name)).join('');
    $('replacement').value = state.applied ? current.candidate : '';
    $('undo').disabled = !state.applied && !state.noise;
  }

  function renderEffect() {
    const current = selectedEdge();
    const present = !!current;
    $('effect-content').hidden = !present;
    $('effect-empty').hidden = present;
    if (!present) {
      $('effect-heading').textContent = 'Effect of this edit';
      $('effect-mode').textContent = 'No measurement';
      $('edit-methods').innerHTML = '<p>This Pokémon has no recorded replacement in the pilot. The two sampled edits for this original team are marked by blue dots above.</p>';
      return;
    }
    const edge = state.noise ? team().edges.find(e => e.kind === 'null') : current;
    const showMeasurement = state.noise || state.applied;
    const delta = showMeasurement ? edge.delta * 100 : 0;
    const ci = edge.ci95.map(v => v * 100);
    const pos = value => (value + axisLimit) / (2 * axisLimit) * 100;
    $('delta-dot').style.transform = `translateX(${pos(delta)}%)`;
    $('delta-interval').style.transform = `translateX(${pos(ci[0])}%) scaleX(${(ci[1] - ci[0]) / (2 * axisLimit)})`;
    $('ci-low').style.transform = `translateX(${pos(ci[0])}%)`;
    $('ci-high').style.transform = `translateX(${pos(ci[1])}%)`;
    ['delta-interval', 'ci-low', 'ci-high'].forEach(id => $(id).style.opacity = showMeasurement ? '1' : '0');
    $('delta-value').innerHTML = `${signed(delta)} <small>pp</small>`;
    $('win-rate').textContent = `${percentage(team().original)} → ${percentage(team().original + (showMeasurement ? edge.delta : 0))}`;
    $('interval-label').textContent = showMeasurement ? `95% interval: ${signed(ci[0])} to ${signed(ci[1])} pp` : 'Original result · no new comparison';
    $('effect-heading').textContent = state.noise ? 'Effect of an unchanged-team repeat' : 'Effect of this edit';
    $('effect-mode').textContent = state.noise ? 'Repeat − original' : state.applied ? (edge.primary_sample === false ? 'Additional member edit' : 'Original sampled edit') : 'Same recorded baseline';
    const intervalReading = ci[1] < 0 ? 'This pointwise interval is entirely below zero.' : ci[0] > 0 ? 'This pointwise interval is entirely above zero.' : 'This pointwise interval includes zero; the direction is uncertain.';
    $('interpretation').textContent = !showMeasurement ? 'You are looking at the original team. Select a recorded replacement to see its measured effect.' : state.noise ? `Same team, new battle panels: ${signed(delta)} pp. The true edit effect is zero; this difference comes from repeat evaluation.` : `${current.edit.old_move} → ${current.edit.new_name}. ${intervalReading}`;
    $('noise-toggle').setAttribute('aria-pressed', state.noise);
    $('noise-toggle').textContent = state.noise ? 'Return to the selected edit' : 'Show an unchanged-team repeat';
    const aria = showMeasurement ? `${state.noise ? 'Unchanged-team repeat' : 'Recorded move replacement'}: change ${signed(delta)} percentage points, 95 percent interval ${signed(ci[0])} to ${signed(ci[1])}. Original win rate ${percentage(team().original)}, ${state.noise ? 'repeat' : 'edited'} win rate ${percentage(team().original + edge.delta)}.` : `Original team, observed win rate ${percentage(team().original)}. No edit selected and no interval displayed.`;
    $('delta-chart').setAttribute('aria-label', aria);
    renderMethods(showMeasurement ? edge : null);
  }

  function renderMethods(edge) {
    const original = team().baselinePanels;
    const total = panels => `${panels.reduce((n, p) => n + p.wins, 0)} / ${panels.reduce((n, p) => n + p.battles, 0)}`;
    $('edit-methods').innerHTML = `<p><strong>Measured pilot, 1 October 2026.</strong> Every candidate was evaluated in three independent panels of 50 battles against the same frozen pool of 50 opponent entries. A fixed behavior-cloned battle policy controls both sides. Ties count as non-wins.</p><table><caption class="sr-only">Observed wins and battle counts</caption><thead><tr><th scope="col">Candidate</th><th scope="col">Panel 1</th><th scope="col">Panel 2</th><th scope="col">Panel 3</th><th scope="col">Total wins / battles</th></tr></thead><tbody><tr><th scope="row">Original</th>${original.map(p => `<td>${p.wins} / ${p.battles}</td>`).join('')}<td>${total(original)}</td></tr>${edge ? `<tr><th scope="row">${state.noise ? 'Unchanged repeat' : 'Edited'}</th>${edge.panels.map(p => `<td>${p.wins} / ${p.battles}</td>`).join('')}<td>${total(edge.panels)}</td></tr>` : ''}</tbody></table><p>The error bar is a pointwise 95% Newcombe/Wilson score interval for the difference between two win proportions, pooled across panels. It is approximate for the fixed opponent strata and is not adjusted for testing multiple edits. An interval that includes zero does not establish that the effect is zero.</p><p><strong>What stays fixed:</strong> the other moves, Pokémon, items, abilities and spreads; the battle policy; and the opponent pool, including its frozen duplicate weights and self matchups. Only the indicated move is replaced. Format: <code>${escapeHTML(data.format)}</code>.</p><p>These observations measure local move sensitivity under this protocol. One large edit effect does not establish mathematical non-smoothness or performance under optimal play.</p>`;
    const samplingNote = edge?.primary_sample === false
      ? 'This extra edit was measured to cover this team member. It reuses the saved original panels and is excluded from the original two-edit cohort and benchmark summaries.'
      : 'This result belongs to the original pilot sample. Additional member edits are available above, while the original cohort and benchmark summaries remain fixed.';
    $('edit-methods').insertAdjacentHTML('afterbegin', `<p><strong>Reused from the matchup database.</strong> ${samplingNote} Refreshing or changing controls reads saved results; it does not run battles.</p>`);
  }

  $('cohort').addEventListener('change', event => selectTeam(rememberedTeams[event.target.value]));
  $('team').addEventListener('change', event => selectTeam(event.target.value));
  $('replacement').addEventListener('change', event => {
    if (event.target.value) state.edgeId = event.target.value;
    state.applied = !!event.target.value;
    state.noise = false;
    $('old-move').textContent = selectedEdge().edit.old_move;
    $('undo').disabled = !state.applied;
    renderEffect();
    announce($('delta-chart').getAttribute('aria-label'));
  });
  $('undo').addEventListener('click', () => {
    state.applied = false;
    state.noise = false;
    $('replacement').value = '';
    $('undo').disabled = true;
    renderEffect();
    $('replacement').focus({ preventScroll: true });
    announce('Edit undone. Showing the original result.');
  });
  $('noise-toggle').addEventListener('click', () => {
    state.noise = !state.noise;
    $('undo').disabled = !state.applied && !state.noise;
    renderEffect();
    announce($('delta-chart').getAttribute('aria-label'));
  });

  function renderGroups() {
    const corrected = state.metric === 'corrected';
    const domain = corrected ? [-10, 100] : [0, 12];
    const ticks = corrected ? [0, 25, 50, 75, 100] : [0, 3, 6, 9, 12];
    const unit = corrected ? 'pp²' : 'pp';
    const value = t => corrected ? t.corrected * 10000 : t.absolute * 100;
    const position = v => (v - domain[0]) / (domain[1] - domain[0]) * 100;
    $('metric-explanation').textContent = corrected ? 'Estimate how much expected win rate changes after accounting for battle noise. Negative estimates are possible and are kept. Both groups use the same scale.' : 'Average the absolute measured changes for each team. This view still contains battle noise; it is not the latent distribution of true effects.';
    $('group-chart').innerHTML = ['random', 'tournament'].map(group => `<div class="group-row"><div class="group-label">${labels[group]}<small>4 originals · 2 edits each</small></div><div class="group-track">${ticks.map(t => `<span class="group-grid ${t === 0 ? 'zero' : ''}" style="--position:${position(t)}%" aria-hidden="true"></span>`).join('')}${data.teams.filter(t => t.group === group).map((t, i) => `<button type="button" class="team-dot" data-team="${t.id}" aria-current="${t.id === state.teamId}" style="--x:${position(value(t))}%;--y:${[14, 38, 62, 86][i]}%;--dot-color:${colors[group]}" aria-label="${t.label}, ${labels[group]}, ${number(value(t), 2)} ${unit}. Open recorded edits."><span class="dot-label">${t.label}</span></button>`).join('')}</div></div>`).join('') + `<div class="group-axis"><span></span><div class="axis-labels">${ticks.map(t => `<span style="--position:${position(t)}%">${t}</span>`).join('')}</div></div><span class="axis-unit">${corrected ? 'Noise-corrected mean squared change' : 'Observed mean absolute change'} (${unit})</span>`;
    $('group-summary').innerHTML = ['random', 'tournament'].map(group => {
      const s = data.summary[group];
      const mean = corrected ? s.mean * 10000 : data.teams.filter(t => t.group === group).reduce((n, t) => n + t.absolute * 100, 0) / s.clusters;
      return `<div class="summary-stat"><span>${labels[group]} · group mean</span><strong style="color:${colors[group]}">${number(mean, 2)} ${unit}</strong><p>${corrected ? `95% cluster interval: ${number(s.ci95[0] * 10000, 2)} to ${number(s.ci95[1] * 10000, 2)} ${unit}` : 'Descriptive mean; no interval computed for this view.'}</p><p>Original mean win rate: ${percentage(s.mean_original_win_rate)}</p></div>`;
    }).join('');
  }
  $('metric').addEventListener('change', e => { state.metric = e.target.value; renderGroups(); announce($('metric-explanation').textContent); });
  $('group-chart').addEventListener('click', e => {
    const button = e.target.closest('[data-team]');
    if (!button) return;
    selectTeam(button.dataset.team);
    changeTab('edit', true);
  });

  function setStepOptions() {
    const continuous = state.benchmark === 'hartmann6';
    const choices = data.benchmarks.filter(b => b.domain === state.benchmark);
    $('benchmark-step').innerHTML = choices.map(b => option(b.step, continuous ? `${b.step} along one coordinate` : `${b.step} stage${b.step === 1 ? '' : 's'} replaced`)).join('');
    state.step = continuous ? .01 : 1;
    $('benchmark-step').value = state.step;
  }
  function renderBenchmarks() {
    const reference = data.summary[state.reference];
    const benchmark = data.benchmarks.find(b => b.domain === state.benchmark && b.step === state.step);
    const continuous = state.benchmark === 'hartmann6';
    const name = continuous ? 'Hartmann-6' : 'Pest Control';
    const width = Math.max(280, Math.round($('benchmark-chart').clientWidth));
    const compact = width < 620;
    const symlog = v => Math.sign(v) * Math.log10(1 + Math.abs(v) / 1e-9);
    const lo = symlog(-.001), hi = symlog(.01);
    const start = compact ? 22 : 130, end = width - 26;
    const x = v => start + (symlog(v) - lo) / (hi - lo) * (end - start);
    const ticks = compact ? [-.001, 0, .00001, .01] : [-.001, -.00001, -1e-7, 0, 1e-7, .00001, .001, .01];
    const tickLabel = compact ? ['−10⁻³', '0', '10⁻⁵', '10⁻²'] : ['−10⁻³', '−10⁻⁵', '−10⁻⁷', '0', '10⁻⁷', '10⁻⁵', '10⁻³', '10⁻²'];
    const chart = $('benchmark-chart');
    const build = chart.dataset.width !== String(width);
    if (build) {
      const row = (id, y, color) => `<rect id="${id}-span" class="bench-moving" x="0" y="${y-1}" width="1" height="2" fill="${color}"/><path id="${id}-low" class="bench-moving" d="M0,${y-6}v12" stroke="${color}"/><path id="${id}-high" class="bench-moving" d="M0,${y-6}v12" stroke="${color}"/><circle id="${id}-point" class="bench-moving" cx="0" cy="${y}" r="5.5" fill="${color}" stroke="white" stroke-width="1.5"/>`;
      chart.innerHTML = `<svg viewBox="0 0 ${width} ${compact ? 225 : 210}" role="img" aria-labelledby="bench-title bench-desc"><title id="bench-title"></title><desc id="bench-desc"></desc>${ticks.map((v,i) => `<line x1="${x(v)}" x2="${x(v)}" y1="${compact ? 28 : 20}" y2="145" stroke="${v === 0 ? '#9baaba' : '#e6ecf1'}" ${v === 0 ? 'stroke-dasharray="4 4"' : ''}/><text x="${x(v)}" y="166" font-size="11" fill="#687789" text-anchor="middle">${tickLabel[i]}</text>`).join('')}<text x="0" y="${compact ? 16 : 59}" font-size="13" fill="#2362a4">Pokémon</text><text id="bench-domain-label" x="0" y="${compact ? 94 : 116}" font-size="13" fill="#35766c"></text>${row('bench-reference',55,'#2362a4')}${row('bench-domain',112,'#35766c')}${compact ? `<text x="0" y="194" font-size="11" fill="#687789">Symmetric log scale · linear near zero</text><text x="0" y="213" font-size="10" fill="#687789">Scale parameter 10⁻⁹ · fixed across all comparisons</text>` : `<text x="${end}" y="196" font-size="12" fill="#687789" text-anchor="end">Symmetric log scale · linear near zero (scale parameter 10⁻⁹)</text>`}</svg>`;
      chart.dataset.width = width;
    }
    $('bench-title').textContent = `Normalized local sensitivity: Pokémon and ${name}`;
    $('bench-desc').textContent = `Symmetric logarithmic scale, linear near zero. ${labels[state.reference]}: mean ${scientific(reference.mean)}, interval ${scientific(reference.ci95[0])} to ${scientific(reference.ci95[1])}. ${name}, step ${state.step}: mean ${scientific(benchmark.summary.mean)}, interval ${scientific(benchmark.summary.ci95[0])} to ${scientific(benchmark.summary.ci95[1])}.`;
    $('bench-domain-label').textContent = name;
    const positionRows = () => {
      for (const [id, summary] of [['bench-reference', reference], ['bench-domain', benchmark.summary]]) {
        $(id + '-span').style.transform = `translateX(${x(summary.ci95[0])}px) scaleX(${x(summary.ci95[1]) - x(summary.ci95[0])})`;
        $(id + '-low').style.transform = `translateX(${x(summary.ci95[0])}px)`;
        $(id + '-high').style.transform = `translateX(${x(summary.ci95[1])}px)`;
        $(id + '-point').style.transform = `translateX(${x(summary.mean)}px)`;
      }
    };
    if (build) instantly(positionRows); else positionRows();
    $('benchmark-definitions').innerHTML = `<div><h2>Pokémon · one move replacement</h2><p>${labels[state.reference]}. The original team, battle policy and opponent pool stay fixed.</p><span class="benchmark-value">${scientific(reference.mean)}</span><p>95% interval: ${scientific(reference.ci95[0])} to ${scientific(reference.ci95[1])}<br>Objective bound: 1 win probability</p></div><div><h2>${name} · ${continuous ? 'one coordinate' : `${state.step} categorical stage${state.step === 1 ? '' : 's'}`}</h2><p>${continuous ? `Signed step of ${state.step} in one of six coordinates. Reference points and direction choices stay fixed across step sizes.` : `Replace ${state.step} of 25 stage decisions. Five choices per stage; independent stochastic evaluations.`}</p><span class="benchmark-value benchmark-domain-value">${scientific(benchmark.summary.mean)}</span><p>95% interval: ${scientific(benchmark.summary.ci95[0])} to ${scientific(benchmark.summary.ci95[1])}<br>Objective bound: ${benchmark.fixed_scale}</p></div>`;
  }
  new ResizeObserver(() => {
    if (state.tab === 'benchmarks') renderBenchmarks();
  }).observe($('benchmark-chart'));
  $('benchmark').addEventListener('change', e => { state.benchmark = e.target.value; setStepOptions(); renderBenchmarks(); });
  $('benchmark-step').addEventListener('change', e => { state.step = Number(e.target.value); renderBenchmarks(); announce(`${state.benchmark === 'hartmann6' ? 'Hartmann-6' : 'Pest Control'} step ${state.step}. Pokémon reference unchanged.`); });
  $('benchmark-cohort').addEventListener('change', e => { state.reference = e.target.value; renderBenchmarks(); });

  function changeTab(name, focus = false, updateHash = true) {
    if (!['edit', 'compare', 'benchmarks'].includes(name)) name = 'edit';
    state.tab = name;
    document.querySelectorAll('[data-tab]').forEach(button => {
      const active = button.dataset.tab === name;
      button.setAttribute('aria-selected', active);
      button.tabIndex = active ? 0 : -1;
      $(`panel-${button.dataset.tab}`).hidden = !active;
    });
    if (name === 'compare') renderGroups();
    if (name === 'benchmarks') renderBenchmarks();
    if (focus) $(`tab-${name}`).focus({ preventScroll: true });
    if (updateHash) history.replaceState(null, '', `#${name}`);
  }
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.addEventListener('click', () => changeTab(button.dataset.tab));
    button.addEventListener('keydown', event => {
      const tabs = [...document.querySelectorAll('[data-tab]')];
      let index = tabs.indexOf(button);
      if (event.key === 'ArrowRight') index = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') index = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = tabs.length - 1;
      else return;
      event.preventDefault();
      changeTab(tabs[index].dataset.tab, true);
    });
  });
  window.addEventListener('hashchange', () => changeTab(location.hash.slice(1), false, false));
  selectTeam(state.teamId, false);
  setStepOptions();
  changeTab(location.hash.slice(1), false, false);
})();
