const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../artifacts/win-rate-frequency.html'), 'utf8');
const refreshSource = html.slice(html.indexOf("let lastUpdate="), html.indexOf('refresh();setInterval'));
function fixture() {
  const elements = {};
  const initial = { saved: true };
  let response = { ok: false, status: 404 };
  const context = vm.createContext({
    DATA: initial, render() {}, fmt: String, Date,
    $: id => elements[id] ||= { hidden: true, textContent: '' },
    fetch: async () => response,
  });
  vm.runInContext(refreshSource, context);
  return {
    context, initial, elements,
    respond(value) { response = value; },
    refresh: () => vm.runInContext('refresh()', context),
  };
}
function snapshot() {
  const counts = Array(101).fill(0); counts[50] = 1;
  return {
    updated_at: '2026-10-14T12:00:00Z', status: 'running',
    added_labels: 1, target_labels: 2, unique_battles: 49, detail: 'In progress.',
    dataset: 'results.npz', execution: { workers: 3, concurrency: 12 },
    data: Object.fromEntries(['all','model','real','fill','hps'].map(key =>
      [key, { counts, labeled: 1, unlabeled: 1, rows: 2 }])),
  };
}
test('initial HTTP failure identifies the embedded saved snapshot', async () => {
  const f = fixture(); await f.refresh();
  assert.equal(f.context.DATA, f.initial);
  assert.equal(f.elements['job-status'].hidden, false);
  assert.match(f.elements['job-status'].textContent, /Live results unavailable.*saved VGCPastes snapshot/);
});
test('HTTP failure retains live counts and recovery clears stale warning without a new timestamp', async () => {
  const f = fixture(), live = snapshot();
  f.respond({ ok: true, json: async () => live }); await f.refresh();
  assert.equal(f.context.DATA, live.data);
  assert.match(f.elements.provenance.textContent, /3 workers, concurrency 12/);
  f.respond({ ok: false, status: 500 }); await f.refresh();
  assert.equal(f.context.DATA, live.data);
  assert.match(f.elements['job-status'].textContent, /Could not refresh.*saved results/);
  f.respond({ ok: true, json: async () => live }); await f.refresh();
  assert.match(f.elements['job-status'].textContent, /Generating labels/);
  assert.doesNotMatch(f.elements['job-status'].textContent, /Could not refresh/);
});
