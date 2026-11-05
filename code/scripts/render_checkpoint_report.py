#!/usr/bin/env python3
"""Render the recorded checkpoint comparison as a self-contained HTML figure."""
import argparse
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask-versus-got error by training epoch</title>
<style>
:root { color-scheme: light dark; --cp-blue: var(--chart-1, #6ca8fa); --cp-green: var(--success, #58b994); }
* { box-sizing: border-box; }
html, body { margin: 0; background: transparent; }
body { color: var(--foreground, #dce3ef); font: 14px/1.5 var(--font-sans, system-ui, sans-serif); }
.cp-report { width: 100%; }
.cp-lead { margin: 0 0 5px; font-size: 20px; line-height: 1.3; font-weight: 650; }
.cp-note { margin: 5px 0 12px; color: var(--muted-foreground, #9da8b8); font-size: 13px; }
.cp-chart { display: block; width: 100%; height: 260px; overflow: visible; }
.cp-chart text { fill: var(--muted-foreground, #9da8b8); font-family: inherit; font-size: 11px; }
.cp-chart .cp-grid { stroke: var(--border, #394353); stroke-width: 1; }
.cp-chart .cp-axis-title { fill: var(--foreground, #dce3ef); font-size: 12px; }
.cp-rule { border: 0; border-top: 1px solid var(--border, #394353); margin: 16px 0; }
.cp-inspect { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
.cp-inspect label { font-weight: 600; }
.cp-inspect select { color: var(--foreground, #dce3ef); background: var(--secondary, #222d3b); border: 1px solid var(--border, #536173); border-radius: var(--radius, 6px); font: inherit; padding: 6px 30px 6px 9px; min-height: 36px; }
.cp-inspect select:focus-visible, summary:focus-visible { outline: 2px solid var(--ring, #7fb4ff); outline-offset: 3px; }
.cp-stats { display: flex; flex-wrap: wrap; gap: 5px 24px; margin: 10px 0 2px; }
.cp-stats span { color: var(--muted-foreground, #9da8b8); }
.cp-stats strong { color: var(--foreground, #dce3ef); font-weight: 650; font-variant-numeric: tabular-nums; }
.cp-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12px; margin: 3px 0; }
.cp-key { display: inline-flex; align-items: center; gap: 6px; }
.cp-swatch { width: 20px; border-top: 2px solid currentColor; }
.cp-blue { color: var(--cp-blue); } .cp-green { color: var(--cp-green); }
.cp-gray { color: var(--muted-foreground, #9da8b8); } .cp-dashed { border-top-style: dashed; }
.cp-table-wrap { overflow-x: auto; margin-top: 10px; }
.cp-table { border-collapse: collapse; font-size: 12px; width: 100%; font-variant-numeric: tabular-nums; }
.cp-table th, .cp-table td { padding: 7px 7px; border-bottom: 1px solid var(--border, #394353); text-align: right; white-space: nowrap; }
.cp-table th:first-child, .cp-table td:first-child { text-align: left; }
.cp-table th { font-weight: 600; color: var(--muted-foreground, #9da8b8); }
.cp-table tr[aria-current="true"] { background: var(--accent-surface, #23334a); }
.cp-more { margin-top: 14px; }
.cp-more summary { cursor: pointer; width: fit-content; padding: 3px 0; }
.cp-more p { margin: 8px 0; }
.cp-link { color: var(--cp-blue); }
@media (prefers-color-scheme: light) { body { color: var(--foreground, #202b3b); } .cp-chart .cp-axis-title { fill: var(--foreground, #202b3b); } .cp-stats strong { color: var(--foreground, #202b3b); } }
</style></head><body><main class="cp-report">
<p class="cp-lead">Error falls early, then flattens by epoch 34.</p>
<p class="cp-note">Recorded checkpoints from one training run. The same sampling seed is used at every epoch. Lower error means achieved win rates are closer to requests.</p>
<svg id="cp-trend" class="cp-chart" role="img" aria-labelledby="cp-trend-title cp-trend-desc"><title id="cp-trend-title">Ask-versus-got error by training epoch</title><desc id="cp-trend-desc">Error falls from 0.0175 at epoch 1 to about 0.006 at epoch 34, then stays near that level through epoch 80. All checkpoints remain above the near-line threshold of 0.001.</desc></svg>
<p class="cp-note" id="cp-trend-caption">Score covers requests from 1/24 to 17/24. Dots are measured checkpoints; connecting lines guide the eye. Thin bars show 95% intervals from resampling whole teams and shared opponent columns. They exclude variation between training runs. The dashed reference is the near-line threshold, error under 0.001.</p>
<hr class="cp-rule">
<div class="cp-inspect"><label for="cp-epoch">Inspect a checkpoint</label><select id="cp-epoch">__OPTIONS__</select></div>
<p class="cp-note">Change the epoch to see where achieved win rates still miss the requests. The epoch 1 curve stays visible for comparison.</p>
<div class="cp-stats" id="cp-stats" aria-live="polite"></div>
<div class="cp-legend"><span class="cp-key cp-blue"><i class="cp-swatch"></i><span id="cp-selected-label">Epoch 34</span></span><span class="cp-key cp-gray"><i class="cp-swatch cp-dashed"></i>Epoch 1</span><span class="cp-key cp-green"><i class="cp-swatch"></i>Got = ask</span></div>
<svg id="cp-agreement" class="cp-chart" role="img" aria-labelledby="cp-agreement-title cp-agreement-desc"><title id="cp-agreement-title">Requested versus achieved win rate</title><desc id="cp-agreement-desc">Each point is the mean achieved win rate among 64 teams generated at that request. The diagonal shows exact agreement.</desc></svg>
<p class="cp-note">Dashed curve segment and shaded band cover requests 18/24 to 23/24, where training examples are scarce. Those requests are scored separately.</p>
<details class="cp-more"><summary>All checkpoints, diversity and scoring details</summary>
<div class="cp-table-wrap"><table class="cp-table"><thead><tr><th>Epoch</th><th>Error<br>1/24–17/24</th><th>Error<br>18/24–23/24</th><th>Distinct<br>line-ups</th><th>New<br>line-ups</th><th>Full<br>66 points</th><th>Team spread<br>pp</th></tr></thead><tbody id="cp-table-body">__ROWS__</tbody></table></div>
<p class="cp-note">A line-up is a team's six species, independent of slot order. Counts are unique line-ups across all 1,472 generated teams. New line-ups are absent from the fixed reference described below. Full 66 points is the share of Pokémon spending all available Stat Points. Team spread is the average within-request standard deviation of team win rates, in percentage points.</p>
<p class="cp-note" id="cp-reference"></p>
<p class="cp-note" id="cp-uncertainty"></p>
<p class="cp-note">Every checkpoint uses training seed 0 and sampling seed 1, with 64 teams at each of 23 requests. Each team plays the 49 distinct top-50 opponents under the behaviour-cloning policy on both sides, with shuffled slots. The opponent represented twice in the top-50 list has weight 2 in the win-rate mean.</p>
<p class="cp-note">The model generates all 84 fields. Stat Alignment is repaired to match the moves. Teams come from the labelled-only model trained with label noise ±1/48; these are checkpoint samples without a search or elite-selection rule.</p>
<p class="cp-note" id="cp-repeats"></p>
<p class="cp-note">This exploratory comparison comes from one training run. Sampling and opponent resampling do not measure variation between independent training runs. Epoch 80 is the training cap and the lowest validation-loss checkpoint; these observations do not establish the best epoch beyond that cap.</p>
<p class="cp-note" id="cp-audit"></p>
<details><summary>Recompute from the saved observations</summary><pre id="cp-reproduce" style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px"></pre></details>
<p class="cp-note"><a class="cp-link" href="https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/105" target="_blank" rel="noreferrer">Training checkpoints decision ticket</a></p>
</details>
</main><script id="cp-data" type="application/json">__DATA__</script><script>
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('cp-data').textContent);
  const epochs = data.epochs;
  const select = document.getElementById('cp-epoch');
  const trend = document.getElementById('cp-trend');
  const agreement = document.getElementById('cp-agreement');
  const f4 = v => Number(v).toFixed(4);
  const pct = v => (100 * Number(v)).toFixed(1) + '%';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const line = (x1,y1,x2,y2,attrs='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
  const text = (x,y,s,attrs='') => `<text x="${x}" y="${y}" ${attrs}>${esc(s)}</text>`;
  const path = (rows,x,y,key,attrs='') => `<path d="${rows.map((r,i)=>(i?'L':'M')+x(r)+','+y(r[key])).join(' ')}" fill="none" ${attrs}/>`;
  function layout(svg) {
    const width = Math.max(280, Math.round(svg.getBoundingClientRect().width));
    svg.setAttribute('viewBox', `0 0 ${width} 260`);
    return {w:width,h:260,l:57,r:20,t:22,b:43};
  }
  function drawTrend(selected) {
    const p = layout(trend);
    const top = Math.max(.02, ...epochs.map(e => e.mse_ci ? e.mse_ci[1] * 1.1 : e.mse * 1.1));
    const ymax = Math.ceil(top / .005) * .005;
    const x = e => p.l + (e.epoch - 1) / 79 * (p.w-p.l-p.r);
    const y = v => p.h-p.b-v/ymax*(p.h-p.t-p.b);
    let out = '<title id="cp-trend-title">Ask-versus-got error by training epoch</title><desc id="cp-trend-desc">Measured error on requests 1/24 through 17/24, using sampling seed 1 at all ten checkpoints. Lower is better.</desc>';
    for(let i=0;i<=Math.round(ymax/.005);i++) { const v=i*.005; out += line(p.l,y(v),p.w-p.r,y(v),'class="cp-grid"') + text(p.l-8,y(v)+4,v.toFixed(3),'text-anchor="end"'); }
    for(const e of [1,21,34,55,80]) out += text(x({epoch:e}),p.h-p.b+19,e,'text-anchor="middle"');
    out += text(p.l,13,'Ask-versus-got error','class="cp-axis-title"');
    out += text((p.l+p.w-p.r)/2,254,'Training epoch','text-anchor="middle" class="cp-axis-title"');
    out += line(p.l,y(.001),p.w-p.r,y(.001),'stroke="var(--cp-green)" stroke-dasharray="5 4"') + text(p.w-p.r,y(.001)-6,'Near line < 0.001','text-anchor="end" style="fill:var(--cp-green)"');
    for(const e of epochs) if(e.mse_ci) {
      out += line(x(e),y(e.mse_ci[0]),x(e),y(e.mse_ci[1]),'stroke="var(--cp-blue)" opacity=".35"');
    }
    out += path(epochs,x,y,'mse','stroke="var(--cp-blue)" stroke-width="2.2"');
    for(const e of epochs) out += `<circle cx="${x(e)}" cy="${y(e.mse)}" r="${e.epoch===selected.epoch?5.5:3.5}" fill="var(--cp-blue)" stroke="var(--foreground, #dce3ef)" stroke-width="${e.epoch===selected.epoch?1.5:0}"><title>Epoch ${e.epoch}: ${e.mse.toFixed(6)}</title></circle>`;
    trend.innerHTML=out;
  }
  function drawAgreement(selected) {
    const p = layout(agreement);
    const x = r => p.l + r.ask*(p.w-p.l-p.r);
    const y = v => p.h-p.b-v*(p.h-p.t-p.b);
    let out='<title id="cp-agreement-title">Requested versus achieved win rate at epoch '+selected.epoch+'</title><desc id="cp-agreement-desc">Mean achieved win rates at 23 measured requests, with epoch 1 retained as a baseline and got equals ask as a diagonal reference.</desc>';
    out += `<rect x="${x({ask:17.5/24})}" y="${p.t}" width="${p.w-p.r-x({ask:17.5/24})}" height="${p.h-p.t-p.b}" fill="var(--warning, #c99547)" opacity=".08"/>`;
    for(const v of [0,.25,.5,.75,1]) { out += line(p.l,y(v),p.w-p.r,y(v),'class="cp-grid"')+text(p.l-8,y(v)+4,Math.round(v*100)+'%','text-anchor="end"')+text(x({ask:v}),p.h-p.b+19,Math.round(v*100)+'%','text-anchor="middle"'); }
    out += text(p.l,13,'Achieved win rate','class="cp-axis-title"')+text((p.l+p.w-p.r)/2,254,'Requested win rate','text-anchor="middle" class="cp-axis-title"');
    out += line(x({ask:0}),y(0),x({ask:1}),y(1),'stroke="var(--cp-green)" stroke-width="1.5" opacity=".8"');
    out += path(epochs[0].asks,x,y,'win','stroke="var(--muted-foreground, #9da8b8)" stroke-dasharray="4 4" stroke-width="1.5"');
    const scored=selected.asks.filter(a=>a.numerator<=17), upper=selected.asks.filter(a=>a.numerator>=17);
    out += path(scored,x,y,'win','stroke="var(--cp-blue)" stroke-width="2.3"') + path(upper,x,y,'win','stroke="var(--cp-blue)" stroke-width="2.3" stroke-dasharray="5 4"');
    for(const r of selected.asks) {
      if(Number.isFinite(r.lo)&&Number.isFinite(r.hi)) out += line(x(r),y(Math.max(0,r.lo)),x(r),y(Math.min(1,r.hi)),'stroke="var(--cp-blue)" opacity=".3"');
      out += `<circle cx="${x(r)}" cy="${y(r.win)}" r="2.7" fill="var(--cp-blue)"><title>Ask ${r.numerator}/24: got ${pct(r.win)}, team spread ${(100*r.sd).toFixed(1)} percentage points</title></circle>`;
    }
    agreement.innerHTML=out;
  }
  function render() {
    const selected=epochs.find(e=>e.epoch===Number(select.value))||epochs[0];
    document.getElementById('cp-selected-label').textContent='Epoch '+selected.epoch;
    document.getElementById('cp-stats').innerHTML=`<span>Error <strong>${f4(selected.mse)}</strong></span><span>Upper-request error <strong>${f4(selected.upper_mse)}</strong></span><span>Root mean squared gap <strong>${(100*selected.rmse).toFixed(1)} pp</strong></span>`;
    for(const row of document.querySelectorAll('[data-cp-epoch]')) row.setAttribute('aria-current',String(Number(row.dataset.cpEpoch)===selected.epoch));
    drawTrend(selected); drawAgreement(selected);
  }
  const p=data.provenance||{};
  const r=p.reference||{};
  const reference = p.reference_description || r.description || (r.scope ? `${Number(r.rows).toLocaleString()} ${r.scope}, containing ${Number(r.unique_lineups).toLocaleString()} distinct line-ups. The snapshot includes rows later rejected by rule tables.` : 'See metrics.json for the fixed reference and its recorded source hash.');
  document.getElementById('cp-reference').textContent='New line-up reference: '+reference;
  const interval = p.interval_method || p.interval || p.bootstrap?.description || 'Intervals describe sampling uncertainty within the recorded training run.';
  document.getElementById('cp-uncertainty').textContent='Uncertainty: '+interval;
  const pooled=data.pooled_repeats||[];
  document.getElementById('cp-repeats').textContent=pooled.length ? 'Existing three-sampling-seed summaries are kept separate: '+pooled.map(e=>'epoch '+e.epoch+', error '+f4(e.mse)).join('; ')+'. They are not substituted into the one-seed epoch curve. Sampling-repeatability is a separate decision ticket.' : '';
  document.getElementById('cp-audit').textContent=p.audit||'';
  document.getElementById('cp-reproduce').textContent=p.reproduce||'';
  select.addEventListener('change',render);
  if(typeof ResizeObserver!=='undefined') new ResizeObserver(render).observe(document.querySelector('.cp-report'));
  render();
})();
</script></body></html>'''


def render(source, output):
    data = json.loads(source.read_text())
    epochs = data["epochs"]
    if not epochs or any(len(e["asks"]) != 23 for e in epochs):
        raise ValueError("Every recorded epoch must have all 23 asks")
    options = "".join(
        f'<option value="{e["epoch"]}"{" selected" if e["epoch"] == 34 else ""}>Epoch {e["epoch"]}</option>'
        for e in epochs
    )
    rows = "".join(
        f'<tr data-cp-epoch="{e["epoch"]}"><td>Epoch {e["epoch"]}</td><td>{e["mse"]:.4f}</td>'
        f'<td>{e["upper_mse"]:.4f}</td><td>{e["distinct_lineups"]:,}</td><td>{e["new_lineups"]:,}</td>'
        f'<td>{100 * e["full_points_share"]:.1f}%</td><td>{100 * e["mean_within_ask_sd"]:.1f}</td></tr>'
        for e in epochs
    )
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    page = TEMPLATE.replace("__OPTIONS__", options).replace("__ROWS__", rows).replace("__DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    print(f"Wrote {output}, {len(page):,} characters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "reports/training-checkpoints-2026-11-05"
    parser.add_argument("source", nargs="?", type=Path, default=default / "metrics.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--png", action="store_true", help="Also export a static scientific figure; requires matplotlib")
    args = parser.parse_args()
    render(args.source, args.output or args.source.with_name("index.html"))
    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        data = json.loads(args.source.read_text())
        rows = data['epochs']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        epoch = [r['epoch'] for r in rows]
        error = np.array([r['mse'] for r in rows])
        bounds = np.array([r['mse_ci'] for r in rows]).T
        axes[0].errorbar(epoch, error, yerr=np.array([error-bounds[0], bounds[1]-error]),
                         fmt='o-', color='#2463b5', linewidth=1.8, capsize=3)
        axes[0].axhline(.001, color='#24784d', linestyle='--', label='Near line: error < 0.001')
        axes[0].set(xlabel='Training epoch', ylabel='Ask-versus-got error', ylim=(0, .03),
                    title='Error falls early, then flattens')
        axes[0].set_xticks([1, 21, 34, 55, 80])
        axes[0].legend(fontsize=9, loc='upper right')
        for ep, color, style in [(1, '#777777', '--'), (34, '#2463b5', '-'), (80, '#b96014', '-')]:
            r = next(r for r in rows if r['epoch'] == ep)
            axes[1].plot([a['ask'] for a in r['asks']], [a['win'] for a in r['asks']],
                         style, marker='.', color=color, label=f'Epoch {ep}')
        axes[1].plot([0,1], [0,1], color='#24784d', linewidth=1, label='Got = ask')
        axes[1].axvspan(17.5/24, 1, color='#b96014', alpha=.07)
        axes[1].set(xlabel='Requested win rate', ylabel='Achieved win rate', xlim=(0,1), ylim=(0,1),
                    title='Higher requests still miss the diagonal')
        axes[1].legend(fontsize=9, loc='upper left')
        for ax in axes:
            ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('84-field model, labelled teams only, label noise ±1/48', fontsize=13)
        fig.supxlabel('One training run, sampling seed 1, 64 teams at each of 23 asks.\n'
                      'Error scores asks 1/24 to 17/24. Bars: 95% team-and-opponent bootstrap intervals.', fontsize=9)
        fig.savefig(args.source.with_name('checkpoint-error.png'), dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
