#!/usr/bin/env python3
"""Render the retrospective checkpoint storyboard without changing the observations."""
import argparse
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training helps early; the ask-versus-got gap remains</title>
<style>
:root { color-scheme: light dark; --cp-ink:var(--foreground,#e8edf4); --cp-muted:var(--muted-foreground,#b5c0cf); --cp-grid:var(--border,#414b5b); --cp-early:#aab5c4; --cp-34:color-mix(in srgb,var(--chart-1,#69b6ef) 85%,var(--cp-ink)); --cp-80:color-mix(in srgb,var(--warning,#f1a86a) 75%,var(--cp-ink)); --cp-target:var(--success,#68c7a8); --cp-extra:var(--chart-4,#c4a5fa); --cp-highlight:var(--accent-surface,#263343); }
* { box-sizing:border-box; }
html,body { margin:0; background:transparent; }
body { color:var(--cp-ink); font:16px/24px var(--font-sans,system-ui,sans-serif); }
.cp-report { width:100%; }
h1,h2,p { margin:0; }
h1 { font-size:24px; line-height:32px; font-weight:700; letter-spacing:-.02em; max-width:35rem; }
h2 { font-size:20px; line-height:24px; font-weight:650; letter-spacing:-.01em; }
.cp-intro { margin:8px 0 24px; color:var(--cp-muted); max-width:35rem; }
.cp-note { margin:8px 0 0; font-size:16px; line-height:24px; color:var(--cp-muted); max-width:35rem; }
.cp-section + .cp-section { border-top:1px solid var(--cp-grid); margin-top:32px; padding-top:24px; }
.cp-facts { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin:16px 0 8px; }
.cp-fact { border-left:3px solid var(--cp-grid); padding-left:12px; }
.cp-fact strong { display:block; font-size:20px; line-height:24px; font-weight:650; }
.cp-fact span { display:block; color:var(--cp-muted); font-size:12px; line-height:16px; margin-top:4px; }
.cp-chart { display:block; width:100%; height:304px; overflow:visible; margin-top:8px; }
.cp-chart text { fill:var(--cp-muted); font-family:inherit; font-size:12px; }
.cp-chart .cp-axis-title { fill:var(--cp-ink); font-size:12px; }
.cp-chart .cp-grid { stroke:var(--cp-grid); stroke-width:1; opacity:.7; }
.cp-chart .cp-emphasis { font-weight:650; fill:var(--cp-ink); }
.cp-readings { display:flex; flex-wrap:wrap; gap:8px 24px; margin:16px 0 8px; align-items:baseline; }
.cp-readings strong { font-weight:650; font-variant-numeric:tabular-nums; }
.cp-readings .cp-note { margin:0; }
.cp-34 { color:var(--cp-34); } .cp-80 { color:var(--cp-80); } .cp-1 { color:var(--cp-early); } .cp-target { color:var(--cp-target); }
.cp-keys { display:flex; flex-wrap:wrap; gap:8px 24px; font-size:12px; line-height:16px; margin-top:8px; }
.cp-key { display:inline-flex; align-items:center; gap:8px; }
.cp-swatch { display:inline-block; width:20px; border-top:2px solid currentColor; }
.cp-dashed { border-top-style:dashed; }
.cp-more { margin-top:24px; border-top:1px solid var(--cp-grid); padding-top:16px; }
summary { cursor:pointer; font-size:16px; line-height:24px; width:fit-content; }
summary:focus-visible,select:focus-visible,a:focus-visible { outline:2px solid var(--ring,#8cc8ff); outline-offset:4px; }
.cp-inspect { display:flex; flex-wrap:wrap; align-items:center; gap:8px 16px; margin-top:16px; }
.cp-inspect label { font-weight:600; }
select { font:inherit; color:var(--cp-ink); background:var(--secondary,#263343); border:1px solid var(--cp-muted); border-radius:var(--radius,6px); padding:8px 32px 8px 8px; min-height:40px; }
.cp-stats { display:flex; flex-wrap:wrap; gap:8px 24px; margin-top:16px; font-size:12px; line-height:16px; color:var(--cp-muted); }
.cp-stats strong { color:var(--cp-ink); font-variant-numeric:tabular-nums; }
.cp-table-wrap { overflow-x:auto; margin-top:16px; }
.cp-table { border-collapse:collapse; width:100%; font-size:12px; line-height:16px; font-variant-numeric:tabular-nums; }
.cp-table th,.cp-table td { text-align:right; white-space:nowrap; padding:8px; border-bottom:1px solid var(--cp-grid); }
.cp-table th:first-child,.cp-table td:first-child { text-align:left; }
.cp-table th { font-weight:600; color:var(--cp-muted); }
.cp-table tr[aria-current="true"] { background:var(--cp-highlight); }
.cp-more p { margin-top:16px; }
a { color:var(--cp-34); text-underline-offset:3px; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; font:12px/16px var(--font-mono,monospace); color:var(--cp-muted); }
@media(prefers-color-scheme:light) { :root { --cp-ink:var(--foreground,#172a3a); --cp-muted:var(--muted-foreground,#475a6b); --cp-grid:var(--border,#d4dce3); --cp-early:#687786; --cp-34:color-mix(in srgb,var(--chart-1,#0072b2) 85%,var(--cp-ink)); --cp-80:color-mix(in srgb,var(--warning,#b65b13) 55%,var(--cp-ink)); --cp-target:color-mix(in srgb,var(--success,#087354) 60%,var(--cp-ink)); --cp-extra:var(--chart-4,#8055b4); --cp-highlight:var(--accent-surface,#e8f2fa); } select { background:var(--secondary,#f0f4f7); } }
@media(max-width:420px) { .cp-facts { gap:16px; } .cp-fact { padding-left:8px; } }
</style></head><body><main class="cp-report">
<h1 class="cp-lead">Training helps early. The remaining gap persists.</h1>
<p class="cp-intro">More epochs initially bring achieved win rates closer to requests. The later checkpoints show no clear further reduction in error.</p>
<section class="cp-section" aria-labelledby="cp-trend-heading">
<h2 id="cp-trend-heading">Most observed improvement is early</h2>
<div class="cp-facts"><div class="cp-fact"><strong>66.2% lower error</strong><span>Epoch 1 → epoch 34, same sampling seed.</span></div><div class="cp-fact"><strong>Further gain uncertain</strong><span>Epochs 34–80 stay near 0.006.</span></div></div>
<svg id="cp-trend" class="cp-chart" role="img" aria-labelledby="cp-trend-title cp-trend-desc"></svg>
<p class="cp-note">Error is the mean squared gap between each request and its mean achieved win rate, over requests 1/24 to 17/24. Lower is better. Dots are observed checkpoints; lines guide the eye. Thin bars are 95% intervals from whole-team and shared-opponent bootstraps. They exclude variation between training runs.</p>
</section>
<section class="cp-section" aria-labelledby="cp-agreement-heading">
<h2 id="cp-agreement-heading">Higher requests still miss the diagonal</h2>
<div class="cp-readings"><strong>Ask 70.8%</strong><span class="cp-34">Epoch 34 gets <strong id="cp-got34">51.8%</strong></span><span class="cp-80">Epoch 80 gets <strong id="cp-got80">52.8%</strong></span></div>
<p class="cp-note">This request is 17/24, inside the scored range. Training longer leaves a large observed gap here; these two means do not establish a better checkpoint.</p>
<div class="cp-keys"><span class="cp-key cp-34"><i class="cp-swatch"></i>Epoch 34</span><span class="cp-key cp-80"><i class="cp-swatch"></i>Epoch 80</span><span class="cp-key cp-1"><i class="cp-swatch cp-dashed"></i>Epoch 1</span><span class="cp-key cp-target"><i class="cp-swatch"></i>Got = ask</span></div>
<svg id="cp-agreement" class="cp-chart" role="img" aria-labelledby="cp-agreement-title cp-agreement-desc"></svg>
<p class="cp-note">The vertical mark shows the remaining miss at request 17/24. The shaded region covers requests 18/24 to 23/24, where training examples are scarce; their error is reported separately. Curves show 64-team means at each request.</p>
</section>
<details class="cp-more" id="cp-explorer-details"><summary>Inspect all ten checkpoints and supporting evidence</summary>
<div class="cp-inspect"><label for="cp-epoch">Checkpoint to inspect</label><select id="cp-epoch">__OPTIONS__</select></div>
<div class="cp-stats" id="cp-stats" aria-live="polite"></div>
<p class="cp-note">The selected checkpoint appears beside the fixed epoch-34 and epoch-80 curves. Color keeps the same meaning in every figure.</p>
<p class="cp-note" id="cp-selected-label"></p>
<svg id="cp-explorer" class="cp-chart" role="img" aria-labelledby="cp-explorer-title cp-explorer-desc"></svg>
<div class="cp-table-wrap"><table class="cp-table"><thead><tr><th>Checkpoint</th><th>Error<br>1/24–17/24</th><th>Error<br>18/24–23/24</th><th>Distinct<br>line-ups</th><th>New<br>line-ups</th><th>Full<br>66 points</th><th>Team spread<br>percentage points</th><th>Slope<br>all 23 requests</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="cp-note">A line-up is a team's six species, independent of slot order. Counts are unique across all 1,472 generated teams. Full 66 points is the share of Pokémon spending all available Stat Points. Team spread is the average within-request standard deviation of team win rates.</p>
<p class="cp-note">Slope is a secondary diagnostic: a straight-line fit of mean achieved win rate against requested win rate across all 23 requests. A slope near 1 can still miss the diagonal, so ask-versus-got error remains the primary measure.</p>
<p class="cp-note" id="cp-reference"></p><p class="cp-note" id="cp-uncertainty"></p>
<p class="cp-note">One training run, seed 0; sampling seed 1 at every checkpoint; 64 teams at each of 23 requests. Each team plays the 49 distinct top-50 opponents with behaviour-cloning on both sides and shuffled slots. The duplicated meta entry has score weight 2, giving total weight 50.</p>
<p class="cp-note">The labelled-only model generates all 84 fields with label noise ±1/48. Stat Alignment is repaired to match the moves. These samples use no search or elite-selection rule.</p>
<p class="cp-note" id="cp-repeats"></p>
<p class="cp-note">This is exploratory evidence from one training run. Epoch 34 is the lowest observed one-seed error, not a proven optimum. Epoch 80 is the training cap and lowest validation-loss checkpoint. No measured checkpoint reaches the near-line threshold of 0.001.</p>
<p class="cp-note" id="cp-audit"></p>
<p class="cp-note">The <a href="https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/t3/checkpoint-error-by-epochs/code/reports/training-checkpoints-2026-11-05/storyboard.json" target="_blank">figure storyboard</a> was written retrospectively from the recorded results. It orders the explanation as early improvement, remaining mismatch, then a single-view teaser. It is not a pre-registration.</p>
<p class="cp-note"><a href="https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/t3/checkpoint-error-by-epochs/code/reports/training-checkpoints-2026-11-05/metrics.json" target="_blank">Per-epoch data</a> · <a href="https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/t3/checkpoint-error-by-epochs/code/reports/training-checkpoints-2026-11-05/asks.csv" target="_blank">Per-request data</a> · <a href="https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/t3/checkpoint-error-by-epochs/code/reports/training-checkpoints-2026-11-05/checkpoint-teaser.svg" target="_blank">Vector teaser</a></p>
<details><summary>Recompute from the saved observations</summary><pre id="cp-reproduce"></pre></details>
<p class="cp-note"><a href="https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/105" target="_blank" rel="noreferrer">Training checkpoints decision ticket</a></p>
</details></main><script id="cp-data" type="application/json">__DATA__</script><script>
(() => {
'use strict';
const data=JSON.parse(document.getElementById('cp-data').textContent), epochs=data.epochs;
const byEpoch=n=>epochs.find(e=>e.epoch===n), select=document.getElementById('cp-epoch');
const color=n=>n===1?'var(--cp-early)':n===34?'var(--cp-34)':n===80?'var(--cp-80)':'var(--cp-extra)';
const pct=v=>(100*v).toFixed(1)+'%', esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const line=(x1,y1,x2,y2,a='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${a}/>`;
const text=(x,y,s,a='')=>`<text x="${x}" y="${y}" ${a}>${esc(s)}</text>`;
const path=(rows,x,y,key,a='')=>`<path d="${rows.map((r,i)=>(i?'L':'M')+x(r)+','+y(r[key])).join(' ')}" fill="none" ${a}/>`;
function layout(svg){const w=Math.max(280,Math.round(svg.getBoundingClientRect().width));svg.setAttribute('viewBox',`0 0 ${w} 304`);return {w,h:304,l:56,r:20,t:40,b:48};}
function drawTrend(){
 const svg=document.getElementById('cp-trend'),p=layout(svg),x=e=>p.l+(e-1)/79*(p.w-p.l-p.r),y=v=>p.h-p.b-v/.03*(p.h-p.t-p.b);
 let out='<title id="cp-trend-title">Most observed improvement is early</title><desc id="cp-trend-desc">Error drops 66.2 percent from epoch 1 to epoch 34. Further improvement between epochs 34 and 80 is uncertain. No checkpoint reaches error under 0.001.</desc>';
 for(const v of [0,.005,.01,.015,.02,.025,.03])out+=line(p.l,y(v),p.w-p.r,y(v),'class="cp-grid"')+text(p.l-8,y(v)+4,v.toFixed(3),'text-anchor="end"');
 for(const n of [1,21,34,55,80])out+=text(x(n),p.h-p.b+20,n,'text-anchor="middle"');
 out+=text(p.l,14,'Ask-versus-got error','class="cp-axis-title"')+text((p.l+p.w-p.r)/2,300,'Training epoch','text-anchor="middle" class="cp-axis-title"');
 out+=line(x(34),29,x(80),29,'stroke="var(--cp-muted)"')+line(x(34),26,x(34),34,'stroke="var(--cp-muted)"')+line(x(80),26,x(80),34,'stroke="var(--cp-muted)"')+text((x(34)+x(80))/2,20,'Later gain uncertain','text-anchor="middle"');
 out+=line(p.l,y(.001),p.w-p.r,y(.001),'stroke="var(--cp-target)" stroke-dasharray="5 4"')+text(p.w-p.r,y(.001)-7,'Target < 0.001: none reaches it','text-anchor="end" style="fill:var(--cp-target)"');
 for(const e of epochs)out+=line(x(e.epoch),y(e.mse_ci[0]),x(e.epoch),y(e.mse_ci[1]),'stroke="var(--cp-muted)" opacity=".25"');
 out+=path(epochs,r=>x(r.epoch),y,'mse','stroke="var(--cp-muted)" stroke-width="2"');
 for(const e of epochs)out+=`<circle cx="${x(e.epoch)}" cy="${y(e.mse)}" r="${[1,34,80].includes(e.epoch)?4.5:3}" fill="${[1,34,80].includes(e.epoch)?color(e.epoch):'var(--cp-muted)'}"><title>Epoch ${e.epoch}: ${e.mse.toFixed(6)}</title></circle>`;
 out+=text(x(1)+10,y(byEpoch(1).mse)-10,'0.01750','class="cp-emphasis"')+line(x(34),y(byEpoch(34).mse)-5,x(34)+8,y(.012)+5,'stroke="var(--cp-34)" stroke-width="1"')+text(x(34)+12,y(.012),'Epoch 34: 0.00591','style="fill:var(--cp-34)"')+text(x(80),y(byEpoch(80).mse)-14,'0.00600','text-anchor="end" style="fill:var(--cp-80)"');
 svg.innerHTML=out;
}
function drawAgreement(id,selected=null){
 const svg=document.getElementById(id),p=layout(svg),x=r=>p.l+r.ask*(p.w-p.l-p.r),y=v=>p.h-p.b-v*(p.h-p.t-p.b);
 let out=`<title id="${id}-title">${selected?'Requested versus achieved win rate at epoch '+selected.epoch:'Higher requests still miss the diagonal'}</title><desc id="${id}-desc">Epochs 34 and 80 remain visible together. At a request of 70.8 percent they achieve ${pct(byEpoch(34).asks[16].win)} and ${pct(byEpoch(80).asks[16].win)}. The diagonal is exact agreement.</desc>`;
 out+=`<rect x="${x({ask:17.5/24})}" y="${p.t}" width="${p.w-p.r-x({ask:17.5/24})}" height="${p.h-p.t-p.b}" fill="var(--cp-muted)" opacity=".07"/>`;
 for(const v of [0,.25,.5,.75,1])out+=line(p.l,y(v),p.w-p.r,y(v),'class="cp-grid"')+text(p.l-8,y(v)+4,Math.round(v*100)+'%','text-anchor="end"')+text(x({ask:v}),p.h-p.b+20,Math.round(v*100)+'%','text-anchor="middle"');
 out+=text(p.l,20,'Achieved win rate','class="cp-axis-title"')+text((p.l+p.w-p.r)/2,300,'Requested win rate','text-anchor="middle" class="cp-axis-title"');
 out+=line(x({ask:0}),y(0),x({ask:1}),y(1),'stroke="var(--cp-target)" stroke-width="1.5"');
 let shown=[1,34,80];if(selected&&!shown.includes(selected.epoch))shown.push(selected.epoch);
 for(const n of shown){const e=byEpoch(n),style=`stroke="${color(n)}" stroke-width="${n===1?1.2:2.3}" ${n===1?'stroke-dasharray="4 4" opacity=".65"':''}`;out+=path(e.asks.filter(a=>a.numerator<=17),x,y,'win',style)+path(e.asks.filter(a=>a.numerator>=17),x,y,'win',style+' stroke-dasharray="4 4"');if(n!==1)for(const a of e.asks)out+=`<circle cx="${x(a)}" cy="${y(a.win)}" r="2.4" fill="${color(n)}"><title>Epoch ${n}, ask ${a.numerator}/24: got ${pct(a.win)}</title></circle>`;}
 const ask=17/24,got=byEpoch(80).asks[16].win;
 out+=line(x({ask}),y(got),x({ask}),y(ask),'stroke="var(--cp-ink)" stroke-dasharray="2 3" opacity=".65"')+text(x({ask})+8,y((ask+got)/2)+4,'Gap','class="cp-emphasis"');
 svg.innerHTML=out;
}
function inspect(){const e=byEpoch(Number(select.value))||epochs[0];document.getElementById('cp-selected-label').textContent=`Epoch ${e.epoch}, highlighted in ${e.epoch===1?'gray':e.epoch===34?'blue':e.epoch===80?'orange':'purple'}.`;
 document.getElementById('cp-stats').innerHTML=`<span>Error <strong>${e.mse.toFixed(4)}</strong></span><span>Upper-request error <strong>${e.upper_mse.toFixed(4)}</strong></span><span>Root mean squared gap <strong>${(100*e.rmse).toFixed(1)} percentage points</strong></span>`;
 for(const r of document.querySelectorAll('[data-cp-epoch]'))r.setAttribute('aria-current',String(Number(r.dataset.cpEpoch)===e.epoch));drawAgreement('cp-explorer',e);}
const p=data.provenance||{};document.getElementById('cp-got34').textContent=pct(byEpoch(34).asks[16].win);document.getElementById('cp-got80').textContent=pct(byEpoch(80).asks[16].win);
document.getElementById('cp-reference').textContent='New line-up reference: '+p.reference_description;
document.getElementById('cp-uncertainty').textContent='Uncertainty: '+p.interval;
document.getElementById('cp-repeats').textContent='Three-sampling-seed summaries stay separate: '+data.pooled_repeats.map(e=>`epoch ${e.epoch}, error ${e.mse.toFixed(4)}`).join('; ')+'. They are not substituted into the one-seed curve.';
document.getElementById('cp-audit').textContent=p.audit||'';document.getElementById('cp-reproduce').textContent=p.reproduce||'';
function render(){drawTrend();drawAgreement('cp-agreement');inspect();}
select.addEventListener('change',inspect);document.getElementById('cp-explorer-details').addEventListener('toggle',inspect);
if(typeof ResizeObserver!=='undefined')new ResizeObserver(render).observe(document.querySelector('.cp-report'));render();
})();
</script></body></html>'''


def render(source, output):
    data = json.loads(source.read_text())
    epochs = data['epochs']
    if not epochs or any(len(e['asks']) != 23 for e in epochs):
        raise ValueError('Every recorded epoch must have all 23 asks')
    options = ''.join(f'<option value="{e["epoch"]}"{" selected" if e["epoch"] == 80 else ""}>Epoch {e["epoch"]}</option>' for e in epochs)
    rows = ''.join(
        f'<tr data-cp-epoch="{e["epoch"]}"><td>Epoch {e["epoch"]}</td><td>{e["mse"]:.4f}</td>'
        f'<td>{e["upper_mse"]:.4f}</td><td>{e["distinct_lineups"]:,}</td><td>{e["new_lineups"]:,}</td>'
        f'<td>{100*e["full_points_share"]:.1f}%</td><td>{100*e["mean_within_ask_sd"]:.1f}</td>'
        f'<td>{e["slope_all_asks"]:.4f}</td></tr>' for e in epochs)
    payload = json.dumps(data, separators=(',', ':'), ensure_ascii=True).replace('<', '\\u003c')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(TEMPLATE.replace('__OPTIONS__', options).replace('__ROWS__', rows).replace('__DATA__', payload))
    print(f'Wrote {output}')


def export_figures(source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    matplotlib.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':12,
                                 'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12})
    data = json.loads(source.read_text())
    rows = data['epochs']
    by_epoch = {r['epoch']:r for r in rows}
    colors = {1:'#687786',34:'#0072b2',80:'#b65b13'}
    ink, muted, grid, target = '#172a3a','#475a6b','#d9e0e6','#087354'

    def setup(ax):
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(grid)
        ax.tick_params(colors=muted, length=0, pad=8)
        ax.grid(axis='y', color=grid, linewidth=.7)
        ax.set_axisbelow(True)

    def trend(ax):
        setup(ax)
        xx=np.array([r['epoch'] for r in rows]); yy=np.array([r['mse'] for r in rows]);ci=np.array([r['mse_ci'] for r in rows]).T
        ax.errorbar(xx,yy,yerr=[yy-ci[0],ci[1]-yy],fmt='none',ecolor='#c2cbd3',linewidth=1,zorder=1)
        ax.plot(xx,yy,color=muted,linewidth=1.8,zorder=2)
        ax.scatter(xx,yy,color=muted,s=18,zorder=3)
        for ep in (1,34,80):
            r=by_epoch[ep];ax.scatter([ep],[r['mse']],color=colors[ep],s=50,zorder=4)
        ax.axhline(.001,color=target,linestyle=(0,(4,3)),linewidth=1.2)
        ax.text(79,.0017,'Target < 0.001: no checkpoint reaches it',ha='right',va='bottom',fontsize=12,color=target)
        ax.set(xlim=(-1,82),ylim=(0,.029),xlabel='Training epoch',ylabel='Ask-versus-got error')
        ax.set_xticks([1,21,34,55,80]);ax.set_yticks([0,.01,.02])
        ax.text(3,.0186,'Epoch 1\n0.01750',fontsize=12,color=colors[1],va='bottom')
        ax.annotate('Epoch 34\n0.00591',xy=(34,by_epoch[34]['mse']),xytext=(38,.012),ha='left',va='bottom',fontsize=12,color=colors[34],arrowprops={'arrowstyle':'-','color':colors[34],'linewidth':.8})
        ax.text(80,.0076,'Epoch 80\n0.00600',ha='right',va='bottom',fontsize=12,color=colors[80])
        ax.plot([34,34,80,80],[.0255,.0262,.0262,.0255],color=muted,linewidth=1)
        ax.text(57,.027,'Further improvement uncertain',ha='center',fontsize=12,color=muted)

    def agreement(ax):
        setup(ax)
        ax.axvspan(17.5/24,1,color=muted,alpha=.05)
        ax.plot([0,1],[0,1],color=target,linewidth=1.3)
        ax.text(.98,.99,'Got = ask',ha='right',va='bottom',fontsize=12,color=target)
        for ep in (1,34,80):
            row=by_epoch[ep]
            ax.plot([a['ask'] for a in row['asks']],[a['win'] for a in row['asks']],
                    color=colors[ep],linewidth=1.2 if ep==1 else 2.2,linestyle='--' if ep==1 else '-',
                    marker='.' if ep!=1 else None,markersize=4,alpha=.65 if ep==1 else 1)
        for ep,ypos in ((34,.40),(80,.31)):
            ax.annotate(f'Epoch {ep}',xy=(23/24,by_epoch[ep]['asks'][-1]['win']),xytext=(.99,ypos),ha='right',
                        color=colors[ep],fontsize=12,arrowprops={'arrowstyle':'-','color':colors[ep],'linewidth':.8})
        ax.text(.88,.18,'Epoch 1',color=colors[1],fontsize=12)
        ask=17/24;got=by_epoch[80]['asks'][16]['win']
        ax.plot([ask,ask],[got,ask],color=ink,linewidth=1,linestyle=(0,(2,3)))
        ax.text(ask+.02,(ask+got)/2,'Gap remains',fontsize=12,color=ink)
        ax.set(xlim=(0,1),ylim=(0,1.06),xlabel='Requested win rate',ylabel='Achieved win rate')
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1])

    # Evidence sequence: each panel carries one sentence, read from top to bottom.
    fig=plt.figure(figsize=(12,11.2),facecolor='white')
    fig.text(.08,.96,'Training helps early. The remaining gap persists.',fontsize=24,weight='bold',color=ink)
    fig.text(.08,.922,'Same sampling seed at all ten checkpoints; one training run.',fontsize=12,color=muted)
    ax1=fig.add_axes([.10,.60,.84,.255]);trend(ax1)
    fig.text(.08,.885,'1  Most observed improvement is early',fontsize=20,weight='bold',color=ink)
    fig.text(.08,.54,'66.2% lower error from epoch 1 to 34. The later checkpoints show no clear further reduction.',fontsize=12,color=muted)
    ax2=fig.add_axes([.10,.16,.84,.255]);agreement(ax2)
    fig.text(.08,.46,'2  Higher requests still miss the diagonal',fontsize=20,weight='bold',color=ink)
    fig.text(.08,.08,'Ask 70.8%: epoch 34 gets 51.8%; epoch 80 gets 52.8%. This request is inside the scored range.',fontsize=12,color=ink)
    fig.text(.08,.035,'Error scores requests 1/24–17/24. Shading marks sparse-training requests 18/24–23/24.\nBars: 95% whole-team and shared-opponent intervals, excluding independent training-run variation.',fontsize=12,color=muted,linespacing=1.5)
    for extension in ('png','svg'):fig.savefig(source.with_name(f'checkpoint-error.{extension}'),dpi=160,facecolor='white')
    plt.close(fig)
    # Teaser: one annotated trajectory, then one concrete request/achievement example.
    fig=plt.figure(figsize=(12,7.2),facecolor='white')
    fig.text(.075,.94,'More training helps early. It does not close the gap.',fontsize=24,weight='bold',color=ink)
    fig.text(.075,.89,'66.2% lower error by epoch 34; further improvement through epoch 80 remains uncertain.',fontsize=12,color=muted)
    ax=fig.add_axes([.10,.38,.84,.43]);trend(ax)
    fig.text(.075,.22,'A request still missed',fontsize=16,weight='bold',color=ink)
    fig.text(.075,.16,'Ask 70.8%',fontsize=20,weight='bold',color=ink)
    fig.text(.29,.16,'→',fontsize=24,color=muted)
    fig.text(.36,.16,'51.8%',fontsize=24,weight='bold',color=colors[34]);fig.text(.36,.115,'got at epoch 34',fontsize=12,color=colors[34])
    fig.text(.60,.16,'52.8%',fontsize=24,weight='bold',color=colors[80]);fig.text(.60,.115,'got at epoch 80',fontsize=12,color=colors[80])
    fig.text(.80,.16,'Gap persists',fontsize=16,weight='bold',color=ink);fig.text(.80,.115,'within scored requests',fontsize=12,color=muted)
    fig.text(.075,.045,'One training run; sampling seed 1; 64 teams per request. Error scores requests 1/24–17/24.\nBars resample teams and shared opponents. Independent training-run variation is unmeasured.',fontsize=12,color=muted,linespacing=1.5)
    for extension in ('png','svg'):fig.savefig(source.with_name(f'checkpoint-teaser.{extension}'),dpi=160,facecolor='white')
    plt.close(fig)
    for name in ('checkpoint-error.svg', 'checkpoint-teaser.svg'):
        svg = source.with_name(name)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    print('Wrote storyboard evidence and teaser, each as text-preserving SVG and PNG.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parents[1]/'reports/training-checkpoints-2026-11-05'
    parser.add_argument('source',nargs='?',type=Path,default=default/'metrics.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--png',action='store_true',help='Also export the evidence and teaser as PNG and text-preserving SVG; requires matplotlib')
    args=parser.parse_args()
    render(args.source,args.output or args.source.with_name('index.html'))
    if args.png:export_figures(args.source)


if __name__=='__main__':
    main()
