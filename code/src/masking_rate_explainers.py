"""Seven one-idea explainer pages for the masking-schedule rows (docs/masking-rate-rows.md), one per question the owner asked
on 2026-11-11. Diagrams are plain HTML and SVG. The one chart is ECharts: `teams_by_ask_option` builds its option as a
Python dict, `write_chart` draws it as its own page, and the tables explainer embeds the same dict. Numbers come from the
stored probe and table results, or are quoted from the pre-registration.

    /tmp/vgc-pilot/.venv/bin/python src/masking_rate_explainers.py      (from code/)

Each page is written twice: a standalone copy under docs/figures/masking_rate_rows/ that follows light and dark, and a
copy without its own background under verify-evidence/, which a chat thread can frame.
"""
import html
import json
from pathlib import Path

from echarts_chart import DARK, write_chart

CODE = Path(__file__).resolve().parent.parent
OUT = CODE / "docs/figures/masking_rate_rows"
FRAMED = CODE.parent / "verify-evidence/figure-pages/2026-11-11-masking-rate-rows/framed"
RES = Path.home() / "vgc-data/results"
ECHARTS = "https://cdn.jsdelivr.net/npm/echarts@6.1.0/dist/echarts.min.js"

SHARED = """
*{box-sizing:border-box} h1{font-size:17px;line-height:1.3;margin:0 0 4px;font-weight:650} h2{font-size:14px;margin:18px 0 6px;font-weight:650}
p{margin:0 0 8px} .sub{color:var(--soft);font-size:13px;line-height:1.45;margin:0 0 12px} .note{color:var(--soft);font-size:12px;line-height:1.45;margin:8px 0 0}
.grid{display:grid;gap:10px} .g2{grid-template-columns:repeat(2,minmax(0,1fr))} .g3{grid-template-columns:repeat(3,minmax(0,1fr))}
@media (max-width:560px){.g2,.g3{grid-template-columns:1fr}}
.box{background:var(--box);border:1px solid var(--edge);border-radius:var(--r);padding:14px 16px}
.box h3{margin:0 0 4px;font-size:14px;line-height:1.3;font-weight:650} .box p{font-size:13px;line-height:1.45;color:var(--soft);margin:0}
.big{font-size:22px;font-weight:700;line-height:1.15;color:var(--ink);font-variant-numeric:tabular-nums}
code,.mono{font-family:var(--mono);font-size:12px} .chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:650;margin-top:8px}
.ok{color:var(--ok);background:color-mix(in srgb,var(--ok) 16%,transparent)} .bad{color:var(--bad);background:color-mix(in srgb,var(--bad) 16%,transparent)}
.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 16%,transparent)} .plain{color:var(--soft);background:color-mix(in srgb,var(--soft) 14%,transparent)}
.flow{display:flex;align-items:stretch;gap:6px;flex-wrap:wrap} .flow .box{flex:1 1 150px} .arrow{align-self:center;color:var(--soft);font-size:18px}
table{border-collapse:collapse;width:100%;font-size:12.5px} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--edge);vertical-align:top;line-height:1.35}
th{color:var(--soft);font-weight:600;font-size:12px} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.same{background:color-mix(in srgb,var(--ok) 15%,transparent)} .diff{background:color-mix(in srgb,var(--warn) 15%,transparent)} .none{color:var(--soft)}
.key{display:inline-flex;align-items:center;gap:6px;margin-right:14px;font-size:12px;color:var(--soft)} .sw{width:12px;height:12px;border-radius:3px;display:inline-block;border:1px solid var(--edge)}
svg text{fill:var(--soft);font-size:9px;font-family:inherit} svg .ink{fill:var(--ink)}
"""
STANDALONE = """
html{background:#fcfcfb;color:#0b0b0b;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Helvetica,Arial,sans-serif}
body{margin:0;--ink:#0b0b0b;--soft:#52514e;--box:#f3f2ee;--edge:#dcdbd5;--c1:#2a78d6;--c2:#1baf7a;--c3:#c98a00;--ok:#16803c;--warn:#a16207;--bad:#c0392b;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace;--r:10px} main{max-width:760px;margin:0 auto;padding:24px 16px}
@media (prefers-color-scheme:dark){html{background:#161615;color:#f2f1ee}
body{--ink:#f2f1ee;--soft:#a9a8a2;--box:#1f1f1e;--edge:#34342f;--c1:#5b9bf0;--c2:#2dd4a4;--c3:#f5c04a;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171}}
"""
FRAME = """
body{--ink:var(--foreground);--soft:var(--muted-foreground);--box:var(--card);--edge:var(--border);--c1:var(--chart-1);--c2:var(--chart-2);--c3:var(--chart-3);
--ok:var(--success);--warn:var(--warning);--bad:var(--destructive);--mono:var(--font-mono);--r:var(--radius)}
"""
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{theme}{shared}</style></head><body><main>
{body}
</main>{script}</body></html>
"""


def write(name, title, body, script=""):
    for folder, theme in ((OUT, STANDALONE), (FRAMED, FRAME)):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.html").write_text(PAGE.format(title=html.escape(title), theme=theme, shared=SHARED, body=body, script=script))
    return OUT / f"{name}.html"


def chip(text, kind="plain", style=""):
    return f'<span class="chip {kind}" style="{style}">{text}</span>' if text else ""


def box(title, text, big="", chip_text="", kind="plain"):
    head = f'<div class="big">{big}</div>' if big else ""
    return f'<div class="box">{head}<h3>{title}</h3><p>{text}</p>{chip(chip_text, kind)}</div>'


def probes():
    p = json.load(open(RES / "masking_rate_rows_preflight.json"))
    t = p["t_sensitivity"]; asks = list(t["by_ask"].values())
    fields = [a["share_of_fields_changed"] for a in asks]
    lineups = round(sum(a["share_of_line_ups_changed"] * t["teams_per_ask"] for a in asks))
    smoke, replay = p["training_smoke"], p["recipe_replay"]
    body = f"""<h1>A probe is a quick check before the expensive steps</h1>
<p class="sub">Its numbers are what that check measured. No probe here used a battle.</p>
<div class="flow">
  {box("Probes", "Six checks on the plan and the code.", "minutes", "0 battles", "ok")}<div class="arrow">→</div>
  {box("Training", "Two models, per-field and linear.", "13 hours")}<div class="arrow">→</div>
  {box("Battles", "Every row at guidance 2, 4 and 8.", "8 hours", "2.3 million battles")}
</div>
<h2>The six probes, one number each</h2>
<div class="grid g3">
  {box("Does the linear rate hide the right share?", "Largest gap from the share it should hide.", f"{p['hidden_share']['largest_gap_any_field']:.3f}", "pass", "ok")}
  {box("Is the corrected t computed right?", "Gap from the corrected-t ticket's table.", f"{p['corrected_t']['largest_gap_from_ticket_table']:.5f}", "pass", "ok")}
  {box("Does the model react to the corrected t?", "Largest shift in its choices. The bar was 0.01.", f"{t['largest_total_variation_corrected']:.3f}", "fail: too small to matter", "bad")}
  {box("Does the loss reproduce?", "Validation loss now against the logged one.", f"{p['cross_rate_loss']['per_field_rate']:.4f} = {p['cross_rate_loss']['logged_at_epoch_80']:.4f}", "pass", "ok")}
  {box("Does a fresh network learn with the linear rate?", "Loss on 32 teams, start to end.", f"{smoke['linear']['first_validation_loss']:.2f} → {smoke['linear']['last_validation_loss']:.2f}", "pass", "ok")}
  {box("Does today's code train like the old run?", "Gap in first-epoch loss. The bar was 0.01.", f"{replay['gaps']['training_loss']:.5f}", "pass", "ok")}
</div>
<h2>"The probe's numbers" for Per-field, corrected t</h2>
<div class="grid g3">
  {box("Largest shift in the model's choices", "0 is no change, 1 is a complete change.", f"{t['largest_total_variation_corrected']:.3f}")}
  {box("Fields that came out different", "Same sampling seed, corrected t against today's.", f"{min(fields):.0%} to {max(fields):.0%}")}
  {box("Line-ups that changed", "Teams whose six species differ.", f"{lineups} of {t['teams_per_ask'] * len(asks)}")}
</div>
<p class="note">Reporting the row "from the probe's numbers" means the results table shows these three figures for it, with no battles. Measured on today's default model, 2026-11-11.</p>"""
    return write("explain_1_probes", "What a probe is", body)


def squares():
    """84 squares in species-first order at the halfway pass: 42 filled, 42 hidden."""
    groups = [("Species", 6, True), ("Ability and item", 12, True), ("Moves", 24, True), ("Stat Alignment", 6, False), ("Stat Points", 36, False)]
    size, gap, per_row, x0, out, n = 14, 3, 42, 0, [], 0
    for name, count, filled in groups:
        start = n
        for _ in range(count):
            x, y = x0 + (n % per_row) * (size + gap), (n // per_row) * (size + gap + 22)
            style = "fill:var(--c1)" if filled else "fill:none;stroke:var(--soft);stroke-dasharray:2 2"
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="3" style="{style}"/>'); n += 1
        lx, ly = x0 + (start % per_row) * (size + gap), (start // per_row) * (size + gap + 22) + size + 13
        out.append(f'<text x="{lx}" y="{ly}" style="font-size:11px">{name} · {count}</text>')
    width = per_row * (size + gap)
    return f'<svg viewBox="0 0 {width} 84" width="100%" role="img" aria-label="84 fields, 42 filled and 42 hidden">{"".join(out)}</svg>'


def corrected_t():
    p = json.load(open(RES / "masking_rate_rows_preflight.json"))
    per_pass, today = p["corrected_t"]["per_pass"], p["corrected_t"]["today"]
    hidden = p["hidden_share"]["by_t"]["0.5"]["per_field_expected_hidden_share"]
    rows = "".join(f"<tr><td>{k} of 84</td><td class=num>{today[k]:.2f}</td><td class=num>{per_pass[k]:.2f}</td></tr>" for k in (6, 18, 42, 48))
    body = f"""<h1>What the corrected t corrects</h1>
<p class="sub">t is a number the model is told: how much of the team is hidden. Training and sampling disagree about what a given t means.</p>
<div class="box"><h3>Halfway through sampling: 42 fields filled, 42 still hidden</h3>{squares()}</div>
<div class="grid g3" style="margin-top:10px">
  {box("Today the sampler says", "It counts filled fields: 1 − 42/84.", "t = 0.50")}
  {box("But in training, t = 0.50 meant", "Species and items are almost never hidden, so less is hidden than t says.", f"{hidden:.0%} hidden")}
  {box("The corrected t says", "The t at which training hid half the team.", f"t = {per_pass[42]:.2f}", "the fix", "ok")}
</div>
<div class="grid g2" style="margin-top:10px">
  <div class="box"><h3>Across the 84 passes</h3><table><tr><th>Fields filled</th><th class=num>Today's t</th><th class=num>Corrected t</th></tr>{rows}</table></div>
  <div class="box"><h3>Does the fix matter?</h3><p>The model's choices moved by <b style="color:var(--ink)">{p['t_sensitivity']['largest_total_variation_corrected']:.3f}</b> at most. The bar was 0.01. The model barely reads t.</p>
  <span class="chip warn">How it resolves</span><p style="margin-top:6px">After the new per-field model trains, the same check runs on it. Under 0.01: no battles, the row is reported from the check. 0.01 or more: the row is battled.</p></div>
</div>
<p class="note">Per-field masking schedule: species hidden with chance t^16, ability and item t^4, moves t^1.5, Stat Alignment and Stat Points t. Measured on today's default model.</p>"""
    return write("explain_2_corrected_t", "What the corrected t corrects", body)


def plan():
    body = f"""<h1>The plan after your two decisions</h1>
<p class="sub">Shuffled-only table for training, and guidance 2, 4 and 8 for every battled row.</p>
<div class="flow">
  {box("One table", "Labels from shuffled-slot battles only.", "214,598 teams", "your choice", "ok")}<div class="arrow">→</div>
  {box("Two trainings", "Per-field, the new control. Linear, the new arm.", "2 × 6.7 hours")}<div class="arrow">→</div>
  {box("Three battled rows × three guidance values", "Control. Linear, species-first. Linear, random order.", "9 grids", "1.95 million battles, 6.9 hours")}
</div>
<h2>What each decision changed</h2>
<div class="grid g2">
  <div class="box"><h3>Guidance 2, 4 and 8</h3><table>
    <tr><th></th><th class=num>Before</th><th class=num>Now</th></tr>
    <tr><td>Each row is</td><td class=num>one point</td><td class=num>a 3-point curve</td></tr>
    <tr><td>Grids</td><td class=num>3</td><td class=num>9</td></tr>
    <tr><td>Battles</td><td class=num>0.65 million</td><td class=num>1.95 million</td></tr>
    <tr><td>Battle time</td><td class=num>2.3 hours</td><td class=num>6.9 hours</td></tr></table></div>
  <div class="box"><h3>Shuffled-only table</h3><table>
    <tr><th></th><th class=num>Before</th><th class=num>Now</th></tr>
    <tr><td>Control</td><td class=num>today's default model</td><td class=num>a new per-field training</td></tr>
    <tr><td>Trainings</td><td class=num>1</td><td class=num>2</td></tr>
    <tr><td>Training time</td><td class=num>5.4 hours</td><td class=num>13.4 hours</td></tr>
    <tr><td>Today's default model</td><td class=num>the control</td><td class=num>a reference only</td></tr></table></div>
</div>
<p class="note">Why a new control: today's default model trained on the older table. Compared with a linear model from the shuffled-only table, the table and the masking schedule would both differ, and a gap could not be pinned on either. Per-field, corrected t is battled only if its check reaches the bar, 3 more grids and 2.3 hours.</p>"""
    return write("explain_3_plan", "The plan after two decisions", body)


def teams_by_ask_option(old, new):
    """Labelled training teams at each ask, older table against shuffled-only table: the ECharts option as a dict."""
    blue, grey, plain = "#2a78d6", "#a9a8a2", "#6f6e69"
    line = lambda name, counts, color, width: dict(name=name, type="line", data=counts[1:24], symbol="none", emphasis=dict(focus="series"),
                                                   lineStyle=dict(color=color, width=width), itemStyle=dict(color=color))
    shuffled = line("Shuffled-only table", new["counts"], blue, 2.5)
    shuffled["markArea"] = dict(silent=True, itemStyle=dict(color=blue, opacity=0.08), label=dict(color=plain, formatter="strong teams", position="insideTop"),
                                data=[[dict(xAxis="13/24"), dict(xAxis="17/24")]])
    return dict(legend=dict(top=0), grid=dict(left=52, right=10, top=34, bottom=28), tooltip=dict(trigger="axis"),
                xAxis=dict(type="category", data=[f"{k}/24" for k in range(1, 24)]), yAxis=dict(type="value"),
                series=[line("Older table", old["counts"], grey, 2), shuffled])


def tables():
    t = json.load(open(RES / "masking_rate_rows_tables.json")); old, new = t["lossdata84.npz"], t["lossdata84_shuffled_only.npz"]
    strong = lambda x: sum(x["counts"][13:18])
    body = f"""<h1>The two training tables</h1>
<p class="sub">"The control's table" was the older table that today's default model trained on. You chose the shuffled-only one.</p>
<div class="grid g2">
  <div class="box"><h3>Older table</h3><p>Labels pool every battle on record, including battles where the six Pokémon kept their listed order. As few as 24 battles for a label.</p>
    <div class="big" style="margin-top:8px">{old['train']:,} teams</div><p>Mean label {old['mean']:.2f}. Near asks 13/24 to 17/24: {strong(old):,}.</p><span class="chip plain">today's default model</span></div>
  <div class="box"><h3>Shuffled-only table</h3><p>Every label comes from shuffled-slot battles alone, at least 49 for each team. The battle player's slot habit cannot tilt it.</p>
    <div class="big" style="margin-top:8px">{new['train']:,} teams</div><p>Mean label {new['mean']:.2f}. Near asks 13/24 to 17/24: {strong(new):,}.</p><span class="chip ok">your choice</span></div>
</div>
<h2>Labelled training teams at each ask</h2>
<div id="chart" style="width:100%;height:260px"></div>
<p class="note">Only 783 labelled teams are in both tables. On those, the older label is higher by 0.014 on average, and half differ by more than 0.05. The cost of the choice: fewer than half as many strong teams to learn the high asks from. The gain: labels the slot order cannot bias.</p>"""
    option = teams_by_ask_option(old, new)
    write_chart(option, OUT / "04_teams_by_ask.html", title="The shuffled-only table holds fewer strong teams",
                subtitle="Labelled training teams at each ask, the win rate nearest each team's label.",
                note=f"Older table: {old['train']:,} labelled training teams. Shuffled-only table: {new['train']:,}. Near asks 13/24 to 17/24: "
                     f"{strong(old):,} against {strong(new):,}. Source: masking_rate_rows_tables.json.")
    script = f"""<script src="{ECHARTS}"></script><script>
const option = {json.dumps(option)}, darkMap = {json.dumps({k.lower(): v for k, v in DARK.items()})};
const swap = o => typeof o === "string" ? (darkMap[o.toLowerCase()] || o) : Array.isArray(o) ? o.map(swap)
  : o && typeof o === "object" ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, swap(v)])) : o;
const el = document.getElementById("chart"), dark = matchMedia("(prefers-color-scheme: dark)");
let chart; function draw() {{ if (chart) chart.dispose(); chart = echarts.init(el, dark.matches ? "dark" : null, {{renderer: "svg"}});
  chart.setOption({{backgroundColor: "transparent", ...(dark.matches ? swap(option) : option)}}); }}
draw(); addEventListener("resize", () => chart.resize()); dark.addEventListener("change", draw);
</script>"""
    return write("explain_4_tables", "The two training tables", body, script)


def sketch(kind):
    """A frame sketch with no real numbers."""
    axes = '<path d="M30 8V92H192" style="fill:none;stroke:var(--soft)"/>'
    if kind == "curves":
        body = (axes + '<text x="96" y="107">line-ups →</text><text x="4" y="52" transform="rotate(-90 12 52)">error →</text>'
                '<path d="M52 30L96 52L150 62" style="fill:none;stroke:var(--soft);stroke-width:2"/><path d="M58 34L104 56L158 66" style="fill:none;stroke:var(--c1);stroke-width:2"/>'
                '<path d="M92 22L138 40L182 50" style="fill:none;stroke:var(--c2);stroke-width:2"/>'
                '<text x="44" y="24">2</text><text x="92" y="68">4</text><text x="146" y="78">8</text>')
    elif kind == "got":
        body = (axes + '<text x="100" y="107">ask →</text><path d="M30 92L186 14" style="fill:none;stroke:var(--soft);stroke-dasharray:4 3"/>'
                '<path d="M30 92L110 54L150 48L190 48" style="fill:none;stroke:var(--soft);stroke-width:2"/><path d="M30 92L112 52L152 45L190 44" style="fill:none;stroke:var(--c1);stroke-width:2"/>'
                '<text x="122" y="22">got = ask</text>')
    elif kind == "ratio":
        body = (axes + '<path d="M30 70H192" style="stroke:var(--soft);stroke-dasharray:4 3"/><text x="166" y="66">1 ×</text>'
                '<rect x="56" y="18" width="34" height="74" style="fill:var(--soft)"/><text class="ink" x="57" y="14">2.8 ×</text>'
                '<rect x="118" y="46" width="34" height="46" style="fill:none;stroke:var(--c1);stroke-width:2;stroke-dasharray:4 3"/><text class="ink" x="128" y="42">?</text>'
                '<text x="48" y="107">per-field</text><text x="118" y="107">linear</text>')
    elif kind == "grid":
        cell = lambda x, y, text, tone: (f'<rect x="{x}" y="{y}" width="70" height="34" rx="4" style="fill:{tone};stroke:var(--edge)"/>'
                                         f'<text class="ink" x="{x + 35}" y="{y + 21}" text-anchor="middle">{text}</text>')
        body = ('<text x="62" y="12">per-field rate</text><text x="140" y="12">linear rate</text><text x="2" y="38">per-field</text><text x="2" y="76">linear</text>'
                + cell(52, 18, "low", "color-mix(in srgb,var(--ok) 18%,transparent)") + cell(126, 18, "higher", "none")
                + cell(52, 56, "a bit higher?", "none") + cell(126, 56, "low", "color-mix(in srgb,var(--ok) 18%,transparent)")
                + '<text x="2" y="12">trained ↓</text><text x="52" y="105">scored under →</text>')
    else:
        body = (axes + '<path d="M30 40H192" style="stroke:var(--soft);stroke-dasharray:4 3"/><text x="150" y="36">bar 0.01</text>'
                + "".join(f'<rect x="{44 + i * 30}" y="{92 - h}" width="18" height="{h}" style="fill:var(--c1)"/>' for i, h in enumerate((4, 6, 12, 18, 14)))
                + '<text x="66" y="107">five field types</text>')
    return f'<svg viewBox="0 0 200 110" width="100%" role="img">{body}</svg>'


FRAMES = [
    ("curves", "1. Error against line-ups", "A row is better when its curve sits lower and further right than the control's.",
     "Linear, species-first lies on the control's curve. Linear, random order sits right and higher.", "A linear curve wholly below the control's."),
    ("got", "2. Got against ask", "The linear masking schedule lifts got above 12/24, or it doesn't.",
     "The two lines overlap and stop rising at the same got.", "The linear line keeps rising where the control stops."),
    ("ratio", "3. Random order's error as a multiple of species-first's", "Random order stops failing once training shows every mix of hidden and visible fields.",
     "The multiple falls from 2.8 and stays above 1.", "It stays near 2.8, so the order itself is the problem."),
    ("grid", "4. Validation loss under both masking schedules", "Each model is best under the rate it trained on. How much does the linear model give up on the species-first path?",
     "Linear model lower under the linear rate, slightly higher under the per-field rate.", "The linear model is lower under both."),
    ("bars", "5. How far the choices move when told another t", "The model barely reads t.",
     "Under 0.01 on every field type, as on today's default model.", "0.01 or more: Per-field, corrected t gets its grids."),
]


def storyboard():
    cards = "".join(f"""<div class="box"><h3>{title}</h3>{sketch(kind)}
<p><b style="color:var(--ink)">Reader takes:</b> {takes}</p><p style="margin-top:4px"><b style="color:var(--ok)">Expected:</b> {expected}</p>
<p style="margin-top:4px"><b style="color:var(--warn)">Surprise:</b> {surprise}</p></div>""" for kind, title, takes, expected, surprise in FRAMES)
    body = f"""<h1>Storyboard: the five figures this experiment will draw</h1>
<p class="sub">Sketched before any training or battle, so no frame holds a real number yet. Grey is the control, blue and green are the linear rows.</p>
<div class="grid g2">{cards}
<div class="box"><h3>What feeds each frame</h3><p>Two trainings feed frame 4. Nine grids feed frames 1 to 3. One check on the new per-field model feeds frame 5. A run that fed no frame was dropped: the linear model's readings at epochs 34 and 55.</p></div></div>"""
    return write("storyboard", "Storyboard for the masking-schedule rows", body)


BUILD = [
    ("1. Two training arms", "train84.py:174, :58, :54", "One per-field arm, on the older table. Early stopping uses the 0.001 default.",
     "Per-field and linear arms on the shuffled-only table. Linear is <code>sched=\"1,1,1,1,1\"</code>. Both pass <code>early_stopping_threshold=0.0</code>.", "Hidden share at t is t under the linear rate"),
    ("2. The sampler's t", "diffusion84.py:403", "<code>t = 1.0 - step / COLS</code>, always.",
     "An option for the corrected t. Today's rule stays the default.", "Matches the probe script's corrected t at every pass"),
    ("3. Guidance", "ask_grid84.py:78", "One constant, <code>GUIDANCE = 2.0</code>.", "A setting: 2, 4 and 8, each with its own results file.", "Guidance 2 reproduces a stored cell"),
    ("4. Ask-grid rows", "ask_grid84.py:48, :387", "The probe step always samples species-first.",
     "Four code IDs, each with its own decoder in the probe step and the grid. Cells with no ask.", "Each code ID loads its checkpoint and decoder"),
    ("5. Random order", "remask84.py:246", "<code>RandomDecodingArm</code> exists and was battled on 2026-11-07.", "Wired in as the decoder of Linear, random order, lookahead on.", "The existing lookahead tests"),
]


def build_items():
    cards = "".join(f"""<div class="box"><h3>{title}</h3><span class="mono" style="color:var(--soft)">{where}</span>
<p style="margin-top:8px"><b style="color:var(--soft)">Today:</b> {today}</p><p style="margin-top:4px"><b style="color:var(--ink)">After:</b> {after}</p>
<span class="chip plain">test: {test}</span></div>""" for title, where, today, after, test in BUILD)
    body = f"""<h1>What the code ticket builds</h1>
<p class="sub">Add the linear masking schedule and the corrected t to the 84-field code, with tests. Five build items, each with the line it changes.</p>
<div class="flow" style="margin-bottom:10px">
  {box("Train", "Item 1", "", "train84.py")}<div class="arrow">→</div>{box("Sample", "Items 2, 3 and 5", "", "diffusion84.py, remask84.py")}<div class="arrow">→</div>{box("Battle", "Item 4", "", "ask_grid84.py")}
</div>
<div class="grid g2">{cards}
<div class="box"><h3>Unchanged by default</h3><p>Every new setting defaults to today's behaviour, so stored runs and their checkpoints keep working.</p></div></div>"""
    return write("explain_6_code_ticket", "What the code ticket builds", body)


S, D, N = "same", "diff", "none"
MATRIX = [
    ("Masking schedule", "Set per field type: species t^16 down to Stat Points t", (D, "One rate, linear by default. §6 learns one per token value"), (D, "Linear: every token t"), "tested here"),
    ("Loss weight", "Exponent ÷ t", (S, "w ÷ t, and 1 ÷ t when linear"), (S, "1 ÷ t, Eq. 3, the linear case"), ""),
    ("Dividing the loss by 84", "Yes", (D, "Sums, no division"), (S, "Divides by t × length, Algorithm 1"), "was listed as ours"),
    ("How t is drawn in training", "Evenly spaced across the batch", (S, "Algorithm 1"), (D, "Uniform at random"), ""),
    ("Time input", "The network is told t", (S, "Yes, and says it is optional"), (D, "No t given"), ""),
    ("Slot shuffling", "Random Pokémon order each batch", (N, "Not in the paper"), (N, "Not in the paper"), ""),
    ("Legal filter in the loss", "Forbidden values set to −10,000", (N, "Only the mask token is excluded"), (N, "Not in the paper"), ""),
    ("Legal filter while sampling", "On, at every draw. If no value is legal, every value reopens", (N, "Not in the paper"), (N, "Not in the paper"), ""),
    ("Ask dropout in training", "15% of teams train with no ask", (N, "No guidance at all"), (D, "Never drops the prompt"), ""),
    ("Guidance mixing", "Push toward the ask by a weight", (N, "Left for future work"), (S, "Eq. 16, used only in an appendix test"), ""),
    ("Decoding order", "Fixed: species first", (D, "Random"), (D, "Random, or most confident first"), "tested here"),
    ("t told while sampling", "1 − filled/84, one field per pass", (D, "Its grid time, with the training rate"), (D, "None"), "tested: no effect"),
    ("The draw", "Sampled at temperature 1", (S, "Sampled"), (D, "Takes the most likely value"), ""),
    ("Validation draws", "4 fixed draws for each team", (D, "32 draws, on one benchmark scored by likelihood"), (D, "An exact count hidden. 128 draws, or 1 for one-token answers"), ""),
    ("Stat Points as tokens", "Values 0 to 32, order ignored", (S, "Pixel values as 256 tokens, order ignored"), (N, "Not in the paper"), ""),
]
WORD = {S: "Same", D: "Differs"}


def departures():
    def cell(c):
        kind, detail = c
        return f'<td class="{kind}">{detail}</td>' if kind == N else f'<td class="{kind}"><b>{WORD[kind]}</b><br>{detail}</td>'

    def part(name, tag):
        return f"<td><b>{name}</b><br>{chip(tag, 'plain', 'margin-top:3px')}</td>" if tag else f"<td><b>{name}</b></td>"

    rows = "".join(f"<tr>{part(name, tag)}<td>{ours}</td>{cell(md4)}{cell(llada)}</tr>" for name, ours, md4, llada, tag in MATRIX)
    count = lambda i, kind: sum(row[i][0] == kind for row in MATRIX)
    either = sum(S in (row[2][0], row[3][0]) for row in MATRIX)
    body = f"""<h1>Where our 84-field code matches MD4 and LLaDA, and where it departs</h1>
<p class="sub">{either} of {len(MATRIX)} parts match at least one paper. MD4 matches {count(2, S)}, LLaDA matches {count(3, S)}.</p>
<p><span class="key"><span class="sw same"></span>Same as ours</span><span class="key"><span class="sw diff"></span>Differs from ours</span><span class="key"><span class="sw"></span>Grey text: the paper has no such part</span></p>
<table><tr><th style="width:22%">Part</th><th style="width:26%">Ours</th><th style="width:26%">MD4</th><th style="width:26%">LLaDA</th></tr>{rows}</table>
<p class="note">MD4: Shi et al., NeurIPS 2024, arXiv 2406.04329. LLaDA: Nie et al., NeurIPS 2025, arXiv 2502.09992. Each cell was checked against the paper's LaTeX source on 2026-11-11, and a second pass by three subagents the same day corrected five cells. The ticket's "loss weight divided by 84" row is split in two here, because the two halves have different answers.</p>"""
    return write("explain_7_departures", "Departures matrix", body)


if __name__ == "__main__":
    for page in (probes(), corrected_t(), plan(), tables(), storyboard(), build_items(), departures()): print(page)
