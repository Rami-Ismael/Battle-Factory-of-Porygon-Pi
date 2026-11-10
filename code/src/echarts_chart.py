"""Write an ECharts chart as a standalone HTML page. ECharts is the project's default chart library (AGENTS.md, Charts).

    from echarts_chart import write_chart
    write_chart({"xAxis": {...}, "yAxis": {...}, "series": [...]}, out / "figure.html", title="...")

The option is the dict documented at https://echarts.apache.org/en/option.html, passed through as JSON, so
formatters are ECharts template strings such as "{b}: {c}". A string that starts with "js:" becomes a JavaScript
function in the page, for what templates cannot say: {"formatter": "js:v => Math.round(v * 100) + '%'"}.
The page follows the reader's light or dark setting with ECharts' built-in themes and redraws on resize.
In dark mode, colors listed in DARK (plus any `dark` mapping passed in) are swapped for their dark-surface partners,
so greys and deep blues stay readable.
"""
import html
import json
from pathlib import Path

ECHARTS = "https://cdn.jsdelivr.net/npm/echarts@6.1.0/dist/echarts.min.js"
# Light-surface color -> dark-surface partner, applied to every string in the option when the page is dark.
DARK = {"#0b0b0b": "#f2f1ee", "#1f1f1e": "#f2f1ee", "#52514e": "#c3c2b7", "#5f5e5a": "#c3c2b7", "#6f6e69": "#a9a8a2", "#fff": "#161615", "#ffffff": "#161615"}
# When a list, write_chart appends (path, option, texts, keyword arguments) here instead of writing; scripts/test_chart_ports.py sets it.
CAPTURE = None

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--ink:#f2f1ee;--ink2:#a9a8a2;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Helvetica,Arial,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:24px 16px}}
h1{{font-size:20px;line-height:1.25;margin:0}}
h1,.sub,.note{{white-space:pre-line}}
.sub,.note{{color:var(--ink2);margin:4px 0 12px}}
.note{{font-size:12.5px;margin-top:10px}}
#chart{{width:100%;height:{height}px}}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div id="chart"></div>
<p class="note">{note}</p>
</main>
<script src="{echarts}"></script>
<script>
const revive = o => typeof o === "string" && o.startsWith("js:") ? (0, eval)("(" + o.slice(3) + ")")
  : Array.isArray(o) ? o.map(revive) : o && typeof o === "object" ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, revive(v)])) : o;
const option = revive({option}), darkMap = {dark};
const swap = o => typeof o === "string" ? (darkMap[o.toLowerCase()] || o) : Array.isArray(o) ? o.map(swap)
  : o && typeof o === "object" && typeof o !== "function" ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, swap(v)])) : o;
const el = document.getElementById("chart"), dark = matchMedia("(prefers-color-scheme: dark)");
let chart;
function draw() {{
  if (chart) chart.dispose();
  chart = echarts.init(el, dark.matches ? "dark" : null, {{renderer: "svg"}});
  chart.setOption({{backgroundColor: "transparent", ...(dark.matches ? swap(option) : option)}});
}}
dark.addEventListener("change", draw);
addEventListener("resize", () => chart.resize());
draw();
</script>
</body>
</html>
"""


def write_chart(option: dict, path, title: str, subtitle: str = "", note: str = "", height: int = 420, dark: dict | None = None) -> Path:
    """Write the page to path and return it. Title, subtitle and note are plain text.
    dark maps extra light-surface colors to the colors to use on the dark surface."""
    path = Path(path)
    if CAPTURE is not None:
        CAPTURE.append((path, option, [title, subtitle, note], dict(height=height, dark=dark)))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    esc = html.escape
    path.write_text(PAGE.format(title=esc(title), subtitle=esc(subtitle), note=esc(note), height=int(height),
                                echarts=ECHARTS, option=json.dumps(option, ensure_ascii=False).replace("</", "<\\/"),
                                dark=json.dumps({k.lower(): v for k, v in {**DARK, **(dark or {})}.items()})))
    return path


def band(name, color, xs, lo, hi, stack, opacity=0.13, **axes):
    """A shaded interval between lo and hi, as three line series: the two thin edges and a stacked fill between them.
    Give each band its own stack id; pass xAxisIndex / yAxisIndex for a chart with several grids."""
    xs, lo, hi = list(map(float, xs)), list(map(float, lo)), list(map(float, hi))
    common = {"type": "line", "name": name, "color": color, "symbol": "none", "silent": True, "tooltip": {"show": False}, **axes}
    edge = {"width": 0.6, "opacity": 0.5}
    return [{**common, "lineStyle": edge, "stack": stack, "data": [list(p) for p in zip(xs, lo)]},
            {**common, "lineStyle": {"width": 0}, "stack": stack, "areaStyle": {"opacity": opacity},
             "data": [[x, h - lw] for x, lw, h in zip(xs, lo, hi)]},
            {**common, "lineStyle": edge, "data": [list(p) for p in zip(xs, hi)]}]
