"""What a chart says, independent of the library that draws it: its words and its data points.

Used by scripts/test_chart_ports.py to move charts from matplotlib to ECharts one script at a time.
A golden record is taken from the matplotlib figure before a port; the ported ECharts option must
still carry every golden word and every golden data point.

- Words: figure and axes titles, axis labels, text and annotations, legend entries. Tick labels are left out,
  because ECharts picks its own ticks.
- Points: every (x, y) drawn in data coordinates: line vertices, scatter positions, bar tops, error-bar ends and
  shaded-band edges. Reference lines drawn in axes coordinates (axhline, axvline, axvspan) are decoration and left out.

This module never imports matplotlib; it reads the figure object it is handed.
"""
import math

DIGITS = 6


def _num(v):
    v = float(v)
    return None if not math.isfinite(v) else round(v, DIGITS) + 0.0


def _pairs(xs, ys):
    out = set()
    for x, y in zip(xs, ys):
        p = (_num(x), _num(y))
        if None not in p:
            out.add(p)
    return out


def _text(t):
    s = t.get_text().strip() if hasattr(t, "get_text") else str(t).strip()
    return s


def _figure_texts(fig):
    texts = [fig._suptitle] if getattr(fig, "_suptitle", None) else []
    texts += list(fig.texts)
    for legend in getattr(fig, "legends", []):
        texts += legend.get_texts()
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        if ax.get_legend():
            texts += ax.get_legend().get_texts()
    return {s for s in map(_text, texts) if s}


def _line_points(ax):
    return set().union(*[_pairs(line.get_xdata(orig=False), line.get_ydata(orig=False))
                         for line in ax.lines if line.get_transform() == ax.transData])


def _collection_points(ax):
    pts = set()
    for coll in ax.collections:
        if coll.get_transform() == ax.transData:
            for path in coll.get_paths():  # error bars, shaded bands
                pts |= _pairs(path.vertices[:, 0], path.vertices[:, 1])
        off = coll.get_offsets()
        if coll.get_offset_transform() == ax.transData and len(off):  # scatter markers
            pts |= _pairs(off[:, 0], off[:, 1])
    return pts


def _bar_points(ax):
    pts = set()
    for container in ax.containers:
        if type(container).__name__ != "BarContainer":
            continue
        horizontal = getattr(container, "orientation", "vertical") == "horizontal"
        for bar in container.patches:
            x, y, w, h = bar.get_x(), bar.get_y(), bar.get_width(), bar.get_height()
            pts |= _pairs([x + w if horizontal else x + w / 2], [y + h / 2 if horizontal else y + h])
    return pts


def _axes_points(ax):
    return _line_points(ax) | _collection_points(ax) | _bar_points(ax)


def record_figure(fig):
    """Record a matplotlib figure as {"texts": [...], "points": [[x, y], ...]}."""
    pts = set()
    for ax in fig.axes:
        pts |= _axes_points(ax)
    return {"texts": sorted(_figure_texts(fig)), "points": sorted([list(p) for p in pts])}


TEXT_KEYS = {"name", "text", "subtext", "formatter"}


def _option_texts(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in TEXT_KEYS and isinstance(v, str):
                out.add(v.strip())
            elif k == "data" and isinstance(v, list) and all(isinstance(s, str) for s in v):
                out |= {s.strip() for s in v}  # legend and category names
            else:
                _option_texts(v, out)
    elif isinstance(node, list):
        for v in node:
            _option_texts(v, out)
    return out


def _category_axes(option):
    def kinds(axes):
        axes = axes if isinstance(axes, list) else [axes or {}]
        return [a.get("type") == "category" for a in axes]
    return kinds(option.get("xAxis")), kinds(option.get("yAxis"))


def _series_points(series, ycat):
    pts = set()
    for i, d in enumerate(series.get("data", [])):
        v = d.get("value") if isinstance(d, dict) else d
        if isinstance(v, (int, float)):
            pair = (v, i) if ycat else (i, v)
            pts |= _pairs([pair[0]], [pair[1]])
        elif isinstance(v, list) and len(v) >= 2 and all(isinstance(n, (int, float)) for n in v):
            pts |= _pairs([v[0]] * (len(v) - 1), v[1:])  # [x, y] or [x, low, high]
    return pts


def record_option(option, texts=()):
    """Record an ECharts option (plus page title, subtitle and notes) in the same shape as record_figure."""
    _, ycat = _category_axes(option)
    pts = set()
    for s in option.get("series", []):
        pts |= _series_points(s, ycat[s.get("yAxisIndex", 0)] if ycat else False)
    words = _option_texts(option, set()) | {t.strip() for t in texts if t and t.strip()}
    return {"texts": sorted(w for w in words if w), "points": sorted([list(p) for p in pts])}


def missing(golden, record, digits=DIGITS - 1):
    """Golden words and points the record does not carry, as two lists. Empty lists mean the port says the same.
    Points match after rounding both sides to `digits` decimals."""
    words = sorted(set(golden["texts"]) - set(record["texts"]))
    have = {(round(x, digits), round(y, digits)) for x, y in record["points"]}
    lost = [p for p in golden["points"] if (round(p[0], digits), round(p[1], digits)) not in have]
    return words, lost
