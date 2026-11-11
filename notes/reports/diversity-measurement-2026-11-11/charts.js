// Charts of the 2026-11-11 diversity page. Reads window.D (embedded by build.py), draws with ECharts 6.1.0,
// follows the light or dark setting, and redraws on resize.
(function () {
  const D = window.D;
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const isDark = () => {
    const t = document.documentElement.dataset.theme;
    return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const fmt = v => v == null || Number.isNaN(v) ? "n/a" : Math.abs(v) >= 100 ? Math.round(v).toLocaleString("en-US")
    : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
  const pct = v => (100 * v).toFixed(1) + "%";
  const row = k => D.rows.find(r => r.key === k);
  const avg = (keys, m) => keys.reduce((a, k) => a + row(k)[m], 0) / keys.length;
  const S = [1, 2, 3];
  const MODEL = S.map(s => "model_s" + s), TRAIN = S.map(s => "train_s" + s), TALL = S.map(s => "train_all_s" + s);
  const charts = [];

  function base() {
    return {
      backgroundColor: "transparent", animation: false,
      textStyle: {color: css("--ink2"), fontFamily: "ui-sans-serif, system-ui, -apple-system, sans-serif"},
      grid: {left: 8, right: 24, top: 36, bottom: 28, containLabel: true},
      legend: {top: 0, left: 0, textStyle: {color: css("--ink2")}, itemWidth: 12, itemHeight: 10},
    };
  }
  const axisStyle = () => ({axisLine: {lineStyle: {color: css("--rule")}}, splitLine: {lineStyle: {color: css("--grid")}},
    axisLabel: {color: css("--ink2")}});

  // Figure 1: model value / training value per measure, with the noise floor band.
  function ratioOption() {
    const R = D.ratio.slice().reverse();
    const names = R.map(r => r.level + " · " + r.name);
    const band = R.map((r, i) => [i, Math.max(r.lo, 0.3), Math.min(r.hi, 3.5)]);
    const seed = s => R.map((r, i) => [Math.min(Math.max(r.model[s] / r.train[s], 0.3), 3.5), i]);
    const tall = R.map((r, i) => [Math.min(Math.max(r.train_all.reduce((a, b) => a + b) / 3 / r.tmean, 0.3), 3.5), i]);
    return {
      ...base(),
      tooltip: {trigger: "axis", axisPointer: {type: "shadow"}, confine: true, formatter: ps => {
        const r = R[ps[0].dataIndex];
        return `<b>${r.level} · ${r.name}</b><br>Default model: ${r.model.map(fmt).join(", ")}<br>` +
          `Training teams matched to the asks: ${r.train.map(fmt).join(", ")}<br>` +
          `Training teams drawn over all win rates: ${r.train_all.map(fmt).join(", ")}<br>` +
          `Validation ${fmt(r.val)} · one team repeated ${fmt(r.copied)} · random legal ${fmt(r.random)}<br>` +
          `Differs beyond the noise floor: ${r.differs ? "yes" : "no"}`;
      }},
      grid: {left: 8, right: 24, top: 36, bottom: 36, containLabel: true},
      xAxis: {type: "log", min: 0.3, max: 3.5, name: "model ÷ training, matched to the asks", nameLocation: "middle", nameGap: 24,
        ...axisStyle()},
      yAxis: {type: "category", data: names, ...axisStyle(), axisLabel: {color: css("--ink"), fontSize: 11}},
      series: [
        {name: "noise floor", type: "custom", renderItem: (params, api) => {
          const a = api.coord([api.value(1), api.value(0)]), b = api.coord([api.value(2), api.value(0)]);
          const h = api.size([0, 1])[1] * 0.55;
          return {type: "rect", shape: {x: a[0], y: a[1] - h / 2, width: Math.max(2, b[0] - a[0]), height: h, r: 3},
            style: {fill: css("--band")}};
        }, encode: {x: [1, 2], y: 0}, data: band, z: 1, itemStyle: {color: css("--band")}},
        ...[0, 1, 2].map(s => ({name: s === 0 ? "default model, one dot per sampling seed" : "default model, seed " + (s + 1),
          type: "scatter", data: seed(s), symbolSize: 9, itemStyle: {color: css("--model")}, z: 3, emphasis: {focus: "series"}})),
        {name: "training teams drawn over all win rates, mean of 3 draws", type: "scatter", data: tall, symbol: "diamond",
          symbolSize: 10, itemStyle: {color: "transparent", borderColor: css("--trainall"), borderWidth: 2}, z: 4},
        {name: "ratio 1", type: "line", data: [], markLine: {silent: true, symbol: "none", label: {show: false},
          lineStyle: {color: css("--ink3"), type: "solid"}, data: [{xAxis: 1}]}},
      ],
      legend: {...base().legend, data: ["default model, one dot per sampling seed", "training teams drawn over all win rates, mean of 3 draws", "noise floor"]},
    };
  }

  // Horizontal bars, one category per sample row.
  function barRows(keys, series, xName, opts = {}) {
    const names = keys.map(k => row(k).name).reverse();
    return {
      ...base(),
      tooltip: {trigger: "axis", axisPointer: {type: "shadow"}, confine: true, valueFormatter: opts.fmt || fmt},
      xAxis: {type: "value", name: xName, nameLocation: "middle", nameGap: 26, max: opts.max, ...axisStyle(),
        axisLabel: {color: css("--ink2"), formatter: opts.axisFmt}},
      yAxis: {type: "category", data: names, ...axisStyle(), axisLabel: {color: css("--ink"), fontSize: 11}},
      grid: {left: 8, right: 24, top: opts.top || 36, bottom: 40, containLabel: true},
      series: series.map(s => ({type: "bar", barMaxWidth: 16, emphasis: {focus: "series"}, ...s,
        data: keys.map(k => s.value(row(k))).reverse()})),
    };
  }
  const rowColor = k => k.startsWith("model") ? css("--model") : k.startsWith("train_all") ? css("--trainall")
    : k.startsWith("train") ? css("--train") : k === "val" ? css("--val") : css("--anchor");

  function buildsOption() {
    const keys = D.rows.map(r => r.key);
    return barRows(keys, [{name: "builds per species at " + D.equal_uses + " uses", value: r => ({value: r.builds_per_species,
      itemStyle: {color: rowColor(r.key)}})}], "distinct builds among " + D.equal_uses + " uses of one species, mean over species");
  }

  function noveltyOption() {
    const keys = D.rows.map(r => r.key).filter(k => k !== "random" || true);
    const parts = [["novelty_swaps_0", "zero swaps, a training line-up copied", "--s0"], ["novelty_swaps_1", "one swap", "--s1"],
      ["novelty_swaps_2", "two swaps", "--s2"], ["novelty_swaps_3plus", "three or more, a new line-up", "--s3"]];
    return barRows(keys, parts.map(([m, name, c]) => ({name, stack: "n", value: r => r[m], itemStyle: {color: css(c)}})),
      "share of teams", {max: 1, fmt: pct, axisFmt: v => Math.round(v * 100) + "%", top: 56});
  }

  function nearestOption() {
    const keys = D.rows.map(r => r.key);
    const parts = [["nearest_share_copied_zero_fields", "0 fields, a training team copied", "--s0"],
      ["nearest_share_one_to_five_fields", "1 to 5 fields", "--s1"], ["nearest_share_six_to_twenty_fields", "6 to 20 fields", "--s2"],
      ["nearest_share_over_twenty_fields", "over 20 fields", "--s3"]];
    return barRows(keys, parts.map(([m, name, c]) => ({name, stack: "d", value: r => r[m], itemStyle: {color: css(c)}})),
      "share of teams", {max: 1, fmt: pct, axisFmt: v => Math.round(v * 100) + "%", top: 56});
  }

  function archetypeOption() {
    const groups = [["default model, mean of 3 seeds", MODEL, "--model"], ["training teams matched to the asks, mean of 3 draws", TRAIN, "--train"],
      ["training teams drawn over all win rates, mean of 3 draws", TALL, "--trainall"], ["validation teams", ["val"], "--val"],
      ["random legal teams", ["random"], "--anchor"]];
    const A = D.archetypes.slice().reverse();
    return {
      ...base(), legend: {...base().legend, type: "plain"},
      tooltip: {trigger: "axis", axisPointer: {type: "shadow"}, confine: true, valueFormatter: pct},
      grid: {left: 8, right: 24, top: 76, bottom: 40, containLabel: true},
      xAxis: {type: "value", name: "share of teams in the archetype", nameLocation: "middle", nameGap: 26, ...axisStyle(),
        axisLabel: {color: css("--ink2"), formatter: v => Math.round(v * 100) + "%"}},
      yAxis: {type: "category", data: A, ...axisStyle(), axisLabel: {color: css("--ink"), fontSize: 12}},
      series: groups.map(([name, keys, c]) => ({name, type: "bar", barMaxWidth: 8, emphasis: {focus: "series"},
        itemStyle: {color: css(c)}, data: A.map(a => avg(keys, "archetype_" + a))})),
    };
  }

  function anchorsOption() {
    const levels = [["Line-up", "vendi_lineup"], ["Builds", "vendi_sets"], ["Stat Points", "vendi_statpoints"], ["Whole team", "vendi_team"]];
    const pts = [["one team repeated", ["copied"], "--anchor", "circle"], ["random legal teams", ["random"], "--anchor", "rect"],
      ["validation teams", ["val"], "--val", "triangle"], ["training teams matched to the asks", TRAIN, "--train", "circle"],
      ["training teams drawn over all win rates", TALL, "--trainall", "diamond"], ["default model", MODEL, "--model", "circle"]];
    return {
      ...base(),
      tooltip: {trigger: "axis", axisPointer: {type: "shadow"}, confine: true, valueFormatter: fmt},
      grid: {left: 8, right: 24, top: 56, bottom: 40, containLabel: true},
      xAxis: {type: "log", min: 1, max: D.n, name: "Vendi Score: effective number of different teams, log scale", nameLocation: "middle",
        nameGap: 26, ...axisStyle()},
      yAxis: {type: "category", data: levels.map(l => l[0]).reverse(), ...axisStyle(), axisLabel: {color: css("--ink"), fontWeight: 600}},
      series: pts.map(([name, keys, c, sym]) => ({name, type: "scatter", symbol: sym, symbolSize: 11, emphasis: {focus: "series"},
        itemStyle: {color: css(c), borderColor: css("--bg"), borderWidth: 1},
        data: levels.slice().reverse().map(([, m]) => avg(keys, m))})),
    };
  }

  const specs = [["ratio", ratioOption], ["builds", buildsOption], ["novelty", noveltyOption], ["nearest", nearestOption],
    ["archetypes", archetypeOption], ["anchors", anchorsOption]];
  function draw() {
    charts.forEach(c => c.dispose()); charts.length = 0;
    for (const [id, make] of specs) {
      const el = document.getElementById(id); if (!el) continue;
      const c = echarts.init(el, isDark() ? "dark" : null, {renderer: "svg"});
      c.setOption(make()); charts.push(c);
    }
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
  addEventListener("resize", () => charts.forEach(c => c.resize()));
  draw();
})();
