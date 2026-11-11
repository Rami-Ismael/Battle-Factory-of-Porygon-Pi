// Render each ECharts option written by build_figures.py to a static SVG with ECharts server-side rendering.
//   NODE_PATH=<dir with echarts@6.1.0 in node_modules> node render_svg.mjs OPTIONS_DIR figures
// Text stays as SVG text. Install echarts outside the repo, for example in a scratch directory.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(path.join(process.env.NODE_PATH || process.cwd(), "x.js"));
const echarts = require("echarts");
const [src, dst] = process.argv.slice(2);
fs.mkdirSync(dst, { recursive: true });
for (const f of fs.readdirSync(src).filter(f => f.endsWith(".json") && f !== "shown_numbers.json")) {
  const option = JSON.parse(fs.readFileSync(path.join(src, f), "utf8"));
  const { width, height } = option;
  delete option.width; delete option.height;
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width, height });
  chart.setOption(option);
  const svg = chart.renderToSVGString();
  chart.dispose();
  fs.writeFileSync(path.join(dst, f.replace(/\.json$/, ".svg")), svg);
  console.log("wrote", path.join(dst, f.replace(/\.json$/, ".svg")));
}
