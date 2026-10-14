// Runs the finder page's own <script> in Node against a stub DOM, then evaluates expressions in that scope.
// Usage: node no_kingambit_finder_harness.js <page.html> '<json array of expressions>'
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const exprs = JSON.parse(process.argv[3] || "[]");
const m = html.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
if (!m) { console.log(JSON.stringify({error: "no script block"})); process.exit(1); }

function stub() {   // an object that accepts any property, call or construction
  const f = function () {};
  const text = new Set(["value", "textContent", "innerHTML", "innerText", "className", "id", "title", "href", "type", "name"]);
  return new Proxy(f, {
    get: (t, k) => {
      if (typeof k === "symbol") return k === Symbol.iterator ? function* () {} : k === Symbol.toPrimitive ? () => "" : undefined;
      if (k === "length") return 0;
      if (k === "then") return undefined;
      if (text.has(k)) return "";
      if (k === "dataset") return {};
      return stub();
    },
    set: () => true, apply: () => stub(), construct: () => stub(),
  });
}
const ctx = vm.createContext({
  document: stub(), navigator: stub(), console, setTimeout, clearTimeout, URL, URLSearchParams, Math, JSON, Set, Map, Object, Array,
  matchMedia: () => ({matches: false, addEventListener() {}}), getComputedStyle: () => stub(),
  location: {search: "", href: "http://localhost/page.html", protocol: "http:"}, history: {replaceState() {}},
  addEventListener() {}, requestAnimationFrame: f => 0, cancelAnimationFrame() {}, innerWidth: 1180, innerHeight: 820, scrollY: 0,
  ResizeObserver: class { observe() {} disconnect() {} }, MutationObserver: class { observe() {} disconnect() {} },
  IntersectionObserver: class { observe() {} disconnect() {} }, PointerEvent: class {}, MouseEvent: class {}, Event: class {},
  CSS: {escape: s => String(s)}, performance: {now: () => 0}, getSelection: () => stub(), CustomEvent: class {}, KeyboardEvent: class {},
});
ctx.window = ctx;
try { new vm.Script(m[1]).runInContext(ctx); }
catch (e) { console.log(JSON.stringify({error: "script threw: " + e.message + " @ " + String(e.stack).split("\n").slice(0,40).join(" | ")})); process.exit(1); }
const out = exprs.map(x => { try { return vm.runInContext(x, ctx); } catch (e) { return "ERR " + e.message; } });
console.log(JSON.stringify({ok: true, out}));
