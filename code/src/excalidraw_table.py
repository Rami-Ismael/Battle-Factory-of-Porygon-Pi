"""Draw a Markdown table as an Excalidraw scene, and export it as a light and a dark PNG.

GitHub renders a Markdown table in one colour. This script turns the table into Excalidraw
elements, one filled rectangle per cell with its text bound inside, so each row carries its own
colour. It writes three files beside each other:

  <name>.excalidraw   the scene, which opens and edits in Excalidraw
  <name>-light.png    Excalidraw's own export
  <name>-dark.png     the same scene exported with Excalidraw's dark mode

Both PNGs carry their own page colour, white and Excalidraw's dark grey, so that a picture stays readable
when a reader's GitHub theme and system theme disagree.

Run:  python src/excalidraw_table.py <markdown file> <output folder> <name> [--table 0] [--widths 220,430,630]

For a design that is not a plain grid, build the element skeletons in Python and call `render_scene`.
`missing_from_scene` then lists any number or word of the table that the scene dropped.

How it works: Excalidraw's library (`@excalidraw/excalidraw` 0.18.0, loaded from esm.sh) lays out the
text and draws the scene inside headless Chrome. One Chrome run needs the network for the library and
its fonts. Chrome is driven over its DevTools socket, which needs the `websockets` package.
"""
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
EXCALIDRAW = "https://esm.sh/@excalidraw/excalidraw@0.18.0"
INK, PAPER, HEADER_FILL = "#1e1e1e", "#ffffff", "#343a40"
# label fill, body tint, bullet colour: the pastel, its lightest shade and its strong shade in Excalidraw's own palette
BLUE, TEAL, ORANGE = ("#a5d8ff", "#e7f5ff", "#1971c2"), ("#96f2d7", "#e6fcf5", "#099268"), ("#ffd8a8", "#fff4e6", "#e8590c")
VIOLET, YELLOW, GREY = ("#d0bfff", "#f3f0ff", "#6741d9"), ("#ffec99", "#fff9db", "#f08c00"), ("#dee2e6", "#f8f9fa", "#495057")
CYCLE = (BLUE, TEAL, ORANGE, VIOLET, YELLOW)
# the three rows of the masking-rate experiment keep the colours their figures use
ROW_COLOURS = (("random", ORANGE), ("control", BLUE), ("per-field", BLUE), ("species-first", TEAL))
BULLET_FROM_SENTENCES = 4                                            # a column turns into bullets when one of its cells has this many sentences
SENTENCE_END = re.compile(r"(?<=[A-Za-z0-9%])\. (?=[A-Z0-9])")


def tables_in(markdown):
    """Every Markdown table in the text, each as (first line index, last line index, header cells, rows of cells)."""
    lines, found, start = markdown.splitlines(), [], None
    for index, line in enumerate([*lines, ""]):
        if line.strip().startswith("|"):
            start = index if start is None else start
            continue
        if start is not None and index - start >= 3 and re.fullmatch(r"[|\s:-]+", lines[start + 1]):
            cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in lines[start:index]]
            found.append((start, index - 1, cells[0], cells[2:]))
        start = None
    return found


def plain(cell):
    """A cell's text without the Markdown marks Excalidraw can't draw."""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cell).replace("**", "").replace("`", "")


def sentences(cell):
    return SENTENCE_END.split(cell)


def colours_of(name, position):
    for word, colours in ROW_COLOURS:
        if word in name.lower(): return colours
    return CYCLE[position % len(CYCLE)]


def spec_of(header, rows, widths=None, body_font_size=20, header_font_size=22):
    """The scene's plan: column widths, the header, and each row's colours and text blocks."""
    header, rows = [plain(cell) for cell in header], [[plain(cell) for cell in row] for row in rows]
    columns = len(header)
    if widths is None:                                               # share 1,280 px by the square root of each column's longest cell
        longest = [max(len(header[c]), *(len(row[c]) for row in rows)) ** 0.5 for c in range(columns)]
        widths = [max(140, round(1280 * share / sum(longest))) for share in longest]
    bulleted = [c > 0 and max(len(sentences(row[c])) for row in rows) >= BULLET_FROM_SENTENCES for c in range(columns)]
    plan = []
    for position, row in enumerate(rows):
        label, tint, strong = colours_of(row[0], position)
        cells = [[dict(text=part, bullet=True) for part in sentences(cell)] if bulleted[c] else [dict(text=cell, bullet=False)]
                 for c, cell in enumerate(row)]
        plan.append(dict(label=label, tint=tint, strong=strong, cells=cells))
    return dict(widths=widths, header=header, rows=plan, body_font_size=body_font_size, header_font_size=header_font_size,
                ink=INK, paper=PAPER, header_fill=HEADER_FILL)


PAGE = """<!doctype html><meta charset="utf-8"><title>working</title><body>
<script type="application/json" id="spec">__SPEC__</script>
<script type="module">
const fail = error => { document.body.insertAdjacentHTML("beforeend", "<pre id='error'></pre>"); document.getElementById("error").textContent = String(error && error.stack || error); document.title = "failed"; };
try {
  const { convertToExcalidrawElements, exportToSvg } = await import("__EXCALIDRAW__");
  const spec = JSON.parse(document.getElementById("spec").textContent);
  const PAD = 14, GAP = 10, DOT = 9, INDENT = 22, HAND = 5;          // HAND is Excalifont, Excalidraw's hand-drawn font
  const widthOf = (column, block) => spec.widths[column] - 2 * PAD - (block.bullet ? INDENT : 0);
  const label = (text, size, align, vertical, colour) => ({ text, fontSize: size, fontFamily: HAND, textAlign: align, verticalAlign: vertical, strokeColor: colour });
  const holder = (id, x, y, width, height, text) => ({ type: "rectangle", id, x, y, width, height, strokeColor: "transparent", backgroundColor: "transparent", roughness: 0, label: text });
  // The fonts must be in the page before Excalidraw measures text, or the text wraps for the wrong font. Excalidraw's export carries the fonts a scene uses, so export once and adopt them.
  const adopt = async elements => {
    const fonts = (await exportToSvg({ elements, appState: { exportBackground: false }, files: {} })).querySelector("style.style-fonts");
    if (fonts) document.head.appendChild(document.importNode(fonts, true));
    await Promise.all([...document.fonts].map(face => face.load().catch(() => null)));
    await document.fonts.ready;
  };
  let skeleton = spec.skeleton;
  if (skeleton) await adopt(convertToExcalidrawElements(skeleton, { regenerateIds: false }));
  else {
  const cellsOf = () => [spec.header.map(text => [{ text, bullet: false }]), ...spec.rows.map(row => row.cells)];

  // Pass 1 measures: every text block sits in a container of its column's width, and Excalidraw grows the container to fit the wrapped text.
  const measure = () => {
    const skeleton = [];
    cellsOf().forEach((row, r) => row.forEach((blocks, c) => blocks.forEach((block, b) =>
      skeleton.push(holder(`m-${r}-${c}-${b}`, 0, 0, widthOf(c, block), 10, label(block.text, r ? spec.body_font_size : spec.header_font_size, r && c ? "left" : "center", "top", spec.ink))))));
    return convertToExcalidrawElements(skeleton, { regenerateIds: false });
  };
  await adopt(measure());
  const measured = Object.fromEntries(measure().filter(element => element.type === "rectangle").map(element => [element.id, element.height]));

  // Pass 2 places: a filled rectangle for each cell, then its text blocks stacked inside, with a dot before each bullet.
  skeleton = [];
  const lefts = spec.widths.map((_, c) => spec.widths.slice(0, c).reduce((sum, width) => sum + width, 0));
  let top = 0;
  cellsOf().forEach((row, r) => {
    const stack = (c, blocks) => blocks.reduce((sum, _, b) => sum + measured[`m-${r}-${c}-${b}`], 0) + GAP * (blocks.length - 1);
    const height = Math.max(...row.map((blocks, c) => stack(c, blocks))) + 2 * PAD;
    const colours = r ? spec.rows[r - 1] : null;
    row.forEach((blocks, c) => {
      const fill = !r ? spec.header_fill : c ? colours.tint : colours.label;
      skeleton.push({ type: "rectangle", id: `cell-${r}-${c}`, x: lefts[c], y: top, width: spec.widths[c], height, strokeColor: spec.ink, backgroundColor: fill, fillStyle: "solid", strokeWidth: 1, roughness: 1 });
      const single = blocks.length === 1 && !blocks[0].bullet;
      let y = top + PAD;
      blocks.forEach((block, b) => {
        const size = r ? spec.body_font_size : spec.header_font_size, own = measured[`m-${r}-${c}-${b}`], x = lefts[c] + PAD + (block.bullet ? INDENT : 0);
        if (block.bullet) skeleton.push({ type: "ellipse", id: `dot-${r}-${c}-${b}`, x: lefts[c] + PAD + 2, y: y + 5 + size * 0.625 - DOT / 2, width: DOT, height: DOT, strokeColor: colours.strong, backgroundColor: colours.strong, fillStyle: "solid", strokeWidth: 1, roughness: 1 });
        skeleton.push(holder(`text-${r}-${c}-${b}`, x, y, widthOf(c, block), single ? height - 2 * PAD : own,
          label(block.text, size, r && c ? "left" : "center", single && !(r && c) ? "middle" : "top", r ? spec.ink : spec.paper)));
        y += own + GAP;
      });
    });
    top += height;
  });
  }
  const elements = convertToExcalidrawElements(skeleton, { regenerateIds: false });
  elements.forEach((element, index) => { element.seed = 7919 + 104729 * index; element.versionNonce = 1 + index; });    // fixed seeds, so the hand-drawn lines are the same on every run
  for (const [theme, dark] of [["light", false], ["dark", true]]) {
    const svg = await exportToSvg({ elements, appState: { exportBackground: spec.background !== false, viewBackgroundColor: "#ffffff", exportWithDarkMode: dark }, files: {}, exportPadding: 12 });
    const box = document.createElement("div"); box.id = theme; box.appendChild(svg); document.body.appendChild(box);
  }
  const scene = { type: "excalidraw", version: 2, source: "excalidraw_table.py", elements, appState: { viewBackgroundColor: "#ffffff", gridSize: null }, files: {} };
  const bytes = new TextEncoder().encode(JSON.stringify(scene));
  let binary = ""; bytes.forEach(byte => { binary += String.fromCharCode(byte); });
  document.body.insertAdjacentHTML("beforeend", `<pre id="scene">${btoa(binary)}</pre>`);
  document.title = "done";
} catch (error) { fail(error); }
</script>"""


class Browser:
    """Headless Chrome, driven over its DevTools socket. `--dump-dom` can't be used: it returns before the library has loaded."""

    def __init__(self, profile):
        self.process = subprocess.Popen([CHROME, "--headless=new", "--hide-scrollbars", "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file, deadline = Path(profile) / "DevToolsActivePort", time.monotonic() + 30
        while not port_file.exists():
            if time.monotonic() > deadline: raise RuntimeError("Chrome did not open its DevTools port")
            time.sleep(0.1)
        port = port_file.read_text().split()[0]
        pages = [target for target in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")) if target["type"] == "page"]
        self.socket, self.sent = connect(pages[0]["webSocketDebuggerUrl"], max_size=None), 0

    def call(self, method, **parameters):
        self.sent += 1
        self.socket.send(json.dumps(dict(id=self.sent, method=method, params=parameters)))
        while True:
            answer = json.loads(self.socket.recv(timeout=120))
            if answer.get("id") == self.sent:
                if "error" in answer: raise RuntimeError(f"{method}: {answer['error']}")
                return answer["result"]

    def value(self, expression):
        return self.call("Runtime.evaluate", expression=expression, returnByValue=True)["result"].get("value")

    def close(self):
        self.socket.close(); self.process.terminate(); self.process.wait(timeout=10)


def render(spec, folder, name, scale=2, wait=120, attempts=3):
    """Draw the spec with Excalidraw and write <name>.excalidraw, <name>-light.png and <name>-dark.png into the folder.
    The library comes over the network, so a failed fetch is tried again."""
    for attempt in range(attempts):
        try:
            return render_once(spec, folder, name, scale, wait)
        except RuntimeError as error:
            if "fetch" not in str(error) or attempt == attempts - 1: raise
            time.sleep(3)
    return None


def render_once(spec, folder, name, scale=2, wait=120):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        page = Path(work) / "table.html"
        page.write_text(PAGE.replace("__SPEC__", json.dumps(spec).replace("</", "<\\/")).replace("__EXCALIDRAW__", EXCALIDRAW), encoding="utf-8")
        browser = Browser(Path(work) / "profile")
        try:
            browser.call("Page.navigate", url=page.as_uri())
            deadline = time.monotonic() + wait
            while (state := browser.value("document.title")) not in ("done", "failed"):
                if time.monotonic() > deadline: raise RuntimeError(f"Excalidraw did not finish in {wait} s; the page says {state!r}")
                time.sleep(0.25)
            if state == "failed": raise RuntimeError(f"Excalidraw did not draw the table: {browser.value('document.getElementById(`error`).textContent')}")
            written = [folder / f"{name}.excalidraw"]
            written[0].write_bytes(base64.b64decode(browser.value("document.getElementById('scene').textContent")))
            browser.call("Emulation.setDefaultBackgroundColorOverride", color=dict(r=0, g=0, b=0, a=0))
            for theme in ("light", "dark"):
                width, height = browser.value(f"""(() => {{ document.body.style.margin = 0;
                    for (const box of document.body.children) box.style.display = box.id === "{theme}" ? "block" : "none";
                    const svg = document.querySelector("#{theme} svg"); svg.style.display = "block";
                    return [Math.ceil(svg.getBoundingClientRect().width), Math.ceil(svg.getBoundingClientRect().height)]; }})()""")
                browser.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=False)
                shot = browser.call("Page.captureScreenshot", format="png", clip=dict(x=0, y=0, width=width, height=height, scale=1))
                written.append(folder / f"{name}-{theme}.png")
                written[-1].write_bytes(base64.b64decode(shot["data"]))
        finally:
            browser.close()
    return written


def render_scene(skeleton, folder, name, scale=2, background=True):
    """Draw a free-form scene. `skeleton` is a list of element skeletons as `convertToExcalidrawElements` takes them:
    rectangle, ellipse, diamond, text, line and arrow dicts, where a shape's `label` is text bound inside it."""
    return render(dict(skeleton=skeleton, background=background), folder, name, scale)


def texts_of(scene_path):
    """Every text a scene shows, unwrapped."""
    return [element.get("originalText") or element["text"] for element in json.loads(Path(scene_path).read_text(encoding="utf-8"))["elements"]
            if element["type"] == "text" and not element.get("isDeleted")]


NUMBER, WORD = re.compile(r"\d[\d,]*(?:\.\d+)?"), re.compile(r"[A-Za-z][A-Za-z'’-]{3,}")


def missing_from_scene(header, rows, scene_path):
    """The numbers and the words of four letters or more that the table has and the scene does not show. An empty list means no cell was dropped."""
    shown = " ".join(texts_of(scene_path))
    shown_numbers, shown_words = set(NUMBER.findall(shown)), {word.lower() for word in WORD.findall(shown)}
    cells = " ".join(plain(cell) for row in [header, *rows] for cell in row)
    return sorted({number for number in NUMBER.findall(cells) if number not in shown_numbers} | {word for word in WORD.findall(cells) if word.lower() not in shown_words})


def main(arguments):
    source, folder, name, *options = arguments
    chosen = dict(zip(options[::2], options[1::2], strict=True))
    _, _, header, rows = tables_in(Path(source).read_text(encoding="utf-8"))[int(chosen.get("--table", 0))]
    widths = [int(width) for width in chosen["--widths"].split(",")] if "--widths" in chosen else None
    for path in render(spec_of(header, rows, widths), folder, name): print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
