#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Drive figure pages in headless Chrome and report named checks. Run from anywhere inside the repo.

  uv run .claude/skills/verify-figure-pages/check_page.py --doctor
  uv run .claude/skills/verify-figure-pages/check_page.py load PAGE.html [PAGE2.html ...] [--state "item=sash"] [--viewport 800x900] --out DIR
  uv run .claude/skills/verify-figure-pages/check_page.py journey JOURNEY.json --out DIR
  uv run .claude/skills/verify-figure-pages/check_page.py same URL=FILE [URL=FILE ...]

load     Opens each page in every state, at 1440x900 and 390x844, light and dark. It scrolls the whole page first, so
         content that appears on scroll is in its final state, then checks load-errors, load-overflow, load-images
         and load-charts and saves a full-page screenshot. --state adds a query string; give it once per state.
         --viewport replaces the two default sizes; give it once per size.
journey  Replays a reader's actions from a JSON file and checks what the page shows after each one.
same     Checks that a published copy (a URL) has the same bytes as a local file: still-published.

Pages are served from the repo root by a throwaway local server on a random port, closed on exit. Chrome is the
installed Google Chrome, headless, with its own temporary profile. Exit code 1 on any FAIL.
"""
import argparse
import functools
import hashlib
import http.server
import json
import socketserver
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

VIEWPORTS = [(1440, 900), (390, 844)]
THEMES = ["light", "dark"]
MEASURE = """() => {
  const doc = document.documentElement;
  const images = [...document.images].filter(i => i.complete && !i.naturalWidth).map(i => (i.currentSrc || i.src).slice(0, 80));
  const charts = [...document.querySelectorAll('[_echarts_instance_]')];
  const drawn = charts.filter(c => [...c.querySelectorAll('canvas, svg')].some(e => e.getBoundingClientRect().width > 0)).length;
  let small = 99; const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let node;
  while ((node = walk.nextNode())) {
    if (!node.textContent.trim()) continue;
    const style = getComputedStyle(node.parentElement);
    if (style.display === 'none' || style.visibility === 'hidden' || +style.opacity === 0) continue;
    small = Math.min(small, parseFloat(style.fontSize));
  }
  return {overflow: doc.scrollWidth - doc.clientWidth, broken: images, charts: charts.length, drawn,
          wantsCharts: !!document.querySelector('script[src*="echarts"]'), hasCharts: typeof echarts !== 'undefined', smallest: small};
}"""


def repo_root():
    return Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/favicon.ico"):          # no page ships one; a 404 here would read as a page error
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


def serve(root):
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def watch(page):
    """Collect what a reader would never see but a broken page produces: console errors, page errors, failed requests."""
    errors = []
    page.on("console", lambda m: errors.append("console: " + m.text[:200]) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append("page error: " + str(e)[:200]))
    page.on("requestfailed", lambda r: errors.append("request failed: " + r.url[:120]))
    page.on("response", lambda r: errors.append(f"HTTP {r.status}: {r.url[:120]}") if r.status >= 400 else None)
    return errors


def launch(playwright):
    return playwright.chromium.launch(channel="chrome")


def doctor():
    from playwright.sync_api import sync_playwright
    ok = True
    try:
        root = repo_root()
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip()
        dirty = len(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=root).stdout.splitlines())
        print(f"PASS repo root: {root} at {commit}, {dirty} uncommitted paths")
    except Exception as error:
        ok = False
        print(f"FAIL repo root: not inside a git checkout ({error})")
    try:
        with sync_playwright() as p:
            browser = launch(p)
            print(f"PASS chrome: headless Google Chrome {browser.version} starts")
            browser.close()
    except Exception as error:
        ok = False
        print(f"FAIL chrome: the installed Google Chrome did not start ({str(error)[:160]})")
    print("RESULT", "pass" if ok else "fail")
    return ok


def load(args):
    from playwright.sync_api import sync_playwright
    root, out = repo_root(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    server, base = serve(root)
    records, failed = [], 0
    with sync_playwright() as p:
        browser = launch(p)
        for target in args.targets:
            rel = Path(target).resolve().relative_to(root).as_posix()
            for state in args.state or [""]:
                for width, height in ([tuple(map(int, v.lower().split('x'))) for v in args.viewport] if args.viewport else VIEWPORTS):
                    for theme in THEMES:
                        page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme)
                        errors = watch(page)
                        page.goto(base + rel + ("?" + state if state else ""), wait_until="load")
                        total, y = page.evaluate("document.documentElement.scrollHeight"), 0
                        while y < total:                   # a reader's scroll, so scroll-triggered content reaches its final state
                            page.evaluate(f"window.scrollTo(0, {y})")
                            page.wait_for_timeout(120)
                            y += int(height * 0.8)
                        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
                        page.wait_for_timeout(args.settle)
                        page.evaluate("window.scrollTo(0, 0)")
                        m = page.evaluate(MEASURE)
                        name = f"{rel.replace('/', '_').removesuffix('.html')}{'-' + state.replace('&', '-').replace('=', '-') if state else ''}-{width}-{theme}.png"
                        page.screenshot(path=str(out / name), full_page=True)
                        page.close()
                        checks = {
                            "load-errors": (not errors, errors[0] if errors else "none"),
                            "load-overflow": (m["overflow"] <= 0, f"{m['overflow']} px of sideways scroll"),
                            "load-images": (not m["broken"], m["broken"][0] if m["broken"] else "none broken"),
                            "load-charts": (m["drawn"] == m["charts"] and (m["hasCharts"] or not m["wantsCharts"]),
                                            "the chart library did not load" if m["wantsCharts"] and not m["hasCharts"] else f"{m['drawn']} of {m['charts']} drawn"),
                        }
                        bad = [f"{check}: {detail}" for check, (ok, detail) in checks.items() if not ok]
                        failed += bool(bad)
                        records.append(dict(page=rel, state=state, viewport=f"{width}x{height}", theme=theme, screenshot=name, smallest_text_px=m["smallest"],
                                            checks={check: dict(ok=ok, detail=detail) for check, (ok, detail) in checks.items()}))
                        print(f"{'FAIL' if bad else 'PASS'} {rel}{' ?' + state if state else ''} {width}x{height} {theme} | "
                              + ("; ".join(bad) if bad else f"errors 0, sideways scroll 0, broken images 0, charts {m['drawn']}/{m['charts']}, smallest text {m['smallest']:.1f} px"))
        browser.close()
    server.shutdown()
    (out / "load.json").write_text(json.dumps(dict(captures=records), indent=1))
    print(f"RESULT {'fail' if failed else 'pass'} | {len(records)} captures, {failed} failed | {out / 'load.json'}")
    return not failed


def locate(page, handle):
    """A handle is a CSS selector, or {"role": ..., "name": ..., "exact": bool} for an accessible name."""
    if isinstance(handle, str):
        return page.locator(handle)
    return page.get_by_role(handle["role"], name=handle["name"], exact=handle.get("exact", False))


def observe(page, step):
    target = locate(page, step["expect"])
    if "count" in step:
        return target.count(), step["count"]
    if "attr" in step:
        return target.first.get_attribute(step["attr"]), step["value"]
    text = (target.first.text_content() or "").strip() if target.count() else None
    if "contains" in step:
        return (step["contains"] if text and step["contains"] in text else text), step["contains"]
    return text, step["text"]


def journey(args):
    from playwright.sync_api import sync_playwright
    root, out = repo_root(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(args.targets[0]).read_text())
    width, height = map(int, spec.get("viewport", "1440x900").split("x"))
    server, base = serve(root)
    results, failed = [], 0
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=spec.get("theme", "dark"))
        errors = watch(page)
        page.goto(base + spec["page"] + ("?" + spec["query"] if spec.get("query") else ""), wait_until="load")
        page.wait_for_timeout(300)
        for step in spec["steps"]:
            label = step.get("id", "")
            if "scroll" in step:
                locate(page, step["scroll"]).first.evaluate("e => e.scrollIntoView({block: 'center'})")
                results.append(dict(id=label, did=f"scroll to {step['scroll']}"))
            elif "click" in step:
                locate(page, step["click"]).first.click()
                results.append(dict(id=label, did=f"click {step['click']}"))
            elif "press" in step:
                page.keyboard.press(step["press"])
                results.append(dict(id=label, did=f"press {step['press']}"))
            elif "shot" in step:
                page.screenshot(path=str(out / f"{step['shot']}.png"))
                results.append(dict(id=label, did=f"screenshot {step['shot']}.png"))
            elif "expect" in step:
                waited, limit = 0, step.get("within", 2000)
                seen, want = observe(page, step)
                while seen != want and waited < limit:      # numbers roll and bars move; wait for the end state, up to the step's limit
                    page.wait_for_timeout(50)
                    waited += 50
                    seen, want = observe(page, step)
                ok = seen == want
                failed += not ok
                results.append(dict(id=label, expect=step["expect"], want=want, saw=seen, ok=ok, waited_ms=waited))
                print(f"{'PASS' if ok else 'FAIL'} {label} | {step['expect']} shows {want!r}" + ("" if ok else f", saw {seen!r} after {waited} ms"))
        if errors:
            failed += 1
            print(f"FAIL journey-errors | {errors[0]}")
        else:
            print("PASS journey-errors | no console error, page error or failed request during the journey")
        browser.close()
    server.shutdown()
    (out / "journey.json").write_text(json.dumps(dict(journey=str(args.targets[0]), page=spec["page"], steps=results, errors=errors), indent=1))
    print(f"RESULT {'fail' if failed else 'pass'} | {sum('ok' in r for r in results)} expectations, {failed} failed | {out / 'journey.json'}")
    return not failed


def same(args):
    failed = 0
    for pair in args.targets:
        url, _, local = pair.rpartition("=")
        remote = hashlib.sha256(urllib.request.urlopen(url, timeout=30).read()).hexdigest()
        mine = hashlib.sha256(Path(local).read_bytes()).hexdigest()
        failed += remote != mine
        print(f"{'PASS' if remote == mine else 'FAIL'} still-published | {local} {'equals' if remote == mine else 'differs from'} {url[:110]}")
    print("RESULT", "fail" if failed else "pass")
    return not failed


def main():
    if "--doctor" in sys.argv:
        sys.exit(0 if doctor() else 1)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["load", "journey", "same"])
    parser.add_argument("targets", nargs="+", help="pages for load, one journey file for journey, URL=FILE pairs for same")
    parser.add_argument("--state", action="append", help="query string for one page state, such as item=sash; repeat for more states")
    parser.add_argument("--viewport", action="append", help="WxH to load at, in place of 1440x900 and 390x844; repeat for more sizes")
    parser.add_argument("--out", default="verify-evidence/figure-pages/run", help="evidence folder")
    parser.add_argument("--settle", type=int, default=1800, help="ms to let scroll-triggered motion finish before measuring")
    args = parser.parse_args()
    ok = {"load": load, "journey": journey, "same": same}[args.command](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
