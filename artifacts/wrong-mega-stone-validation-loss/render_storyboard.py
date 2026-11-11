# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Render storyboard.html to storyboard-light.png and storyboard-dark.png, the image in issue #258.

    uv run artifacts/wrong-mega-stone-validation-loss/render_storyboard.py

Run build.py first. The board is 760 CSS pixels wide and is captured at twice the pixel density, in the installed
Google Chrome. It also prints the smallest text size and any page error, so a broken render is caught here.
"""
import functools
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
SMALLEST_TEXT = """() => { let m = 99; const w = document.createTreeWalker(document.querySelector('#board'), NodeFilter.SHOW_TEXT); let n;
  while ((n = w.nextNode())) if (n.textContent.trim()) m = Math.min(m, parseFloat(getComputedStyle(n.parentElement).fontSize)); return m; }"""


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(HERE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/storyboard.html"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        for theme in ("light", "dark"):
            page = browser.new_page(viewport={"width": 800, "height": 900}, device_scale_factor=2, color_scheme=theme)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{url}?theme={theme}", wait_until="load")
            page.wait_for_timeout(400)
            board = page.query_selector("#board")
            box = board.bounding_box()
            broken = page.evaluate("[...document.images].filter(i => !i.naturalWidth).length")
            board.screenshot(path=str(HERE / f"storyboard-{theme}.png"))
            print(f"{theme}: {round(box['width'])} x {round(box['height'])} CSS px, smallest text {page.evaluate(SMALLEST_TEXT)} px, "
                  f"broken images {broken}, page errors {errors}")
            page.close()
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
