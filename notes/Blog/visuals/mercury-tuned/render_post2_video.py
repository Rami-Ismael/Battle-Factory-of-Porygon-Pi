"""Render post2-mercury-tuned.html to post2-mercury-tuned.mp4 (1080x1350, 60 fps) and post2-mercury-tuned-final.png.

Each frame is drawn at an exact time through the page's own render(), so the video matches what ?t= shows.
Uses the installed Chrome, so Playwright downloads no browser.

    uv run --with playwright python notes/Blog/visuals/mercury-tuned/render_post2_video.py
"""
import base64
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PAGE = HERE / "post2-mercury-tuned.html"
FPS = 60
GRAB = "s => { t = s; playing = false; render(); return document.getElementById('stage').toDataURL('image/png'); }"


def png(uri):
    return base64.b64decode(uri.split(",", 1)[1])


def main():
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1080, "height": 1350})
        page.goto(PAGE.as_uri() + "?capture=1&t=0")
        page.wait_for_function("Object.values(imgs).length > 0 && Object.values(imgs).every(im => im.complete)")
        seconds = page.evaluate("T")
        for i in range(round(FPS * seconds) + 1):
            Path(tmp, f"f{i:05d}.png").write_bytes(png(page.evaluate(GRAB, i / FPS)))
        (HERE / "post2-mercury-tuned-final.png").write_bytes(png(page.evaluate(GRAB, seconds)))
        browser.close()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                        str(HERE / "post2-mercury-tuned.mp4")], check=True)
    print("wrote post2-mercury-tuned.mp4 and post2-mercury-tuned-final.png")


if __name__ == "__main__":
    main()
