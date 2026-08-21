"""Drive verify/water.html in a real, GPU-backed, headFUL Chromium window.

Headless Chromium has no compositor, so requestAnimationFrame never fires and every frame would be
captured mid-nothing; and `file://` is not a secure context, so `navigator.gpu` would not exist at
all. Hence: localhost + a headful window, exactly as verify/shots.py does.

    .venv/Scripts/python.exe runs/.../verify/serve.py            # in one shell
    .venv/Scripts/python.exe runs/.../verify/water.py [jobs...]  # in another
"""
import asyncio
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

RUN = pathlib.Path(__file__).resolve().parents[1]
HERE = RUN / "verify"
PORT = 9346
SERVE = 8742
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def find_browser():
    for c in (EDGE, r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(c):
            return c
    raise SystemExit("no Chromium-family browser found")


def launch(profile, url):
    return subprocess.Popen(
        [find_browser(), "--remote-debugging-port=%d" % PORT, "--user-data-dir=%s" % profile,
         "--no-first-run", "--no-default-browser-check", "--window-size=900,900",
         "--window-position=0,0", "--disable-backgrounding-occluded-windows",
         "--disable-features=CalculateNativeWinOcclusion", "--disable-renderer-backgrounding",
         "--disable-background-timer-throttling",
         # Dawn quantises timestamp-query results (65.536 us granularity was observed here), which
         # is coarser than the entire thing being measured. Turning it off is the difference between
         # a number and a staircase; the amplified slope in water.html cross-checks it either way.
         "--enable-unsafe-webgpu",
         "--disable-dawn-features=timestamp_quantization",
         url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def page_ws(match, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            data = json.loads(urllib.request.urlopen(
                "http://127.0.0.1:%d/json/list" % PORT, timeout=2).read())
            for t in data:
                if t.get("type") == "page" and match in t.get("url", ""):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise SystemExit("could not reach the DevTools endpoint")


async def wait_done(ws, budget):
    """The page sets document.title to CAPDONE / CAPFAIL when it is finished."""
    i = 0
    t0 = time.time()
    last = ""
    while time.time() - t0 < budget:
        i += 1
        await ws.send(json.dumps({"id": i, "method": "Runtime.evaluate", "params": {
            "expression": "[document.title, document.getElementById('log').textContent.slice(-400)]",
            "returnByValue": True}}))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)
        val = msg.get("result", {}).get("result", {}).get("value")
        if not val:
            await asyncio.sleep(1.0)
            continue
        title, log = val
        if log != last:
            tail = log.strip().splitlines()[-1:] or [""]
            print("    ", tail[0][:120], flush=True)
            last = log
        if title in ("CAPDONE", "CAPFAIL"):
            return title, log
        await asyncio.sleep(1.0)
    return "TIMEOUT", last


async def one(url, match, budget=900):
    profile = pathlib.Path(tempfile.gettempdir()) / ("mpm_water_%d" % int(time.time() * 1000))
    shutil.rmtree(profile, ignore_errors=True)
    proc = launch(str(profile), url)
    try:
        async with websockets.connect(page_ws(match), max_size=64 * 1024 * 1024) as ws:
            return await wait_done(ws, budget)
    finally:
        try:
            async with websockets.connect(json.loads(urllib.request.urlopen(
                    "http://127.0.0.1:%d/json/version" % PORT, timeout=3).read()
                    )["webSocketDebuggerUrl"]) as bws:
                await bws.send(json.dumps({"id": 1, "method": "Browser.close"}))
                await asyncio.sleep(1.0)
        except Exception:
            pass
        try:
            proc.terminate(); proc.wait(timeout=8)
        except Exception:
            pass
        shutil.rmtree(profile, ignore_errors=True)


# (label, query string). Cost is measured at three resolutions on purpose: a screen-space cost that
# does not grow with pixels is not being measured.
JOBS = {
    "smoke": ("water.html", "scene=shipped&frames=4&res=600&tag=smoke"),
    "shipped": ("water.html", "scene=shipped&frames=90&res=600&tag=shipped&clip=1&cost=1"),
    "pool": ("water.html", "scene=pool&frames=90&res=600&tag=pool&clip=1"),
    "cost480": ("water.html", "scene=shipped&frames=40&res=480&tag=c480&cost=1&reps=80"),
    "cost720": ("water.html", "scene=shipped&frames=40&res=720&tag=c720&cost=1&reps=80"),
    "cost1080": ("water.html", "scene=shipped&frames=40&res=1080&tag=c1080&cost=1&reps=80"),
    # The ORIGINAL run's captures whose right-hand side is the current build: re-shot, because a
    # clip captioned "the shipped page" that shows water the shipped page no longer draws is a
    # false record. The `_mvp` and `old_` halves are drawn with the previous shading and are
    # untouched by this rework, so they are not re-shot.
    "cap_new_shipped": ("cap.html", "b=new&scene=shipped&frames=90&res=600&t=new"),
    "cap_new_three": ("cap.html", "b=new&scene=three&frames=90&res=600&t=new"),
}


async def main():
    want = sys.argv[1:] or ["smoke"]
    bad = 0
    for w in want:
        page, q = JOBS[w]
        url = "http://localhost:%d/verify/%s?%s" % (SERVE, page, q)
        print("==", w, url, flush=True)
        title, log = await one(url, page)
        print("  ->", title, flush=True)
        if title != "CAPDONE":
            bad += 1
            print(log[-1500:])
        await asyncio.sleep(1.0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
