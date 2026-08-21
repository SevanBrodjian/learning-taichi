"""Screenshot the SHIPPED PAGE, before and after, at four real viewport sizes.

A claim about what fits on a screen cannot be made from CSS -- it is a claim about a rendered page,
so the evidence has to be a rendered page. This drives a real, GPU-backed, headFUL Chromium window
over the DevTools protocol (headless Chromium has no compositor, so requestAnimationFrame never
fires and the simulation would be captured mid-nothing), overrides the device metrics per viewport,
and writes one PNG per (page, viewport).

It also measures, from the live layout rather than from the stylesheet:
  * the field's rendered width and height, and whether they are EQUAL -- the previous layout let the
    square field stretch on any viewport taller than it is wide, which does not crop the simulation,
    it distorts it;
  * what fraction of the viewport the field occupies;
  * the smallest tappable control, because a 29 px button is not a touch target.

    .venv/Scripts/python.exe runs/.../verify/shots.py
"""
import asyncio
import base64
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import urllib.request

RUN = pathlib.Path(__file__).resolve().parents[1]
HERE = RUN / "verify"
SHOTS = HERE / "shots"
PORT = 9344
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

import websockets                                   # noqa: E402

# The throwaway browser profile goes to the SYSTEM TEMP DIR, never inside the run. A Chromium profile
# contains component directories nested deeply enough to exceed Windows' path limit, and if one
# survives the cleanup (it will, whenever the browser is still holding a handle) it is left inside a
# tree that the data server walks -- which takes /api/index and /api/overview to a 500 and the whole
# dashboard with them. Nothing that is not an artifact belongs under runs/.

# (label, css width, css height, dpr). The two phone entries are the same device in both
# orientations, because portrait and landscape are different layout problems.
VIEWPORTS = [
    ("phone_portrait", 390, 844, 3),
    ("phone_landscape", 844, 390, 3),
    ("tablet_portrait", 820, 1180, 2),
    ("laptop", 1280, 800, 2),
    ("desktop", 1920, 1080, 1),
]
PAGES = [("before", "http://localhost:8742/old/demo.html"),
         ("after", "http://localhost:8742/web/demo.html")]


def find_browser():
    for c in (EDGE, r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(c):
            return c
    raise SystemExit("no Chromium-family browser found")


def launch(profile, url):
    return subprocess.Popen(
        [find_browser(), "--remote-debugging-port=%d" % PORT, "--user-data-dir=%s" % profile,
         "--no-first-run", "--no-default-browser-check", "--window-size=1400,1000",
         "--window-position=0,0", "--disable-backgrounding-occluded-windows",
         "--disable-features=CalculateNativeWinOcclusion", "--disable-renderer-backgrounding",
         "--disable-background-timer-throttling", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def page_ws(match, timeout=25):
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


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.i = 0
        self.pending = {}
        self.reader = asyncio.create_task(self._pump())

    async def _pump(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self.pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
        except Exception:
            pass

    async def call(self, method, **params):
        self.i += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.i] = fut
        await self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        msg = await asyncio.wait_for(fut, timeout=60)
        if "error" in msg:
            raise RuntimeError("%s: %s" % (method, msg["error"]))
        return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression="(async()=>{%s})()" % expr,
                            awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:400])
        return r["result"].get("value")

    async def shot(self, path):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path.write_bytes(base64.b64decode(r["data"]))


MEASURE = """
const f = document.querySelector('.frame').getBoundingClientRect();
const c = document.querySelector('canvas');
const cr = c.getBoundingClientRect();
const btns = [...document.querySelectorAll('.mpm4 button')].map(b => b.getBoundingClientRect());
const ctl = document.querySelector('.ctl').getBoundingClientRect();
return {
  vw: innerWidth, vh: innerHeight, dpr: devicePixelRatio,
  field_w: Math.round(f.width), field_h: Math.round(f.height),
  field_square: Math.abs(f.width - f.height) < 1.5,
  aspect_error_pct: 100 * Math.abs(f.width - f.height) / Math.max(f.width, f.height),
  field_frac_of_viewport: (f.width * f.height) / (innerWidth * innerHeight),
  canvas_backing: [c.width, c.height],
  canvas_css: [Math.round(cr.width), Math.round(cr.height)],
  controls_h: Math.round(ctl.height), controls_w: Math.round(ctl.width),
  min_button_h: btns.length ? Math.round(Math.min(...btns.map(b => b.height))) : null,
  n_buttons: btns.length,
  doc_overflow_x: document.documentElement.scrollWidth - innerWidth,
  n: (window.DEMO && window.DEMO.sim) ? window.DEMO.sim.n : null,
};
"""


async def run():
    SHOTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, url in PAGES:
        profile = pathlib.Path(tempfile.gettempdir()) / ("mpm_shotprof_%d" % int(time.time() * 1000))
        shutil.rmtree(profile, ignore_errors=True)
        proc = launch(str(profile), url)
        try:
            async with websockets.connect(page_ws("demo.html"), max_size=64 * 1024 * 1024) as ws:
                c = CDP(ws)
                await c.call("Page.enable")
                await c.call("Runtime.enable")
                await asyncio.sleep(2.0)
                boot = await c.js("const d = await window.DEMO.ready;"
                                  "return { gpu: !!navigator.gpu, ver: MPM4.PARAMS.physics_version,"
                                  " treat: MPM4.RENDER_TREATMENT || 'mvp' };")
                print(tag, boot)
                if not boot or not boot.get("gpu"):
                    raise SystemExit("no WebGPU in this window -- the shots would be meaningless")
                for name, w, h, dpr in VIEWPORTS:
                    await c.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                                 deviceScaleFactor=dpr, mobile=w < 500)
                    await asyncio.sleep(1.6)          # let ResizeObserver + a few frames land
                    m = await c.js(MEASURE)
                    out["%s_%s" % (tag, name)] = dict(m, page=tag, viewport=name,
                                                      physics_version=boot["ver"],
                                                      treatment=boot["treat"])
                    await c.shot(SHOTS / ("%s_%s.png" % (tag, name)))
                    print("  %-16s field %sx%s square=%s  ctl %spx  minbtn %spx  overflowX %s"
                          % (name, m["field_w"], m["field_h"], m["field_square"],
                             m["controls_h"], m["min_button_h"], m["doc_overflow_x"]))
                await c.call("Emulation.clearDeviceMetricsOverride")
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
            subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq MATTER*"], capture_output=True)
            shutil.rmtree(profile, ignore_errors=True)
        await asyncio.sleep(1.0)
    (HERE / "layout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", HERE / "layout.json")


if __name__ == "__main__":
    asyncio.run(run())
