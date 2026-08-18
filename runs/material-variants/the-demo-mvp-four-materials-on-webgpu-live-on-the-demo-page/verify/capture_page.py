"""Capture the SHIPPED page -- chrome, HUD and all -- while a real GPU drives it.

Why this is not a screenshot of a canvas: the deliverable is a page, so the evidence has to be the
page. It is driven through the Chrome DevTools Protocol against a real (headful, off-screen) Edge
window, because:

  * headless Chromium does not expose WebGPU, so a `--screenshot` run only ever captures the
    graceful-degradation path (which is worth capturing too, and is captured, as fallback.png);
  * `requestAnimationFrame` does not fire in a window that is not compositing, and it SHOULD not --
    a visitor's battery depends on that -- so the page must really be on a GPU-backed surface;
  * `Input.dispatchMouseEvent` delivers genuine pointer events, so pouring, grabbing and erasing are
    exercised exactly the way a visitor exercises them, not by calling internal functions.

Writes demo_capture.mp4 (real time) + demo_page.png / demo_views.png / fallback.png.

    .venv/Scripts/python.exe runs/.../verify/capture_page.py
"""
import asyncio
import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

RUN = pathlib.Path(__file__).resolve().parents[1]
HERE = RUN / "verify"
SHOTS = HERE / "shots"
PORT = 9333
URL = "http://localhost:8741/web/demo.html"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
W, H = 1280, 900

import websockets                                   # noqa: E402


def find_browser():
    for c in (EDGE, r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(c):
            return c
    raise SystemExit("no Chromium-family browser found")


def launch(profile, headless=False, extra=()):
    args = [find_browser(),
            "--remote-debugging-port=%d" % PORT,
            "--user-data-dir=%s" % profile,
            "--no-first-run", "--no-default-browser-check",
            "--window-size=%d,%d" % (W, H),
            # ON SCREEN, deliberately. Chromium throttles Page.startScreencast for a window the
            # compositor believes nobody can see, and an off-screen window delivered 0.9 fps of a
            # 133 fps page -- a recording that would have libelled the thing it was recording.
            "--window-position=0,0",
            "--disable-backgrounding-occluded-windows",
            "--disable-features=CalculateNativeWinOcclusion",
            "--disable-renderer-backgrounding", "--disable-background-timer-throttling",
            URL]
    if headless:
        args.insert(1, "--headless=new")
    args[1:1] = list(extra)
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def page_ws(timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            data = json.loads(urllib.request.urlopen(
                "http://127.0.0.1:%d/json/list" % PORT, timeout=2).read())
            for t in data:
                if t.get("type") == "page" and "demo.html" in t.get("url", ""):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise SystemExit("could not reach the DevTools endpoint")


class CDP:
    """A minimal CDP client with a real reader loop, because the screencast interleaves EVENTS with
    command responses -- a naive "read until my id comes back" client throws the frames away."""

    def __init__(self, ws):
        self.ws = ws
        self.i = 0
        self.pending = {}
        self.frames = []          # (monotonic-ish page timestamp, png bytes)
        self.recording = False
        self.reader = asyncio.create_task(self._pump())

    async def _pump(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self.pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                elif msg.get("method") == "Page.screencastFrame":
                    p = msg["params"]
                    if self.recording:
                        self.frames.append((p["metadata"].get("timestamp", time.time()),
                                            base64.b64decode(p["data"])))
                    await self.ws.send(json.dumps({"id": -1, "method": "Page.screencastFrameAck",
                                                   "params": {"sessionId": p["sessionId"]}}))
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

    async def js(self, expr, wait=True):
        r = await self.call("Runtime.evaluate", expression="(async()=>{%s})()" % expr,
                            awaitPromise=wait, returnByValue=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:400])
        return r["result"].get("value")

    async def shot(self, path):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path.write_bytes(base64.b64decode(r["data"]))

    async def shot_bytes(self):
        r = await self.call("Page.captureScreenshot", format="png")
        return base64.b64decode(r["data"])

    async def mouse(self, kind, x, y, buttons=0):
        await self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left",
                        buttons=buttons, clickCount=1 if kind != "mouseMoved" else 0)


async def run():
    SHOTS.mkdir(parents=True, exist_ok=True)
    for f in SHOTS.glob("*.png"):
        f.unlink()
    # A FRESH profile per run, and the browser is closed through CDP at the end. msedge.exe is a
    # launcher that exits immediately, so Popen.terminate() kills the launcher and leaves the real
    # browser holding the debugging port -- the next run then attaches to a page still running the
    # PREVIOUS build of demo4.js and silently measures the wrong code. That happened once already.
    profile = HERE / ("_cdpprofile_%d" % int(time.time()))
    shutil.rmtree(profile, ignore_errors=True)
    proc = launch(str(profile))
    browser_ws = None
    try:
        ws_url = page_ws()
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            c = CDP(ws)
            await c.call("Page.enable")
            await c.call("Runtime.enable")
            await asyncio.sleep(2.0)
            info = await c.js("const d = await window.DEMO.ready; await new Promise(r=>setTimeout(r,900));"
                              "return { gpu: !!navigator.gpu, n: d.sim.n, dt: d.stats().dt,"
                              " fps: d.stats().fps, spf: d.stats().spf, achieved: d.stats().achieved,"
                              " errors: MPM4.errors() };")
            print("page:", json.dumps(info))
            stamp = await c.js("const r = await fetch('/web/demo4.js?x='+Date.now());"
                               "const t = await r.text();"
                               "return { frame_rate_independent: t.includes('lastFrameAt'),"
                               "         bytes: t.length };")
            print("served demo4.js:", json.dumps(stamp))
            assert stamp["frame_rate_independent"], "the page is running a stale build"
            if not info or not info.get("gpu"):
                raise SystemExit("this window has no WebGPU -- capture would be meaningless")

            # canvas rect in CSS pixels, for real pointer input
            box = await c.js("const r = document.querySelector('canvas').getBoundingClientRect();"
                             "return {x:r.left, y:r.top, w:r.width, h:r.height};")

            def pt(sx, sy):
                return box["x"] + sx * box["w"], box["y"] + (1 - sy) * box["h"]

            async def click_btn(sel):
                await c.js("document.querySelector(%s).click(); return 1;" % json.dumps(sel))

            # stream from the compositor instead of round-tripping a screenshot per frame: a
            # Page.captureScreenshot loop tops out near 9 fps and makes a 130 fps page look broken
            await c.call("Page.startScreencast", format="png", quality=92, everyNthFrame=1,
                         maxWidth=W, maxHeight=H)
            c.recording = True
            t_start = time.time()

            async def grab_for(seconds, label=""):
                await asyncio.sleep(seconds)

            async def drag(path, seconds, buttons=1):
                x, y = pt(*path[0])
                await c.mouse("mouseMoved", x, y)
                await c.mouse("mousePressed", x, y, buttons=1)
                t0 = time.time()
                n = len(path) - 1
                while True:
                    u = (time.time() - t0) / seconds
                    if u >= 1:
                        break
                    seg = min(n - 1, int(u * n))
                    f = u * n - seg
                    a, b = path[seg], path[seg + 1]
                    x, y = pt(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                    await c.mouse("mouseMoved", x, y, buttons=1)
                    await asyncio.sleep(1 / 90)
                x, y = pt(*path[-1])
                await c.mouse("mouseReleased", x, y)

            # ---- the scripted session -------------------------------------------------------
            await click_btn('[data-act="reset"]')
            await grab_for(2.2)                                        # the opening scene lands
            await c.shot(SHOTS / "still_material.png")

            await click_btn('[data-tool="sand"]')                      # pour a stream of sand
            await drag([(0.30, 0.92), (0.44, 0.90), (0.58, 0.92), (0.70, 0.90)], 2.6)
            await grab_for(1.0)
            await c.shot(SHOTS / "still_pour.png")

            await click_btn('[data-view="grid"]')                      # the grid actually carries it
            await grab_for(1.6)
            await c.shot(SHOTS / "still_grid.png")

            await click_btn('[data-view="pts"]')
            await grab_for(1.4)
            await c.shot(SHOTS / "still_pts.png")

            await click_btn('[data-view="blob"]')
            await grab_for(0.5)

            await click_btn('[data-tool="grab"]')                      # drag the material around
            await drag([(0.22, 0.10), (0.40, 0.30), (0.62, 0.12), (0.80, 0.26)], 2.6)
            await grab_for(0.9)
            await c.shot(SHOTS / "still_grab.png")

            await click_btn('[data-tool="erase"]')                     # carve a channel out
            await drag([(0.24, 0.06), (0.50, 0.06), (0.74, 0.06)], 1.8)
            await grab_for(1.1)
            await c.shot(SHOTS / "still_erase.png")

            await click_btn('[data-act="reset"]')
            await grab_for(1.8)
            c.recording = False
            await c.call("Page.stopScreencast")
            wall = time.time() - t_start

            final = await c.js("const d = window.DEMO; return { n: d.sim.n, counts: d.counts(),"
                               " fps: d.stats().fps, achieved: d.stats().achieved, spf: d.stats().spf,"
                               " spfFull: d.stats().spfFull, dt: d.stats().dt,"
                               " hud: [...document.querySelectorAll('.chip')].map(e=>e.textContent),"
                               " errors: MPM4.errors() };")
            print("final:", json.dumps(final, indent=1))

            for i, (_, b) in enumerate(c.frames):
                (SHOTS / ("cap_%04d.png" % i)).write_bytes(b)
            fps = len(c.frames) / max(wall, 1e-6)
            (HERE / "out" / "capture.json").write_text(json.dumps(
                {"frames": len(c.frames), "wall_seconds": wall, "capture_fps": fps,
                 "page": info, "final": final}, indent=2), encoding="utf-8")
            print("captured %d frames over %.1f s -> %.1f fps (played back at that rate = real time)"
                  % (len(c.frames), wall, fps))
            return fps
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
            proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            pass
        subprocess.run(["taskkill", "/F", "/FI",
                        "WINDOWTITLE eq MATTER*"], capture_output=True)
        shutil.rmtree(profile, ignore_errors=True)


def assemble(fps):
    import imageio_ffmpeg
    exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    dst = RUN / "demo_capture.mp4"
    subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", "%.3f" % fps,
                    "-i", str(SHOTS / "cap_%04d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", "30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "20",
                    str(dst)], check=True)
    print("wrote", dst)


async def fallback_shot():
    """The other half of graceful degradation: what a browser WITHOUT WebGPU sees.

    The obvious instrument -- headless Chromium -- is a TRAP, and it produced one wrong artifact
    before this comment existed. Headless Edge *does* expose `navigator.gpu` and *does* return an
    adapter; what it lacks is a compositor, so requestAnimationFrame never fires and the page sits
    blank with its readouts at "--". Screenshotting that and captioning it "no WebGPU" would have
    been a picture of a completely different failure.

    So the API is hidden from a REAL GPU-backed window instead, before the page's first script
    runs. The page cannot tell the difference, and every other part of it still composites."""
    profile = HERE / ("_hlprofile_%d" % int(time.time()))
    shutil.rmtree(profile, ignore_errors=True)
    out = RUN / "demo_no_webgpu.png"
    proc = launch(str(profile))
    try:
        async with websockets.connect(page_ws(), max_size=32 * 1024 * 1024) as ws:
            c = CDP(ws)
            await c.call("Page.enable")
            await c.call("Runtime.enable")
            await c.call("Page.addScriptToEvaluateOnNewDocument", source=(
                "Object.defineProperty(Navigator.prototype, 'gpu', "
                "{ get: function () { return undefined; }, configurable: true });"))
            await c.call("Page.reload", ignoreCache=True)
            await asyncio.sleep(4.0)
            state = await c.js(
                "return { gpu: !!navigator.gpu, secure: window.isSecureContext,"
                " fallbackShown: !document.querySelector('.fallback').hidden,"
                " head: document.querySelector('.fallback h2').textContent,"
                " why: document.querySelector('[data-k=why]').textContent,"
                " badge: document.querySelector('[data-k=badge]').textContent };")
            print("no-webgpu path:", json.dumps(state))
            assert state["fallbackShown"] and not state["gpu"], "the degradation path did not engage"
            await c.shot(out)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        subprocess.run(["taskkill", "/F", "/IM", "msedge.exe"], capture_output=True)
        shutil.rmtree(profile, ignore_errors=True)
    print("wrote", out, out.exists())


if __name__ == "__main__":
    f = asyncio.run(run())
    assemble(f)
    asyncio.run(fallback_shot())
