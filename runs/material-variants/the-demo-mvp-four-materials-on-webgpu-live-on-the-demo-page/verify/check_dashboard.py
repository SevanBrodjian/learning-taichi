"""Open the REAL dashboard on port 5174, switch to the Demo tab, and check it actually runs there.

The standalone page working is not evidence that the Demo tab works: the tab goes through Vite,
React and a GENERATED ES-module copy of the same code, any of which can break while the plain page
stays fine. So this drives the shipped surface, clicks the controls on it, and screenshots it.

One trap already caught here: Vite caches its transform of a module, and a browser that asks for the
un-suffixed URL can be handed the pre-edit transform long after the file changed -- so this always
warms the module with a cache-busting query first and asserts the served source is the new one.

    .venv/Scripts/python.exe runs/.../verify/check_dashboard.py
"""
import asyncio
import json
import pathlib
import shutil
import subprocess
import time
import urllib.request

import websockets

import capture_page as CP

RUN = pathlib.Path(__file__).resolve().parents[1]
SHOTS = RUN / "verify" / "shots"
DASH = "http://localhost:5174/"


def dash_ws(timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            for t in json.loads(urllib.request.urlopen(
                    "http://127.0.0.1:%d/json/list" % CP.PORT, timeout=2).read()):
                if t.get("type") == "page" and "5174" in t.get("url", ""):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise SystemExit("no dashboard page target")


async def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    profile = RUN / "verify" / ("_dash_%d" % int(time.time()))
    shutil.rmtree(profile, ignore_errors=True)
    CP.URL = DASH
    proc = CP.launch(str(profile))
    out = {}
    try:
        async with websockets.connect(dash_ws(), max_size=64 * 1024 * 1024) as ws:
            c = CP.CDP(ws)
            await c.call("Page.enable")
            await c.call("Runtime.enable")
            await c.call("Network.enable")
            await c.call("Network.setCacheDisabled", cacheDisabled=True)
            await asyncio.sleep(3.0)

            # force Vite to re-transform, then reload so the page picks the fresh module up
            served = await c.js(
                "const t = await (await fetch('/src/components/DemoView.jsx?bust='+Date.now())).text();"
                "await fetch('/src/components/mpm/demo4.js?bust='+Date.now());"
                "return { imports_bundle: t.includes('mpm/demo4'), bytes: t.length };")
            out["served_demoview"] = served
            assert served["imports_bundle"], "Vite is still serving a DemoView without the bundle"
            await c.call("Page.reload", ignoreCache=True)
            await asyncio.sleep(4.0)

            out["clicked_demo_tab"] = await c.js(
                "const b=[...document.querySelectorAll('button,a')]"
                ".find(e=>/^\\s*demo\\s*$/i.test(e.textContent));"
                "if(b){b.click();} return !!b;")
            for _ in range(20):                        # the GPU device takes a moment to arrive
                await asyncio.sleep(1.0)
                if await c.js("return !!document.querySelector('.mpm4 canvas');"):
                    break
            await asyncio.sleep(4.0)

            out["state"] = await c.js(
                "const h=document.querySelector('.mpm4');"
                "if(!h) return {mounted:false, main:(document.querySelector('main')||{}).className};"
                "return {mounted:true,"
                " chips:[...h.querySelectorAll('.chip')].map(e=>e.textContent),"
                " buttons:h.querySelectorAll('button').length,"
                " fallback_shown:!h.querySelector('.fallback').hidden,"
                " canvas:[h.querySelector('canvas').width,h.querySelector('canvas').height]};")
            print("demo tab:", json.dumps(out["state"], indent=1))
            assert out["state"].get("mounted"), "the Demo tab did not mount the simulation"
            await c.shot(SHOTS / "dashboard_demo_tab.png")

            for v in ("grid", "pts", "blob"):
                await c.js('document.querySelector(".mpm4 [data-view=' + v + ']").click(); return 1;')
                await asyncio.sleep(1.2)
                await c.shot(SHOTS / ("dashboard_view_%s.png" % v))
            for t in ("elastic", "snow", "sand", "grab", "erase", "fluid"):
                await c.js('document.querySelector(".mpm4 [data-tool=' + t + ']").click(); return 1;')
            await c.js('document.querySelector(".mpm4 [data-act=clear]").click();'
                       'document.querySelector(".mpm4 [data-act=reset]").click(); return 1;')
            await asyncio.sleep(2.0)
            out["after_controls"] = await c.js(
                "const h=document.querySelector('.mpm4');"
                "return {chips:[...h.querySelectorAll('.chip')].map(e=>e.textContent),"
                " tool_on:[...h.querySelectorAll('[data-tool].on')].map(e=>e.dataset.tool),"
                " view_on:[...h.querySelectorAll('[data-view].on')].map(e=>e.dataset.view),"
                " webgpu_errors: window.MPM4 ? MPM4.errors() : 'no global'};")
            print("after clicking every control:", json.dumps(out["after_controls"], indent=1))
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        subprocess.run(["taskkill", "/F", "/IM", "msedge.exe"], capture_output=True)
        shutil.rmtree(profile, ignore_errors=True)
    (RUN / "verify" / "out" / "dashboard_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
