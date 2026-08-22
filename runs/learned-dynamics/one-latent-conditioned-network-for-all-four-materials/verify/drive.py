"""Run the WebGPU harness unattended: start the localhost server, open a real GPU-backed browser
window at it, wait for `out/done.json`, then shut everything down.

Why a real headful window and not `--headless`: headless Chromium historically does not expose
WebGPU, and a page that never composites does not get a GPU-backed surface. The whole measurement is
about a GPU, so the browser has to actually be on one. The window is kept on-screen and the
occlusion/backgrounding throttles are disabled, because a window Chromium thinks nobody can see gets
its timers and its compositor throttled -- which would libel the thing being measured.

Why localhost: `navigator.gpu` is only exposed in a secure context, and `http://localhost` is one
where `file://` and a LAN IP are not.

    .venv/Scripts/python.exe runs/.../verify/drive.py [--timeout 900] [--keep]
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
PORT = 8752
DEBUG_PORT = 9344
CANDIDATES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]


def find_browser():
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    raise SystemExit("no Chromium-family browser found")


def devtools_alive():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/version", timeout=2).read()
        return True
    except Exception:
        return False


def wait_server(timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/verify/job.json", timeout=2).read()
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--page", default="verify/harness.html")
    ap.add_argument("--keep", action="store_true", help="leave the browser open afterwards")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("done.json", "bench.json"):
        if (OUT / f).exists():
            (OUT / f).unlink()

    srv = subprocess.Popen([sys.executable, str(HERE / "serve.py"), str(PORT)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not wait_server():
        srv.terminate()
        raise SystemExit("server did not come up")
    print(f"server up on http://localhost:{PORT}/", flush=True)

    profile = HERE / "_profile"
    if profile.exists():
        shutil.rmtree(profile, ignore_errors=True)
    url = f"http://localhost:{PORT}/{a.page}"
    browser = subprocess.Popen([
        find_browser(),
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={profile}",
        "--no-first-run", "--no-default-browser-check",
        "--window-size=1100,900", "--window-position=0,0",
        "--disable-backgrounding-occluded-windows",
        "--disable-features=CalculateNativeWinOcclusion",
        "--disable-renderer-backgrounding", "--disable-background-timer-throttling",
        "--enable-unsafe-webgpu",
        url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("browser launched at", url, flush=True)

    t0 = time.time()
    rc = 1
    try:
        while time.time() - t0 < a.timeout:
            if (OUT / "done.json").exists():
                time.sleep(1.0)                       # let the last POST land
                print("done.json:", (OUT / "done.json").read_text(), flush=True)
                rc = 0
                break
            # NOTE: do NOT treat browser.poll() as "the browser died". Edge and Chrome both
            # launch a detached child and let the launcher process exit immediately, so polling the
            # handle reports an exit within a second of a perfectly healthy window. The DevTools
            # endpoint is the real liveness signal.
            if time.time() - t0 > 30 and not devtools_alive():
                print("no DevTools endpoint -- the browser really is gone", flush=True)
                break
            time.sleep(2.0)
        else:
            print(f"TIMEOUT after {a.timeout}s", flush=True)
    finally:
        if not a.keep:
            # the launcher handle is usually already dead (see the note above), so close the real
            # window through DevTools and then sweep any stragglers from THIS profile only
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{DEBUG_PORT}/json/close/browser", timeout=3).read()
            except Exception:
                pass
            time.sleep(1.5)
            try:
                browser.terminate(); browser.wait(timeout=5)
            except Exception:
                pass
            subprocess.run(["taskkill", "/F", "/FI",
                            f"WINDOWTITLE eq *{a.page}*"], capture_output=True)
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()
    if (OUT / "bench.json").exists():
        b = json.loads((OUT / "bench.json").read_text())
        if b.get("fatal"):
            print("HARNESS FATAL:", b["fatal"], flush=True)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
