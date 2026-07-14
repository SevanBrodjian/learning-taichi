#!/usr/bin/env python3
"""Keep the data server alive.

Launches `harness/server/app.py` and restarts it if it dies OR stops responding on /api/health — so a
worker's memory spike (which has OOM-killed the server before) can never again leave the dashboard staring
at an API error. The dashboard self-heals within seconds.

Run this INSTEAD of the raw server:
    python harness/tools/serve_watchdog.py
Env: DASHBOARD_PORT (default 8732). This process is the long-lived supervisor; keep it running on main.
"""
import os
import subprocess
import sys
import time
import urllib.request

PORT = int(os.environ.get("DASHBOARD_PORT", "8732"))
HEALTH = f"http://localhost:{PORT}/api/health"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def responding() -> bool:
    try:
        with urllib.request.urlopen(HEALTH, timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def launch():
    print("[watchdog] launching data server", flush=True)
    return subprocess.Popen([sys.executable, "harness/server/app.py"], cwd=REPO)


def main():
    proc = None
    misses = 0
    while True:
        if proc is None or proc.poll() is not None:
            if proc is not None:
                print("[watchdog] server process exited — restarting", flush=True)
            proc = launch()
            time.sleep(6)  # let it bind the port before health-checking
            misses = 0
        elif not responding():
            misses += 1
            if misses >= 2:  # two consecutive misses => hung/unresponsive, not just a slow request
                print("[watchdog] server unresponsive — killing and restarting", flush=True)
                try:
                    proc.kill()
                except Exception:
                    pass
                proc = None
                misses = 0
                continue
        else:
            misses = 0
        time.sleep(10)


if __name__ == "__main__":
    main()
