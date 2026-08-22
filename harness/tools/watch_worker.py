#!/usr/bin/env python3
"""Adaptive check-in on a running worker (the orchestrator's periodic health watch).

Sleeps one check interval (or until the task's soft time budget, whichever is sooner), then reports the
worker's health and EXITS — which re-invokes the orchestrator to act:

    exit 0  HEALTHY     : status is fresh and under budget -> re-arm (launch another watch, keep going).
    exit 1  STALE       : status has not updated in a while -> the worker likely stalled / ended its turn
                          on a background job -> intervene (nudge to converge, or take over).
    exit 2  OVER_BUDGET : elapsed passed the task's budget -> converge (review the on-disk result / take
                          over); if a manifest already exists it very likely has a complete result.

Budgets are a SOFT expectation set from the effort tier (and tunable per task on the dashboard), NOT a
hard cap — a long-but-productive task keeps its fresh status and stays HEALTHY. Run in the BACKGROUND
(never a foreground wait — those time out).

Usage:
  python harness/tools/watch_worker.py --direction <d> --task <t> --budget <min> --started <unix_ts>
                                       [--interval 20] [--stale 25]
"""
import argparse
import json
import pathlib
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="Periodic health check on a running worker.")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--budget", type=float, required=True, help="soft time budget in minutes")
    ap.add_argument("--started", type=float, required=True, help="unix ts the worker started")
    ap.add_argument("--interval", type=float, default=20.0, help="check interval in minutes")
    ap.add_argument("--stale", type=float, default=25.0, help="status considered stale after this many minutes")
    ap.add_argument("--quiet", type=float, default=12.0,
                    help="no file written under the run dir for this many minutes (default 12)")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()

    deadline = a.started + a.budget * 60.0
    wake = min(time.time() + a.interval * 60.0, deadline)
    time.sleep(max(5.0, wake - time.time()))

    base = pathlib.Path(a.root) / "runs" / a.direction / a.task
    sf, mf = base / "status.json", base / "manifest.json"
    now = time.time()
    elapsed = (now - a.started) / 60.0
    step, updated = "(no status yet)", 0
    if sf.is_file():
        try:
            s = json.loads(sf.read_text("utf-8"))
            step, updated = s.get("step", "?"), int(s.get("updated", 0))
        except Exception:
            pass
    status_age = (now - updated) / 60.0 if updated else 1e9

    # A worker that is WORKING but not TALKING is not stalled. Judging liveness by the status string
    # alone raised two false STALE alarms on genuinely healthy runs -- a long capture or a training
    # loop legitimately goes quiet for half an hour with nothing worth reporting, while its files keep
    # moving. File mtimes under the run directory are the honest signal; the status string is a
    # courtesy the worker may forget. Stale is now BOTH: nothing said AND nothing written.
    newest = 0.0
    if base.is_dir():
        for f in base.rglob("*"):
            if f.name == "status.json":
                continue          # written by the status call itself; would mask a real stall
            try:
                if f.is_file():
                    newest = max(newest, f.stat().st_mtime)
            except OSError:
                continue
    file_age = (now - newest) / 60.0 if newest else 1e9
    has_manifest = mf.is_file()
    print(f"{a.direction}/{a.task}  elapsed={elapsed:.0f}m budget={a.budget:.0f}m "
          f"status_age={status_age:.0f}m file_age={file_age:.0f}m step='{step}' manifest={'yes' if has_manifest else 'no'}")

    if status_age > a.stale and file_age > a.quiet:
        print("STALE")
        sys.exit(1)
    if elapsed >= a.budget:
        print("OVER_BUDGET")
        sys.exit(2)
    print("HEALTHY")
    sys.exit(0)


if __name__ == "__main__":
    main()
