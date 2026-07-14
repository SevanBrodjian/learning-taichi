#!/usr/bin/env python3
"""Wake helper for the contract-approval gate.

Polls a contract decision file until it is resolved (Approve / Reject) or its auto-run deadline passes,
then EXITS — which re-invokes the orchestrator to act:
    exit 0  -> APPROVED  : spawn the worker now.
    exit 1  -> REJECTED  : send the task back to the queue with the note.
    exit 2  -> TIMEOUT   : the deadline passed with no decision -> auto-run the task AS-IS.

Run this in the BACKGROUND (never a foreground blocking wait — those time out). The orchestrator launches
it after posting a contract; when it exits, the orchestrator wakes and does the right thing. This is how a
contract auto-runs on approval OR after the timeout without the user having to come back and say so.

Usage:
    python harness/tools/await_contract.py --id <contract-stem> --deadline <unix_ts> [--interval 20]
"""
import argparse
import pathlib
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="Wait for a contract to be approved/rejected, or time out.")
    ap.add_argument("--id", required=True, help="contract decision id (the .md filename stem)")
    ap.add_argument("--deadline", type=float, required=True, help="unix timestamp to auto-run at")
    ap.add_argument("--interval", type=float, default=20.0, help="poll seconds (default 20)")
    ap.add_argument("--root", default=".", help="repo root (default cwd)")
    a = ap.parse_args()
    f = pathlib.Path(a.root) / "coordination" / "decisions" / f"{a.id}.md"
    while True:
        txt = f.read_text("utf-8", errors="ignore") if f.is_file() else ""
        if "Resolution: APPROVED" in txt:
            print(f"APPROVED {a.id}")
            sys.exit(0)
        if "Resolution: REJECTED" in txt:
            print(f"REJECTED {a.id}")
            sys.exit(1)
        if time.time() >= a.deadline:
            print(f"TIMEOUT {a.id} (auto-run as-is)")
            sys.exit(2)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
