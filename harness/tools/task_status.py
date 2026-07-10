#!/usr/bin/env python3
"""Worker -> dashboard live status.

A running worker calls this a handful of times over a run to say what step it is on. It writes
`runs/<direction>/<task>/status.json`, which the data server reads (`live_statuses` in
harness/server/app.py) so an Active task on the board reads as *running* with a one-line note,
instead of an undifferentiated "Active".

This is deliberately lightweight and infrequent — a few coarse milestones ("training the net",
"rendering final videos"), not a per-iteration log. The file is ephemeral and gitignored; the
manifest is the durable record. When the run finishes, either call `--state done` or just let the
finished manifest supersede it (the client stops showing a live badge once the task is Done).

Usage:
  python harness/tools/task_status.py --direction <dir> --task <id> --step "training the conditioned net"
  python harness/tools/task_status.py --direction <dir> --task <id> --state blocked --step "waiting on GPU"
  python harness/tools/task_status.py --direction <dir> --task <id> --state done --step "results on disk"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def write_status(root: Path, direction: str, task: str, step: str, state: str) -> Path:
    d = root / "runs" / direction / task
    d.mkdir(parents=True, exist_ok=True)
    target = d / "status.json"
    target.write_text(
        json.dumps({"state": state, "step": step, "updated": int(time.time())}, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a worker's live status for the dashboard.")
    ap.add_argument("--direction", required=True, help="direction id (e.g. material-variants)")
    ap.add_argument("--task", required=True, help="task id")
    ap.add_argument("--step", required=True, help="a few words: the current step / what we're waiting on")
    ap.add_argument("--state", default="running", choices=["running", "blocked", "done"])
    ap.add_argument("--root", default=".", help="repo root the run dir lives under (default: cwd)")
    a = ap.parse_args()
    target = write_status(Path(a.root).resolve(), a.direction, a.task, a.step, a.state)
    print(f"status [{a.state}] {a.direction}/{a.task}: {a.step}  -> {target}")


if __name__ == "__main__":
    main()
