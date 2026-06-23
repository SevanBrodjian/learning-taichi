#!/usr/bin/env python3
"""ntfy notifier for agent progress/gate pings.

Two levels (see CLAUDE.md):
  progress  non-blocking FYI (run started, loss updates, report drafted) — emit freely
  gate      needs the user (milestone decision, divergence, hard block) — use sparingly

Config via env:
  NTFY_TOPIC   the ntfy topic to post to (REQUIRED; treat it like a password)
  NTFY_SERVER  defaults to https://ntfy.sh

Usage:
  python tools/notify.py "epoch 200, loss 0.031"
  python tools/notify.py --level gate "DiffMPM milestone ready for review"
"""
import argparse
import os
import sys
import urllib.request

NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")

LEVELS = {
    "progress": {"priority": "low", "tags": "hourglass_flowing_sand"},
    "gate": {"priority": "high", "tags": "rotating_light"},
}


def notify(message: str, level: str = "progress", title: str | None = None,
           topic: str | None = None) -> int:
    cfg = LEVELS.get(level, LEVELS["progress"])
    topic = topic or os.environ.get("NTFY_TOPIC")
    if not topic:
        raise RuntimeError("NTFY_TOPIC is not set (and --topic not given)")
    req = urllib.request.Request(
        f"{NTFY_SERVER}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Title": title or f"learning-taichi [{level}]",
            "Priority": cfg["priority"],
            "Tags": cfg["tags"],
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Post an ntfy notification.")
    p.add_argument("message")
    p.add_argument("--level", choices=list(LEVELS), default="progress")
    p.add_argument("--title")
    p.add_argument("--topic", help="overrides NTFY_TOPIC")
    args = p.parse_args()
    try:
        status = notify(args.message, args.level, args.title, args.topic)
        print(f"ntfy {status}")
    except Exception as e:  # noqa: BLE001 — CLI: report and exit non-zero
        print(f"notify failed: {e}", file=sys.stderr)
        sys.exit(1)
