#!/usr/bin/env python3
"""ntfy notifier for agent progress/gate pings.

Two levels (see CLAUDE.md):
  progress  non-blocking FYI (run started, loss updates, report drafted) — emit freely
  gate      needs the user (milestone decision, divergence, hard block) — use sparingly

Topic resolution (it is a secret — never commit it):
  --topic arg  >  NTFY_TOPIC env var  >  ~/.learning-taichi/ntfy_topic
  The file lives OUTSIDE the repo so every worktree/subagent can read it without it syncing to GitHub.
  NTFY_SERVER  defaults to https://ntfy.sh

Usage:
  python tools/notify.py "epoch 200, loss 0.031"
  python tools/notify.py --level gate "DiffMPM milestone ready for review"
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")

# The topic is a secret and must never land in the repo (it would sync to GitHub). It is resolved from,
# in order: an explicit --topic, the NTFY_TOPIC env var, or a file at a fixed path OUTSIDE the repo, so
# every worktree and spawned subagent can read it without it being committed.
TOPIC_FILE = Path.home() / ".learning-taichi" / "ntfy_topic"


def _resolve_topic(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("NTFY_TOPIC")
    if env:
        return env
    if TOPIC_FILE.is_file():
        topic = TOPIC_FILE.read_text(encoding="utf-8").strip()
        if topic:
            return topic
    return None

LEVELS = {
    "progress": {"priority": "low", "tags": "hourglass_flowing_sand"},
    "gate": {"priority": "high", "tags": "rotating_light"},
}


def notify(message: str, level: str = "progress", title: str | None = None,
           topic: str | None = None) -> int:
    cfg = LEVELS.get(level, LEVELS["progress"])
    topic = _resolve_topic(topic)
    if not topic:
        raise RuntimeError(
            f"No ntfy topic: set NTFY_TOPIC, pass --topic, or write it to {TOPIC_FILE}"
        )
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
