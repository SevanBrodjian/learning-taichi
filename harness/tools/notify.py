#!/usr/bin/env python3
"""ntfy notifier for agent pings.

The pings are human status, not technical reports (see CLAUDE.md -> Notifications). Workers own the
routine ones and fire two per run:
  --kind started   one sentence: what it is starting        (low priority)
  --kind finished  one sentence: results are on disk         (default priority)
  --kind blocked   one sentence: hit a hard stop, needs help (high priority)
  --kind note      one sentence: an optional mid-run FYI      (low priority)
The orchestrator pings sparingly, mostly:
  --kind gate      one sentence: a decision the user must make (high priority)

Each message is a single plain sentence the agent writes itself. Never dump metrics or a report.

Topic resolution (it is a secret — never commit it):
  --topic arg  >  NTFY_TOPIC env var  >  ~/.learning-taichi/ntfy_topic
  The file lives OUTSIDE the repo so every worktree/subagent can read it without it syncing to GitHub.
  NTFY_SERVER  defaults to https://ntfy.sh

Usage:
  python harness/tools/notify.py --kind started  --task fluid-vs-snow "Starting the fluid-vs-snow sweep."
  python harness/tools/notify.py --kind finished --task fluid-vs-snow "Done; results are on disk."
  python harness/tools/notify.py --kind gate "Which direction should lead next, materials or learned dynamics?"
  python harness/tools/notify.py "epoch 200, loss 0.031"          # legacy: defaults to a low-priority note
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


# The category an agent picks. label -> (priority, tags, human label shown in the title).
KINDS = {
    "started":  {"priority": "low",     "tags": "rocket",          "label": "started"},
    "finished": {"priority": "default", "tags": "white_check_mark", "label": "finished"},
    "note":     {"priority": "low",     "tags": "speech_balloon",  "label": "note"},
    "blocked":  {"priority": "high",    "tags": "no_entry",        "label": "blocked"},
    "gate":     {"priority": "high",    "tags": "rotating_light",  "label": "needs you"},
}

# Legacy --level kept working: progress -> note, gate -> gate.
LEVEL_TO_KIND = {"progress": "note", "gate": "gate"}


def notify(message: str, kind: str = "note", task: str | None = None,
           title: str | None = None, topic: str | None = None) -> int:
    cfg = KINDS.get(kind, KINDS["note"])
    topic = _resolve_topic(topic)
    if not topic:
        raise RuntimeError(
            f"No ntfy topic: set NTFY_TOPIC, pass --topic, or write it to {TOPIC_FILE}"
        )
    if not title:
        parts = ["learning-taichi"]
        if task:
            parts.append(task)
        parts.append(cfg["label"])
        title = " · ".join(parts)
    req = urllib.request.Request(
        f"{NTFY_SERVER}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Title": title,
            "Priority": cfg["priority"],
            "Tags": cfg["tags"],
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Post a human status ping to ntfy.")
    p.add_argument("message", help="one plain sentence of status (not a report)")
    p.add_argument("--kind", choices=list(KINDS), help="category of ping")
    p.add_argument("--task", help="task id, shown in the notification title")
    p.add_argument("--level", choices=list(LEVEL_TO_KIND), help="legacy alias for --kind")
    p.add_argument("--title", help="override the auto-generated title")
    p.add_argument("--topic", help="overrides NTFY_TOPIC")
    args = p.parse_args()
    kind = args.kind or LEVEL_TO_KIND.get(args.level or "", "note")
    try:
        status = notify(args.message, kind, args.task, args.title, args.topic)
        print(f"ntfy {status}")
    except Exception as e:  # noqa: BLE001 — CLI: report and exit non-zero
        print(f"notify failed: {e}", file=sys.stderr)
        sys.exit(1)
