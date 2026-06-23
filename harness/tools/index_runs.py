#!/usr/bin/env python3
"""Aggregate every runs/<branch>/<run-id>/manifest.json into runs/index.json.

index.json powers the dashboard branch-switcher + run list. See runs/README.md.
"""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    runs_dir = os.path.join(REPO, "runs")
    entries = []
    for mpath in glob.glob(os.path.join(runs_dir, "**", "manifest.json"), recursive=True):
        try:
            with open(mpath) as fh:
                m = json.load(fh)
        except Exception as e:  # noqa: BLE001
            print(f"skip {mpath}: {e}")
            continue
        rel = os.path.relpath(mpath, REPO).replace(os.sep, "/")
        entries.append({
            "run_id": m.get("run_id"),
            "branch": m.get("branch"),
            "title": m.get("title", m.get("run_id")),
            "status": m.get("status", "unknown"),
            "created": m.get("created"),
            "manifest": rel,
        })
    entries.sort(key=lambda e: (e.get("created") or ""), reverse=True)
    index = {"schema_version": "0", "runs": entries}
    with open(os.path.join(runs_dir, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"indexed {len(entries)} run(s) -> runs/index.json")


if __name__ == "__main__":
    main()
