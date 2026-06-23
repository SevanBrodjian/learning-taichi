#!/usr/bin/env python3
"""Copy runs/ + reports/ + runs/index.json into dashboard/public/data so the standalone dashboard can
serve them at /data. Later (site integration) this is replaced by a live API; see dashboard/README.md.
"""
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "dashboard", "public", "data")


def copy_tree(src_rel):
    src = os.path.join(REPO, src_rel)
    dst = os.path.join(DATA, src_rel)
    if not os.path.isdir(src):
        print(f"skip {src_rel} (missing)")
        return
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"synced {src_rel}/ -> dashboard/public/data/{src_rel}/")


def main():
    os.makedirs(DATA, exist_ok=True)
    copy_tree("runs")
    copy_tree("reports")
    index = os.path.join(REPO, "runs", "index.json")
    if os.path.isfile(index):
        shutil.copy2(index, os.path.join(DATA, "index.json"))
        print("synced runs/index.json -> dashboard/public/data/index.json")
    else:
        print("no runs/index.json yet — run tools/index_runs.py first")


if __name__ == "__main__":
    main()
