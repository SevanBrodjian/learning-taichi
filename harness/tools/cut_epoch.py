#!/usr/bin/env python3
"""Cut an epoch — freeze the four things that are only meaningful together.

An epoch is a cut across the whole project at an inflection point, so that "what did this look like in
August" is a question you can open rather than a git archaeology exercise. The full rationale (including
what an epoch is NOT — it is not a folder that tasks get swept into) is `coordination/epochs/README.md`.
This tool implements the cut described there and nothing else.

What it freezes, in one act:
  1. THE REPORT.  `reports/research_report.md` copied to the epoch as `report.md`, having PASSED
     `spec/examination.md`. The epoch does not close until it passes, so a cut without a passing
     verdict.md needs an explicit --force and is stamped `forced: true` in the json.
  2. THE DEMO.    `harness/dashboard/src/components/mpm/` copied to `demo-versions/<n>-<slug>/` with a
     plain index.html that loads it. This is a COPY rather than a build because the transplant contract
     holds: that directory imports nothing from the harness. Freezing it is the second thing that
     constraint buys.
  3. THE PHYSICS VERSION, from `sim.physics` — the canonical, imported definition, never re-derived here.
     The demo's generated params.js carries its own stamp; both go in the json and a mismatch is called
     out loudly, because a frozen demo whose physics version is unknown is a curiosity, not evidence.
  4. THE TASK SET AND GRAPH at that instant: every task's id, ref, title, status and tags, and every
     edge with its kind.

Deliberate by design. Nothing calls this automatically, and it commits nothing — it writes to disk and
tells you what it wrote, so the cut is reviewed like any other change.

Usage:
  python harness/tools/cut_epoch.py 1 first-demo --title "The first demo"
  python harness/tools/cut_epoch.py 1 first-demo --dry-run          # say what it would do, write nothing
  python harness/tools/cut_epoch.py 1 first-demo --force            # cut without a PASS verdict
  python harness/tools/cut_epoch.py --serve 1-first-demo            # verify the frozen demo actually runs

`--serve` matters more than it looks: WebGPU is only exposed in a SECURE CONTEXT, so a frozen demo
opened from file:// shows nothing and proves nothing. It serves the frozen directory over
http://localhost, which is a secure context, so "the copy runs" is something you can actually check.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = ROOT / "coordination" / "epochs"
DEMO_SRC = ROOT / "harness" / "dashboard" / "src" / "components" / "mpm"
DEMO_OUT = ROOT / "demo-versions"
REPORT_SRC = ROOT / "reports" / "research_report.md"
DIRECTIONS = ROOT / "coordination" / "directions"

# The demo's own module graph. demo4.js is the entry; it imports mpm4.js, which imports params.js;
# demo4.css.js is the stylesheet as a string. Anything else in that directory is copied too (a new file
# there is more likely a new part of the demo than junk), but these four must be present.
DEMO_REQUIRED = ("demo4.js", "demo4.css.js", "mpm4.js", "params.js")

INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<!-- Frozen demo build. Epoch {n} ({slug}), cut {cut}, physics {physics}.
     Copied verbatim from harness/dashboard/src/components/mpm/ by harness/tools/cut_epoch.py.
     Self-contained: no framework, no data server, no network. Serve it over http(s) — WebGPU needs a
     secure context, so file:// will not start it. -->
<style>
  html, body {{ margin: 0; height: 100%; background: #05070b; overflow: hidden; }}
  #demo {{ position: fixed; inset: 0; }}
  #boot {{ position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
          color: #6f8794; font: 12px/1.6 ui-monospace, Menlo, Consolas, monospace; letter-spacing: .1em;
          text-transform: uppercase; }}
</style>
<div id="boot">loading</div>
<div id="demo"></div>
<script type="module">
  import MPMDemo4 from './demo4.js';
  import DEMO_CSS from './demo4.css.js';
  const style = document.createElement('style');
  style.textContent = DEMO_CSS;
  document.head.appendChild(style);
  const boot = document.getElementById('boot');
  try {{
    MPMDemo4.mount(document.getElementById('demo'), {{}});
    boot.remove();
  }} catch (e) {{
    boot.textContent = 'failed to start: ' + String((e && e.message) || e);
  }}
</script>
"""


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def physics_version() -> str:
    """From sim.physics — the canonical definition, imported, never re-derived here (CLAUDE.md)."""
    sys.path.insert(0, str(ROOT))
    try:
        import sim.physics as physics
    except Exception as e:  # an epoch without a provable physics version is not worth cutting
        raise SystemExit(f"cannot import sim.physics ({e}); an epoch must record its ground truth")
    return physics.VERSION


def demo_params_version() -> str | None:
    f = DEMO_SRC / "params.js"
    if not f.is_file():
        return None
    m = re.search(r"physics_version:\s*(phys-[0-9a-f]+)", f.read_text("utf-8", errors="ignore"))
    return m.group(1) if m else None


def _parents(task: dict) -> list[dict]:
    """`follow_up_of` is [{id, dir, kind}] now; the older plain-id and bare-string forms still read."""
    v = task.get("follow_up_of")
    if not v:
        return []
    if isinstance(v, (str, dict)):
        v = [v]
    out = []
    for p in v:
        if isinstance(p, str):
            out.append({"id": p, "dir": None, "kind": None})
        elif isinstance(p, dict) and p.get("id"):
            out.append({"id": p["id"], "dir": p.get("dir"), "kind": p.get("kind")})
    return out


def collect_graph() -> tuple[list[dict], list[dict]]:
    """Every task in every direction, and every edge between them, as they stand right now."""
    tasks: list[dict] = []
    edges: list[dict] = []
    for f in sorted(DIRECTIONS.glob("*.json")):
        try:
            data = json.loads(f.read_text("utf-8"))
        except Exception:
            print(f"  ! skipping unreadable {f.name}")
            continue
        did = data.get("id", f.stem)
        for t in data.get("tasks", []):
            tasks.append({
                "id": t.get("id"),
                "dir": did,
                "ref": t.get("ref"),
                "title": t.get("title"),
                "status": t.get("status"),
                "tags": t.get("tags", []),
            })
            for p in _parents(t):
                edges.append({
                    "parent": {"id": p["id"], "dir": p["dir"] or did},
                    "child": {"id": t.get("id"), "dir": did},
                    "kind": p["kind"] or "extends",
                })
    tasks.sort(key=lambda t: (t.get("ref") or "~", t.get("id") or ""))
    return tasks, edges


def read_verdict(epoch_dir: Path, verdict_arg: str | None) -> tuple[str | None, Path | None]:
    """PASS / REVISE, from the grader's verdict.md (spec/examination.md defines the format)."""
    src = Path(verdict_arg).resolve() if verdict_arg else (epoch_dir / "verdict.md")
    if not src.is_file():
        return None, None
    head = src.read_text("utf-8", errors="ignore")[:2000].upper()
    if "REVISE" in head and head.find("REVISE") < (head.find("PASS") if "PASS" in head else 10**9):
        return "REVISE", src
    if "PASS" in head:
        return "PASS", src
    return None, src


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def freeze_demo(out: Path, n: int, slug: str, title: str, cut: str, physics: str,
                dry: bool) -> list[dict]:
    missing = [f for f in DEMO_REQUIRED if not (DEMO_SRC / f).is_file()]
    if missing:
        raise SystemExit(f"demo source is incomplete, missing: {', '.join(missing)}")
    files = []
    if not dry:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
    for src in sorted(p for p in DEMO_SRC.iterdir() if p.is_file()):
        if not dry:
            shutil.copy2(src, out / src.name)
        files.append({"name": src.name, "bytes": src.stat().st_size, "sha256": sha256(src)})
    if not dry:
        (out / "index.html").write_text(
            INDEX_HTML.format(title=title, n=n, slug=slug, cut=cut, physics=physics),
            encoding="utf-8", newline="\n")
        files.append({"name": "index.html", "bytes": (out / "index.html").stat().st_size,
                      "sha256": sha256(out / "index.html")})
    return files


def cut(n: int, slug: str, title: str | None, verdict_arg: str | None,
        force: bool, replace: bool, dry: bool) -> None:
    eid = f"{n}-{slug}"
    epoch_dir = EPOCHS / eid
    demo_dir = DEMO_OUT / eid
    title = title or slug.replace("-", " ").capitalize()
    cut_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    if (epoch_dir / "epoch.json").is_file() and not replace:
        raise SystemExit(f"{eid} has already been cut; pass --replace to overwrite it")

    # 1. the report, and the verdict that lets the epoch close at all
    verdict, verdict_src = read_verdict(epoch_dir, verdict_arg)
    if verdict != "PASS" and not force:
        where = verdict_src or (epoch_dir / "verdict.md")
        raise SystemExit(
            f"the epoch does not close until its report passes (spec/examination.md).\n"
            f"  verdict: {verdict or 'none found'}  ({where})\n"
            f"  cut anyway with --force; the json will record forced: true")
    if not REPORT_SRC.is_file() and not force:
        raise SystemExit(f"{REPORT_SRC} does not exist; --force to cut without a report")

    # 2. the physics version — imported, never re-derived
    phys = physics_version()
    params_phys = demo_params_version()

    # 3. the task set and graph
    tasks, edges = collect_graph()

    print(f"cutting epoch {n} — {title}")
    print(f"  physics      {phys}" + ("" if params_phys == phys else f"   demo params.js says {params_phys}"))
    if params_phys and params_phys != phys:
        print("  ! the frozen demo was generated from a DIFFERENT physics version than sim.physics is on.")
        print("    Both are recorded; treat the demo's behaviour as reproducing ITS stamp, not the repo's.")
    print(f"  tasks        {len(tasks)} across {len(set(t['dir'] for t in tasks))} directions, {len(edges)} edges")
    print(f"  report       {'(missing)' if not REPORT_SRC.is_file() else REPORT_SRC.relative_to(ROOT)}"
          f"   verdict: {verdict or 'none'}{' (FORCED)' if verdict != 'PASS' else ''}")

    demo_files = freeze_demo(demo_dir, n, slug, f"{title} — epoch {n}", cut_at, phys, dry)
    print(f"  demo         {len(demo_files)} files -> {demo_dir.relative_to(ROOT)}")

    report_info = {"source": "reports/research_report.md", "frozen": None,
                   "bytes": None, "sha256": None, "verdict": verdict,
                   "verdict_file": str(verdict_src.relative_to(ROOT)).replace("\\", "/") if verdict_src else None}
    if not dry:
        epoch_dir.mkdir(parents=True, exist_ok=True)
        if REPORT_SRC.is_file():
            shutil.copy2(REPORT_SRC, epoch_dir / "report.md")
            report_info.update({"frozen": f"coordination/epochs/{eid}/report.md",
                                "bytes": (epoch_dir / "report.md").stat().st_size,
                                "sha256": sha256(epoch_dir / "report.md")})
        if verdict_src and verdict_src != epoch_dir / "verdict.md":
            shutil.copy2(verdict_src, epoch_dir / "verdict.md")
            report_info["verdict_file"] = f"coordination/epochs/{eid}/verdict.md"

    epoch = {
        "schema_version": "1",
        "id": eid,
        "n": n,
        "slug": slug,
        "title": title,
        "cut": cut_at,
        "physics_version": phys,
        "report_verdict": verdict,
        "forced": verdict != "PASS",
        "report": report_info,
        "demo": {
            "path": f"demo-versions/{eid}",
            "source": "harness/dashboard/src/components/mpm",
            "entry": "index.html",
            "params_physics_version": params_phys,
            "physics_matches": params_phys == phys,
            "files": demo_files,
        },
        "git": git_state(),
        "tasks": tasks,
        "edges": edges,
        "cut_by": "harness/tools/cut_epoch.py",
    }

    if dry:
        print("\n--dry-run: nothing written. epoch.json would be:\n")
        print(json.dumps({k: v for k, v in epoch.items() if k not in ("tasks", "edges")}, indent=2))
        return

    (epoch_dir / "epoch.json").write_text(json.dumps(epoch, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {(epoch_dir / 'epoch.json').relative_to(ROOT)}")
    print(f"verify the frozen build actually runs (WebGPU needs a secure context):")
    print(f"  python harness/tools/cut_epoch.py --serve {eid}")
    print("then review and commit:")
    print(f"  git add coordination/epochs/{eid} demo-versions/{eid} && git commit -m \"epoch {n}: {title}\"")


def serve(eid: str, port: int) -> None:
    """Serve one frozen build over http://localhost — a secure context, which navigator.gpu requires.
    Opening index.html from the filesystem does not test anything: modules and WebGPU both refuse."""
    import functools
    import http.server
    import socketserver

    d = DEMO_OUT / eid
    if not (d / "index.html").is_file():
        raise SystemExit(f"no frozen demo at {d}")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"frozen demo {eid} -> http://localhost:{port}/   (ctrl-c to stop)")
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", nargs="?", help="epoch number, e.g. 1")
    ap.add_argument("slug", nargs="?", help="short slug, e.g. first-demo")
    ap.add_argument("--title", help="human title (defaults to the slug)")
    ap.add_argument("--verdict", help="path to the grader's verdict.md, if it is not already in the epoch dir")
    ap.add_argument("--force", action="store_true", help="cut without a PASS verdict (recorded as forced)")
    ap.add_argument("--replace", action="store_true", help="overwrite an epoch that was already cut")
    ap.add_argument("--dry-run", action="store_true", help="say what it would do, write nothing")
    ap.add_argument("--serve", metavar="EPOCH_ID", help="serve a frozen demo over http://localhost to verify it")
    ap.add_argument("--port", type=int, default=8788)
    a = ap.parse_args()

    if a.serve:
        serve(a.serve, a.port)
        return
    if not a.n or not a.slug:
        ap.error("give an epoch number and a slug, e.g. `cut_epoch.py 1 first-demo`")
    try:
        n = int(a.n)
    except ValueError:
        ap.error("the epoch number must be an integer")
    slug = re.sub(r"[^a-z0-9]+", "-", a.slug.lower()).strip("-")
    if not slug:
        ap.error("the slug must contain something url-safe")
    cut(n, slug, a.title, a.verdict, a.force, a.replace, a.dry_run)


if __name__ == "__main__":
    main()
