#!/usr/bin/env python3
"""Master data server for the learning-taichi dashboard.

Runs once on the MAIN checkout and presents a single, live view of runs across the main
repo AND every git worktree, so one dashboard sees all parallel agent branches without
merging anything. It replaces the static "copy into dashboard/public/data" hack and is the
natural bridge to a future Django API (identical JSON shapes).

Design:
  - /api/index                     unified run list across all worktrees (newest first).
  - /api/data/{root}/{path}        serve any file (manifest/metrics/media/report), resolved
                                   inside the worktree that owns it. manifest.json responses
                                   are rewritten so their media/metrics/report paths become
                                   absolute /api/data/... URLs -> the dashboard stays dumb.

Run:  python harness/server/app.py        # http://localhost:8732
Env:  DASHBOARD_PORT (default 8732)

This module keeps its data logic in plain functions (list_roots / build_index /
rewrite_manifest) so they are unit-testable without a running server.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

# harness/server/app.py -> repo root is three parents up.
MAIN_ROOT = Path(__file__).resolve().parents[2]
PORT = int(os.environ.get("DASHBOARD_PORT", "8732"))


def _root_id(branch: str, path: Path) -> str:
    """URL-safe, stable id for a worktree. Prefer the leaf dir name (unique per worktree)."""
    name = path.name or branch or "main"
    return name.replace("/", "~")


def list_roots() -> dict[str, Path]:
    """{root_id: absolute_path} for the main checkout + every linked worktree.

    Rebuilt per request (cheap) so a freshly created worktree shows up with no restart.
    """
    roots: dict[str, Path] = {}
    out = ""
    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=MAIN_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        pass

    entries: list[dict] = []
    cur: dict = {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            cur = {"path": line[len("worktree "):].strip()}
            entries.append(cur)
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):].strip().replace("refs/heads/", "")
    if not entries:  # not a git repo / git missing -> fall back to this checkout only
        entries = [{"path": str(MAIN_ROOT), "branch": "main"}]

    for e in entries:
        p = Path(e["path"]).resolve()
        roots[_root_id(e.get("branch", p.name), p)] = p
    return roots


def build_index(roots: dict[str, Path] | None = None) -> dict:
    """Aggregate every runs/*/*/manifest.json across all roots into one index payload."""
    roots = roots if roots is not None else list_roots()
    runs: list[dict] = []
    seen: set[tuple[str, str]] = set()  # a merged run lives in main AND its worktree; show it once
    for rid, root in roots.items():
        # Recursive: branch slugs contain slashes (runs/<branch>/<run-id>/manifest.json),
        # so the run dir sits at a variable depth under runs/.
        for mf in sorted(root.glob("runs/**/manifest.json")):
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                continue
            key = (m.get("branch", rid), m.get("run_id", mf.parent.name))
            if key in seen:
                continue
            seen.add(key)
            rel = mf.relative_to(root).as_posix()
            runs.append({
                "run_id": m.get("run_id", mf.parent.name),
                "branch": m.get("branch", rid),
                "title": m.get("title", mf.parent.name),
                "status": m.get("status", "unknown"),
                "created": m.get("created", ""),
                "root": rid,
                "manifest": f"/api/data/{rid}/{rel}",
            })
    runs.sort(key=lambda r: r.get("created", ""), reverse=True)
    return {"schema_version": "1", "runs": runs}


def rewrite_manifest(m: dict, rid: str) -> dict:
    """Turn a manifest's repo-relative paths into absolute /api/data/{rid}/... URLs."""
    def url(p):
        return f"/api/data/{rid}/{p}" if isinstance(p, str) and not p.startswith("/api/") else p

    metrics = m.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("series"), str):
        metrics["series"] = url(metrics["series"])
    for key in ("media", "reports"):
        section = m.get(key)
        if isinstance(section, dict):
            m[key] = {k: url(v) for k, v in section.items()}
    return m


def resolve(rid: str, path: str) -> Path:
    """Resolve {rid}/{path} to an absolute file, guarding against path traversal."""
    roots = list_roots()
    root = roots.get(rid)
    if root is None:
        raise FileNotFoundError(rid)
    target = (root / path).resolve()
    if os.path.commonpath([str(target), str(root)]) != str(root):
        raise PermissionError(path)
    if not target.is_file():
        raise FileNotFoundError(path)
    return target


# ---- shared, repo-level doc sets (training / directions / reports / decisions) ----
# These are NOT per-branch: they live on the checkout the server runs in (main), so they read
# from MAIN_ROOT only, and resolve to /api/data/<main>/... URLs the dashboard fetches as usual.

def main_root_id() -> str:
    for rid, root in list_roots().items():
        if root == MAIN_ROOT:
            return rid
    return _root_id("main", MAIN_ROOT)


def _shared_url(rel: str) -> str:
    return f"/api/data/{main_root_id()}/{rel}"


def training_toc() -> dict:
    """The textbook TOC from reports/training/index.json, with section URLs resolved."""
    idx = MAIN_ROOT / "reports" / "training" / "index.json"
    if not idx.is_file():
        return {"title": "Training", "groups": []}
    toc = json.loads(idx.read_text("utf-8"))
    for group in toc.get("groups", []):
        for sec in group.get("sections", []):
            if "file" in sec:
                sec["url"] = _shared_url(f"reports/training/{sec['file']}")
                fp = MAIN_ROOT / "reports" / "training" / sec["file"]
                # mtime powers the client-side "New" tag (new or edited-since-last-read).
                sec["mtime"] = int(fp.stat().st_mtime) if fp.is_file() else 0
    return toc


def directions_doc() -> dict:
    """Structured directions for the board, from coordination/directions.json, with the narrative
    research_directions.md as a fallback url."""
    j = MAIN_ROOT / "coordination" / "directions.json"
    md = MAIN_ROOT / "coordination" / "research_directions.md"
    out = {"directions": [], "md_url": _shared_url("coordination/research_directions.md") if md.is_file() else None}
    if j.is_file():
        try:
            out["directions"] = json.loads(j.read_text("utf-8")).get("directions", [])
        except Exception:
            pass
    return out


def reports_list() -> dict:
    out = []
    rr = MAIN_ROOT / "reports" / "research_report.md"
    if rr.is_file():
        out.append({"id": "research_report", "title": "Research report",
                    "url": _shared_url("reports/research_report.md")})
    return {"reports": out}


def decisions_list() -> dict:
    """The inbox: open decisions awaiting the user (coordination/decisions/*.md)."""
    d = MAIN_ROOT / "coordination" / "decisions"
    items = []
    if d.is_dir():
        for f in sorted(d.glob("*.md")):
            if f.name.lower() == "readme.md":
                continue
            items.append({"id": f.stem, "title": f.stem.replace("-", " "),
                          "url": _shared_url(f"coordination/decisions/{f.name}")})
    return {"decisions": items}


# ---- Direction -> Task model (schema v2) ----
# A direction is coordination/directions/<id>.json; a task's detail lives at
# runs/<direction>/<task-id>/manifest.json (schema_version "2"). The Overview<->Task link is structural:
# a task has a detail iff that manifest exists. No agent maintains the link.

def _v2_tasks() -> dict:
    """(direction, task_id) -> artifact info, for every schema-2 manifest across all roots."""
    out: dict = {}
    for rid, root in list_roots().items():
        for mf in sorted(root.glob("runs/**/manifest.json")):
            try:
                m = json.loads(mf.read_text("utf-8"))
            except Exception:
                continue
            if str(m.get("schema_version")) != "2":
                continue
            d, t = m.get("direction"), m.get("task_id")
            if not d or not t or (d, t) in out:
                continue
            out[(d, t)] = {"root": rid, "rel": mf.relative_to(root).as_posix(),
                           "status": m.get("status", "done")}
    return out


def rewrite_task(m: dict, rid: str) -> dict:
    """Rewrite a schema-2 task manifest's result src/series into absolute /api/data URLs."""
    def url(p):
        return f"/api/data/{rid}/{p}" if isinstance(p, str) and not p.startswith("/api/") else p
    for r in m.get("results", []):
        if isinstance(r, dict):
            for k in ("src", "series"):
                if isinstance(r.get(k), str):
                    r[k] = url(r[k])
    return m


def overview() -> dict:
    """Directions (coordination/directions/*.json on main) joined with their executed tasks."""
    ddir = MAIN_ROOT / "coordination" / "directions"
    arts = _v2_tasks()
    directions = []
    for f in (sorted(ddir.glob("*.json")) if ddir.is_dir() else []):
        try:
            d = json.loads(f.read_text("utf-8"))
        except Exception:
            continue
        did = d.get("id", f.stem)
        tasks = []
        for t in d.get("tasks", []):
            tid = t.get("id")
            has = (did, tid) in arts
            tasks.append({
                "id": tid, "title": t.get("title", tid), "status": t.get("status", "proposed"),
                "note": t.get("note", ""),
                "has_artifact": has, "detail": f"/api/task/{did}/{tid}" if has else None,
                "rework_history": t.get("rework_history", []),
                "follow_up_of": t.get("follow_up_of"),
                "follow_ups": t.get("follow_ups", []),
            })
        directions.append({
            "id": did, "name": d.get("name", did), "status": d.get("status", "proposed"),
            "summary": d.get("summary", ""), "tasks": tasks,
        })
    listed = {(dd["id"], t["id"]) for dd in directions for t in dd["tasks"]}
    orphans = [{"direction": k[0], "task": k[1]} for k in arts if k not in listed]
    return {"directions": directions, "orphans": orphans}


def task_detail(direction: str, task: str) -> dict | None:
    arts = _v2_tasks()
    info = arts.get((direction, task))
    if not info:
        return None
    m = rewrite_task(json.loads(resolve(info["root"], info["rel"]).read_text("utf-8")), info["root"])
    # The direction file is authoritative for board status and the task graph (rework, follow-ups).
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if f.is_file():
        try:
            dtasks = json.loads(f.read_text("utf-8")).get("tasks", [])
            by_id = {t.get("id"): t for t in dtasks}

            def ref(tid):
                tt = by_id.get(tid)
                if not tt:
                    return None
                return {"direction": direction, "id": tid, "title": tt.get("title", tid),
                        "status": tt.get("status", "proposed"),
                        "has_artifact": (direction, tid) in arts}

            this = by_id.get(task, {})
            m["status"] = this.get("status", m.get("status"))
            m["rework_history"] = this.get("rework_history", [])
            parent = this.get("follow_up_of")
            m["follow_up_of"] = ref(parent) if parent else None
            m["follow_ups"] = [r for r in (ref(x) for x in this.get("follow_ups", [])) if r]
        except Exception:
            pass
    return m


# ---- write-back (Overview drag / Mark Done) + ntfy notification feed ----

def _git_commit(path: Path, msg: str, cwd: Path = MAIN_ROOT) -> None:
    """Persist a dashboard edit. Non-fatal: the file write already took effect for the live server.
    The commit is scoped to `path` (`git commit -- <path>`) so that a concurrent editor's unrelated
    staged changes are never swept into a dashboard commit — an unscoped `git commit` commits the whole
    index, which once clobbered an in-progress hand edit."""
    try:
        subprocess.run(["git", "add", str(path)], cwd=cwd, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", msg, "--", str(path)],
                       cwd=cwd, check=True, capture_output=True)
    except Exception:
        pass


def set_task_status(direction: str, task: str, status: str, note: str | None = None) -> dict:
    """Atomically change a task's status in its direction file (the Overview write-back)."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    found = False
    for t in data.get("tasks", []):
        if t.get("id") == task:
            t["status"] = status
            if note:
                hist = t.get("rework_history") or []
                hist.append({"time": int(time.time()), "note": note})
                t["rework_history"] = hist
            found = True
            break
    if not found:
        return {"ok": False, "error": "no such task"}
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _git_commit(f, f"dashboard: {direction}/{task} -> {status}")
    return {"ok": True}


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s or "untitled"


def resolve_write(rid: str, path: str) -> tuple[Path, Path]:
    """Resolve {rid}/{path} for writing, guarding traversal. Returns (target, owning_root).
    Unlike `resolve`, the file need not already exist."""
    roots = list_roots()
    root = roots.get(rid)
    if root is None:
        raise FileNotFoundError(rid)
    target = (root / path).resolve()
    if os.path.commonpath([str(target), str(root)]) != str(root):
        raise PermissionError(path)
    return target, root


def write_file(rid: str, path: str, content: str) -> dict:
    """Write back an edited markdown doc (training page, report, decision). Markdown only."""
    if not path.endswith(".md"):
        return {"ok": False, "error": "only .md files are editable from the dashboard"}
    target, root = resolve_write(rid, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git_commit(target, f"dashboard: edit {path}", cwd=root)
    return {"ok": True}


def create_task(direction: str, title: str, note: str = "",
                status: str = "proposed", task_id: str | None = None) -> dict:
    """Add a task to an existing direction (Overview authoring)."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    tid = task_id or _slug(title)
    if any(t.get("id") == tid for t in data.get("tasks", [])):
        return {"ok": False, "error": "a task with that id already exists"}
    data.setdefault("tasks", []).append({"id": tid, "title": title, "status": status, "note": note})
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _git_commit(f, f"dashboard: add task {direction}/{tid}")
    return {"ok": True, "id": tid}


def edit_task(direction: str, task: str, title: str | None = None, note: str | None = None) -> dict:
    """Refine an existing task's title and/or note (Overview authoring)."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    found = False
    for t in data.get("tasks", []):
        if t.get("id") == task:
            if title is not None:
                t["title"] = title
            if note is not None:
                t["note"] = note
            found = True
            break
    if not found:
        return {"ok": False, "error": "no such task"}
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _git_commit(f, f"dashboard: edit task {direction}/{task}")
    return {"ok": True}


def create_direction(name: str, summary: str = "",
                     did: str | None = None, status: str = "proposed") -> dict:
    """Create a new direction file (Overview authoring)."""
    did = did or _slug(name)
    f = MAIN_ROOT / "coordination" / "directions" / f"{did}.json"
    if f.is_file():
        return {"ok": False, "error": "a direction with that id already exists"}
    f.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": did, "name": name, "status": status, "summary": summary, "tasks": []}
    f.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _git_commit(f, f"dashboard: add direction {did}")
    return {"ok": True, "id": did}


def delete_task(direction: str, task: str) -> dict:
    """Remove a task from its direction file entirely (the dashboard Delete). Any run artifacts on
    disk are left in place — git history keeps them recoverable and they no longer surface, since the
    board is driven by the direction file, not by orphaned runs."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    tasks = data.get("tasks", [])
    kept = [t for t in tasks if t.get("id") != task]
    if len(kept) == len(tasks):
        return {"ok": False, "error": "no such task"}
    # Sever any follow-up links that referenced the deleted task, so no dangling refs remain.
    for t in kept:
        if t.get("follow_up_of") == task:
            t.pop("follow_up_of", None)
        if task in (t.get("follow_ups") or []):
            t["follow_ups"] = [x for x in t["follow_ups"] if x != task]
    data["tasks"] = kept
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _git_commit(f, f"dashboard: delete task {direction}/{task}")
    return {"ok": True}


def propose_follow_up(direction: str, parent: str, title: str, note: str = "") -> dict:
    """Create a proposed follow-up to an existing task, linked both ways: the parent gains the new id
    in `follow_ups`, the child records `follow_up_of` = parent. Lives in the parent's direction."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    tasks = data.get("tasks", [])
    if not any(t.get("id") == parent for t in tasks):
        return {"ok": False, "error": "no such parent task"}
    existing = {t.get("id") for t in tasks}
    base = _slug(title)
    tid, i = base, 2
    while tid in existing:
        tid, i = f"{base}-{i}", i + 1
    tasks.append({"id": tid, "title": title, "status": "proposed",
                  "note": note, "follow_up_of": parent})
    for t in tasks:
        if t.get("id") == parent:
            fu = t.get("follow_ups") or []
            fu.append(tid)
            t["follow_ups"] = fu
    data["tasks"] = tasks
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _git_commit(f, f"dashboard: follow-up {direction}/{tid} of {parent}")
    return {"ok": True, "id": tid}


def _topic() -> str | None:
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        return topic
    tf = Path.home() / ".learning-taichi" / "ntfy_topic"
    if tf.is_file():
        return tf.read_text("utf-8").strip() or None
    return None


def notifications() -> dict:
    """Pull the recent ntfy message cache for our topic, server-side, so the secret topic never
    reaches the browser."""
    import urllib.request
    topic = _topic()
    if not topic:
        return {"configured": False, "notifications": []}
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
    try:
        since = int(time.time()) - 7 * 86400  # ntfy `since` wants a unix ts / duration in h,m,s
        with urllib.request.urlopen(f"{server}/{topic}/json?poll=1&since={since}", timeout=8) as resp:
            lines = resp.read().decode("utf-8").splitlines()
    except Exception as e:
        return {"configured": True, "notifications": [], "error": str(e)}
    msgs = []
    for ln in lines:
        try:
            m = json.loads(ln)
        except Exception:
            continue
        if m.get("event") != "message":
            continue
        msgs.append({
            "id": m.get("id"), "time": m.get("time", 0), "title": m.get("title", ""),
            "message": m.get("message", ""), "priority": m.get("priority", 3), "tags": m.get("tags", []),
        })
    msgs.sort(key=lambda x: x.get("time", 0), reverse=True)
    return {"configured": True, "notifications": msgs}


def _build_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="learning-taichi data server")
    # Local dev: the Vite dashboard runs on a different port; allow it to fetch from here.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/api/index")
    def api_index():
        return build_index()

    @app.get("/api/data/{rid}/{path:path}")
    def api_data(rid: str, path: str):
        try:
            target = resolve(rid, path)
        except PermissionError:
            raise HTTPException(403, "forbidden")
        except FileNotFoundError:
            raise HTTPException(404, "not found")
        if target.name == "manifest.json":
            m = json.loads(target.read_text("utf-8"))
            rw = rewrite_task if str(m.get("schema_version")) == "2" else rewrite_manifest
            return JSONResponse(rw(m, rid))
        return FileResponse(target)

    @app.get("/api/training")
    def api_training():
        return training_toc()

    @app.get("/api/directions")
    def api_directions():
        return directions_doc()

    @app.get("/api/reports")
    def api_reports():
        return reports_list()

    @app.get("/api/decisions")
    def api_decisions():
        return decisions_list()

    @app.get("/api/overview")
    def api_overview():
        return overview()

    @app.get("/api/task/{direction}/{task}")
    def api_task(direction: str, task: str):
        d = task_detail(direction, task)
        if d is None:
            raise HTTPException(404, "task not found")
        return d

    @app.post("/api/task-status")
    def api_task_status(payload: dict):
        d, t, s = payload.get("direction"), payload.get("task"), payload.get("status")
        if not (d and t and s):
            raise HTTPException(400, "direction, task, status required")
        return set_task_status(d, t, s, payload.get("note"))

    @app.post("/api/file")
    def api_file(payload: dict):
        rid, path, content = payload.get("rid"), payload.get("path"), payload.get("content")
        if rid is None or path is None or content is None:
            raise HTTPException(400, "rid, path, content required")
        try:
            return write_file(rid, path, content)
        except PermissionError:
            raise HTTPException(403, "forbidden")
        except FileNotFoundError:
            raise HTTPException(404, "not found")

    @app.post("/api/task-create")
    def api_task_create(payload: dict):
        d, title = payload.get("direction"), payload.get("title")
        if not (d and title):
            raise HTTPException(400, "direction, title required")
        return create_task(d, title, payload.get("note", ""),
                           payload.get("status", "proposed"), payload.get("id"))

    @app.post("/api/task-edit")
    def api_task_edit(payload: dict):
        d, t = payload.get("direction"), payload.get("task")
        if not (d and t):
            raise HTTPException(400, "direction, task required")
        return edit_task(d, t, payload.get("title"), payload.get("note"))

    @app.post("/api/direction-create")
    def api_direction_create(payload: dict):
        name = payload.get("name")
        if not name:
            raise HTTPException(400, "name required")
        return create_direction(name, payload.get("summary", ""),
                               payload.get("id"), payload.get("status", "proposed"))

    @app.post("/api/task-delete")
    def api_task_delete(payload: dict):
        d, t = payload.get("direction"), payload.get("task")
        if not (d and t):
            raise HTTPException(400, "direction, task required")
        return delete_task(d, t)

    @app.post("/api/task-follow-up")
    def api_task_follow_up(payload: dict):
        d, p, title = payload.get("direction"), payload.get("parent"), payload.get("title")
        if not (d and p and title):
            raise HTTPException(400, "direction, parent, title required")
        return propose_follow_up(d, p, title, payload.get("note", ""))

    @app.get("/api/notifications")
    def api_notifications():
        return notifications()

    return app


# Lazily built so the helper functions import cleanly without FastAPI installed (tests).
app = None
try:  # pragma: no cover - import-time convenience
    app = _build_app()
except Exception:
    app = None


if __name__ == "__main__":
    import uvicorn

    if app is None:
        app = _build_app()
    print(f"learning-taichi data server -> http://localhost:{PORT}  (root: {MAIN_ROOT})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
