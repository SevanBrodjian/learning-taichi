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
import subprocess
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
            })
        directions.append({
            "id": did, "name": d.get("name", did), "status": d.get("status", "proposed"),
            "summary": d.get("summary", ""), "tasks": tasks,
        })
    listed = {(dd["id"], t["id"]) for dd in directions for t in dd["tasks"]}
    orphans = [{"direction": k[0], "task": k[1]} for k in arts if k not in listed]
    return {"directions": directions, "orphans": orphans}


def task_detail(direction: str, task: str) -> dict | None:
    info = _v2_tasks().get((direction, task))
    if not info:
        return None
    target = resolve(info["root"], info["rel"])
    return rewrite_task(json.loads(target.read_text("utf-8")), info["root"])


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
