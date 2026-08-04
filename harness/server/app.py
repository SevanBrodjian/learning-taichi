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


def _json_safe(o):
    """Recursively replace NaN / Infinity floats with None. A worker's manifest can carry a NaN metric
    (e.g. a degenerate held-out value); Python's json.load accepts it, but Starlette's JSONResponse
    serializes with allow_nan=False and 500s — which hangs the task page forever. Sanitizing here makes
    the data server resilient to any NaN-bearing manifest instead of dying on it."""
    import math
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


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
    """The inbox: open decisions awaiting the user (coordination/decisions/*.md). A file whose name
    contains 'contract' is a task contract the user can Approve/Reject before the run spawns."""
    d = MAIN_ROOT / "coordination" / "decisions"
    items = []
    if d.is_dir():
        for f in sorted(d.glob("*.md")):
            if f.name.lower() == "readme.md":
                continue
            try:
                txt = f.read_text("utf-8", errors="ignore")
            except Exception:
                txt = ""
            mt = re.search(r"auto_run_at:\s*(\d+)", txt)
            items.append({"id": f.stem, "title": f.stem.replace("-", " ").replace("_", " / "),
                          "url": _shared_url(f"coordination/decisions/{f.name}"),
                          "kind": "contract" if "contract" in f.stem.lower() else "note",
                          "resolved": "**Resolution:" in txt,
                          "auto_run_at": int(mt.group(1)) if mt else None})
    return {"decisions": items}


def resolve_decision(decision_id: str, resolution: str, note: str = "") -> dict:
    """Record the user's Approve/Reject on a decision (esp. a task contract) by appending a resolution
    marker to the decision file. The orchestrator reads this to know whether to spawn the run."""
    f = MAIN_ROOT / "coordination" / "decisions" / f"{decision_id}.md"
    if not f.is_file():
        return {"ok": False, "error": "no such decision"}
    res = "APPROVED" if resolution == "approve" else "REJECTED"
    line = f"\n\n**Resolution: {res}**"
    if note:
        line += f" — {note}"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    _git_commit(f, f"dashboard: decision {decision_id} -> {res}")
    return {"ok": True}


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


def live_statuses() -> dict:
    """(direction, task) -> live status a worker wrote at runs/<dir>/<task>/status.json.

    This is ephemeral, gitignored, per-machine state (NOT a manifest field): a running worker calls
    harness/tools/task_status.py a handful of times to say what step it is on, and this surfaces that
    on the board so an Active task reads as *running* with a one-line note, not just "Active". A stale
    file (old `updated`) means the worker is likely gone; the client dims it accordingly using `age`.
    """
    out: dict = {}
    now = int(time.time())
    for rid, root in list_roots().items():
        # runs/<direction>/<task>/status.json — directions and task ids are flat slugs (no slashes).
        for sf in sorted(root.glob("runs/*/*/status.json")):
            parts = sf.relative_to(root).parts  # ("runs", <dir>, <task>, "status.json")
            if len(parts) != 4:
                continue
            d, t = parts[1], parts[2]
            if (d, t) in out:
                continue
            try:
                s = json.loads(sf.read_text("utf-8"))
            except Exception:
                continue
            updated = int(s.get("updated", 0)) or 0
            out[(d, t)] = {
                "state": s.get("state", "running"),
                "step": s.get("step", ""),
                "updated": updated,
                "age": (now - updated) if updated else None,
            }
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
    live = live_statuses()
    _tidx = _task_index()          # built once; parent refs resolve across directions against it
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
                "effort": t.get("effort", "standard"),
                "budget_minutes": t.get("budget_minutes") or default_budget(t.get("effort", "standard")),
                "tags": t.get("tags", []),
                "live": live.get((did, tid)),
                "has_artifact": has, "detail": f"/api/task/{did}/{tid}" if has else None,
                "rework_history": t.get("rework_history", []),
                # Normalized + direction-resolved so the Map can draw cross-direction, typed edges.
                "follow_up_of": _overview_parents(t, did, _tidx),
                "follow_ups": t.get("follow_ups", []),
                "notes": t.get("notes", []),
            })
        directions.append({
            "id": did, "name": d.get("name", did), "status": d.get("status", "proposed"),
            "summary": d.get("summary", ""), "tasks": tasks,
        })
    listed = {(dd["id"], t["id"]) for dd in directions for t in dd["tasks"]}
    orphans = [{"direction": k[0], "task": k[1]} for k in arts if k not in listed]
    return {"directions": directions, "orphans": orphans}


# Edge kinds. A follow-up that overturned its parent must not read like one that built on it.
EDGE_KINDS = ("extends", "re-does", "refutes", "applies", "prerequisite-of")
DEFAULT_KIND = "extends"


def _overview_parents(t: dict, did: str, idx: dict) -> list:
    """Parents of `t`, resolved to a concrete direction so the graph can cross direction boundaries.
    Unresolvable refs are dropped rather than drawn as dangling edges. `idx` is built ONCE by the caller —
    the overview is polled every few seconds, so this must not re-read the direction files per task."""
    out = []
    for p in _as_parent_list(t.get("follow_up_of")):
        r = _resolve_parent(p, did, idx)
        if r:
            out.append({"id": r[1], "dir": r[0], "kind": p["kind"]})
    return out


def _as_parent_list(v) -> list:
    """Normalize `follow_up_of` into a uniform list of {id, dir, kind} dicts.

    Three historical shapes are accepted, so nothing on disk has to be migrated in lockstep:
      "task-id"                            legacy single parent, same direction
      ["task-id", ...]                     legacy multi-parent, same direction
      [{"id":..,"dir":..,"kind":..}, ...]  current: cross-direction and typed
    `dir` may be None, meaning "resolve by id across all directions"."""
    if not v:
        return []
    raw = v if isinstance(v, list) else [v]
    out = []
    for x in raw:
        if not x:
            continue
        if isinstance(x, str):
            out.append({"id": x, "dir": None, "kind": DEFAULT_KIND})
        elif isinstance(x, dict) and x.get("id"):
            out.append({"id": x["id"], "dir": x.get("dir"),
                        "kind": x.get("kind") if x.get("kind") in EDGE_KINDS else DEFAULT_KIND})
    return out


def _parent_ids(v) -> list:
    """Just the task ids, for code that only cares about identity (pruning, dedupe)."""
    return [p["id"] for p in _as_parent_list(v)]


def _task_index() -> dict:
    """(direction, task_id) -> task dict, across every direction file. Lets a parent reference resolve
    across directions, which is what makes the graph a real lineage instead of five disconnected trees."""
    idx = {}
    ddir = MAIN_ROOT / "coordination" / "directions"
    if ddir.is_dir():
        for f in sorted(ddir.glob("*.json")):
            try:
                d = json.loads(f.read_text("utf-8"))
            except Exception:
                continue
            did = d.get("id", f.stem)
            for t in d.get("tasks", []):
                if t.get("id"):
                    idx[(did, t["id"])] = t
    return idx


def _resolve_parent(p: dict, fallback_dir: str, idx: dict):
    """Resolve one normalized parent ref to (direction, task) or None. An explicit `dir` wins; then the
    referring task's own direction; then a unique match on id anywhere."""
    if p.get("dir") and (p["dir"], p["id"]) in idx:
        return p["dir"], p["id"]
    if (fallback_dir, p["id"]) in idx:
        return fallback_dir, p["id"]
    hits = [k for k in idx if k[1] == p["id"]]
    return hits[0] if len(hits) == 1 else None


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
            tidx = _task_index()

            def ref(tid, tdir=None, kind=None):
                """Resolve a task reference, possibly in ANOTHER direction (the graph is cross-direction
                now), into the shape the task page renders."""
                d2 = tdir or direction
                tt = tidx.get((d2, tid))
                if tt is None and (direction, tid) in tidx:
                    d2, tt = direction, tidx[(direction, tid)]
                if tt is None:
                    hits = [k for k in tidx if k[1] == tid]
                    if len(hits) != 1:
                        return None
                    d2, tt = hits[0][0], tidx[hits[0]]
                out = {"direction": d2, "id": tid, "title": tt.get("title", tid),
                       "status": tt.get("status", "proposed"),
                       "has_artifact": (d2, tid) in arts}
                if kind:
                    out["kind"] = kind
                return out

            this = by_id.get(task, {})
            m["status"] = this.get("status", m.get("status"))
            m["effort"] = this.get("effort", "standard")
            m["budget_minutes"] = this.get("budget_minutes") or default_budget(this.get("effort", "standard"))
            m["live"] = live_statuses().get((direction, task))
            m["rework_history"] = this.get("rework_history", [])
            m["notes"] = this.get("notes", [])          # the user's own passive margin notes
            # follow_up_of is now a LIST of resolved refs (a proposal may follow up on several parents).
            m["follow_up_of"] = [r for r in (ref(p["id"], p.get("dir"), p.get("kind"))
                                             for p in _as_parent_list(this.get("follow_up_of"))) if r]
            m["follow_ups"] = [r for r in (ref(x if isinstance(x, str) else x.get("id"),
                                               None if isinstance(x, str) else x.get("dir"))
                                           for x in this.get("follow_ups", [])) if r]
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


EFFORTS = ("quick", "standard", "deep")
# Default adaptive time budget (minutes) per effort tier — a soft expectation, not a hard cap. The
# orchestrator watches a running worker against this (harness/tools/watch_worker.py) and intervenes if it
# goes silent or blows past it. The user can override per task on the dashboard.
EFFORT_BUDGET = {"quick": 15, "standard": 40, "deep": 90}


def default_budget(effort: str) -> int:
    return EFFORT_BUDGET.get(effort or "standard", 40)


def set_task_budget(direction: str, task: str, minutes) -> dict:
    """Set a task's adaptive time budget in minutes (the soft expectation the orchestrator watches against)."""
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        return {"ok": False, "error": "minutes must be an integer"}
    minutes = max(1, min(600, minutes))
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    for t in data.get("tasks", []):
        if t.get("id") == task:
            t["budget_minutes"] = minutes
            f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            _git_commit(f, f"dashboard: {direction}/{task} budget -> {minutes}m")
            return {"ok": True}
    return {"ok": False, "error": "no such task"}


def add_task_note(direction: str, task: str, text: str, author: str = "Sevan") -> dict:
    """Append a passive note to a task. A note NEVER changes status -- it is the user's own margin
    comment (a question, a doubt, a conclusion they reached) attached to the task and kept with it
    across re-runs, which is why it lives in the direction file and not in the run manifest."""
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "empty note"}
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    for t in data.get("tasks", []):
        if t.get("id") == task:
            note = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "author": author, "text": text}
            t.setdefault("notes", []).append(note)
            f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            _git_commit(f, f"note: {direction}/{task} — {text[:60]}")
            return {"ok": True, "note": note}
    return {"ok": False, "error": "no such task"}


def delete_task_note(direction: str, task: str, ts: str) -> dict:
    """Remove a note by timestamp."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    for t in data.get("tasks", []):
        if t.get("id") == task:
            before = len(t.get("notes", []))
            t["notes"] = [n for n in t.get("notes", []) if n.get("ts") != ts]
            if len(t["notes"]) == before:
                return {"ok": False, "error": "no such note"}
            f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            _git_commit(f, f"note: removed one from {direction}/{task}")
            return {"ok": True}
    return {"ok": False, "error": "no such task"}


def set_task_effort(direction: str, task: str, effort: str) -> dict:
    """Set a task's intensity tier (quick | standard | deep). The orchestrator reads this when it
    spawns the worker: it picks the model and reasoning effort and how long the worker is expected to
    persist (see the /execute skill and coordination/tasks/_TEMPLATE.md)."""
    if effort not in EFFORTS:
        return {"ok": False, "error": f"effort must be one of {EFFORTS}"}
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    for t in data.get("tasks", []):
        if t.get("id") == task:
            t["effort"] = effort
            f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            _git_commit(f, f"dashboard: {direction}/{task} effort -> {effort}")
            return {"ok": True}
    return {"ok": False, "error": "no such task"}


def set_task_tags(direction: str, task: str, tags: list) -> dict:
    """Set a task's tags (the sorting/filtering axis that replaces directions-as-containers)."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    clean = [str(t).strip() for t in (tags or []) if str(t).strip()]
    for t in data.get("tasks", []):
        if t.get("id") == task:
            t["tags"] = list(dict.fromkeys(clean))
            f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            _git_commit(f, f"dashboard: {direction}/{task} tags")
            return {"ok": True}
    return {"ok": False, "error": "no such task"}


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
                status: str = "proposed", task_id: str | None = None,
                effort: str = "standard") -> dict:
    """Add a task to an existing direction (Overview authoring)."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    tid = task_id or _slug(title)
    if any(t.get("id") == tid for t in data.get("tasks", [])):
        return {"ok": False, "error": "a task with that id already exists"}
    effort = effort if effort in EFFORTS else "standard"
    data.setdefault("tasks", []).append(
        {"id": tid, "title": title, "status": status, "note": note, "effort": effort})
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
    data["tasks"] = kept
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Sever every reference to the deleted task so no dangling edge remains. Edges are CROSS-DIRECTION
    # now, so this has to sweep all direction files, not just this one.
    touched = [f]
    for g in sorted((MAIN_ROOT / "coordination" / "directions").glob("*.json")):
        try:
            gd = json.loads(g.read_text("utf-8"))
        except Exception:
            continue
        gdid = gd.get("id", g.stem)
        changed = False
        for t in gd.get("tasks", []):
            parents = _as_parent_list(t.get("follow_up_of"))
            keptp = [p for p in parents
                     if not (p["id"] == task and (p.get("dir") in (None, direction) or gdid == direction))]
            if len(keptp) != len(parents):
                changed = True
                if keptp:
                    t["follow_up_of"] = keptp
                else:
                    t.pop("follow_up_of", None)
            fu = t.get("follow_ups") or []
            keptf = [x for x in fu if (x.get("id") if isinstance(x, dict) else x) != task]
            if len(keptf) != len(fu):
                changed = True
                t["follow_ups"] = keptf
        if changed:
            g.write_text(json.dumps(gd, indent=2) + "\n", encoding="utf-8")
            if g != f:
                touched.append(g)
    for g in touched:
        _git_commit(g, f"dashboard: delete task {direction}/{task}")
    return {"ok": True}


def propose_follow_up(direction: str, parents, title: str, note: str = "") -> dict:
    """Create a proposed follow-up to one OR MORE existing tasks in the same direction, linked both ways:
    every parent gains the new id in its `follow_ups`, and the child records `follow_up_of` = the parent
    id (a single string) or the list of parent ids when there are several. `parents` accepts a single id
    or a list of ids. Lives in the parents' direction (the graph is direction-local)."""
    f = MAIN_ROOT / "coordination" / "directions" / f"{direction}.json"
    if not f.is_file():
        return {"ok": False, "error": "no such direction"}
    data = json.loads(f.read_text("utf-8"))
    tasks = data.get("tasks", [])
    parents = [parents] if isinstance(parents, str) else list(parents or [])
    parents = list(dict.fromkeys(p for p in parents if p))  # dedupe, drop empties, preserve order
    if not parents:
        return {"ok": False, "error": "at least one parent task required"}
    existing = {t.get("id") for t in tasks}
    missing = [p for p in parents if p not in existing]
    if missing:
        return {"ok": False, "error": f"no such parent task(s): {', '.join(missing)}"}
    base = _slug(title)
    tid, i = base, 2
    while tid in existing:
        tid, i = f"{base}-{i}", i + 1
    # Written in the normalized {id, dir, kind} form. The kind is a PLACEHOLDER: the user's citation is a
    # hint, and the orchestrator re-derives the real edges (and their kinds) when it reviews the task.
    tasks.append({"id": tid, "title": title, "status": "proposed", "note": note,
                  "follow_up_of": [{"id": p, "dir": direction, "kind": DEFAULT_KIND} for p in parents]})
    pset = set(parents)
    for t in tasks:
        if t.get("id") in pset:
            fu = t.get("follow_ups") or []
            if tid not in fu:
                fu.append(tid)
            t["follow_ups"] = fu
    data["tasks"] = tasks
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _git_commit(f, f"dashboard: follow-up {direction}/{tid} of {', '.join(parents)}")
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
            return JSONResponse(_json_safe(rw(m, rid)))
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

    @app.get("/api/definitions")
    def api_definitions():
        """The standardization registry (spec/registry/*.json) flattened into hoverable terms, so the
        dashboard can define a metric or a material in place instead of every task reinventing them.
        Metrics are hand-authored; materials are generated from sim.physics by harness/tools/sync_registry.py."""
        reg = MAIN_ROOT / "spec" / "registry"
        terms: dict = {}
        if not reg.is_dir():
            return terms
        try:
            mf = reg / "metrics.json"
            if mf.is_file():
                terms.update({k: v for k, v in json.loads(mf.read_text(encoding="utf-8")).items()
                              if not k.startswith("_")})
        except Exception:
            pass
        try:
            af = reg / "materials.json"
            if af.is_file():
                doc = json.loads(af.read_text(encoding="utf-8"))
                meaning = doc.get("_param_meaning", {})
                for name, e in (doc.get("materials") or {}).items():
                    params = ", ".join(f"{k}={e[k]}" for k in ("E", "dt", "xi", "tc", "ts") if k in e)
                    drift = e.get("_known_drift") or []
                    caution = None
                    if drift:
                        caution = ("KNOWN DRIFT -- tasks that reimplemented instead of importing: "
                                   + "; ".join(f"{d['where']} uses {d['param']}={d['actual']} "
                                               f"(canonical {d['canonical']})" for d in drift))
                    terms[name] = {
                        "label": name,
                        "short": f"Canonical {name}. Frozen in sim/physics/core.py; "
                                 f"every task must import these, never redefine them.",
                        "formula": params,
                        "units": "; ".join(f"{k}: {v}" for k, v in meaning.items() if k in e),
                        "range": f"physics_version {doc.get('physics_version', '?')}",
                        "source": doc.get("_source", "sim/physics/core.py"),
                        **({"caution": caution} if caution else {}),
                    }
        except Exception:
            pass
        return terms

    @app.get("/api/decisions")
    def api_decisions():
        return decisions_list()

    @app.post("/api/decision-resolve")
    def api_decision_resolve(payload: dict):
        did, resn = payload.get("id"), payload.get("resolution")
        if not (did and resn in ("approve", "reject")):
            raise HTTPException(400, "id and resolution (approve|reject) required")
        return resolve_decision(did, resn, payload.get("note", ""))

    @app.get("/api/overview")
    def api_overview():
        return overview()

    @app.get("/api/task/{direction}/{task}")
    def api_task(direction: str, task: str):
        d = task_detail(direction, task)
        if d is None:
            raise HTTPException(404, "task not found")
        return JSONResponse(_json_safe(d))

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
                           payload.get("status", "proposed"), payload.get("id"),
                           payload.get("effort", "standard"))

    @app.post("/api/task-effort")
    def api_task_effort(payload: dict):
        d, t, e = payload.get("direction"), payload.get("task"), payload.get("effort")
        if not (d and t and e):
            raise HTTPException(400, "direction, task, effort required")
        return set_task_effort(d, t, e)

    @app.post("/api/task-note")
    def api_task_note(payload: dict):
        d, t = payload.get("direction"), payload.get("task")
        if not (d and t):
            raise HTTPException(400, "direction, task required")
        return add_task_note(d, t, payload.get("text", ""), payload.get("author", "Sevan"))

    @app.post("/api/task-note-delete")
    def api_task_note_delete(payload: dict):
        d, t, ts = payload.get("direction"), payload.get("task"), payload.get("ts")
        if not (d and t and ts):
            raise HTTPException(400, "direction, task, ts required")
        return delete_task_note(d, t, ts)

    @app.post("/api/task-tags")
    def api_task_tags(payload: dict):
        d, t = payload.get("direction"), payload.get("task")
        if not (d and t):
            raise HTTPException(400, "direction, task required")
        return set_task_tags(d, t, payload.get("tags", []))

    @app.post("/api/task-budget")
    def api_task_budget(payload: dict):
        d, t, mn = payload.get("direction"), payload.get("task"), payload.get("minutes")
        if not (d and t and mn is not None):
            raise HTTPException(400, "direction, task, minutes required")
        return set_task_budget(d, t, mn)

    @app.get("/api/health")
    def api_health():
        return {"ok": True}

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
        d, title = payload.get("direction"), payload.get("title")
        # `parents` (list) is preferred; `parent` (single) stays accepted for older callers.
        parents = payload.get("parents")
        if parents is None and payload.get("parent"):
            parents = [payload.get("parent")]
        if not (d and parents and title):
            raise HTTPException(400, "direction, parent(s), title required")
        return propose_follow_up(d, parents, title, payload.get("note", ""))

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
