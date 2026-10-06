#!/usr/bin/env python3
"""Export a PUBLIC, READ-ONLY, STATIC snapshot of the dashboard ("the lab") and the WebGPU demo.

The live dashboard reads a live data server (harness/server/app.py). This tool freezes the GET side of
that server into plain files, builds the dashboard in its read-only public mode against them, builds the
standalone demo, and then refuses to finish if anything private survived.

    <out>/index.html, assets/       the dashboard, `npm run build:public` (relative base: any subpath works)
    <out>/snapshot/meta.json        snapshot date + which opt-in views are present
    <out>/snapshot/api/*.json       /api/index, overview, directions, reports, training, definitions,
                                    epochs, tags (+ decisions only if opted in)
    <out>/snapshot/api/task/<dir>/<task>.json        every task with a result
    <out>/snapshot/api/data/<rid>/<path>             only the files those responses reference
    <out>/demo/index.html           the WebGPU MPM demo, one self-contained file (`npm run build:demo`)

Responses are computed by IMPORTING THE SERVER and calling its own GET route handlers, so the snapshot
has exactly the live shapes. The server is pointed at the data checkout (--source, default: the main
worktree) and restricted to it alone; live worker status is never exported.

NEVER EXPORTED: /api/notebook and reports/notebook/ (Sevan's scratchpad), /api/notifications (the ntfy
topic is a secret), /api/health, every POST endpoint, coordination/decisions/ (the approval inbox) unless
--include-decisions, reports/research_report.md (hand-written by Sevan) unless --include-research-report,
orphan runs that are not on the board, and any of Sevan's task notes the personal-content check flags
(unless --include-flagged).

PRIVACY PASS. Every exported string is scrubbed (absolute/home paths, tailnet hosts, local machine names,
tailnet/private IPs, known emails, browser user-agent strings, ntfy topics, token/key shapes), every
media file is re-encoded or re-saved without metadata, and the finished output tree is scanned for the
same patterns. Any hit is a hard failure (exit 2): fix the source or extend the scrub, never ignore it.

Run from the checkout that holds this file, with the data checkout's venv (FastAPI + Pillow + an ffmpeg
via imageio-ffmpeg) and Node on PATH for the two builds:

    python harness/tools/export_public.py --source <data checkout> --out <lab dir>

Useful flags: --skip-build (snapshot only), --date YYYY-MM-DD, --max-file-mb 8, --budget-mb 75,
--report-json <file> (machine-readable summary, kept OUT of the bundle: it quotes flagged notes).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve()
CHECKOUT = HERE.parents[2]                      # the checkout holding this tool, the server and dashboard
DASHBOARD = CHECKOUT / "harness" / "dashboard"
SERVER_PY = CHECKOUT / "harness" / "server" / "app.py"

# The top-level things this tool owns inside --out. Only these are ever deleted/replaced there.
OWNED = ("index.html", "assets", "snapshot", "demo")
REPORT_NAME = "PUBLIC_EXPORT_REPORT.md"      # written by hand next to the bundle; not part of it

MEDIA_VIDEO = {".mp4", ".webm", ".mov", ".m4v"}
MEDIA_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif"}
TEXT_DATA = {".json", ".md", ".txt", ".csv", ".svg"}
BINARY_DATA = {".f32", ".i32", ".u32", ".f16", ".bin"}
EXPORTABLE = MEDIA_VIDEO | MEDIA_IMAGE | TEXT_DATA | BINARY_DATA

# Paths (relative to the data checkout) that may never be exported, whatever references them.
DENY_ALWAYS = (
    re.compile(r"^reports/notebook(/|$)"),
    re.compile(r"^\.git(/|$)"),
    re.compile(r"(^|/)\.venv(/|$)"),
    re.compile(r"(^|/)node_modules(/|$)"),
    re.compile(r"^agents/"),
    re.compile(r"^coordination/(shared_memory|tasks|threads)/"),
    re.compile(r"(^|/)status\.json$"),
)
RESEARCH_REPORT = "reports/research_report.md"
EPOCH_PRIVATE = re.compile(r"^coordination/epochs/[^/]+/(report|verdict|EXAM)\.md$")
DECISIONS_DIR = re.compile(r"^coordination/decisions/")
# Anything referenced from outside these roots is refused (and reported) rather than silently copied.
ALLOWED_ROOTS = ("runs/", "reports/training/", "coordination/research_directions.md", "spec/registry/")


# ------------------------------------------------------------------------------------------------ util

def die(msg: str, code: int = 1):
    print(f"\nEXPORT FAILED: {msg}", file=sys.stderr)
    sys.exit(code)


def git(*args, cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout
    except Exception:
        return ""


def main_worktree(start: Path) -> Path:
    """The first entry of `git worktree list` is the main checkout -- where the live data lives."""
    for line in git("worktree", "list", "--porcelain", cwd=start).splitlines():
        if line.startswith("worktree "):
            return Path(line[len("worktree "):].strip()).resolve()
    return start


def human(n: int) -> str:
    return f"{n / 1048576:.2f} MB" if n >= 1048576 else f"{n / 1024:.1f} KB"


# ----------------------------------------------------------------------------- privacy: local secrets

def local_literals(source: Path) -> dict[str, list[str]]:
    """Identifiers of THIS machine/tailnet/person that must never appear in the bundle. Gathered at run
    time so the list is never written down anywhere. Values are only ever compared, never printed."""
    lit: dict[str, set[str]] = {"host": set(), "ip": set(), "email": set(), "ntfy": set(), "user": set()}

    def add(kind, v, min_len=4):
        v = (v or "").strip().rstrip(".")
        if len(v) >= min_len and v.lower() not in {"localhost", "sevan", "mac", "owner", "user"}:
            lit[kind].add(v)

    for cmd in (["hostname"], ["scutil", "--get", "ComputerName"], ["scutil", "--get", "LocalHostName"]):
        try:
            add("host", subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout)
        except Exception:
            pass
    try:
        ts = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True,
                                       timeout=10).stdout or "{}")
        suffix = (ts.get("MagicDNSSuffix") or "").strip(".")
        if suffix:
            add("host", suffix)
            add("host", suffix.split(".")[0])
        for n in [ts.get("Self") or {}] + list((ts.get("Peer") or {}).values()):
            dns = (n.get("DNSName") or "").strip(".")
            add("host", dns)
            if dns:
                add("host", dns.split(".")[0], min_len=6)
            add("host", n.get("HostName"), min_len=6)
            for ip in n.get("TailscaleIPs") or []:
                add("ip", ip, min_len=7)
    except Exception:
        pass
    for key in ("user.email",):
        add("email", git("config", key, cwd=source))
    for em in set(git("log", "--format=%ae%n%ce", cwd=source).split()):
        add("email", em)
    topic = os.environ.get("NTFY_TOPIC")
    tf = Path.home() / ".learning-taichi" / "ntfy_topic"
    if not topic and tf.is_file():
        topic = tf.read_text("utf-8").strip()
    if topic:
        add("ntfy", topic, min_len=3)
    add("user", Path.home().name, min_len=3)
    return {k: sorted(v, key=len, reverse=True) for k, v in lit.items()}


# Generic shapes, scrubbed in every exported string AND scanned for in the finished tree.
_SEP = r"[\\/]{1,2}"
REPO_ROOT_ABS = re.compile(
    rf"(?:[A-Za-z]:)?{_SEP}(?:Users|home){_SEP}[^\\/\s\"'<>`]+{_SEP}(?:[^\\/\s\"'<>`]+{_SEP})*?"
    rf"learning-taichi(?:-public)?(?:{_SEP}\.claude{_SEP}worktrees{_SEP}[^\\/\s\"'<>`]+)?(?:{_SEP}|(?=[\s\"'<>`]|$))")
HOME_ABS = re.compile(rf"(?:[A-Za-z]:)?{_SEP}(?:Users|home){_SEP}[^\\/\s\"'<>`]+")
WIN_ABS = re.compile(r"\b[A-Za-z]:[\\/]{1,2}(?=[A-Za-z0-9_.-]+[\\/])")
# A home path flattened into a directory name by a tool (e.g. "C--Users-<name>-Projects-...").
SLUGGED_HOME = re.compile(r"(?:\b[A-Za-z]-|\B)-?-Users-[A-Za-z0-9]+(?:-[A-Za-z0-9._]+)*")
TS_NET = re.compile(r"\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.ts\.net\.?", re.I)
NTFY_URL = re.compile(r"(ntfy\.sh/)[A-Za-z0-9_-]{3,}")
USER_AGENT = re.compile(r"Mozilla/5\.0 \([^)]*\)[^\"'<>\n]*")
PRIVATE_IP = re.compile(
    r"(?<![\w./-])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|169\.254)"
    r"\.\d{1,3}\.\d{1,3}(?![\w.])")
SECRET_SHAPES = [
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"), re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"), re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"), re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+_\-]{16,}"),
]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?!(?:png|jpe?g|gif|webp|svg|mp4|webm|json|js|css|md|html?|f32|i32)\b)[A-Za-z]{2,}\b")
# Everything else that is a public IPv4 literal (not loopback/unspecified) is reported for review.
ANY_IPV4 = re.compile(r"(?<![\w./-])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")
IPV4_OK = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}


class Scrubber:
    def __init__(self, literals: dict[str, list[str]]):
        self.lit = literals
        self.hits: dict[str, int] = {}

    def _count(self, kind, n):
        if n:
            self.hits[kind] = self.hits.get(kind, 0) + n

    def text(self, s: str) -> str:
        if not s:
            return s
        s, n = USER_AGENT.subn("[browser user agent removed]", s); self._count("user-agent", n)
        s, n = REPO_ROOT_ABS.subn("", s); self._count("abs-repo-path", n)
        s, n = HOME_ABS.subn("~", s); self._count("home-path", n)
        s, n = SLUGGED_HOME.subn("[path]", s); self._count("slugged-home-path", n)
        s, n = TS_NET.subn("[host]", s); self._count("tailnet-host", n)
        s, n = NTFY_URL.subn(r"\1[topic]", s); self._count("ntfy-url", n)
        s, n = PRIVATE_IP.subn("[ip]", s); self._count("private-ip", n)
        for pat in SECRET_SHAPES:
            s, n = pat.subn("[redacted]", s); self._count("secret-shape", n)
        for kind in ("ntfy", "email", "ip", "host"):
            for v in self.lit.get(kind, []):
                if v in s:
                    self._count(f"local-{kind}", s.count(v))
                    s = s.replace(v, "[redacted]")
                elif kind == "host" and v.lower() in s.lower():
                    s, n = re.compile(re.escape(v), re.I).subn("[redacted]", s); self._count("local-host", n)
        return s


def scan_text(t: str, literals: dict[str, list[str]], binary: bool = False) -> list[tuple[str, str]]:
    """Return (category, excerpt) for every forbidden shape left in a string. For a binary file `t` is its
    printable runs, where compressed bytes routinely spell short "x@y.zz" / dotted-quad shapes by chance,
    so only the unambiguous shapes (paths, hosts, URLs, user agents, key prefixes, local literals) apply."""
    out = []

    def hit(cat, m_start, m_end):
        out.append((cat, t[max(0, m_start - 40):m_end + 40].replace("\n", " ")))

    shapes = [("abs-home-path", HOME_ABS), ("slugged-home-path", SLUGGED_HOME), ("tailnet-host", TS_NET),
              ("ntfy-url", NTFY_URL), ("user-agent", USER_AGENT)]
    if not binary:
        shapes += [("windows-abs-path", WIN_ABS), ("private-ip", PRIVATE_IP), ("email", EMAIL)]
    for cat, pat in shapes:
        for m in pat.finditer(t):
            hit(cat, m.start(), m.end())
    for pat in (SECRET_SHAPES[:-1] if binary else SECRET_SHAPES):
        for m in pat.finditer(t):
            hit("secret-shape", m.start(), m.end())
    if not binary:
        for m in ANY_IPV4.finditer(t):
            if m.group(0) not in IPV4_OK:
                hit("ipv4-literal", m.start(), m.end())
    low = t.lower()
    for kind in ("ntfy", "email", "ip", "host"):
        for v in literals.get(kind, []):
            i = low.find(v.lower())
            if i >= 0:
                # Never echo the secret itself into the log.
                out.append((f"local-{kind}", f"<{kind} literal at offset {i}>"))
    return out


def scan_tree(root: Path, literals, exclude: set[str]) -> dict:
    """Scan every file under `root`. Text files are scanned whole; binary files by their printable runs
    (container metadata, PNG text chunks). Returns {files_scanned, hits:[{file, category, excerpt}]}."""
    hits, n = [], 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in exclude:
            continue
        n += 1
        data = p.read_bytes()
        binary = p.suffix.lower() not in {".html", ".js", ".css", ".json", ".md", ".txt", ".svg", ".csv",
                                          ".webmanifest"}
        if not binary:
            text = data.decode("utf-8", errors="replace")
        else:
            text = "\n".join(m.decode("ascii") for m in re.findall(rb"[\x20-\x7e]{8,}", data))
        for cat, ex in scan_text(text, literals, binary=binary):
            hits.append({"file": rel, "category": cat, "excerpt": ex[:200]})
    return {"files_scanned": n, "hits": hits}


# ------------------------------------------------------------------- personal-content check (notes)

# Sevan's own words on the board: a task's seed `note`, its `rework_history` send-back notes and its
# margin `notes`. Research instructions stay; anything that reads as personal is flagged and, by
# default, left out. The word list is deliberately broad: a false flag costs one line in the report.
PERSONAL = re.compile(r"\b(" + "|".join([
    # emotional / venting
    r"frustrat\w*", r"annoy\w*", r"angry", r"upset", r"sad", r"depress\w*", r"anxi\w*", r"stress\w*",
    r"tired", r"exhaust\w*", r"burn(?:ed|t)?[ -]?out", r"hate\w*", r"sorry", r"ugh+", r"wtf", r"damn\w*",
    r"fuck\w*", r"shit\w*", r"crap\w*", r"stupid", r"idiot\w*", r"worr(?:y|ied|ies)", r"scared", r"lonely",
    r"overwhelm\w*", r"cry\w*", r"miserable", r"hopeless",
    # other people
    r"girlfriend", r"boyfriend", r"wife", r"husband", r"mom", r"dad", r"mother", r"father", r"parents?",
    r"brother", r"sister", r"friends?", r"roommates?", r"boss", r"manager", r"advis[eo]r", r"professor",
    r"prof", r"cowork\w*", r"colleagues?", r"labmates?", r"he", r"she", r"him", r"her",
    # private life / plans
    r"interview\w*", r"jobs?", r"salary", r"recruit\w*", r"resign\w*", r"visa", r"doctor", r"therap\w*",
    r"medical", r"sick", r"hospital", r"vacation", r"holiday", r"birthday", r"wedding", r"rent",
    r"money", r"bank", r"loan", r"internship", r"application", r"apply(?:ing)?", r"grad school", r"phd",
    r"date", r"dating", r"personal", r"private", r"secret", r"phone", r"address",
]) + r")\b", re.I)


def personal_hits(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in PERSONAL.finditer(text or "")})


# ---------------------------------------------------------------------------------- server bridge

def load_server(source: Path):
    """Import harness/server/app.py and aim it at `source` ONLY: no other worktree, no live worker
    status. Returns (module, {GET path: endpoint}, rid)."""
    spec = importlib.util.spec_from_file_location("lt_server_for_export", SERVER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rid = mod._root_id("main", source)
    mod.MAIN_ROOT = source
    mod.TAGS_FILE = source / "coordination" / "tags.json"
    mod.list_roots = lambda: {rid: source}
    mod.live_statuses = lambda: {}          # ephemeral per-machine worker state: never part of a snapshot
    app = mod._build_app()
    routes = {}
    for r in app.routes:
        if "GET" in (getattr(r, "methods", None) or set()):
            routes[r.path] = r.endpoint
    return mod, routes, rid


def call_json(mod, routes, route, **kw):
    """Call a GET route handler and return what the client would have parsed."""
    res = routes[route](**kw)
    if hasattr(res, "body"):
        return json.loads(res.body)
    return json.loads(json.dumps(mod._json_safe(res)))


# --------------------------------------------------------------------------------------- exporter

API_DATA_ABS = re.compile(r"(?<![\w/.-])/api/data/")
SNAP_REF = re.compile(r"snapshot/api/data/([A-Za-z0-9_.~-]+)/([^\s\"'<>`)\]\\]+)")


class Exporter:
    def __init__(self, args):
        self.args = args
        self.source = Path(args.source).resolve()
        self.out = Path(args.out).resolve()
        self.snap = self.out / "snapshot"
        self.literals = local_literals(self.source)
        self.scrub = Scrubber(self.literals)
        self.mod, self.routes, self.rid = load_server(self.source)
        self.json_files: dict[str, object] = {}       # snapshot-relative path -> JSON payload
        self.text_files: dict[str, str] = {}          # snapshot-relative path -> text (md)
        self.refs: dict[str, set[str]] = {}           # data path (source-relative) -> why
        self.refused: list[dict] = []
        self.flagged: list[dict] = []
        self.excluded: list[dict] = []
        self.skipped: list[dict] = []
        self.media_log: list[dict] = []
        self.cache = Path(args.cache_dir or (Path(tempfile.gettempdir()) / "lt-public-export-cache"))

    # ---- strings
    def clean(self, obj):
        """Rewrite /api/data URLs to the relative snapshot and scrub every string, recursively."""
        if isinstance(obj, str):
            return self.scrub.text(API_DATA_ABS.sub("snapshot/api/data/", obj))
        if isinstance(obj, list):
            return [self.clean(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.clean(v) for k, v in obj.items()}
        return obj

    def collect_refs(self, obj, why: str, base_html: str | None = None):
        """Find every snapshot/api/data/<rid>/<path> mentioned anywhere in `obj` (after clean())."""
        def walk(x):
            if isinstance(x, str):
                for m in SNAP_REF.finditer(x):
                    rid, path = m.group(1), urllib.parse.unquote(m.group(2)).rstrip(".,;:!?")
                    if rid != self.rid:
                        self.refused.append({"path": f"{rid}/{path}", "why": why, "reason": "foreign root"})
                        continue
                    self.add_ref(path, why, text=x)
            elif isinstance(x, list):
                for y in x:
                    walk(y)
            elif isinstance(x, dict):
                for y in x.values():
                    walk(y)
        walk(obj)

    def add_ref(self, path: str, why: str, text: str = ""):
        path = path.split("#")[0].split("?")[0]
        target = (self.source / path)
        try:
            target = target.resolve()
            target.relative_to(self.source)
        except Exception:
            self.refused.append({"path": path, "why": why, "reason": "outside the data checkout"})
            return
        rel = target.relative_to(self.source).as_posix()
        if target.is_dir():
            # A base URL a bespoke page builds file names from. Pull in the media/data files under it that
            # the referring text actually names: by relative path, by a distinctive basename, or by a name
            # its script ASSEMBLES ("clip_" + scene + "_" + mat + ".mp4", `${base}/cells/nn_st${r}.png`).
            for f in sorted(target.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in EXPORTABLE:
                    continue
                frel = f.relative_to(target).as_posix()
                if frel in text or (len(f.name) >= 8 and f.name in text):
                    self.add_ref(f.relative_to(self.source).as_posix(), why + " (named under a base url)", text="")
                elif self.assembled_name(frel, text):
                    self.add_ref(f.relative_to(self.source).as_posix(), why + " (name assembled by its script)",
                                 text="")
            return
        self.refs.setdefault(rel, set()).add(why)

    # A piece of code standing in for one token of a file name: `${expr}` in a template literal, or
    # `" + expr + "` in string concatenation (either quote style).
    _EXPR = r"""(?:\$\{[^}`]{1,60}\}|"\s*\+\s*[^"+;\n]{1,60}?\s*\+\s*"|'\s*\+\s*[^'+;\n]{1,60}?\s*\+\s*')"""

    def assembled_name(self, rel: str, text: str) -> bool:
        """True if `text` contains code that builds `rel` with at least one token supplied by a variable.
        Each alphanumeric token of the name may appear literally or as an expression; the first token must
        be literal (an anchor), so this cannot match arbitrary concatenation."""
        if "+" not in text and "${" not in text:
            return False
        stem, ext = os.path.splitext(rel)
        toks = re.findall(r"[A-Za-z]+|\d+|[^A-Za-z\d]", stem)
        words = [i for i, t in enumerate(toks) if t[0].isalnum()]
        if len(words) < 2:
            return False
        first = words[0]
        parts = []
        for i, t in enumerate(toks):
            if not t[0].isalnum():
                parts.append(re.escape(t))
            elif i == first:
                parts.append(re.escape(t))
            else:
                parts.append(f"(?:{re.escape(t)}|{self._EXPR})")
        pat = "".join(parts) + re.escape(ext)
        for m in re.finditer(pat, text):
            if re.search(self._EXPR, m.group(0)):
                return True
        return False

    # ---- personal-content check on Sevan's notes
    def review_notes(self, ov: dict):
        manual = {}
        if self.args.review_file:
            manual = json.loads(Path(self.args.review_file).read_text("utf-8"))
        self.reviewed_clear = []
        for d in ov.get("directions", []):
            summ = d.get("summary") or ""
            if personal_hits(summ):
                self.flagged.append({"id": f"{d['id']}:summary", "ref": None, "task": d.get("name"),
                                     "field": "direction summary", "index": None,
                                     "source": f"coordination/directions/{d['id']}.json",
                                     "matched": personal_hits(summ), "manual_reason": None,
                                     "excerpt": summ[:220], "published": True,
                                     "note": "direction summaries are always published; review by hand"})
            for t in d.get("tasks", []):
                key = f"{d['id']}/{t['id']}"
                src = f"coordination/directions/{d['id']}.json"
                fields = [("note", None, t.get("note") or "")]
                fields += [("rework_history", i, h.get("note") or "") for i, h in enumerate(t.get("rework_history") or [])]
                fields += [("notes", i, n.get("text") or "") for i, n in enumerate(t.get("notes") or [])]
                drop_rw, drop_notes = set(), set()
                for field, i, text in fields:
                    if not text:
                        continue
                    fid = f"{key}:{field}" + (f"[{i}]" if i is not None else "")
                    words = personal_hits(text)
                    rev = manual.get(fid) or {}
                    decision = rev.get("decision")
                    if not words and decision != "withhold":
                        continue
                    excerpt = text[:220] + ("…" if len(text) > 220 else "")
                    if decision == "publish":
                        self.reviewed_clear.append({"id": fid, "ref": t.get("ref"), "task": t.get("title"),
                                                    "field": field, "source": src, "matched": words,
                                                    "reason": rev.get("reason"), "excerpt": excerpt})
                        continue
                    entry = {"id": fid, "ref": t.get("ref"), "task": t.get("title"), "field": field, "index": i,
                             "source": src, "matched": words, "manual_reason": rev.get("reason"),
                             "excerpt": excerpt, "published": bool(self.args.include_flagged)}
                    self.flagged.append(entry)
                    if self.args.include_flagged:
                        continue
                    if field == "note":
                        t["note"] = ""
                    elif field == "rework_history":
                        drop_rw.add(i)
                    else:
                        drop_notes.add(i)
                if drop_rw:
                    t["rework_history"] = [h for j, h in enumerate(t.get("rework_history") or []) if j not in drop_rw]
                    t["rework_note"] = (t["rework_history"][-1].get("note") if t["rework_history"] else None)
                if drop_notes:
                    t["notes"] = [n for j, n in enumerate(t.get("notes") or []) if j not in drop_notes]
        return ov

    def apply_note_flags_to_detail(self, direction: str, task: str, detail: dict):
        """Mirror the overview's note decisions onto the task page payload (same fields, same rules)."""
        if self.args.include_flagged:
            return detail
        key = f"{direction}/{task}"
        rw_drop = {f["index"] for f in self.flagged if f["id"].startswith(key + ":rework_history[")}
        nt_drop = {f["index"] for f in self.flagged if f["id"].startswith(key + ":notes[")}
        if rw_drop:
            detail["rework_history"] = [h for j, h in enumerate(detail.get("rework_history") or []) if j not in rw_drop]
        if nt_drop:
            detail["notes"] = [n for j, n in enumerate(detail.get("notes") or []) if j not in nt_drop]
        return detail

    # ---- endpoints
    def build_snapshot(self):
        r = self.routes
        ov = call_json(self.mod, r, "/api/overview")
        if ov.get("orphans"):
            for o in ov["orphans"]:
                self.excluded.append({"what": f"run {o['direction']}/{o['task']}",
                                      "why": "orphan: has a manifest but is not on the board (deleted or superseded)"})
        ov["orphans"] = []
        ov = self.review_notes(ov)
        on_board = set()
        for d in ov.get("directions", []):
            for t in d.get("tasks", []):
                t["live"] = None
                if t.get("detail"):
                    on_board.add((d["id"], t["id"]))
                    t["detail"] = f"snapshot/api/task/{d['id']}/{t['id']}.json"
        ov = self.clean(ov)
        self.json_files["api/overview.json"] = ov

        # Every task with a result.
        for (did, tid) in sorted(on_board):
            det = call_json(self.mod, r, "/api/task/{direction}/{task}", direction=did, task=tid)
            det["live"] = None
            det = self.apply_note_flags_to_detail(did, tid, det)
            det = self.clean(det)
            self.collect_refs(det, f"task {did}/{tid}")
            # A bespoke page may build media names from its run dir without spelling the full URL.
            html = det.get("custom_html") or ""
            if html:
                run_dir = f"runs/{did}/{tid}"
                if (self.source / run_dir).is_dir():
                    self.add_ref(run_dir, f"task {did}/{tid} bespoke page", text=html)
            self.json_files[f"api/task/{did}/{tid}.json"] = det

        # Run index (+ the manifests it points at), restricted to runs that are on the board.
        idx = call_json(self.mod, r, "/api/index")
        keep = []
        for run in idx.get("runs", []):
            m = re.match(rf"^/api/data/{re.escape(self.rid)}/runs/([^/]+)/([^/]+)/manifest\.json$", run.get("manifest", ""))
            if not m or (m.group(1), m.group(2)) not in on_board:
                continue
            rel = f"runs/{m.group(1)}/{m.group(2)}/manifest.json"
            man = call_json(self.mod, r, "/api/data/{rid}/{path:path}", rid=self.rid, path=rel)
            if str(man.get("schema_version")) == "2":
                man = self.apply_note_flags_to_detail(m.group(1), m.group(2), man)
            man = self.clean(man)
            self.collect_refs(man, f"manifest {rel}")
            self.json_files[f"api/data/{self.rid}/{rel}"] = man
            keep.append(run)
        idx["runs"] = keep
        self.json_files["api/index.json"] = self.clean(idx)

        # Textbook.
        toc = call_json(self.mod, r, "/api/training")
        for g in toc.get("groups", []):
            for s in g.get("sections", []):
                s["mtime"] = 0      # the "New" badge is per-device reading state; meaningless in public
        toc = self.clean(toc)
        self.collect_refs(toc, "training toc")
        self.json_files["api/training.json"] = toc

        # Directions (the narrative backlog doc it points at is research planning, not personal).
        dirs = self.clean(call_json(self.mod, r, "/api/directions"))
        self.collect_refs(dirs, "directions")
        self.json_files["api/directions.json"] = dirs

        # Reports: Sevan's hand-written report is his, not a deliverable, unless explicitly opted in.
        reps = call_json(self.mod, r, "/api/reports")
        if not self.args.include_research_report:
            for x in reps.get("reports", []):
                self.excluded.append({"what": f"report '{x.get('title')}' ({RESEARCH_REPORT})",
                                      "why": "hand-written by Sevan; opt in with --include-research-report"})
            reps["reports"] = []
        reps = self.clean(reps)
        self.collect_refs(reps, "reports")
        self.json_files["api/reports.json"] = reps

        # Epochs: the copied report/verdict/exam of each epoch are as private as the report itself.
        eps = call_json(self.mod, r, "/api/epochs")
        for e in eps.get("epochs", []):
            if not self.args.include_research_report:
                for k in ("report_url", "verdict_url", "exam_url"):
                    e[k] = None
                e["report_verdict"] = None
                e["report_score"] = None
            if e.get("demo_url"):
                self.excluded.append({"what": f"epoch {e.get('id')} frozen demo build", "why": "not exported"})
                e["demo_url"] = None
        eps = self.clean(eps)
        self.collect_refs(eps, "epochs")
        self.json_files["api/epochs.json"] = eps

        self.json_files["api/definitions.json"] = self.clean(call_json(self.mod, r, "/api/definitions"))
        self.json_files["api/tags.json"] = self.clean(call_json(self.mod, r, "/api/tags"))

        if self.args.include_decisions:
            dec = self.clean(call_json(self.mod, r, "/api/decisions"))
            for x in dec.get("decisions", []):
                x["auto_run_at"] = None
            self.collect_refs(dec, "decisions")
            self.json_files["api/decisions.json"] = dec

        # Always-excluded endpoints, recorded for the report.
        for ep, why in (("/api/notebook", "Sevan's thinking space (reports/notebook/)"),
                        ("/api/notifications", "ntfy feed; the topic is a secret"),
                        ("/api/health", "liveness probe of a server that does not exist here"),
                        ("/api/decisions", "approval inbox; opt in with --include-decisions"),
                        ("POST /api/*", "every write endpoint (status, notes, tags, effort, budget, files, "
                                        "uploads, create/delete/follow-up, decision resolve)")):
            if ep == "/api/decisions" and self.args.include_decisions:
                continue
            self.excluded.append({"what": ep, "why": why})
        missing = [p for p in self.routes if p.startswith("/api/") and p not in {
            "/api/index", "/api/data/{rid}/{path:path}", "/api/training", "/api/directions", "/api/reports",
            "/api/definitions", "/api/decisions", "/api/overview", "/api/task/{direction}/{task}",
            "/api/notebook", "/api/epochs", "/api/tags", "/api/health", "/api/notifications"}]
        if missing:
            die(f"the server has GET endpoints this exporter does not know how to classify: {missing}. "
                "Decide whether each is public, then add it here.")

    # ---- referenced files
    def gate(self, rel: str) -> str | None:
        """Why `rel` may not be exported, or None if it may."""
        for pat in DENY_ALWAYS:
            if pat.search(rel):
                return "never exported"
        if rel == RESEARCH_REPORT and not self.args.include_research_report:
            return "Sevan's hand-written report (opt-in)"
        if EPOCH_PRIVATE.search(rel) and not self.args.include_research_report:
            return "epoch copy of the report/verdict (opt-in)"
        if DECISIONS_DIR.search(rel) and not self.args.include_decisions:
            return "approval inbox (opt-in)"
        if not (rel.startswith(ALLOWED_ROOTS) or (self.args.include_research_report and rel == RESEARCH_REPORT)
                or (self.args.include_decisions and DECISIONS_DIR.search(rel))):
            return "outside the exportable roots"
        if Path(rel).suffix.lower() not in EXPORTABLE:
            return f"file type {Path(rel).suffix or '(none)'} is not exported"
        return None

    def ffmpeg(self) -> str | None:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        try:
            import imageio_ffmpeg  # noqa: PLC0415
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def process_video(self, src: Path, dst: Path) -> dict:
        """Re-encode to H.264 inside a 1280x720 box, no audio, NO metadata. Keeps the smaller of the
        re-encode and a metadata-stripped remux of the original."""
        ff = self.ffmpeg()
        if not ff:
            if not self.args.no_transcode:
                die("no ffmpeg found (PATH or imageio-ffmpeg); install one or pass --no-transcode")
            shutil.copyfile(src, dst)
            return {"action": "copied (no ffmpeg)"}
        key = hashlib.sha1(src.read_bytes() + f"v1|crf{self.args.crf}".encode()).hexdigest()
        cached = self.cache / f"{key}{src.suffix.lower()}"
        if cached.is_file():
            shutil.copyfile(cached, dst)
            return {"action": "re-encoded (cached)"}
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp_enc = self.cache / f"{key}.enc{src.suffix.lower()}"
        tmp_mux = self.cache / f"{key}.mux{src.suffix.lower()}"
        vf = ("scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease,"
              "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p")
        enc = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an",
               "-map_metadata", "-1", "-map_chapters", "-1", "-vf", vf, "-c:v", "libx264", "-preset", "slow",
               "-crf", str(self.args.crf), "-movflags", "+faststart", str(tmp_enc)]
        mux = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an",
               "-map_metadata", "-1", "-map_chapters", "-1", "-c", "copy", "-movflags", "+faststart", str(tmp_mux)]
        ok_enc = src.suffix.lower() in {".mp4", ".m4v", ".mov"} and subprocess.run(enc).returncode == 0 and tmp_enc.is_file()
        ok_mux = subprocess.run(mux).returncode == 0 and tmp_mux.is_file()
        cands = [p for p, ok in ((tmp_enc, ok_enc), (tmp_mux, ok_mux)) if ok]
        if not cands:
            die(f"ffmpeg could not process {src}")
        best = min(cands, key=lambda p: p.stat().st_size)
        shutil.copyfile(best, cached)
        shutil.copyfile(best, dst)
        for p in (tmp_enc, tmp_mux):
            p.unlink(missing_ok=True)
        return {"action": "re-encoded" if best == tmp_enc else "remuxed (metadata stripped)"}

    PNG_KEEP = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"sBIT",
                b"pHYs", b"bKGD", b"acTL", b"fcTL", b"fdAT"}

    def strip_png(self, src: Path, dst: Path) -> int:
        """Copy a PNG keeping only image/colour chunks: text (tEXt/iTXt/zTXt), tIME, eXIf and any private
        chunk are dropped, and the pixel data is left byte-identical (no re-compression)."""
        import struct  # noqa: PLC0415
        b = src.read_bytes()
        if b[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG")
        out, i, dropped = [b[:8]], 8, 0
        while i < len(b):
            n = struct.unpack(">I", b[i:i + 4])[0]
            typ = b[i + 4:i + 8]
            chunk = b[i:i + 12 + n]
            if typ in self.PNG_KEEP:
                out.append(chunk)
            else:
                dropped += 1
            i += 12 + n
            if typ == b"IEND":
                break
        dst.write_bytes(b"".join(out))
        return dropped

    def process_image(self, src: Path, dst: Path) -> dict:
        """Strip metadata (EXIF / PNG text chunks); downscale anything over --max-image-px. PNGs within the
        size limit are stripped at the chunk level so their compression is untouched."""
        if src.suffix.lower() == ".png":
            try:
                from PIL import Image  # noqa: PLC0415
                with Image.open(src) as im:
                    big = max(im.size)
            except Exception:
                big = 0
            if big <= self.args.max_image_px:
                dropped = self.strip_png(src, dst)
                return {"action": f"metadata stripped ({dropped} chunk{'s' if dropped != 1 else ''})"}
        try:
            from PIL import Image  # noqa: PLC0415
        except Exception:
            die("Pillow is required to strip image metadata (pip install pillow)")
        with Image.open(src) as im:
            fmt = im.format
            if getattr(im, "is_animated", False):
                # Animated GIF/WebP: keep frames, drop metadata.
                frames = []
                try:
                    while True:
                        frames.append(im.copy())
                        im.seek(im.tell() + 1)
                except EOFError:
                    pass
                frames[0].save(dst, format=fmt, save_all=True, append_images=frames[1:],
                               loop=im.info.get("loop", 0), duration=im.info.get("duration", 100))
                return {"action": "re-saved (animated, metadata stripped)"}
            im.load()
            out = im
            action = "re-saved (metadata stripped)"
            big = max(im.size)
            if big > self.args.max_image_px:
                k = self.args.max_image_px / big
                out = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
                action = f"downscaled {im.width}x{im.height} -> {out.width}x{out.height}"
            kw = {"optimize": True} if fmt in ("PNG", "JPEG") else {}
            if fmt == "JPEG":
                kw["quality"] = 88
            out.save(dst, format=fmt, **kw)
        return {"action": action}

    def write_text_file(self, rel: str, src: Path, dst: Path):
        t = src.read_text("utf-8", errors="replace")
        t = self.clean(t)
        dst.write_text(t, encoding="utf-8")

    def export_files(self):
        data_root = self.snap / "api" / "data" / self.rid
        # Markdown (training pages, the directions doc) can reference more media: resolve those first.
        pending = sorted(self.refs)
        seen = set()
        while pending:
            rel = pending.pop(0)
            if rel in seen:
                continue
            seen.add(rel)
            why = self.gate(rel)
            if why:
                self.refused.append({"path": rel, "why": sorted(self.refs.get(rel, [])), "reason": why})
                continue
            src = self.source / rel
            if not src.is_file():
                self.refused.append({"path": rel, "why": sorted(self.refs.get(rel, [])), "reason": "missing on disk"})
                continue
            if src.suffix.lower() in {".md", ".txt", ".svg", ".csv"}:
                t = self.clean(src.read_text("utf-8", errors="replace"))
                before = set(self.refs)
                self.collect_refs(t, f"doc {rel}")
                pending.extend(sorted(set(self.refs) - before))
                self.text_files[f"api/data/{self.rid}/{rel}"] = t

        # Process media/data into the snapshot.
        cap = int(self.args.max_file_mb * 1048576)
        items = []
        for rel in sorted(self.refs):
            if self.gate(rel) or not (self.source / rel).is_file() or f"api/data/{self.rid}/{rel}" in self.text_files:
                continue
            items.append(rel)
        for rel in items:
            src = self.source / rel
            dst = data_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            ext = src.suffix.lower()
            orig = src.stat().st_size
            if f"api/data/{self.rid}/{rel}" in self.json_files:
                continue    # manifests are written from the server's rewritten response
            if src.name == "manifest.json":
                self.refused.append({"path": rel, "why": sorted(self.refs[rel]),
                                     "reason": "a manifest not on the board is never exported"})
                continue
            if ext in MEDIA_VIDEO:
                info = self.process_video(src, dst)
            elif ext in MEDIA_IMAGE:
                info = self.process_image(src, dst)
            elif ext == ".json":
                try:
                    j = json.loads(src.read_text("utf-8"))
                except Exception:
                    j = None
                if j is None:
                    self.refused.append({"path": rel, "why": sorted(self.refs[rel]), "reason": "unparseable JSON"})
                    continue
                dst.write_text(json.dumps(self.clean(self.mod._json_safe(j)), separators=(",", ":")), "utf-8")
                info = {"action": "scrubbed JSON"}
            else:
                shutil.copyfile(src, dst)
                info = {"action": "copied"}
            size = dst.stat().st_size
            entry = {"path": rel, "original": orig, "exported": size, **info, "why": sorted(self.refs[rel])}
            if size > cap:
                dst.unlink()
                entry["skipped"] = f"over the {self.args.max_file_mb:g} MB per-file cap after processing"
                self.skipped.append(entry)
            else:
                self.media_log.append(entry)

        # Total budget: drop the largest videos first until under it.
        budget = int(self.args.budget_mb * 1048576)
        total = sum(e["exported"] for e in self.media_log)
        if total > budget:
            for e in sorted([e for e in self.media_log if Path(e["path"]).suffix.lower() in MEDIA_VIDEO],
                            key=lambda e: -e["exported"]):
                if total <= budget:
                    break
                (data_root / e["path"]).unlink(missing_ok=True)
                e["skipped"] = f"dropped to fit the {self.args.budget_mb:g} MB total budget"
                total -= e["exported"]
                self.skipped.append(e)
            self.media_log = [e for e in self.media_log if "skipped" not in e]

    def mark_omitted(self):
        """Degrade gracefully where media was skipped: results say so instead of rendering a broken
        player, and markdown embeds become a plain note."""
        gone = {f"snapshot/api/data/{self.rid}/{e['path']}" for e in self.skipped}
        gone |= {f"snapshot/api/data/{self.rid}/{r['path']}" for r in self.refused
                 if Path(r["path"]).suffix.lower() in MEDIA_VIDEO | MEDIA_IMAGE}
        if not gone:
            return

        def fix_results(payload):
            for r in payload.get("results") or []:
                if isinstance(r, dict) and r.get("src") in gone:
                    r["omitted"] = True
                    r.pop("src", None)

        for k, v in self.json_files.items():
            if k.startswith("api/task/") or k.endswith("/manifest.json"):
                fix_results(v)
        md_img = re.compile(r"!\[([^\]]*)\]\((snapshot/api/data/[^)\s]+)\)")
        for k, t in list(self.text_files.items()):
            self.text_files[k] = md_img.sub(
                lambda m: (f"*[{m.group(1) or 'media'} — not included in this public snapshot]*"
                           if m.group(2) in gone else m.group(0)), t)

    # ---- write + build + scan
    def write_snapshot(self):
        for name in OWNED:
            p = self.out / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        self.snap.mkdir(parents=True, exist_ok=True)
        self.export_files()
        self.mark_omitted()
        for rel, payload in self.json_files.items():
            p = self.snap / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "utf-8")
        for rel, t in self.text_files.items():
            p = self.snap / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t, "utf-8")
        meta = {
            "snapshot_date": self.args.date,
            "views": {"reports": bool(self.json_files["api/reports.json"].get("reports")
                                      or self.json_files["api/epochs.json"].get("epochs")),
                      "decisions": bool(self.args.include_decisions)},
            "read_only": True,
        }
        (self.snap / "meta.json").write_text(json.dumps(meta, indent=1), "utf-8")

    def build_bundles(self):
        npm = shutil.which("npm")
        if not npm:
            die("npm not found on PATH (Node 24: PATH=/opt/homebrew/opt/node@24/bin:$PATH)")
        if not (DASHBOARD / "node_modules").is_dir():
            subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=DASHBOARD, check=True)
        subprocess.run([npm, "run", "build:public"], cwd=DASHBOARD, check=True)
        subprocess.run([npm, "run", "build:demo"], cwd=DASHBOARD, check=True)
        dist = DASHBOARD / "dist-public"
        for p in dist.iterdir():
            if p.name == "snapshot":
                continue
            dst = self.out / p.name
            if p.is_dir():
                shutil.copytree(p, dst)
            else:
                shutil.copyfile(p, dst)
        (self.out / "demo").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DASHBOARD / "dist-demo" / "index.html", self.out / "demo" / "index.html")

    def run(self):
        print(f"source: data checkout '{self.source.name}' (root id '{self.rid}')")
        self.build_snapshot()
        self.write_snapshot()
        if not self.args.skip_build:
            self.build_bundles()
        scan = scan_tree(self.out, self.literals, exclude={REPORT_NAME})
        sizes = {"total": 0, "snapshot": 0, "snapshot_media": 0, "bundle": 0, "demo": 0}
        for p in self.out.rglob("*"):
            if not p.is_file() or p.name == REPORT_NAME:
                continue
            n = p.stat().st_size
            rel = p.relative_to(self.out).as_posix()
            sizes["total"] += n
            if rel.startswith("snapshot/"):
                sizes["snapshot"] += n
                if Path(rel).suffix.lower() in MEDIA_VIDEO | MEDIA_IMAGE:
                    sizes["snapshot_media"] += n
            elif rel.startswith("demo/"):
                sizes["demo"] += n
            else:
                sizes["bundle"] += n
        report = {
            "snapshot_date": self.args.date,
            "root_id": self.rid,
            "options": {k: getattr(self.args, k) for k in ("include_research_report", "include_decisions",
                                                           "include_flagged", "max_file_mb", "budget_mb", "crf",
                                                           "max_image_px", "skip_build")},
            "endpoints": sorted(k for k in self.json_files if not k.startswith("api/data/")),
            "tasks": sorted(k[len("api/task/"):-5] for k in self.json_files if k.startswith("api/task/")),
            "docs": sorted(self.text_files),
            "files": self.media_log,
            "skipped": self.skipped,
            "refused": self.refused,
            "excluded": self.excluded,
            "flagged_notes": self.flagged,
            "reviewed_published": getattr(self, "reviewed_clear", []),
            "scrub_replacements": self.scrub.hits,
            "scan": {"files_scanned": scan["files_scanned"], "hit_count": len(scan["hits"]),
                     "categories": sorted({h["category"] for h in scan["hits"]})},
            "sizes": sizes,
        }
        if self.args.report_json:
            Path(self.args.report_json).write_text(json.dumps(report, indent=1, default=list), "utf-8")
        print(f"tasks: {len(report['tasks'])}  docs: {len(self.text_files)}  files: {len(self.media_log)}  "
              f"skipped: {len(self.skipped)}  refused refs: {len(self.refused)}  flagged notes: {len(self.flagged)}")
        print(f"scrubbed: {self.scrub.hits or 'nothing'}")
        print(f"size: total {human(sizes['total'])} (snapshot {human(sizes['snapshot'])}, of which media "
              f"{human(sizes['snapshot_media'])}; dashboard {human(sizes['bundle'])}; demo {human(sizes['demo'])})")
        for e in self.skipped:
            print(f"  SKIPPED {e['path']}  {human(e['original'])} -> {human(e['exported'])}: {e['skipped']}")
        for f in self.flagged:
            print(f"  FLAGGED {f['id']} {'(published)' if f['published'] else '(withheld)'}: {f['matched'] or f['manual_reason']}")
        print(f"scan: {scan['files_scanned']} files, {len(scan['hits'])} hits")
        if scan["hits"]:
            for h in scan["hits"][:50]:
                print(f"  LEAK [{h['category']}] {h['file']}: {h['excerpt']}", file=sys.stderr)
            die(f"{len(scan['hits'])} forbidden pattern(s) remain in {self.out.name}/ -- nothing should be "
                "published from this tree until they are fixed", code=2)
        if sizes["total"] > (self.args.budget_mb + 10) * 1048576:
            die(f"bundle is {human(sizes['total'])}, over the budget", code=3)
        print("OK: snapshot exported, scan clean.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=None, help="data checkout (default: the main git worktree)")
    ap.add_argument("--out", required=True, help="output folder for the lab (index.html, snapshot/, demo/)")
    ap.add_argument("--date", default=datetime.date.today().isoformat(), help="snapshot date shown (YYYY-MM-DD)")
    ap.add_argument("--include-research-report", action="store_true",
                    help="OPT-IN: publish reports/research_report.md (hand-written by Sevan) and epoch report copies")
    ap.add_argument("--include-decisions", action="store_true",
                    help="OPT-IN: publish coordination/decisions/ as a read-only Decisions view")
    ap.add_argument("--include-flagged", action="store_true",
                    help="OPT-IN: publish task notes the personal-content check flagged")
    ap.add_argument("--review-file", default=None,
                    help='JSON {"<dir>/<task>:<field>[i]": {"decision": "withhold"|"publish", "reason": "..."}}: '
                         'hand review of notes. "withhold" flags a note the word check missed; "publish" clears '
                         'an automatic flag that was reviewed as plain research content.')
    ap.add_argument("--max-file-mb", type=float, default=8.0, help="per-file cap after processing")
    ap.add_argument("--budget-mb", type=float, default=75.0, help="total budget for exported data files")
    ap.add_argument("--crf", type=int, default=26, help="x264 quality for re-encoded video (lower = better)")
    ap.add_argument("--max-image-px", type=int, default=3200, help="downscale images whose long side exceeds this")
    ap.add_argument("--no-transcode", action="store_true", help="copy videos as-is if no ffmpeg is available")
    ap.add_argument("--cache-dir", default=None, help="video re-encode cache (default: system temp)")
    ap.add_argument("--skip-build", action="store_true", help="write the snapshot only; do not build/copy bundles")
    ap.add_argument("--report-json", default=None, help="write a machine-readable summary here (keep it out of --out)")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        die("--date must be YYYY-MM-DD")
    args.source = str(Path(args.source).resolve() if args.source else main_worktree(CHECKOUT))
    if not (Path(args.source) / "coordination" / "directions").is_dir():
        die(f"{args.source} does not look like a learning-taichi data checkout")
    out = Path(args.out).resolve()
    src = Path(args.source)
    if out == src or src in out.parents:
        die("--out must be outside the data checkout (the export never writes into it)")
    out.mkdir(parents=True, exist_ok=True)
    if args.report_json and Path(args.report_json).resolve().is_relative_to(out):
        die("--report-json must live outside --out (it quotes flagged notes)")
    Exporter(args).run()


if __name__ == "__main__":
    main()
