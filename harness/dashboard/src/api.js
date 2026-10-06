import { API_BASE } from "./config.js";
import { PUBLIC, SNAPSHOT_ROOT } from "./publicMode.js";

// Public snapshot: the GET endpoints that were exported as static files. Anything else (notebook,
// notifications, decisions unless opted in, health) is simply not there, and asking for it fails locally
// instead of reaching for a server.
const SNAPSHOT_ENDPOINTS = new Set([
  "/api/index", "/api/overview", "/api/directions", "/api/reports", "/api/training",
  "/api/definitions", "/api/epochs", "/api/tags", "/api/decisions",
]);

// Map a live-server path to its place in the static snapshot. Returns null for anything not exported.
// The exporter already rewrites URLs inside the data to `snapshot/...`, so those pass straight through.
export function snapshotUrl(path) {
  if (!path) return null;
  if (path.startsWith(SNAPSHOT_ROOT)) return path;
  if (path.startsWith("/api/data/")) return SNAPSHOT_ROOT + path.slice(1);
  const t = /^\/api\/task\/([^/]+)\/([^/?#]+)$/.exec(path);
  if (t) return `${SNAPSHOT_ROOT}api/task/${t[1]}/${t[2]}.json`;
  if (SNAPSHOT_ENDPOINTS.has(path)) return `${SNAPSHOT_ROOT}${path.slice(1)}.json`;
  return null;
}

// The data server returns fully-resolved /api/data/... URLs inside the index and every manifest,
// so callers just pass those paths straight back here. API_BASE is "" in dev (same-origin proxy).
async function get(path, kind) {
  let url = `${API_BASE}${path}`;
  if (PUBLIC) {
    url = snapshotUrl(path);
    if (!url) throw new Error(`${path} is not part of this snapshot`);
  }
  const res = await fetch(url, PUBLIC ? undefined : { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return kind === "text" ? res.text() : res.json();
}

// Every write in public mode resolves to a refusal without touching the network. The UI that would call
// these is not rendered in public mode either; this is the second lock, not the first.
const READ_ONLY = () => Promise.resolve({ ok: false, error: "read-only snapshot" });

export const fetchIndex = () => get("/api/index", "json");
export const fetchJSON = (path) => get(path, "json");
export const fetchText = (path) => get(path, "text");

// Media (video/frames) is referenced by an already-resolved /api/data/... URL in the manifest.
export const mediaUrl = (path) => (PUBLIC ? snapshotUrl(path) || path : `${API_BASE}${path}`);

// Shared, repo-level doc sets (served from the main checkout).
export const fetchTraining = () => get("/api/training", "json");
export const fetchDirections = () => get("/api/directions", "json");
export const fetchReports = () => get("/api/reports", "json");
export const fetchDecisions = () => get("/api/decisions", "json");
// Approve/reject a decision (esp. a task contract) — appends a resolution the orchestrator reads.
export const resolveDecision = (id, resolution, note) =>
  post("/api/decision-resolve", { id, resolution, note });

// Direction -> Task model: the Overview board and a single task's detail.
export const fetchOverview = () => get("/api/overview", "json");
export const fetchTask = (detail) => get(detail, "json"); // detail is /api/task/<dir>/<task>

// Write-back: change a task's board status (drag / Mark Done / send back).
export const setTaskStatus = (direction, task, status, note) => PUBLIC ? READ_ONLY() :
  fetch(`${API_BASE}/api/task-status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, task, status, note }),
  }).then((r) => r.json());

// Set a task's intensity tier (quick | standard | deep). The orchestrator reads it at spawn time.
export const setTaskEffort = (direction, task, effort) => PUBLIC ? READ_ONLY() :
  fetch(`${API_BASE}/api/task-effort`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, task, effort }),
  }).then((r) => r.json());

// Set a task's adaptive time budget in minutes (a soft expectation the orchestrator watches against).
export const setTaskBudget = (direction, task, minutes) => PUBLIC ? READ_ONLY() :
  fetch(`${API_BASE}/api/task-budget`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, task, minutes }),
  }).then((r) => r.json());

const post = (path, payload) => PUBLIC ? READ_ONLY() :
  fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then((r) => r.json());

// Edit any displayed markdown doc back to disk. A doc url is /api/data/<rid>/<path>.
export function parseDataUrl(url) {
  const m = (url || "").match(/^\/api\/data\/([^/]+)\/(.+)$/);
  return m ? { rid: m[1], path: m[2] } : null;
}
// `commit: false` is for autosave — it lands on disk (which is what protects the writing) without
// making a git commit per pause in typing. The deliberate save commits.
export const saveFile = (url, content, commit = true) => {
  const p = parseDataUrl(url);
  if (!p) return Promise.resolve({ ok: false, error: "not an editable doc url" });
  return post("/api/file", { rid: p.rid, path: p.path, content, commit });
};

// Last-ditch save when the tab is being closed/hidden: fetch() gets cancelled on unload, sendBeacon
// does not. Fire-and-forget by design — there is no response to read at that point.
export const beaconSave = (url, content) => {
  const p = parseDataUrl(url);
  if (PUBLIC || !p || !navigator.sendBeacon) return false;
  const body = new Blob([JSON.stringify({ rid: p.rid, path: p.path, content, commit: false })],
                        { type: "application/json" });
  try { return navigator.sendBeacon(`${API_BASE}/api/file`, body); } catch { return false; }
};

// The notebook: Sevan's hand-written thinking space. The server tells us where it lives and what its
// relative image refs resolve against.
export const fetchNotebook = () => get("/api/notebook", "json");

// Binary-safe image upload (a pasted screenshot, a photo of paper). /api/file is markdown-only and
// would corrupt a PNG, so bytes go base64 over their own endpoint.
export const uploadImage = (rid, dir, filename, dataB64) =>
  post("/api/upload", { rid, dir, filename, data_b64: dataB64 });

// Epochs that have actually been cut (harness/tools/cut_epoch.py). Read-only.
export const fetchEpochs = () => get("/api/epochs", "json");

// Overview authoring: add/edit tasks and create directions, all committed server-side.
// A task is created with TAGS, not a direction. The server picks the storage direction itself — that
// file is an implementation detail behind the graph, not something the user should have to think about.
export const createTask = (title, note, status, tags) =>
  post("/api/task-create", { title, note, status, tags });
export const editTask = (direction, task, title, note) =>
  post("/api/task-edit", { direction, task, title, note });
export const createDirection = (name, summary) =>
  post("/api/direction-create", { name, summary });

// Remove a task/proposal from its direction file entirely (dashboard Delete).
export const deleteTask = (direction, task) =>
  post("/api/task-delete", { direction, task });

// Spin a completed task into a linked, proposed follow-up (both sides record the link). `parents` is an
// array of parent task ids in the same direction — a proposal can follow up on several tasks at once.
export const proposeFollowUp = (direction, parents, title, note, tags) =>
  post("/api/task-follow-up", { direction, parents, title, note, tags });

// ntfy notification feed (the server holds the secret topic; it never reaches the browser).
export const fetchNotifications = () => get("/api/notifications", "json");

// Canonical metric registry (spec/definitions.json). Rendered as hover definitions so a reader never has
// to guess what "roundness" means, and so tasks stop reinventing metrics.
export const fetchDefinitions = () => get("/api/definitions", "json");

// Passive user notes on a task. A note never changes status — it is a margin comment kept with the task.
export const addTaskNote = (direction, task, text) =>
  post("/api/task-note", { direction, task, text });
export const deleteTaskNote = (direction, task, ts) =>
  post("/api/task-note-delete", { direction, task, ts });

// Tag registry. Tags used to be a hard-coded array duplicated across three components, so a new one
// needed a code edit; they come from the server now (registry file UNION tags in use).
export const fetchTags = () => get("/api/tags", "json");
export const createTag = (name, color) =>
  post("/api/tag-create", { name, color });

// Set a task's tags. Tags are how the board is navigated (the Tasks list and the Map both filter by
// them), so this is available at any status -- a finished task is often exactly when you know what it
// should have been filed under.
export const setTaskTags = (direction, task, tags) =>
  post("/api/task-tags", { direction, task, tags });
