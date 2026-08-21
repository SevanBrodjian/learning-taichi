import { API_BASE } from "./config.js";

// The data server returns fully-resolved /api/data/... URLs inside the index and every manifest,
// so callers just pass those paths straight back here. API_BASE is "" in dev (same-origin proxy).
async function get(path, kind) {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return kind === "text" ? res.text() : res.json();
}

export const fetchIndex = () => get("/api/index", "json");
export const fetchJSON = (path) => get(path, "json");
export const fetchText = (path) => get(path, "text");

// Media (video/frames) is referenced by an already-resolved /api/data/... URL in the manifest.
export const mediaUrl = (path) => `${API_BASE}${path}`;

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
export const setTaskStatus = (direction, task, status, note) =>
  fetch(`${API_BASE}/api/task-status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, task, status, note }),
  }).then((r) => r.json());

// Set a task's intensity tier (quick | standard | deep). The orchestrator reads it at spawn time.
export const setTaskEffort = (direction, task, effort) =>
  fetch(`${API_BASE}/api/task-effort`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, task, effort }),
  }).then((r) => r.json());

// Set a task's adaptive time budget in minutes (a soft expectation the orchestrator watches against).
export const setTaskBudget = (direction, task, minutes) =>
  fetch(`${API_BASE}/api/task-budget`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, task, minutes }),
  }).then((r) => r.json());

const post = (path, payload) =>
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
  if (!p || !navigator.sendBeacon) return false;
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
