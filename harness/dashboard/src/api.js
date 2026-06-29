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
export const saveFile = (url, content) => {
  const p = parseDataUrl(url);
  if (!p) return Promise.resolve({ ok: false, error: "not an editable doc url" });
  return post("/api/file", { rid: p.rid, path: p.path, content });
};

// Overview authoring: add/edit tasks and create directions, all committed server-side.
export const createTask = (direction, title, note, status) =>
  post("/api/task-create", { direction, title, note, status });
export const editTask = (direction, task, title, note) =>
  post("/api/task-edit", { direction, task, title, note });
export const createDirection = (name, summary) =>
  post("/api/direction-create", { name, summary });

// ntfy notification feed (the server holds the secret topic; it never reaches the browser).
export const fetchNotifications = () => get("/api/notifications", "json");
