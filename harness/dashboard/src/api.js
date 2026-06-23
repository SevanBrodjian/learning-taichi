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
