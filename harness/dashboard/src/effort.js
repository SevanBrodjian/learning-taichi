// Per-task intensity tiers. These map (in the /execute skill and coordination/tasks/_TEMPLATE.md) to
// the model + reasoning effort + persistence the orchestrator gives the spawned worker. The dashboard
// only sets the label; the orchestrator interprets it at spawn time.
export const EFFORTS = [
  { id: "quick", label: "Quick", hint: "Cheap / short — light learning tasks (Sonnet, low effort)" },
  { id: "standard", label: "Standard", hint: "Default depth (Opus, normal effort)" },
  { id: "deep", label: "Deep", hint: "Hard, long-running — persists (Opus, high effort)" },
];
export const EFFORT_IDS = EFFORTS.map((e) => e.id);
export const effortMeta = (id) => EFFORTS.find((e) => e.id === id) || EFFORTS[1];

// A worker's live status (runs/<dir>/<task>/status.json) is considered "current" only if it was
// updated recently. A stale file means the worker is very likely gone, so we stop claiming "running".
const FRESH_SECONDS = 20 * 60;
export const isLiveRunning = (live) =>
  !!live && live.state === "running" && typeof live.age === "number" && live.age < FRESH_SECONDS;
export const isLiveBlocked = (live) =>
  !!live && live.state === "blocked" && typeof live.age === "number" && live.age < FRESH_SECONDS;
