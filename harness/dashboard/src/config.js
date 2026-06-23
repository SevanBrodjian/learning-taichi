// Base for the data API. Empty string = same-origin: the Vite dev server (and the production
// host later) proxies `/api/*` to the harness data server (harness/server), which serves a live
// view across all worktrees. For site integration, point the proxy at the Django API instead —
// the JSON shapes are identical, so nothing in the dashboard changes.
export const API_BASE = "";
