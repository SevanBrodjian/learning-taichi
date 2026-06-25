import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standalone dev server for the dashboard. `host: true` exposes it on the LAN so it can later be
// opened from an iPad. Port 5174 stays clear of a CRA site on 3000.
// `/api/*` is proxied to the harness data server (harness/server, default :8732). This keeps
// the dashboard same-origin (no CORS) and means an iPad hitting http://<host>:5174 reaches the
// data server on the same host transparently. Override the target with DASHBOARD_API.
const API_TARGET = process.env.DASHBOARD_API || "http://localhost:8732";

// strictPort: never silently fall back to 5175+. The user's iPad PWA is pinned to http://<host>:5174,
// so the dashboard must run on 5174 or fail loudly (then kill the stale instance and restart). Never
// run two instances.
export default defineConfig({
  plugins: [react()],
  server: { port: 5174, strictPort: true, host: true, proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
  preview: { port: 5174, strictPort: true, host: true, proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
});
