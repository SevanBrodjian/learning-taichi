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
// Tailnet hosts must be allow-listed or Vite's DNS-rebinding guard 403s them. The leading dot is a
// subdomain wildcard, so any machine on this tailnet can serve the dashboard.
//
// WHY THIS MATTERS beyond convenience: `navigator.gpu` (and every other powerful API) is only exposed in a
// SECURE CONTEXT. localhost qualifies; http://<lan-ip>:5174 does not. Reaching the dashboard over plain
// HTTP from the iPad/phone/Mac therefore hides WebGPU entirely, which reads as "this device has no WebGPU"
// when it actually means "this origin is not trusted". `tailscale serve --bg --https=443
// http://localhost:5174` fronts this server with a real Let's Encrypt cert on
// https://<machine>.<tailnet>.ts.net, which is a secure context on every device with no cert to install.
const ALLOWED_HOSTS = [".ts.net"];

export default defineConfig({
  plugins: [react()],
  server: { port: 5174, strictPort: true, host: true, allowedHosts: ALLOWED_HOSTS,
            proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
  preview: { port: 5174, strictPort: true, host: true, allowedHosts: ALLOWED_HOSTS,
             proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
});
