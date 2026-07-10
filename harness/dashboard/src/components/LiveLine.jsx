import { isLiveRunning, isLiveBlocked } from "../effort.js";

// A worker's live status line: a pulsing dot plus the current step ("training the net"). Shows only
// while the status file is fresh; a stale file (worker gone) renders nothing so it never lies.
export default function LiveLine({ live, className = "" }) {
  if (!live) return null;
  const running = isLiveRunning(live);
  const blocked = isLiveBlocked(live);
  if (!running && !blocked) return null;
  return (
    <div className={`live-line ${blocked ? "blocked" : "running"} ${className}`}>
      <span className="live-dot" />
      <span className="live-step">{live.step || (blocked ? "blocked" : "running")}</span>
    </div>
  );
}
