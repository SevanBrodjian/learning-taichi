import { useEffect, useState } from "react";
import { fetchDirections } from "../api.js";
import DocView from "./DocView.jsx";

// Status columns, in pipeline order. "active" = a worker/worktree is on it right now.
const COLUMNS = [
  { id: "active", label: "Active" },
  { id: "queued", label: "Queued" },
  { id: "proposed", label: "Proposed" },
  { id: "done", label: "Done" },
  { id: "parked", label: "Parked" },
];

export default function DirectionsView() {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetchDirections()
      .then(setData)
      .catch(() => setData({ directions: [], md_url: null }));
  }, []);

  if (!data) return <div className="muted pad">Loading…</div>;
  const dirs = data.directions || [];
  // No structured directions yet: fall back to the narrative markdown.
  if (dirs.length === 0) return <DocView url={data.md_url} empty="No research directions yet." />;

  // Always show Active + Queued (even if empty, so the pipeline is legible); other columns only if used.
  const cols = COLUMNS.filter(
    (c) => c.id === "active" || c.id === "queued" || dirs.some((d) => (d.status || "proposed") === c.id)
  );

  return (
    <div className="directions">
      <div className="board">
        {cols.map((c) => {
          const items = dirs.filter((d) => (d.status || "proposed") === c.id);
          return (
            <div className={`board-col col-${c.id}`} key={c.id}>
              <div className="board-col-head">
                {c.label} <span className="board-count">{items.length}</span>
              </div>
              {items.map((d) => (
                <div className="dir-card" key={d.id}>
                  <div className="dir-title">{d.title}</div>
                  {d.branch && <div className="dir-branch">{d.branch.split("/").pop()}</div>}
                  {d.note && <div className="dir-note">{d.note}</div>}
                </div>
              ))}
              {items.length === 0 && <div className="muted dir-empty">none</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
