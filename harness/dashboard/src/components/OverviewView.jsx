import { useState } from "react";

// The live pipeline. Done is intentionally NOT a column (it would crowd the board); it lives in a
// collapsed section below. Cards are tasks, filterable by direction.
const COLUMNS = [
  { id: "proposed", label: "Proposed" },
  { id: "queued", label: "Queued" },
  { id: "active", label: "Active" },
];

export default function OverviewView({ overview, onOpenTask }) {
  const [filter, setFilter] = useState("all");
  const [showDone, setShowDone] = useState(false);

  if (!overview) return <div className="muted pad">Loading…</div>;
  const dirs = overview.directions || [];

  const tasks = dirs
    .filter((d) => filter === "all" || d.id === filter)
    .flatMap((d) => d.tasks.map((t) => ({ ...t, direction: d.id, directionName: d.name })));

  const done = tasks.filter((t) => t.status === "done");
  const columns = COLUMNS.map((c) => ({ ...c, items: tasks.filter((t) => t.status === c.id) }));

  return (
    <div className="overview">
      <div className="dir-filter">
        <button className={`chip ${filter === "all" ? "active" : ""}`} onClick={() => setFilter("all")}>All</button>
        {dirs.map((d) => (
          <button key={d.id} className={`chip ${filter === d.id ? "active" : ""}`} title={d.summary} onClick={() => setFilter(d.id)}>
            {d.name}
          </button>
        ))}
      </div>

      <div className="board3">
        {columns.map((c) => (
          <div className={`board-col col-${c.id}`} key={c.id}>
            <div className="board-col-head">{c.label} <span className="board-count">{c.items.length}</span></div>
            {c.items.map((t) => (
              <button
                key={t.direction + "/" + t.id}
                className={`task-card ${t.has_artifact ? "has" : ""}`}
                disabled={!t.has_artifact}
                onClick={() => t.has_artifact && onOpenTask(t)}
              >
                <div className="task-card-title">{t.title}</div>
                <div className="task-card-dir">{t.directionName}</div>
                {t.has_artifact && <div className="task-card-open">view result →</div>}
              </button>
            ))}
            {c.items.length === 0 && <div className="muted dir-empty">none</div>}
          </div>
        ))}
      </div>

      <div className="done-section">
        <button className="done-toggle" onClick={() => setShowDone((s) => !s)}>
          {showDone ? "▾" : "▸"} Done ({done.length})
        </button>
        {showDone && (
          <div className="done-list">
            {done.map((t) => (
              <button key={t.direction + "/" + t.id} className="done-item" onClick={() => onOpenTask(t)}>
                <span className="task-card-title">{t.title}</span>
                <span className="task-card-dir">{t.directionName}</span>
              </button>
            ))}
            {done.length === 0 && <div className="muted dir-empty">nothing done yet</div>}
          </div>
        )}
      </div>
    </div>
  );
}
