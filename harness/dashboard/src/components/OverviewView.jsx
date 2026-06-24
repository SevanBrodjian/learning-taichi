import { useState } from "react";

// The live pipeline. Done is intentionally NOT a column (it would crowd the board); it lives in a
// collapsed section below. Cards are tasks, filterable by direction. A proposed task (no artifact yet)
// opens a detail modal instead of navigating, so you can judge its worth before queueing it.
const COLUMNS = [
  { id: "proposed", label: "Proposed" },
  { id: "queued", label: "Queued" },
  { id: "active", label: "Active" },
];

export default function OverviewView({ overview, onOpenTask }) {
  const [filter, setFilter] = useState("all");
  const [showDone, setShowDone] = useState(false);
  const [modal, setModal] = useState(null);

  if (!overview) return <div className="muted pad">Loading…</div>;
  const dirs = overview.directions || [];

  const tasks = dirs
    .filter((d) => filter === "all" || d.id === filter)
    .flatMap((d) => d.tasks.map((t) => ({ ...t, direction: d.id, directionName: d.name })));

  const done = tasks.filter((t) => t.status === "done");
  const columns = COLUMNS.map((c) => ({ ...c, items: tasks.filter((t) => t.status === c.id) }));

  const openCard = (t) => (t.has_artifact ? onOpenTask(t) : setModal(t));

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
              <button key={t.direction + "/" + t.id} className={`task-card ${t.has_artifact ? "has" : ""}`} onClick={() => openCard(t)}>
                <div className="task-card-title">{t.title}</div>
                <div className="task-card-dir">{t.directionName}</div>
                <div className="task-card-open">{t.has_artifact ? "view result →" : "details →"}</div>
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

      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setModal(null)} aria-label="close">×</button>
            <div className="modal-dir">{modal.directionName}</div>
            <h3>{modal.title}</h3>
            <p className="modal-note">{modal.note || "No description yet."}</p>
            <div className="modal-status">Status: {modal.status}</div>
          </div>
        </div>
      )}
    </div>
  );
}
