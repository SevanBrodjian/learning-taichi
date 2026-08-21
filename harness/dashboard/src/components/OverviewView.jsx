import TagPicker from "./TagPicker.jsx";
import { useEffect, useState } from "react";
import { setTaskStatus, setTaskEffort, setTaskBudget, createTask, editTask, deleteTask } from "../api.js";
import { EFFORTS, effortMeta } from "../effort.js";
import LiveLine from "./LiveLine.jsx";

// Quick / Standard / Deep picker. Used in the task modal; writes back immediately.
function EffortPicker({ value, onPick, disabled }) {
  return (
    <div className="effort-picker">
      {EFFORTS.map((e) => (
        <button
          key={e.id}
          type="button"
          title={e.hint}
          disabled={disabled}
          className={`effort-opt ${value === e.id ? "active" : ""}`}
          onClick={() => onPick(e.id)}
        >
          {e.label}
        </button>
      ))}
    </div>
  );
}

// The live pipeline. Done is a collapsed section below (not a crowding column). Cards are tasks,
// filterable by direction. Proposed and queued cards can be dragged between those two columns
// (the user's call). A proposed/queued card with no artifact opens a detail modal; a card with a
// result navigates to its task. Tasks and directions can also be authored here (+ Task / + Direction).
const COLUMNS = [
  { id: "proposed", label: "Proposed" },
  { id: "queued", label: "Queued" },
  { id: "active", label: "Active" },
];
const DRAGGABLE = new Set(["proposed", "queued"]);

// The task-detail / edit modal. Holds its OWN draft state so typing never re-renders the board behind
// it (that, plus pausing the board poll while a field is focused, is what fixes the input lag).
function TaskModal({ task, onClose, onChanged }) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({ title: task.title, note: task.note || "" });
  const [effort, setEffort] = useState(task.effort || "standard"); // local so the picker updates live
  const [budget, setBudget] = useState(task.budget_minutes || 40); // adaptive time expectation (minutes)
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [busy, setBusy] = useState(false);

  const move = (status) => { setTaskStatus(task.direction, task.id, status).then(onChanged); onClose(); };
  const saveEdit = async () => {
    setBusy(true);
    try {
      await editTask(task.direction, task.id, form.title.trim(), form.note);
      setEditing(false);
      onChanged();
    } finally { setBusy(false); }
  };
  const doDelete = async () => {
    setBusy(true);
    try {
      const r = await deleteTask(task.direction, task.id);
      if (r && r.ok) { onChanged(); onClose(); }
    } finally { setBusy(false); }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="close">×</button>
        <div className="modal-dir">{task.directionName}</div>
        {editing ? (
          <div className="author-form">
            <label>Title</label>
            <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
            <label>Note (the seed for the worker brief)</label>
            <textarea rows={5} value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} />
            <div className="modal-actions">
              <button className="act-btn" onClick={() => setEditing(false)}>Cancel</button>
              <button className="act-btn primary" onClick={saveEdit} disabled={busy || !form.title.trim()}>Save</button>
            </div>
          </div>
        ) : (
          <>
            <h3>{form.title}</h3>
            <p className="modal-note">{form.note || "No description yet."}</p>
            <LiveLine live={task.live} />
            <div className="modal-status">Status: {task.status}</div>
            <div className="modal-effort">
              <span className="modal-effort-label">Intensity</span>
              <EffortPicker
                value={effort}
                disabled={busy}
                onPick={(e) => { setEffort(e); setTaskEffort(task.direction, task.id, e).then(onChanged); }}
              />
              <span className="modal-effort-hint">{effortMeta(effort).hint}</span>
            </div>
            <div className="modal-effort">
              <span className="modal-effort-label">Time budget</span>
              <span className="budget-edit">
                <input type="number" min="1" max="600" step="5" value={budget} disabled={busy}
                  onChange={(e) => setBudget(e.target.value)}
                  onBlur={() => { const v = Math.max(1, Math.min(600, parseInt(budget, 10) || 40)); setBudget(v); setTaskBudget(task.direction, task.id, v).then(onChanged); }} />
                <span className="budget-unit">min</span>
              </span>
              <span className="modal-effort-hint">Soft expectation — the orchestrator checks in and steps in if a worker goes silent or blows past this.</span>
            </div>
            {task.rework_history?.length > 0 && (
              <div className="modal-rework">
                <span className="modal-rework-flag">⚑ Sent back with notes</span>
                <ul>{task.rework_history.map((h, i) => <li key={i}>{h.note}</li>)}</ul>
              </div>
            )}
            {confirmDelete ? (
              <div className="confirm-box">
                <p><strong>Delete this task?</strong> It is removed from the dashboard entirely. This cannot be undone here.</p>
                <div className="modal-actions">
                  <button className="act-btn" onClick={() => setConfirmDelete(false)}>Cancel</button>
                  <button className="act-btn danger" disabled={busy} onClick={doDelete}>Delete</button>
                </div>
              </div>
            ) : (
              <div className="modal-actions">
                {task.status === "proposed" && (
                  <button className="act-btn primary" onClick={() => move("queued")}>Queue</button>
                )}
                {task.status === "queued" && (
                  <button className="act-btn" onClick={() => move("proposed")}>Move to Proposed</button>
                )}
                <button className="act-btn" onClick={() => setEditing(true)}>Edit</button>
                <button className="act-btn danger" onClick={() => setConfirmDelete(true)}>Delete</button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// New-task authoring. Directions are gone from the UI entirely — a task is described by its TAGS and its
// place in the graph, and the storage direction is chosen server-side as an implementation detail.
// Compact date shown on a card so the board reads chronologically at a glance.
const shortDate = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const s = d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
  return d.getFullYear() === new Date().getFullYear() ? s : `${s} ${String(d.getFullYear()).slice(2)}`;
};

const TAGS = ["gradients", "materials", "learned", "rendering", "demo"];
const TAG_COLORS = { gradients: "#4cc2ff", materials: "#ffb037", learned: "#c98bff", rendering: "#5ee0c8", demo: "#ff7bb0" };

function AuthorModal({ onClose, onChanged, tagOptions, onTagCreated }) {
  const [form, setForm] = useState({ title: "", note: "", tags: [] });
  const [busy, setBusy] = useState(false);
  const toggle = (t) =>
    setForm((f) => ({ ...f, tags: f.tags.includes(t) ? f.tags.filter((x) => x !== t) : [...f.tags, t] }));

  const submit = async () => {
    if (!form.title.trim()) return;
    setBusy(true);
    try {
      await createTask(form.title.trim(), form.note.trim(), "queued", form.tags);
      onChanged();
      onClose();
    } finally { setBusy(false); }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="close">×</button>
        <h3>New task</h3>
        <div className="author-form">
          <label>Title</label>
          <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })}
                 placeholder="What the task should accomplish" />
          <label>Tags</label>
          <TagPicker options={tagOptions && tagOptions.length ? tagOptions
                                : TAGS.map((t) => ({ name: t, color: TAG_COLORS[t] }))}
                     value={form.tags}
                     onChange={(tags) => setForm((f) => ({ ...f, tags }))}
                     onTagCreated={onTagCreated} />
          <label>Note (seed for the worker brief)</label>
          <textarea rows={5} value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })}
                    placeholder="A sentence or two; the orchestrator expands this into a full brief." />
          <p className="author-hint">
            The orchestrator places this in the task graph and derives its links — and re-checks the whole
            graph after it runs, when the result shows what it really was.
          </p>
          <div className="modal-actions">
            <button className="act-btn" onClick={onClose}>Cancel</button>
            <button className="act-btn primary" onClick={submit} disabled={busy || !form.title.trim()}>Add to Queued</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function OverviewView({ overview, onOpenTask, onChange, focus, onFocusHandled,
                                       tagOptions, onTagCreated }) {
  const [filter, setFilter] = useState("all");
  const [showDone, setShowDone] = useState(false);
  const [modal, setModal] = useState(null);
  const [drag, setDrag] = useState(null);
  const [over, setOver] = useState(null);
  const [author, setAuthor] = useState(null); // truthy while the New-task modal is open

  const dirs = overview?.directions || [];

  // Follow-up navigation from a task view: focus a proposal on the board (set its direction filter and
  // open its modal). Runs land in the task view instead (handled up in App), so here focus is a proposal.
  useEffect(() => {
    if (!focus || !dirs.length) return;
    const d = dirs.find((x) => x.id === focus.direction);
    const t = d?.tasks.find((x) => x.id === focus.id);
    if (t) {
      setFilter(focus.direction);
      setModal({ ...t, direction: d.id, directionName: d.name });
    }
    onFocusHandled && onFocusHandled();
  }, [focus]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!overview) return <div className="muted pad">Loading…</div>;

  // Tasks are filtered by TAG now. `direction` still rides along because it is the storage key the
  // status/edit endpoints need — it is no longer anything the user sees or chooses.
  const tasks = dirs
    .flatMap((d) => d.tasks.map((t) => ({ ...t, direction: d.id, directionName: d.name })))
    .filter((t) => filter === "all" || (t.tags || []).includes(filter));

  const done = tasks.filter((t) => t.status === "done");
  const columns = COLUMNS.map((c) => ({ ...c, items: tasks.filter((t) => t.status === c.id) }));

  const openCard = (t) => (t.has_artifact ? onOpenTask(t) : setModal(t));
  const refresh = () => onChange && onChange();

  const drop = (colId) => {
    if (drag && DRAGGABLE.has(colId) && drag.status !== colId) {
      setTaskStatus(drag.direction, drag.id, colId).then(refresh);
    }
    setDrag(null);
    setOver(null);
  };


  return (
    <div className="overview">
      <div className="dir-filter">
        <button className={`chip ${filter === "all" ? "active" : ""}`} onClick={() => setFilter("all")}>All</button>
        {TAGS.map((t) => (
          <button key={t} className={`chip tagchip ${filter === t ? "active" : ""}`}
                  style={{ "--tc": TAG_COLORS[t] }} onClick={() => setFilter(t)}>
            {t}
          </button>
        ))}
        <span className="dir-filter-spacer" />
        <button className="chip add" onClick={() => setAuthor(true)}>+ Task</button>
      </div>

      <div className="board3">
        {columns.map((c) => (
          <div
            className={`board-col col-${c.id} ${over === c.id && DRAGGABLE.has(c.id) ? "drop-ok" : ""}`}
            key={c.id}
            onDragOver={(e) => { if (drag && DRAGGABLE.has(c.id)) { e.preventDefault(); setOver(c.id); } }}
            onDragLeave={() => setOver((o) => (o === c.id ? null : o))}
            onDrop={() => drop(c.id)}
          >
            <div className="board-col-head">{c.label} <span className="board-count">{c.items.length}</span></div>
            {c.items.map((t) => (
              <button
                key={t.direction + "/" + t.id}
                className={`task-card ${t.has_artifact ? "has" : ""} ${DRAGGABLE.has(t.status) ? "drag" : ""}`}
                draggable={DRAGGABLE.has(t.status)}
                onDragStart={() => setDrag({ direction: t.direction, id: t.id, status: t.status })}
                onDragEnd={() => { setDrag(null); setOver(null); }}
                onClick={() => openCard(t)}
              >
                <div className="task-card-title">{t.title}</div>
                <div className="task-card-stamp">
                  {t.ref && <span className="task-card-ref">{t.ref}</span>}
                  {t.created && <span className="task-card-date">{shortDate(t.created)}</span>}
                </div>
                <div className="task-card-dir">
                  {t.directionName}
                  {t.rework_history?.length > 0 && <span className="card-flag" title="sent back with notes">⚑</span>}
                  {t.effort && t.effort !== "standard" && (
                    <span className={`effort-tag effort-${t.effort}`} title={effortMeta(t.effort).hint}>
                      {effortMeta(t.effort).label}
                    </span>
                  )}
                </div>
                {c.id === "active" && <LiveLine live={t.live} />}
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
        <TaskModal
          task={modal}
          onClose={() => setModal(null)}
          onChanged={refresh}
        />
      )}

      {author && (
        <AuthorModal tagOptions={tagOptions} onTagCreated={onTagCreated} onClose={() => setAuthor(null)} onChanged={refresh} />
      )}
    </div>
  );
}
