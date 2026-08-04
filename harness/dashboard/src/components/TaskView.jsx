import { useEffect, useRef, useState } from "react";
import { fetchTask, fetchJSON, fetchText, fetchTraining, fetchOverview, setTaskStatus, deleteTask, proposeFollowUp, fetchDefinitions, addTaskNote, deleteTaskNote } from "../api.js";
import LossChart from "./LossChart.jsx";
import MarkdownReport from "./MarkdownReport.jsx";
import VideoPlayer from "./VideoPlayer.jsx";
import LiveLine from "./LiveLine.jsx";

// A loss plot whose series is fetched from a referenced metrics.json.
function LossResult({ series, log }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let alive = true;
    fetchJSON(series)
      .then((j) => alive && setData((Array.isArray(j) ? j : j.loss || []).map((v, i) => ({ iter: i, loss: v }))))
      .catch(() => {});
    return () => { alive = false; };
  }, [series]);
  return <LossChart series={data} log={log !== false} />;
}

// Injected into every bespoke page so the frame can size itself to its content. A designed page must not
// be trapped in a short scrolling box -- that was the whole reason custom_html read as a footnote.
const AUTOSIZE = `
<script>(function(){
  function send(){
    var d=document.documentElement,b=document.body;
    var h=Math.max(d?d.scrollHeight:0,b?b.scrollHeight:0,d?d.offsetHeight:0);
    try{parent.postMessage({__taskFrame:true,height:h},'*');}catch(e){}
  }
  window.addEventListener('load',send); window.addEventListener('resize',send);
  if(window.ResizeObserver){try{new ResizeObserver(send).observe(document.documentElement);}catch(e){}}
  setTimeout(send,50); setTimeout(send,400); setTimeout(send,1500);
})();</script>`;

// The task's own page: arbitrary self-contained HTML/JS the task authored to present its result the way
// that result deserves. Sandboxed (scripts only, no same-origin), so it cannot reach the parent or network.
function BespokePage({ html }) {
  const ref = useRef(null);
  const [height, setHeight] = useState(null);
  useEffect(() => {
    function onMsg(e) {
      if (!e.data || e.data.__taskFrame !== true) return;
      if (!ref.current || e.source !== ref.current.contentWindow) return;
      const h = Number(e.data.height);
      if (Number.isFinite(h)) setHeight(Math.min(Math.max(h, 200), 6000));
    }
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, []);
  return (
    <iframe
      ref={ref}
      className="custom-frame lead"
      style={height ? { height: height + "px" } : undefined}
      srcDoc={html + AUTOSIZE}
      sandbox="allow-scripts"
      title="task result"
    />
  );
}

// ── Metric definitions on hover ────────────────────────────────────────────────────────────────────
// The registry (spec/definitions.json) is loaded once and shared, so any table header naming a known
// metric explains itself. Undefined metrics were a real defect: "trajectory RMSE" is neither an RMS nor a
// centre-of-mass distance, and a wrong mechanism built on that misreading reached three artifacts.
let _defsCache = null;
function useDefinitions() {
  const [defs, setDefs] = useState(_defsCache);
  useEffect(() => {
    if (_defsCache) return;
    let alive = true;
    fetchDefinitions()
      .then((d) => { _defsCache = d || {}; if (alive) setDefs(_defsCache); })
      .catch(() => { _defsCache = {}; });
    return () => { alive = false; };
  }, []);
  return defs || {};
}

const _norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "");

// Match a column/label against the registry by key or display label.
function lookupDef(defs, text) {
  const t = _norm(text);
  if (!t) return null;
  for (const [k, v] of Object.entries(defs)) {
    if (t === _norm(k) || t === _norm(v.label)) return v;
  }
  for (const [k, v] of Object.entries(defs)) {
    if (t.includes(_norm(k)) || t.includes(_norm(v.label))) return v;
  }
  return null;
}

// A real popover, not a `title` attribute. Native tooltips never fire on touch, and this dashboard is a
// pinned PWA on an iPad -- so the definition has to open on CLICK as well as hover, or it does not exist.
function DefTerm({ text, defs }) {
  const d = lookupDef(defs, text);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState(null);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const close = (e) => { if (!ref.current || !ref.current.contains(e.target)) setOpen(false); };
    const esc = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, [open]);

  if (!d) return <>{text}</>;

  const show = () => {
    const r = ref.current && ref.current.getBoundingClientRect();
    if (r) setPos({ left: Math.min(r.left, window.innerWidth - 380), top: r.bottom + 6 });
    setOpen(true);
  };

  return (
    <span
      ref={ref}
      className={"defterm" + (open ? " open" : "")}
      onClick={(e) => { e.stopPropagation(); open ? setOpen(false) : show(); }}
      onMouseEnter={show}
      onMouseLeave={() => setOpen(false)}
    >
      {text}
      {open && pos && (
        <span className="defpop" style={{ left: pos.left, top: pos.top }} onClick={(e) => e.stopPropagation()}>
          <span className="defpop-h">{d.label || text}</span>
          {d.short && <span className="defpop-s">{d.short}</span>}
          {d.formula && <span className="defpop-r"><b>Formula</b> {d.formula}</span>}
          {d.units && <span className="defpop-r"><b>Units</b> {d.units}</span>}
          {d.range && <span className="defpop-r"><b>Range</b> {d.range}</span>}
          {d.caution && <span className="defpop-w">⚠ {d.caution}</span>}
          {d.source && <span className="defpop-src">{d.source}</span>}
        </span>
      )}
    </span>
  );
}

// ── The user's notes ───────────────────────────────────────────────────────────────────────────────
// A passive margin comment: a question, a doubt, a conclusion reached. It never changes task status.
// Rolled up to a single strip by default and unrolls on click, so a task with notes is marked without
// the notes eating the page.
function TaskNotes({ task, onChange }) {
  const notes = task.notes || [];
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    const text = draft.trim();
    if (!text) return;
    setBusy(true);
    try {
      await addTaskNote(task.direction, task.task_id, text);
      setDraft("");
      onChange && onChange();
    } finally { setBusy(false); }
  };
  const remove = async (ts) => {
    setBusy(true);
    try { await deleteTaskNote(task.direction, task.task_id, ts); onChange && onChange(); }
    finally { setBusy(false); }
  };

  return (
    <div className={"notes" + (open ? " open" : "") + (notes.length ? " has" : "")}>
      <button className="notes-tab" onClick={() => setOpen((o) => !o)}>
        <span className="notes-pin">🗒</span>
        <span className="notes-label">
          {notes.length ? `${notes.length} note${notes.length > 1 ? "s" : ""}` : "Add a note"}
        </span>
        <span className="notes-chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="notes-body">
          {notes.map((n) => (
            <div className="note" key={n.ts}>
              <div className="note-meta">
                <span>{n.author || "Sevan"}</span>
                <span className="note-ts">{String(n.ts).replace("T", " ").slice(0, 16)}</span>
                <button className="note-x" title="Delete note" disabled={busy} onClick={() => remove(n.ts)}>×</button>
              </div>
              <div className="note-text">{n.text}</div>
            </div>
          ))}
          <textarea
            className="note-input"
            rows={2}
            placeholder="A question, a doubt, something you liked, a conclusion you reached…"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); }}
          />
          <div className="note-actions">
            <span className="note-hint">⌘/Ctrl + Enter</span>
            <button className="act-btn primary" disabled={busy || !draft.trim()} onClick={submit}>Add note</button>
          </div>
        </div>
      )}
    </div>
  );
}

// One typed result. Anything absent or unknown renders nothing (no placeholders cluttering the view).
function Result({ r, defs = {} }) {
  let body = null;
  if (r.type === "video" && r.src) {
    body = <VideoPlayer src={r.src} />;
  } else if (r.type === "image" && r.src) {
    body = <img className="task-image" src={r.src} alt={r.caption || ""} />;
  } else if (r.type === "plot" && r.series) {
    body = <LossResult series={r.series} log={r.log} />;
  } else if (r.type === "table" && Array.isArray(r.rows)) {
    body = (
      <table className="result-table">
        {r.columns && (
          <thead><tr>{r.columns.map((c, i) => <th key={i}><DefTerm text={c} defs={defs} /></th>)}</tr></thead>
        )}
        <tbody>
          {r.rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>)}
        </tbody>
      </table>
    );
  }
  if (!body) return null;
  return (
    <figure className={"result type-" + r.type}>
      {body}
      {r.caption && <figcaption>{r.caption}</figcaption>}
    </figure>
  );
}

// A clickable reference to another task (a follow-up parent/child). Navigates to the run if it has
// one, otherwise jumps to the board where the proposal lives.
function TaskRef({ r, onOpenRef }) {
  return (
    <button className="task-ref" onClick={() => onOpenRef && onOpenRef(r.direction, r.id, r.has_artifact)}>
      <span className="task-ref-title">{r.title}</span>
      <span className={`status status-${r.status === "done" ? "done" : "active"} task-ref-status`}>{r.status}</span>
    </button>
  );
}

export default function TaskView({ detail, reloadToken, onChange, onDeleted, onOpenRef, onOpenTraining }) {
  const defs = useDefinitions();
  const [task, setTask] = useState(null);
  const [refs, setRefs] = useState([]);
  const [sendBack, setSendBack] = useState(false);   // reject / reopen -> back to queue with a note
  const [sendNote, setSendNote] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [proposing, setProposing] = useState(false);
  const [followForm, setFollowForm] = useState({ title: "", note: "" });
  const [siblings, setSiblings] = useState([]);        // other tasks in this direction (candidate extra parents)
  const [extraParents, setExtraParents] = useState([]); // ids of the additional parents the user checked
  const [citeQuery, setCiteQuery] = useState("");       // suggester filter
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!detail) { setTask(null); setRefs([]); return; }
    let alive = true;
    setTask(null);
    setRefs([]);
    setSendBack(false); setSendNote(""); setConfirmDelete(false); setProposing(false);
    setExtraParents([]); setSiblings([]);
    fetchTask(detail)
      .then((t) => {
        if (!alive) return;
        setTask(t);
        const ids = t.training_refs || [];
        if (ids.length) {
          fetchTraining()
            .then((toc) => {
              const all = (toc.groups || []).flatMap((g) => g.sections);
              const chosen = ids.map((id) => all.find((s) => s.id === id)).filter(Boolean);
              return Promise.all(
                chosen.map((s) => fetchText(s.url).then((b) => ({ ...s, body: b })).catch(() => ({ ...s, body: "*unavailable*" })))
              );
            })
            .then((loaded) => alive && setRefs(loaded))
            .catch(() => {});
        }
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [detail, reloadToken]);

  // When the propose form opens, pull EVERY other task as a citation candidate. The graph is
  // cross-direction now, and citing is a hint the orchestrator refines — so there is no reason to hide
  // tasks from another direction, which is exactly how connections got missed before.
  useEffect(() => {
    if (!proposing || !task) return;
    let alive = true;
    fetchOverview()
      .then((ov) => {
        const all = (ov.directions || []).flatMap((d) =>
          (d.tasks || []).map((t) => ({ ...t, direction: d.id, directionName: d.name })));
        const others = all.filter((t) => !(t.direction === task.direction && t.id === task.task_id));
        if (alive) setSiblings(others);
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [proposing, task?.direction, task?.task_id]);

  // Suggester ranking: already-checked first (so a citation never scrolls away), then same-direction and
  // same-tag tasks, then the rest. Free-text filters across title, direction and tags.
  const suggested = (() => {
    const q = citeQuery.trim().toLowerCase();
    const mine = new Set(task?.tags || []);
    const scored = siblings
      .filter((s) => !q || `${s.title} ${s.directionName} ${(s.tags || []).join(" ")}`.toLowerCase().includes(q))
      .map((s) => {
        let score = 0;
        if (extraParents.includes(s.id)) score -= 100;
        if (s.direction === task?.direction) score -= 10;
        score -= (s.tags || []).filter((t) => mine.has(t)).length * 4;
        if (s.status === "done") score -= 1;
        return { s, score };
      })
      .sort((a, b) => a.score - b.score || a.s.title.localeCompare(b.s.title));
    return scored.slice(0, q ? 40 : 14).map((x) => x.s);
  })();

  if (!detail) return <div className="muted pad">Select a task.</div>;
  if (!task) return <div className="muted pad">Loading task…</div>;

  const results = (task.results || []).filter(Boolean);
  const media = results.filter((r) => r.type !== "table"); // video/image/plot -> masonry
  const tableResults = results.filter((r) => r.type === "table"); // full-width, below
  const isDone = task.status === "done";
  const act = (status, note) => setTaskStatus(task.direction, task.task_id, status, note).then(() => onChange && onChange());

  const doDelete = async () => {
    setBusy(true);
    try {
      const r = await deleteTask(task.direction, task.task_id);
      if (r && r.ok) { onDeleted && onDeleted(); onChange && onChange(); }
    } finally {
      setBusy(false);
      setConfirmDelete(false);
    }
  };

  const submitFollowUp = async () => {
    if (!followForm.title.trim()) return;
    setBusy(true);
    try {
      // This task is always a parent; any checked siblings are additional parents.
      const parents = [task.task_id, ...extraParents];
      const r = await proposeFollowUp(task.direction, parents, followForm.title.trim(), followForm.note.trim());
      if (r && r.ok) {
        setProposing(false);
        setFollowForm({ title: "", note: "" });
        setExtraParents([]);
        onChange && onChange(); // refresh so the new follow-up shows up under "Follow-ups"
      }
    } finally {
      setBusy(false);
    }
  };
  const toggleParent = (id) =>
    setExtraParents((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));

  return (
    <div className="taskview">
      <div className="task-head">
        <div className="task-head-title">
          <h1>{task.title}</h1>
          <span className={`status status-${isDone ? "done" : "active"}`}>{isDone ? "Done" : "Active"}</span>
        </div>
        <div className="task-actions">
          {isDone ? (
            <button className="act-btn" onClick={() => setSendBack(true)}>Reopen</button>
          ) : (
            <>
              <button className="act-btn primary" onClick={() => act("done")}>Mark Done</button>
              <button className="act-btn" onClick={() => setSendBack(true)}>Send back to queue</button>
            </>
          )}
          <button className="act-btn" onClick={() => setProposing((p) => !p)}>Propose follow-up</button>
          <button className="act-btn danger" onClick={() => setConfirmDelete(true)}>Delete</button>
        </div>
      </div>
      <TaskNotes task={task} onChange={onChange} />
      {!isDone && <LiveLine live={task.live} className="task-live" />}

      {confirmDelete && (
        <div className="confirm-box">
          <p><strong>Delete this task?</strong> It is removed from the board and the Tasks list entirely.
          The run files stay on disk (recoverable from git), but nothing links to them anymore.</p>
          <div className="reopen-actions">
            <button className="act-btn" onClick={() => setConfirmDelete(false)}>Cancel</button>
            <button className="act-btn danger" disabled={busy} onClick={doDelete}>Delete</button>
          </div>
        </div>
      )}

      {sendBack && (
        <div className="reopen-box">
          <p>
            This sends the task back to the <strong>Queue</strong> with a flagged note so a worker knows
            what to change or extend on the next run. {isDone ? "It has already run" : "It is currently active"};
            say what should be different, otherwise there is nothing to act on.
          </p>
          <textarea
            className="reopen-input"
            value={sendNote}
            onChange={(e) => setSendNote(e.target.value)}
            placeholder="What needs changing or extending?"
            rows={3}
          />
          <div className="reopen-actions">
            <button className="act-btn" onClick={() => { setSendBack(false); setSendNote(""); }}>Cancel</button>
            <button
              className="act-btn primary"
              disabled={!sendNote.trim()}
              onClick={() => { act("queued", sendNote.trim()); setSendBack(false); setSendNote(""); }}
            >
              Send back with this note
            </button>
          </div>
        </div>
      )}

      {proposing && (
        <div className="reopen-box">
          <p>
            Spin this result into a <strong>proposed follow-up</strong> in the same direction — an
            extension or the next question it raises. The new proposal links back here, and this task
            links out to it. Check any other tasks in this direction it also builds on to link them too.
          </p>
          <div className="author-form">
            <label>Follow-up title</label>
            <input
              value={followForm.title}
              onChange={(e) => setFollowForm({ ...followForm, title: e.target.value })}
              placeholder="What the follow-up should accomplish"
            />
            <label>Note (seed for the worker brief)</label>
            <textarea
              rows={4}
              value={followForm.note}
              onChange={(e) => setFollowForm({ ...followForm, note: e.target.value })}
              placeholder="What to build on, and what this task left open."
            />
            {siblings.length > 0 && (
              <>
                <label>Also follows up on {extraParents.length > 0 ? `(${extraParents.length + 1} tasks)` : "(optional)"}</label>
                <input
                  className="parent-search"
                  placeholder="Search tasks to cite…"
                  value={citeQuery}
                  onChange={(e) => setCiteQuery(e.target.value)}
                />
                <div className="parent-picker">
                  <label className="parent-opt fixed" title="the task you are proposing from is always a parent">
                    <input type="checkbox" checked readOnly />
                    <span>{task.title}</span>
                  </label>
                  {suggested.map((s) => (
                    <label key={`${s.direction}/${s.id}`} className="parent-opt">
                      <input type="checkbox" checked={extraParents.includes(s.id)} onChange={() => toggleParent(s.id)} />
                      <span>{s.title}</span>
                      <span className="parent-opt-dir">{s.directionName}</span>
                      <span className={`status status-${s.status === "done" ? "done" : "active"} parent-opt-status`}>{s.status}</span>
                    </label>
                  ))}
                  {suggested.length === 0 && <div className="muted pad">No match.</div>}
                </div>
                <p className="parent-hint">
                  Citing is a <b>hint</b>, not the final graph — the orchestrator derives the real links
                  (and their kind: extends / re-does / refutes / applies) when it reviews the task.
                </p>
              </>
            )}
            <div className="reopen-actions">
              <button className="act-btn" onClick={() => setProposing(false)}>Cancel</button>
              <button className="act-btn primary" disabled={busy || !followForm.title.trim()} onClick={submitFollowUp}>
                Create proposal
              </button>
            </div>
          </div>
        </div>
      )}

      {((task.follow_up_of && task.follow_up_of.length > 0) || (task.follow_ups && task.follow_ups.length > 0)) && (
        <div className="followups">
          {task.follow_up_of && task.follow_up_of.length > 0 && (
            <div className="followups-row">
              <span className="followups-label">Follows up on</span>
              <div className="followups-list">
                {task.follow_up_of.map((r) => <TaskRef key={r.id} r={r} onOpenRef={onOpenRef} />)}
              </div>
            </div>
          )}
          {task.follow_ups && task.follow_ups.length > 0 && (
            <div className="followups-row">
              <span className="followups-label">Follow-ups</span>
              <div className="followups-list">
                {task.follow_ups.map((r) => <TaskRef key={r.id} r={r} onOpenRef={onOpenRef} />)}
              </div>
            </div>
          )}
        </div>
      )}

      {task.rework_history?.length > 0 && (
        <div className="rework-banner">
          <strong>This task was sent back to the queue.</strong> The result below reflects the last run.
          Requested changes:
          <ul>
            {task.rework_history.map((h, i) => <li key={i}>{h.note}</li>)}
          </ul>
        </div>
      )}

      {/* One sentence, first thing on the page: the punchline, for scanning many tasks quickly. */}
      {task.tldr && (
        <section className="task-block tldr">
          <h2>TL;DR</h2>
          <p>{task.tldr}</p>
        </section>
      )}

      {task.objective && (
        <section className="task-block">
          <h2>Objective</h2>
          <p>{task.objective}</p>
        </section>
      )}

      {/* Layered in the order a reader needs them:
            1. the tight summary (always shown, the human-legible anchor),
            2. the task's OWN page, if it authored one -- the main event, not a footnote,
            3. everything else (raw results, full findings, hypothesis, limits) as the evidence layer.
          A task with no bespoke page keeps its results grid shown directly, so older tasks do not regress. */}
      {(() => {
        const summaryText = task.summary || task.findings;
        const detail = [];
        if (task.full_report) detail.push(["Full findings", task.full_report]);
        else if (task.summary && task.findings && task.findings !== task.summary)
          detail.push(["Full findings", task.findings]);
        if (task.hypothesis) detail.push(["Why / hypothesis", task.hypothesis]);
        if (task.limitations) detail.push(["Limitations and scope", task.limitations]);

        const bespoke = !!task.custom_html;
        const resultsBody = results.length > 0 && (
          <>
            {media.length > 0 && (
              <div className="results-grid">
                {media.map((r, i) => <Result key={i} r={r} defs={defs} />)}
              </div>
            )}
            {tableResults.map((r, i) => <Result key={"t" + i} r={r} defs={defs} />)}
          </>
        );
        const resultsBlock = resultsBody && (
          <section className="task-block">
            <h2>Results</h2>
            {resultsBody}
          </section>
        );
        const prose = detail.map(([h, body], i) => (
          <section className="task-block" key={i}>
            <h2>{h}</h2>
            <p>{body}</p>
          </section>
        ));

        return (
          <>
            {summaryText && (
              <section className="task-block">
                <h2>Summary</h2>
                <p>{summaryText}</p>
              </section>
            )}

            {bespoke && <BespokePage html={task.custom_html} />}

            {bespoke
              ? (resultsBlock || prose.length > 0) && (
                  <details className="full-report evidence">
                    <summary>Evidence &amp; detail</summary>
                    {resultsBlock}
                    {prose}
                  </details>
                )
              : (
                  <>
                    {resultsBlock}
                    {prose.length > 0 && (
                      <details className="full-report">
                        <summary>Full report</summary>
                        {prose}
                      </details>
                    )}
                  </>
                )}
          </>
        );
      })()}

      {refs.length > 0 && (
        <section className="task-block">
          <h2>From the textbook</h2>
          {refs.map((s) => (
            <details key={s.id} className="xclude">
              <summary>
                <span className="xclude-title">{s.title}</span>
                <button
                  className="xclude-open"
                  title="Open this section in the Training report"
                  onClick={(e) => { e.preventDefault(); e.stopPropagation(); onOpenTraining && onOpenTraining(s.id); }}
                >
                  Open in Training ↗
                </button>
              </summary>
              <MarkdownReport markdown={s.body} />
            </details>
          ))}
        </section>
      )}
    </div>
  );
}
