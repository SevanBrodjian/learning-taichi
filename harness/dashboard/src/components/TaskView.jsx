import { useEffect, useRef, useState } from "react";
import { fetchTask, fetchJSON, fetchText, fetchTraining, fetchOverview, setTaskStatus, deleteTask, proposeFollowUp } from "../api.js";
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

// One typed result. Anything absent or unknown renders nothing (no placeholders cluttering the view).
function Result({ r }) {
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
          <thead><tr>{r.columns.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
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
  const [task, setTask] = useState(null);
  const [refs, setRefs] = useState([]);
  const [sendBack, setSendBack] = useState(false);   // reject / reopen -> back to queue with a note
  const [sendNote, setSendNote] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [proposing, setProposing] = useState(false);
  const [followForm, setFollowForm] = useState({ title: "", note: "" });
  const [siblings, setSiblings] = useState([]);        // other tasks in this direction (candidate extra parents)
  const [extraParents, setExtraParents] = useState([]); // ids of the additional parents the user checked
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

  // When the propose form opens, pull the other tasks in this direction as candidate extra parents so a
  // proposal can follow up on several at once (the graph is direction-local).
  useEffect(() => {
    if (!proposing || !task) return;
    let alive = true;
    fetchOverview()
      .then((ov) => {
        const dir = (ov.directions || []).find((d) => d.id === task.direction);
        const others = (dir?.tasks || []).filter((t) => t.id !== task.task_id);
        if (alive) setSiblings(others);
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [proposing, task?.direction, task?.task_id]);

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
                <div className="parent-picker">
                  <label className="parent-opt fixed" title="the task you are proposing from is always a parent">
                    <input type="checkbox" checked readOnly />
                    <span>{task.title}</span>
                  </label>
                  {siblings.map((s) => (
                    <label key={s.id} className="parent-opt">
                      <input type="checkbox" checked={extraParents.includes(s.id)} onChange={() => toggleParent(s.id)} />
                      <span>{s.title}</span>
                      <span className={`status status-${s.status === "done" ? "done" : "active"} parent-opt-status`}>{s.status}</span>
                    </label>
                  ))}
                </div>
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
                {media.map((r, i) => <Result key={i} r={r} />)}
              </div>
            )}
            {tableResults.map((r, i) => <Result key={"t" + i} r={r} />)}
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
