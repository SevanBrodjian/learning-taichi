import { useEffect, useRef, useState } from "react";
import { fetchTask, fetchJSON, fetchText, fetchTraining, setTaskStatus } from "../api.js";
import LossChart from "./LossChart.jsx";
import MarkdownReport from "./MarkdownReport.jsx";

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

// Clean autoplay loop with on-demand controls. Native `controls` on iOS keeps a big overlay up over an
// autoplaying video until tapped, which is obstructive when flicking through tasks. Instead the video
// plays clean and a minimal control bar (play/pause + scrub) appears on hover (desktop) or tap (touch).
function VideoResult({ src }) {
  const ref = useRef(null);
  const [playing, setPlaying] = useState(true);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);
  const [show, setShow] = useState(false);
  const toggle = () => {
    const v = ref.current;
    if (v) (v.paused ? v.play() : v.pause());
  };
  const fullscreen = () => {
    const v = ref.current;
    if (!v) return;
    if (v.requestFullscreen) v.requestFullscreen();
    else if (v.webkitEnterFullscreen) v.webkitEnterFullscreen(); // iOS Safari (native fullscreen player)
  };
  return (
    <div className="vid" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)} onClick={() => setShow((s) => !s)}>
      <video
        ref={ref}
        className="task-video"
        src={src}
        autoPlay loop muted playsInline preload="auto"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(e) => setCur(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDur(e.currentTarget.duration || 0)}
      />
      <div className={`vid-controls ${show ? "show" : ""}`} onClick={(e) => e.stopPropagation()}>
        <button className="vid-btn" onClick={toggle} aria-label={playing ? "pause" : "play"}>{playing ? "❚❚" : "▶"}</button>
        <input
          className="vid-seek" type="range" min="0" max={dur || 0} step="0.01" value={cur}
          onChange={(e) => { if (ref.current) ref.current.currentTime = parseFloat(e.target.value); }}
        />
        <button className="vid-btn" onClick={fullscreen} aria-label="fullscreen">⛶</button>
      </div>
    </div>
  );
}

// One typed result. Anything absent or unknown renders nothing (no placeholders cluttering the view).
function Result({ r }) {
  let body = null;
  if (r.type === "video" && r.src) {
    body = <VideoResult src={r.src} />;
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
    <figure className="result">
      {body}
      {r.caption && <figcaption>{r.caption}</figcaption>}
    </figure>
  );
}

export default function TaskView({ detail, reloadToken, onChange }) {
  const [task, setTask] = useState(null);
  const [refs, setRefs] = useState([]);
  const [reopening, setReopening] = useState(false);
  const [reopenNote, setReopenNote] = useState("");

  useEffect(() => {
    if (!detail) { setTask(null); setRefs([]); return; }
    let alive = true;
    setTask(null);
    setRefs([]);
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

  if (!detail) return <div className="muted pad">Select a task.</div>;
  if (!task) return <div className="muted pad">Loading task…</div>;

  const results = (task.results || []).filter(Boolean);
  const act = (status, note) => setTaskStatus(task.direction, task.task_id, status, note).then(() => onChange && onChange());

  return (
    <div className="taskview">
      <div className="task-head">
        <h1>{task.title}</h1>
        <span className={`status status-${task.status === "done" ? "done" : "active"}`}>
          {task.status === "done" ? "Done" : "Active"}
        </span>
        <div className="task-actions">
          {task.status === "done" ? (
            <button className="act-btn" onClick={() => setReopening(true)}>Reopen</button>
          ) : (
            <button className="act-btn primary" onClick={() => act("done")}>Mark Done</button>
          )}
        </div>
      </div>

      {reopening && (
        <div className="reopen-box">
          <p>
            Reopening re-queues a task that has <em>already run</em>. Say what should change or be
            extended, otherwise a worker has no way to know what to do differently.
          </p>
          <textarea
            className="reopen-input"
            value={reopenNote}
            onChange={(e) => setReopenNote(e.target.value)}
            placeholder="What needs changing or extending?"
            rows={3}
          />
          <div className="reopen-actions">
            <button className="act-btn" onClick={() => { setReopening(false); setReopenNote(""); }}>Cancel</button>
            <button
              className="act-btn primary"
              disabled={!reopenNote.trim()}
              onClick={() => { act("queued", reopenNote.trim()); setReopening(false); setReopenNote(""); }}
            >
              Reopen with this note
            </button>
          </div>
        </div>
      )}

      {task.rework_history?.length > 0 && (
        <div className="rework-banner">
          <strong>This task already ran and was reopened.</strong> The result below reflects the last run.
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

      {task.findings && (
        <section className="task-block">
          <h2>Findings</h2>
          <p>{task.findings}</p>
        </section>
      )}

      {results.length > 0 && (
        <section className="task-block">
          <h2>Results</h2>
          <div className="results-grid">
            {results.map((r, i) => <Result key={i} r={r} />)}
          </div>
        </section>
      )}

      {task.custom_html && (
        <section className="task-block">
          <h2>Interactive</h2>
          <iframe className="custom-frame" srcDoc={task.custom_html} sandbox="allow-scripts" title="custom result" />
        </section>
      )}

      {refs.length > 0 && (
        <section className="task-block">
          <h2>From the textbook</h2>
          {refs.map((s) => (
            <details key={s.id} className="xclude">
              <summary>{s.title}</summary>
              <MarkdownReport markdown={s.body} />
            </details>
          ))}
        </section>
      )}
    </div>
  );
}
