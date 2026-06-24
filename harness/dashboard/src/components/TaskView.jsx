import { useEffect, useState } from "react";
import { fetchTask, fetchJSON, fetchText, fetchTraining } from "../api.js";
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

// One typed result. Anything absent or unknown renders nothing (no placeholders cluttering the view).
function Result({ r }) {
  let body = null;
  if (r.type === "video" && r.src) {
    body = <video className="task-video" src={r.src} autoPlay loop muted playsInline controls />;
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

export default function TaskView({ detail }) {
  const [task, setTask] = useState(null);
  const [refs, setRefs] = useState([]);

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
  }, [detail]);

  if (!detail) return <div className="muted pad">Select a task.</div>;
  if (!task) return <div className="muted pad">Loading task…</div>;

  const results = (task.results || []).filter(Boolean);

  return (
    <div className="taskview">
      <div className="task-head">
        <h1>{task.title}</h1>
        <span className={`status status-${task.status === "done" ? "done" : "active"}`}>
          {task.status === "done" ? "Done" : "Active"}
        </span>
      </div>

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
