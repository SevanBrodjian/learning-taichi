import { useEffect, useState } from "react";
import { fetchJSON, fetchText, fetchTraining } from "../api.js";
import LossChart from "./LossChart.jsx";
import MarkdownReport from "./MarkdownReport.jsx";
import SimPlayer from "./SimPlayer.jsx";

function toSeries(j) {
  const loss = Array.isArray(j) ? j : j.loss || [];
  return loss.map((v, i) => ({ iter: i, loss: v }));
}
const fmt = (x) => Number(x).toExponential(2);

export default function RunView({ manifest }) {
  const [series, setSeries] = useState(null);
  const [refs, setRefs] = useState([]); // transcluded textbook sections [{id,title,body}]
  const [legacy, setLegacy] = useState(null); // back-compat single training report

  useEffect(() => {
    let alive = true;
    setSeries(null);
    setRefs([]);
    setLegacy(null);

    const sp = manifest?.metrics?.series;
    if (sp) fetchJSON(sp).then((j) => alive && setSeries(toSeries(j))).catch(() => {});

    const refIds = manifest?.training_refs;
    if (refIds?.length) {
      fetchTraining()
        .then((toc) => {
          const all = (toc.groups || []).flatMap((g) => g.sections);
          const chosen = refIds.map((id) => all.find((s) => s.id === id)).filter(Boolean);
          return Promise.all(
            chosen.map((s) =>
              fetchText(s.url)
                .then((body) => ({ ...s, body }))
                .catch(() => ({ ...s, body: "*Section unavailable.*" }))
            )
          );
        })
        .then((loaded) => alive && setRefs(loaded))
        .catch(() => {});
    } else if (manifest?.reports?.training) {
      fetchText(manifest.reports.training).then((t) => alive && setLegacy(t)).catch(() => {});
    }
    return () => {
      alive = false;
    };
  }, [manifest]);

  if (!manifest) return <div className="muted pad">Select a run.</div>;
  const params = manifest.params || {};

  return (
    <div className="runview">
      <h1>{manifest.title || manifest.run_id}</h1>
      {manifest.summary && <p className="summary">{manifest.summary}</p>}
      <div className="meta">
        <span className={`status status-${manifest.status}`}>{manifest.status || "unknown"}</span>
        {manifest.metrics?.final_loss != null && <span>final loss {fmt(manifest.metrics.final_loss)}</span>}
        {manifest.metrics?.iterations != null && <span>{manifest.metrics.iterations} iters</span>}
        {manifest.created && <span className="muted">{manifest.created}</span>}
      </div>

      <div className="grid2">
        <section className="card">
          <h2>Optimization loss</h2>
          <LossChart series={series} />
        </section>
        <section className="card">
          <h2>Simulation</h2>
          <SimPlayer media={manifest.media} />
        </section>
      </div>

      {Object.keys(params).length > 0 && (
        <section className="card">
          <h2>Parameters</h2>
          <table className="params">
            <tbody>
              {Object.entries(params).map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{typeof v === "object" ? JSON.stringify(v) : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {(refs.length > 0 || legacy) && (
        <section className="card report">
          <h2>From the textbook</h2>
          {refs.length > 0 && (
            <p className="muted xclude-note">
              The teaching lives once in the <strong>Training</strong> textbook; this run links to the
              relevant sections.
            </p>
          )}
          {refs.map((s) => (
            <details key={s.id} className="xclude" open>
              <summary>{s.title}</summary>
              <MarkdownReport markdown={s.body} />
            </details>
          ))}
          {legacy && <MarkdownReport markdown={legacy} />}
        </section>
      )}
    </div>
  );
}
