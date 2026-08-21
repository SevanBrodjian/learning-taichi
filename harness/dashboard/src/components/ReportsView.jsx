import { useEffect, useState } from "react";
import { fetchEpochs, fetchReports } from "../api.js";
import DocView from "./DocView.jsx";
import EpochCard from "./EpochCard.jsx";

// The shippable research report(s), plus the epochs that froze earlier ones. An epoch is a cut across
// the whole project at an inflection point, and its report is the copy that passed the examination —
// so it belongs beside the live report rather than in a tab of its own (the nav does not need to grow
// for something read a few times a year).
export default function ReportsView() {
  const [reports, setReports] = useState(null);
  const [epochs, setEpochs] = useState([]);
  const [active, setActive] = useState(undefined); // {kind:"report"|"epoch", url, epoch?}

  useEffect(() => {
    fetchReports()
      .then((d) => {
        const list = d.reports || [];
        setReports(list);
        setActive(list[0] ? { kind: "report", url: list[0].url } : { kind: "report", url: null });
      })
      .catch(() => {
        setReports([]);
        setActive({ kind: "report", url: null });
      });
    fetchEpochs().then((d) => setEpochs(d.epochs || [])).catch(() => setEpochs([]));
  }, []);

  if (reports === null) return <div className="muted pad">Loading…</div>;

  // Nothing cut yet and a single report: the page is just the report, exactly as it was.
  if (!epochs.length && reports.length <= 1) {
    return (
      <DocView
        url={active?.url ?? null}
        empty="The shippable research report has not been started yet."
      />
    );
  }

  const isActive = (kind, key) =>
    active?.kind === kind && (kind === "report" ? active.url === key : active.epoch?.id === key);

  return (
    <div className="doc-layout">
      <nav className="toc">
        <div className="toc-group-title">Reports</div>
        {(reports.length ? reports : [{ id: "research_report", title: "Research report", url: null }]).map((r) => (
          <button
            key={r.id}
            className={`toc-item ${isActive("report", r.url) ? "active" : ""}`}
            onClick={() => setActive({ kind: "report", url: r.url })}
          >
            <span className="toc-item-title">{r.title}</span>
          </button>
        ))}
        {epochs.length > 0 && <div className="toc-group-title">Epochs</div>}
        {epochs.map((e) => (
          <button
            key={e.id}
            className={`toc-item ${isActive("epoch", e.id) ? "active" : ""}`}
            onClick={() => setActive({ kind: "epoch", url: e.report_url, epoch: e })}
          >
            <span className="toc-item-title">{e.n}. {e.title}</span>
          </button>
        ))}
      </nav>
      <article className="doc-body">
        {active?.kind === "epoch" ? (
          <>
            <EpochCard epoch={active.epoch} />
            <DocView
              url={active.url}
              readOnly
              empty="This epoch was cut without a report copy."
            />
          </>
        ) : (
          <DocView url={active?.url ?? null} empty="The shippable research report has not been started yet." />
        )}
      </article>
    </div>
  );
}
