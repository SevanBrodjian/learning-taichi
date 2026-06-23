import { useEffect, useState } from "react";
import { fetchReports } from "../api.js";
import DocView from "./DocView.jsx";

// The shippable research report(s). Slow-growing; often empty early on.
export default function ReportsView() {
  const [reports, setReports] = useState(null);
  const [activeUrl, setActiveUrl] = useState(undefined);

  useEffect(() => {
    fetchReports()
      .then((d) => {
        const list = d.reports || [];
        setReports(list);
        setActiveUrl(list[0]?.url ?? null);
      })
      .catch(() => {
        setReports([]);
        setActiveUrl(null);
      });
  }, []);

  if (reports === null) return <div className="muted pad">Loading…</div>;
  if (reports.length <= 1) {
    return <DocView url={activeUrl} empty="The shippable research report has not been started yet." />;
  }
  return (
    <div className="doc-layout">
      <nav className="toc">
        <div className="toc-group-title">Reports</div>
        {reports.map((r) => (
          <button
            key={r.id}
            className={`toc-item ${r.url === activeUrl ? "active" : ""}`}
            onClick={() => setActiveUrl(r.url)}
          >
            {r.title}
          </button>
        ))}
      </nav>
      <article className="doc-body">
        <DocView url={activeUrl} />
      </article>
    </div>
  );
}
