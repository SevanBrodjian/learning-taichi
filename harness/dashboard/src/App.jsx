import { useEffect, useMemo, useState } from "react";
import { fetchIndex, fetchJSON } from "./api.js";
import RunView from "./components/RunView.jsx";
import TrainingView from "./components/TrainingView.jsx";
import DirectionsView from "./components/DirectionsView.jsx";
import ReportsView from "./components/ReportsView.jsx";
import InboxView from "./components/InboxView.jsx";

// Top-level sections. Runs is the experiment browser (Direction -> Run); the rest are the
// shared, repo-level doc sets served from main.
const SECTIONS = [
  { id: "runs", label: "Runs" },
  { id: "training", label: "Training" },
  { id: "directions", label: "Directions" },
  { id: "reports", label: "Reports" },
  { id: "inbox", label: "Inbox" },
];

const loadManifest = (entry) => fetchJSON(entry.manifest);

export default function App() {
  const [section, setSection] = useState("runs");
  const [index, setIndex] = useState(null);
  const [error, setError] = useState(null);
  const [branch, setBranch] = useState(null);
  const [runId, setRunId] = useState(null);
  const [manifest, setManifest] = useState(null);

  useEffect(() => {
    fetchIndex().then(setIndex).catch((e) => setError(String(e)));
  }, []);

  const runs = index?.runs ?? [];
  const branches = useMemo(() => [...new Set(runs.map((r) => r.branch))], [index]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (branches.length && (branch == null || !branches.includes(branch))) setBranch(branches[0]);
  }, [branches]); // eslint-disable-line react-hooks/exhaustive-deps

  const branchRuns = useMemo(() => runs.filter((r) => r.branch === branch), [runs, branch]);

  useEffect(() => {
    setRunId(branchRuns.length ? branchRuns[0].run_id : null);
  }, [branch, index]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const entry = runs.find((r) => r.branch === branch && r.run_id === runId);
    if (!entry) {
      setManifest(null);
      return;
    }
    let alive = true;
    loadManifest(entry)
      .then((m) => alive && setManifest(m))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [runId, branch, index]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="dot" /> learning-taichi
        </div>

        <nav className="sections">
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              className={`section-tab ${s.id === section ? "active" : ""}`}
              onClick={() => setSection(s.id)}
            >
              {s.label}
            </button>
          ))}
        </nav>

        {error && <div className="error">{error}</div>}

        {section === "runs" && (
          <>
            {branches.length > 1 && (
              <div className="side-block">
                <div className="side-label">Direction</div>
                <div className="branches">
                  {branches.map((b) => (
                    <button
                      key={b}
                      className={`branch ${b === branch ? "active" : ""}`}
                      onClick={() => setBranch(b)}
                      title={b}
                    >
                      {b.split("/").pop()}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="side-label">Runs</div>
            <nav className="runlist">
              {branchRuns.map((r) => (
                <button
                  key={r.run_id}
                  className={`runitem ${r.run_id === runId ? "active" : ""}`}
                  onClick={() => setRunId(r.run_id)}
                >
                  <span className="runtitle">{r.title || r.run_id}</span>
                  <span className={`status status-${r.status}`}>{r.status}</span>
                </button>
              ))}
              {index && branchRuns.length === 0 && <div className="muted pad">No runs yet.</div>}
              {!index && !error && <div className="muted pad">Loading runs…</div>}
            </nav>
          </>
        )}
      </aside>

      <main className="content">
        {section === "runs" && <RunView manifest={manifest} />}
        {section === "training" && <TrainingView />}
        {section === "directions" && <DirectionsView />}
        {section === "reports" && <ReportsView />}
        {section === "inbox" && <InboxView />}
      </main>
    </div>
  );
}
