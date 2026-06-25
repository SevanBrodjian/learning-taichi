import { useEffect, useMemo, useState } from "react";
import { fetchOverview } from "./api.js";
import OverviewView from "./components/OverviewView.jsx";
import TaskView from "./components/TaskView.jsx";
import TrainingView from "./components/TrainingView.jsx";
import ReportsView from "./components/ReportsView.jsx";
import InboxView from "./components/InboxView.jsx";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "tasks", label: "Tasks" },
  { id: "training", label: "Training" },
  { id: "reports", label: "Reports" },
  { id: "inbox", label: "Inbox" },
];

export default function App() {
  const [section, setSection] = useState("overview");
  const [overview, setOverview] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [taskFilter, setTaskFilter] = useState("all");
  const [reloadToken, setReloadToken] = useState(0); // bump to force task-detail refetch

  // After any write-back (drag / Mark Done) re-pull the board and re-fetch the open task.
  const reloadOverview = () => fetchOverview().then(setOverview).catch((e) => setError(String(e)));
  // Write-backs also bump the token so the open task re-fetches. The poll below only refreshes the
  // board, so external changes (orchestrator flips, finished workers) appear live without remounting
  // and restarting an open task's video.
  const reloadAll = () => { reloadOverview(); setReloadToken((k) => k + 1); };
  useEffect(() => {
    reloadOverview();
    const id = setInterval(() => { if (!document.hidden) reloadOverview(); }, 4000);
    return () => clearInterval(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const realTasks = useMemo(() => {
    const dirs = overview?.directions || [];
    return dirs.flatMap((d) =>
      d.tasks
        .filter((t) => t.has_artifact)
        .map((t) => ({ ...t, direction: d.id, directionName: d.name, key: `${d.id}/${t.id}` }))
    );
  }, [overview]);

  useEffect(() => {
    if (!selected && realTasks.length) setSelected(realTasks[0]);
  }, [realTasks]); // eslint-disable-line react-hooks/exhaustive-deps

  const openTask = (t) => {
    setSelected({ ...t, key: `${t.direction}/${t.id}` });
    setSection("tasks");
  };

  const groups = useMemo(() => {
    const m = new Map();
    for (const t of realTasks) {
      if (!m.has(t.direction)) m.set(t.direction, { name: t.directionName, tasks: [] });
      m.get(t.direction).tasks.push(t);
    }
    return [...m.entries()].map(([id, v]) => ({ id, ...v }));
  }, [realTasks]);

  const visibleGroups = taskFilter === "all" ? groups : groups.filter((g) => g.id === taskFilter);

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

        {section === "tasks" && (
          <nav className="task-nav">
            {groups.length > 1 && (
              <select className="task-filter" value={taskFilter} onChange={(e) => setTaskFilter(e.target.value)}>
                <option value="all">All directions</option>
                {groups.map((g) => (
                  <option key={g.id} value={g.id}>{g.name}</option>
                ))}
              </select>
            )}
            {visibleGroups.map((g) => (
              <div className="task-group" key={g.id}>
                <div className="side-label">{g.name}</div>
                {g.tasks.map((t) => (
                  <button
                    key={t.key}
                    className={`runitem ${selected?.key === t.key ? "active" : ""}`}
                    onClick={() => setSelected(t)}
                  >
                    <span className="runtitle">{t.title}</span>
                    <span className={`status status-${t.status === "done" ? "done" : "active"}`}>
                      {t.status === "done" ? "Done" : "Active"}
                    </span>
                  </button>
                ))}
              </div>
            ))}
            {groups.length === 0 && <div className="muted pad">No tasks with results yet.</div>}
          </nav>
        )}
      </aside>

      <main className="content">
        {section === "overview" && <OverviewView overview={overview} onOpenTask={openTask} onChange={reloadAll} />}
        {section === "tasks" && <TaskView detail={selected?.detail} reloadToken={reloadToken} onChange={reloadAll} />}
        {section === "training" && <TrainingView />}
        {section === "reports" && <ReportsView />}
        {section === "inbox" && <InboxView />}
      </main>
    </div>
  );
}
