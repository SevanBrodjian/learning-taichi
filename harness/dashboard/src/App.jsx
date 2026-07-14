import { useEffect, useMemo, useState } from "react";
import { fetchOverview, fetchTraining, fetchNotifications } from "./api.js";
import OverviewView from "./components/OverviewView.jsx";
import GraphView from "./components/GraphView.jsx";
import TaskView from "./components/TaskView.jsx";
import TrainingView from "./components/TrainingView.jsx";
import ReportsView from "./components/ReportsView.jsx";
import InboxView from "./components/InboxView.jsx";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "map", label: "Map" },
  { id: "tasks", label: "Tasks" },
  { id: "training", label: "Training" },
  { id: "reports", label: "Reports" },
  { id: "inbox", label: "Inbox" },
];

// The iPad PWA reloads from scratch when resumed after even a few seconds in the background, so we
// persist the current place (section, filter, open task) and restore it on load (#12).
const PLACE_KEY = "lt_place";
const READ_KEY = "lt_training_read";     // written by TrainingView; read here for the "New" badge
const SEEN_KEY = "lt_notifs_seen";       // notification ids whose badge has been cleared (Inbox opened)
const loadPlace = () => {
  try { return JSON.parse(localStorage.getItem(PLACE_KEY) || "{}"); } catch { return {}; }
};
const loadReadMap = () => {
  try { return JSON.parse(localStorage.getItem(READ_KEY) || "{}"); } catch { return {}; }
};
const loadSeen = () => {
  try { return new Set(JSON.parse(localStorage.getItem(SEEN_KEY) || "[]")); } catch { return new Set(); }
};

// A field is focused when the user is typing/selecting; we pause background refetches then so a
// mid-keystroke re-render can't stutter the input (the reported typing lag).
const isTyping = () => {
  const el = document.activeElement;
  return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
};

// Phone layout kicks in only below 600px — iPad (>=768) and laptops are untouched.
function useMediaQuery(query) {
  const [match, setMatch] = useState(() => (typeof window !== "undefined" ? window.matchMedia(query).matches : false));
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setMatch(mq.matches);
    on();
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, [query]);
  return match;
}

export default function App() {
  const [place] = useState(loadPlace); // snapshot once at mount
  const [section, setSection] = useState(place.section || "overview");
  const [overview, setOverview] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [taskFilter, setTaskFilter] = useState(place.taskFilter || "all");
  const [reloadToken, setReloadToken] = useState(0); // bump to force task-detail refetch
  const [overviewFocus, setOverviewFocus] = useState(null); // follow-up nav target on the board

  // Badge data: the training TOC (for New sections) and the ntfy feed (for unread), lifted here so the
  // counts can sit on the nav. InboxView reuses the same feed instead of re-fetching.
  const [trainingToc, setTrainingToc] = useState(null);
  const [notifData, setNotifData] = useState(null);
  const [seenNotifs, setSeenNotifs] = useState(loadSeen);
  const [badgeToken, setBadgeToken] = useState(0); // bumped when a training section is read
  const [trainingTarget, setTrainingTarget] = useState(null); // deep-link: open this section in Training

  const isPhone = useMediaQuery("(max-width: 600px)");

  // Persist place whenever it changes.
  useEffect(() => {
    localStorage.setItem(
      PLACE_KEY,
      JSON.stringify({ section, taskFilter, selectedKey: selected?.key })
    );
  }, [section, taskFilter, selected?.key]);

  // After any write-back (drag / Mark Done) re-pull the board and re-fetch the open task.
  const reloadOverview = () => fetchOverview().then(setOverview).catch((e) => setError(String(e)));
  // Write-backs also bump the token so the open task re-fetches. The poll below only refreshes the
  // board, so external changes (orchestrator flips, finished workers) appear live without remounting
  // and restarting an open task's video.
  const reloadAll = () => { reloadOverview(); setReloadToken((k) => k + 1); };
  useEffect(() => {
    reloadOverview();
    const id = setInterval(() => { if (!document.hidden && !isTyping()) reloadOverview(); }, 4000);
    return () => clearInterval(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Badge feeds, refreshed on a slower cadence than the board.
  useEffect(() => {
    const loadBadges = () => {
      fetchTraining().then(setTrainingToc).catch(() => {});
      fetchNotifications().then(setNotifData).catch(() => {});
    };
    loadBadges();
    const id = setInterval(() => { if (!document.hidden) loadBadges(); }, 20000);
    return () => clearInterval(id);
  }, []);

  const realTasks = useMemo(() => {
    const dirs = overview?.directions || [];
    return dirs.flatMap((d) =>
      d.tasks
        .filter((t) => t.has_artifact)
        .map((t) => ({ ...t, direction: d.id, directionName: d.name, key: `${d.id}/${t.id}` }))
    );
  }, [overview]);

  useEffect(() => {
    if (selected || !realTasks.length) return;
    const restored = place.selectedKey && realTasks.find((t) => t.key === place.selectedKey);
    setSelected(restored || realTasks[0]);
  }, [realTasks]); // eslint-disable-line react-hooks/exhaustive-deps

  const openTask = (t) => {
    setSelected({ ...t, key: `${t.direction}/${t.id}` });
    setSection("tasks");
  };

  // Navigate a follow-up link: to the run if it has one, otherwise to its proposal on the board.
  const openRef = (direction, id, hasArtifact) => {
    if (hasArtifact) {
      const t = realTasks.find((x) => x.direction === direction && x.id === id);
      if (t) { setSelected(t); setSection("tasks"); return; }
    }
    setOverviewFocus({ direction, id });
    setSection("overview");
  };

  const onTaskDeleted = () => { setSelected(null); setSection("overview"); };

  // A task page's embedded textbook section links out to the full Training view, opened on that section.
  const openTraining = (sectionId) => { setTrainingTarget(sectionId); setSection("training"); };

  const groups = useMemo(() => {
    const m = new Map();
    for (const t of realTasks) {
      if (!m.has(t.direction)) m.set(t.direction, { name: t.directionName, tasks: [] });
      m.get(t.direction).tasks.push(t);
    }
    return [...m.entries()].map(([id, v]) => ({ id, ...v }));
  }, [realTasks]);

  const visibleGroups = taskFilter === "all" ? groups : groups.filter((g) => g.id === taskFilter);

  // Badge counts. New training = sections never opened or edited since last open (per this device).
  const trainingSections = useMemo(() => (trainingToc?.groups || []).flatMap((g) => g.sections), [trainingToc]);
  const newTrainingCount = useMemo(() => {
    const rm = loadReadMap();
    return trainingSections.filter((s) => s.mtime && (!(s.id in rm) || s.mtime > rm[s.id])).length;
  }, [trainingSections, badgeToken, section]);
  const notifList = notifData?.notifications || [];
  const unreadNotifCount = useMemo(
    () => notifList.filter((n) => !seenNotifs.has(n.id)).length,
    [notifList, seenNotifs]
  );
  const sectionBadge = (id) => (id === "training" ? newTrainingCount : id === "inbox" ? unreadNotifCount : 0);

  // Opening the Inbox clears the unread badge (all current notifications become "seen").
  useEffect(() => {
    if (section !== "inbox" || !notifList.length) return;
    setSeenNotifs((prev) => {
      const next = new Set(prev);
      let changed = false;
      for (const n of notifList) if (!next.has(n.id)) { next.add(n.id); changed = true; }
      if (changed) localStorage.setItem(SEEN_KEY, JSON.stringify([...next]));
      return changed ? next : prev;
    });
  }, [section, notifData]); // eslint-disable-line react-hooks/exhaustive-deps

  // Training and Reports are two-pane doc views: the section list and the article get independent
  // scrolling (via .content-split) so reaching the rest of the TOC doesn't require scrolling the
  // article to the bottom first.
  const isSplit = section === "training" || section === "reports";
  const mainContent = (
    <main className={`content${isSplit ? " content-split" : ""}${section === "map" ? " content-map" : ""}`}>
      {section === "overview" && (
        <OverviewView
          overview={overview}
          onOpenTask={openTask}
          onChange={reloadAll}
          focus={overviewFocus}
          onFocusHandled={() => setOverviewFocus(null)}
        />
      )}
      {section === "map" && (
        <GraphView overview={overview} onOpenTask={openTask} onOpenRef={openRef} />
      )}
      {section === "tasks" && (
        <TaskView
          detail={selected?.detail}
          reloadToken={reloadToken}
          onChange={reloadAll}
          onDeleted={onTaskDeleted}
          onOpenRef={openRef}
          onOpenTraining={openTraining}
        />
      )}
      {section === "training" && (
        <TrainingView
          onRead={() => setBadgeToken((t) => t + 1)}
          target={trainingTarget}
          onTargetHandled={() => setTrainingTarget(null)}
        />
      )}
      {section === "reports" && <ReportsView />}
      {section === "inbox" && <InboxView notifData={notifData} />}
    </main>
  );

  if (isPhone) {
    return (
      <div className="app phone">
        <header className="mobile-bar">
          <div className="brand"><span className="dot" /> learning-taichi</div>
          <select className="mobile-select" value={section} onChange={(e) => setSection(e.target.value)}>
            {SECTIONS.map((s) => {
              const b = sectionBadge(s.id);
              return <option key={s.id} value={s.id}>{s.label}{b > 0 ? ` (${b})` : ""}</option>;
            })}
          </select>
          {section === "tasks" && groups.length > 0 && (
            <select
              className="mobile-select"
              value={selected?.key || ""}
              onChange={(e) => { const t = realTasks.find((x) => x.key === e.target.value); if (t) setSelected(t); }}
            >
              {groups.map((g) => (
                <optgroup key={g.id} label={g.name}>
                  {g.tasks.map((t) => <option key={t.key} value={t.key}>{t.title}</option>)}
                </optgroup>
              ))}
            </select>
          )}
          {error && <div className="error">{error}</div>}
        </header>
        {mainContent}
      </div>
    );
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="dot" /> learning-taichi
        </div>

        <nav className="sections">
          {SECTIONS.map((s) => {
            const b = sectionBadge(s.id);
            return (
              <button
                key={s.id}
                className={`section-tab ${s.id === section ? "active" : ""}`}
                onClick={() => setSection(s.id)}
              >
                {s.label}
                {b > 0 && <span className="tab-badge">{b}</span>}
              </button>
            );
          })}
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

      {mainContent}
    </div>
  );
}
