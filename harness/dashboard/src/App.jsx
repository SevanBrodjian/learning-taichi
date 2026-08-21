import { useEffect, useMemo, useState } from "react";
import { fetchOverview, fetchTraining, fetchNotifications, fetchTags } from "./api.js";
import OverviewView from "./components/OverviewView.jsx";
import GraphView from "./components/GraphView.jsx";
import TaskView from "./components/TaskView.jsx";
import TrainingView from "./components/TrainingView.jsx";
import ReportsView from "./components/ReportsView.jsx";
import InboxView from "./components/InboxView.jsx";
import DemoView from "./components/DemoView.jsx";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "map", label: "Map" },
  { id: "tasks", label: "Tasks" },
  { id: "training", label: "Training" },
  { id: "reports", label: "Reports" },
  { id: "inbox", label: "Inbox" },
  { id: "demo", label: "Demo" },
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
// Compact date for the task list: "12 Aug" this year, "12 Aug 25" otherwise.
const shortDate = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const s = d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
  return d.getFullYear() === new Date().getFullYear()
    ? s
    : `${s} ${String(d.getFullYear()).slice(2)}`;
};

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
  const [taskSort, setTaskSort] = useState(place.taskSort || "newest");
  const [reloadToken, setReloadToken] = useState(0); // bump to force task-detail refetch
  const [overviewFocus, setOverviewFocus] = useState(null); // follow-up nav target on the board
  const [mapFocus, setMapFocus] = useState(null);         // "dir/id" to centre + select on the Map
  const [tagReg, setTagReg] = useState(null);             // server tag registry (see fetchTags)

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
      JSON.stringify({ section, taskFilter, taskSort, selectedKey: selected?.key })
    );
  }, [section, taskFilter, taskSort, selected?.key]);

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
      fetchTags().then(setTagReg).catch(() => {});
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

  // Jump from a task page to that task on the Map, selected and centred.
  const openOnMap = (direction, id) => { setMapFocus(`${direction}/${id}`); setSection("map"); };

  const onTaskDeleted = () => { setSelected(null); setSection("overview"); };

  // A task page's embedded textbook section links out to the full Training view, opened on that section.
  const openTraining = (sectionId) => { setTrainingTarget(sectionId); setSection("training"); };

  // One source of truth for tags. The registry can contain a tag no task uses yet (that is the point of
  // being able to make one), and the board can contain a tag the registry has not caught up with, so the
  // union is what the UI shows.
  const allTags = useMemo(() => {
    const m = new Map();
    for (const t of tagReg?.tags || []) m.set(t.name, t.color);
    for (const t of realTasks) for (const tg of t.tags || []) if (!m.has(tg)) m.set(tg, null);
    return [...m.entries()].map(([name, color]) => ({ name, color }));
  }, [tagReg, realTasks]);
  const tagColor = (name) => allTags.find((t) => t.name === name)?.color || "#7f8ea3";


  // Tasks are filtered by TAG now, not by direction — directions are storage, not the user's model
  // (CLAUDE.md), and the Map already sorts by tag. A 26-item flat list was hard to scan, so it also
  // sorts, and every row carries its ref and date.
  const visibleTasks = useMemo(() => {
    const list = taskFilter === "all"
      ? realTasks
      : realTasks.filter((t) => (t.tags || []).includes(taskFilter));
    const when = (t) => (t.created ? Date.parse(t.created) || 0 : 0);
    const refNum = (t) => {
      const m = /^T-(\d+)$/.exec(t.ref || "");
      return m ? Number(m[1]) : Number.MAX_SAFE_INTEGER;
    };
    const cmp = {
      newest: (a, b) => when(b) - when(a) || refNum(b) - refNum(a),
      oldest: (a, b) => when(a) - when(b) || refNum(a) - refNum(b),
      ref: (a, b) => refNum(a) - refNum(b),
      title: (a, b) => String(a.title).localeCompare(String(b.title)),
      active: (a, b) =>
        (a.status === "done" ? 1 : 0) - (b.status === "done" ? 1 : 0) || when(b) - when(a),
    }[taskSort] || ((a, b) => when(b) - when(a));
    return [...list].sort(cmp);
  }, [realTasks, taskFilter, taskSort]);

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
    <main className={`content${isSplit ? " content-split" : ""}${section === "map" ? " content-map" : ""}${section === "demo" ? " content-demo" : ""}`}>
      {section === "overview" && (
        <OverviewView
          overview={overview}
          onOpenTask={openTask}
          onChange={reloadAll}
          tagOptions={allTags}
          onTagCreated={() => fetchTags().then(setTagReg).catch(() => {})}
          focus={overviewFocus}
          onFocusHandled={() => setOverviewFocus(null)}
        />
      )}
      {section === "map" && (
        <GraphView overview={overview} onOpenTask={openTask} onOpenRef={openRef}
                   focusTask={mapFocus} onFocusHandled={() => setMapFocus(null)}
                   tagColors={allTags} />
      )}
      {section === "tasks" && (
        <TaskView
          detail={selected?.detail}
          reloadToken={reloadToken}
          onChange={reloadAll}
          onDeleted={onTaskDeleted}
          onOpenRef={openRef}
          onOpenOnMap={openOnMap}
          tagOptions={allTags}
          onTagCreated={() => fetchTags().then(setTagReg).catch(() => {})}
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
      {section === "demo" && <DemoView />}
    </main>
  );

  // The Tasks page is the ONLY page that wants a sidebar, so the sidebar belongs to it rather than to
  // the shell (where it sat as a blank spacer on every other page).
  const taskSidebar = (
    <nav className="task-nav">
      {allTags.length > 0 && (
        <div className="task-tagbar">
          <button className={`gtag ${taskFilter === "all" ? "active" : ""}`}
                  style={{ "--tc": "#7f8ea3" }}
                  onClick={() => setTaskFilter("all")}>all</button>
          {allTags.map((t) => (
            <button key={t.name}
                    className={`gtag ${taskFilter === t.name ? "active" : ""}`}
                    style={{ "--tc": tagColor(t.name) }}
                    onClick={() => setTaskFilter(taskFilter === t.name ? "all" : t.name)}>
              {t.name}
            </button>
          ))}
        </div>
      )}
      <div className="task-sortbar">
        <span className="task-count">{visibleTasks.length} task{visibleTasks.length === 1 ? "" : "s"}</span>
        <select className="task-sort" value={taskSort} onChange={(e) => setTaskSort(e.target.value)}>
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="ref">By ref</option>
          <option value="title">By title</option>
          <option value="active">Active first</option>
        </select>
      </div>
      <div className="task-group">
        {visibleTasks.map((t) => (
          <button
            key={t.key}
            className={`runitem ${selected?.key === t.key ? "active" : ""}`}
            onClick={() => setSelected(t)}
          >
            <span className="runline">
              {t.ref && <span className="runref">{t.ref}</span>}
              <span className="runtitle">{t.title}</span>
            </span>
            <span className="runmeta">
              <span className="rundate">{shortDate(t.created)}</span>
              <span className={`status status-${t.status === "done" ? "done" : "active"}`}>
                {t.status === "done" ? "Done" : "Active"}
              </span>
            </span>
          </button>
        ))}
      </div>
      {realTasks.length === 0 && <div className="muted pad">No tasks with results yet.</div>}
      {realTasks.length > 0 && visibleTasks.length === 0 && (
        <div className="muted pad">No tasks tagged “{taskFilter}”.</div>
      )}
    </nav>
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
          {section === "tasks" && visibleTasks.length > 0 && (
            <select
              className="mobile-select"
              value={selected?.key || ""}
              onChange={(e) => { const t = realTasks.find((x) => x.key === e.target.value); if (t) setSelected(t); }}
            >
              {visibleTasks.map((t) => (
                <option key={t.key} value={t.key}>{t.ref ? `${t.ref} · ` : ""}{t.title}</option>
              ))}
            </select>
          )}
          {error && <div className="error">{error}</div>}
        </header>
        {mainContent}
      </div>
    );
  }

  // Shell: a fixed tab strip on top, one full-width page beneath. The shell itself never scrolls —
  // each page scrolls its own content, and pages that shouldn't scroll (Map, Demo) simply don't.
  return (
    <div className="app shell">
      <header className="tabbar">
        <div className="brand"><span className="dot" /> learning-taichi</div>
        <nav className="tabstrip">
          {SECTIONS.map((s) => {
            const b = sectionBadge(s.id);
            return (
              <button
                key={s.id}
                className={`tab-item ${s.id === section ? "active" : ""}${s.id === "demo" ? " tab-demo" : ""}`}
                onClick={() => setSection(s.id)}
              >
                {s.label}
                {b > 0 && <span className="tab-badge">{b}</span>}
              </button>
            );
          })}
        </nav>
        {error && <div className="error tabbar-error">{error}</div>}
      </header>

      {section === "tasks" ? (
        <div className="page page-tasks">
          <aside className="task-side">{taskSidebar}</aside>
          {mainContent}
        </div>
      ) : (
        <div className="page">{mainContent}</div>
      )}
    </div>
  );
}
