import { mediaUrl } from "../api.js";

// One cut epoch, read-only (coordination/epochs/README.md). An epoch is only meaningful as a bundle —
// the report as it passed, the frozen demo, the physics version that demo was built from, and the task
// graph at that instant — so the card shows the four together rather than letting them drift apart.
// Cutting one is a deliberate act (harness/tools/cut_epoch.py); nothing here can create or change one.
export default function EpochCard({ epoch }) {
  if (!epoch) return null;
  const cut = epoch.cut ? new Date(epoch.cut) : null;
  const when = cut && !isNaN(cut)
    ? cut.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })
    : epoch.cut;
  // The examination's own words, never re-graded here: PASS / RESUBMIT / INSUFFICIENT COVERAGE and the
  // overall percentage. `forced` means the epoch was cut without a pass — visible, not hidden.
  const verdict = epoch.report_verdict;
  const passed = String(verdict || "").toUpperCase() === "PASS";
  const score = typeof epoch.report_score === "number" ? ` ${epoch.report_score}%` : "";

  return (
    <div className="epoch-card">
      <div className="epoch-head">
        <span className="epoch-n">Epoch {epoch.n}</span>
        <span className="epoch-title">{epoch.title}</span>
        {verdict
          ? <span className={`epoch-verdict ${passed ? "v-pass" : "v-revise"}`}>{verdict}{score}</span>
          : epoch.forced && <span className="epoch-verdict v-revise">cut without a verdict</span>}
      </div>
      <dl className="epoch-facts">
        <div><dt>Cut</dt><dd>{when || "—"}</dd></div>
        <div><dt>Physics</dt><dd><code>{epoch.physics_version || "—"}</code></dd></div>
        <div><dt>Tasks</dt><dd>{epoch.task_count} · {epoch.edge_count} edges</dd></div>
        <div>
          <dt>Frozen demo</dt>
          <dd>
            {epoch.demo_url
              ? <a href={mediaUrl(epoch.demo_url)} target="_blank" rel="noreferrer">Open ↗</a>
              : <span className="muted">not frozen</span>}
          </dd>
        </div>
      </dl>
    </div>
  );
}
