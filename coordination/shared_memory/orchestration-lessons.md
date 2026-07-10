# Orchestration lessons (hard-won, durable)

Cross-agent operational facts for whoever orchestrates next. Complements `CLAUDE.md` (the rules) and
`working-with-sevan.md` (the collaborator). These are things that actually bit a real session.

## Reviewing workers — evidence discipline is the load-bearing job
- **View the media yourself.** Two workers this project overclaimed in ways only the *pictures* revealed: a
  "smooth material morph" that was actually a degenerate spray, and "edge-exactness" that was 14–17% off the
  true sim. Reading the manifest prose is not enough — open the stills, sample video frames (`ffmpeg -sseof
  -1 -i clip.mp4 -frames:v 1 out.png`), and check every claim against `metrics.json`. Send back overclaims.
- **Watch for reframed claims.** A worker "verified edge-exactness" by silently redefining it (state-rule
  parity, not the material vs the true sim). Insist claims match the quantity the user asked about.
- A **negative / honest-tradeoff result is a real result** here and often the most valuable — do not let a
  worker (or yourself) dress it up. The best pages in the textbook are honest negatives.

## Workers punt on long background jobs (runtime friction, not laziness)
- Subagents running long GPU training/rendering **end their turn on a detached background job** ("I'll wait
  for the run to complete") — it's a runtime turn-budget cutoff, not a choice, and the `finish-in-turn`
  spec rule does not fully prevent it. It happened ~5× on `one-nn-for-three-materials`.
- **The work often already completed on disk.** Before reacting, check the run dir mtimes and whether
  `manifest.json` was written last — the punt message is frequently stale.
- To finalize: **`SendMessage(<agentId>, ...)`** resumes the agent with its context. If it keeps punting,
  **constrain the resume to narrative-only (no training/render)** so it fits one turn, or finalize yourself.
- **Schedule GPU-timing tasks alone.** A benchmark (the GPU-renderer speedup) must run with no other
  GPU-heavy worker or the numbers are corrupted (the standing GPU-contention lesson, applied to timings).

## Shared files and concurrency
- **`reports/training/index.json` races.** When several workers run in parallel they all want to append their
  training page to the index → last-writer-wins can drop entries or corrupt JSON. Standard now: **workers do
  NOT edit `index.json`; they report their page id/title/file and the orchestrator registers it at review.**
- **Manifest last, only-existing media.** A worker wrote its manifest up-front listing all planned scenes,
  then rendered them slowly → the dashboard showed broken tiles for the un-rendered ones. Now a spec rule:
  write the manifest last, `results[]` referencing only files that exist; the orchestrator verifies every
  `src` resolves before committing.
- **The live data server commits dashboard actions.** `_git_commit` was fixed to scope to its path (it used
  to commit the *whole* index and clobbered concurrently-staged hand edits). Still, do not leave unrelated
  files staged while the user is actively clicking the dashboard.

## Running the services
- **Data server:** `.venv\Scripts\python.exe harness\server\app.py` on port **8732**. The uv base python
  lacks uvicorn/fastapi — always use the repo `.venv`. Start detached (PowerShell `Start-Process ... -PassThru
  -WindowStyle Hidden -RedirectStandardOutput/Error`) so it survives the session. It is NOT reload-mode, so a
  code change to `app.py` needs a restart (kill the PID on 8732, relaunch).
- **Dashboard:** Vite on **5174** (strictPort). One instance only.
- **Interactive results** render via the manifest `custom_html` field (sandboxed iframe, scripts allowed,
  strict CSP) — must be fully self-contained (no external hosts/CDNs).

## Headless rendering / checks
- `imageio-ffmpeg` ships ffmpeg at `.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-*.exe` (no
  ffprobe — use `ffmpeg -i file 2>&1 | grep Duration`). Render sims headless (no `ti.GUI`) → frames → mp4.
- **KaTeX check** headlessly with the node `katex` in `harness/dashboard/node_modules` (see the scratch
  `katexcheck.mjs` pattern): extract `$$…$$` / `$…$`, `renderToString(throwOnError)`. Catches the silent
  "red paragraph" before it reaches the iPad.
- **Link check**: every training `[[link]]` target must be a section id in `reports/training/index.json`
  (decorative emphasis aside). Cheap python scan; run before committing a page.
