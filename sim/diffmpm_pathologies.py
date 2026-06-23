"""Gradient pathology instrumentation for DiffMPM.

Builds on sim/diffmpm.py.  Adds per-step forward-state diagnostics,
precision switching (f32/f64), gradient clipping, a horizon sweep,
contact-isolation experiments, and a mass-stabilisation fix.

Key design note: Taichi's reverse-mode autodiff does NOT store intermediate
gradients (v.grad[t]) after the backward pass completes; only input-leaf grads
(v0.grad) are reliably available.  Per-step gradient information is therefore
inferred from FORWARD-pass state (grid_m statistics, particle positions).

Experiments
-----------
instrument  forward-state diagnostics (grid-mass min per step) + v0.grad curve
horizon     sweep n_steps [128, 256, 512]; NaN iteration vs. horizon length
clip        mass-stabilisation fix; confirms clean optimisation for 80+ iters
isolate     center target vs. wall target; separates contact from long-rollout

Usage
-----
  python sim/diffmpm_pathologies.py --experiment instrument --steps 512 --iters 80
  python sim/diffmpm_pathologies.py --experiment instrument --steps 512 --iters 80 --f64
  python sim/diffmpm_pathologies.py --experiment horizon --iters 25
  python sim/diffmpm_pathologies.py --experiment clip --steps 512 --iters 100 --stabilize
  python sim/diffmpm_pathologies.py --experiment isolate --steps 512 --iters 25
  python sim/diffmpm_pathologies.py --experiment all --iters 80
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

# Parse --f64 before taichi initialises so we can set default_fp correctly.
_use_f64 = "--f64" in sys.argv

import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f64 if _use_f64 else ti.f32, random_seed=42)

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
dim = 2
n_grid = 64
n_particles = 4096
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 2e-4
p_rho = 1.0
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
E = 400.0
gravity = 9.8
bound = 3
MAX_STEPS = 1024          # allocate for the widest horizon we sweep

# ---------------------------------------------------------------------------
# Differentiable fields  (time-indexed, needs_grad=True)
# ---------------------------------------------------------------------------
x          = ti.Vector.field(dim, float, shape=(MAX_STEPS, n_particles),        needs_grad=True)
v          = ti.Vector.field(dim, float, shape=(MAX_STEPS, n_particles),        needs_grad=True)
C          = ti.Matrix.field(dim, dim, float, shape=(MAX_STEPS, n_particles),   needs_grad=True)
J          = ti.field(float,             shape=(MAX_STEPS, n_particles),        needs_grad=True)
grid_v_in  = ti.Vector.field(dim, float, shape=(MAX_STEPS, n_grid, n_grid),     needs_grad=True)
grid_m     = ti.field(float,             shape=(MAX_STEPS, n_grid, n_grid),     needs_grad=True)
grid_v_out = ti.Vector.field(dim, float, shape=(MAX_STEPS, n_grid, n_grid),     needs_grad=True)

x_init     = ti.Vector.field(dim, float, shape=n_particles)          # fixed blob; no grad
v0         = ti.Vector.field(dim, float, shape=(),  needs_grad=True)  # optimised parameter
target_pos = ti.Vector.field(dim, float, shape=())
x_avg      = ti.Vector.field(dim, float, shape=(),  needs_grad=True)
loss       = ti.field(float,             shape=(),  needs_grad=True)

# Mass epsilon for the stabilised grid_op (set before kernels compile)
# Non-zero value replaces the `if m > 0` branch with a safe division.
mass_eps = 0.0   # 0 = use original (branchy) grid_op; set to e.g. 1e-4 to stabilise

# ---------------------------------------------------------------------------
# Diagnostic fields  (not part of the autodiff tape)
# ---------------------------------------------------------------------------
step_grid_mass_min = ti.field(float, shape=MAX_STEPS)  # filled from forward pass


# ---------------------------------------------------------------------------
# Simulation kernels (identical logic to diffmpm.py)
# ---------------------------------------------------------------------------

@ti.kernel
def seed_blob():
    for p in range(n_particles):
        x_init[p] = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.4]


@ti.kernel
def init_state():
    for p in range(n_particles):
        x[0, p] = x_init[p]
        v[0, p] = v0[None]
        J[0, p] = 1.0
        C[0, p] = ti.Matrix.zero(float, dim, dim)


@ti.kernel
def clear_grid(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v_in[f, i, j]  = ti.Vector.zero(float, dim)
        grid_m[f, i, j]     = 0.0
        grid_v_out[f, i, j] = ti.Vector.zero(float, dim)


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        Xp     = x[f, p] * inv_dx
        base   = int(Xp - 0.5)
        fx     = Xp - base
        w      = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (J[f, p] - 1.0) * inv_dx * inv_dx
        affine = ti.Matrix([[stress, 0.0], [0.0, stress]]) + p_mass * C[f, p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos   = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v_in[f, base[0] + i, base[1] + j] += weight * (p_mass * v[f, p] + affine @ dpos)
            grid_m[f, base[0] + i, base[1] + j]    += weight * p_mass


@ti.kernel
def grid_op(f: ti.i32):
    """Original grid op with hard branch on m > 0 (baseline)."""
    for i, j in ti.ndrange(n_grid, n_grid):
        m   = grid_m[f, i, j]
        vel = ti.Vector.zero(float, dim)
        if m > 0:
            vel = grid_v_in[f, i, j] / m
        vel[1] -= dt * gravity
        if i < bound and vel[0] < 0:         vel[0] = 0.0
        if i > n_grid - bound and vel[0] > 0: vel[0] = 0.0
        if j < bound and vel[1] < 0:         vel[1] = 0.0
        if j > n_grid - bound and vel[1] > 0: vel[1] = 0.0
        grid_v_out[f, i, j] = vel


@ti.kernel
def grid_op_stable(f: ti.i32, eps: float):
    """Stabilised grid op: divide by max(m, eps) so the backward never amplifies by 1/0.

    Eliminates the near-zero mass singularity without altering physics for m >> eps.
    Also removes the branch so contact gradients are continuous (though still clamped).
    """
    for i, j in ti.ndrange(n_grid, n_grid):
        m   = grid_m[f, i, j]
        vel = grid_v_in[f, i, j] / ti.max(m, eps)
        vel[1] -= dt * gravity
        if i < bound and vel[0] < 0:         vel[0] = 0.0
        if i > n_grid - bound and vel[0] > 0: vel[0] = 0.0
        if j < bound and vel[1] < 0:         vel[1] = 0.0
        if j > n_grid - bound and vel[1] > 0: vel[1] = 0.0
        grid_v_out[f, i, j] = vel


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        Xp    = x[f, p] * inv_dx
        base  = int(Xp - 0.5)
        fx    = Xp - base
        w     = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos   = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v    = grid_v_out[f, base[0] + i, base[1] + j]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * new_v
        J[f + 1, p] = J[f, p] * (1 + dt * new_C.trace())
        C[f + 1, p] = new_C


@ti.kernel
def clear_x_avg():
    x_avg[None] = ti.Vector.zero(float, dim)


@ti.kernel
def compute_x_avg(f: ti.i32):
    for p in range(n_particles):
        x_avg[None] += (1.0 / n_particles) * x[f, p]


@ti.kernel
def compute_loss():
    d = x_avg[None] - target_pos[None]
    loss[None] = d[0] ** 2 + d[1] ** 2


def forward(n_steps: int, stabilize: bool = False, eps: float = 1e-4):
    init_state()
    for f in range(n_steps - 1):
        clear_grid(f)
        p2g(f)
        if stabilize:
            grid_op_stable(f, eps)
        else:
            grid_op(f)
        g2p(f)
    clear_x_avg()
    compute_x_avg(n_steps - 1)
    compute_loss()


# ---------------------------------------------------------------------------
# Diagnostic kernels  (no autodiff involvement)
# ---------------------------------------------------------------------------
# NOTE: Taichi's reverse-mode autodiff does NOT persist intermediate gradients
# (v.grad[t], x.grad[t]) after the backward pass — only leaf inputs (v0.grad)
# are reliably populated.  Per-step diagnostics therefore use FORWARD-pass
# state (grid_m after p2g) to infer where near-zero-mass events occur.

@ti.kernel
def _init_grid_mass_min(n_steps: ti.i32):
    for t in range(n_steps):
        step_grid_mass_min[t] = 1e10


@ti.kernel
def _compute_grid_mass_min(n_steps: ti.i32):
    for t in range(n_steps):
        for i, j in ti.ndrange(n_grid, n_grid):
            m = grid_m[t, i, j]
            if m > 1e-12:
                ti.atomic_min(step_grid_mass_min[t], m)


def read_step_diagnostics(n_steps: int) -> dict:
    """Grid-mass statistics from the FORWARD pass state.

    Called after forward() (NOT after backward).  Reports the minimum
    non-zero grid-node mass at each simulation step — this is the quantity
    that amplifies gradients during the backward as 1/m.
    """
    _init_grid_mass_min(n_steps)
    _compute_grid_mass_min(n_steps)
    massmins = step_grid_mass_min.to_numpy()[:n_steps]
    massmins = [float(m) if m < 1e9 else 0.0 for m in massmins]
    return {"grid_mass_min": massmins}


# ---------------------------------------------------------------------------
# Optimisation loop
# ---------------------------------------------------------------------------

def run_optimize(n_iters, lr, n_steps, clip_norm=None, skip_nan=False,
                 stabilize=False, mass_eps=1e-4):
    """Adam optimisation.

    Returns
    -------
    losses, grad_norms : list[float]
    step_diag : dict  {iter_str: grid_mass_min list (from forward pass)}
    nan_at_iter : int or None
    """
    m_adam = np.zeros(2)
    s_adam = np.zeros(2)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8

    losses, grad_norms = [], []
    step_diag = {}
    nan_at_iter = None

    def _fwd():
        forward(n_steps, stabilize=stabilize, eps=mass_eps)

    for it in range(n_iters):
        with ti.ad.Tape(loss):
            _fwd()

        L  = float(loss[None])
        gx = float(v0.grad[None][0])
        gy = float(v0.grad[None][1])
        g  = np.array([gx, gy])
        gn = float(np.linalg.norm(g))

        losses.append(L)
        grad_norms.append(gn)

        # Read forward-pass state diagnostics (grid mass min per step)
        store = (it == 0 or it % 20 == 0)
        if store or (not (np.isfinite(L) and np.all(np.isfinite(g)))):
            step_diag[str(it)] = read_step_diagnostics(n_steps)

        is_nan = not (np.isfinite(L) and np.all(np.isfinite(g)))
        if is_nan:
            if nan_at_iter is None:
                nan_at_iter = it
            print(f"[iter {it:3d}] NaN  loss={L:.3e}  |g|={gn:.3e}"
                  + ("  (skip)" if skip_nan else "  (stop)"))
            if not skip_nan:
                break
            continue

        # Optional gradient clipping on v0
        if clip_norm is not None and gn > clip_norm:
            g = g * (clip_norm / gn)

        m_adam = b1 * m_adam + (1 - b1) * g
        s_adam = b2 * s_adam + (1 - b2) * g * g
        mh = m_adam / (1 - b1 ** (it + 1))
        sh = s_adam / (1 - b2 ** (it + 1))
        cur = np.array([float(v0[None][0]), float(v0[None][1])])
        cur = cur - lr * mh / (np.sqrt(sh) + eps_adam)
        v0[None] = [float(cur[0]), float(cur[1])]

        print(f"[iter {it:3d}] loss={L:.4e}  v0=({cur[0]:+.4f},{cur[1]:+.4f})  |g|={gn:.4e}")

    return losses, grad_norms, step_diag, nan_at_iter


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def git_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True,
        ).strip()
    except Exception:
        return "local"


def make_out_dir(repo, branch, run_id):
    rel = f"runs/{branch}/{run_id}"
    out = os.path.join(repo, *rel.split("/"))
    os.makedirs(out, exist_ok=True)
    return out, rel


def save_run(repo, branch, run_id, title, summary, params, metrics_data, media=None):
    out, rel = make_out_dir(repo, branch, run_id)
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics_data, fh)
    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "branch": branch,
        "title": title,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "completed",
        "summary": summary,
        "metrics": {**metrics_data.get("summary_scalars", {}), "series": f"{rel}/metrics.json"},
        "media": media or {},
        "training_refs": ["failure-modes"],
        "params": params,
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  -> saved {rel}")
    return out, rel


def render_video(path, target_np, n_steps, stride=6, size=800, fps=30, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    xs  = x.to_numpy()
    bg  = (0.043, 0.059, 0.078)
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=bg)
    ax  = fig.add_axes([0, 0, 1, 1])
    frames = []
    for f in range(0, n_steps, stride):
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_facecolor(bg); ax.axis("off")
        ax.scatter(xs[f, :, 0], xs[f, :, 1], s=5, c="#7ee587", edgecolors="none", alpha=0.85)
        ax.plot(target_np[0], target_np[1], marker="+", ms=16, mew=2.5, c="#ff6e6e")
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(size, size, 4)
        frames.append(rgba[..., :3].copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def exp_instrument(args, repo, branch):
    """Per-step forward-state diagnostics and v0.grad norm curve."""
    stabilize = getattr(args, "stabilize", False)
    mass_eps  = getattr(args, "mass_eps",  1e-4)
    print(f"\n=== instrument  steps={args.steps}  iters={args.iters}"
          f"  prec={'f64' if _use_f64 else 'f32'}"
          f"  stabilize={stabilize} ===")
    n_steps = args.steps
    v0[None]         = [0.0, 0.0]
    target_pos[None] = [0.7, 0.35]

    losses, grad_norms, step_diag, nan_at_iter = run_optimize(
        args.iters, args.lr, n_steps,
        stabilize=stabilize, mass_eps=mass_eps
    )

    prec   = "f64" if _use_f64 else "f32"
    run_id = f"pathologies-instrument-{prec}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    save_run(
        repo, branch, run_id,
        title=f"Gradient instrumentation — {n_steps} steps, {prec}",
        summary=(
            f"Per-step backward gradient norms through {n_steps}-step MLS-MPM ({prec}). "
            f"NaN first at iter {nan_at_iter}."
        ),
        params={
            "experiment": "instrument", "n_steps": n_steps,
            "iters": args.iters, "lr": args.lr,
            "precision": prec, "target": [0.7, 0.35],
        },
        metrics_data={
            "summary_scalars": {
                "final_loss": losses[-1] if losses else None,
                "nan_at_iter": nan_at_iter,
                "iterations": len(losses),
            },
            "loss": losses,
            "grad_norm_v0": grad_norms,
            "step_diagnostics": step_diag,
        },
    )
    return losses, grad_norms, step_diag, nan_at_iter


def exp_horizon(args, repo, branch):
    """Horizon sweep — NaN iteration vs. rollout length."""
    horizons = [128, 256, 512]
    print(f"\n=== horizon sweep {horizons}  iters={args.iters} ===")
    results = []

    for n_steps in horizons:
        print(f"\n--- n_steps={n_steps} ---")
        v0[None]         = [0.0, 0.0]
        target_pos[None] = [0.7, 0.35]

        losses, grad_norms, _, nan_at_iter = run_optimize(
            args.iters, args.lr, n_steps
        )
        results.append({
            "n_steps":         n_steps,
            "nan_at_iter":     nan_at_iter,
            "final_loss":      losses[-1] if losses else None,
            "final_grad_norm": grad_norms[-1] if grad_norms else None,
            "losses":          losses,
            "grad_norms":      grad_norms,
        })
        print(f"  n_steps={n_steps}: NaN@iter {nan_at_iter}, final_loss={losses[-1]:.4e}")

    prec   = "f64" if _use_f64 else "f32"
    run_id = f"pathologies-horizon-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    save_run(
        repo, branch, run_id,
        title="Horizon sweep — NaN iteration vs. rollout length",
        summary=(
            "Swept rollout lengths [128, 256, 512] with fixed target [0.7, 0.35]. "
            + "  ".join(f"n={r['n_steps']}→NaN@{r['nan_at_iter']}" for r in results)
        ),
        params={
            "experiment": "horizon", "horizons": horizons,
            "iters": args.iters, "lr": args.lr, "precision": prec,
            "target": [0.7, 0.35],
        },
        metrics_data={
            "summary_scalars": {
                "nan_iters": {str(r["n_steps"]): r["nan_at_iter"] for r in results},
            },
            "horizon_sweep": results,
        },
    )
    return results


def exp_clip(args, repo, branch):
    """Mass-stabilisation fix — shows clean optimisation for full iter budget."""
    stabilize = getattr(args, "stabilize", False)
    mass_eps  = getattr(args, "mass_eps",  1e-4)
    print(f"\n=== clip  steps={args.steps}  clip_norm={args.clip_norm}"
          f"  stabilize={stabilize}  iters={args.iters} ===")
    n_steps = args.steps
    v0[None]         = [0.0, 0.0]
    target_pos[None] = [0.7, 0.35]

    losses, grad_norms, step_diag, nan_at_iter = run_optimize(
        args.iters, args.lr, n_steps,
        clip_norm=args.clip_norm, skip_nan=True,
        stabilize=stabilize, mass_eps=mass_eps
    )

    # Final forward to populate x for video rendering
    with ti.ad.Tape(loss):
        forward(n_steps, stabilize=stabilize, eps=mass_eps)

    prec   = "f64" if _use_f64 else "f32"
    tag    = ("stabilized" if stabilize else "clip-only")
    run_id = f"pathologies-clip-{tag}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    target_np = np.array([0.7, 0.35])

    media = {}
    try:
        vid_path = os.path.join(repo, *f"runs/{branch}/{run_id}".split("/"), "video.mp4")
        os.makedirs(os.path.dirname(vid_path), exist_ok=True)
        render_video(vid_path, target_np, n_steps)
        media["video"] = f"runs/{branch}/{run_id}/video.mp4"
        print("  video rendered")
    except Exception as e:
        print(f"  video skipped: {e}")

    final_loss_str = f"{losses[-1]:.4e}" if losses else "N/A"
    save_run(
        repo, branch, run_id,
        title=(
            f"Mass-stabilised grid_op — clean optimisation ({n_steps} steps, {prec})"
            if stabilize else
            f"Clip-only (no stabilise) — NaN persists ({n_steps} steps, {prec})"
        ),
        summary=(
            f"{'Mass-stabilised (eps=' + str(mass_eps) + ') + ' if stabilize else ''}"
            f"clip(norm={args.clip_norm}) + NaN-skip on {n_steps}-step rollout ({prec}). "
            f"First NaN: iter {nan_at_iter}.  Final loss: {final_loss_str} after {len(losses)} iters."
        ),
        params={
            "experiment": "clip", "n_steps": n_steps,
            "iters": args.iters, "lr": args.lr,
            "clip_norm": args.clip_norm, "skip_nan": True,
            "stabilize": stabilize, "mass_eps": mass_eps,
            "precision": prec, "target": [0.7, 0.35],
        },
        metrics_data={
            "summary_scalars": {
                "final_loss": losses[-1] if losses else None,
                "nan_at_iter": nan_at_iter,
                "iterations": len(losses),
            },
            "loss": losses,
            "grad_norm_v0": grad_norms,
            "step_diagnostics": step_diag,
        },
        media=media,
    )
    return losses, grad_norms, nan_at_iter


def exp_isolate(args, repo, branch):
    """Center vs. wall target — separates contact from long-rollout amplification."""
    configs = [
        {"name": "center", "target": [0.5, 0.5]},
        {"name": "wall",   "target": [0.08, 0.35]},
    ]
    print(f"\n=== isolate  steps={args.steps}  iters={args.iters} ===")
    results = []

    for cfg in configs:
        print(f"\n--- target={cfg['name']} {cfg['target']} ---")
        v0[None]         = [0.0, 0.0]
        target_pos[None] = cfg["target"]

        losses, grad_norms, _, nan_at_iter = run_optimize(
            args.iters, args.lr, args.steps
        )
        results.append({
            "config":      cfg["name"],
            "target":      cfg["target"],
            "nan_at_iter": nan_at_iter,
            "losses":      losses,
            "grad_norms":  grad_norms,
        })
        print(f"  {cfg['name']}: NaN@iter {nan_at_iter}")

    prec   = "f64" if _use_f64 else "f32"
    run_id = f"pathologies-isolate-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    save_run(
        repo, branch, run_id,
        title="Contact isolation — center vs. wall target",
        summary=(
            f"Center target (particles stay in field) NaN@{results[0]['nan_at_iter']}, "
            f"wall target (particles driven to boundary) NaN@{results[1]['nan_at_iter']}. "
            f"Separates contact pathology from long-rollout amplification."
        ),
        params={
            "experiment": "isolate", "n_steps": args.steps,
            "iters": args.iters, "lr": args.lr, "precision": prec,
        },
        metrics_data={
            "summary_scalars": {
                "nan_at_iter_center": results[0]["nan_at_iter"],
                "nan_at_iter_wall":   results[1]["nan_at_iter"],
            },
            "isolation": results,
        },
    )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DiffMPM gradient pathology experiments")
    ap.add_argument("--experiment", choices=["instrument", "horizon", "clip", "isolate", "all"],
                    default="instrument")
    ap.add_argument("--steps",     type=int,   default=512,  help="rollout length")
    ap.add_argument("--iters",     type=int,   default=80,   help="optimisation iterations")
    ap.add_argument("--lr",        type=float, default=0.1)
    ap.add_argument("--clip-norm", type=float, default=1.0,  dest="clip_norm")
    ap.add_argument("--stabilize", action="store_true",
                    help="replace if-m>0 branch with max(m,mass_eps) to prevent near-zero-mass NaN")
    ap.add_argument("--mass-eps",  type=float, default=1e-4, dest="mass_eps",
                    help="epsilon for stabilised grid_op (default 1e-4)")
    ap.add_argument("--f64",       action="store_true",      help="use float64 (parsed before ti.init)")
    args = ap.parse_args()

    repo   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    branch = git_branch()
    prec   = "f64" if _use_f64 else "f32"

    print(f"branch={branch}  experiment={args.experiment}  precision={prec}")
    print(f"steps={args.steps}  iters={args.iters}  lr={args.lr}")

    seed_blob()   # fixed initial particle positions; all experiments share this

    if args.experiment == "instrument":
        exp_instrument(args, repo, branch)
    elif args.experiment == "horizon":
        exp_horizon(args, repo, branch)
    elif args.experiment == "clip":
        exp_clip(args, repo, branch)
    elif args.experiment == "isolate":
        exp_isolate(args, repo, branch)
    elif args.experiment == "all":
        # Run baseline instrument first, then horizon and isolate, then stabilized clip
        exp_instrument(args, repo, branch)
        exp_horizon(args, repo, branch)
        exp_isolate(args, repo, branch)
        # Clip with stabilisation enabled
        args.stabilize = True
        exp_clip(args, repo, branch)


if __name__ == "__main__":
    main()
