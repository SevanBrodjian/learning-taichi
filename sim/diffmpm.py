"""Differentiable MLS-MPM — Phase 1 core.

Forward seed: ``sim/mpm88.py``. Here the rollout is **time-indexed** and **differentiable**: we
optimize a single shared initial velocity ``v0`` so the blob's center of mass reaches a target, by
backpropagating through the whole simulation with Taichi's autodiff tape.

Design notes: ``agents/claude/elegant-bassi-cb7174/diffmpm_design.md``.
Output contract: ``runs/README.md`` (writes a run folder + manifest; refresh the index separately).

Usage:
    python sim/diffmpm.py --iters 80 --lr 0.1            # full run + video
    python sim/diffmpm.py --iters 5 --no-video          # quick gradient/loss smoke test
"""
import argparse
import datetime
import json
import os
import subprocess

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --- parameters ---
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
max_steps = 512

# --- time-indexed differentiable fields ---
_scalar = lambda: ti.field(float, shape=(max_steps, n_particles), needs_grad=True)
_vec = lambda: ti.Vector.field(dim, float, shape=(max_steps, n_particles), needs_grad=True)
_mat = lambda: ti.Matrix.field(dim, dim, float, shape=(max_steps, n_particles), needs_grad=True)

x, v, C, J = _vec(), _vec(), _mat(), _scalar()
grid_v_in = ti.Vector.field(dim, float, shape=(max_steps, n_grid, n_grid), needs_grad=True)
grid_m = ti.field(float, shape=(max_steps, n_grid, n_grid), needs_grad=True)
grid_v_out = ti.Vector.field(dim, float, shape=(max_steps, n_grid, n_grid), needs_grad=True)

x_init = ti.Vector.field(dim, float, shape=n_particles)          # fixed blob (no grad)
v0 = ti.Vector.field(dim, float, shape=(), needs_grad=True)       # the optimized parameter
target = ti.Vector.field(dim, float, shape=())
x_avg = ti.Vector.field(dim, float, shape=(), needs_grad=True)
loss = ti.field(float, shape=(), needs_grad=True)


@ti.kernel
def seed_blob():
    for p in range(n_particles):
        x_init[p] = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.4]  # ~[0.2,0.5] x [0.4,0.7]


@ti.kernel
def init_state():
    for p in range(n_particles):
        x[0, p] = x_init[p]
        v[0, p] = v0[None]              # v0 enters the autodiff graph here
        J[0, p] = 1.0
        C[0, p] = ti.Matrix.zero(float, dim, dim)


@ti.kernel
def clear_grid(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v_in[f, i, j] = ti.Vector.zero(float, dim)
        grid_m[f, i, j] = 0.0
        grid_v_out[f, i, j] = ti.Vector.zero(float, dim)


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        Xp = x[f, p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (J[f, p] - 1.0) * inv_dx * inv_dx
        affine = ti.Matrix([[stress, 0.0], [0.0, stress]]) + p_mass * C[f, p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v_in[f, base[0] + i, base[1] + j] += weight * (p_mass * v[f, p] + affine @ dpos)
            grid_m[f, base[0] + i, base[1] + j] += weight * p_mass


@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[f, i, j]
        vel = ti.Vector.zero(float, dim)
        if m > 0:
            vel = grid_v_in[f, i, j] / m
        vel[1] -= dt * gravity
        if i < bound and vel[0] < 0:
            vel[0] = 0.0
        if i > n_grid - bound and vel[0] > 0:
            vel[0] = 0.0
        if j < bound and vel[1] < 0:
            vel[1] = 0.0
        if j > n_grid - bound and vel[1] > 0:
            vel[1] = 0.0
        grid_v_out[f, i, j] = vel


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        Xp = x[f, p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v_out[f, base[0] + i, base[1] + j]
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
    d = x_avg[None] - target[None]
    loss[None] = d[0] ** 2 + d[1] ** 2


def forward():
    init_state()
    for f in range(max_steps - 1):
        clear_grid(f)
        p2g(f)
        grid_op(f)
        g2p(f)
    clear_x_avg()
    compute_x_avg(max_steps - 1)
    compute_loss()


def optimize(n_iter, lr):
    """Adam on the 2-D v0 (robust to gradient scaling); returns the loss history."""
    m = np.zeros(2)
    s = np.zeros(2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    for it in range(n_iter):
        with ti.ad.Tape(loss):
            forward()
        L = float(loss[None])
        losses.append(L)
        g = np.array([v0.grad[None][0], v0.grad[None][1]])
        if not np.all(np.isfinite(g)) or not np.isfinite(L):
            print(f"[iter {it}] non-finite (loss={L}, grad={g}) — stopping")
            break
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1))
        sh = s / (1 - b2 ** (it + 1))
        cur = np.array([v0[None][0], v0[None][1]])
        cur = cur - lr * mh / (np.sqrt(sh) + eps)
        v0[None] = [float(cur[0]), float(cur[1])]
        print(f"[iter {it:3d}] loss={L:.6f}  v0=({cur[0]:+.3f},{cur[1]:+.3f})  |grad|={np.linalg.norm(g):.4f}")
    return losses


def render_video(path, target_np, stride=6, size=800, fps=30, dpi=100):
    """High-quality headless render: anti-aliased particle splats via matplotlib (Agg), no ti.GUI.

    Scatter gives smooth round particles at high resolution, which reads far better than the old
    single-pixel blocks. 800px is a multiple of 16 so the encoder needs no padding.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    xs = x.to_numpy()  # (max_steps, n_particles, 2)
    bg = (0.043, 0.059, 0.078)
    # facecolor on the figure itself: with axis("off") the axes patch is not drawn, so the figure
    # background is what shows. Without this the canvas renders white.
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    frames = []
    for f in range(0, max_steps, stride):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(bg)
        ax.axis("off")
        ax.scatter(xs[f, :, 0], xs[f, :, 1], s=5, c="#7ee587", edgecolors="none", alpha=0.85)
        ax.plot(target_np[0], target_np[1], marker="+", ms=16, mew=2.5, c="#ff6e6e")
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(size, size, 4)
        frames.append(rgba[..., :3].copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def git_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True,
        ).strip()
    except Exception:
        return "local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--target", type=float, nargs=2, default=[0.7, 0.35])
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    branch = git_branch()
    run_id = "diffmpm-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rel_dir = f"runs/{branch}/{run_id}"
    out_dir = os.path.join(repo, *rel_dir.split("/"))
    os.makedirs(out_dir, exist_ok=True)

    seed_blob()
    v0[None] = [0.0, 0.0]
    target[None] = [args.target[0], args.target[1]]

    t0 = datetime.datetime.now(datetime.timezone.utc)
    losses = optimize(args.iters, args.lr)
    forward()  # final rollout with the optimized v0 (populates x for rendering)
    final_loss = float(loss[None])

    media = {}
    if not args.no_video:
        try:
            render_video(os.path.join(out_dir, "video.mp4"), np.array(args.target))
            media["video"] = f"{rel_dir}/video.mp4"
        except Exception as e:  # noqa: BLE001
            print(f"video render skipped: {e}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"loss": losses}, fh)

    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "branch": branch,
        "title": "DiffMPM: optimize initial velocity to reach a target",
        "created": t0.isoformat(),
        "status": "completed" if np.isfinite(final_loss) else "failed",
        "summary": (
            f"Backprop through {max_steps} MLS-MPM steps to optimize a shared initial velocity so the "
            f"blob's center of mass reaches {tuple(args.target)}. Adam, {len(losses)} iters."
        ),
        "metrics": {
            "final_loss": final_loss,
            "iterations": len(losses),
            "series": f"{rel_dir}/metrics.json",
        },
        "media": media,
        # Run pages transclude these textbook sections (by id in reports/training/index.json)
        # instead of restating the teaching. See harness/server /api/training.
        "training_refs": ["mls-mpm-forward", "differentiating-the-rollout", "failure-modes"],
        "params": {
            "optimizer": "adam", "lr": args.lr, "n_grid": n_grid, "n_particles": n_particles,
            "steps": max_steps, "dt": dt, "E": E, "target": list(args.target),
            "v0_final": [float(v0[None][0]), float(v0[None][1])],
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nfinal loss {final_loss:.6f}  ->  {rel_dir}")
    print(f"v0* = ({v0[None][0]:+.3f}, {v0[None][1]:+.3f})")


if __name__ == "__main__":
    main()
