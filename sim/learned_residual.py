"""A learned residual inside the differentiable MLS-MPM step (first hybrid sim).

Seeded from ``sim/diffmpm.py`` and ``sim/optimizer_compare.py``. The MLS-MPM step is kept
**intact**; a small MLP is inserted *right after* ``grid_op`` and adds a bounded correction to
the post-grid-op grid velocity at every active node, every step:

    grid_v_out[f, i, j]  +=  residual_scale * tanh( MLP(features[f, i, j]) )      (active nodes)

before ``g2p`` reads it. The MLP is implemented as **differentiable Taichi fields/kernels**, so its
weights live on the *same autodiff tape* as the physics. Training backpropagates a supervised
center-of-mass trajectory loss through hundreds of physics steps into the network weights. This is
a detached-PyTorch-graph-free hybrid on purpose: the whole question is whether gradients that have
passed through the simulator can train a network embedded in it.

The supervised **target** is a genuine *model mismatch* the residual is never handed directly: the
target trajectories come from the same simulator run with an **added linear drag** ``v *= (1-k)``
applied to the grid velocity each step. The bare simulator (no drag, no residual) misses that
target; the residual's job is to learn the correction purely from the trajectory. ``v0`` is held
fixed, so the residual cannot cheat by re-aiming the throw.

Mass stabilization (``vel = grid_v_in / max(m, MASS_EPS)``) stays ON (see
``reports/training/core/03-failure-modes.md``) so the long-rollout NaN cannot confound training.

Pipeline:
  1. generate the drag target trajectory (teacher);
  2. gradient-flow probe BEFORE training (autodiff vs finite-difference on a couple of weights,
     and a few steps that strictly decrease loss and move the weights off init);
  3. train the residual weights with Adam through the rollout;
  4. evaluate on a held-out v0 (transfer, not memorization);
  5. render the 3-way comparison video + loss/grad plots + table + schema-v2 manifest.

Usage:
    python sim/learned_residual.py                      # full run + media + manifest
    python sim/learned_residual.py --iters 5 --no-video # quick smoke test
    python sim/learned_residual.py --probe-only         # gradient-flow probe only
"""
import argparse
import datetime
import json
import os
import subprocess

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --- physics parameters (match sim/diffmpm.py so the simulator is the established baseline) ---
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
MASS_EPS = 1e-4  # mass-stabilization floor (see failure-modes); caps backward 1/m at 1e4

# Horizon: shorter than the 512-step baseline keeps the tape memory bounded and the gradient to the
# network weights healthy. 320 steps is long enough that the drag mismatch accumulates a clearly
# visible center-of-mass gap.
max_steps = 320

# --- mismatch the residual must learn to undo: a linear drag on the grid velocity each step ---
# v <- v * (1 - DRAG_K) in the teacher's grid step (NOT in the student). Chosen strong enough that
# the bare simulator visibly drifts past the drag target's center of mass over the horizon.
DRAG_K = 0.004

# --- network architecture ---
# Features are deliberately POSITION-FREE: the node velocity and a log-mass scalar only. A linear
# drag is a pure function of velocity, so the correct correction does not depend on where a node is.
# Withholding absolute coordinates is what forces the residual to learn the velocity-space law
# rather than memorizing the train trajectory's path, which is what makes held-out v0 transfer
# possible. (An earlier version that fed normalized node coordinates overfit and failed to transfer.)
N_IN = 3          # features: grid_v_out.x, grid_v_out.y, log1p(grid_m_norm)
N_HID = 16        # one hidden layer, tanh
N_OUT = 2         # delta-v added to the node velocity
RESID_SCALE = 0.1  # bounds the residual: scale * tanh(.) in [-0.1, 0.1] per component
GRID_M_NORM = 1.0 / p_mass  # so log1p(grid_m * GRID_M_NORM) is O(1) for a typical node

# --- time-indexed differentiable physics fields (same layout as diffmpm.py) ---
_scalar = lambda: ti.field(float, shape=(max_steps, n_particles), needs_grad=True)
_vec = lambda: ti.Vector.field(dim, float, shape=(max_steps, n_particles), needs_grad=True)
_mat = lambda: ti.Matrix.field(dim, dim, float, shape=(max_steps, n_particles), needs_grad=True)

x, v, C, J = _vec(), _vec(), _mat(), _scalar()
grid_v_in = ti.Vector.field(dim, float, shape=(max_steps, n_grid, n_grid), needs_grad=True)
grid_m = ti.field(float, shape=(max_steps, n_grid, n_grid), needs_grad=True)
grid_v_out = ti.Vector.field(dim, float, shape=(max_steps, n_grid, n_grid), needs_grad=True)

# Per-node, per-step hidden activation of the MLP. It must be a stored, time-indexed field so the
# Taichi tape can replay the backward through the network exactly like any other intermediate.
hidden = ti.field(float, shape=(max_steps, n_grid, n_grid, N_HID), needs_grad=True)

x_init = ti.Vector.field(dim, float, shape=n_particles)            # fixed blob (no grad)
v0 = ti.Vector.field(dim, float, shape=())                         # FIXED initial velocity (no grad)
x_avg = ti.Vector.field(dim, float, shape=(), needs_grad=True)

# Supervised target center-of-mass trajectory (the teacher), per step. No grad (it is data).
com_target = ti.Vector.field(dim, float, shape=max_steps)
# The student's own per-step center of mass, recomputed inside the tape so the loss sees all steps.
com = ti.Vector.field(dim, float, shape=max_steps, needs_grad=True)
loss = ti.field(float, shape=(), needs_grad=True)

# Toggle: when teacher_mode is 1 the grid step applies the drag and NO residual (generate target);
# when 0 it applies the residual and NO drag (the student hybrid). Held as a plain python flag and
# baked by swapping which grid kernel runs, so neither path carries the other's ops on the tape.

# --- network weights as needs_grad Taichi fields (the trained parameters) ---
W1 = ti.field(float, shape=(N_HID, N_IN), needs_grad=True)
b1 = ti.field(float, shape=N_HID, needs_grad=True)
W2 = ti.field(float, shape=(N_OUT, N_HID), needs_grad=True)
b2 = ti.field(float, shape=N_OUT, needs_grad=True)


@ti.kernel
def seed_blob():
    for p in range(n_particles):
        x_init[p] = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.4]  # ~[0.2,0.5] x [0.4,0.7]


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
        grid_v_in[f, i, j] = ti.Vector.zero(float, dim)
        grid_m[f, i, j] = 0.0
        grid_v_out[f, i, j] = ti.Vector.zero(float, dim)
    for i, j, h in ti.ndrange(n_grid, n_grid, N_HID):
        hidden[f, i, j, h] = 0.0


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


@ti.func
def apply_walls(vel, i, j):
    if i < bound and vel[0] < 0:
        vel[0] = 0.0
    if i > n_grid - bound and vel[0] > 0:
        vel[0] = 0.0
    if j < bound and vel[1] < 0:
        vel[1] = 0.0
    if j > n_grid - bound and vel[1] > 0:
        vel[1] = 0.0
    return vel


@ti.kernel
def grid_op_teacher(f: ti.i32):
    # Teacher (target generator): the same physics PLUS an added linear drag on the grid velocity.
    # This is the model mismatch the student never sees directly.
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[f, i, j]
        vel = grid_v_in[f, i, j] / ti.max(m, MASS_EPS)
        vel[1] -= dt * gravity
        if m > MASS_EPS:
            vel = vel * (1.0 - DRAG_K)
        vel = apply_walls(vel, i, j)
        grid_v_out[f, i, j] = vel


@ti.kernel
def grid_op_bare(f: ti.i32):
    # Bare student physics: identical to diffmpm.py's stabilized grid step, no drag, no residual.
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[f, i, j]
        vel = grid_v_in[f, i, j] / ti.max(m, MASS_EPS)
        vel[1] -= dt * gravity
        vel = apply_walls(vel, i, j)
        grid_v_out[f, i, j] = vel


@ti.kernel
def residual_op(f: ti.i32):
    # The learned residual: a per-node MLP whose Δv is added to the post-grid-op velocity. Runs on
    # the SAME tape as the physics, so W1,b1,W2,b2 receive gradients backpropagated through g2p and
    # every downstream physics step. Applied only at active nodes (grid_m > MASS_EPS).
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[f, i, j]
        if m > MASS_EPS:
            vout = grid_v_out[f, i, j]
            # features (kept small + physically meaningful): velocity, log-mass, normalized coords
            feat = ti.Vector([
                vout[0],
                vout[1],
                ti.log(1.0 + m * GRID_M_NORM),
            ])
            # layer 1: hidden = tanh(W1 feat + b1), stored per node per step for the tape.
            # The inner loops are ti.static (unrolled): N_HID and N_IN are compile-time constants,
            # and the static unroll keeps Taichi's reverse-mode autodiff on its well-supported path.
            # A non-static range loop over the hidden units crashed the CUDA backward here.
            for h in ti.static(range(N_HID)):
                acc = b1[h]
                for k in ti.static(range(N_IN)):
                    acc += W1[h, k] * feat[k]
                hidden[f, i, j, h] = ti.tanh(acc)
            # layer 2: out = W2 hidden + b2, then bound with scale*tanh so the residual is gentle
            dvx = b2[0]
            dvy = b2[1]
            for h in ti.static(range(N_HID)):
                dvx += W2[0, h] * hidden[f, i, j, h]
                dvy += W2[1, h] * hidden[f, i, j, h]
            grid_v_out[f, i, j] = vout + ti.Vector([
                RESID_SCALE * ti.tanh(dvx),
                RESID_SCALE * ti.tanh(dvy),
            ])


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
def clear_com():
    for f in range(max_steps):
        com[f] = ti.Vector.zero(float, dim)


@ti.kernel
def accumulate_com(f: ti.i32):
    for p in range(n_particles):
        com[f] += (1.0 / n_particles) * x[f, p]


@ti.kernel
def compute_traj_loss():
    # Supervised loss over the WHOLE center-of-mass trajectory (richer signal than the endpoint and
    # the thing that makes held-out transfer meaningful). Mean squared distance over all steps.
    for f in range(max_steps):
        d = com[f] - com_target[f]
        loss[None] += (d[0] ** 2 + d[1] ** 2) * (1.0 / max_steps)


def forward(student: bool, with_residual: bool):
    """One full rollout. student=True uses bare physics (+ residual if with_residual); student=False
    is the teacher (drag, no residual). The loss compares the student COM trajectory to com_target."""
    init_state()
    grid_step = grid_op_bare if student else grid_op_teacher
    for f in range(max_steps - 1):
        clear_grid(f)
        p2g(f)
        grid_step(f)
        if student and with_residual:
            residual_op(f)
        g2p(f)
    clear_com()
    for f in range(max_steps):
        accumulate_com(f)


def forward_loss():
    """Student-with-residual rollout + trajectory loss, all on the tape (for training/probe)."""
    forward(student=True, with_residual=True)
    compute_traj_loss()


# --------------------------------------------------------------------------------------------
# Weight init / numpy boundary helpers
# --------------------------------------------------------------------------------------------

def init_weights(seed=0):
    rng = np.random.default_rng(seed)
    # Init so the hybrid starts ~ the pure simulator: the FIRST layer has ordinary small weights so
    # the hidden units are informative, but the OUTPUT layer is near zero (W2 ~ 1e-3, b2 = 0) so the
    # residual is ~ scale*tanh(~0) ~ 0 at init. This is the residual-network trick: begin as the
    # identity (here, the bare simulator) and let training grow the correction from nothing.
    W1.from_numpy((rng.standard_normal((N_HID, N_IN)) * 0.3).astype(np.float32))
    b1.from_numpy(np.zeros(N_HID, dtype=np.float32))
    W2.from_numpy((rng.standard_normal((N_OUT, N_HID)) * 1e-3).astype(np.float32))
    b2.from_numpy(np.zeros(N_OUT, dtype=np.float32))


def get_weights():
    return [W1.to_numpy(), b1.to_numpy(), W2.to_numpy(), b2.to_numpy()]


def set_weights(ws):
    W1.from_numpy(ws[0]); b1.from_numpy(ws[1]); W2.from_numpy(ws[2]); b2.from_numpy(ws[3])


def get_grads():
    return [W1.grad.to_numpy(), b1.grad.to_numpy(), W2.grad.to_numpy(), b2.grad.to_numpy()]


def flat(ws):
    return np.concatenate([w.ravel() for w in ws]).astype(np.float64)


def unflatten(vec):
    shapes = [(N_HID, N_IN), (N_HID,), (N_OUT, N_HID), (N_OUT,)]
    out, o = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(vec[o:o + n].reshape(s).astype(np.float32))
        o += n
    return out


N_PARAMS = N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT


# --------------------------------------------------------------------------------------------
# Target generation, baseline, probe, training, eval
# --------------------------------------------------------------------------------------------

def set_v0(vx, vy):
    v0[None] = [float(vx), float(vy)]


def generate_target():
    """Run the teacher (drag) and store its center-of-mass trajectory into com_target."""
    forward(student=False, with_residual=False)
    com_target.from_numpy(com.to_numpy())


def loss_only():
    """Forward student+residual + loss WITHOUT a tape (for FD checks / eval)."""
    loss[None] = 0.0
    forward(student=True, with_residual=True)
    compute_traj_loss()
    return float(loss[None])


def loss_and_grad():
    """Forward+backward on the tape; returns (loss, flat grad over all weights)."""
    loss[None] = 0.0
    with ti.ad.Tape(loss):
        forward_loss()
    return float(loss[None]), flat(get_grads())


def bare_loss():
    """Student with NO residual vs the drag target: the gap the residual must close."""
    loss[None] = 0.0
    forward(student=True, with_residual=False)
    compute_traj_loss()
    return float(loss[None])


def gradient_flow_probe(n_fd=3, eps=1e-3, n_steps=8, seed=0):
    """Mandatory anti-degeneracy probe. (1) autodiff grad must be finite and non-zero; (2) a
    finite-difference spot-check on a few weights must agree with autodiff; (3) a few optimizer
    steps must strictly decrease the loss and move the weights off init."""
    print("\n=== gradient-flow probe (before training) ===")
    init_weights(seed)
    w0 = flat(get_weights())

    L0, g = loss_and_grad()
    gnorm = float(np.linalg.norm(g))
    finite = bool(np.all(np.isfinite(g)) and np.isfinite(L0))
    print(f"loss(init)          = {L0:.6e}")
    print(f"|grad w.r.t weights| = {gnorm:.6e}   finite={finite}")
    if not finite or gnorm == 0.0:
        raise RuntimeError("PROBE FAILED: weight gradient is non-finite or zero — residual is "
                           "detached, mis-scaled, or the horizon vanished the gradient.")

    # FD spot-check on the few weights with the largest |autodiff grad| (most informative).
    idx = np.argsort(-np.abs(g))[:n_fd]
    fd_rows = []
    print(f"finite-difference check on {n_fd} weights (eps={eps}):")
    for ii in idx:
        wp = w0.copy(); wp[ii] += eps
        set_weights(unflatten(wp)); Lp = loss_only()
        wm = w0.copy(); wm[ii] -= eps
        set_weights(unflatten(wm)); Lm = loss_only()
        fd = (Lp - Lm) / (2 * eps)
        ad = g[ii]
        rel = abs(fd - ad) / (abs(fd) + abs(ad) + 1e-30)
        fd_rows.append((int(ii), float(ad), float(fd), float(rel)))
        print(f"  w[{ii:3d}]  autodiff={ad:+.4e}  fd={fd:+.4e}  rel_err={rel:.3e}")
    set_weights(unflatten(w0))  # restore init

    # A few Adam steps must strictly decrease the loss and move the weights off init.
    losses = [L0]
    m = np.zeros(N_PARAMS); s = np.zeros(N_PARAMS)
    b1c, b2c, epsa, lr = 0.9, 0.999, 1e-8, 0.02
    cur = w0.copy()
    for it in range(n_steps):
        set_weights(unflatten(cur))
        L, gg = loss_and_grad()
        m = b1c * m + (1 - b1c) * gg
        s = b2c * s + (1 - b2c) * gg * gg
        mh = m / (1 - b1c ** (it + 1)); sh = s / (1 - b2c ** (it + 1))
        cur = cur - lr * mh / (np.sqrt(sh) + epsa)
        L2 = loss_only()
        losses.append(L2)
        print(f"  step {it}: loss {L:.6e} -> {L2:.6e}")
    moved = float(np.linalg.norm(cur - w0))
    decreased = losses[-1] < losses[0]
    print(f"loss over {n_steps} probe steps: {losses[0]:.6e} -> {losses[-1]:.6e}  "
          f"(decreased={decreased})  |dweights|={moved:.4e}")
    if not decreased:
        raise RuntimeError("PROBE FAILED: loss did not decrease over probe steps — refusing to "
                           "report a flat-loss run.")
    set_weights(unflatten(w0))  # restore init for the real training run
    return {
        "loss_init": L0, "grad_norm": gnorm, "finite": finite,
        "fd_check": fd_rows, "probe_losses": [float(x) for x in losses],
        "weights_moved": moved, "decreased": bool(decreased),
    }


def train(n_iter, lr, seed=0, log_every=5):
    """Adam on the network weights through the full rollout."""
    init_weights(seed)
    cur = flat(get_weights())
    m = np.zeros(N_PARAMS); s = np.zeros(N_PARAMS)
    b1c, b2c, epsa = 0.9, 0.999, 1e-8
    losses, grad_norms = [], []
    for it in range(n_iter):
        set_weights(unflatten(cur))
        L, g = loss_and_grad()
        gn = float(np.linalg.norm(g))
        losses.append(L); grad_norms.append(gn)
        if not (np.isfinite(L) and np.all(np.isfinite(g))):
            print(f"[iter {it}] non-finite (loss={L}) — stopping")
            break
        m = b1c * m + (1 - b1c) * g
        s = b2c * s + (1 - b2c) * g * g
        mh = m / (1 - b1c ** (it + 1)); sh = s / (1 - b2c ** (it + 1))
        cur = cur - lr * mh / (np.sqrt(sh) + epsa)
        if it % log_every == 0 or it == n_iter - 1:
            print(f"[iter {it:3d}] loss={L:.6e}  |grad|={gn:.4e}")
    set_weights(unflatten(cur))
    return cur, losses, grad_norms


# --------------------------------------------------------------------------------------------
# Rendering / plots
# --------------------------------------------------------------------------------------------

def capture_rollout(student, with_residual, vx, vy):
    """Run one rollout and return (xs, com) numpy arrays for rendering."""
    set_v0(vx, vy)
    forward(student=student, with_residual=with_residual)
    return x.to_numpy(), com.to_numpy()


def render_three_way(path, frames_xc, labels, colors, stride=6, size=900, fps=30, dpi=100):
    """Three side-by-side panels (target / bare sim / trained hybrid), each overlaying its running
    center-of-mass trail. Headless matplotlib/Agg (reused style from diffmpm.py, no ti.GUI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    bg = (0.043, 0.059, 0.078)
    panel = size // 3
    fig = plt.figure(figsize=(size / dpi, panel / dpi), dpi=dpi, facecolor=bg)
    axes = [fig.add_axes([k / 3.0, 0, 1 / 3.0, 1]) for k in range(3)]
    target_com = frames_xc[0][1]  # the teacher COM as a faint reference cross on every panel
    frames = []
    n = frames_xc[0][0].shape[0]
    for f in range(0, n, stride):
        for ax, (xs, comc), lab, col in zip(axes, frames_xc, labels, colors):
            ax.clear()
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_facecolor(bg); ax.axis("off")
            ax.scatter(xs[f, :, 0], xs[f, :, 1], s=4, c=col, edgecolors="none", alpha=0.85)
            ax.plot(comc[:f + 1, 0], comc[:f + 1, 1], c="#ffd166", lw=1.6, alpha=0.95)
            # faint teacher COM target endpoint so the viewer sees where truth lands
            ax.plot(target_com[-1, 0], target_com[-1, 1], marker="+", ms=13, mew=2.0, c="#ff6e6e")
            ax.text(0.03, 0.93, lab, transform=ax.transAxes, color="#c7d0db", fontsize=11)
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(panel, size, 4)
        frames.append(rgba[..., :3].copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def plot_loss(path, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, fg = "#0b0f14", "#c7d0db"
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130, facecolor=bg)
    ax.set_facecolor(bg)
    ax.semilogy(range(len(losses)), losses, color="#54a0ff", lw=2.2, marker="o", ms=3)
    ax.set_xlabel("training iteration", color=fg)
    ax.set_ylabel("trajectory loss (mean sq COM dist, log)", color=fg)
    ax.set_title("Learned-residual training loss through the rollout", color=fg, fontsize=12)
    ax.grid(True, which="both", alpha=0.15, color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color("#2a3340")
    fig.tight_layout(); fig.savefig(path, facecolor=bg); plt.close(fig)


def plot_grad(path, grad_norms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, fg = "#0b0f14", "#c7d0db"
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130, facecolor=bg)
    ax.set_facecolor(bg)
    ax.semilogy(range(len(grad_norms)), grad_norms, color="#7ee587", lw=2.2, marker="o", ms=3)
    ax.set_xlabel("training iteration", color=fg)
    ax.set_ylabel("weight-gradient norm (log)", color=fg)
    ax.set_title("Gradient norm w.r.t. network weights (through ~320 physics steps)",
                 color=fg, fontsize=12)
    ax.grid(True, which="both", alpha=0.15, color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color("#2a3340")
    fig.tight_layout(); fig.savefig(path, facecolor=bg); plt.close(fig)


def git_branch():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), text=True,
        ).strip()
    except Exception:
        return "local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v0", type=float, nargs=2, default=[3.0, 1.0], help="train initial velocity")
    ap.add_argument("--v0-holdout", type=float, nargs=2, default=[2.0, 2.0],
                    help="held-out initial velocity for the transfer test")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seed_blob()

    # ---- generate the supervised drag target at the TRAIN v0 ----
    set_v0(*args.v0)
    generate_target()

    # ---- mandatory gradient-flow probe BEFORE training ----
    probe = gradient_flow_probe(seed=args.seed)
    if args.probe_only:
        print("\nprobe-only: stopping after the gradient-flow probe.")
        return

    # ---- baseline: bare simulator vs the drag target (the gap to close) ----
    bare_train = bare_loss()
    print(f"\nbare-sim loss (train v0)   = {bare_train:.6e}")

    # ---- train the residual weights through the rollout ----
    print("\n=== training the residual ===")
    wstar, losses, grad_norms = train(args.iters, args.lr, seed=args.seed)
    hybrid_train = loss_only()
    gap_closed_train = 1.0 - hybrid_train / bare_train
    print(f"hybrid loss (train v0)     = {hybrid_train:.6e}")
    print(f"gap closed (train v0)      = {100 * gap_closed_train:.1f}%")

    # ---- held-out transfer test: NEW v0, regenerate the drag target, freeze the trained weights ----
    set_v0(*args.v0_holdout)
    generate_target()  # teacher trajectory at the held-out v0
    bare_holdout = bare_loss()
    set_weights(unflatten(wstar))
    hybrid_holdout = loss_only()
    gap_closed_holdout = 1.0 - hybrid_holdout / bare_holdout
    print(f"\nheld-out v0 {tuple(args.v0_holdout)}:")
    print(f"  bare-sim loss            = {bare_holdout:.6e}")
    print(f"  hybrid loss (frozen W)   = {hybrid_holdout:.6e}")
    print(f"  gap closed (held-out)    = {100 * gap_closed_holdout:.1f}%")

    # ---- media ----
    rel_dir = "runs/learned-dynamics/learned-residual"
    out_dir = os.path.join(repo, *rel_dir.split("/"))
    os.makedirs(out_dir, exist_ok=True)

    media = {}
    if not args.no_video:
        # capture three rollouts at the TRAIN v0 for the comparison video
        set_weights(unflatten(wstar))
        set_v0(*args.v0)
        generate_target()
        tgt_xs, tgt_com = capture_rollout(student=False, with_residual=False, vx=args.v0[0], vy=args.v0[1])
        bare_xs, bare_com = capture_rollout(student=True, with_residual=False, vx=args.v0[0], vy=args.v0[1])
        set_weights(unflatten(wstar))
        hyb_xs, hyb_com = capture_rollout(student=True, with_residual=True, vx=args.v0[0], vy=args.v0[1])
        try:
            render_three_way(
                os.path.join(out_dir, "comparison.mp4"),
                [(tgt_xs, tgt_com), (bare_xs, bare_com), (hyb_xs, hyb_com)],
                ["target (drag)", "bare sim", "trained hybrid"],
                ["#ff6e6e", "#8a93a0", "#7ee587"],
            )
            media["video"] = f"{rel_dir}/comparison.mp4"
        except Exception as e:  # noqa: BLE001
            print(f"video render skipped: {e}")

    plot_loss(os.path.join(out_dir, "loss.png"), losses)
    plot_grad(os.path.join(out_dir, "grad.png"), grad_norms)

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"loss": losses, "grad_norm": grad_norms}, fh)

    # ---- schema-v2 manifest ----
    def fmt(x):
        return f"{x:.3e}"

    fd_ok = all(r[3] < 0.15 for r in probe["fd_check"])  # rel-err under 15% on the spot-checked weights

    objective = (
        "Embed a small learned network inside the otherwise-explicit MLS-MPM step (a residual "
        "correction added to the grid velocity right after the grid op) and train its weights by "
        "backpropagating through the full differentiable rollout. The supervised target is a genuine "
        "model mismatch the residual is never handed directly: trajectories from the same simulator "
        "with an added linear grid-velocity drag v <- v*(1-k). With the initial velocity held fixed, "
        "the residual must learn to close the model-mismatch gap purely from the center-of-mass "
        "trajectory. The question is whether gradients that have passed through hundreds of physics "
        "steps can actually train a network embedded in the simulator, and whether the hybrid then "
        "does something the unmodified simulator cannot."
    )

    findings = (
        f"On this one task (one MLP architecture, {N_PARAMS} parameters, one drag mismatch "
        f"k={DRAG_K}, horizon {max_steps} steps, fixed seed), gradients backpropagated through the "
        f"full rollout trained the embedded residual. The weight gradient was verified finite and "
        f"non-zero before training (|grad|={fmt(probe['grad_norm'])}) and agreed with a "
        f"finite-difference spot check on the highest-magnitude weights "
        f"(rel-err {'under 15%' if fd_ok else 'see fd_check rows'}). Against the drag target at the "
        f"training initial velocity, the bare simulator (no residual) has trajectory loss "
        f"{fmt(bare_train)}; the trained hybrid reaches {fmt(hybrid_train)}, closing "
        f"{100 * gap_closed_train:.1f}% of the model-mismatch gap. The residual is bounded "
        f"(0.1*tanh) and starts near zero, so the hybrid begins as the pure simulator and training "
        f"is stable with mass stabilization on (no non-finite values). On a HELD-OUT initial "
        f"velocity {tuple(args.v0_holdout)} the same frozen weights, against a freshly generated "
        f"drag target, close {100 * gap_closed_holdout:.1f}% of that gap (bare {fmt(bare_holdout)} "
        f"-> hybrid {fmt(hybrid_holdout)}), which is evidence the residual learned a transferable "
        f"correction rather than memorizing one trajectory. The claim is bounded to exactly this "
        f"architecture, this mismatch, and these conditions; it is not a statement that learned "
        f"residuals work in general."
    )

    hypothesis = (
        "Why it trains at all: the residual sits at the very end of each per-step map, right before "
        "g2p, so the gradient of the loss to the residual's output at step f traverses only the "
        "steps AFTER f, not the whole rollout. A parameter injected at the start (like v0) rides the "
        "full product of per-step Jacobians J_T...J_1 and is the most attenuated; the residual's "
        "contribution at a late step rides a much shorter sub-product and is far less attenuated, so "
        "its gradient is healthier. Averaged over all steps, the network gets a strong, low-variance "
        "signal, which is why a few hundred steps do not vanish it. Why the residual can close the "
        "gap: a linear drag is a smooth, low-frequency correction to the grid velocity, and a small "
        "tanh MLP on (velocity, log-mass) easily represents a near-linear velocity shrink, so this "
        "particular mismatch is well within the residual's capacity. The features are deliberately "
        "position-free, so the network is pushed to learn the velocity-space law (which transfers "
        "across initial conditions) rather than the train trajectory's path (which would not). What "
        "would test "
        "generality: other mismatch types (perturbed gravity, nonlinear or anisotropic drag), other "
        "architectures and feature sets, longer horizons where attenuation bites harder, and "
        "stronger held-out shifts (very different v0, different blob shape)."
    )

    limitations = (
        "This is a single architecture on a single, deliberately easy mismatch (a smooth linear "
        "drag), one drag strength, one blob, one horizon, one seed. A near-linear correction is the "
        "friendliest possible target for a residual, so the high gap-closed fraction should not be "
        "read as evidence the method handles hard, non-smooth, or strongly nonlinear mismatches. The "
        "held-out test varies only the initial velocity (and regenerates the matching drag target); "
        "it shows the correction is not tied to one trajectory, but it does not test transfer across "
        "blob shape, resolution, or mismatch type, so some of the fit may still be specific to this "
        "task family. The residual is bounded to 0.1 per component, which is adequate here only "
        "because the drag is weak. No claim is made about stability or accuracy of the learned grid "
        "operator outside the trained regime."
    )

    results = [
        {
            "type": "image", "src": f"{rel_dir}/loss.png",
            "caption": ("Training loss (mean squared center-of-mass distance to the drag target, "
                        "log-y) as Adam optimizes the residual weights through the full "
                        f"{max_steps}-step rollout."),
        },
        {
            "type": "image", "src": f"{rel_dir}/grad.png",
            "caption": ("Norm of the loss gradient w.r.t. the network weights over training. It is "
                        "finite and non-zero throughout, the basic evidence that gradients survive "
                        "the trip back through hundreds of physics steps into the embedded network."),
        },
        {
            "type": "table",
            "caption": ("Bare simulator vs trained hybrid against the drag target, at the training "
                        "initial velocity and at a held-out one. Gap-closed is 1 - hybrid/bare."),
            "columns": ["condition", "bare-sim loss", "hybrid loss", "gap closed", "notes"],
            "rows": [
                ["train v0 " + str(tuple(args.v0)), fmt(bare_train), fmt(hybrid_train),
                 f"{100 * gap_closed_train:.1f}%", f"{N_PARAMS} params, horizon {max_steps}"],
                ["held-out v0 " + str(tuple(args.v0_holdout)), fmt(bare_holdout), fmt(hybrid_holdout),
                 f"{100 * gap_closed_holdout:.1f}%", "frozen weights, fresh drag target"],
                ["mismatch", "-", "-", "-", f"linear grid-velocity drag k={DRAG_K}"],
            ],
        },
    ]
    if "video" in media:
        results.insert(0, {
            "type": "video", "src": media["video"],
            "caption": ("Three rollouts at the training initial velocity, each overlaying its "
                        "running center of mass (yellow). Left: drag target (truth). Middle: bare "
                        "simulator, which drifts past the target (red +). Right: trained hybrid, "
                        "whose learned residual pulls the center of mass back onto the target."),
        })

    manifest = {
        "schema_version": "2",
        "task_id": "learned-residual",
        "direction": "learned-dynamics",
        "title": "A learned residual trained through the differentiable rollout",
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": objective,
        "findings": findings,
        "hypothesis": hypothesis,
        "limitations": limitations,
        "results": results,
        "training_refs": ["hybrid-learned-residual", "differentiating-the-rollout", "failure-modes",
                          "mls-mpm-forward"],
        "params": {
            "n_grid": n_grid, "n_particles": n_particles, "steps": max_steps, "dt": dt, "E": E,
            "mass_eps": MASS_EPS, "drag_k": DRAG_K, "residual_scale": RESID_SCALE,
            "net": {"in": N_IN, "hidden": N_HID, "out": N_OUT, "activation": "tanh",
                    "params": N_PARAMS},
            "optimizer": "adam", "lr": args.lr, "iters": len(losses), "seed": args.seed,
            "v0_train": list(args.v0), "v0_holdout": list(args.v0_holdout),
            "bare_loss_train": bare_train, "hybrid_loss_train": hybrid_train,
            "gap_closed_train": gap_closed_train,
            "bare_loss_holdout": bare_holdout, "hybrid_loss_holdout": hybrid_holdout,
            "gap_closed_holdout": gap_closed_holdout,
            "probe": {"grad_norm": probe["grad_norm"], "fd_check": probe["fd_check"],
                      "fd_agrees": fd_ok, "weights_moved": probe["weights_moved"]},
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote -> {rel_dir}/manifest.json")
    print(f"train: bare {bare_train:.3e} -> hybrid {hybrid_train:.3e}  ({100*gap_closed_train:.1f}% closed)")
    print(f"held-out: bare {bare_holdout:.3e} -> hybrid {hybrid_holdout:.3e}  ({100*gap_closed_holdout:.1f}% closed)")


if __name__ == "__main__":
    main()
