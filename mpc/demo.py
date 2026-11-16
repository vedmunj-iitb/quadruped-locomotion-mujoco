"""MPC stage demo: the convex MPC trot in MuJoCo, and the MPC stage's media.

Written in 2026.

    python -m mpc.demo                       # the demo: stand, trot, 0.5 and 1.0 m/s, sideways, turn
    python -m mpc.demo --vx 0.7 --seconds 8  # one constant command (after 1 s of standing)
    python -m mpc.demo --vx 0.4 --yaw-rate 0.5 --gif turn.gif
"""
import os
import sys

if "--gif" in sys.argv:   # a GIF renders offscreen: pick EGL before MuJoCo is first imported
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from common import go2
from common.video import Recorder, WindowClosed, save_gif
from mpc.controller import STAND, TROT, simulate

MEDIA = Path(__file__).resolve().parents[1] / "media"
COLORS = {"FL": "#1f77b4", "FR": "#ff7f0e", "RL": "#2ca02c", "RR": "#d62728"}
ARROW_SCALE = 0.004           # connector length per N. MuJoCo draws an arrow at half the
                              # connector's length (checked against a capsule), so 2 mm per N
MU = 0.5                      # friction coefficient used by the MPC (pyramid)
MU_SIM = 0.8                  # friction coefficient of the feet in the MuJoCo model
# The demo: (end time in s, forward m/s, sideways m/s, yaw rate rad/s, gait, label)
DEMO = [
    (1.5, 0.0, 0.0, 0.0, STAND, "stand"),
    (3.0, 0.0, 0.0, 0.0, TROT, "trot in place"),
    (6.0, 0.5, 0.0, 0.0, TROT, "forward 0.5 m/s"),
    (9.0, 1.0, 0.0, 0.0, TROT, "forward 1.0 m/s"),
    (11.5, 0.0, 0.3, 0.0, TROT, "sideways 0.3 m/s"),
    (14.0, 0.5, 0.0, 0.8, TROT, "turning 0.8 rad/s"),
    (15.5, 0.0, 0.0, 0.0, TROT, "trot in place"),
    (17.0, 0.0, 0.0, 0.0, STAND, "stand"),
]


def schedule_from(segments):
    def schedule(t):
        for end, vx, vy, wz, gait, _ in segments:
            if t < end:
                return vx, vy, wz, gait
        return segments[-1][1:5]
    return schedule


def segment_table(log, segments):
    """Over the second half of each segment, after it has settled: the mean speeds, turn rate
    and height, and the largest roll and pitch."""
    rows, start = [], 0.0
    for end, vx, vy, wz, gait, label in segments:
        sel = (log["t"] >= (start + end) / 2) & (log["t"] < end)
        rows.append(dict(label=label, cmd=(vx, vy, wz), vx=log["vel"][sel, 0].mean(),
                         vy=log["vel"][sel, 1].mean(), wz=log["yaw_rate"][sel].mean(),
                         com_z=log["com"][sel, 2].mean(),
                         roll=np.degrees(np.abs(log["rpy"][sel, 0])).max(),
                         pitch=np.degrees(np.abs(log["rpy"][sel, 1])).max()))
        start = end
    return rows


def add_geom(scene, kind, rgba, size=(0, 0, 0), pos=(0, 0, 0)):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array(size, float), np.array(pos, float),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
    scene.ngeom += 1
    return g


def draw_mpc(ctrl):
    """Recorder callback: an arrow for each stance foot's MPC force, and dots on the floor under
    the MPC's predicted centre of mass for the next 0.36 s."""
    def extra(scene, data):
        for i in range(scene.ngeom):              # see-through robot, so the arrows show
            g = scene.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and data.model.geom_bodyid[g.objid] > 0:
                g.rgba[3] = 0.45
        for leg in range(4):
            f = ctrl.forces[leg]
            if not ctrl.sched.in_stance(leg) or f[2] < 5.0:
                continue
            start = data.geom_xpos[ctrl.feet_geoms[leg]] - [0.0, 0.0, go2.FOOT_RADIUS]
            g = add_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, (0.75, 1.0, 0.1, 1.0))
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, start,
                                     start + ARROW_SCALE * f)
        if ctrl.prediction is not None:
            for p in ctrl.prediction[:, 3:6]:
                add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (1.0, 0.55, 0.1, 1.0),
                         size=(0.012, 0, 0), pos=(p[0], p[1], 0.004))
    return extra


def segment_at(t):
    for seg in DEMO:
        if t < seg[0]:
            return seg
    return DEMO[-1]


def record_demo(model):
    """Run the demo once, with a recorder that also notes what to write on each frame."""
    rec = Recorder(model, 360, 225, fps=12, distance=1.6, azimuth=115, elevation=-14,
                   start=1.5, end=14.0)
    recent = deque(maxlen=int(TROT.period / model.opt.timestep))    # last gait cycle
    labels = []

    def hook(data, ctrl):
        if rec.extra is None:
            rec.extra = draw_mpc(ctrl)
        heading = go2.rot_z(go2.body_rpy(data)[2])
        recent.append(heading.T @ data.qvel[:3])
        n = len(rec.frames)
        rec(data)
        if len(rec.frames) > n:
            seg = segment_at(data.time)
            v = np.mean(recent, axis=0)
            labels.append(f"{seg[5]}: asked {seg[1]:.1f}, {seg[2]:.1f}  got {v[0]:.2f}, "
                          f"{v[1]:.2f} m/s")

    log = simulate(schedule_from(DEMO), DEMO[-1][0], frame_hook=hook, model=model)
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=11)
    frames = []
    for f, text in zip(rec.frames, labels):
        img = Image.fromarray(f)
        ImageDraw.Draw(img).text((8, 6), text, fill="white", font=font)
        frames.append(np.asarray(img))
    save_gif(frames, MEDIA / "mpc_trot.gif", fps=12, colors=96)
    return log


def shade_stand(ax):
    start = 0.0
    for end, *_, gait, _label in DEMO:
        if gait.name == "stand":
            ax.axvspan(start, end, color="0.92", lw=0)
        start = end


def velocity_plot(log):
    import matplotlib.pyplot as plt
    t = log["t"]
    n = int(TROT.period / (t[1] - t[0]))
    smooth = lambda x: np.convolve(x, np.ones(n) / n, mode="same")   # one gait cycle average
    signals = [("forward speed (m/s)", log["vel"][:, 0], 0),
               ("sideways speed (m/s)", log["vel"][:, 1], 1),
               ("yaw rate (rad/s)", log["yaw_rate"], 2)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    for ax, (name, meas, i) in zip(axes, signals):
        shade_stand(ax)
        ax.plot(t, meas, color="#9ecae1", lw=0.6, label="measured")
        ax.plot(t, smooth(meas), color="#08519c", lw=1.6, label="measured, one cycle average")
        ax.plot(t, log["target"][:, i], color="k", lw=1.2, ls="--", label="command")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left", ncol=3)
    axes[-1].set_xlabel("time (s), grey: standing")
    fig.suptitle("Velocity tracking of the MPC trot (MuJoCo, Unitree Go2 model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEDIA / "mpc_velocity.png", dpi=110)
    plt.close(fig)


def force_plot(log):
    """Vertical force of each foot: what the MPC asked for and what MuJoCo's contact gave."""
    import matplotlib.pyplot as plt
    t0, t1 = 4.5, 4.5 + 3 * TROT.period
    sel = (log["t"] >= t0) & (log["t"] < t1)
    t = log["t"][sel]
    fig, axes = plt.subplots(4, 1, figsize=(9, 6.5), sharex=True, sharey=True)
    for leg, ax in enumerate(axes):
        name = go2.LEGS[leg]
        ax.fill_between(t, 0, 200, where=log["plan"][sel, leg], step="post", color=COLORS[name],
                        alpha=0.12, lw=0)
        ax.plot(t, log["forces"][sel, leg, 2], color=COLORS[name], lw=1.8, label="MPC")
        ax.plot(t, log["contact_forces"][sel, leg, 2], color="k", lw=0.8, label="MuJoCo contact")
        ax.set_ylabel(f"{name} fz (N)")
        ax.set_ylim(-5, 200)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("time (s), trot at 0.5 m/s, shaded: planned stance")
    fig.suptitle("Ground reaction force of each foot, three trot cycles", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEDIA / "mpc_forces.png", dpi=110)
    plt.close(fig)


def friction_plot(log):
    """Every foot force divided by its normal force: (fx/fz, fy/fz). The MPC keeps them in the
    pyramid |fx|, |fy| <= 0.5 fz, which fits inside MuJoCo's friction cone of 0.8."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    for ax, key, title in [(axes[0], "forces", "MPC forces"),
                           (axes[1], "contact_forces", "MuJoCo contact forces")]:
        F = log[key][log["plan"]]                    # stance feet only
        F = F[F[:, 2] > 10.0]
        ax.scatter(F[:, 0] / F[:, 2], F[:, 1] / F[:, 2], s=2, alpha=0.3, color="#08519c")
        ax.add_patch(plt.Rectangle((-MU, -MU), 2 * MU, 2 * MU, fill=False, color="#2ca02c", lw=1.5,
                                   label=f"MPC pyramid, mu = {MU}"))
        ax.add_patch(plt.Circle((0, 0), MU_SIM, fill=False, color="#d62728", lw=1.5,
                                label=f"friction cone of the model, mu = {MU_SIM}"))
        ax.set_aspect("equal")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_xlabel("fx / fz")
        ax.set_title(f"{title} ({len(F)} samples)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fy / fz")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(MEDIA / "mpc_friction.png", dpi=110)
    plt.close(fig)


def solve_time_plot(log):
    import matplotlib.pyplot as plt
    st = 1000 * log["solve_times"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(st, bins=40, color="#08519c")
    ax.axvline(np.median(st), color="k", ls="--", lw=1, label=f"median {np.median(st):.2f} ms")
    ax.set_xlabel("OSQP time per MPC solve: update, solve and polish (ms)")
    ax.set_ylabel("solves")
    ax.set_title(f"{len(st)} solves in the {DEMO[-1][0]:.0f} s demo (this PC)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / "mpc_solve_time.png", dpi=110)
    plt.close(fig)


def media():
    """Run the demo once and write the MPC stage's media, printing the numbers."""
    import matplotlib
    matplotlib.use("Agg")
    MEDIA.mkdir(exist_ok=True)
    model = go2.load_model()
    log = record_demo(model)
    velocity_plot(log)
    force_plot(log)
    friction_plot(log)
    solve_time_plot(log)

    for r in segment_table(log, DEMO):
        print(f"{r['label']:18s} command {r['cmd']}  measured vx {r['vx']:.3f} vy {r['vy']:.3f} "
              f"wz {r['wz']:.3f}  CoM z {r['com_z']:.3f}  max roll {r['roll']:.1f} pitch {r['pitch']:.1f}")
    st = 1000 * log["solve_times"]
    print(f"MPC solves {len(st)} (OSQP update, solve and polish): median {np.median(st):.2f} ms, "
          f"99 % below {np.percentile(st, 99):.2f} ms, slowest {st.max():.2f} ms")
    F, C, plan = log["forces"], log["contact_forces"], log["plan"]
    mpc_ratio = (np.abs(F[..., :2]).max(-1) / np.maximum(F[..., 2], 1e-9))[plan & (F[..., 2] > 1)]
    swing_force = np.abs(F[~plan]).max() if (~plan).any() else 0.0
    loaded = plan & (C[..., 2] > 10.0)
    sim_ratio = np.hypot(C[..., 0], C[..., 1])[loaded] / C[..., 2][loaded]          # cone measure
    outside = np.abs(C[..., :2]).max(-1)[loaded] / C[..., 2][loaded] > MU          # pyramid measure
    print(f"MPC: largest max(|fx|, |fy|) / fz on stance feet {mpc_ratio.max():.3f}, largest fz "
          f"{F[..., 2].max():.1f} N, largest |f| on swing feet {swing_force:.4f} N")
    print(f"MuJoCo contacts (stance feet, fz > 10 N): {100 * np.mean(outside):.1f} % outside the MPC "
          f"pyramid, largest |ft|/fz {sim_ratio.max():.3f} (model friction {MU_SIM})")
    print(f"lowest body height {log['pos'][:, 2].min():.3f} m, largest tilt "
          f"{np.degrees(np.abs(log['rpy'][:, :2])).max():.1f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, help="forward speed command (m/s)")
    ap.add_argument("--vy", type=float, default=0.0, help="sideways speed command (m/s, left +)")
    ap.add_argument("--yaw-rate", type=float, default=0.0, help="turn rate command (rad/s)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--gif", help="also write a GIF of the run here")
    ap.add_argument("--view", action="store_true", help="watch the run in the MuJoCo viewer")
    args = ap.parse_args()
    if args.vx is None and args.vy == 0.0 and args.yaw_rate == 0.0:
        segments = DEMO
    else:
        segments = [(1.0, 0.0, 0.0, 0.0, STAND, "stand"),
                    (args.seconds, args.vx or 0.0, args.vy, args.yaw_rate, TROT, "trot")]

    model = go2.load_model()
    hook, close, data = None, None, None
    if args.view:
        from common.video import viewer_hook
        data = mujoco.MjData(model)
        hook, close = viewer_hook(model, data, extra=lambda scene, d, ctrl: draw_mpc(ctrl)(scene, d))
    elif args.gif:
        rec = Recorder(model, 480, 320, fps=20, distance=1.6)
        hook = lambda data, ctrl: rec(data)
    try:
        log = simulate(schedule_from(segments), segments[-1][0], frame_hook=hook, model=model,
                       data=data)
    except WindowClosed:
        print(f"viewer closed at t = {data.time:.1f} s")
        return
    finally:
        if close is not None:
            close()
    if args.gif and not args.view:
        save_gif(rec.frames, args.gif)

    print(f"{'segment':20s} {'command (vx, vy, wz)':>22s} {'measured vx':>11s} {'vy':>6s} "
          f"{'wz':>6s} {'CoM z':>6s} {'roll':>5s} {'pitch':>5s}")
    for r in segment_table(log, segments):
        print(f"{r['label']:20s} {str(r['cmd']):>22s} {r['vx']:11.3f} {r['vy']:6.3f} {r['wz']:6.3f} "
              f"{r['com_z']:6.3f} {r['roll']:5.1f} {r['pitch']:5.1f}")
    st = 1000 * log["solve_times"]
    print(f"OSQP solves: {len(st)}, median {np.median(st):.2f} ms, slowest {st.max():.2f} ms")


if __name__ == "__main__":
    main()
