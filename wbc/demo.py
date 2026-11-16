"""WBC stage demo: the whole body QP standing in MuJoCo, and the WBC stage's media.

Written in 2026.

    python -m wbc.demo pose                    # pose tracking while standing, prints the errors
    python -m wbc.demo push                    # three pushes, prints displacement and recovery
    python -m wbc.demo push --force 200        # one sideways push of 200 N instead
    python -m wbc.demo swing                   # stands on three feet, the fourth draws circles
    python -m wbc.demo swing --forces          # the same, with static desired forces as f_des
    python -m wbc.demo forces                  # stands with f_des = m g / 4 on every foot
    python -m wbc.demo pose --gif pose.gif     # also writes a GIF (no window)
"""
import os
import sys

if "--gif" in sys.argv:   # a GIF renders offscreen: pick EGL before MuJoCo is first imported
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import go2
from common.video import Recorder, WindowClosed, save_gif
from wbc import scenarios as run
from wbc.wbc import foot_forces

MEDIA = Path(__file__).resolve().parents[1] / "media"
COLORS = {"FL": "#1f77b4", "FR": "#ff7f0e", "RL": "#2ca02c", "RR": "#d62728"}
PUSH_N_PER_M = 500.0             # push arrow: 1 cm per 5 N
FOOT_N_PER_M = 250.0             # ground force arrows: 1 cm per 2.5 N
FPS = 12                         # GIF frame rate
POSE_FPS = 8                     # the pose clip moves slowly, so fewer frames are enough
SLOW = 4                         # push clips play 4 times slower round each push
FONT = ImageFont.load_default(size=12)


def label(frames, texts):
    out = []
    for f, text in zip(frames, texts):
        img = Image.fromarray(f)
        ImageDraw.Draw(img).text((8, 6), text, fill="white", font=FONT)
        out.append(np.asarray(img))
    return out


def add_geom(scene, kind, size, rgba, start=None, end=None, pos=None, width=0.0):
    """Add one shape to the scene that is about to be drawn: an arrow from start to end, or a
    sphere at pos."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array(size, float), np.zeros(3) if pos is None else pos,
                        np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    if start is not None:
        mujoco.mjv_connector(g, kind, width, start, end)
    scene.ngeom += 1


def force_arrows(model):
    """Recorder.extra: red arrow for the push, green arrows for the ground force on each foot."""
    geoms = go2.foot_geoms(model)

    def draw(scene, data):
        push = data.xfrc_applied[1, :3]
        if np.linalg.norm(push) > 0:
            u = push / np.linalg.norm(push)
            tip = data.xipos[1] - 0.12 * u            # stop short of the trunk's centre of mass
            add_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), (0.9, 0.1, 0.1, 1),
                     start=tip - u * np.linalg.norm(push) / PUSH_N_PER_M, end=tip, width=0.014)
        for leg, f in enumerate(foot_forces(model, data, geoms)):
            if f[2] > 1.0:
                base = data.geom_xpos[geoms[leg]] - [0.0, 0.0, go2.FOOT_RADIUS]
                add_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), (0.2, 0.85, 0.3, 1),
                         start=base, end=base + f / FOOT_N_PER_M, width=0.010)
    return draw


def move_name(t):
    k = int((t - 1.0) // run.SEGMENT)
    if t < 1.0 or k >= len(run.MOVES):
        return "standing still"
    return {"roll": "roll ±15°", "pitch": "pitch ±15°", "yaw": "yaw ±20°",
            "height": "height ±6 cm", "circle": "roll and pitch circle, 10°"}[run.MOVES[k][0]]


def pose_demo(model):
    rec = Recorder(model, 300, 225, fps=POSE_FPS, distance=1.3, azimuth=135, elevation=-20)
    log = run.simulate(run.POSE_SECONDS, run.dance, frame_hook=lambda d, w: rec(d), model=model)
    times = np.arange(len(rec.frames)) / POSE_FPS
    save_gif(label(rec.frames, [move_name(t) for t in times]), MEDIA / "wbc_pose.gif",
             fps=POSE_FPS)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    rows = [("roll (deg)", np.degrees(log["rpy"][:, 0]), np.degrees(log["rpy_ref"][:, 0])),
            ("pitch (deg)", np.degrees(log["rpy"][:, 1]), np.degrees(log["rpy_ref"][:, 1])),
            ("yaw (deg)", np.degrees(log["rpy"][:, 2]), np.degrees(log["rpy_ref"][:, 2])),
            ("height (cm)", 100 * log["pos"][:, 2], 100 * log["pos_ref"][:, 2])]
    for ax, (name, measured, target) in zip(axes, rows):
        ax.plot(log["t"], target, color="0.6", lw=3, label="target")
        ax.plot(log["t"], measured, color="#1f77b4", lw=1.2, label="simulation")
        ax.set_ylabel(name)
        for k in range(len(run.MOVES) + 1):
            ax.axvline(1.0 + k * run.SEGMENT, color="0.85", lw=0.8)
    for k, (name, _) in enumerate(run.MOVES):
        axes[0].text(1.0 + (k + 0.5) * run.SEGMENT, 19, name, ha="center", fontsize=9)
    axes[0].set_ylim(-20, 24)
    axes[0].legend(loc="lower left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Body pose targets and the simulated body, standing on four feet", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / "wbc_pose.png", dpi=110)
    plt.close(fig)
    return log


def push_clip(model, seconds, pushes, captions):
    """Frames of a push run: SLOW times slower from 0.1 s before each push to 0.7 s after."""
    rec = Recorder(model, 320, 240, fps=FPS * SLOW, distance=1.6, azimuth=120, elevation=-22)
    rec.extra = force_arrows(model)
    log = run.simulate(seconds, run.hold, pushes, frame_hook=lambda d, w: rec(d), model=model)
    frames, texts = [], []
    for k, frame in enumerate(rec.frames):
        t = k / (FPS * SLOW)
        near = [i for i, (s, _, _) in enumerate(pushes) if s - 0.1 <= t < s + 0.7]
        if near:
            frames.append(frame)
            texts.append(f"{captions[near[0]]}, {SLOW}x slower")
        elif k % SLOW == 0:
            frames.append(frame)
            texts.append("red: push, green: ground forces")
    return log, label(frames, texts)


def push_plots(log):
    sel = log["t"] >= 0.5                  # after the robot has settled onto its feet
    t = log["t"][sel]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    err = 1000 * (log["pos"][sel] - log["pos_ref"][sel])
    axes[0].plot(t, err[:, 0], label="x (forward)")
    axes[0].plot(t, err[:, 1], label="y (left)")
    axes[0].plot(t, err[:, 2], label="z (height)")
    axes[0].set_ylabel("body off target (mm)")
    axes[0].set_ylim(-120, 140)
    axes[1].plot(t, np.degrees(log["rpy"][sel, 0]), label="roll")
    axes[1].plot(t, np.degrees(log["rpy"][sel, 1]), label="pitch")
    axes[1].plot(t, np.degrees(log["rpy"][sel, 2]), label="yaw")
    axes[1].set_ylabel("body angles (deg)")
    dt = t[1] - t[0]
    for leg, name in enumerate(go2.LEGS):
        axes[2].plot(t, 1000 * np.cumsum(log["slip"][sel, leg]) * dt, color=COLORS[name],
                     label=name)
    axes[2].set_ylabel("distance slid (mm)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        for start, duration, force in run.PUSHES:
            ax.axvspan(start, start + duration, color="#d62728", alpha=0.25, lw=0)
        ax.legend(fontsize=8, loc="lower right", ncol=4)
    for start, duration, force in run.PUSHES:
        f = np.asarray(force)
        where = "sideways" if f[0] == 0 else "forwards"
        axes[0].text(start + 0.05, 115, f"{np.linalg.norm(f):.0f} N {where}", fontsize=8,
                     color="#d62728")
    fig.suptitle("Three pushes of 0.1 s on the trunk (red bands), from 0.5 s", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / "wbc_push.png", dpi=110)
    plt.close(fig)

    start = run.PUSHES[2][0]
    sel = (log["t"] >= start - 0.1) & (log["t"] < start + 0.9)
    ts = log["t"][sel] - start
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for leg, name in enumerate(go2.LEGS):
        fm, fc = log["f_meas"][sel, leg], log["f_cmd"][sel, leg]
        a1.plot(ts, fm[:, 2], color=COLORS[name], label=f"{name} simulation")
        a1.plot(ts, fc[:, 2], color=COLORS[name], ls="--", lw=1, label=f"{name} asked by the QP")
        ratio = np.linalg.norm(fm[:, :2], axis=1) / np.maximum(fm[:, 2], 1e-3)
        asked = np.linalg.norm(fc[:, :2], axis=1) / fc[:, 2]
        a2.plot(ts, np.where(fm[:, 2] > 1.0, ratio, np.nan), color=COLORS[name], label=name)
        a2.plot(ts, asked, color=COLORS[name], ls="--", lw=1)
    a2.axhline(0.5, color="k", ls=":", lw=1)
    a2.text(0.62, 0.52, "friction cone in the QP (0.5)", fontsize=8)
    a2.axhline(0.8, color="k", lw=1)
    a2.text(0.62, 0.82, "friction of the foot in the model (0.8)", fontsize=8)
    a1.set_ylabel("normal force (N)")
    a1.legend(fontsize=7, ncol=4, loc="upper right")
    a2.set_ylabel("sideways / normal force\n(dashed: asked by the QP)")
    a2.set_ylim(0, 1.0)
    a2.set_xlabel(f"time from the start of the push (s), {np.linalg.norm(run.PUSHES[2][2]):.0f} N "
                  "sideways for 0.1 s")
    a2.legend(fontsize=8, ncol=4, loc="upper right")
    for ax in (a1, a2):
        ax.axvspan(0, 0.1, color="#d62728", alpha=0.25, lw=0)
    fig.suptitle("Foot forces during the push, and how much of the friction cone each foot uses",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / "wbc_forces.png", dpi=110)
    plt.close(fig)


def swing_demo(model):
    """Three-legged stand: GIF with the target path drawn as yellow dots, and a tracking plot."""
    rec = Recorder(model, 360, 270, fps=FPS, distance=1.1, azimuth=-115, elevation=-12)
    p0 = {}

    def dots(scene, data):
        if "p" not in p0 and data.time >= run.T_UP:   # the controller anchors its path here too:
            p0["p"] = data.geom_xpos[go2.foot_geoms(model)[run.SWING_LEG]].copy()   # at lift-off
        if "p" in p0 and data.time < run.T_LANDED + 0.5:
            for s in np.linspace(run.T_UP, run.T_LANDED, 60):
                add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005] * 3, (1.0, 0.85, 0.1, 1),
                         pos=run.swing_target(s, p0["p"])[0])
    rec.extra = dots
    log = run.simulate(run.SWING_SECONDS, run.lean, frame_hook=lambda d, w: rec(d), model=model,
                       plan=run.three_leg_plan(model))
    times = np.arange(len(rec.frames)) / FPS
    texts = ["leaning over the other three feet" if t < run.T_UP else
             "front left foot follows the yellow path" if t < run.T_LANDED else "back on four feet"
             for t in times]
    save_gif(label(rec.frames, texts), MEDIA / "wbc_swing.gif", fps=FPS)

    sel = (log["t"] > run.T_UP) & (log["t"] < run.T_LANDED)
    foot, ref = log["feet"][sel, run.SWING_LEG][:-1], log["swing_ref"][sel][1:]
    origin = log["feet"][np.searchsorted(log["t"], run.T_UP) - 1, run.SWING_LEG]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[1, 1.4]))
    a1.plot(100 * (ref[:, 0] - origin[0]), 100 * (ref[:, 2] - origin[2]), color="0.6", lw=3,
            label="target")
    a1.plot(100 * (foot[:, 0] - origin[0]), 100 * (foot[:, 2] - origin[2]), color="#1f77b4",
            lw=1.2, label="simulation")
    a1.set_aspect("equal")
    a1.set_xlabel("forward (cm)")
    a1.set_ylabel("up (cm)")
    a1.set_title("Front left foot, side view", fontsize=10, loc="left")
    a1.legend(fontsize=8, loc="lower right")
    ts = log["t"][sel][1:]
    a2.plot(ts, 1000 * np.linalg.norm(foot - ref, axis=1), label="swing foot")
    a2.plot(log["t"], 1000 * np.linalg.norm(log["pos"] - log["pos_ref"], axis=1),
            label="body position")
    a2.axvspan(run.T_UP, run.T_LANDED, color="0.9", lw=0)
    a2.set_xlabel("time (s), grey: foot in the air")
    a2.set_ylabel("error (mm)")
    a2.set_title("Tracking errors", fontsize=10, loc="left")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / "wbc_swing.png", dpi=110)
    plt.close(fig)
    return log


def media():
    """Run the pose, push and swing scenarios and write the WBC stage's media."""
    MEDIA.mkdir(exist_ok=True)
    model = go2.load_model()

    log = pose_demo(model)
    print("Pose demo (largest and RMS error):")
    for name, (peak, rms) in run.pose_errors(log).items():
        unit = "mm" if name == "height" else "deg"
        print(f"  {name:6s} {peak:.2f} {unit}, RMS {rms:.2f} {unit}")
    print(f"  QP {log['qp_ms'].mean():.3f} ms mean, {log['qp_ms'].max():.2f} ms max, control step "
          f"{log['step_ms'].mean():.2f} ms mean, failures {log['failures']}")
    plain = run.simulate(run.POSE_SECONDS, run.dance, model=model, friction_comp=False)
    print("  without the friction term:", ", ".join(
        f"{n} {p:.2f}" for n, (p, _) in run.pose_errors(plain).items()))

    captions = [f"{np.linalg.norm(f):.0f} N {'sideways' if f[0] == 0 else 'forwards'}, 0.1 s"
                for _, _, f in run.PUSHES]
    log, frames = push_clip(model, run.PUSH_SECONDS, run.PUSHES, captions)
    print("Push demo:")
    for start, duration, force in run.PUSHES:
        peak, tilt, rec_t, slide = run.push_response(log, start, duration)
        print(f"  {np.linalg.norm(force):.0f} N at {start} s: moved {1000 * peak:.1f} mm, tilt "
              f"{tilt:.2f} deg, back within 5 mm and 1 deg after {rec_t:.2f} s, slid "
              f"{1000 * slide:.1f} mm")
    print(f"  upright at the end: {np.degrees(np.abs(log['rpy'][-1, :2])).max() < 5}, "
          f"failures {log['failures']}")
    push_plots(log)
    fall, fall_frames = push_clip(model, 3.0, run.TOO_STRONG,
                                  ["150 N sideways: too strong, it tips over"])
    print(f"  150 N sideways: largest tilt {np.degrees(np.abs(fall['rpy'][:, :2])).max():.0f} deg")
    save_gif(frames + fall_frames, MEDIA / "wbc_push.gif", fps=FPS)

    log = swing_demo(model)
    peak, rms, body, ang = run.swing_errors(log)
    print(f"Three-legged stand: swing foot error largest {peak:.1f} mm, RMS {rms:.1f} mm, body "
          f"position error largest {body:.1f} mm, angle {ang:.2f} deg, QP "
          f"{log['qp_ms'].mean():.3f} ms mean, failures {log['failures']}")
    print("written:", ", ".join(f"{p.name} {p.stat().st_size / 1e6:.2f} MB"
                                for p in sorted(MEDIA.iterdir())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=["pose", "push", "swing", "forces"])
    ap.add_argument("--force", type=float, help="push: one push of this many newtons at 1 s, to "
                    "the robot's left")
    ap.add_argument("--forces", action="store_true", help="swing: give static forces as f_des")
    ap.add_argument("--gif", help="also write a GIF of the run")
    ap.add_argument("--view", action="store_true", help="watch the run in the MuJoCo viewer")
    args = ap.parse_args()
    model = go2.load_model()
    path, pushes, plan = run.hold, (), None
    if args.demo == "pose":
        path, seconds = run.dance, run.POSE_SECONDS
    elif args.demo == "push":
        seconds = run.PUSH_SECONDS
        pushes = run.PUSHES if args.force is None else ((1.0, 0.1, (0.0, args.force, 0.0)),)
        if args.force is not None:
            seconds = 3.5
    elif args.demo == "swing":
        path, seconds, plan = run.lean, run.SWING_SECONDS, run.three_leg_plan(model, args.forces)
    else:
        seconds = 3.0
        weight = model.body_subtreemass[1] * 9.81
        plan = lambda t, data: {"f_des": np.tile([0.0, 0.0, weight / 4], (4, 1))}
    hook, close, data = None, None, None
    if args.view:
        from common.video import viewer_hook
        data = mujoco.MjData(model)
        arrows = force_arrows(model)
        hook, close = viewer_hook(model, data, extra=lambda scene, d, w: arrows(scene, d))
    elif args.gif:
        rec = Recorder(model, 480, 360, fps=15, distance=1.6, azimuth=135, elevation=-20)
        hook = lambda data, wbc: rec(data)
    try:
        log = run.simulate(seconds, path, pushes, frame_hook=hook, model=model, plan=plan, data=data)
    except WindowClosed:
        print(f"viewer closed at t = {data.time:.1f} s")
        return
    finally:
        if close is not None:
            close()
    if args.gif and not args.view:
        save_gif(rec.frames, args.gif, fps=15)
    if args.demo == "pose":
        for name, (peak, rms) in run.pose_errors(log).items():
            unit = "mm" if name == "height" else "deg"
            print(f"{name:6s}: largest error {peak:.2f} {unit}, RMS {rms:.2f} {unit}")
    elif args.demo == "push":
        for start, duration, force in pushes:
            peak, tilt, rec_t, slide = run.push_response(log, start, duration)
            print(f"push {np.linalg.norm(force):.0f} N for {duration} s at {start} s: body moved "
                  f"{1e3 * peak:.1f} mm, tilted {tilt:.2f} deg, back within 5 mm and 1 deg after "
                  f"{rec_t:.2f} s, longest foot slide {1e3 * slide:.1f} mm")
    elif args.demo == "swing":
        peak, rms, body, ang = run.swing_errors(log)
        print(f"swing foot: largest error {peak:.1f} mm, RMS {rms:.1f} mm. Body: largest position "
              f"error {body:.1f} mm, largest angle error {ang:.2f} deg")
    else:
        sel = log["t"] >= 1.0
        print(f"QP forces minus f_des: largest {np.abs(log['f_cmd'][sel] - log['f_des'][sel]).max():.2f} N. "
              f"Simulated foot forces minus f_des: largest "
              f"{np.abs(log['f_meas'][sel] - log['f_des'][sel]).max():.2f} N. Body: largest position "
              f"error {1e3 * np.linalg.norm(log['pos'] - log['pos_ref'], axis=1)[sel].max():.1f} mm, "
              f"largest tilt {np.degrees(np.abs(log['rpy'][sel, :2])).max():.2f} deg")
    upright = np.degrees(np.abs(log["rpy"][-1, :2])).max() < 5.0
    print(f"upright at the end: {upright}. QP solve {log['qp_ms'].mean():.3f} ms mean, "
          f"{log['qp_ms'].max():.2f} ms max. Whole control step {log['step_ms'].mean():.2f} ms "
          f"mean. QP failures: {log['failures']}")


if __name__ == "__main__":
    main()
