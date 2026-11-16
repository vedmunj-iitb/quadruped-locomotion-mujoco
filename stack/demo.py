"""Whole stack demo: foothold planner, gait, MPC and WBC together on the course, in the MuJoCo
viewer or to a GIF. Also the stack's media: the course with and without the foothold planner,
and a flat ground speed run.

Written in 2026.

    python -m stack.demo                    # viewer: the whole stack trots over the course
    python -m stack.demo --flat --vx 0.5    # flat ground, one speed
    python -m stack.demo --raibert-only     # no foothold selection: feet land on the first guess
    python -m stack.demo --gif stack.gif    # no window, writes a GIF
"""
import os
import sys

if "--gif" in sys.argv:   # a GIF renders offscreen: pick EGL before MuJoCo is first imported
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import mujoco
import numpy as np

from common import go2
from common.metrics import fall_index, hole_drops
from common.video import Recorder, save_gif
from footholds.terrain import make_course
from stack.stack import STAND, TROT, run_stack

MEDIA = Path(__file__).resolve().parents[1] / "media"
COURSE_SPEED = 0.4
COURSE = [(1.0, 0.0, STAND), (9.5, COURSE_SPEED, TROT), (10.5, 0.0, TROT), (11.5, 0.0, STAND)]
FLAT = [(1.0, 0.0, STAND), (5.0, 0.3, TROT), (9.0, 0.6, TROT), (10.0, 0.0, TROT), (11.0, 0.0, STAND)]
ARROW_SCALE = 0.004    # connector length per N (MuJoCo draws the arrow at half of it: 2 mm per N)


def schedule_from(segments):
    """(end time, forward speed, gait) segments -> schedule(t) for run_stack."""
    def schedule(t):
        for end, vx, gait in segments:
            if t < end:
                return vx, 0.0, 0.0, gait
        return 0.0, 0.0, 0.0, segments[-1][2]
    return schedule


def summary(log, ctrl, terrain=None, segments=COURSE):
    """Distance, fall, trot speed (from 3 s to the end of the last moving segment, or the fall),
    largest tilt, feet into a hole, steps moved by the planner and timings, all up to the fall."""
    end = fall_index(log, terrain)
    last = end if end is not None else len(log["t"]) - 1
    moving_end = max(e for e, vx, _ in segments if vx != 0.0)
    trot = (log["t"] >= 3.0) & (log["t"] < moving_end) & (np.arange(len(log["t"])) < last)
    steps = [s for s in ctrl.steps if s[0] < log["t"][last]]
    moved = [np.linalg.norm(c[:2] - n) > 0.01 for _, _, n, c in steps]
    n_mpc = len(log["solve_times"])
    return dict(distance=log["pos"][last, 0] - log["pos"][0, 0],
                fell_at=None if end is None else log["t"][end],
                speed=log["vel"][trot, 0].mean() if trot.any() else float("nan"),
                tilt=np.degrees(np.abs(log["rpy"][:last, :2])).max(), steps=len(steps),
                moved=int(sum(moved)), drops=[t for t, _ in hole_drops(log) if t <= log["t"][last]],
                mpc_ms=1000 * np.median(log["solve_times"][:n_mpc]),
                wbc_ms=1000 * np.mean(log["wbc_times"][:last + 1]),
                failures=ctrl.wbc.failures)


def draw(ctrl, keep=12):
    """Recorder callback: the chosen footholds (green discs) and the MPC ground forces (arrows)."""
    def extra(scene, data):
        def add(kind, size, pos, rgba):
            if scene.ngeom >= scene.maxgeom:
                return None
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, kind, np.array(size, float), np.array(pos, float),
                                np.eye(3).ravel(), np.array(rgba, np.float32))
            scene.ngeom += 1
            return g
        for _, _, _, chosen in ctrl.steps[-keep:]:
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, (0.022, 0.003, 0), (chosen[0], chosen[1], chosen[2] + 0.003),
                (0.1, 0.85, 0.2, 1.0))
        for leg in range(4):
            f = ctrl.forces[leg]
            if ctrl.sched.in_stance(leg) and f[2] > 5.0:
                start = data.geom_xpos[ctrl.feet_geoms[leg]] - [0.0, 0.0, go2.FOOT_RADIUS]
                g = add(mujoco.mjtGeom.mjGEOM_ARROW, (0, 0, 0), (0, 0, 0), (0.75, 1.0, 0.1, 1.0))
                if g is not None:
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, start, start + ARROW_SCALE * f)
    return extra


def ablation_plot(results):
    """Distance along the course against time, with and without the foothold planner."""
    import matplotlib.pyplot as plt
    terrain = make_course()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=False,
                                 gridspec_kw=dict(height_ratios=[1.6, 1]))
    styles = {"full stack": ("#1f77b4", "-"), "without the foothold planner": ("#ff7f0e", "-.")}
    for name, (log, ctrl, s) in results.items():
        end = fall_index(log, terrain)
        color, ls = styles[name]
        a1.plot(log["t"], log["pos"][:, 0], color=color, ls=ls, lw=1.8, label=name)
        if end is not None:
            a1.plot(log["t"][end], log["pos"][end, 0], "x", color=color, ms=10, mew=2.5)
            a1.annotate("tipped over 30 degrees", (log["t"][end], log["pos"][end, 0]),
                        textcoords="offset points", xytext=(8, -12), fontsize=8, color=color)
    for xx, label in ((0.59, "gap 8 cm"), (1.05, "step 5 cm"), (1.485, "gap 7 cm"), (2.05, "ramp"),
                      (2.7, "potholes")):
        a1.axhline(xx, color="0.8", lw=0.8, zorder=0)
        a1.annotate(label, (0.2, xx), fontsize=7.5, color="0.4", va="bottom")
    a1.set_xlabel("time (s)")
    a1.set_ylabel("body x along the course (m)")
    a1.set_title("Body position along the course in each run", fontsize=10, loc="left")
    a1.legend(fontsize=8, loc="lower right")
    x = terrain.x0 + terrain.res * np.arange(terrain.nx)
    a2.plot(x, terrain.h[terrain.index(0.0, 0.14)[0], :], color="0.3", lw=1, label="height at y = +0.14 m")
    a2.plot(x, terrain.h[terrain.index(0.0, -0.14)[0], :], color="0.3", lw=1, ls="--",
            label="height at y = -0.14 m")
    a2.set_xlim(-0.4, 4.6)
    a2.set_xlabel("x (m)")
    a2.set_ylabel("height (m)")
    a2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(MEDIA / "stack_ablation.png", dpi=110)
    plt.close(fig)


def media():
    """The course with the whole stack (GIF) and without the foothold planner, and the flat
    ground speed run. Prints the numbers of the README."""
    import matplotlib
    matplotlib.use("Agg")
    MEDIA.mkdir(exist_ok=True)
    terrain = make_course()
    model = go2.load_model(terrain.add_to_spec)
    results = {}
    runs = {"full stack": dict(), "without the foothold planner": dict(select=False)}
    for name, opts in runs.items():
        rec = None
        if name == "full stack":
            rec = Recorder(model, 360, 225, fps=12, distance=1.5, azimuth=115, elevation=-22,
                           end=COURSE[-1][0])

        def hook(data, ctrl, rec=rec):
            if rec is not None:
                if rec.extra is None:
                    rec.extra = draw(ctrl)
                rec(data)

        log, ctrl = run_stack(schedule_from(COURSE), COURSE[-1][0], terrain=terrain, frame_hook=hook,
                              model=model, **opts)
        s = summary(log, ctrl, terrain)
        results[name] = (log, ctrl, s)
        state = f"fell at t = {s['fell_at']:.1f} s" if s["fell_at"] is not None else "did not fall"
        drops = ", ".join(f"{t:.1f} s" for t in s["drops"]) or "none"
        print(f"stack course, {name}: {s['distance']:.2f} m, {state}, trot speed {s['speed']:.3f} m/s "
              f"(command {COURSE_SPEED}), largest tilt {s['tilt']:.1f} deg, feet into a hole: {drops}, "
              f"{s['moved']} of {s['steps']} steps moved, MPC median {s['mpc_ms']:.2f} ms, "
              f"WBC mean {s['wbc_ms']:.2f} ms, QP failures {s['failures']}")
        if rec is not None:
            save_gif(rec.frames, MEDIA / "stack_course.gif", fps=12, colors=96)
    ablation_plot(results)

    flat, ctrl = run_stack(schedule_from(FLAT), FLAT[-1][0])
    for (start, (end, vx, gait)) in zip([0.0] + [e for e, _, _ in FLAT], FLAT):
        sel = (flat["t"] >= (start + end) / 2) & (flat["t"] < end)
        print(f"stack flat, {gait.name} {vx:.1f} m/s: measured {flat['vel'][sel, 0].mean():.3f} m/s, "
              f"largest tilt {np.degrees(np.abs(flat['rpy'][sel, :2])).max():.1f} deg")
    print(f"stack flat: QP failures {ctrl.wbc.failures}, lowest body {flat['pos'][:, 2].min():.3f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", action="store_true", help="flat ground instead of the course")
    ap.add_argument("--vx", type=float, default=None, help="forward speed on flat ground (m/s)")
    ap.add_argument("--raibert-only", action="store_true", help="no foothold selection")
    ap.add_argument("--seconds", type=float, help="length of the run (after the schedule the robot "
                    "stands still, so a long run keeps the viewer open to explore)")
    ap.add_argument("--gif", help="write a GIF here instead of opening the viewer")
    args = ap.parse_args()
    if args.flat:
        terrain = None
        segments = FLAT if args.vx is None else [(1.0, 0.0, STAND), (9.0, args.vx, TROT),
                                                 (10.0, 0.0, TROT), (11.0, 0.0, STAND)]
    else:
        terrain = make_course()
        segments = COURSE
    model = go2.load_model(None if terrain is None else terrain.add_to_spec)
    seconds = segments[-1][0] if args.seconds is None else args.seconds
    run = dict(terrain=terrain, select=not args.raibert_only, model=model)
    if args.gif:
        rec = Recorder(model, 400, 250, fps=12, distance=1.5, azimuth=115, elevation=-22, end=seconds)

        def hook(data, ctrl):
            if rec.extra is None:
                rec.extra = draw(ctrl)
            rec(data)
        log, ctrl = run_stack(schedule_from(segments), seconds, frame_hook=hook, **run)
        save_gif(rec.frames, args.gif, fps=12)
    else:
        from common.video import WindowClosed, viewer_hook
        data = mujoco.MjData(model)
        show, close = viewer_hook(model, data, extra=lambda scene, d, ctrl: draw(ctrl)(scene, d))
        try:
            log, ctrl = run_stack(schedule_from(segments), seconds, frame_hook=show, data=data, **run)
        except WindowClosed:
            print(f"viewer closed at t = {data.time:.1f} s")
            return
        finally:
            close()
    s = summary(log, ctrl, terrain, segments)
    state = f"fell at t = {s['fell_at']:.1f} s" if s["fell_at"] is not None else "did not fall"
    speed = "too short to measure" if np.isnan(s["speed"]) else f"{s['speed']:.3f} m/s"
    print(f"covered {s['distance']:.2f} m, {state}, mean trot speed (from 3 s) {speed}, "
          f"largest tilt {s['tilt']:.1f} deg, QP failures {s['failures']}")


if __name__ == "__main__":
    main()
