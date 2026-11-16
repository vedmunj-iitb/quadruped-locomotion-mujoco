"""Foothold stage demo: walk the course with the gait stage's joint PD controller, landing on
planned footholds, in the MuJoCo viewer or to a GIF. Also the foothold stage's media.

Written in 2026.

    python -m footholds.demo                      # viewer, walk with foothold selection
    python -m footholds.demo --gait trot          # trot at 0.4 m/s
    python -m footholds.demo --gait trot --raibert-only --gif fall.gif
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
from footholds.planner import foothold_costs, total_cost
from footholds.terrain import GROUND, make_course
from footholds.walker import walk_course

MEDIA = Path(__file__).resolve().parents[1] / "media"
SPEEDS = {"walk": 0.2, "trot": 0.4}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]   # FL, FR, RL, RR
MOVED = 0.01   # a step counts as moved when the planner shifted it by more than one grid cell (m)
RUNS = {   # name: settings, the runs in the README table
    "walk, foothold selection": dict(gait="walk", speed=0.2, seconds=19.0, select=True),
    "walk, Raibert point only": dict(gait="walk", speed=0.2, seconds=19.0, select=False),
    "trot, foothold selection": dict(gait="trot", speed=0.4, seconds=12.0, select=True),
    "trot, Raibert point only": dict(gait="trot", speed=0.4, seconds=12.0, select=False),
}


def summary(log, ctrl, terrain):
    """Distance covered, whether and where it fell, feet that dropped into a hole, steps moved by
    the planner, landing error."""
    tilt = np.degrees(np.abs(log["rpy"][:, :2])).max(axis=1)
    fell = fall_index(log, terrain)
    end = fell if fell is not None else len(log["t"]) - 1
    g = ctrl.sched.gait
    landed = [s for s in ctrl.steps if s[0] + (1 - g.duty) * g.period < log["t"][end]]
    moved = [np.linalg.norm(c[:2] - n) for _, _, n, c in landed]
    # Landing error: where the foot first touches down after the step was planned (after it
    # has left the ground, since lift-off comes late), against the chosen foothold.
    errors = []
    dt = log["t"][1] - log["t"][0]
    for t, leg, _, chosen in landed:
        i0 = int(round(t / dt))
        up = np.flatnonzero(~log["contact"][i0:end, leg])
        if not len(up):
            continue
        down = np.flatnonzero(log["contact"][i0 + up[0]:end, leg])
        if len(down):
            errors.append(np.linalg.norm(log["feet"][i0 + up[0] + down[0], leg, :2] - chosen[:2]))
    return dict(distance=log["pos"][end, 0] - log["pos"][0, 0],
                fell_at=log["t"][fell] if fell is not None else None,
                fell_x=log["pos"][fell, 0] if fell is not None else None,
                drops=[(t, go2.LEGS[leg]) for t, leg in hole_drops(log) if t <= log["t"][end]],
                tilt=tilt[:end].max(), steps=len(landed),
                moved=sum(m > MOVED for m in moved), largest_move=max(moved),
                landing_median=np.median(errors), landing_max=max(errors),
                lowest_foot=log["feet"][:end, :, 2].min())


def foothold_markers(ctrl, keep=16):
    """Recorder callback: green discs at chosen footholds, red ones at moved Raibert points."""
    def draw(scene, data):
        for t, leg, nominal, chosen in ctrl.steps[-keep:]:
            spots = [(chosen, [0.1, 0.85, 0.2, 1.0])]
            if np.linalg.norm(chosen[:2] - nominal) > MOVED:
                spots.append((np.array([*nominal, chosen[2]]), [0.95, 0.15, 0.1, 1.0]))
            for p, rgba in spots:
                if scene.ngeom >= scene.maxgeom:
                    return
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    np.array([0.022, 0.003, 0.0]), np.array([p[0], p[1], p[2] + 0.003]),
                                    np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
                scene.ngeom += 1
    return draw


def course_figure(terrain, log, ctrl, name):
    """Top view of the course with every planned step, and the height profile along it."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.6), gridspec_kw=dict(height_ratios=[1.6, 1]),
                                 sharex=True)
    ext = [terrain.x0, terrain.x0 + (terrain.nx - 1) * terrain.res,
           terrain.y0, terrain.y0 + (terrain.ny - 1) * terrain.res]
    im = a1.imshow(terrain.h, origin="lower", extent=ext, cmap="gist_earth", vmin=0, vmax=0.25)
    for t, leg, nominal, chosen in ctrl.steps:
        if np.linalg.norm(chosen[:2] - nominal) > MOVED:
            a1.plot([nominal[0], chosen[0]], [nominal[1], chosen[1]], color="w", lw=0.8)
            a1.plot(*nominal, "x", color="red", ms=5, mew=1.5)
        a1.plot(*chosen[:2], "o", color=COLORS[leg], ms=4.5, mec="k", mew=0.5)
    a1.plot(log["pos"][:, 0], log["pos"][:, 1], color="w", lw=1, ls="--")
    a1.set_xlim(-0.4, log["pos"][-1, 0] + 0.35)
    a1.set_ylim(-0.45, 0.45)
    a1.set_aspect("equal")
    a1.set_ylabel("y (m)")
    a1.set_title(f"{name}: chosen footholds (dots, one colour per leg), Raibert points the "
                 "planner moved (red x), body path (dashed)", fontsize=9, loc="left")
    fig.colorbar(im, ax=a1, label="height (m)", fraction=0.025, pad=0.01)
    x = terrain.x0 + terrain.res * np.arange(terrain.nx)
    for yy, ls in ((0.14, "-"), (-0.14, "--")):
        a2.plot(x, terrain.h[terrain.index(0.0, yy)[0], :], ls=ls, color="0.3", lw=1,
                label=f"height at y = {yy:+.2f} m")
    a2.set_xlabel("x (m)")
    a2.set_ylabel("height (m)")
    a2.set_ylim(-0.01, 0.2)
    a2.legend(fontsize=8, loc="lower left")
    for label, xx in (("gap 8 cm", 0.59), ("step up 5 cm", 1.05), ("gap 7 cm", 1.485),
                      ("ramp", 2.05), ("potholes", 2.7)):
        a2.annotate(label, (xx, 0.185), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / "footholds_course.png", dpi=110)
    plt.close(fig)


def costmap_figure(terrain, ctrl):
    """The cost terms round the Raibert point the planner moved furthest, near the first gap."""
    import matplotlib.pyplot as plt
    t, leg, nominal, chosen = max((s for s in ctrl.steps if s[2][0] < 1.0),
                                  key=lambda s: np.linalg.norm(s[3][:2] - s[2]))
    hip = np.array([nominal[0], nominal[1], GROUND + 0.30])
    (iy, ix), terms, feasible = foothold_costs(terrain, nominal, hip)
    panels = [("slope (rise over run)", terms["slope"]), ("edge", terms["edge"]),
              ("reach", terms["reach"]),
              ("total cost, grey: not allowed", np.where(feasible, total_cost(terms), np.nan))]
    rows, cols = iy - iy.min(), ix - ix.min()          # candidates as an image round the window
    x0, y0 = terrain.point(iy.min(), ix.min())
    ext = [x0 - terrain.res / 2, x0 + (cols.max() + 0.5) * terrain.res,
           y0 - terrain.res / 2, y0 + (rows.max() + 0.5) * terrain.res]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (title, value) in zip(axes, panels):
        img = np.full((rows.max() + 1, cols.max() + 1), np.nan)
        if title.startswith("total"):
            grey = np.full(img.shape, np.nan)
            grey[rows[~feasible], cols[~feasible]] = 1.0
            ax.imshow(grey, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=4)
        img[rows, cols] = value
        im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis")
        ax.axvline(0.545, color="w", lw=0.8, ls=":")     # the gap's cells drawn round their
        ax.axvline(0.625, color="w", lw=0.8, ls=":")     # grid points: 0.55 to 0.62
        ax.plot(*nominal, "x", color="red", ms=9, mew=2, label="Raibert point")
        ax.plot(*chosen[:2], "o", color="w", mec="k", ms=8, label="chosen")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("y (m)")
    axes[3].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{go2.LEGS[leg]} foot at t = {t:.2f} s: the Raibert point is in the 8 cm gap "
                 f"(dotted lines), so the planner moves it {100 * np.linalg.norm(chosen[:2] - nominal):.1f} cm",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / "footholds_costmap.png", dpi=110)
    plt.close(fig)


def media():
    """The four course runs, the foothold stage's media and the numbers of the README table."""
    import matplotlib
    matplotlib.use("Agg")
    MEDIA.mkdir(exist_ok=True)
    terrain = make_course()
    model = go2.load_model(terrain.add_to_spec)
    results = {}
    for name, settings in RUNS.items():
        rec = None
        if name.startswith("trot"):
            rec = Recorder(model, 400, 225, fps=12, distance=1.5, azimuth=115, elevation=-22,
                           end=10.0 if settings["select"] else 3.5)
        def draw_footholds(ctrl, rec=rec):
            if rec is not None:
                rec.extra = foothold_markers(ctrl)
        log, ctrl = walk_course(model=model, terrain=terrain, frame_hook=rec,
                                on_start=draw_footholds, **settings)
        results[name] = (log, ctrl, summary(log, ctrl, terrain))
        s = results[name][2]
        fell = f"fell at t = {s['fell_at']:.1f} s, x = {s['fell_x']:.2f} m" if s["fell_at"] else "did not fall"
        drops = ", ".join(f"{leg} at {t:.1f} s" for t, leg in s["drops"]) or "none"
        print(f"footholds {name}: {s['distance']:.2f} m, {fell}, feet into a hole: {drops}, "
              f"largest tilt before that {s['tilt']:.1f} deg, "
              f"{s['moved']} of {s['steps']} steps moved (largest {100 * s['largest_move']:.1f} cm), "
              f"landing error median {1000 * s['landing_median']:.0f} mm, largest "
              f"{1000 * s['landing_max']:.0f} mm, lowest foot centre {s['lowest_foot']:.3f} m")
        if rec is not None:
            out = "footholds_trot.gif" if settings["select"] else "footholds_no_selection.gif"
            save_gif(rec.frames, MEDIA / out, fps=12, colors=96)
    course_figure(terrain, *results["trot, foothold selection"][:2], "trot at 0.4 m/s")
    costmap_figure(terrain, results["trot, foothold selection"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", choices=list(SPEEDS), default="walk")
    ap.add_argument("--raibert-only", action="store_true",
                    help="land on the Raibert point, without foothold selection")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--gif", help="write a GIF here instead of opening the viewer")
    args = ap.parse_args()
    seconds = args.seconds or (19.0 if args.gait == "walk" else 12.0)
    terrain = make_course()
    model = go2.load_model(terrain.add_to_spec)
    run = dict(gait=args.gait, speed=SPEEDS[args.gait], seconds=seconds,
               select=not args.raibert_only, model=model, terrain=terrain)

    if args.gif:
        rec = Recorder(model, 480, 270, fps=12, distance=1.5, azimuth=115, elevation=-22,
                       end=seconds)
        log, ctrl = walk_course(frame_hook=rec, **run)
        save_gif(rec.frames, args.gif, fps=12)
    else:
        from common.video import WindowClosed, viewer_hook
        data = mujoco.MjData(model)
        markers = {}

        def keep(ctrl):                              # the walker exists once the run starts
            markers["draw"] = foothold_markers(ctrl)

        show, close = viewer_hook(model, data, extra=lambda scene, d: markers["draw"](scene, d))
        try:
            log, ctrl = walk_course(frame_hook=show, data=data, on_start=keep, **run)
        except WindowClosed:
            print(f"viewer closed at t = {data.time:.1f} s")
            return
        finally:
            close()

    fell = fall_index(log, terrain)
    last = fell if fell is not None else -1
    end = f"fell at t = {log['t'][last]:.1f} s" if fell is not None else "did not fall"
    print(f"{args.gait}, {'Raibert point only' if args.raibert_only else 'foothold selection'}: "
          f"covered {log['pos'][last, 0] - log['pos'][0, 0]:.2f} m, {end}")


if __name__ == "__main__":
    main()
